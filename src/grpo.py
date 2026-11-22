"""GRPO with outcome-based gradient routing.

Core pieces, kept separate so each can be tested:
  - group_advantages: group-normalized advantages (Lec 12 GRPO)
  - routing_masks:    which sequences send gradients to S and to F, per mode
  - sequence_losses:  clipped GRPO objective + KL penalty, one loss per sequence
  - routed_backward:  backpropagates the S-loss only into S and the F-loss only into F

Routing modes:
  grpo     standard GRPO, every sequence updates both adapters
           (two parallel adapters trained on the same gradients = one wider adapter)
  outcome  incorrect sequences update only F, correct sequences update only S
  random   correct/incorrect labels are shuffled within the batch before routing,
           so F gets exactly as many sequences as under outcome routing
           (the key control)
  psr      only correct sequences are used, updating both adapters
  nsr      only incorrect sequences are used, updating both adapters
"""

import torch

MODES = ("grpo", "outcome", "random", "psr", "nsr")


def group_advantages(rewards, group_size, eps=1e-4):
    """rewards: float tensor of shape [n_prompts * group_size], grouped consecutively.

    A_i = (r_i - mean(group)) / (std(group) + eps). Groups where every sample
    got the same reward have zero advantage (no learning signal).
    """
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, unbiased=False)
    return ((r - mean) / (std + eps)).view(-1)


def routing_masks(correct, mode, generator=None):
    """correct: bool tensor [N]. Returns float masks (m_S, m_F) of shape [N]."""
    c = correct.float()
    ones = torch.ones_like(c)
    if mode == "grpo":
        return ones, ones
    if mode == "outcome":
        return c, 1.0 - c
    if mode == "random":
        perm = torch.randperm(len(c), generator=generator).to(c.device)
        shuffled = c[perm]
        return shuffled, 1.0 - shuffled
    if mode == "psr":
        return c, c
    if mode == "nsr":
        return 1.0 - c, 1.0 - c
    raise ValueError(f"unknown mode {mode}; choose from {MODES}")


def token_logprobs(model, input_ids, attention_mask, prompt_len):
    """Log-probabilities of the generated tokens under the model.

    input_ids, attention_mask: [B, T] = left-padded prompt (length prompt_len)
    followed by the generated tokens. Returns [B, T - prompt_len]: the log-prob of
    each generated token input_ids[:, prompt_len:]. Only the logits for those
    positions are computed, which saves a lot of memory with long prompts.
    """
    T = input_ids.shape[1]
    L = T - prompt_len
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                  position_ids=position_ids, use_cache=False)
    try:
        logits = model(**kwargs, logits_to_keep=L + 1).logits
    except TypeError:  # older transformers without logits_to_keep
        logits = model(**kwargs).logits[:, -(L + 1):]
    logits = logits[:, :-1]  # positions prompt_len-1 .. T-2 predict tokens prompt_len .. T-1
    targets = input_ids[:, prompt_len:]
    V = logits.shape[-1]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none")
    return -nll.view(targets.shape)


def sequence_losses(logp, old_logp, ref_logp, advantages, gen_mask, clip_eps=0.2, beta=0.0):
    """Per-sequence GRPO loss (to minimize).

    logp, old_logp, ref_logp: [B, L] log-probs of the generated tokens (policy,
    policy at rollout time, reference = base model). gen_mask: [B, L], 1 for real
    generated tokens (through the first EOS). advantages: [B]. Token losses are
    averaged over each sequence's generated tokens.
      ratio      = exp(logp - old_logp)
      objective  = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
      KL (k3)    = exp(ref - logp) - (ref - logp) - 1
      loss       = -(objective - beta * KL)
    """
    mask = gen_mask.float()
    ratio = torch.exp(logp - old_logp)
    A = advantages.unsqueeze(1)
    obj = torch.minimum(ratio * A, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A)
    if beta > 0 and ref_logp is not None:
        d = ref_logp - logp
        kl = torch.exp(d) - d - 1
        obj = obj - beta * kl
    n_tokens = mask.sum(dim=1).clamp(min=1)
    return -(obj * mask).sum(dim=1) / n_tokens


def routed_backward(seq_losses, m_S, m_F, n_total, S_params, F_params):
    """Accumulate gradients: the S-loss only into S_params, the F-loss only into F_params.

    seq_losses: [b] losses for this micro-batch. m_S, m_F: [b] masks.
    n_total: number of sequences in the full batch (for consistent scaling
    across micro-batches).
    """
    loss_S = (seq_losses * m_S).sum() / n_total
    loss_F = (seq_losses * m_F).sum() / n_total
    targets = [(loss_S, S_params), (loss_F, F_params)]
    active = [(loss, params) for (loss, params), m in zip(targets, (m_S, m_F))
              if loss.requires_grad and m.sum() > 0]  # skip an adapter with nothing routed to it
    for i, (loss, params) in enumerate(active):
        grads = torch.autograd.grad(loss, params, retain_graph=(i < len(active) - 1), allow_unused=True)
        for p, g in zip(params, grads):
            if g is None:
                continue
            p.grad = g.detach() if p.grad is None else p.grad + g.detach()
    return loss_S.item(), loss_F.item()


def completion_mask_from_generation(gen_ids, eos_token_ids):
    """Mask over generated tokens: 1 up to and including the first EOS, 0 after.

    gen_ids: [B, L] generated token ids (no prompt). Returns (mask [B, L], finished [B]).
    """
    is_eos = torch.zeros_like(gen_ids, dtype=torch.bool)
    for e in eos_token_ids:
        is_eos |= gen_ids == e
    # positions strictly after the first EOS
    after = (is_eos.long().cumsum(dim=1) - is_eos.long()) > 0
    mask = (~after).long()
    finished = is_eos.any(dim=1)
    return mask, finished
