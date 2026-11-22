"""Train the success/failure adapters with GRPO and gradient routing.

Example (from the repo root):
    # quick check on a T4 (Kaggle/Colab): a few steps, short generations
    python scripts/train_grpo.py --mode outcome --steps 5 --prompts_per_step 2 \
        --max_new_tokens 256 --name debug_outcome

    # milestone run on an L4/A100 (GCP)
    python scripts/train_grpo.py --mode outcome --steps 100 --name outcome_seed0

Modes: grpo, outcome, random, psr, nsr (see src/grpo.py).
Outputs go to runs/<name>/: log.jsonl (one line per step), adapters_step*.pt,
config.json, and samples.jsonl (a few rollouts per step for inspection).
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from adapters import (  # noqa: E402
    adapter_state_dict,
    attach_adapters,
    count_params,
    set_adapter_scales,
)
from checker import has_valid_format, is_correct  # noqa: E402
from data import load_math_train_dev  # noqa: E402
from grpo import (  # noqa: E402
    MODES,
    completion_mask_from_generation,
    group_advantages,
    routed_backward,
    routing_masks,
    sequence_losses,
    token_logprobs,
)

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def pick_dtype(name):
    if name != "auto":
        return getattr(torch, name)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    # T4: float16 training of a frozen base is fragile, so use float32 there
    return torch.float32


def build_prompt(tokenizer, problem):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def eos_ids(tokenizer, model):
    ids = set()
    gen_eos = model.generation_config.eos_token_id
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


@torch.no_grad()
def rollout(model, tokenizer, prompts, group_size, max_new_tokens, temperature):
    """Sample group_size completions per prompt. Returns token tensors and texts."""
    model.eval()
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    prompt_mask = enc["attention_mask"].repeat_interleave(group_size, dim=0)
    gen = out[:, prompt_len:]
    gen_mask, finished = completion_mask_from_generation(gen, eos_ids(tokenizer, model))
    attention_mask = torch.cat([prompt_mask, gen_mask], dim=1)
    # replace tokens after the first EOS with padding before decoding
    gen_clean = torch.where(gen_mask.bool(), gen, torch.full_like(gen, tokenizer.pad_token_id))
    texts = tokenizer.batch_decode(gen_clean, skip_special_tokens=True)
    return out, attention_mask, prompt_len, gen_mask, texts, finished


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--prompts_per_step", type=int, default=8)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--micro_batch", type=int, default=4, help="sequences per forward/backward")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.001, help="KL penalty toward the base model")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--adapter_width", type=int, default=256)
    p.add_argument("--max_level", type=int, default=5, help="train only on MATH levels <= this")
    p.add_argument("--n_train_problems", type=int, default=None, help="use a fixed subset (e.g. 20 for an overfit check)")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_grad_checkpointing", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    route_gen = torch.Generator().manual_seed(args.seed + 1)

    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ----- data -----
    train, _dev = load_math_train_dev()
    def level_num(ex):
        last = str(ex["level"]).split()[-1]  # "Level 3" -> 3; "Level ?" -> unknown
        return int(last) if last.isdigit() else None

    train = [ex for ex in train if level_num(ex) is not None and level_num(ex) <= args.max_level]
    rng = random.Random(args.seed)
    rng.shuffle(train)
    if args.n_train_problems:
        train = train[: args.n_train_problems]
    print(f"{len(train)} training problems")

    # ----- model -----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = pick_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    S_params, F_params = attach_adapters(model, width=args.adapter_width)
    print(f"dtype={dtype}  adapter params: S={count_params(S_params):,}  F={count_params(F_params):,}")
    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = True  # generation still uses the KV cache in eval mode
    opt = torch.optim.AdamW(S_params + F_params, lr=args.lr, weight_decay=0.0)

    log_f = open(os.path.join(out_dir, "log.jsonl"), "a")
    samples_f = open(os.path.join(out_dir, "samples.jsonl"), "a")
    cursor = 0

    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch = []
        for _ in range(args.prompts_per_step):
            if cursor >= len(train):
                rng.shuffle(train)
                cursor = 0
            batch.append(train[cursor])
            cursor += 1
        prompts = [build_prompt(tokenizer, ex["problem"]) for ex in batch]

        # ----- rollouts (both adapters active) -----
        set_adapter_scales(model, 1.0, 1.0)
        input_ids, attn, prompt_len, gen_mask, texts, finished = rollout(
            model, tokenizer, prompts, args.group_size, args.max_new_tokens, args.temperature)
        golds = [ex["answer"] for ex in batch for _ in range(args.group_size)]
        correct = torch.tensor([is_correct(t, g) for t, g in zip(texts, golds)])
        valid = [has_valid_format(t) for t in texts]
        rewards = correct.float()
        adv = group_advantages(rewards, args.group_size)
        m_S, m_F = routing_masks(correct, args.mode, generator=route_gen)
        N = len(texts)

        # ----- old and reference log-probs (no grad) -----
        model.eval()
        old_lp, ref_lp = [], []
        with torch.no_grad():
            for i in range(0, N, args.micro_batch):
                sl = slice(i, i + args.micro_batch)
                set_adapter_scales(model, 1.0, 1.0)
                old_lp.append(token_logprobs(model, input_ids[sl], attn[sl], prompt_len))
                if args.beta > 0:
                    set_adapter_scales(model, 0.0, 0.0)  # base model = reference
                    ref_lp.append(token_logprobs(model, input_ids[sl], attn[sl], prompt_len))
        set_adapter_scales(model, 1.0, 1.0)

        # ----- policy update with routing -----
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_S_total = loss_F_total = 0.0
        for j, i in enumerate(range(0, N, args.micro_batch)):
            sl = slice(i, i + args.micro_batch)
            if adv[sl].abs().sum() == 0 and args.beta == 0:
                continue  # no learning signal in this micro-batch
            lp = token_logprobs(model, input_ids[sl], attn[sl], prompt_len)
            losses = sequence_losses(
                lp, old_lp[j], ref_lp[j] if ref_lp else None, adv[sl].to(lp.device),
                gen_mask[sl], clip_eps=args.clip_eps, beta=args.beta)
            lS, lF = routed_backward(losses, m_S[sl].to(lp.device), m_F[sl].to(lp.device), N,
                                     S_params, F_params)
            loss_S_total += lS
            loss_F_total += lF
        grad_norm = torch.nn.utils.clip_grad_norm_(S_params + F_params, 1.0).item()
        opt.step()

        gen_lens = gen_mask.sum(dim=1).float()
        record = {
            "step": step,
            "reward": rewards.mean().item(),
            "format_validity": sum(valid) / N,
            "finished_rate": finished.float().mean().item(),
            "mean_gen_tokens": gen_lens.mean().item(),
            "frac_to_F": m_F.mean().item(),
            "frac_to_S": m_S.mean().item(),
            "loss_S": loss_S_total,
            "loss_F": loss_F_total,
            "grad_norm": grad_norm,
            "seconds": round(time.time() - t0, 1),
        }
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()
        samples_f.write(json.dumps({"step": step, "problem": batch[0]["problem"],
                                    "answer": batch[0]["answer"],
                                    "completions": texts[: args.group_size],
                                    "correct": correct[: args.group_size].tolist()}) + "\n")
        samples_f.flush()
        print(f"step {step:4d} | reward {record['reward']:.3f} | valid {record['format_validity']:.2f} | "
              f"len {record['mean_gen_tokens']:.0f} | gnorm {grad_norm:.3f} | {record['seconds']}s", flush=True)

        if step % args.save_every == 0 or step == args.steps:
            torch.save(adapter_state_dict(model), os.path.join(out_dir, f"adapters_step{step}.pt"))

    log_f.close()
    samples_f.close()
    print(f"Done. Logs and adapters in {out_dir}/")


if __name__ == "__main__":
    main()
