"""Tests for the adapters, GRPO loss, and gradient routing, on a tiny random Qwen2
model (CPU, a few seconds). Run: python -m pytest tests/ -q"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adapters import (  # noqa: E402
    adapter_state_dict,
    attach_adapters,
    load_adapter_state_dict,
    set_adapter_scales,
)
from grpo import (  # noqa: E402
    completion_mask_from_generation,
    group_advantages,
    routed_backward,
    routing_masks,
    sequence_losses,
    token_logprobs,
)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.Qwen2Config(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return transformers.Qwen2ForCausalLM(cfg).eval()


def batch(B=4, P=5, L=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 100, (B, P + L), generator=g)
    attn = torch.ones_like(ids)
    attn[0, :2] = 0  # left padding on one prompt
    return ids, attn, P


def nonzero_adapters(model, scale=0.1):
    """Give the adapters non-zero output weights so they affect the model."""
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.S.down.weight.normal_(0, scale)
            layer.mlp.F.down.weight.normal_(0, scale)


def test_zero_init_matches_base():
    model = tiny_model()
    ids, attn, P = batch()
    base = token_logprobs(model, ids, attn, P)
    attach_adapters(model, width=8)
    with_adapters = token_logprobs(model, ids, attn, P)
    assert torch.allclose(base, with_adapters, atol=1e-6)


def test_only_adapters_trainable():
    model = tiny_model()
    S, F = attach_adapters(model, width=8)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in S + F}
    assert len(S) == len(F) > 0


def test_removing_both_adapters_gives_base():
    model = tiny_model()
    ids, attn, P = batch()
    base = token_logprobs(model, ids, attn, P)
    attach_adapters(model, width=8)
    nonzero_adapters(model)
    assert not torch.allclose(base, token_logprobs(model, ids, attn, P))
    set_adapter_scales(model, 0.0, 0.0)
    assert torch.allclose(base, token_logprobs(model, ids, attn, P), atol=1e-6)


def test_save_and_load_adapters():
    model = tiny_model()
    ids, attn, P = batch()
    attach_adapters(model, width=8)
    nonzero_adapters(model)
    before = token_logprobs(model, ids, attn, P)
    state = adapter_state_dict(model)
    other = tiny_model()  # same seed, same base weights
    attach_adapters(other, width=8)
    load_adapter_state_dict(other, state)
    assert torch.allclose(before, token_logprobs(other, ids, attn, P), atol=1e-6)


def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    a = group_advantages(r, group_size=4)
    assert torch.allclose(a[:4], torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=1e-3)
    assert torch.all(a[4:] == 0)  # all correct -> no signal


def test_routing_masks():
    c = torch.tensor([True, False, False, True, False, False])
    mS, mF = routing_masks(c, "outcome")
    assert mS.tolist() == [1, 0, 0, 1, 0, 0] and mF.tolist() == [0, 1, 1, 0, 1, 1]
    mS, mF = routing_masks(c, "random", generator=torch.Generator().manual_seed(0))
    assert mS.sum() == c.sum() and torch.all(mS + mF == 1)  # same counts, just shuffled
    mS, mF = routing_masks(c, "grpo")
    assert torch.all(mS == 1) and torch.all(mF == 1)
    mS, mF = routing_masks(c, "nsr")
    assert mS.tolist() == mF.tolist() == [0, 1, 1, 0, 1, 1]


def _grads_after_routing(correct, mode):
    model = tiny_model()
    ids, attn, P = batch(B=len(correct))
    S, F = attach_adapters(model, width=8)
    nonzero_adapters(model)
    model.train()
    gen_mask = torch.ones(len(correct), ids.shape[1] - P)
    with torch.no_grad():
        old = token_logprobs(model, ids, attn, P)
    lp = token_logprobs(model, ids, attn, P)
    adv = torch.tensor([1.0 if c else -1.0 for c in correct])
    losses = sequence_losses(lp, old, None, adv, gen_mask)
    mS, mF = routing_masks(torch.tensor(correct), mode)
    routed_backward(losses, mS, mF, len(correct), S, F)
    gS = sum(p.grad.abs().sum().item() for p in S if p.grad is not None)
    gF = sum(p.grad.abs().sum().item() for p in F if p.grad is not None)
    return gS, gF


def test_outcome_routing_all_correct_only_updates_S():
    gS, gF = _grads_after_routing([True, True, True, True], "outcome")
    assert gS > 0 and gF == 0


def test_outcome_routing_all_wrong_only_updates_F():
    gS, gF = _grads_after_routing([False, False, False, False], "outcome")
    assert gF > 0 and gS == 0


def test_outcome_routing_matches_separate_losses():
    """The F gradient under outcome routing equals the gradient of the
    incorrect-only loss, so correct answers contribute nothing to F."""
    model = tiny_model()
    ids, attn, P = batch()
    S, F = attach_adapters(model, width=8)
    nonzero_adapters(model)
    model.train()
    correct = torch.tensor([True, False, True, False])
    adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
    gen_mask = torch.ones(4, ids.shape[1] - P)
    with torch.no_grad():
        old = token_logprobs(model, ids, attn, P)
    losses = sequence_losses(token_logprobs(model, ids, attn, P), old, None, adv, gen_mask)
    mS, mF = routing_masks(correct, "outcome")
    routed_backward(losses, mS, mF, 4, S, F)
    routed_F = [p.grad.clone() for p in F]

    for p in S + F:
        p.grad = None
    losses = sequence_losses(token_logprobs(model, ids, attn, P), old, None, adv, gen_mask)
    ((losses * (~correct).float()).sum() / 4).backward()
    for g_routed, p in zip(routed_F, F):
        assert torch.allclose(g_routed, p.grad, atol=1e-7)


def test_completion_mask_stops_after_first_eos():
    gen = torch.tensor([[5, 6, 9, 7, 7], [5, 6, 7, 8, 1]])
    mask, finished = completion_mask_from_generation(gen, eos_token_ids=[9])
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert finished.tolist() == [True, False]


def test_kl_is_zero_against_itself_and_positive_otherwise():
    lp = torch.log(torch.tensor([[0.5, 0.25]]))
    mask = torch.ones(1, 2)
    zero_adv = torch.zeros(1)
    assert torch.allclose(sequence_losses(lp, lp, lp, zero_adv, mask, beta=1.0), torch.zeros(1))
    ref = torch.log(torch.tensor([[0.25, 0.5]]))
    assert sequence_losses(lp, lp, ref, zero_adv, mask, beta=1.0).item() > 0
