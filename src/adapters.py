"""Success and failure adapters attached to a frozen causal LM.

Each Transformer block's MLP is wrapped so that its output becomes

    mlp(x) + scale_S * S(x) + scale_F * F(x)

where S and F are small parallel SwiGLU MLPs (hidden width `width`) that read
the same input as the original MLP. Their output projections start at zero,
so at initialization the model is exactly the base model.

The scales let us remove an adapter at evaluation time (scale 0), keep it
(scale 1), or turn it up or down (any alpha).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUAdapter(nn.Module):
    def __init__(self, d_model, width):
        super().__init__()
        self.gate = nn.Linear(d_model, width, bias=False)
        self.up = nn.Linear(d_model, width, bias=False)
        self.down = nn.Linear(width, d_model, bias=False)
        nn.init.zeros_(self.down.weight)  # start exactly at the base model

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MLPWithAdapters(nn.Module):
    """Wraps a block's MLP and adds the two adapters in parallel."""

    def __init__(self, mlp, d_model, width, adapter_dtype=torch.float32):
        super().__init__()
        self.mlp = mlp
        self.S = SwiGLUAdapter(d_model, width).to(adapter_dtype)
        self.F = SwiGLUAdapter(d_model, width).to(adapter_dtype)
        self.scale_S = 1.0
        self.scale_F = 1.0

    def forward(self, x):
        out = self.mlp(x)
        if self.scale_S == 0.0 and self.scale_F == 0.0:
            return out
        xa = x.to(self.S.down.weight.dtype)
        extra = 0.0
        if self.scale_S != 0.0:
            extra = extra + self.scale_S * self.S(xa)
        if self.scale_F != 0.0:
            extra = extra + self.scale_F * self.F(xa)
        return out + extra.to(out.dtype)


def _decoder_layers(model):
    # Qwen2 / Llama-style models keep their blocks at model.model.layers
    return model.model.layers


def attach_adapters(model, width=256, adapter_dtype=torch.float32):
    """Freeze the model and add an S and an F adapter to every block.

    Returns (S_params, F_params): lists of the trainable parameters.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.config.hidden_size
    device = next(model.parameters()).device
    for layer in _decoder_layers(model):
        layer.mlp = MLPWithAdapters(layer.mlp, d_model, width, adapter_dtype).to(device)
    return adapter_params(model)


def adapter_modules(model):
    return [layer.mlp for layer in _decoder_layers(model) if isinstance(layer.mlp, MLPWithAdapters)]


def adapter_params(model):
    S_params, F_params = [], []
    for m in adapter_modules(model):
        S_params += list(m.S.parameters())
        F_params += list(m.F.parameters())
    return S_params, F_params


def set_adapter_scales(model, scale_S=1.0, scale_F=1.0):
    """E.g. (1, 0) removes the failure adapter, (0, 1) removes the success adapter,
    (0, 0) gives back the base model."""
    for m in adapter_modules(model):
        m.scale_S = float(scale_S)
        m.scale_F = float(scale_F)


def adapter_state_dict(model):
    """Only the adapter weights (small), for saving checkpoints."""
    return {
        f"{i}.{name}": t.detach().cpu()
        for i, m in enumerate(adapter_modules(model))
        for name, t in list(m.S.state_dict(prefix="S.").items()) + list(m.F.state_dict(prefix="F.").items())
    }


def load_adapter_state_dict(model, state):
    for i, m in enumerate(adapter_modules(model)):
        m.S.load_state_dict({k[len(f"{i}.S."):]: v for k, v in state.items() if k.startswith(f"{i}.S.")})
        m.F.load_state_dict({k[len(f"{i}.F."):]: v for k, v in state.items() if k.startswith(f"{i}.F.")})


def count_params(params):
    return sum(p.numel() for p in params)
