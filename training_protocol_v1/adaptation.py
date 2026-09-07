"""Pinned ordinary LoRA layer; never attached to Qwen during preparation."""

import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank != 8 or alpha != 16:
            raise ValueError("Frozen pilot requires nn.Linear, rank 8 and alpha 16")
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device="cpu"))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device="cpu"))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)


def attach_for_future_training(model):
    """Explicit future-training helper. No call from baseline/preflight code."""
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("LoRA is already attached")
    layers = model.model.layers
    if len(layers) != 24 or model.device.type != "cpu" or model.dtype != torch.float32:
        raise ValueError("Architecture/device/dtype differs from the protocol")
    model.requires_grad_(False)
    for layer in layers:
        for name in ("q_proj", "v_proj"):
            setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name)))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != 540672:
        raise ValueError("Unexpected adapter parameter count")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model
