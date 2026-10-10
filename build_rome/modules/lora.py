# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

import math

import torch
import torch.nn.functional as F
from torch import nn


def apply_lora_to_attention(attn, rank: int, alpha: float):
    """Apply LoRA to self or cross attention layers if enabled."""
    # Prevent double injection
    if getattr(attn, "_lora_applied", False):
        return

    def inject_lora(linear: nn.Linear):
        linear.lora_A = nn.Parameter(torch.zeros(rank, linear.in_features))
        nn.init.kaiming_uniform_(linear.lora_A, a=math.sqrt(5))
        linear.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        linear.scaling = alpha / rank

        linear._original_forward = linear.forward

        def forward_with_lora(x):
            delta = F.linear(x, linear.lora_B @ linear.lora_A) * linear.scaling
            return linear._original_forward(x) + delta

        linear.forward = forward_with_lora

    if attn._type == "self":
        inject_lora(attn.to_qkv)
    else:
        inject_lora(attn.to_q)
        inject_lora(attn.to_kv)
    inject_lora(attn.to_out)

    attn._lora_applied = True


def apply_lora(model, lora_cfg):
    for p in model.parameters():
        p.requires_grad_(False)

    lora_rank = lora_cfg["r"]
    lora_alpha = lora_cfg["alpha"]

    for block in model.blocks:
        apply_lora_to_attention(block.self_attn, lora_rank, lora_alpha)
        apply_lora_to_attention(block.cross_attn, lora_rank, lora_alpha)

    return model
