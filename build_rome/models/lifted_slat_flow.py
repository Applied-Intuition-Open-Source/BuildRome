# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Structured-latent (shape) flow model with depth-lifted conditioning.

A TRELLIS.2 img2shape SLat flow transformer on one window's sparse 64^3
tokens, with the depth-lift branch of ``LiftedSSFlowModel``: each token center
is projected into the view, depth-tested against the pack depth, and receives
the DINO feature of its patch, injected before every block through per-block
zero-init linears.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import sparse as sp
from ..modules.lifting import DEFAULT_TAU_M, lift_patch_features_to_tokens
from ..modules.utils import manual_cast
from ..utils.window_conventions import CHUNK_SIZE
from .structured_latent_flow import SLatFlowModel


class LiftedSLatFlowModel(SLatFlowModel):
    def __init__(self, *args, depth_lift_tau_m: float = DEFAULT_TAU_M, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_lift_tau_m = float(depth_lift_tau_m)
        # created after the parent's convert_to: these stay float32
        self.depth_lift_proj_linears = nn.ModuleList(
            [nn.Linear(self.cond_channels, self.model_channels) for _ in range(self.num_blocks)])
        for linear in self.depth_lift_proj_linears:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

    def token_world_coords(self, coords, depth_m, lateral_m) -> torch.Tensor:
        """[N, 4] sparse (batch, x, y, z) voxels -> [N, 3] floor-frame token
        centers (lateral, depth, up) in meters."""
        pitch = CHUNK_SIZE / self.resolution
        local = (coords[:, 1:].float() + 0.5) * pitch
        local[:, 0] -= CHUNK_SIZE / 2.0
        local[:, 1] -= CHUNK_SIZE / 2.0
        b = coords[:, 0].long()
        local[:, 0] += lateral_m.float().view(-1)[b]
        local[:, 1] += depth_m.float().view(-1)[b]
        return local

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: dict,                                  # {"cond_2D": [B, T, D], optional "cond_2D_hi"}
        depth_m: torch.Tensor,                       # [B] window center depth
        lateral_m: torch.Tensor,                     # [B] window center lateral offset
        lift_depth: Optional[torch.Tensor] = None,   # [B, H, W] pack depth (meters)
        lift_valid: Optional[torch.Tensor] = None,   # [B, H, W] bool
        lift_K: Optional[torch.Tensor] = None,       # [B, 3, 3] pixel intrinsics
        lift_c2y: Optional[torch.Tensor] = None,     # [B, 4, 4] camera -> floor frame
        z_off_m: Optional[torch.Tensor] = None,      # unused: SLat tokens are floor-anchored
        cond_hi: Optional[torch.Tensor] = None,      # unused: its encoding rides cond["cond_2D_hi"]
    ) -> sp.SparseTensor:
        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond_2D = manual_cast(cond["cond_2D"], self.dtype)
        if self.pe_mode == "ape":
            h = h + manual_cast(self.pos_embedder(h.coords[:, 1:]), self.dtype)

        lift_feats = None
        if lift_depth is not None:
            lift_src = cond.get("cond_2D_hi")
            if lift_src is None:
                lift_src = cond["cond_2D"]
            lift_feats = lift_patch_features_to_tokens(
                lift_src.float(), self.token_world_coords(h.coords, depth_m, lateral_m), h.coords[:, 0],
                lift_depth, lift_valid, lift_K, lift_c2y, tau_m=self.depth_lift_tau_m)

        for i, block in enumerate(self.blocks):
            if lift_feats is not None:
                h = h.replace(h.feats + manual_cast(self.depth_lift_proj_linears[i](lift_feats), h.feats.dtype))
            h = block(h, t_emb, cond_2D)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.out_layer(h)
