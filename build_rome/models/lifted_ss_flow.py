# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Sparse-structure flow model with depth-lifted geometric conditioning.

A TRELLIS.2 SS flow transformer operating on one 3 m window (16^3 latent
tokens), with two branches driven by the view's monocular depth pack, each
injected before every block through per-block zero-init linears (so an
untrained branch is a no-op and warm starts are bit-exact):

* depth lift: every voxel center is projected into the view with the pack's
  camera, depth-tested against the pack depth (only voxels within
  ``depth_lift_tau_m`` of the visible surface pass), and receives the DINO
  feature of the patch it lands in. The gather prefers the hi-res token grid
  ``cond["cond_2D_hi"]`` when present; a CFG-dropped image lifts zeros.
* signed clearance: every voxel center receives s = z_voxel - d_image
  (clamped to +-1 m; negative = observed free space, ~0 = visible surface,
  positive = occluded), Fourier-encoded and MLP'd. Voxels without a trusted
  pixel contribute exact zeros.

Absolute position never enters the model: windows see geometry only through
these branches. Token order matches the DiT flatten: token k = (x*16 + y)*16
+ z with axes (x = lateral, y = depth, z = up) in the window frame.
"""
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.lifting import DEFAULT_TAU_M, lift_depth_clearance_to_grid, lift_patch_features_to_grid
from ..modules.utils import manual_cast
from ..utils.window_conventions import CHUNK_SIZE
from .sparse_structure_flow import SparseStructureFlowModel

# finest sinusoid period = one SS latent cell (CHUNK_SIZE / 16)
COORD_SCALE = 2.0 * math.pi / (CHUNK_SIZE / 16.0)


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
    """[N] -> [N, dim] sin/cos features, identical to TimestepEmbedder's."""
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DepthClearanceEncoder(nn.Module):
    """``[...]`` signed clearance (meters) -> ``[..., cond_channels]``."""

    def __init__(self, cond_channels: int, freq_dim: int = 128):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, cond_channels, bias=True),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels, bias=True),
        )

    def forward(self, clearance_m: torch.Tensor) -> torch.Tensor:
        lead = clearance_m.shape
        feats = sinusoidal_embedding(clearance_m.reshape(-1).float() * COORD_SCALE, self.freq_dim)
        return self.mlp(feats.reshape(*lead, -1))


def _zero_linears(n: int, c_in: int, c_out: int) -> nn.ModuleList:
    linears = nn.ModuleList([nn.Linear(c_in, c_out) for _ in range(n)])
    for linear in linears:
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
    return linears


class LiftedSSFlowModel(SparseStructureFlowModel):
    def __init__(self, *args, depth_lift_tau_m: float = DEFAULT_TAU_M, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_lift_tau_m = float(depth_lift_tau_m)
        # created after the parent's convert_to: these stay float32
        self.depth_lift_proj_linears = _zero_linears(self.num_blocks, self.cond_channels, self.model_channels)
        self.depth_geom_encoder = DepthClearanceEncoder(self.cond_channels)
        self.depth_geom_proj_linears = _zero_linears(self.num_blocks, self.cond_channels, self.model_channels)

        # window-local token centers in meters [R^3, 3], in DiT token order;
        # lateral/depth window-centered, up floor-anchored in [0, CHUNK_SIZE)
        pitch = CHUNK_SIZE / self.resolution
        axes = [torch.arange(self.resolution, dtype=torch.float32) for _ in range(3)]
        local = (torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3) + 0.5) * pitch
        local[:, 0] -= CHUNK_SIZE / 2.0
        local[:, 1] -= CHUNK_SIZE / 2.0
        self.register_buffer("voxel_local_offsets_m", local)

    def token_world_coords(self, depth_m, lateral_m, z_off_m=None) -> torch.Tensor:
        """[B, R^3, 3] floor-frame token centers (lateral, depth, up) in meters.
        z_off_m is the window base height above the fitted floor (None -> 0)."""
        z = z_off_m.float().view(-1) if z_off_m is not None else torch.zeros_like(depth_m.float().view(-1))
        window = torch.stack([lateral_m.float().view(-1), depth_m.float().view(-1), z], dim=-1)
        return self.voxel_local_offsets_m[None] + window[:, None, :]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict,                                  # {"cond_2D": [B, T, D], optional "cond_2D_hi"}
        depth_m: torch.Tensor,                       # [B] window center depth
        lateral_m: torch.Tensor,                     # [B] window center lateral offset
        z_off_m: Optional[torch.Tensor] = None,      # [B] window base above the floor
        lift_depth: Optional[torch.Tensor] = None,   # [B, H, W] pack depth (meters)
        lift_valid: Optional[torch.Tensor] = None,   # [B, H, W] bool
        lift_K: Optional[torch.Tensor] = None,       # [B, 3, 3] pixel intrinsics
        lift_c2y: Optional[torch.Tensor] = None,     # [B, 4, 4] camera -> floor frame
        cond_hi: Optional[torch.Tensor] = None,      # unused: its encoding rides cond["cond_2D_hi"]
    ) -> torch.Tensor:
        B = x.shape[0]
        assert [*x.shape] == [B, self.in_channels, *[self.resolution] * 3], f"input shape {tuple(x.shape)}"
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        h = manual_cast(h, self.dtype)
        cond_2D = manual_cast(cond["cond_2D"], self.dtype)

        lift_feats = geom_feats = None
        if lift_depth is not None:
            coords = self.token_world_coords(depth_m, lateral_m, z_off_m)
            lift_src = cond.get("cond_2D_hi")
            if lift_src is None:
                lift_src = cond["cond_2D"]
            lift_feats = lift_patch_features_to_grid(
                lift_src.float(), coords, lift_depth, lift_valid, lift_K, lift_c2y, tau_m=self.depth_lift_tau_m)
            s, ok = lift_depth_clearance_to_grid(coords, lift_depth, lift_valid, lift_K, lift_c2y)
            geom_feats = self.depth_geom_encoder(s)
            geom_feats = geom_feats * ok.unsqueeze(-1).to(geom_feats.dtype)

        for i, block in enumerate(self.blocks):
            if lift_feats is not None:
                h += manual_cast(self.depth_lift_proj_linears[i](lift_feats), h.dtype)
                h += manual_cast(self.depth_geom_proj_linears[i](geom_feats), h.dtype)
            h = block(h, t_emb, cond_2D, self.rope_phases)

        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        return h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()
