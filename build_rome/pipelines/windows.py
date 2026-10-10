# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Window helpers shared by inference: window transforms, per-window
conditioning, and single-window sampling with RePaint-style pinning.

Windows are 3 m cubes in the view's floor frame (x = lateral, y = depth,
z = up), given as (depth_m, lateral_m, z_off_m). Overlapping windows share
latent cells only when their offsets are integer multiples of the latent
pitch (CHUNK_SIZE / 16).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from build_rome.utils.window_conventions import CHUNK_SIZE

from ..modules.sparse.basic import SparseTensor
from ..pipelines.samplers import FlowEulerGuidanceIntervalSampler


def relative_translations(layout: Sequence[Tuple[float, float, float]]) -> List[torch.Tensor]:
    """Per-window offsets from window 0 in CHUNK_SIZE units, latent axis
    order (x=lateral, y=depth, z=up). Pure translations: all windows share
    one yaw frame. Layout entries are (depth, lateral, z = window base height)."""
    d0, l0, z0 = layout[0]
    return [torch.tensor([(l - l0) / CHUNK_SIZE, (d - d0) / CHUNK_SIZE, (z - z0) / CHUNK_SIZE])
            for d, l, z in layout]


def window_world_transform(depth_m: float, lateral_m: float,
                           z_m: float = 0.0) -> np.ndarray:
    """[4,4] normalized chunk coords -> shared yaw frame (the visualization
    'world'): scale 3, lift to floor anchor (+ the stacked-layer base z_m),
    translate to the window center."""
    return np.array(
        [
            [CHUNK_SIZE, 0.0, 0.0, lateral_m],
            [0.0, CHUNK_SIZE, 0.0, depth_m],
            [0.0, 0.0, CHUNK_SIZE, CHUNK_SIZE / 2.0 + z_m],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------- sampling


def make_window_conditioning(extractor, image_chw: torch.Tensor,
                             layout: Sequence[Tuple[float, float, float]], device):
    """Per-window (cond, neg_cond, chunk_kwargs) for one shared view image.

    The image is DINO-encoded ONCE; every window gets the same cond_2D and a
    zeroed neg_cond. The window position goes into chunk_kwargs so the CFG
    sampler passes it to BOTH the cond and the neg_cond forward — the position
    never takes the null path, matching training.
    """
    feats = extractor(image_chw[None].to(device))  # [1, T, D]
    cond = {"cond_2D": feats}
    neg_cond = {"cond_2D": torch.zeros_like(feats)}
    conds = [cond] * len(layout)
    neg_conds = [neg_cond] * len(layout)
    chunk_kwargs = [
        {
            "depth_m": torch.tensor([d], device=device, dtype=torch.float32),
            "lateral_m": torch.tensor([l], device=device, dtype=torch.float32),
            "z_off_m": torch.tensor([z], device=device, dtype=torch.float32),
        }
        for d, l, z in layout
    ]
    return conds, neg_conds, chunk_kwargs


def make_val_sampler(sigma_min: float) -> FlowEulerGuidanceIntervalSampler:
    """CFG + guidance-interval sampler: consumes guidance_* kwargs so only
    (depth_m, lateral_m) reach the model forward."""
    return FlowEulerGuidanceIntervalSampler(sigma_min)

@torch.no_grad()
def sample_window(sampler, model, noise, cond: dict, neg_cond: dict, steps: int = 50,
                  rescale_t: float = 1.0, guidance_strength: float = 3.0,
                  guidance_interval: Tuple[float, float] = (0.0, 1.0), guidance_rescale: float = 0.0,
                  chunk_kwargs: Optional[dict] = None, inpaint_x0=None, inpaint_mask=None):
    """Euler flow sampling of ONE window, with RePaint-style pinning.

    ``noise`` (dense [1, C, R, R, R] tensor or SparseTensor) only supplies the
    shape; values are drawn here (sparse: one draw per token in world-key
    order). After every step the pinned cells / tokens (``inpaint_mask``) are
    reset onto the exact flow trajectory of ``inpaint_x0``,
    x_t = (1 - t) x0 + (smin + (1 - smin) t) eps, with eps the window's own
    initial noise, so at t = 0 they equal ``inpaint_x0``.
    """
    R = model.resolution
    if isinstance(noise, torch.Tensor):
        x = torch.randn_like(noise)
        eps = x.clone()
    else:
        xyz = noise.coords[:, 1:]
        keys = xyz[:, 0].long() * (R * R) + xyz[:, 1].long() * R + xyz[:, 2].long()
        unique_keys = torch.unique(keys)
        draw = torch.randn(len(unique_keys), noise.feats.shape[1], device=noise.feats.device,
                           dtype=noise.feats.dtype)
        x = SparseTensor(feats=draw[torch.searchsorted(unique_keys, keys)], coords=noise.coords)
        eps = x.feats.clone()
    pin = inpaint_mask is not None and bool(inpaint_mask.any())
    smin = float(getattr(sampler, "sigma_min", 1e-5))

    def _repaint(x, t_now):
        x_obs = (1.0 - t_now) * inpaint_x0 + (smin + (1.0 - smin) * t_now) * eps
        if isinstance(x, torch.Tensor):
            return torch.where(inpaint_mask, x_obs.to(x.dtype), x)
        return x.replace(torch.where(inpaint_mask[:, None], x_obs.to(x.feats.dtype), x.feats))

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = (rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)).tolist()
    for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
        v = sampler._inference_model(model, x, t, cond=cond, neg_cond=neg_cond,
                                     guidance_strength=guidance_strength, guidance_interval=guidance_interval,
                                     guidance_rescale=guidance_rescale, **(chunk_kwargs or {}))
        x = x - (t - t_prev) * v
        if pin:
            x = _repaint(x, t_prev)
    return x


def slice_shared_structure(
    world_coords: torch.Tensor,
    layout: Sequence[Tuple[float, float, float]],
    resolution: int,
) -> Tuple[List[Optional[torch.Tensor]], List[np.ndarray]]:
    """Per-window local coords sliced from ONE shared world voxel set.

    ``world_coords``: [K, 3] integer voxels on the shared lattice
    (resolution cells per CHUNK_SIZE, axes x=lateral, y=depth, z=up).
    Windows whose cube contains no active voxel get None (skipped at the
    SLAT stage — no latent, no participation, mirroring the dataloader's
    exclusion of empty windows). Presence of a world voxel in two windows is
    exact by construction: the sparse scatter key is the world voxel index.
    """
    locals_, offsets = [], []
    for d, l, z in layout:
        off = np.array(
            [
                round(l / CHUNK_SIZE * resolution),
                round(d / CHUNK_SIZE * resolution),
                round(z / CHUNK_SIZE * resolution),
            ],
            dtype=np.int64,
        )
        local = world_coords - torch.from_numpy(off).to(world_coords.device)[None, :]
        inside = ((local >= 0) & (local < resolution)).all(dim=1)
        offsets.append(off)
        locals_.append(local[inside].contiguous() if bool(inside.any()) else None)
    return locals_, offsets
