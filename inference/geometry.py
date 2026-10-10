# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Monocular geometry for conditioning: the MoGe-3 depth pack (depth, camera,
gravity-aligned floor frame) and helpers on its point cloud."""
import hashlib
import os
import subprocess

import torch

# depth-discontinuity rtol for the layout cloud (bleed streaks between a
# foreground edge and the background)
EDGE_RTOL = 0.04


def ensure_moge3_pack(image_path: str, moge3_python: str, cache_dir: str) -> str:
    """Build (or reuse) the MoGe-3 pack for an image, cached by image content.

    MoGe-3's dependency pins conflict with the TRELLIS environment, so the pack
    is built by inference/moge3_pack_once.py under ``moge3_python`` (which may
    be the current interpreter when both live in one environment)."""
    with open(image_path, "rb") as f:
        digest = hashlib.sha1(f.read() + b"|center-crop|cascade").hexdigest()[:12]  # salted with the pack recipe
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(cache_dir, exist_ok=True)
    pack_path = os.path.join(cache_dir, f"{stem}_{digest}_moge3.npz")
    if os.path.isfile(pack_path):
        print(f"[moge3] cached pack: {pack_path}")
        return pack_path
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moge3_pack_once.py")
    print(f"[moge3] building pack with {moge3_python} ...")
    subprocess.run([moge3_python, runner, "--image", image_path, "--out", pack_path], check=True)
    return pack_path


def load_lift_kwargs(pack_path: str, device) -> dict:
    """Depth pack -> the conditioning tensors both denoisers consume (batch 1):
    metric depth, validity, pixel intrinsics, camera-to-floor-frame transform."""
    from build_rome.modules.lifting import load_da3_pack

    pack = load_da3_pack(pack_path)
    assert bool(pack["floor_ok"]), f"pack {pack_path}: floor fit failed"
    print(f"[geometry] pack {pack_path}: camera height {float(pack['camera_height']):.2f} m")
    return {
        "lift_depth": pack["depth"].float()[None].to(device),
        "lift_valid": (~pack["sky"].bool())[None].to(device),
        "lift_K": pack["K"].float()[None].to(device),
        "lift_c2y": pack["cam_to_yaw"].float()[None].to(device),
    }


def apply_scene_scale(lift_kwargs: dict, scale: float) -> dict:
    """Uniformly scale the evidence about the camera's floor footpoint. K is
    unchanged: uniform scaling preserves every pixel projection."""
    lift_kwargs["lift_depth"] = lift_kwargs["lift_depth"] * scale
    c2y = lift_kwargs["lift_c2y"].clone()
    c2y[:, :3, 3] *= scale
    lift_kwargs["lift_c2y"] = c2y
    return lift_kwargs


@torch.no_grad()
def filtered_cloud(lift_kwargs: dict) -> torch.Tensor:
    """Metric floor-frame cloud [N, 3] (lateral, depth, up) with depth-edge
    pixels removed. A per-pixel edge test is used instead of a density-based
    flyer filter, whose fixed voxel size deletes far surfaces (density falls
    as 1/z^2)."""
    from build_rome.modules.lifting import backproject_depth, depth_edge_mask

    depth = lift_kwargs["lift_depth"][0]
    valid = lift_kwargs["lift_valid"][0] & (depth > 1e-6)
    valid = valid & ~depth_edge_mask(depth, valid, EDGE_RTOL)
    c2y = lift_kwargs["lift_c2y"][0]
    pts = backproject_depth(depth, lift_kwargs["lift_K"][0]) @ c2y[:3, :3].T + c2y[:3, 3]
    return pts[valid.reshape(-1)]
