# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Depth-pack geometry and depth-grounded feature lifting (single definition).

Everything that interprets a per-view depth pack lives here: back-projection
and the gravity-aligned floor fit that WRITE packs (inference/moge3_pack_once),
and the depth-tested lift / signed clearance that the models READ them with.
One definition guarantees overlapping windows receive bit-identical features
at shared positions, and that train/test conditioning match.

Frames and conventions:

* CAMERA frame: OpenCV pinhole: x right, y down, z forward (depth).
* FLOOR (camera-yaw) frame, the window frame: x lateral (right), y depth
  (camera forward projected to horizontal), z up; floor at z = 0, camera at
  (0, 0, camera_height). Window centers are (lateral_m, depth_m) in it.
* The pack's ``K`` is pixel-space for the pack's own ``depth`` resolution;
  projection normalizes by (W, H), so a pack works with any resize of the
  same image.

A depth pack (npz) carries::

    depth          fp16 [H, W]   metric depth (meters)
    conf           fp16 [H, W]   confidence (ones when the estimator has none)
    sky            bool [H, W]   invalid / sky mask
    K              fp32 [3, 3]   pixel-space intrinsics for [H, W]
    cam_to_yaw     fp32 [4, 4]   camera frame -> floor frame (rigid)
    camera_height  fp32 scalar   meters above the fitted floor
    floor_ok       bool          floor fit succeeded
    floor_inlier_frac fp32       inlier fraction of the fitted plane
    version        int           DA3_PACK_VERSION
    model          str           estimator that produced the pack
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DA3_PACK_VERSION = 1

# Depth-test tolerance: ~1.5x the SS latent pitch (3 m / 16 = 0.1875 m).
DEFAULT_TAU_M = 0.28

# Signed-clearance saturation for the depth-only geometry lift: fine detail
# near the surface comes from the Fourier encoding; beyond +-1 m the signal
# is just "well in front of" / "well behind" the visible surface.
DEFAULT_CLEARANCE_CLAMP_M = 1.0

# Level-camera fallback height when the floor fit fails (typical eye/phone
# height; the pack's floor_ok=False flag marks these views as low-confidence).
FALLBACK_CAMERA_HEIGHT_M = 1.6

# OpenCV camera frame: y points DOWN, so "up" prior is -y.
_UP_PRIOR = (0.0, -1.0, 0.0)


# ------------------------------------------------------------ back-projection


def backproject_depth(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Metric depth map -> [H*W, 3] points in the CAMERA frame.

    ``depth`` [H, W] (meters), ``K`` [3, 3] pixel-space for that resolution.
    """
    H, W = depth.shape
    v, u = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=torch.float32),
        torch.arange(W, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth.float()
    x = (u + 0.5 - cx) / fx * z
    y = (v + 0.5 - cy) / fy * z
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


# ------------------------------------------------------------------ floor fit


def depth_normals(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Per-pixel surface normals [H, W, 3] in the CAMERA frame from a metric
    depth map (central differences on the back-projected point grid),
    oriented toward the camera (n . view_dir < 0). Border pixels get zeros.
    """
    H, W = depth.shape
    pts = backproject_depth(depth, K).reshape(H, W, 3)
    dx = pts[1:-1, 2:] - pts[1:-1, :-2]
    dy = pts[2:, 1:-1] - pts[:-2, 1:-1]
    n = torch.cross(dx, dy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    # Orient toward the camera: the ray to the point is pts itself.
    flip = (n * pts[1:-1, 1:-1]).sum(dim=-1) > 0
    n = torch.where(flip.unsqueeze(-1), -n, n)
    out = torch.zeros(H, W, 3, device=depth.device)
    out[1:-1, 1:-1] = n
    return out


def fit_floor_plane(
    points_cam: torch.Tensor,
    normals_cam: torch.Tensor,
    max_points: int = 60000,
    inlier_tau_m: float = 0.05,
    max_tilt_deg: float = 35.0,
    height_range_m: Tuple[float, float] = (0.3, 4.0),
    min_inlier_frac: float = 0.005,
    height_bin_m: float = 0.06,
    mode_min_frac: float = 0.15,
    anchor_height: bool = False,
) -> Tuple[torch.Tensor, float, float, bool]:
    """Gravity-then-height floor fit in the CAMERA frame.

    Returns ``(up_cam, camera_height_m, inlier_frac, ok)`` where ``up_cam``
    is the unit floor normal pointing from the floor toward the camera and
    ``inlier_frac`` is the floor fraction of the (subsampled) point cloud.

    Two decoupled, mode-seeking stages instead of plane RANSAC (which is
    brittle when the floor covers ~10% of the image):

    1. GRAVITY from normals: every horizontal surface (floor, seats, tables,
       ceiling flipped) shares the up direction, so the mean of the up-ish
       normal cluster — refined by two reselection rounds at 15 deg — is a
       far larger consensus set than any single plane.
    2. FLOOR HEIGHT as the lowest populated mode of the 1D height histogram
       ``h = -up . p`` over points whose OWN normal is up-ish: the floor is
       the lowest horizontal surface with meaningful support.

    A final least-squares plane on the floor inliers refines both. Failure
    returns ``ok=False`` with the height prior (FALLBACK_CAMERA_HEIGHT_M) and
    a TWO-TIER up: once stage 1 has a normal consensus, later failures (floor
    not visible / refinement diverged) keep that estimated up — a pitched
    camera still gets a gravity-correct frame, only the height is a guess.
    Only a stage-1 failure falls back to the level-camera up prior.
    """
    up_prior = torch.tensor(_UP_PRIOR, device=points_cam.device)
    fallback = (up_prior, FALLBACK_CAMERA_HEIGHT_M, 0.0, False)

    finite = torch.isfinite(points_cam).all(dim=1)
    pts, nrm = points_cam[finite], normals_cam[finite]
    if pts.shape[0] < 100:
        return fallback
    if pts.shape[0] > max_points:
        idx = torch.randperm(
            pts.shape[0], generator=torch.Generator(device="cpu").manual_seed(0)
        )[:max_points].to(pts.device)
        pts, nrm = pts[idx], nrm[idx]
    N = pts.shape[0]

    # -- 1. gravity: mean-shift on the up-ish normal cluster ----------------
    flip = (nrm @ up_prior) < 0
    nrm = torch.where(flip.unsqueeze(-1), -nrm, nrm)
    real = nrm.norm(dim=1) > 0.5  # border pixels are zeroed
    up = up_prior
    for cos_sel in (math.cos(math.radians(max_tilt_deg)), math.cos(math.radians(15.0)), math.cos(math.radians(15.0))):
        sel = real & ((nrm @ up) >= cos_sel)
        if int(sel.sum()) < 50:
            return fallback
        up = nrm[sel].mean(dim=0)
        up = up / up.norm().clamp(min=1e-9)

    # Gravity is trusted from here on: any later failure anchors the floor at
    # the EVIDENCE'S BOTTOM instead of a blind height prior (GenRecon-style
    # percentile bounds, 1D): h = -(up . p) grows downward, so a high
    # quantile of h is the underside of the point pile. It is always defined,
    # cannot land on a rooftop (never the bottom), and is robust to the <2%
    # of below-ground spike outliers that min() would chase. Where a floor is
    # visible this coincides with it; where the mode/refinement stages fail
    # (tiny visible floor, MoGe-2 scale drift outside height_range, sparse
    # up-ish normals in vegetation) this keeps a usable gravity-correct frame
    # rather than the level-camera 1.6 m guess. Reported ok=True: the frame
    # is anchored to evidence; the carve-side ray cross-check still guards
    # data generation.
    h_all = -(pts @ up)
    anchor_h = float(torch.quantile(h_all, 0.98).clamp(min=0.15))
    anchor_frac = float(((h_all - anchor_h).abs() < inlier_tau_m).float().mean())
    fallback = (up, anchor_h, anchor_frac, True)
    if anchor_height:
        # anchor-primary mode (--ground anchor): ground = the
        # percentile bottom of the projected points, consensus up, no
        # plane-fit stage. Deterministic and always defined once gravity is.
        return fallback

    # -- 2. floor height: lowest strong mode of h = -up.p (up-ish points) ---
    horiz = real & ((nrm @ up) >= math.cos(math.radians(20.0)))
    h = -(pts @ up)
    h_sel = h[horiz]
    h_sel = h_sel[(h_sel >= height_range_m[0]) & (h_sel <= height_range_m[1])]
    if h_sel.shape[0] < min_inlier_frac * N:
        return fallback
    nbins = int((height_range_m[1] - height_range_m[0]) / height_bin_m) + 1
    counts = torch.histc(h_sel, bins=nbins, min=height_range_m[0], max=height_range_m[1])
    strong = counts >= mode_min_frac * counts.max()
    # Highest h = lowest surface (floor is furthest below the camera).
    pick = int(strong.nonzero(as_tuple=True)[0].max())
    h_floor = height_range_m[0] + (pick + 0.5) * (height_range_m[1] - height_range_m[0]) / nbins

    # -- 3. refine: least-squares plane on the floor band -------------------
    band = horiz & ((h - h_floor).abs() < 2 * inlier_tau_m)
    if int(band.sum()) < min_inlier_frac * N:
        return fallback
    inl = pts[band]
    centroid = inl.mean(dim=0)
    _, _, Vh = torch.linalg.svd(inl - centroid, full_matrices=False)
    up_ref = Vh[-1]
    if (up_ref @ up_prior) < 0:
        up_ref = -up_ref
    if (up_ref @ up) < math.cos(math.radians(max_tilt_deg)):
        return fallback  # refinement diverged from the normal consensus
    # Plane through the centroid with normal `up_ref`; the camera sits at
    # the origin, so its signed height above the floor is -up . centroid.
    height = float(-(up_ref @ centroid))
    inlier_frac = float(inl.shape[0] / N)
    if not (height_range_m[0] <= height <= height_range_m[1]):
        return fallback
    return up_ref, height, inlier_frac, True


def camera_to_yaw_transform(up_cam: torch.Tensor, camera_height_m: float) -> torch.Tensor:
    """[4, 4] rigid transform: CAMERA frame -> CAMERA-YAW frame.

    Yaw frame: x lateral (camera right projected to horizontal), y depth
    (camera forward projected to horizontal), z up; camera at
    (0, 0, camera_height), floor at z = 0.
    """
  
    up = up_cam / up_cam.norm().clamp(min=1e-9)
    fwd_cam = torch.tensor([0.0, 0.0, 1.0], device=up.device)
    fwd = fwd_cam - (fwd_cam @ up) * up
    fwd = fwd / fwd.norm().clamp(min=1e-9)
    right = torch.cross(fwd, up, dim=0)  # (right, fwd, up) right-handed
    T = torch.eye(4, device=up.device)
    T[0, :3] = right
    T[1, :3] = fwd
    T[2, :3] = up
    T[2, 3] = camera_height_m
    return T


# ------------------------------------------------------------------ da3 pack


def depth_edge_mask(depth: torch.Tensor, valid: torch.Tensor, rtol: float) -> torch.Tensor:
    """[H, W] bool: pixels whose 3x3 neighborhood spans a RELATIVE depth jump
    greater than ``rtol`` (the MoGe-2 demo's "Remove edges" postprocessing,
    utils3d.depth_edge equivalent). Invalid pixels neither fire nor spread
    into their neighbors. The SINGLE definition shared by inference
    (--moge2-remove-edges) and the dataloader's edge_drop_rtol augmentation,
    so train and test conditioning get the exact same holes."""
    big = 1e6
    d = (depth.float() * valid)[None, None]
    v = valid[None, None]
    dmax = F.max_pool2d(d - big * (~v), 3, 1, 1)
    dmin = -F.max_pool2d(-(d + big * (~v)), 3, 1, 1)
    return ((dmax > dmin.clamp_min(1e-6) * (1.0 + rtol)) & v)[0, 0]


def load_da3_pack(file) -> dict:
    """npz path/file-like -> dict of torch tensors (fp32) + python scalars."""
    data = np.load(file)
    assert int(data["version"]) == DA3_PACK_VERSION, (
        f"da3 pack version {int(data['version'])} != code version {DA3_PACK_VERSION}"
    )
    return {
        "depth": torch.from_numpy(data["depth"]).float(),
        "conf": torch.from_numpy(data["conf"]).float(),
        "sky": torch.from_numpy(data["sky"]),
        "K": torch.from_numpy(data["K"]).float(),
        "cam_to_yaw": torch.from_numpy(data["cam_to_yaw"]).float(),
        "camera_height": float(data["camera_height"]),
        "floor_ok": bool(data["floor_ok"]),
        "floor_inlier_frac": float(data["floor_inlier_frac"]),
    }


# --------------------------------------------------------------- feature lift


def lift_patch_features_to_grid(
    feats: torch.Tensor,          # [B, T, D] patch tokens AFTER the global tokens offset
    coords_yaw_m: torch.Tensor,   # [B, V, 3] absolute camera-yaw coords (lateral, depth, up)
    depth: torch.Tensor,          # [B, H, W] metric depth (the pack's map)
    valid_px: Optional[torch.Tensor],  # [B, H, W] bool (conf/sky mask) or None
    K: torch.Tensor,              # [B, 3, 3] pixel-space intrinsics for [H, W]
    cam_to_yaw: torch.Tensor,     # [B, 4, 4]
    tau_m: Optional[float] = DEFAULT_TAU_M,
    global_tokens: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Depth-tested gather of image patch features onto 3D grid points.

    For every query point: transform to the camera frame, project with K,
    keep only points whose camera depth matches the DA3 depth at that pixel
    within ``tau_m`` (surface voxels; free-space and occluded voxels get
    zeros), then gather the DINO patch feature at the projected patch.

    Determinism: identical (coords_yaw_m row, pack, feats) -> bit-identical
    output row, regardless of which window the row belongs to.

    Returns [B, V, D] (zeros where the depth test or masks reject).
    """
    B, T, D = feats.shape
    patch_res = int(round(math.sqrt(T - global_tokens)))
    assert patch_res * patch_res == T - global_tokens, (
        f"token count {T} minus {global_tokens} global tokens is not a square"
    )
    H, W = depth.shape[-2:]

    # yaw -> camera: p_cam = R^T (p_yaw - t)
    R = cam_to_yaw[:, :3, :3]                      # [B, 3, 3]
    t = cam_to_yaw[:, :3, 3]                       # [B, 3]
    p_cam = torch.einsum("bij,bvi->bvj", R, coords_yaw_m.float() - t[:, None, :])  # [B, V, 3]

    z = p_cam[..., 2]
    valid = z > eps
    safe_z = torch.where(valid, z, torch.ones_like(z))
    u = K[:, 0, 0, None] * p_cam[..., 0] / safe_z + K[:, 0, 2, None]  # [B, V] pixels
    v = K[:, 1, 1, None] * p_cam[..., 1] / safe_z + K[:, 1, 2, None]
    un, vn = u / W, v / H  # normalized uv in [0, 1)
    valid &= (un >= 0.0) & (un < 1.0) & (vn >= 0.0) & (vn < 1.0)

    if tau_m is not None:
        # Depth test at the nearest pixel (nearest, not bilinear: interpolating
        # across depth discontinuities manufactures phantom surfaces).
        px = (u.long()).clamp(0, W - 1)
        py = (v.long()).clamp(0, H - 1)
        flat = (py * W + px).reshape(B, -1)                       # [B, V]
        d_img = depth.reshape(B, -1).float().gather(1, flat)      # [B, V]
        valid &= (z - d_img).abs() < tau_m
        if valid_px is not None:
            valid &= valid_px.reshape(B, -1).gather(1, flat)

    pu = (un * patch_res).long().clamp(0, patch_res - 1)
    pv = (vn * patch_res).long().clamp(0, patch_res - 1)
    token_ids = pv * patch_res + pu + global_tokens               # [B, V]
    gathered = feats.gather(1, token_ids.unsqueeze(-1).expand(-1, -1, D))

    return gathered * valid.unsqueeze(-1).to(gathered.dtype)


def lift_patch_features_to_tokens(
    feats: torch.Tensor,          # [B, T, D] patch tokens AFTER the global tokens offset
    coords_yaw_m: torch.Tensor,   # [N, 3] absolute camera-yaw coords (lateral, depth, up)
    batch_idx: torch.Tensor,      # [N] which sample each token belongs to
    depth: torch.Tensor,          # [B, H, W] metric depth (the pack's map)
    valid_px: Optional[torch.Tensor],  # [B, H, W] bool (conf/sky mask) or None
    K: torch.Tensor,              # [B, 3, 3] pixel-space intrinsics for [H, W]
    cam_to_yaw: torch.Tensor,     # [B, 4, 4]
    tau_m: Optional[float] = DEFAULT_TAU_M,
    global_tokens: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``lift_patch_features_to_grid`` for FLAT sparse tokens.

    Same projection / depth test / gather per point, but the queries are a
    flat [N, 3] token list with a per-token batch index (the SLat models'
    native layout) instead of a dense [B, V, 3] grid. Identical inputs give
    bit-identical rows to the dense function (the SS/SLat consistency the
    grid version guarantees across windows).

    Returns [N, D] (zeros where the depth test or masks reject).
    """
    B, T, D = feats.shape
    patch_res = int(round(math.sqrt(T - global_tokens)))
    assert patch_res * patch_res == T - global_tokens, (
        f"token count {T} minus {global_tokens} global tokens is not a square"
    )
    H, W = depth.shape[-2:]
    b = batch_idx.long()

    # yaw -> camera: p_cam = R^T (p_yaw - t), camera gathered per token.
    R = cam_to_yaw[b, :3, :3]                      # [N, 3, 3]
    t = cam_to_yaw[b, :3, 3]                       # [N, 3]
    p_cam = torch.einsum("nij,ni->nj", R, coords_yaw_m.float() - t)  # [N, 3]

    z = p_cam[:, 2]
    valid = z > eps
    safe_z = torch.where(valid, z, torch.ones_like(z))
    u = K[b, 0, 0] * p_cam[:, 0] / safe_z + K[b, 0, 2]  # [N] pixels
    v = K[b, 1, 1] * p_cam[:, 1] / safe_z + K[b, 1, 2]
    un, vn = u / W, v / H  # normalized uv in [0, 1)
    valid &= (un >= 0.0) & (un < 1.0) & (vn >= 0.0) & (vn < 1.0)

    # Depth test at the nearest pixel (nearest, not bilinear: interpolating
    # across depth discontinuities manufactures phantom surfaces).
    px = (u.long()).clamp(0, W - 1)
    py = (v.long()).clamp(0, H - 1)
    d_img = depth[b, py, px].float()                              # [N]
    valid &= (z - d_img).abs() < tau_m
    if valid_px is not None:
        valid &= valid_px[b, py, px]

    pu = (un * patch_res).long().clamp(0, patch_res - 1)
    pv = (vn * patch_res).long().clamp(0, patch_res - 1)
    token_ids = pv * patch_res + pu + global_tokens               # [N]
    gathered = feats[b, token_ids]                                # [N, D]
    return gathered * valid.unsqueeze(-1).to(gathered.dtype)


def lift_depth_clearance_to_grid(
    coords_yaw_m: torch.Tensor,   # [B, V, 3] absolute camera-yaw coords (lateral, depth, up)
    depth: torch.Tensor,          # [B, H, W] metric depth (the pack's map)
    valid_px: Optional[torch.Tensor],  # [B, H, W] bool (conf/sky mask) or None
    K: torch.Tensor,              # [B, 3, 3] pixel-space intrinsics for [H, W]
    cam_to_yaw: torch.Tensor,     # [B, 4, 4]
    clamp_m: float = DEFAULT_CLEARANCE_CLAMP_M,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Signed depth clearance of 3D grid points against the depth map.

    For every query point: transform to the camera frame, project with K,
    read the depth at the nearest pixel, and return the signed clearance
    ``s = z_point - d_img`` clamped to ``+-clamp_m``: negative = observed
    free space, ~0 = on the visible surface, positive = occluded/unknown.

    Unlike ``lift_patch_features_to_grid`` there is NO depth-band test —
    the band comparison IS the signal. Validity only means "projected to a
    trusted pixel": in front of the camera, in frame, ``valid_px`` true,
    and ``d_img > eps`` (GT packs use depth 0 for "no geometry").

    Determinism: identical (coords_yaw_m row, pack) -> bit-identical output
    row, regardless of which window the row belongs to.

    Returns ``(s, valid)``: [B, V] float (zeros where invalid), [B, V] bool.
    """
    B = depth.shape[0]
    H, W = depth.shape[-2:]

    # yaw -> camera: p_cam = R^T (p_yaw - t)
    R = cam_to_yaw[:, :3, :3]                      # [B, 3, 3]
    t = cam_to_yaw[:, :3, 3]                       # [B, 3]
    p_cam = torch.einsum("bij,bvi->bvj", R, coords_yaw_m.float() - t[:, None, :])  # [B, V, 3]

    z = p_cam[..., 2]
    valid = z > eps
    safe_z = torch.where(valid, z, torch.ones_like(z))
    u = K[:, 0, 0, None] * p_cam[..., 0] / safe_z + K[:, 0, 2, None]  # [B, V] pixels
    v = K[:, 1, 1, None] * p_cam[..., 1] / safe_z + K[:, 1, 2, None]
    un, vn = u / W, v / H  # normalized uv in [0, 1)
    valid &= (un >= 0.0) & (un < 1.0) & (vn >= 0.0) & (vn < 1.0)

    # Nearest pixel, matching lift_patch_features_to_grid (bilinear across
    # depth discontinuities manufactures phantom surfaces).
    px = (u.long()).clamp(0, W - 1)
    py = (v.long()).clamp(0, H - 1)
    flat = (py * W + px).reshape(B, -1)                           # [B, V]
    d_img = depth.reshape(B, -1).float().gather(1, flat)          # [B, V]
    valid &= d_img > eps
    if valid_px is not None:
        valid &= valid_px.reshape(B, -1).gather(1, flat)
   
    s = (z - d_img).clamp(-clamp_m, clamp_m)
    return s * valid.to(s.dtype), valid


