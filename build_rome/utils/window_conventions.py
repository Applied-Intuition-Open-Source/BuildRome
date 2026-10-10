# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""
MDv3 frozen window conventions: continuous chunk offsets + re-frozen camera.

Replaces the grid cell table of ``chunk_conventions.py`` for v3 data
(``rendered_chunks_v3``). ``chunk_conventions.py`` stays untouched: v1 grid
data, its dataloaders and live training runs keep validating against it.

A window is a 3 m cube, floor-anchored (z in [0, CHUNK_SIZE]), placed in the
CAMERA-YAW FRAME at a continuous center (lateral_m, depth_m):

    window center = cam_xy + fwd_yaw * depth_m + right_yaw * lateral_m

sampled per view as
    depth_m   ~ U(*DEPTH_RANGE_M)          (uniform in depth, NOT area-weighted)
    lateral_m ~ U(-w(depth)-H, +w(depth)+H)   H = CHUNK_SIZE / 2
with w(d) = frustum_half_width(d) at the frozen FOV. A window is accepted iff
its cube INTERSECTS the frustum (not center-inside); straddling windows are
intended — inference windows straddle too.

Camera model is re-frozen: FOV_DEG with render-time jitter +-FOV_JITTER_DEG,
NOT stored per sample (deployment input is a center square crop of a phone
photo, ~53-58 deg square-crop FOV on 24-26 mm-equiv phones). Exporters must
assert their lens constants against the stamp at startup.

Occupancy classification (visible-INSTANCE geometry only — architecture never
counts, matching v1 semantics where bare walls/floor never triggered a cell):
distinct VOXEL_RES^3 voxels of the cube containing instance geometry, with
two thresholds EMPTY_MAX_VOX < SLIVER_MAX_VOX:

    <  EMPTY_MAX_VOX   -> "empty":    sidecar only (target = z_empty), no GLB,
                                      never enters stage 2
    <= SLIVER_MAX_VOX  -> "sliver":   normal chunk (never labeled empty)
    >  SLIVER_MAX_VOX  -> "occupied": normal chunk

Rules carried over unchanged from v1: the sidecar is the single source of
truth (directory names are ``chunks/w{k:02d}/``, index only, no floats in
dirnames); dataloaders must call ``validate_sidecar()`` and hard-reject on
mismatch — reject, never remap; ``debug_*`` fields are visualization only.
"""

import math

CHUNK_SIZE = 3.0  # meters, cube edge; cube is floor-anchored (z in [0, 3])

# --- Camera (re-frozen; fixes the v1 45-60 deg FOV drift) ---
FOV_DEG = 55.0
FOV_JITTER_DEG = 2.0  # render-time jitter, NOT stored per sample

# --- Window sampling ---
DEPTH_RANGE_M = (0.5, 7.5)  # window CENTER depth
LATERAL_MARGIN_M = CHUNK_SIZE / 2.0  # lateral bound: |lateral| <= w(depth) + margin
N_WINDOWS = 10  # uniform draws per view (before stratification top-up)

# --- Occupancy classification ---
VOXEL_RES = 64  # voxelization of the cube for instance-vertex counting
EMPTY_MAX_VOX = 32  # strictly below -> empty
SLIVER_MAX_VOX = 512  # (EMPTY_MAX_VOX..SLIVER_MAX_VOX] -> sliver; above -> occupied
OCCUPANCY_CLASSES = ("empty", "sliver", "occupied")

# --- Stratification (per shard, not per view) ---
DEPTH_BIN_M = 1.0  # ledger bins: 1 m depth bins x occupancy class
TOPUP_MAX_PER_VIEW = 4  # targeted draws appended after the uniform N_WINDOWS
TOPUP_MIN_BIN_FRAC = 0.02  # a bin is deficient below this fraction of the shard total


def frustum_half_width(depth_m: float) -> float:
    """Horizontal frustum half-width at depth d for the frozen FOV."""
    return depth_m * math.tan(math.radians(FOV_DEG) / 2.0)


def classify_occupancy(n_voxels: int) -> str:
    """Instance-voxel count -> occupancy class (thresholds EMPTY < SLIVER)."""
    if n_voxels < EMPTY_MAX_VOX:
        return "empty"
    if n_voxels <= SLIVER_MAX_VOX:
        return "sliver"
    return "occupied"


def depth_bin(depth_m: float) -> int:
    """Ledger depth bin index for a window-center depth."""
    lo, hi = DEPTH_RANGE_M
    assert lo <= depth_m <= hi, f"depth {depth_m} outside {DEPTH_RANGE_M}"
    return min(int((depth_m - lo) / DEPTH_BIN_M), int((hi - lo) / DEPTH_BIN_M) - 1)


N_DEPTH_BINS = int((DEPTH_RANGE_M[1] - DEPTH_RANGE_M[0]) / DEPTH_BIN_M)


def window_dirname(k: int) -> str:
    """Stable window dir stem, index only: coordinates live in meta.json."""
    return f"w{k:02d}"


# Stamp written into every sidecar. JSON-native types only so that
# `meta["conventions"] == CONVENTIONS` holds after a json round-trip.
CONVENTIONS = {
    "version": 3,
    "chunk_size_m": CHUNK_SIZE,
    "continuous_offsets": True,
    "depth_range_m": list(DEPTH_RANGE_M),
    "fov_deg": FOV_DEG,
    "fov_jitter_deg": FOV_JITTER_DEG,
    "inference_input": "center_square_crop",
}


def validate_sidecar(meta: dict) -> None:
    """Hard-assert a v3 window sidecar. Range checks replace table membership;
    reject, never remap."""
    assert meta["conventions"] == CONVENTIONS, (
        f"conventions mismatch: sidecar has {meta['conventions']}, expected {CONVENTIONS}"
    )
    depth, lateral = float(meta["depth_m"]), float(meta["lateral_m"])
    assert DEPTH_RANGE_M[0] <= depth <= DEPTH_RANGE_M[1], (
        f"depth_m {depth} outside {DEPTH_RANGE_M}"
    )
    assert abs(lateral) <= frustum_half_width(depth) + LATERAL_MARGIN_M + 1e-6, (
        f"lateral_m {lateral} exceeds w({depth}) + {LATERAL_MARGIN_M} = "
        f"{frustum_half_width(depth) + LATERAL_MARGIN_M:.3f}"
    )
    assert meta["occupancy_class"] in OCCUPANCY_CLASSES, (
        f"unknown occupancy_class {meta['occupancy_class']!r}"
    )
