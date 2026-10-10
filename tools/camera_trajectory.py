# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Parsing and sampling helpers for versioned camera trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraTrajectory:
    positions: np.ndarray
    targets: np.ndarray
    durations: np.ndarray
    fovs: np.ndarray
    closed: bool
    num_frames: int
    fps: int
    legacy: bool = False


def parse_camera_trajectory(
    data: Any,
    *,
    default_num_frames: int = 240,
    default_fps: int = 30,
    default_fov: float = 70.0,
) -> CameraTrajectory:
    """Parse legacy keyframe lists or the camera-path-editor versioned object."""
    legacy = isinstance(data, list)
    if legacy:
        keys = data
        closed = True
        num_frames = default_num_frames
        fps = default_fps
    else:
        if not isinstance(data, dict):
            raise ValueError("trajectory JSON must be a keyframe list or versioned object")
        if data.get("version") != 1:
            raise ValueError(f"unsupported trajectory version: {data.get('version')!r}")
        if data.get("coordinate_system", "z_up") != "z_up":
            raise ValueError("trajectory coordinates must use the renderer's z_up frame")
        if data.get("interpolation", "cubic_interpolating") != "cubic_interpolating":
            raise ValueError("only cubic_interpolating trajectories are supported")
        keys = data.get("keyframes")
        closed = bool(data.get("closed", False))
        num_frames = int(data.get("num_frames", default_num_frames))
        fps = int(data.get("fps", default_fps))

    if not isinstance(keys, list) or len(keys) < 3:
        raise ValueError("need at least 3 keyframes")
    try:
        positions = np.asarray([key["pos"] for key in keys], dtype=np.float64)
        targets = np.asarray([key["target"] for key in keys], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("each keyframe needs numeric pos and target arrays") from exc
    if positions.shape != (len(keys), 3) or targets.shape != positions.shape:
        raise ValueError("keyframe pos and target values must each contain 3 numbers")
    if not np.isfinite(positions).all() or not np.isfinite(targets).all():
        raise ValueError("keyframe coordinates must be finite")

    segment_count = len(keys) if closed else len(keys) - 1
    durations = np.asarray(
        [float(keys[index].get("duration", 1.0)) for index in range(segment_count)],
        dtype=np.float64,
    )
    fovs = np.asarray(
        [float(key.get("fov", default_fov)) for key in keys],
        dtype=np.float64,
    )
    if not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("keyframe durations must be positive finite numbers")
    if not np.isfinite(fovs).all() or np.any((fovs <= 0) | (fovs >= 180)):
        raise ValueError("keyframe FOV values must be between 0 and 180 degrees")
    if num_frames < 2 or fps < 1:
        raise ValueError("num_frames must be at least 2 and fps must be positive")

    return CameraTrajectory(
        positions=positions,
        targets=targets,
        durations=durations,
        fovs=fovs,
        closed=closed,
        num_frames=num_frames,
        fps=fps,
        legacy=legacy,
    )


def evaluate_interpolating_cubic(
    points: np.ndarray,
    durations: np.ndarray,
    times: np.ndarray,
    *,
    closed: bool,
) -> np.ndarray:
    """Evaluate a C1 cubic Hermite spline that interpolates every control point."""
    points = np.asarray(points, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    count = len(points)
    expected_segments = count if closed else count - 1
    if points.ndim != 2 or count < 2:
        raise ValueError("points must have shape [K, D] with K >= 2")
    if durations.shape != (expected_segments,) or np.any(durations <= 0):
        raise ValueError("durations do not match the trajectory segments")

    tangents = np.empty_like(points)
    if closed:
        for index in range(count):
            previous = (index - 1) % count
            following = (index + 1) % count
            tangents[index] = (
                (points[following] - points[previous])
                / (durations[previous] + durations[index])
            )
    else:
        tangents[0] = (points[1] - points[0]) / durations[0]
        tangents[-1] = (points[-1] - points[-2]) / durations[-1]
        for index in range(1, count - 1):
            tangents[index] = (
                (points[index + 1] - points[index - 1])
                / (durations[index - 1] + durations[index])
            )

    total = float(durations.sum())
    sample_times = np.mod(times, total) if closed else np.clip(times, 0.0, total)
    cumulative = np.concatenate([[0.0], np.cumsum(durations)])
    segments = np.searchsorted(cumulative[1:], sample_times, side="right")
    if not closed:
        segments = np.minimum(segments, count - 2)
    local = sample_times - cumulative[segments]
    u = local / durations[segments]
    following = (segments + 1) % count

    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    dt = durations[segments]
    return (
        h00[:, None] * points[segments]
        + h10[:, None] * dt[:, None] * tangents[segments]
        + h01[:, None] * points[following]
        + h11[:, None] * dt[:, None] * tangents[following]
    )


def sample_camera_trajectory(
    trajectory: CameraTrajectory,
    *,
    num_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample positions, targets, and per-frame FOV from a trajectory."""
    frame_count = num_frames or trajectory.num_frames
    times = np.linspace(
        0.0,
        float(trajectory.durations.sum()),
        frame_count,
        endpoint=not trajectory.closed,
    )
    positions = evaluate_interpolating_cubic(
        trajectory.positions, trajectory.durations, times, closed=trajectory.closed
    )
    targets = evaluate_interpolating_cubic(
        trajectory.targets, trajectory.durations, times, closed=trajectory.closed
    )
    fovs = evaluate_interpolating_cubic(
        trajectory.fovs[:, None], trajectory.durations, times, closed=trajectory.closed
    )[:, 0]
    return positions, targets, np.clip(fovs, 1.0, 179.0)
