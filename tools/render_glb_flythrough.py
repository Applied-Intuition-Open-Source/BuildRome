# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Render a room-exploration flythrough of a scene.glb along a spline camera path.

The camera path editor's versioned JSON supports open or closed interpolating
cubic paths, per-segment durations, and per-keyframe FOV. Legacy keyframe lists
keep their original periodic, chord-length spline behavior.

Keyframes can be given explicitly (--keyframes JSON) or auto-generated from the
mesh bounding box: 16 keyframes wrapping around the room TWICE (one closed
spline, so the two turns blend seamlessly), the first turn wide at eye height
and the second smoothly dipping tighter and lower. Each keyframe looks ahead
along the path (walkthrough feel). Scene GLBs from image_to_scene_val.py are
in world space: floor at z=0, meters, z-up.

Usage:
    python tools/render_glb_flythrough.py \
        --glb results/inference/val_gt/<name>/scene.glb --gpu 1

    # custom keyframes (legacy list or camera-path-editor JSON)
    python tools/render_glb_flythrough.py --glb scene.glb \
        --keyframes trajectory.json --y-up
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python tools/render_glb_flythrough.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spline-path flythrough render of a scene GLB.")
    p.add_argument("--glb", required=True, help="Mesh file (.glb; anything trimesh loads).")
    p.add_argument("--out", default=None, help="Output video (default: <glb dir>/flythrough.mp4).")
    p.add_argument("--keyframes", default=None,
                   help="Camera path editor JSON or legacy list of pos/target keyframes. "
                        "Omit to auto-generate an interior loop from the mesh bbox.")
    p.add_argument("--num-frames", type=int, default=None,
                   help="Frame count override (editor JSON value or 240 by default).")
    p.add_argument("--fps", type=int, default=None,
                   help="Output FPS override (editor JSON value or 30 by default).")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--far", type=float, default=None,
                   help="Far clip plane in meters (default: auto = 2x the scene "
                        "bbox diagonal, min 100 — nothing in-scene gets clipped).")
    p.add_argument("--minimap-res", type=int, default=256,
                   help="Top-view minimap size in px (0 disables the inset).")
    p.add_argument("--out-height", type=int, default=None,
                   help="Center-crop the square render to <resolution> x <out-height> "
                        "(e.g. --resolution 1280 --out-height 720 -> 1280x720). The "
                        "per-frame fov is treated as the VERTICAL fov of the cropped "
                        "frame, so editor framing is preserved.")
    p.add_argument("--fov", type=float, default=None,
                   help="FOV override in degrees (editor path values or 70 by default).")
    p.add_argument("--eye-height", type=float, default=1.6, help="Auto-path camera height (m).")
    p.add_argument("--loop-scale", type=float, default=0.55,
                   help="Auto-path loop size as a fraction of the bbox half-extent.")
    p.add_argument("--turns", type=int, default=2,
                   help="Number of times the auto path wraps around the room (orbit mode).")
    p.add_argument("--y-up", action="store_true",
                   help="Input GLB is glTF Y-up (e.g. image_to_scene.py "
                        "scene.glb): convert to the Z-up world frame on load.")
    p.add_argument("--path-mode", default="walk", choices=["walk", "orbit"],
                   help="walk = A* loop through free floor space at constant eye "
                        "height (human walkthrough); orbit = legacy ellipse loop.")
    p.add_argument("--clearance", type=float, default=0.30,
                   help="walk mode: min distance to obstacles (m).")
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def auto_keyframes(vmin, vmax, eye_height: float, loop_scale: float, n_keys: int = 16,
                   n_turns: int = 2):
    """Interior path wrapping ``n_turns`` times around the room, as ONE closed loop.

    The 16 keyframes span two revolutions of an ellipse inside the bbox while
    the radius and height are smoothly modulated over the WHOLE path (wide at
    eye height on the first turn, dipping tighter and lower on the second,
    back to the start): a single periodic spline through all of them gives a
    seamless flythrough with no cut between the turns.
    """
    import numpy as np

    cx, cy = (vmin[0] + vmax[0]) / 2, (vmin[1] + vmax[1]) / 2
    rx = max(0.5, (vmax[0] - vmin[0]) / 2 * loop_scale)
    ry = max(0.5, (vmax[1] - vmin[1]) / 2 * loop_scale)
    z_eye = min(eye_height, vmin[2] + 0.9 * (vmax[2] - vmin[2]))

    t = np.linspace(0, 1, n_keys, endpoint=False)
    angles = 2 * np.pi * n_turns * t
    # 0 -> 1 -> 0 over the full path (periodic): drives the wide->tight->wide
    # radius and the eye-height dip so turn 2 differs from turn 1 smoothly.
    dip = 0.5 - 0.5 * np.cos(2 * np.pi * t)
    radial = 1.0 - 0.4 * dip
    z = z_eye - 0.5 * dip
    positions = np.stack([cx + rx * radial * np.cos(angles),
                          cy + ry * radial * np.sin(angles), z], axis=1)
    # Look at the NEXT keyframe (walkthrough), slightly below the camera.
    targets = np.roll(positions, -1, axis=0).copy()
    targets[:, 2] = np.roll(z, -1) - 0.25
    return positions, targets


def build_walk_path(vertices_np, faces_np, eye_height: float, clearance: float,
                    grid_res: float = 0.05, n_waypoints: int = 6):
    """Human-walkthrough loop through the room's free floor space.

    1. 2D occupancy at body height: cells blocked by geometry in the standing
       band (0.25..1.75 m), walkable only where floor exists below 0.25 m.
    2. Clearance map (distance transform), obstacles dilated by ``clearance``.
    3. Waypoints: farthest-point-sampled high-clearance cells, ordered into a
       tour by angle around their centroid.
    4. A* between consecutive waypoints with a cost that hugs open space.

    Returns (positions[K,3], targets[K,3]) keyframes for the periodic spline,
    or None if the scene has too little walkable space (caller falls back to
    the orbit path).
    """
    import heapq

    import cv2
    import numpy as np

    vmin = vertices_np.min(axis=0)
    vmax = vertices_np.max(axis=0)
    pad = 0.2
    x0, y0 = vmin[0] - pad, vmin[1] - pad
    W = max(8, int(np.ceil((vmax[0] - vmin[0] + 2 * pad) / grid_res)))
    H = max(8, int(np.ceil((vmax[1] - vmin[1] + 2 * pad) / grid_res)))

    # Face centroids classify cells: floor support vs body-height obstacles.
    tri = vertices_np[faces_np]                       # (F, 3, 3)
    cen = tri.mean(axis=1)
    gx = np.clip(((cen[:, 0] - x0) / grid_res).astype(int), 0, W - 1)
    gy = np.clip(((cen[:, 1] - y0) / grid_res).astype(int), 0, H - 1)

    floor = np.zeros((H, W), dtype=np.uint8)
    block = np.zeros((H, W), dtype=np.uint8)
    is_floor = cen[:, 2] < 0.25
    is_block = (cen[:, 2] >= 0.25) & (cen[:, 2] <= 1.75)
    floor[gy[is_floor], gx[is_floor]] = 1
    block[gy[is_block], gx[is_block]] = 1

    # Close small floor holes (SS scenes are voxel-derived and gappy).
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    free = ((floor > 0) & (block == 0)).astype(np.uint8)

    # Clearance in meters, then require the body radius everywhere.
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * grid_res
    walkable = dist > clearance
    if walkable.sum() < 50:
        return None

    def to_world(iy, ix):
        return np.array([x0 + (ix + 0.5) * grid_res, y0 + (iy + 0.5) * grid_res])

    # Waypoints: FPS over comfortably-open cells (median clearance cut keeps
    # them off the walls), angular order -> a roughly convex walking tour.
    ys, xs = np.where(walkable)
    open_cut = max(clearance * 1.5, float(np.median(dist[ys, xs])))
    cand = np.stack([ys, xs], axis=1)[dist[ys, xs] >= open_cut]
    if len(cand) < n_waypoints:
        cand = np.stack([ys, xs], axis=1)
    pts = cand.astype(np.float64)
    sel = [int(np.argmax(dist[cand[:, 0], cand[:, 1]]))]  # start at max clearance
    d2 = np.full(len(pts), np.inf)
    for _ in range(1, n_waypoints):
        d2 = np.minimum(d2, ((pts - pts[sel[-1]]) ** 2).sum(axis=1))
        sel.append(int(np.argmax(d2)))
    way = cand[sel]
    center = way.mean(axis=0)
    way = way[np.argsort(np.arctan2(way[:, 0] - center[0], way[:, 1] - center[1]))]

    # A* between consecutive waypoints; cost prefers high clearance.
    penalty = 1.0 + 4.0 * np.exp(-np.maximum(dist - clearance, 0.0) / 0.4)
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    def astar(a, b):
        start, goal = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
        heap = [(0.0, start)]
        g = {start: 0.0}
        came = {}
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            for dy, dx, step in nbrs:
                ny, nx = cur[0] + dy, cur[1] + dx
                if not (0 <= ny < H and 0 <= nx < W) or not walkable[ny, nx]:
                    continue
                ng = g[cur] + step * penalty[ny, nx]
                if ng < g.get((ny, nx), np.inf):
                    g[(ny, nx)] = ng
                    came[(ny, nx)] = cur
                    h = np.hypot(ny - goal[0], nx - goal[1])
                    heapq.heappush(heap, (ng + h, (ny, nx)))
        return None

    loop_cells = []
    for i in range(len(way)):
        seg = astar(way[i], way[(i + 1) % len(way)])
        if seg is None:  # disconnected free space: straight-line fallback
            seg = [tuple(way[i]), tuple(way[(i + 1) % len(way)])]
        loop_cells.extend(seg[:-1])
    if len(loop_cells) < 8:
        return None

    # Downsample the dense cell path to evenly-spaced spline keyframes.
    xy = np.array([to_world(iy, ix) for iy, ix in loop_cells])
    seglen = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seglen)])[:-1]
    n_keys = min(24, max(8, int(arclen[-1] / 0.75)))
    keys_at = np.linspace(0, arclen[-1], n_keys, endpoint=False)
    key_idx = np.searchsorted(arclen, keys_at)
    key_xy = xy[np.clip(key_idx, 0, len(xy) - 1)]

    z_eye = min(eye_height, vmin[2] + 0.9 * (vmax[2] - vmin[2]))
    positions = np.concatenate([key_xy, np.full((len(key_xy), 1), z_eye)], axis=1)
    # Gaze: at the NEXT keyframe, slightly below eye level (natural forward
    # look-down while walking). The periodic spline smooths the turns.
    targets = np.roll(positions, -1, axis=0).copy()
    targets[:, 2] = z_eye - 0.25
    return positions, targets


def spline_path(points, num_frames: int):
    """Periodic (closed-loop) cubic spline through points, chord-length parameterized.

    points: [K, 3] -> [num_frames, 3]
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    closed = np.vstack([points, points[:1]])  # periodic bc needs first == last
    chords = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(chords)])
    t /= t[-1]
    spline = CubicSpline(t, closed, bc_type="periodic", axis=0)
    u = np.linspace(0, 1, num_frames, endpoint=False)
    return spline(u)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import numpy as np
    import torch
    import trimesh
    import utils3d

    from tools.camera_trajectory import (
        parse_camera_trajectory,
        sample_camera_trajectory,
    )
    from build_rome.pipelines.render import write_video_file
    from build_rome.representations import Mesh
    from build_rome.utils.render_utils import get_renderer

    torch.set_grad_enabled(False)

    loaded = trimesh.load(args.glb, force="mesh", process=False)
    if args.y_up:
        # glTF Y-up -> Blender Z-up world: (x, y, z) -> (x, -z, y)
        v = np.asarray(loaded.vertices)
        loaded.vertices = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    vertices = torch.tensor(np.asarray(loaded.vertices), dtype=torch.float32, device="cuda")
    faces = torch.tensor(np.asarray(loaded.faces), dtype=torch.int32, device="cuda")
    mesh = Mesh(vertices, faces)
    vmin = vertices.min(dim=0).values.cpu().numpy()
    vmax = vertices.max(dim=0).values.cpu().numpy()
    print(f"[mesh] {vertices.shape[0]} verts, {faces.shape[0]} faces, "
          f"bbox {np.round(vmin, 2).tolist()} .. {np.round(vmax, 2).tolist()}")

    path_closed = True
    fps = args.fps or 30
    num_frames = args.num_frames or 240
    if args.keyframes:
        with open(args.keyframes, "r") as f:
            keyframe_data = json.load(f)
        trajectory = parse_camera_trajectory(
            keyframe_data,
            default_num_frames=num_frames,
            default_fps=fps,
            default_fov=args.fov or 70.0,
        )
        num_frames = args.num_frames or trajectory.num_frames
        fps = args.fps or trajectory.fps
        path_closed = trajectory.closed
        if trajectory.legacy:
            # Preserve the original renderer's legacy-list interpolation exactly.
            positions = trajectory.positions
            targets = trajectory.targets
            cam_pos = spline_path(positions, num_frames)
            cam_tgt = spline_path(targets, num_frames)
            fov_degrees = np.full(num_frames, args.fov or trajectory.fovs[0])
        else:
            cam_pos, cam_tgt, fov_degrees = sample_camera_trajectory(
                trajectory, num_frames=num_frames
            )
            if args.fov is not None:
                fov_degrees.fill(args.fov)
    elif args.path_mode == "walk":
        walk = build_walk_path(
            vertices.cpu().numpy(), faces.cpu().numpy().astype(np.int64),
            args.eye_height, args.clearance,
        )
        if walk is None:
            print("[path] WARN: too little walkable floor space; falling back to orbit")
            positions, targets = auto_keyframes(vmin, vmax, args.eye_height, args.loop_scale,
                                                n_turns=args.turns)
        else:
            positions, targets = walk
            print(f"[path] walk mode: {positions.shape[0]} keyframes through free floor space")
    else:
        positions, targets = auto_keyframes(vmin, vmax, args.eye_height, args.loop_scale,
                                            n_turns=args.turns)
    if not args.keyframes:
        cam_pos = spline_path(positions, num_frames)
        cam_tgt = spline_path(targets, num_frames)
        fov_degrees = np.full(num_frames, args.fov or 70.0)
    print(f"[path] {len(cam_pos)} frames "
          f"({'closed' if path_closed else 'open'} spline, {fps} fps)")

    up = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    far = args.far or max(100.0, 2.0 * float(np.linalg.norm(vmax - vmin)))
    print(f"[render] far plane {far:.0f} m")
    renderer = get_renderer(mesh, resolution=args.resolution, near=0.05, far=far,
                            chunk_size=5_000_000)

    # ------------------------------------------------- top-view minimap (once)
    # Ceiling-cut dollhouse render straight down, plus the world->pixel mapping
    # so the camera path / frustum can be drawn on it.
    import cv2

    mini_res = args.minimap_res
    minimap_frame = None
    if mini_res > 0:
        cut = vertices[:, 2] > 2.2  # same dollhouse height as the spiral renders
        keep = ~cut[faces.long()].any(dim=1)
        top_fov = torch.deg2rad(torch.tensor(60.0)).cuda()
        cx, cy = (vmin[0] + vmax[0]) / 2, (vmin[1] + vmax[1]) / 2
        half = max(vmax[0] - vmin[0], vmax[1] - vmin[1]) / 2 * 1.15
        z_cam = vmax[2] + half / np.tan(np.deg2rad(30.0))
        top_eye = torch.tensor([cx, cy, z_cam], dtype=torch.float32, device="cuda")
        top_tgt = torch.tensor([cx, cy, 0.0], dtype=torch.float32, device="cuda")
        top_up = torch.tensor([0.0, 1.0, 0.0], device="cuda")  # straight-down view
        top_extr = utils3d.torch.extrinsics_look_at(top_eye, top_tgt, top_up)
        top_intr = utils3d.torch.intrinsics_from_fov(fov_x=top_fov, fov_y=top_fov)
        top_renderer = get_renderer(Mesh(vertices, faces[keep]), resolution=mini_res,
                                    near=0.05, far=1000.0, chunk_size=5_000_000)
        top_img = top_renderer.render(Mesh(vertices, faces[keep]), top_extr, top_intr,
                                      return_types=["normal"])["normal"]
        top_img = np.clip(top_img.detach().cpu().numpy().transpose(1, 2, 0) * 255,
                          0, 255).astype(np.uint8)

        def to_minimap_px(pts_xyz: np.ndarray) -> np.ndarray:
            """World points -> minimap pixel coords via the top view's pinhole."""
            p = torch.tensor(pts_xyz, dtype=torch.float32, device="cuda")
            cam = p @ top_extr[:3, :3].T + top_extr[:3, 3]
            uv = cam @ top_intr.T
            uv = uv[:, :2] / uv[:, 2:3]  # normalized [0,1] image coords
            return (uv.cpu().numpy() * mini_res)

        path_px = to_minimap_px(cam_pos).astype(np.int32)
        pos_px = to_minimap_px(cam_pos)
        tgt_px = to_minimap_px(cam_tgt)
        # Camera path in red, drawn once on the base minimap.
        cv2.polylines(top_img, [path_px.reshape(-1, 1, 2)], isClosed=path_closed,
                      color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        def minimap_frame(i: int) -> np.ndarray:
            """Minimap with the current camera drawn as a frustum triangle."""
            mini = top_img.copy()
            p = pos_px[i]
            fwd = tgt_px[i] - p
            norm = np.linalg.norm(fwd)
            if norm < 1e-6:
                return mini
            fwd /= norm
            half_ang = np.deg2rad(fov_degrees[i] / 2)
            length = mini_res * 0.12
            tri = [p]
            for s in (+1.0, -1.0):
                c, sn = np.cos(s * half_ang), np.sin(s * half_ang)
                tri.append(p + length * np.array([c * fwd[0] - sn * fwd[1],
                                                  sn * fwd[0] + c * fwd[1]]))
            tri = np.array(tri, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mini, [tri], color=(255, 255, 0), lineType=cv2.LINE_AA)
            cv2.polylines(mini, [tri], isClosed=True, color=(0, 0, 0), thickness=1,
                          lineType=cv2.LINE_AA)
            return mini

    frames = []
    for i in range(num_frames):
        eye = torch.tensor(cam_pos[i], dtype=torch.float32, device="cuda")
        tgt = torch.tensor(cam_tgt[i], dtype=torch.float32, device="cuda")
        extr = utils3d.torch.extrinsics_look_at(eye, tgt, up)
        fov = torch.deg2rad(torch.tensor(float(fov_degrees[i]))).cuda()
        if args.out_height:
            # trajectory fov = vertical fov of the CROPPED frame: widen the
            # square render so its center out-height band spans exactly fov
            fov = 2 * torch.atan(torch.tan(fov / 2) * (args.resolution / args.out_height))
        intr = utils3d.torch.intrinsics_from_fov(fov_x=fov, fov_y=fov)
        res = renderer.render(mesh, extr, intr, return_types=["normal"])["normal"]
        frame = np.clip(res.detach().cpu().numpy().transpose(1, 2, 0) * 255,
                        0, 255).astype(np.uint8)
        if args.out_height:
            ch = int(round(frame.shape[0] * args.out_height / args.resolution))
            y0 = (frame.shape[0] - ch) // 2
            frame = frame[y0:y0 + ch]
        if minimap_frame is not None:
            # Overlay the minimap top-right with a thin white border. The
            # renderer's output size can differ from args.resolution (ssaa),
            # so scale the minimap to stay proportional in the actual frame.
            mini = minimap_frame(i)
            if frame.shape[1] != args.resolution:
                s = frame.shape[1] / args.resolution
                mini = cv2.resize(mini, (int(mini.shape[1] * s), int(mini.shape[0] * s)),
                                  interpolation=cv2.INTER_LINEAR)
            h, w = mini.shape[:2]
            frame[:h + 2, -w - 2:] = 255
            frame[1:h + 1, -w - 1:-1] = mini
        frames.append(frame)
        if (i + 1) % 30 == 0:
            print(f"[render] {i + 1}/{num_frames} frames")

    out = args.out or str(Path(args.glb).parent / "flythrough.mp4")
    write_video_file(np.stack(frames), out, fps=fps)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
