# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Scene-wide joint decoding of SS latents and shape SLats.

All chunk SLats are merged onto a single global-frame ``SparseTensor``, the
decoder runs once (or in spatial groups for very large scenes), and a single
scene mesh is extracted via one joint call to ``flexible_dual_grid_to_mesh``.

Joint frame convention:
  * Origin at ``chunk_centers[0]`` (the chunker's reference chunk center).
  * Scale: ``1 / chunk_size`` (same as chunk-local frames), so ``m_c2o[0]``
    from the chunker is the joint→world transform.
  * Scene aabb in this frame: per axis ``[min_rel_t - 0.5, max_rel_t + 0.5]``,
    where ``rel_t`` is the chunker's ``relative_translation`` (unit =
    ``chunk_size``).

Chunked decode (1024 pipeline only):
  * For ``len(slats) > max_chunks_per_group``, chunks are spatially grouped
    via recursive longest-axis bisection until each leaf has
    ``≤ max_chunks_per_group`` chunks.
  * Each group's joint slat is inflated by ``overlap_r_in`` voxels (R_in
    scale) so the decoder's receptive field is intact at owned-region
    boundaries; outputs are filtered back to the owned region and concatenated.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from o_voxel.convert import flexible_dual_grid_to_mesh

from ..models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from ..modules.sparse import SparseTensor
from ..representations import Mesh


def _merge_chunk_slats(
    slats: List[SparseTensor],
    relative_transl: List[torch.Tensor],
    R_in: int,
    weights: Optional[List[torch.Tensor]] = None,
) -> Tuple[SparseTensor, Dict]:
    """Merge per-chunk SLats into one global-frame batch-1 ``SparseTensor``.

    Each chunk's coords are shifted by ``(offset_k - mins)`` so all chunks
    share a single (0, 0, 0)-origin lattice. Overlap voxels are deduplicated
    and features averaged, weighted per token by ``weights`` when given
    ([N_k] per chunk). With autoregressive one-at-a-time sampling the windows
    agree only where tokens were pinned; a window's uncommitted border ring
    holds its own, different sample, and averaging two valid-but-different
    latents decodes to shredded geometry along the seam. Passing weight 0 for
    border tokens makes each overlap voxel take the latent of the window it
    is interior to (equal-weight mean where every chunk has weight 0).

    Returns ``(joint_slat, meta)`` with ``meta = {"mins", "maxs", "R_in"}``
    at the coarse input scale.
    """
    device = slats[0].feats.device
    dtype = slats[0].feats.dtype

    offsets = [((t.to(device) * R_in).round().long()) for t in relative_transl]
    mins = torch.stack(offsets).min(dim=0).values
    maxs = torch.stack(offsets).max(dim=0).values + R_in
    GY, GZ = (maxs - mins)[1].item(), (maxs - mins)[2].item()

    global_coords = []
    global_feats = []
    global_w = []
    for k, (slat, offset) in enumerate(zip(slats, offsets)):
        assert slat.coords[:, 0].max() == 0, "expected batch index 0 in each chunk"
        xyz = slat.coords[:, 1:] + (offset - mins).unsqueeze(0)
        global_coords.append(xyz)
        global_feats.append(slat.feats)
        global_w.append(weights[k].to(device=device, dtype=dtype).reshape(-1) if weights is not None
                        else torch.ones(slat.feats.shape[0], device=device, dtype=dtype))
    global_coords = torch.cat(global_coords, dim=0)
    global_feats = torch.cat(global_feats, dim=0)
    global_w = torch.cat(global_w, dim=0)
    D = global_feats.shape[1]

    keys = global_coords[:, 0].long() * (GY * GZ) + global_coords[:, 1].long() * GZ + global_coords[:, 2].long()
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    K_u = unique_keys.shape[0]

    # voxels where every contributing chunk has weight 0 fall back to the plain mean
    wsum = torch.zeros(K_u, 1, device=device, dtype=dtype)
    wsum.scatter_add_(0, inv.unsqueeze(1), global_w.unsqueeze(1))
    w = torch.where(wsum[inv, 0] > 0, global_w, torch.ones_like(global_w))
    acc = torch.zeros(K_u, D, device=device, dtype=dtype)
    cnt = torch.zeros(K_u, 1, device=device, dtype=dtype)
    acc.scatter_add_(0, inv.unsqueeze(1).expand(-1, D), global_feats * w.unsqueeze(1))
    cnt.scatter_add_(0, inv.unsqueeze(1), w.unsqueeze(1))
    avg = acc / cnt

    ux = (unique_keys // (GY * GZ)).int()
    uy = ((unique_keys // GZ) % GY).int()
    uz = (unique_keys % GZ).int()
    batch_col = torch.zeros(K_u, 1, dtype=torch.int32, device=device)
    unique_coords = torch.cat([batch_col, torch.stack([ux, uy, uz], dim=1)], dim=1)

    joint_slat = SparseTensor(feats=avg, coords=unique_coords)
    meta = {"mins": mins, "maxs": maxs, "R_in": R_in}
    return joint_slat, meta


def _make_groups(
    relative_transl: List[torch.Tensor],
    R_in: int,
    max_chunks_per_group: int,
    joint_slat: Optional[SparseTensor] = None,
    overlap_r_in: int = 16,
    max_inflated_voxels: Optional[int] = None,
) -> Tuple[List[List[int]], List[int]]:
    """Recursive longest-axis median bisection over chunk centers.

    A leaf is accepted only if BOTH:
      - ``len(leaf) <= max_chunks_per_group``, AND
      - the inflated AABB of the leaf contains ``<= max_inflated_voxels``
        voxels in ``joint_slat`` (when both are provided).

    The inflated count is the *actual* number of voxels the per-group decoder
    forward will consume (after merge dedup, including overlap from
    neighboring chunks) — it is the direct memory governor at res=1024, not
    the pre-merge owned voxel sum. Single-chunk leaves are returned even if
    they exceed the cap (cannot bisect further).

    Returns ``(groups, group_inflated_counts)``. The counts are zero when no
    measurement was performed (joint_slat or max_inflated_voxels missing).
    """
    centers = torch.stack([t.float() * R_in + R_in / 2 for t in relative_transl])

    do_measure = joint_slat is not None and max_inflated_voxels is not None
    if do_measure:
        device = joint_slat.feats.device
        offsets = torch.stack([(t.to(device) * R_in).round().long() for t in relative_transl])
        mins_global = offsets.min(dim=0).values
        offsets_minus_mins = offsets - mins_global  # [K, 3]
        coords_xyz = joint_slat.coords[:, 1:]

        def inflated_count(indices: List[int]) -> int:
            chunk_origins = offsets_minus_mins[torch.tensor(indices, device=device)]
            owned_min = chunk_origins.min(dim=0).values
            owned_max = chunk_origins.max(dim=0).values + R_in
            inf_min = (owned_min - overlap_r_in).to(coords_xyz.dtype)
            inf_max = (owned_max + overlap_r_in).to(coords_xyz.dtype)
            mask = ((coords_xyz >= inf_min) & (coords_xyz < inf_max)).all(dim=1)
            return int(mask.sum().item())

    else:

        def inflated_count(indices: List[int]) -> int:  # pragma: no cover
            return 0

    def is_leaf(indices: List[int]) -> bool:
        if len(indices) <= 1:
            return True  # cannot bisect further
        if len(indices) > max_chunks_per_group:
            return False
        if do_measure and inflated_count(indices) > max_inflated_voxels:
            return False
        return True

    def bisect(indices: List[int]) -> List[List[int]]:
        if is_leaf(indices):
            return [indices]
        sub = centers[torch.tensor(indices)]
        ranges = sub.max(dim=0).values - sub.min(dim=0).values
        axis = int(ranges.argmax().item())
        order = torch.argsort(sub[:, axis]).tolist()
        sorted_indices = [indices[i] for i in order]
        mid = len(sorted_indices) // 2
        return bisect(sorted_indices[:mid]) + bisect(sorted_indices[mid:])

    groups = bisect(list(range(len(relative_transl))))
    counts = [inflated_count(g) for g in groups] if do_measure else [0] * len(groups)
    return groups, counts


def _filter_sparse_to_aabb(
    st: SparseTensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> SparseTensor:
    """Return a new SparseTensor with only voxels whose xyz coord lies in
    ``[aabb_min, aabb_max)``. Coord frame is preserved (no shift)."""
    coords = st.coords
    aabb_min = aabb_min.to(device=coords.device, dtype=coords.dtype)
    aabb_max = aabb_max.to(device=coords.device, dtype=coords.dtype)
    xyz = coords[:, 1:]
    mask = ((xyz >= aabb_min) & (xyz < aabb_max)).all(dim=1)
    return SparseTensor(
        feats=st.feats[mask].contiguous(),
        coords=coords[mask].contiguous(),
    )


def _group_owned_aabb(
    group_chunk_idx: List[int],
    offsets_minus_mins: List[torch.Tensor],
    R_in: int,
) -> torch.Tensor:
    """Owned AABB ``[2, 3]`` at R_in scale in joint-frame integer coords."""
    origins = torch.stack([offsets_minus_mins[i] for i in group_chunk_idx])
    return torch.stack([origins.min(dim=0).values, origins.max(dim=0).values + R_in])


@torch.no_grad()
def joint_decode_sparse_structure(
    dec,
    z_s_list: List[torch.Tensor],
    relative_transl: List[torch.Tensor],
    ss_res: int,
) -> List[torch.Tensor]:
    """Joint sparse-structure decode: place every chunk's latent on a merged
    dense canvas (mean in overlaps), run the SS decoder once, max-pool to the
    ``ss_res`` lattice, split back per chunk.

    Returns a list of ``[1, 1, ss_res, ss_res, ss_res]`` occupancy-logit
    tensors, one per chunk.

    Why this is valid: the SS decoder is a dense 3D CNN with only
    channel-wise normalization (GroupNorm / ChannelLayerNorm), so it is
    translation- and size-equivariant. Running it on a larger merged input
    gives the same result as running it per-chunk with neighbor-aware
    boundaries — no training-distribution shift.
    """
    device = z_s_list[0].device
    dtype = z_s_list[0].dtype

    R_lat = z_s_list[0].shape[-1]
    offsets_lat = [(t.to(device) * R_lat).round().long() for t in relative_transl]
    mins_lat = torch.stack(offsets_lat).min(dim=0).values
    maxs_lat = torch.stack(offsets_lat).max(dim=0).values + R_lat
    B, C = z_s_list[0].shape[:2]
    GX, GY, GZ = (maxs_lat - mins_lat).tolist()

    accum = torch.zeros(B, C, GX, GY, GZ, device=device, dtype=dtype)
    count = torch.zeros(1, 1, GX, GY, GZ, device=device, dtype=dtype)
    for z, off in zip(z_s_list, offsets_lat):
        x, y, zc = (off - mins_lat).tolist()
        accum[:, :, x : x + R_lat, y : y + R_lat, zc : zc + R_lat] += z
        count[:, :, x : x + R_lat, y : y + R_lat, zc : zc + R_lat] += 1
    joint_z = accum / count.clamp(min=1)
    del accum, count

    decoder_dtype = next(dec.parameters()).dtype
    max_cells = int(os.environ.get("JOINT_SS_DEC_MAX_CELLS", 1_500_000))
    if GX * GY * GZ > max_cells:
        # Canvas too large for one decoder pass (activation memory scales
        # with cells; a 2x-upscaled scene OOMs an 80 GB card). Tile along the
        # longest horizontal axis with a 16-latent-cell halo — comfortably
        # above the conv decoder's receptive field, so by the same
        # translation-equivariance argument as above the halo-cropped slabs
        # stitch to the full-canvas result.
        halo = 16
        axis = 2 if GX >= GY else 3          # dims: [B, C, GX, GY, GZ]
        L = joint_z.shape[axis]
        n_slabs = math.ceil(GX * GY * GZ / max_cells)
        seg = math.ceil(L / n_slabs)
        pieces = []
        for s0 in range(0, L, seg):
            s1 = min(s0 + seg, L)
            a0, a1 = max(s0 - halo, 0), min(s1 + halo, L)
            sl_in = [slice(None)] * 5
            sl_in[axis] = slice(a0, a1)
            out = dec(joint_z[tuple(sl_in)].to(dtype=decoder_dtype))
            up = out.shape[axis] // (a1 - a0)
            sl_out = [slice(None)] * 5
            sl_out[axis] = slice((s0 - a0) * up, (s1 - a0) * up)
            pieces.append(out[tuple(sl_out)])
            del out
            torch.cuda.empty_cache()
        decoded_joint = torch.cat(pieces, dim=axis)
        del pieces
        print(f"[joint-ss-dec] canvas {GX}x{GY}x{GZ} latent cells: decoded in "
              f"{math.ceil(L / seg)} slabs along axis {axis} (halo {halo})")
    else:
        decoded_joint = dec(joint_z.to(dtype=decoder_dtype))

    # Per-chunk: decoder upsamples R_lat → R_dec_per_chunk. Same factor at
    # joint scale. Max-pool joint output to ss_res lattice if needed.
    upscale = decoded_joint.shape[-1] // joint_z.shape[-1]
    per_chunk_dec_res = R_lat * upscale
    ratio = per_chunk_dec_res // ss_res
    if ratio > 1:
        decoded_joint = F.max_pool3d(decoded_joint.float(), ratio, ratio, 0)

    # Split at ss_res scale. Offsets scale linearly with resolution.
    offsets_ss = [(t.to(device) * ss_res).round().long() for t in relative_transl]
    mins_ss = torch.stack(offsets_ss).min(dim=0).values
    per_chunk_occ: List[torch.Tensor] = []
    for off in offsets_ss:
        x, y, zc = (off - mins_ss).tolist()
        per_chunk_occ.append(decoded_joint[:, :, x : x + ss_res, y : y + ss_res, zc : zc + ss_res].clone())
    return per_chunk_occ


@torch.no_grad()
def joint_decode_shape(
    decoder,
    slats: List[SparseTensor],
    relative_transl: List[torch.Tensor],
    *,
    max_chunks_per_group: Optional[int] = None,
    max_inflated_voxels: Optional[int] = None,
    overlap_r_in: int = 16,
    weights: Optional[List[torch.Tensor]] = None,
) -> Mesh:
    """Joint shape decode + joint mesh extraction.

    ``weights``: optional per-chunk, per-token merge weights for overlap
    voxels (see :func:`_merge_chunk_slats`).

    With ``max_chunks_per_group is None`` (or ``len(slats) <=`` it AND every
    chunk is under the voxel cap) the decoder runs once on the full merged
    joint slat (original behavior). Otherwise the chunks are spatially grouped
    and decoded group-by-group with overlap.

    Both caps act as governors on per-group memory. The voxel cap (sum of
    pre-merge ``slats[i].coords.shape[0]`` over a group) is the real memory
    constraint at res=1024 — chunk-count alone misses density variation.
    """
    num_up = len(decoder.num_blocks) - 1
    R_in = decoder.resolution // (2**num_up)
    upscale = decoder.resolution // R_in

    joint_slat, meta = _merge_chunk_slats(slats, relative_transl, R_in, weights=weights)
    device = joint_slat.feats.device

    # Poisoned latents (NaN feats) make every to_subdiv logit compare False,
    # which empties the decoder's upsample and crashes the next sparse conv —
    # surface it per source chunk instead.
    nan_rows = int(torch.isnan(joint_slat.feats).any(dim=1).sum())
    if nan_rows:
        per_chunk = [int(torch.isnan(s.feats).any(dim=1).sum()) for s in slats]
        print(f"[joint_decode_shape] WARNING: {nan_rows:,}/{joint_slat.feats.shape[0]:,} merged "
              f"voxels have NaN feats (per chunk: {per_chunk})")

    chunk_cap_active = max_chunks_per_group is not None and len(slats) > max_chunks_per_group
    voxel_cap_active = max_inflated_voxels is not None and int(joint_slat.coords.shape[0]) > max_inflated_voxels
    use_chunked = chunk_cap_active or voxel_cap_active

    # Scene aabb in chunk_size units (joint frame). Derivation: chunk 0's
    # voxel 0 must map to chunk-0-local position -0.5 + 0.5/R_fine, which
    # forces aabb[0] = mins_coarse/R_in - 0.5 per axis.
    aabb = torch.stack(
        [
            meta["mins"].float() / R_in - 0.5,
            meta["maxs"].float() / R_in - 0.5,
        ],
        dim=0,
    ).to(device)
    grid_size_fine = ((meta["maxs"] - meta["mins"]) * upscale).int()
    vm = decoder.voxel_margin

    if not use_chunked:
        # Single-pass: original behavior. Decode → channel-extract → mesh-extract once.
        h_joint = SparseUnetVaeDecoder.forward(decoder, joint_slat)
        vertices = h_joint.replace((1 + 2 * vm) * F.sigmoid(h_joint.feats[..., 0:3]) - vm)
        intersected = h_joint.replace(h_joint.feats[..., 3:6] > 0)
        quad_lerp = h_joint.replace(F.softplus(h_joint.feats[..., 6:7]))
        meshes: List[Mesh] = [
            Mesh(
                *flexible_dual_grid_to_mesh(
                    v.coords[:, 1:],
                    v.feats,
                    i.feats,
                    q.feats,
                    aabb=aabb,
                    grid_size=grid_size_fine,
                    train=False,
                )
            )
            for v, i, q in zip(vertices, intersected, quad_lerp)
        ]
        assert len(meshes) == 1, f"expected batch size 1, got {len(meshes)}"
        return meshes[0]

    # ── Chunked: spatially group chunks, decode + mesh-extract per group ──
    # We never form a global h_joint. ``flexible_dual_grid_to_mesh`` allocates
    # an O(N * 36) intermediate (edge_neighbor_voxel) that exceeds 80 GB at
    # full-scene N. Per-group extraction caps that allocation at per-group
    # voxel count. Each group mesh-extracts on its INFLATED voxel set so
    # boundary adjacency is intact, then keeps only faces whose centroid is
    # in the OWNED AABB. Owned AABBs are unions of overlapping 3 m windows,
    # so neighboring groups both claim the shared overlap slab — the merge
    # below dedupes vertices (by voxel coord) AND faces (by vertex triple)
    # so the final mesh is single-layered and shares vertices across group
    # boundaries.
    offsets = [(t.to(device) * R_in).round().long() for t in relative_transl]
    mins = torch.stack(offsets).min(dim=0).values
    offsets_minus_mins = [(o - mins) for o in offsets]

    chunk_cap = max_chunks_per_group if max_chunks_per_group is not None else len(slats)
    groups, group_inflated_counts = _make_groups(
        relative_transl,
        R_in,
        chunk_cap,
        joint_slat=joint_slat,
        overlap_r_in=overlap_r_in,
        max_inflated_voxels=max_inflated_voxels,
    )
    print(
        f"[joint_decode_shape] chunked: {len(slats)} chunks → {len(groups)} groups "
        f"(sizes: {[len(g) for g in groups]}, inflated-voxel counts: {group_inflated_counts})"
    )

    mesh_v_parts: List[torch.Tensor] = []  # CPU; per-group owned vertices [V_g, 3] float
    mesh_f_parts: List[torch.Tensor] = []  # CPU; per-group owned faces [F_g, 3] int (group-local indices)
    voxel_coords_parts: List[torch.Tensor] = []  # CPU; per-group voxel coords [V_g, 3] int — dedup keys

    for g_idx, group in enumerate(groups):
        owned_aabb = _group_owned_aabb(group, offsets_minus_mins, R_in)
        inflated_aabb = torch.stack([owned_aabb[0] - overlap_r_in, owned_aabb[1] + overlap_r_in])

        slat_g = _filter_sparse_to_aabb(joint_slat, inflated_aabb[0], inflated_aabb[1])
        n_in = slat_g.feats.shape[0]
        try:
            h_g = SparseUnetVaeDecoder.forward(decoder, slat_g)
        except IndexError as e:
            # Empty intermediate in the decoder (all subdiv logits <= 0, e.g.
            # NaN latents): drop the group instead of killing the scene.
            g_nan = int(torch.isnan(slat_g.feats).any(dim=1).sum())
            print(f"[joint_decode_shape] WARNING: group {g_idx} (chunks={group}, in={n_in:,}, "
                  f"nan_voxels={g_nan:,}) failed to decode ({e}) — SKIPPING group")
            torch.cuda.empty_cache()
            continue

        # Channel extraction on the inflated h_g (small enough per-group).
        v_g = h_g.replace((1 + 2 * vm) * F.sigmoid(h_g.feats[..., 0:3]) - vm)
        i_g = h_g.replace(h_g.feats[..., 3:6] > 0)
        q_g = h_g.replace(F.softplus(h_g.feats[..., 6:7]))

        # Mesh-extract on this group's inflated voxel set. coords are joint-frame
        # fine-scale integers; aabb/grid_size are scene-wide so output positions
        # are in the joint world frame and consistent across groups.
        coords_g = v_g.coords[:, 1:]  # [V_g, 3] int
        mesh_v_g, mesh_f_g = flexible_dual_grid_to_mesh(
            coords_g,
            v_g.feats,
            i_g.feats,
            q_g.feats,
            aabb=aabb,
            grid_size=grid_size_fine,
            train=False,
        )

        # Filter faces to those whose centroid lies in owned region (in world).
        # owned_world = owned_aabb_R_in / R_in + aabb[0].
        owned_world_min = (owned_aabb[0].float() / R_in + aabb[0]).to(mesh_v_g.dtype)
        owned_world_max = (owned_aabb[1].float() / R_in + aabb[0]).to(mesh_v_g.dtype)
        face_centroids = mesh_v_g[mesh_f_g.long()].mean(dim=1)  # [F_g, 3]
        in_owned = ((face_centroids >= owned_world_min) & (face_centroids < owned_world_max)).all(dim=1)
        mesh_f_owned = mesh_f_g[in_owned]

        # Compact: keep only vertices referenced by owned faces.
        used_v = torch.unique(mesh_f_owned.flatten().long())
        remap = torch.full((mesh_v_g.shape[0],), -1, dtype=torch.long, device=mesh_v_g.device)
        remap[used_v] = torch.arange(used_v.shape[0], device=mesh_v_g.device)
        mesh_v_compact = mesh_v_g[used_v]
        mesh_f_compact = remap[mesh_f_owned.long()].to(mesh_f_owned.dtype)
        # For dedup later: voxel coords of the compacted vertices.
        voxel_coords_compact = coords_g[used_v].to(torch.int64)

        mesh_v_parts.append(mesh_v_compact.cpu())
        mesh_f_parts.append(mesh_f_compact.cpu())
        voxel_coords_parts.append(voxel_coords_compact.cpu())

        print(
            f"[joint_decode_shape] group {g_idx}: chunks={group} in={n_in:,} → "
            f"verts={mesh_v_compact.shape[0]:,}, faces={mesh_f_compact.shape[0]:,}"
        )

        del slat_g, h_g, v_g, i_g, q_g, coords_g
        del mesh_v_g, mesh_f_g, mesh_f_owned, used_v, remap
        del mesh_v_compact, mesh_f_compact, voxel_coords_compact, face_centroids, in_owned
        torch.cuda.empty_cache()

    # ── Concat per-group meshes + dedupe boundary vertices ────────────────
    # Vertices coming from different groups but corresponding to the same
    # voxel coord must collapse to one entry, otherwise neighboring faces
    # across group boundaries don't share an edge in the index sense.
    cum_offsets: List[int] = []
    cum = 0
    for v in mesh_v_parts:
        cum_offsets.append(cum)
        cum += int(v.shape[0])

    final_verts = torch.cat([v.to(device) for v in mesh_v_parts], dim=0)
    final_faces = torch.cat(
        [f.to(device) + off for f, off in zip(mesh_f_parts, cum_offsets)],
        dim=0,
    )
    voxel_coords_all = torch.cat([c.to(device) for c in voxel_coords_parts], dim=0)
    del mesh_v_parts, mesh_f_parts, voxel_coords_parts

    # Dedupe by voxel coord. The grid is bounded; pack (x, y, z) into one int64.
    GY = int((meta["maxs"] - meta["mins"])[1].item()) * upscale + 1
    GZ = int((meta["maxs"] - meta["mins"])[2].item()) * upscale + 1
    keys = voxel_coords_all[:, 0] * (GY * GZ) + voxel_coords_all[:, 1] * GZ + voxel_coords_all[:, 2]
    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    # Pick one source vertex per unique key. Sort by inverse so identical
    # keys are adjacent; the first row in each run is one valid representative.
    order = torch.argsort(inverse, stable=True)
    inverse_sorted = inverse[order]
    keep_mask = torch.ones(inverse.shape[0], dtype=torch.bool, device=device)
    keep_mask[1:] = inverse_sorted[1:] != inverse_sorted[:-1]
    representative = order[keep_mask]  # one original index per unique key, in unique-key order
    final_verts_dedup = final_verts[representative]
    # chunked gather: a one-shot `inverse[final_faces.long()]` materializes an
    # int64 copy of every face row (24 B/face -> >10 GB at ~0.5B faces) on top
    # of the mesh already resident — the OOM that killed 2x-upscaled scenes
    final_faces_dedup = torch.empty_like(final_faces)
    _B = 50_000_000
    for _i in range(0, final_faces.shape[0], _B):
        final_faces_dedup[_i:_i + _B] = (
            inverse[final_faces[_i:_i + _B].long()].to(final_faces.dtype))
    del final_verts, voxel_coords_all, keys, inverse, order, inverse_sorted, keep_mask, representative
    del final_faces
    torch.cuda.empty_cache()

    # Dedupe faces: overlapping owned AABBs make both groups emit the shared
    # window-overlap slab; after vertex dedup those are exact index triples.
    # Keep the first original face per sorted triple (preserves winding).
    # Runs on CPU: unique(dim=0) over hundreds of millions of int64 rows
    # allocates several times the input, which does not fit next to the mesh
    # on the GPU — host RAM takes it easily, only the keep-index returns.
    n_faces_pre = final_faces_dedup.shape[0]
    sorted_tris = torch.sort(final_faces_dedup.long().cpu(), dim=1).values
    _, tri_inverse = torch.unique(sorted_tris, dim=0, return_inverse=True)
    first_face = torch.full((int(tri_inverse.max().item()) + 1,), n_faces_pre, dtype=torch.long)
    first_face.scatter_reduce_(0, tri_inverse, torch.arange(n_faces_pre), reduce="amin")
    final_faces_dedup = final_faces_dedup[first_face.to(device)]
    del sorted_tris, tri_inverse, first_face
    torch.cuda.empty_cache()

    print(
        f"[joint_decode_shape] merged: {final_verts_dedup.shape[0]:,} verts (dedup from "
        f"{cum:,}), {final_faces_dedup.shape[0]:,} faces (dedup from {n_faces_pre:,})"
    )

    return Mesh(final_verts_dedup, final_faces_dedup)
