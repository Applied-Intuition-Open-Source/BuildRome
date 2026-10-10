# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""The two generation stages for one band of windows.

SS stage: windows are generated one at a time, nearest the camera first
(evidence windows before ring windows). Each window RePaint-pins the latent
cells already committed by its neighbors (autoregressive outpainting), and
optionally by the previous, nearer band (cross-band prior). All window
latents are then decoded jointly into one occupancy canvas.

SLat stage: the SS occupancy is sliced back into the windows; sparse latents
are generated window by window with the same outpainting, then decoded
jointly into one mesh.
"""
import numpy as np
import torch
import torch.nn.functional as F

from inference.models import (SAMPLER_PARAMS, SLAT_DEC, SLAT_MEAN, SLAT_STD, SS_DEC,
                              build_slat_model, build_ss_model)

SS_AR_ERODE = 1      # latent cells (of 16) dropped from a committed window's border
SLAT_AR_ERODE = 4    # token cells (of 64) dropped from a committed window's border
MIN_VOXELS = 64      # windows with fewer decoded voxels count as empty
FILL_HOLES_M = 1.0   # close mesh holes with metric perimeter below this


def _lift_chunk_kwargs(chunk_kwargs, lift_kwargs):
    # the depth pack rides chunk_kwargs, so the CFG negative pass sees it too
    return [{**kw, **lift_kwargs} for kw in chunk_kwargs]


def _with_hires(conds, neg_conds, lift_hi_feats):
    # hi-res lift source rides the CFG cond dict (zeroed in the negative pass)
    return ([{**c, "cond_2D_hi": lift_hi_feats} for c in conds],
            [{**c, "cond_2D_hi": torch.zeros_like(lift_hi_feats)} for c in neg_conds])

def _seed_all(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def window_order(layout, evidence=None):
    """Generation order: nearest the camera first; with ``evidence``, windows
    that see image evidence before the pure outpainting ring."""
    def rank(i):
        d, l, _ = layout[i]
        return (evidence is not None and not evidence[i], d * d + l * l)
    return sorted(range(len(layout)), key=rank)


def _inpaint_kwargs(x0, pin):
    """RePaint arguments for ``sample_window``: nothing pinned -> free sampling."""
    if not bool(pin.any()):
        return {"inpaint_x0": None, "inpaint_mask": None}
    return {"inpaint_x0": x0, "inpaint_mask": pin}


def _interior(cells, R, erode):
    """Mask of cells at least ``erode`` cells away from the window border (the
    border has the weakest receptive field, so only the interior is committed)."""
    return ((cells >= erode) & (cells < R - erode)).all(dim=1)


def _cell_keys(cells):
    """One int64 key per integer cell [N, 3] (band frame), for set lookups."""
    w = cells + 8192  # positive shift for a collision-free key
    return (w[:, 0] * 16384 + w[:, 1]) * 16384 + w[:, 2]


def _overlap_slices(offset, R, erode):
    """Overlap of two R^3 cell grids, window j's origin sitting at ``offset``
    (in window i's cells), shrunk by ``erode`` cells on j's border. Returns
    (slices into i, slices into j), or None if the trusted overlap is empty."""
    lo = [max(o + erode, 0) for o in offset]
    hi = [min(o + R - erode, R) for o in offset]
    if any(a >= b for a, b in zip(lo, hi)):
        return None
    sl_i = tuple(slice(a, b) for a, b in zip(lo, hi))
    sl_j = tuple(slice(a - o, b - o) for a, b, o in zip(lo, hi, offset))
    return sl_i, sl_j


def _pin_ss_neighbors(x0, pin, i, committed, latents, rel_cells, R, erode):
    """RePaint targets for window ``i`` from its already-generated neighbors:
    where a neighbor's trusted (eroded) cube overlaps window i, copy its latent
    and mark the cell pinned. Cells pinned earlier -- by the previous band or
    by an earlier (nearer) neighbor -- are kept. Mutates x0 [1, C, R, R, R]
    and pin [1, 1, R, R, R]."""
    for j in committed:
        sl = _overlap_slices((rel_cells[j] - rel_cells[i]).tolist(), R, erode)
        if sl is None:
            continue
        sl_i, sl_j = sl
        free = ~pin[0, 0][sl_i]
        x0[0][(slice(None),) + sl_i][:, free] = latents[j][0][(slice(None),) + sl_j][:, free].float()
        pin[0, 0][sl_i] |= free


class _CommittedTokens:
    """SLat tokens committed so far, addressed by world cell key. Lookups give
    the first store that has a cell: the previous band is added first, then
    windows in generation order, so nearer always wins."""

    def __init__(self):
        self.keys, self.feats = [], []   # per store: sorted keys, feats in that order

    def add(self, keys, feats):
        keys, perm = keys.sort()
        self.keys.append(keys)
        self.feats.append(feats[perm])

    def lookup(self, keys, channels):
        """(x0 [N, C], pin [N]) for the query cells; unpinned rows are zero."""
        x0 = torch.zeros(len(keys), channels, device=keys.device)
        pin = torch.zeros(len(keys), dtype=torch.bool, device=keys.device)
        for kj, fj in zip(self.keys, self.feats):
            hit = torch.isin(keys, kj) & ~pin
            if bool(hit.any()):
                x0[hit] = fj[torch.searchsorted(kj, keys[hit])].float()
                pin |= hit
        return x0, pin


def cross_band_ss_prior(layout, prior, device, R_lat=16):
    """Pin targets from the previous band: each latent cell of THIS band's
    windows whose center lies in a nearer window's eroded cube takes that
    window's latent, trilinearly resampled. prior: lat [N, 8, 16, 16, 16],
    full / boxes [N, 6] (full / eroded cubes) in this band's canonical frame."""
    from build_rome.pipelines import windows as mdv

    C = mdv.CHUNK_SIZE
    lat = prior["lat"].to(device).float()
    full = prior["full"].to(device).float()
    ero = prior["boxes"].to(device).float()
    ax = (torch.arange(R_lat, dtype=torch.float32, device=device) + 0.5) * (C / R_lat)
    gx, gy, gz = torch.meshgrid(ax - C / 2, ax - C / 2, ax, indexing="ij")
    local = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    x0s, pins, n_pin = [], [], 0
    for d, l, z in layout:
        ctr = local + torch.tensor([l, d, z], device=device)
        x0 = torch.zeros(lat.shape[1], R_lat ** 3, device=device)
        pin = torch.zeros(R_lat ** 3, dtype=torch.bool, device=device)
        for j in range(len(lat)):
            m = ((ctr >= ero[j, :3]) & (ctr <= ero[j, 3:])).all(dim=1) & ~pin
            if not bool(m.any()):
                continue
            lo, hi = full[j, :3], full[j, 3:]
            nrm = 2 * (ctr[m] - lo) / (hi - lo) - 1                 # (lat, dep, up) in [-1, 1]
            grid = nrm[:, [2, 1, 0]].view(1, -1, 1, 1, 3)           # grid_sample: (W=up, H=dep, D=lat)
            smp = F.grid_sample(lat[j:j + 1], grid, mode="bilinear", align_corners=False)
            x0[:, m] = smp.view(lat.shape[1], -1)
            pin |= m
        if not bool(pin.any()):
            x0s.append(None); pins.append(None)
            continue
        x0s.append(x0.view(1, -1, R_lat, R_lat, R_lat))
        pins.append(pin.view(1, 1, R_lat, R_lat, R_lat))
        n_pin += int(pin.sum())
    print(f"[cross-band] {sum(p is not None for p in pins)}/{len(layout)} windows, "
          f"{n_pin} latent cells pinned from {len(lat)} nearer windows")
    return x0s, pins


def cross_band_slat_prior(act_layout, local_coords_act, rel_cells, R, prior, device):
    """Pin targets from the previous band's SLat tokens: every token of THIS
    band inside a nearer window's eroded cube takes the trilinear interpolation
    of the 8 surrounding nearer tokens (weights renormalized over those that
    exist; pinned only if >= half the weight is present). prior: centers
    [M, 3], feats [M, C], pitch, origin [3], boxes [B, 6] in this band's frame.
    Returns (sorted world keys, feats) or (None, None)."""
    from build_rome.pipelines import windows as mdv

    C = mdv.CHUNK_SIZE
    cen = prior["centers"].to(device).float()
    fe = prior["feats"].to(device).float()
    pitch = float(prior["pitch"])
    org = prior["origin"].to(device).float()
    boxes = prior["boxes"].to(device).float()
    g = torch.round((cen - org) / pitch - 0.5).long()
    kp, order = _cell_keys(g).sort()
    fe = fe[order]
    keys, feats = [], []
    for a, (d, l, z) in enumerate(act_layout):
        lc = local_coords_act[a].long().to(device)
        ctr = (lc.float() + 0.5) * (C / R) + torch.tensor([l - C / 2, d - C / 2, z], device=device)
        inside = torch.zeros(len(lc), dtype=torch.bool, device=device)
        for b in boxes:
            inside |= ((ctr >= b[:3]) & (ctr <= b[3:])).all(dim=1)
        if not bool(inside.any()):
            continue
        q = (ctr[inside] - org) / pitch - 0.5          # continuous nearer-lattice coords
        q0 = torch.floor(q).long()
        t = q - q0.float()
        acc = torch.zeros(len(q), fe.shape[1], device=device)
        wsum = torch.zeros(len(q), device=device)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    nb = q0 + torch.tensor([dx, dy, dz], device=device)
                    wt = ((t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                          * (t[:, 2] if dz else 1 - t[:, 2]))
                    kq = _cell_keys(nb)
                    pos = torch.searchsorted(kp, kq).clamp(max=len(kp) - 1)
                    hit = kp[pos] == kq
                    acc[hit] += wt[hit, None] * fe[pos[hit]]
                    wsum[hit] += wt[hit]
        ok = wsum >= 0.5
        if not bool(ok.any()):
            continue
        idx = torch.nonzero(inside).squeeze(1)[ok]
        keys.append(_cell_keys(lc[idx] + rel_cells[a][None, :]))
        feats.append(acc[ok] / wsum[ok, None])
    if not keys:
        print("[cross-band slat] no prior tokens for this band")
        return None, None
    k, f = torch.cat(keys), torch.cat(feats)
    k, first = np.unique(k.cpu().numpy(), return_index=True)
    print(f"[cross-band slat] {len(k):,} tokens interpolated from {len(cen):,} nearer tokens")
    return torch.from_numpy(k).to(device), f[torch.from_numpy(first).to(device)]


def load_ss_models(ss_ckpt, device):
    """(denoiser, occupancy decoder) for the SS stage, built once."""
    from build_rome import models

    return build_ss_model(ss_ckpt, device), models.from_pretrained(SS_DEC).to(device).eval()


def load_slat_models(slat_ckpt, device):
    """(denoiser, shape decoder at o-voxel resolution 1024) for the SLat stage."""
    from build_rome import models

    slat_dec = models.from_pretrained(SLAT_DEC)
    slat_dec.set_resolution(1024)
    return build_slat_model(slat_ckpt, device), slat_dec.to(device).eval()


@torch.no_grad()
def run_ss_stage(models, image, extractor, device, lift_kwargs, lift_hi_feats, layout, evidence,
                 seed, prior=None):
    """Autoregressive SS over ``layout`` (canonical frame) -> (occs, latents,
    world_coords, R). occs: per-window [64, 64, 64] bool occupancy.
    ``models``: the ``(denoiser, ss_dec)`` pair from ``load_ss_models``.

    Windows are sampled nearest-first. Each window RePaint-pins the latent
    cells already committed, with precedence: previous band (``prior``, finer
    and already decoded) > earlier neighbor > later neighbor; a committed
    window contributes its cube eroded by SS_AR_ERODE cells. All latents are
    then decoded jointly into one occupancy canvas."""
    from build_rome.pipelines import windows as mdv
    from build_rome.pipelines.joint_decode import joint_decode_sparse_structure

    denoiser, ss_dec = models
    rel = mdv.relative_translations(layout)
    conds, neg_conds, chunk_kwargs = mdv.make_window_conditioning(extractor, image, layout, device)
    chunk_kwargs = _lift_chunk_kwargs(chunk_kwargs, lift_kwargs)
    conds, neg_conds = _with_hires(conds, neg_conds, lift_hi_feats)
    sampler = mdv.make_val_sampler(1e-5)

    R_lat, C = denoiser.resolution, denoiser.in_channels
    rel_cells = [(t * R_lat).round().long() for t in rel]   # window origins in latent cells
    order = window_order(layout, evidence)
    print(f"[ss] {len(order)} windows, nearest-first, pin erode {SS_AR_ERODE} cells")
    prior_x0 = prior_pin = None
    if prior is not None:
        prior_x0, prior_pin = cross_band_ss_prior(layout, prior, device, R_lat)

    latents = [None] * len(layout)
    for step, i in enumerate(order):
        x0 = torch.zeros(1, C, R_lat, R_lat, R_lat, device=device)
        pin = torch.zeros(1, 1, R_lat, R_lat, R_lat, dtype=torch.bool, device=device)
        if prior_pin is not None and prior_pin[i] is not None:
            x0 = torch.where(prior_pin[i], prior_x0[i], x0)
            pin = pin | prior_pin[i]
        _pin_ss_neighbors(x0, pin, i, order[:step], latents, rel_cells, R_lat, SS_AR_ERODE)
        _seed_all(seed + i)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = mdv.sample_window(
                sampler, denoiser, torch.empty(1, C, R_lat, R_lat, R_lat, device=device),
                conds[i], neg_conds[i], **SAMPLER_PARAMS["ss"], chunk_kwargs=chunk_kwargs[i],
                **_inpaint_kwargs(x0, pin))
        latents[i] = out.float()
        print(f"[ss]   window {step + 1}/{len(order)} (d={layout[i][0]:.2f}, l={layout[i][1]:.2f}) "
              f"pinned {int(pin.sum())}/{R_lat ** 3} cells")
    torch.cuda.empty_cache()

    R = 64  # decoded cells per window (= SLat grid)
    occs = [(logit[0, 0] > 0) for logit in joint_decode_sparse_structure(ss_dec, latents, rel, ss_res=R)]
    world, kept = [], 0
    for occ, (d, l, z) in zip(occs, layout):
        if int(occ.sum()) < MIN_VOXELS:
            continue
        kept += 1
        off = torch.tensor([round(l / mdv.CHUNK_SIZE * R), round(d / mdv.CHUNK_SIZE * R),
                            round(z / mdv.CHUNK_SIZE * R)], device=device, dtype=torch.long)
        world.append(torch.nonzero(occ, as_tuple=False).long() + off[None, :])
    print(f"[ss] {kept}/{len(layout)} windows above {MIN_VOXELS} voxels")
    assert world, "SS predicted an empty scene"
    return occs, latents, torch.unique(torch.cat(world, dim=0), dim=0), R


@torch.no_grad()
def run_slat_stage(models, image, extractor, device, lift_kwargs, lift_hi_feats, layout,
                   world_coords, R, seed, scene_scale, prior=None, max_inflated_voxels=100_000):
    """SLat over the SS occupancy -> (mesh, T0, tokens). The mesh is in the
    window-0-local frame; T0 places it in the band's canonical frame. tokens:
    ([(local coords, normalized feats)], active layout) for the next band.
    ``models``: the ``(denoiser, slat_dec)`` pair from ``load_slat_models``.

    Same outpainting scheme as the SS stage, on sparse tokens: a window
    RePaint-pins every token whose world cell was already committed (previous
    band first, then nearer windows), and commits its own tokens that lie
    SLAT_AR_ERODE cells inside its border."""
    from build_rome.pipelines import windows as mdv
    from build_rome.modules.sparse.basic import SparseTensor
    from build_rome.pipelines.joint_decode import joint_decode_shape

    denoiser, slat_dec = models

    local_coords, _ = mdv.slice_shared_structure(world_coords, layout, R)
    active = [k for k, lc in enumerate(local_coords) if lc is not None and lc.shape[0] >= MIN_VOXELS]
    assert active, "no active SLat windows"
    act_layout = [layout[k] for k in active]
    rel = mdv.relative_translations(act_layout)
    conds, neg_conds, chunk_kwargs = mdv.make_window_conditioning(extractor, image, act_layout, device)
    chunk_kwargs = _lift_chunk_kwargs(chunk_kwargs, lift_kwargs)
    conds, neg_conds = _with_hires(conds, neg_conds, lift_hi_feats)
    sampler = mdv.make_val_sampler(1e-5)
    C_in = denoiser.in_channels
    noise = [SparseTensor(
        coords=torch.cat([torch.zeros_like(local_coords[k][:, :1]), local_coords[k]], dim=1).int(),
        feats=torch.empty(local_coords[k].shape[0], C_in, device=device))
        for k in active]
    print(f"[slat] {len(active)}/{len(layout)} active windows, "
          f"{sum(n.feats.shape[0] for n in noise)} tokens total")
    rel_cells = [(t * R).round().long().to(device) for t in rel]   # window origins in token cells
    order = window_order(act_layout)

    committed = _CommittedTokens()
    if prior is not None:
        pk, pf = cross_band_slat_prior(act_layout, [local_coords[k] for k in active], rel_cells, R,
                                       prior, device)
        if pk is not None:
            committed.add(pk, pf)
    latents = [None] * len(active)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for step, a in enumerate(order):
            lc = local_coords[active[a]].long().to(device)
            cells = lc + rel_cells[a][None, :]                    # band-frame token cells
            x0, pin = committed.lookup(_cell_keys(cells), C_in)
            _seed_all(seed + active[a])
            out = mdv.sample_window(
                sampler, denoiser, noise[a], conds[a], neg_conds[a], **SAMPLER_PARAMS["slat"],
                chunk_kwargs=chunk_kwargs[a], **_inpaint_kwargs(x0, pin))
            latents[a] = out.replace(out.feats.float())
            interior = _interior(lc, R, SLAT_AR_ERODE)
            committed.add(_cell_keys(cells[interior]), latents[a].feats[interior])
            print(f"[slat]   window {step + 1}/{len(order)} (d={act_layout[a][0]:.2f}, "
                  f"l={act_layout[a][1]:.2f}) pinned {int(pin.sum())}/{len(cells)} tokens")
    del committed
    torch.cuda.empty_cache()

    tokens = ([(z.coords[:, 1:].detach().cpu(), z.feats.detach().float().cpu()) for z in latents],
              list(act_layout))
    mean = torch.tensor(SLAT_MEAN, device=device).reshape(1, -1)
    std = torch.tensor(SLAT_STD, device=device).reshape(1, -1)
    slats = [z.replace(z.feats.float() * std + mean) for z in latents]
    # overlap merge: a window's uncommitted border ring (the SLAT_AR_ERODE cells
    # never pinned into its neighbours) gets weight 0, so each overlap voxel is
    # decoded from the window it is interior to instead of a mean of two
    # different samples (which shreds the surface along the seam)
    merge_w = [_interior(z.coords[:, 1:], R, SLAT_AR_ERODE).float() for z in latents]
    mesh = joint_decode_shape(slat_dec, slats, rel, max_inflated_voxels=max_inflated_voxels, weights=merge_w)
    torch.cuda.empty_cache()

    # metric -> window-local units: 1 local unit = CHUNK_SIZE canonical m,
    # canonical = metric * scene_scale
    cap_local = FILL_HOLES_M * scene_scale / mdv.CHUNK_SIZE
    nf0 = int(mesh.faces.shape[0])
    try:
        mesh.fill_holes(max_hole_perimeter=cap_local)
        print(f"[mesh] fill_holes({FILL_HOLES_M:g} m): +{int(mesh.faces.shape[0]) - nf0:,} faces")
    except Exception as e:  # noqa: BLE001 — a fill failure must not kill the run
        print(f"[mesh] fill_holes failed ({e}); continuing unfilled")
    return mesh, mdv.window_world_transform(*act_layout[0]), tokens
