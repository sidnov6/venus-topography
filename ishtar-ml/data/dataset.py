"""Assembling model inputs from tiles.

A tile on disk is a dict of rasters and scalars. `collate_inputs` turns a batch of them
into exactly the channel stack `model.unet.INPUT_CHANNELS` declares, the global
conditioning vector `COND_FEATURES` declares, and the targets the losses consume. Both
orderings live in `model.unet` so the dataset and the network cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from model import physics
from model.unet import COND_FEATURES, INPUT_CHANNELS

from . import augment, masks

LOOKS = ("left", "right", "stereo")

# Normalisation. Elevation is expressed relative to the tile's own GTDR mean so the
# network never sees an absolute planetary radius, and dB is already near zero-mean.
GTDR_SCALE_M = 1000.0
EMISSIVITY_MEAN = 0.85
EMISSIVITY_SCALE = 0.05
RV_SCALE_DB = 5.0


@dataclass
class BatchSpec:
    pixel_size_m: float = 75.0
    gtdr_stride_px: int = 62
    gsdr_cell_m: float = 4641.0
    alt_edge_margin_px: int = 0
    """Posts within this many pixels of the tile border are dropped from `L_alt`.

    0 is right for `data.synthetic`, whose GTDR is produced by the same padded operator
    on the same extent. For real tiles set it to ~3 sigma of the footprint (about 370 px
    at 75 m) and cut tiles with a matching context margin — see `losses.loss_alt`.
    """
    max_gain_db: float = 3.0
    speckle_looks: float = 6.0
    p_drop_right: float = 0.4
    p_drop_stereo: float = 0.4
    augment: bool = True


def _as_bchw(x: Tensor) -> Tensor:
    return x if x.dim() == 4 else x.unsqueeze(1) if x.dim() == 3 else x[None, None]


def build_batch(
    tiles: dict[str, Tensor], spec: BatchSpec, rng: np.random.Generator | None = None
) -> dict[str, Tensor]:
    """Turn a collated tile dict into model inputs and loss targets.

    `tiles` is what the default DataLoader collate produces from `SyntheticVenus` (or the
    Zarr dataset): rasters `(B, H, W)`, scalars `(B,)`, look vectors `(B, 2)`.
    """
    rng = rng or np.random.default_rng()
    B = tiles["dn_left"].shape[0]
    dev = tiles["dn_left"].device

    out: dict[str, Tensor] = {}
    rv: dict[str, Tensor] = {}
    valid: dict[str, Tensor] = {}
    look: dict[str, Tensor] = {}
    theta: dict[str, Tensor] = {}

    for name in LOOKS:
        r, v = physics.rv_from_dn(_as_bchw(tiles[f"dn_{name}"]))
        has = tiles[f"has_{name}"].view(B, 1, 1, 1).bool()
        v = v & has
        if spec.augment:
            # Per-tile, unlike the geometric transform: gain striping is what the model
            # must learn to ignore, and a single offset across the batch would let it be
            # absorbed by a batch statistic instead.
            offsets = torch.from_numpy(
                rng.uniform(-spec.max_gain_db, spec.max_gain_db, size=B).astype(np.float32)
            ).view(B, 1, 1, 1).to(r.device)
            r = r + offsets
        rv[name] = r * v
        valid[name] = v
        look[name] = tiles[f"look_{name}"].to(torch.float32)
        theta[name] = tiles[f"theta_{name}"].view(B, 1, 1, 1).to(torch.float32)

    gtdr_up = _as_bchw(tiles["gtdr_up"])
    gtdr_valid = _as_bchw(tiles["gtdr_valid"]) if "gtdr_valid" in tiles else None
    z_true = _as_bchw(tiles["z_true"]) if "z_true" in tiles else None
    stereo = _as_bchw(tiles["stereo_dem"])
    stereo_valid = _as_bchw(tiles["stereo_valid"])
    emis = _as_bchw(tiles["emissivity"])
    rms = _as_bchw(tiles["rms_slope"])
    lat = tiles["lat_deg"].view(B, 1, 1, 1).to(torch.float32)

    # ---- geometric augmentation: rasters and look vectors together ----
    # One transform per batch rather than per tile. Correctness is unaffected — the look
    # vectors move with the rasters either way — but it does mean a batch of 8 sees one
    # orientation, not eight. Per-tile would need the dihedral applied inside a loop, and
    # the trade is diversity against a slower collate; revisit if the model turns out to
    # be orientation-sensitive on real mosaics, where the striping is directional.
    if spec.augment:
        k, fh, fv = augment.random_dihedral(rng)
        for name in LOOKS:
            rv[name] = augment.apply_dihedral(rv[name], k, fh, fv)
            valid[name] = augment.apply_dihedral(valid[name], k, fh, fv)
            look[name] = augment.dihedral_look_vec(look[name], k, fh, fv)
        gtdr_up = augment.apply_dihedral(gtdr_up, k, fh, fv)
        if gtdr_valid is not None:
            gtdr_valid = augment.apply_dihedral(gtdr_valid, k, fh, fv)
        stereo = augment.apply_dihedral(stereo, k, fh, fv)
        stereo_valid = augment.apply_dihedral(stereo_valid, k, fh, fv)
        emis = augment.apply_dihedral(emis, k, fh, fv)
        rms = augment.apply_dihedral(rms, k, fh, fv)
        if z_true is not None:
            z_true = augment.apply_dihedral(z_true, k, fh, fv)
        # Latitude is left alone: a tile is ~19-38 km, well under a tenth of a degree of
        # incidence-angle variation, so re-deriving it after a north-south flip would be
        # noise. Do revisit this if tiles ever get much larger.

    # ---- random look dropout, so the left-only pathway is trained ----
    if spec.augment:
        for name, p in (("right", spec.p_drop_right), ("stereo", spec.p_drop_stereo)):
            drop = torch.from_numpy((rng.random(B) < p)).view(B, 1, 1, 1).to(dev)
            rv[name] = rv[name] * ~drop
            valid[name] = valid[name] & ~drop

    # ---- input channel stack ----
    gtdr_ref = gtdr_up.mean(dim=(-2, -1), keepdim=True)
    chan = {
        "sar_left_db": rv["left"] / RV_SCALE_DB,
        "mask_left": valid["left"].to(torch.float32),
        "sar_right_db": rv["right"] / RV_SCALE_DB,
        "mask_right": valid["right"].to(torch.float32),
        "sar_stereo_db": rv["stereo"] / RV_SCALE_DB,
        "mask_stereo": valid["stereo"].to(torch.float32),
        "gtdr_up": (gtdr_up - gtdr_ref) / GTDR_SCALE_M,
        "emissivity": (emis - EMISSIVITY_MEAN) / EMISSIVITY_SCALE,
    }
    for name in LOOKS:
        chan[f"theta_{name}_sin"] = torch.sin(theta[name]).expand_as(gtdr_up)
        chan[f"theta_{name}_cos"] = torch.cos(theta[name]).expand_as(gtdr_up)
    # Only the left look vector is fed. The right and stereo looks are its negation by
    # construction (`geometry.look_vector`), so a second pair would be redundant — and
    # after augmentation all three are transformed together, so this stays consistent.
    chan["look_east"] = look["left"][:, 0].view(B, 1, 1, 1).expand_as(gtdr_up)
    chan["look_north"] = look["left"][:, 1].view(B, 1, 1, 1).expand_as(gtdr_up)
    chan["lat_sin"] = torch.sin(torch.deg2rad(lat)).expand_as(gtdr_up)
    chan["lat_cos"] = torch.cos(torch.deg2rad(lat)).expand_as(gtdr_up)

    out["x"] = torch.cat([chan[c] for c in INPUT_CHANNELS], dim=1)

    # ---- global conditioning vector ----
    scalar = {
        "look_east": look["left"][:, 0],
        "look_north": look["left"][:, 1],
        "has_right": valid["right"].flatten(1).any(dim=1).to(torch.float32),
        "has_stereo": valid["stereo"].flatten(1).any(dim=1).to(torch.float32),
    }
    for name in LOOKS:
        scalar[f"theta_{name}_sin"] = torch.sin(theta[name]).view(B)
        scalar[f"theta_{name}_cos"] = torch.cos(theta[name]).view(B)
    out["cond"] = torch.stack([scalar[c] for c in COND_FEATURES], dim=1)

    # ---- targets and auxiliaries ----
    out["gtdr_up"] = gtdr_up
    # GTDR nodata decodes to 0 m and Venus has no sea level, so an absent mask would let
    # `L_alt` anchor the surface to zero wherever the altimeter has a gap.
    out["gtdr_valid"] = (
        gtdr_valid.bool() if gtdr_valid is not None
        else torch.ones_like(gtdr_up, dtype=torch.bool)
    )
    out["stereo_dem"] = stereo
    out["stereo_trust"] = masks.stereo_trust_mask(stereo, stereo_valid, rv["left"], valid["left"])
    out["rms_slope"] = rms
    out["rms_valid"] = torch.ones_like(rms, dtype=torch.bool)
    out["unsupervised"] = masks.unsupervised_mask(
        out["stereo_trust"], valid["right"], valid["stereo"]
    )
    if z_true is not None:
        out["z_true"] = z_true
    for name in LOOKS:
        out[f"rv_{name}"] = rv[name]
        out[f"valid_{name}"] = valid[name]
        out[f"look_{name}"] = look[name]
        out[f"theta_{name}"] = theta[name]
    return out


def drop_second_looks(x: Tensor) -> Tensor:
    """Zero the right and stereo SAR channels and their masks, for the cross-look loss:
    it needs a prediction made as if only the left look existed."""
    x = x.clone()
    for name in ("sar_right_db", "mask_right", "sar_stereo_db", "mask_stereo"):
        x[:, INPUT_CHANNELS.index(name)] = 0.0
    return x
