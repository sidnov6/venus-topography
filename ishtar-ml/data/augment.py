"""Joint geometric and radiometric augmentation.

The one rule that matters: a flip or rotation changes the geometry of the scene, so it
must be applied to the rasters **and** to `look_vec` in the same breath. Flipping the
image without flipping the look vector trains the network on inverted radar physics —
bright slopes read as facing away from the radar — and the result still looks entirely
plausible in a loss curve. Everything geometric therefore goes through
`apply_dihedral`, which transforms both.

Axis convention, matching `data.geometry`: column index increases east, row index
increases south, so north is `-row`. `look_vec` is the down-range beam direction
(away from the radar), as in `model.physics`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def dihedral_look_vec(look_vec: Tensor, k: int, flip_h: bool, flip_v: bool) -> Tensor:
    """Transform an `(east, north)` vector the same way `apply_dihedral` transforms a raster.

    `torch.rot90(x, 1, (-2, -1))` maps a raster point `(r, c) -> (N-1-c, r)`, which sends
    the east axis to north and north to west, i.e. `(e, n) -> (-n, e)`.
    """
    v = look_vec.clone().to(torch.float32)
    for _ in range(k % 4):
        e, n = v[..., 0].clone(), v[..., 1].clone()
        v[..., 0], v[..., 1] = -n, e
    if flip_h:
        v[..., 0] = -v[..., 0]
    if flip_v:
        v[..., 1] = -v[..., 1]
    return v


def apply_dihedral(x: Tensor, k: int, flip_h: bool, flip_v: bool) -> Tensor:
    """One of the eight dihedral transforms of a `(..., H, W)` raster."""
    if k % 4:
        x = torch.rot90(x, k % 4, dims=(-2, -1))
    if flip_h:
        x = torch.flip(x, dims=[-1])
    if flip_v:
        x = torch.flip(x, dims=[-2])
    return x


def random_dihedral(rng: np.random.Generator) -> tuple[int, bool, bool]:
    return int(rng.integers(0, 4)), bool(rng.random() < 0.5), bool(rng.random() < 0.5)


def speckle(rv_db: Tensor, looks: float, rng: np.random.Generator) -> Tensor:
    """Multiplicative gamma speckle applied in linear power, returned in dB."""
    lin = 10 ** (rv_db / 10.0)
    g = torch.from_numpy(rng.gamma(looks, 1.0 / looks, size=tuple(rv_db.shape)).astype(np.float32))
    return 10 * torch.log10((lin * g.to(rv_db.device)).clamp_min(1e-6))


def gain_offset(rv_db: Tensor, max_db: float, rng: np.random.Generator) -> Tensor:
    """Uniform dB offset, imitating orbit-to-orbit gain differences between mosaic strips.

    Paired with the `h_b` brightness head this teaches the network that absolute
    brightness carries no elevation information — only its spatial structure does.
    """
    return rv_db + float(rng.uniform(-max_db, max_db))
