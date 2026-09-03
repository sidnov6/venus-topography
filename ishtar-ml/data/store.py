"""Zarr-backed tile dataset with spatial splits.

Mirrors `data.synthetic.SyntheticVenus` exactly — same keys, same shapes — so training,
evaluation and inference code does not care which one it is given.

Splits are by whole 12 x 12 degree FMAP quadrangle. Never by tile: neighbouring tiles
overlap by the context margin and share terrain, so a random split leaks the validation
set into training and every metric comes out flattering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

RASTER_KEYS = (
    "dn_left", "dn_right", "dn_stereo", "gtdr_up", "gtdr_valid", "stereo_dem",
    "stereo_valid", "emissivity", "rms_slope",
)
SCALAR_KEYS = (
    "theta_left", "theta_right", "theta_stereo", "has_left", "has_right", "has_stereo",
    "has_stereo_dem", "lat_deg",
)
VECTOR_KEYS = ("look_left", "look_right", "look_stereo")


@dataclass
class SplitSpec:
    held_out_quads: tuple[str, ...] = ()
    """Validation quadrangles. Section 2.4 wants a tessera region (Ovda), a plains region
    with small volcanoes, and a crater field."""

    demo_quads: tuple[str, ...] = ()
    """Regions you look at constantly — Maxwell, Maat. These stay in *training*: a metric
    you have been eyeballing for weeks is not a held-out metric."""

    require_stereo: bool = False
    max_abs_lat_deg: float = 80.0


class ZarrTiles(torch.utils.data.Dataset):
    """Read-only view over a tile store written by `data/tile.py`.

    Crops each tile to its core before returning it, unless `keep_margin` is set — the
    margin exists so the footprint convolution in `L_alt` sees real terrain, so it is
    kept for training and dropped for anything that scores per-pixel accuracy.
    """

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        spec: SplitSpec | None = None,
        keep_margin: bool = True,
    ):
        import zarr

        self.root = zarr.open_group(str(path), mode="r")
        self.spec = spec or SplitSpec()
        self.keep_margin = keep_margin
        self.core_px = int(self.root.attrs["core_px"])
        self.margin_px = int(self.root.attrs["margin_px"])
        self.pixel_size_m = float(self.root.attrs["pixel_size_m"])

        quads = np.asarray(self.root["quad"][:])
        lats = np.asarray(self.root["lat_deg"][:])
        held = np.isin(quads, np.asarray(self.spec.held_out_quads, dtype=quads.dtype))
        usable = np.abs(lats) <= self.spec.max_abs_lat_deg
        if self.spec.require_stereo:
            usable &= np.asarray(self.root["has_stereo_dem"][:]) > 0.5

        keep = usable & (held if split == "val" else ~held)
        self.index = np.flatnonzero(keep)
        if self.index.size == 0:
            raise ValueError(
                f"split {split!r} is empty. Held-out quads {self.spec.held_out_quads} may "
                "not exist in this store; check data/tile.py::quad_id output."
            )

    def __len__(self) -> int:
        return int(self.index.size)

    def _crop(self, a: np.ndarray) -> np.ndarray:
        if self.keep_margin:
            return a
        m = self.margin_px
        return a[m : m + self.core_px, m : m + self.core_px]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        j = int(self.index[i])
        out: dict[str, torch.Tensor] = {}
        for k in RASTER_KEYS:
            out[k] = torch.from_numpy(np.ascontiguousarray(self._crop(self.root[k][j])))
        for k in SCALAR_KEYS:
            out[k] = torch.tensor(float(self.root[k][j]), dtype=torch.float32)
        for k in VECTOR_KEYS:
            out[k] = torch.from_numpy(np.asarray(self.root[k][j], dtype=np.float32))
        return out


def quad_summary(path: str | Path) -> dict[str, int]:
    """Tiles per quadrangle — the input to choosing held-out regions.

    Look at this before fixing `HELD_OUT_QUADS`: a validation quad with 40 tiles and no
    stereo coverage produces a metric table that means nothing.
    """
    import zarr

    root = zarr.open_group(str(path), mode="r")
    quads = np.asarray(root["quad"][:])
    uniq, counts = np.unique(quads, return_counts=True)
    return {str(q): int(c) for q, c in zip(uniq, counts)}
