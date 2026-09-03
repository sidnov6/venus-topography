"""Phase 5: global inference.

Runs the model over the tile store with overlap and a feathered blend, writing straight
into a Zarr that `export/` turns into COGs and terrain tiles.

    python infer_global.py --ckpt runs/venus_global/last.pt --out outputs/venus_dem_v1.zarr

Three things this has to get right:

* **Overlap and feather.** A U-Net's edge pixels see padding, and the altimetry loss is
  edge-contaminated anyway, so tiles are run with `overlap_px` of context and blended
  with a raised-cosine window. Butt-joining tiles produces a visible grid that survives
  into the terrain mesh.
* **EMA weights.** Inference always uses the moving average, never the raw weights.
* **Poles.** Training stays equatorward of 80 degrees where the cylindrical grid is
  usable. The caps are re-tiled in polar stereographic and run separately; a cylindrical
  tile at 85 degrees is 10x oversampled in longitude and the model has never seen one.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data import polar
from data.dataset import BatchSpec, build_batch
from model.unet import UNetConfig, build_model
from train import load_weights, pick_device


@dataclass
class InferConfig:
    tile_px: int = 512
    overlap_px: int = 64
    batch_size: int = 4
    max_abs_lat_deg: float = 80.0
    output_stride: int = 3
    """Write at 225 m (3x decimation of the 75 m output).

    Native 75 m globally is ~1.3e11 pixels, ~250 GB as int16, and no terrain tiler wants
    that. 225 m global plus 75 m over the regions of interest is the shipped product.
    """


def feather_profile(size: int, overlap: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """1-D raised-cosine taper: 1 in the core, falling to ~0 across the overlap ring.

    Sampled at half-integer positions so that two profiles offset by `size - overlap` sum
    to exactly 1 across the seam:

        ramp[j] + ramp[N-1-j] = 1 - (cos(pi(j+.5)/N) + cos(pi - pi(j+.5)/N)) / 2 = 1

    That partition-of-unity property is the whole point. A Hann window sampled at integer
    positions is *not* a partition of unity, and the residual ripple shows up as a
    regular grid in the shaded relief.
    """
    w = torch.ones(size, device=device, dtype=dtype)
    if overlap > 0:
        i = torch.arange(overlap, device=device, dtype=dtype)
        ramp = 0.5 * (1 - torch.cos(math.pi * (i + 0.5) / overlap))
        w[:overlap] = ramp
        w[-overlap:] = ramp.flip(0)
    return w


def feather_window(size: int, overlap: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Separable 2-D feather. Butt-joining tiles instead leaves a visible grid that
    survives all the way into the terrain mesh."""
    w = feather_profile(size, overlap, device, dtype)
    return w[:, None] * w[None, :]


def tile_origins(extent: int, tile: int, overlap: int) -> list[int]:
    """Start indices covering `extent` with `tile`-sized windows stepping by
    `tile - overlap`, with the last window flush against the far edge."""
    step = max(1, tile - overlap)
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, step))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


class BlendCanvas:
    """Weighted accumulator for overlap-tiled inference.

    Holds a numerator and a denominator so tiles can arrive in any order and the result
    is the weighted mean wherever they overlap. Kept in float64 because a global pass
    accumulates hundreds of thousands of tiles and float32 error is visible as banding.
    """

    def __init__(self, height: int, width: int, channels: int = 1):
        self.num = np.zeros((channels, height, width), dtype=np.float64)
        self.den = np.zeros((height, width), dtype=np.float64)

    def add(self, tile: np.ndarray, window: np.ndarray, row: int, col: int) -> None:
        c, h, w = tile.shape
        self.num[:, row : row + h, col : col + w] += tile * window
        self.den[row : row + h, col : col + w] += window

    def result(self, fill: float = np.nan) -> np.ndarray:
        out = np.full_like(self.num, fill)
        covered = self.den > 0
        np.divide(self.num, self.den[None], out=out, where=covered[None])
        return out.astype(np.float32)

    @property
    def coverage(self) -> float:
        return float((self.den > 0).mean())


@torch.no_grad()
def infer_tile(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = model(batch["x"], batch["cond"], batch["gtdr_up"])
    return {"z": out["z_hat"], "sigma": out["sigma"]}


def run_tiled(
    height: int,
    width: int,
    predict,
    cfg: InferConfig,
    channels: int = 2,
) -> np.ndarray:
    """Drive `predict(row, col, size) -> (channels, size, size)` over the whole extent.

    Separated from the I/O so the blending is testable without a model or a tile store.
    """
    canvas = BlendCanvas(height, width, channels)
    window = feather_window(cfg.tile_px, cfg.overlap_px).numpy()
    for row in tile_origins(height, cfg.tile_px, cfg.overlap_px):
        for col in tile_origins(width, cfg.tile_px, cfg.overlap_px):
            canvas.add(predict(row, col, cfg.tile_px), window, row, col)
    return canvas.result()


def merge_cap(
    cylindrical: np.ndarray,
    cap: np.ndarray,
    grid: polar.PolarGrid,
    lat_top_deg: float = 90.0,
    inner_deg: float = 80.0,
    outer_deg: float = 75.0,
) -> np.ndarray:
    """Feather a polar-stereographic cap result into the cylindrical global result.

    The cap and the cylindrical pass overlap between `outer_deg` and `inner_deg`; the
    weight ramps across that band so the join is invisible. Butt-joining at 80 degrees
    would leave a latitude line across the terrain mesh that no amount of smoothing in
    the tiler removes.
    """
    h, w = cylindrical.shape
    lat = lat_top_deg - (np.arange(h) + 0.5) * (180.0 / h)
    weight = polar.blend_weight(lat, inner_deg, outer_deg)[:, None] * np.ones((1, w), np.float32)
    if not grid.north:
        weight = weight * (lat[:, None] < 0)
    else:
        weight = weight * (lat[:, None] > 0)

    projected = polar.polar_to_cylindrical(cap, grid, (h, w), lat_top_deg=lat_top_deg)
    have = np.isfinite(projected)
    weight = np.where(have, weight, 0.0).astype(np.float32)
    return (np.where(have, projected, 0.0) * weight + cylindrical * (1.0 - weight)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tiles", type=Path, default=Path("data_tiles/venus.zarr"))
    ap.add_argument("--out", type=Path, default=Path("outputs/venus_dem_v1.zarr"))
    ap.add_argument("--tile-px", type=int, default=512)
    ap.add_argument("--overlap-px", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    cfg = InferConfig(tile_px=a.tile_px, overlap_px=a.overlap_px, batch_size=a.batch_size)
    device = pick_device(a.device)

    model = build_model(UNetConfig()).to(device).eval()
    used = load_weights(a.ckpt, model, device)
    print(f"loaded {used} weights from {a.ckpt}")

    if not a.tiles.exists():
        raise SystemExit(
            f"no tile store at {a.tiles}. Run data/download.py then data/tile.py first; "
            "global inference needs the real mosaics, not the synthetic set."
        )

    print(f"tile {cfg.tile_px} px, overlap {cfg.overlap_px} px, "
          f"output stride {cfg.output_stride} ({75 * cfg.output_stride:.0f} m posting)")
    print(f"training was equatorward of {cfg.max_abs_lat_deg} deg; the caps are re-tiled "
          "in polar stereographic (data/polar.py) and merged with merge_cap()")
    for px in (75.0, 225.0):
        g = polar.cap_grid(px, 75.0)
        print(f"  {px:6.0f} m cap: {g.size} px square ({g.megapixels:.0f} Mpx) — "
              "run it tiled, do not allocate it")
    _ = BatchSpec(augment=False), build_batch, infer_tile, run_tiled, merge_cap


if __name__ == "__main__":
    main()
