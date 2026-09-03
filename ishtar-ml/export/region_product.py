"""Run a trained model over a real Venus region and export it for the globe.

    python -m export.region_product --ckpt runs/real_warm/last.pt --region ovda \
        --out outputs/ovda --globe ../ishtar-globe/public/tiles

Reads the Magellan mosaics by window, runs the network on overlapping tiles, feathers the
result into one canvas, and writes:

  <out>/elevation.npy, sigma.npy, sar.npy, gtdr.npy   the arrays, with a JSON sidecar
  <out>/preview.png                                    a look at what came out
  <globe>/terrain_ishtar_<region>/                     quantized-mesh on the Venus sphere
  <globe>/{sar,colour_relief,hillshade}_<region>/      imagery tiles, geodetic

The blend is the one from `infer_global`: a raised-cosine partition of unity, so adjacent
tiles sum to one across the seam instead of leaving a grid in the terrain mesh.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("VSI_CACHE", "TRUE")
# A stalled HTTP read otherwise blocks forever at 0% CPU, with no error and no progress.
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "20")

from data.dataset import BatchSpec, build_batch  # noqa: E402
from data.ingest import REGIONS, IngestConfig, cut_tile  # noqa: E402
from data.sources import M_PER_DEG, StereoDEM  # noqa: E402
from infer_global import BlendCanvas, feather_window, tile_origins  # noqa: E402
from model.unet import UNetConfig, build_model  # noqa: E402
from train import load_weights, pick_device  # noqa: E402


def region_box(region: str, span_km: float) -> tuple[float, float, float, float]:
    """A square box of `span_km` centred on the region.

    The regions in `data.ingest` are whole provinces, up to 1600 km across. At 75 m that
    is a 21k x 18k pixel canvas and 12 GB of float64 accumulator — the architecture note
    asks for 500 km regions of interest, and that is what this produces.
    """
    west, south, east, north = REGIONS[region]
    if east < west:
        east += 360.0
    clon, clat = (west + east) / 2, (south + north) / 2
    half = (span_km * 1000.0 / M_PER_DEG) / 2.0
    return clon - half, clat - half, clon + half, clat + half


def region_grid(region: str, tile_px: int, pixel_size_m: float, overlap_px: int,
                span_km: float = 400.0):
    """Tile origins and the lon/lat of each tile centre, over one region box."""
    west, south, east, north = region_box(region, span_km)
    deg_per_px = pixel_size_m / M_PER_DEG
    width = int((east - west) / deg_per_px)
    height = int((north - south) / deg_per_px)

    rows = tile_origins(height, tile_px, overlap_px)
    cols = tile_origins(width, tile_px, overlap_px)
    jobs = []
    for r in rows:
        for c in cols:
            lon = west + (c + tile_px / 2) * deg_per_px
            lat = north - (r + tile_px / 2) * deg_per_px
            jobs.append((r, c, lon % 360.0, lat))
    return jobs, height, width


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--region", default="ovda", choices=sorted(REGIONS))
    ap.add_argument("--raw", type=Path, default=Path("data_raw"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--globe", type=Path, default=None, help="write globe tiles here too")
    ap.add_argument("--tile-px", type=int, default=256)
    ap.add_argument("--overlap-px", type=int, default=64)
    ap.add_argument("--pixel-size", type=float, default=75.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--span-km", type=float, default=400.0,
                    help="square box centred on the region; the note asks for ~500 km ROIs")
    ap.add_argument("--max-tiles", type=int, default=0, help="cap, for a quick look")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-level", type=int, default=8)
    a = ap.parse_args()

    out = a.out or Path("outputs") / a.region
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)

    model = build_model(UNetConfig()).to(device).eval()
    used = load_weights(a.ckpt, model, device)
    print(f"loaded {used} weights from {a.ckpt}")

    stereo_img = a.raw / "mosaic_allstereo.img"
    stereo = (StereoDEM.from_label(stereo_img)
              if stereo_img.exists() and stereo_img.with_suffix(".xml").exists() else None)

    cfg = IngestConfig(core_px=a.tile_px, pixel_size_m=a.pixel_size, looks=("left", "right"))
    jobs, height, width = region_grid(a.region, a.tile_px, a.pixel_size, a.overlap_px, a.span_km)
    if a.max_tiles:
        jobs = jobs[: a.max_tiles]
    print(f"{a.region}: {width} x {height} px at {a.pixel_size:.0f} m "
          f"({width * a.pixel_size / 1000:.0f} x {height * a.pixel_size / 1000:.0f} km), "
          f"{len(jobs)} tiles")

    canvas = BlendCanvas(height, width, channels=4)  # z, sigma, sar, gtdr
    window = feather_window(a.tile_px, a.overlap_px).numpy()
    spec = BatchSpec(augment=False, pixel_size_m=a.pixel_size)
    rng = np.random.default_rng(0)

    from concurrent.futures import ThreadPoolExecutor

    def fetch(job):
        r, c, lon, lat = job
        try:
            return r, c, cut_tile(lon, lat, cfg, a.raw, stereo)
        except Exception as exc:
            print(f"  tile {lon:.2f},{lat:.2f}: {type(exc).__name__}: {exc}")
            return r, c, None

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r, c, tile in pool.map(fetch, jobs):
            done += 1
            if tile is None:
                continue
            batch_in = {k: torch.as_tensor(np.asarray(v)[None]) for k, v in tile.arrays.items()}
            b = build_batch({k: v.to(device) for k, v in batch_in.items()}, spec, rng)
            with torch.no_grad():
                o = model(b["x"], b["cond"], b["gtdr_up"])
            stack = np.stack([
                o["z_hat"][0, 0].cpu().numpy(),
                o["sigma"][0, 0].cpu().numpy(),
                b["rv_left"][0, 0].cpu().numpy(),
                b["gtdr_up"][0, 0].cpu().numpy(),
            ])
            canvas.add(stack, window, r, c)
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}  coverage {canvas.coverage:.0%}", flush=True)

    result = canvas.result()
    z, sigma, sar, gtdr = result
    finite = np.isfinite(z)
    if not finite.any():
        raise SystemExit("no tiles produced output")

    west, south, east, north = region_box(a.region, a.span_km)
    meta = {
        "region": a.region, "bounds_deg": [west, south, east, north],
        "pixel_size_m": a.pixel_size, "shape": list(z.shape),
        "checkpoint": str(a.ckpt), "weights": used,
        "elevation_m": {"min": float(np.nanmin(z)), "max": float(np.nanmax(z)),
                        "mean": float(np.nanmean(z))},
        "sigma_m": {"mean": float(np.nanmean(sigma)), "max": float(np.nanmax(sigma))},
        "relief_over_gtdr_m": float(np.nanstd(z - gtdr)),
        "coverage": float(canvas.coverage),
    }
    for name, arr in (("elevation", z), ("sigma", sigma), ("sar", sar), ("gtdr", gtdr)):
        np.save(out / f"{name}.npy", arr.astype(np.float32))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}")
    print(f"  elevation {meta['elevation_m']['min']:.0f} .. {meta['elevation_m']['max']:.0f} m")
    print(f"  the model's own departure from altimetry: {meta['relief_over_gtdr_m']:.1f} m rms")
    print(f"  mean 1 sigma {meta['sigma_m']['mean']:.0f} m")

    _preview(out, z, sigma, sar, gtdr, a.region)

    if a.globe:
        _globe_tiles(a.globe, a.region, z, sigma, sar, (west, south, east, north), a.max_level)


def _preview(out: Path, z, sigma, sar, gtdr, region: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [("Magellan SAR (dB)", sar, "gray", None),
              ("Magellan altimetry", gtdr, "terrain", None),
              ("ISHTAR elevation", z, "terrain", None),
              ("ISHTAR minus altimetry", z - gtdr, "RdBu_r", None),
              ("1 sigma (m)", sigma, "magma", None)]
    lo, hi = np.nanpercentile(z[np.isfinite(z)], [1, 99])
    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 4.4), constrained_layout=True)
    for ax, (title, arr, cmap, _) in zip(axes, fields):
        kw = {"vmin": lo, "vmax": hi} if cmap == "terrain" else {}
        if title.endswith("altimetry") and "minus" in title:
            kw = {}
        im = ax.imshow(arr, cmap=cmap, **kw)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    fig.suptitle(f"{region} — real Magellan data, learned topography", fontsize=12)
    fig.savefig(out / "preview.png", dpi=120)
    plt.close(fig)
    print(f"  preview {out / 'preview.png'}")


def _globe_tiles(globe: Path, region: str, z, sigma, sar, bounds, max_level: int) -> None:
    from export.demo_tiles import colour_relief, hillshade, write_imagery_pyramid
    from export.quantized_mesh import TileBounds, build_pyramid

    west, south, east, north = bounds
    tb = TileBounds(west - 360 if west > 180 else west, south,
                    east - 360 if east > 180 else east, north)
    filled = np.where(np.isfinite(z), z, np.nanmedian(z[np.isfinite(z)]))

    counts = build_pyramid(filled, tb, globe / f"terrain_{region}", max_level=max_level)
    print(f"  terrain_{region}: {sum(counts.values())} tiles to level {max_level}")

    s = np.nan_to_num(sar)
    s = np.clip((s + 12) / 24 * 255, 0, 255).astype(np.uint8)
    imgs = {
        f"sar_{region}": np.repeat(s[..., None], 3, axis=2),
        f"relief_{region}": colour_relief(filled),
        f"hillshade_{region}": np.repeat(hillshade(filled, z_factor=3.0)[..., None], 3, axis=2),
    }
    for name, img in imgs.items():
        n = write_imagery_pyramid(img, globe / name, max_level)
        print(f"  {name}: {sum(n.values())} tiles")


if __name__ == "__main__":
    main()
