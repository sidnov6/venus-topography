"""Cut real Magellan tiles into a training store.

    python -m data.ingest --regions ovda alpha mead --out data_tiles/venus.npz

Everything is sampled onto one lon/lat grid per tile by `data.sources`, so the products'
differing resolutions, origins and extents are resolved in one place. The output matches
`data.synthetic` key for key, so training, evaluation and inference cannot tell the two
apart.

The 75 m SAR mosaics are read by window straight from S3 unless a local copy exists.
A 512 px tile touches roughly 2 x 2 of the mosaics' internal 256 px tiles, so cutting a
few thousand tiles costs a few GB of transfer rather than the 117 GB the full product
would.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry
from .sources import PRODUCTS, M_PER_DEG, StereoDEM, Window, sample

# Reading a remote COG efficiently needs these; without the first, GDAL lists the whole
# bucket prefix on every open.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("VSI_CACHE", "TRUE")
# A stalled HTTP read otherwise blocks forever at 0% CPU, with no error and no progress.
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "20")
os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


# Regions chosen for terrain variety and stereo overlap, not for looks. Ovda and the
# plains quads are the held-out set; Maxwell and Maat are demo sites and stay in training,
# because a metric you have been staring at for weeks is not a held-out metric.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "ovda":      (75.0, -12.0, 100.0, 4.0),      # tessera, held out
    "thetis":    (110.0, -20.0, 135.0, -3.0),    # tessera + deformation
    "alpha":     (0.0, -32.0, 16.0, -18.0),      # tessera and pancake domes
    "mead":      (50.0, 6.0, 65.0, 19.0),        # the largest impact crater
    "guinevere": (325.0, 12.0, 350.0, 32.0),     # smooth plains, low relief
    "lavinia":   (340.0, -47.0, 5.0, -30.0),     # plains with small volcanoes
    "eistla":    (16.0, 8.0, 40.0, 26.0),        # shield volcanoes
    "maxwell":   (352.0, 60.0, 16.0, 72.0),      # demo: highest relief, crosses the meridian
    "maat":      (188.0, 2.0, 202.0, 14.0),      # demo: large shield volcano
    "artemis":   (125.0, -45.0, 150.0, -25.0),   # corona rim
}

HELD_OUT = ("ovda", "guinevere")

RASTER_KEYS = ("dn_left", "dn_right", "dn_stereo", "gtdr_up", "gtdr_valid",
               "stereo_dem", "stereo_valid", "emissivity", "rms_slope")
SCALAR_KEYS = ("theta_left", "theta_right", "theta_stereo", "has_left", "has_right",
               "has_stereo", "has_stereo_dem", "lat_deg")
VECTOR_KEYS = ("look_left", "look_right", "look_stereo")


@dataclass
class IngestConfig:
    pixel_size_m: float = 75.0
    core_px: int = 512
    margin_px: int = 0
    """Context ring for the altimetry footprint. 384 px is the honest value (see
    `losses.loss_alt`), but it multiplies transfer by ~6; the driver defaults to 0 and
    sets `alt_edge_margin_px` instead, which is the cheaper half of the same trade."""

    looks: tuple[str, ...] = ("left", "right")
    """Which SAR looks to read. The stereo-look (Cycle 3) mosaic has no JPEG variant, so
    it is 87 GB uncompressed and its window reads move several times the bytes of the
    other two for 17% coverage. It is off by default and enabled with `--looks`."""

    min_valid_sar: float = 0.55
    """Reject tiles that are mostly inter-swath gap. Magellan's left-look mosaic covers
    92-96% of the planet, but the missing part is in stripes, so individual tiles can be
    almost entirely nodata."""

    max_abs_lat_deg: float = 78.0


@dataclass
class Tile:
    arrays: dict[str, np.ndarray]
    lon: float
    lat: float
    region: str
    quad: str


def tile_centres(region: str, cfg: IngestConfig) -> list[tuple[float, float]]:
    """Non-overlapping tile centres covering a region, skipping the +/-180 seam.

    A region whose west bound exceeds its east bound wraps the meridian; those tiles are
    cut from the eastern part only, because `sources.sample` refuses a window that
    straddles the seam rather than silently returning a mirrored strip.
    """
    west, south, east, north = REGIONS[region]
    if east < west:
        east += 360.0
    step = cfg.core_px * cfg.pixel_size_m / M_PER_DEG

    out = []
    lat = south + step / 2
    while lat < north:
        if abs(lat) <= cfg.max_abs_lat_deg:
            lon = west + step / 2
            while lon < east:
                w = lon - step / 2
                e = lon + step / 2
                # Skip anything crossing the seam in either direction.
                if not (w < 180.0 < e or (w % 360.0) > (e % 360.0)):
                    out.append((lon % 360.0, lat))
                lon += step
        lat += step
    return out


def cut_tile(lon: float, lat: float, cfg: IngestConfig, root: Path,
             stereo: StereoDEM | None) -> Tile | None:
    """Sample every product over one tile. Returns None if the SAR is too gappy."""
    n = cfg.core_px + 2 * cfg.margin_px
    span_km = n * cfg.pixel_size_m / 1000.0
    win = Window.centred(lon, lat, span_km, cfg.pixel_size_m)

    dn_left, ok_left = sample(PRODUCTS["sar_left_75m"].uri(root), win, nodata_out=0.0)
    if float(ok_left.mean()) < cfg.min_valid_sar:
        return None

    if "right" in cfg.looks:
        dn_right, ok_right = sample(PRODUCTS["sar_right_75m"].uri(root), win, nodata_out=0.0)
    else:
        dn_right = np.zeros_like(dn_left); ok_right = np.zeros_like(ok_left)
    if "stereo" in cfg.looks:
        dn_st, ok_st = sample(PRODUCTS["sar_stereo_75m"].uri(root), win, nodata_out=0.0)
    else:
        dn_st = np.zeros_like(dn_left); ok_st = np.zeros_like(ok_left)

    gtdr, ok_gtdr = sample(PRODUCTS["gtdr"].uri(root), win, resampling="cubic", nodata_out=0.0)
    emis, ok_emis = sample(PRODUCTS["gedr"].uri(root), win, resampling="bilinear", nodata_out=0.85)
    gsdr, ok_gsdr = sample(PRODUCTS["gsdr"].uri(root), win, resampling="bilinear", nodata_out=0.0)
    gtdr = np.nan_to_num(gtdr)

    if stereo is not None:
        sdem, ok_sdem = stereo.read(win, nodata_out=0.0)
        # Tie the stereo DEM to the altimetry, per tile.
        #
        # Measured over 50 patches: stereo = 0.982 * GTDR - 795 m, correlation 0.9981,
        # scatter 71 m about that line. The scatter is the Herrick DEM's own quoted 50-100 m
        # vertical accuracy, so the *shape* is right and it is the datum that is off by
        # about 795 m. Stereo gives relative heights; the absolute tie has to come from
        # altimetry, which is what GTDR is for.
        #
        # Removing the median difference per tile also absorbs any regional drift, and it
        # leaves exactly what `L_stereo` should be teaching: the departure from GTDR in
        # the 100 m - 10 km band. `L_alt` keeps the absolute level.
        tie = ok_sdem & ok_gtdr
        if tie.sum() > 64:
            sdem = sdem - float(np.median(sdem[tie] - gtdr[tie]))
        else:
            ok_sdem = np.zeros_like(ok_sdem)
    else:
        sdem = np.zeros((win.height, win.width), np.float32)
        ok_sdem = np.zeros_like(sdem, bool)

    look = {k: geometry.look_vector(k) for k in ("left", "right", "stereo")}
    theta = {k: float(geometry.INCIDENCE_MODELS[k].theta_rad(lat)) for k in look}

    arrays = {
        "dn_left": np.where(ok_left, dn_left, 0).astype(np.float32),
        "dn_right": np.where(ok_right, dn_right, 0).astype(np.float32),
        "dn_stereo": np.where(ok_st, dn_st, 0).astype(np.float32),
        "gtdr_up": np.nan_to_num(gtdr).astype(np.float32),
        "gtdr_valid": ok_gtdr.astype(np.float32),
        "stereo_dem": np.nan_to_num(sdem).astype(np.float32),
        "stereo_valid": ok_sdem.astype(np.float32),
        "emissivity": np.nan_to_num(emis, nan=0.85).astype(np.float32),
        # GSDR is stored with its own scale; `rms_slope` downstream is radians.
        "rms_slope": np.deg2rad(np.nan_to_num(gsdr)).astype(np.float32),
        "theta_left": np.float32(theta["left"]),
        "theta_right": np.float32(theta["right"]),
        "theta_stereo": np.float32(theta["stereo"]),
        "look_left": look["left"], "look_right": look["right"], "look_stereo": look["stereo"],
        "has_left": np.float32(1.0),
        "has_right": np.float32(float(ok_right.mean()) > cfg.min_valid_sar),
        "has_stereo": np.float32(float(ok_st.mean()) > cfg.min_valid_sar),
        "has_stereo_dem": np.float32(float(ok_sdem.mean()) > 0.2),
        "lat_deg": np.float32(lat),
    }
    return Tile(arrays, lon, lat, "", quad=f"q{int((lat + 90) // 12):02d}_{int((lon % 360) // 12):02d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", nargs="*", default=sorted(REGIONS))
    ap.add_argument("--raw", type=Path, default=Path("data_raw"))
    ap.add_argument("--out", type=Path, default=Path("data_tiles/venus.npz"))
    ap.add_argument("--core-px", type=int, default=512)
    ap.add_argument("--pixel-size", type=float, default=75.0)
    ap.add_argument("--limit", type=int, help="stop after this many tiles (for a smoke run)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--looks", default="left,right",
                    help="SAR looks to read; add 'stereo' for the 87 GB Cycle 3 mosaic")
    ap.add_argument("--count-only", action="store_true", help="report the tile count and stop")
    ap.add_argument("--per-region", type=int, default=0,
                    help="cap tiles per region, strided so coverage stays spread out")
    a = ap.parse_args()

    cfg = IngestConfig(core_px=a.core_px, pixel_size_m=a.pixel_size,
                       looks=tuple(x.strip() for x in a.looks.split(",") if x.strip()))
    stereo_img = a.raw / "mosaic_allstereo.img"
    stereo = None
    if stereo_img.exists() and stereo_img.with_suffix(".xml").exists():
        stereo = StereoDEM.from_label(stereo_img)
        if not stereo.is_complete():
            got = stereo_img.stat().st_size / stereo.expected_bytes
            print(f"stereo DEM is {got:.0%} downloaded; rows run north to south, so "
                  f"coverage is complete down to {stereo.available_south_deg():.1f} deg "
                  "and nodata below it")
    if stereo is None:
        print("no stereo DEM available; L_stereo will have nothing to see")
    else:
        print(f"stereo DEM {stereo.width} x {stereo.height} @ {stereo.pixel_size_m:.0f} m, "
              f"bounds {tuple(round(b, 1) for b in stereo.bounds_deg)}")

    def region_tiles(r):
        c = tile_centres(r, cfg)
        if a.per_region and len(c) > a.per_region:
            # Stride rather than truncate: a contiguous prefix would be one corner of the
            # region, and the point of several regions is terrain variety.
            step = len(c) / a.per_region
            c = [c[int(i * step)] for i in range(a.per_region)]
        return c

    jobs = [(r, lon, lat) for r in a.regions for lon, lat in region_tiles(r)]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} candidate tiles of {cfg.core_px} px over {len(a.regions)} regions")
    for r in a.regions:
        print(f"    {r:11s} {len(region_tiles(r)):4d} tiles"
              + ("   [held out]" if r in HELD_OUT else ""))
    print(f"  looks: {', '.join(cfg.looks)}")
    if a.count_only:
        return

    from concurrent.futures import ThreadPoolExecutor

    kept: list[Tile] = []
    regions: list[str] = []

    def work(job):
        region, lon, lat = job
        try:
            t = cut_tile(lon, lat, cfg, a.raw, stereo)
        except Exception as exc:  # a transient S3 failure should not lose the run
            print(f"  {region} {lon:.1f},{lat:.1f}: {type(exc).__name__}: {exc}")
            return None
        return (region, t)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, res in enumerate(pool.map(work, jobs)):
            if res and res[1]:
                regions.append(res[0])
                kept.append(res[1])
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} scanned, {len(kept)} kept", flush=True)

    if not kept:
        raise SystemExit("no tiles survived the coverage filter")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    store = {k: np.stack([t.arrays[k] for t in kept]) for k in RASTER_KEYS + SCALAR_KEYS + VECTOR_KEYS}
    store["region"] = np.array(regions)
    store["quad"] = np.array([t.quad for t in kept])
    store["lon"] = np.array([t.lon for t in kept], np.float32)
    store["lat"] = np.array([t.lat for t in kept], np.float32)
    np.savez_compressed(a.out, **store)

    n_right = int(sum(t.arrays["has_right"] for t in kept))
    n_stereo_look = int(sum(t.arrays["has_stereo"] for t in kept))
    n_stereo_dem = int(sum(t.arrays["has_stereo_dem"] for t in kept))
    print(f"\nwrote {a.out}  ({a.out.stat().st_size / 1e9:.2f} GB)")
    print(f"  {len(kept)} tiles   right-look {n_right} ({100 * n_right / len(kept):.0f}%)   "
          f"stereo-look {n_stereo_look} ({100 * n_stereo_look / len(kept):.0f}%)   "
          f"stereo DEM {n_stereo_dem} ({100 * n_stereo_dem / len(kept):.0f}%)")
    held = sum(1 for r in regions if r in HELD_OUT)
    print(f"  held-out regions {HELD_OUT}: {held} tiles")


if __name__ == "__main__":
    main()
