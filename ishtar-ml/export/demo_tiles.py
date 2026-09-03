"""Generate a runnable demo tile set for the globe, from synthetic Venus.

The real pipeline needs ~300 GB of Magellan products. This produces a small but
structurally identical tile set — geodetic pyramid, quantized-mesh terrain on the Venus
sphere, XYZ imagery — so `ishtar-globe` renders a real globe today and the tiling
conventions can be verified end to end before any download.

    python -m export.demo_tiles --out ../ishtar-globe/public/tiles --max-level 5

Everything it writes is synthetic. The globe labels it as such.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export.quantized_mesh import TileBounds, build_pyramid  # noqa: E402

# Regions of interest, as the real product ships them: global coverage to one level, and
# deeper tiles only inside a handful of boxes. Level 12 globally would be 33 million tiles.
ROIS: dict[str, TileBounds] = {
    "maxwell": TileBounds(-6.0, 60.0, 14.0, 70.0),
    "mead": TileBounds(52.0, 8.0, 62.0, 17.0),
    "alpha": TileBounds(0.0, -30.0, 16.0, -18.0),
}


def planet_dem(height: int = 512, width: int = 1024, seed: int = 0) -> np.ndarray:
    """A whole-planet elevation field with Venus-like statistics.

    Spectrally synthesised so the hypsometry is plausible — most of the surface within a
    kilometre of the mean, with a few highland provinces — rather than uniform noise.
    """
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    f = np.sqrt(fy**2 + fx**2)
    f[0, 0] = 1.0
    amp = f ** (-1.6)
    amp[0, 0] = 0.0
    field = np.real(np.fft.ifft2(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, (height, width)))))
    field = (field - field.mean()) / field.std()

    # Squash toward the mean, then let the tail reach Maxwell-like heights.
    dem = 900.0 * np.tanh(field) + 2600.0 * np.clip(field - 1.2, 0, None) ** 2
    # Taper the poles: a cylindrical grid oversamples them and the seam is ugly.
    lat = np.linspace(90, -90, height)[:, None]
    return (dem * np.cos(np.deg2rad(lat)) ** 0.25).astype(np.float32)


def hillshade(dem: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0,
              z_factor: float = 6.0) -> np.ndarray:
    """Standard hillshade, 0..255. `z_factor` exaggerates because the demo DEM is coarse."""
    dy, dx = np.gradient(dem.astype(np.float64) * z_factor)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(360.0 - azimuth_deg + 90.0)
    alt = np.deg2rad(altitude_deg)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(shade * 255.0, 0, 255).astype(np.uint8)


# Hypsometric ramp, metres -> RGB. Venus spans about -3 km to +11 km.
RAMP = [
    (-3000, (20, 30, 60)), (-1000, (40, 80, 110)), (0, (90, 120, 100)),
    (1000, (150, 140, 90)), (3000, (190, 150, 80)), (6000, (220, 180, 120)),
    (11000, (250, 245, 235)),
]


def colour_relief(dem: np.ndarray) -> np.ndarray:
    stops = np.array([s for s, _ in RAMP], dtype=np.float64)
    cols = np.array([c for _, c in RAMP], dtype=np.float64)
    out = np.empty(dem.shape + (3,), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.interp(dem, stops, cols[:, c]).astype(np.uint8)
    return out


def _smooth_field(shape: tuple[int, int], rng: np.random.Generator, factor: int = 16) -> np.ndarray:
    """Low-frequency unit-variance noise, for the fields that vary slowly across the planet."""
    from PIL import Image

    coarse = rng.normal(size=(max(4, shape[0] // factor), max(4, shape[1] // factor)))
    up = np.asarray(Image.fromarray(coarse).resize((shape[1], shape[0]), Image.BICUBIC))
    return (up - up.mean()) / (up.std() + 1e-9)


def uncertainty_like(dem: np.ndarray, seed: int = 2) -> np.ndarray:
    """A plausible 1-sigma field, as RGBA "confidence fog".

    Uncertainty on the real product is the model's own variance head, and it is high in
    two places: where the terrain is steep enough for layover to mask the physics, and
    where no second look and no stereo DEM ever constrained the cross-track slope. Both
    are imitated here so the layer reads correctly — clear where the model is confident,
    grey where it is not.
    """
    rng = np.random.default_rng(seed)
    dy, dx = np.gradient(dem.astype(np.float64))
    steep = np.hypot(dx, dy)
    steep = (steep - steep.min()) / max(float(np.ptp(steep)), 1e-9)

    # Coverage: a few wide bands of "second look available", the ~17% of the real planet.
    lon = np.linspace(0, 2 * np.pi, dem.shape[1])[None, :]
    covered = (np.sin(3 * lon + 0.7) > 0.75) * np.ones((dem.shape[0], 1))

    sigma = 40.0 + 260.0 * steep + 120.0 * (1 - covered)
    sigma += rng.normal(0, 8, dem.shape)
    alpha = np.clip((sigma - 40.0) / 300.0, 0.0, 1.0)

    out = np.zeros(dem.shape + (4,), np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = 190
    out[..., 3] = (alpha * 200).astype(np.uint8)
    return out


def stereo_coverage_like(dem: np.ndarray) -> np.ndarray:
    """Where a stereo DEM constrained training — about 20% of the real planet."""
    lat = np.linspace(90, -90, dem.shape[0])[:, None]
    lon = np.linspace(0, 360, dem.shape[1])[None, :]
    covered = (np.sin(np.radians(2.5 * lon) + 0.4) > 0.55) & (np.abs(lat) < 70)
    out = np.zeros(dem.shape + (4,), np.uint8)
    out[..., 1] = 220
    out[..., 2] = 140
    out[..., 3] = (covered * 110).astype(np.uint8)
    return out


def emissivity_like(dem: np.ndarray, seed: int = 3) -> np.ndarray:
    """Magellan emissivity: near 0.85 over most of the planet, dropping sharply on the
    high-emissivity-anomaly highlands above about 4.5 km."""
    e = 0.85 + 0.015 * _smooth_field(dem.shape, np.random.default_rng(seed))
    e = np.where(dem > 4500, 0.55, e)
    norm = np.clip((e - 0.5) / 0.4, 0, 1)
    out = np.zeros(dem.shape + (3,), np.uint8)
    out[..., 0] = (255 * (1 - norm)).astype(np.uint8)
    out[..., 1] = (120 + 80 * norm).astype(np.uint8)
    out[..., 2] = (200 * norm).astype(np.uint8)
    return out


def sar_like(dem: np.ndarray, seed: int = 1) -> np.ndarray:
    """A backscatter-looking image: slope shading toward a fixed look plus speckle.

    Not a physical render — the demo DEM is 20 km per pixel, far coarser than anything
    the radar model applies to. It exists so the imagery pyramid has content and the
    alignment between imagery and terrain can be checked.
    """
    rng = np.random.default_rng(seed)
    _, dx = np.gradient(dem.astype(np.float64))
    v = 128 + 90 * np.tanh(dx / 60.0)
    v = v * rng.gamma(8.0, 1 / 8.0, size=dem.shape)
    return np.clip(v, 0, 255).astype(np.uint8)


# The six sites the globe knows about, mirrored from ishtar-globe/src/venus.ts.
# Mead is the alignment check from the architecture note's globe work plan: if the marker
# does not land under the crater in the SAR mosaic, the tiling scheme is wrong.
SITES: list[tuple[str, float, float]] = [
    ("Maxwell Montes", 3.3, 65.2),
    ("Maat Mons", 194.6, 0.5),
    ("Ovda Regio", 85.6, -2.8),
    ("Artemis Corona", 135.0, -35.0),
    ("Mead crater", 57.2, 12.5),
    ("Alpha Regio", 5.0, -25.0),
]

GRATICULE_RGBA = (110, 190, 255, 90)
MARKER_RGBA = (255, 90, 60, 235)


def write_graticule_pyramid(root: Path, max_level: int, tile_size: int = 256,
                            spacing_deg: float = 10.0) -> dict[str, int]:
    """A transparent overlay of lat/lon lines and site markers.

    This is the imagery-alignment check: the marker is drawn from the site's degrees, the
    camera flies to the same degrees, and the terrain is tiled from the same degrees. If
    any of the three disagree — a Web Mercator tiling scheme is the classic way — the
    marker and the feature separate visibly, and `scripts/smoke.mjs` fails on it.
    """
    from PIL import Image, ImageDraw

    counts: dict[str, int] = {}
    for level in range(max_level + 1):
        nx, ny = 2 ** (level + 1), 2**level
        span = 180.0 / 2**level
        for x in range(nx):
            for y in range(ny):
                west = -180.0 + x * span
                south = -90.0 + y * span   # TMS: y increases north
                img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
                d = ImageDraw.Draw(img)

                px = lambda lon: (lon - west) / span * tile_size
                py = lambda lat: (south + span - lat) / span * tile_size

                first = np.ceil(west / spacing_deg) * spacing_deg
                for lon in np.arange(first, west + span + 1e-9, spacing_deg):
                    d.line([(px(lon), 0), (px(lon), tile_size)], fill=GRATICULE_RGBA, width=1)
                first = np.ceil(south / spacing_deg) * spacing_deg
                for lat in np.arange(first, south + span + 1e-9, spacing_deg):
                    d.line([(0, py(lat)), (tile_size, py(lat))], fill=GRATICULE_RGBA, width=1)

                for _, lon_e, lat in SITES:
                    lon = lon_e - 360.0 if lon_e > 180.0 else lon_e
                    if not (west <= lon <= west + span and south <= lat <= south + span):
                        continue
                    cx, cy = px(lon), py(lat)
                    r = max(6, tile_size // 22)
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=MARKER_RGBA, width=3)
                    d.line([(cx - r * 1.8, cy), (cx + r * 1.8, cy)], fill=MARKER_RGBA, width=2)
                    d.line([(cx, cy - r * 1.8), (cx, cy + r * 1.8)], fill=MARKER_RGBA, width=2)

                out = root / str(level) / str(x)
                out.mkdir(parents=True, exist_ok=True)
                img.save(out / f"{y}.png", optimize=True)
        counts[str(level)] = nx * ny
    return counts


def write_imagery_pyramid(image: np.ndarray, root: Path, max_level: int,
                          tile_size: int = 256) -> dict[str, int]:
    """Geodetic (2 x 1) XYZ pyramid, `{z}/{x}/{y}.png`, y increasing *south*.

    Cesium's `UrlTemplateImageryProvider` uses `{reverseY}` for TMS-ordered pyramids; the
    layer definitions in `src/layers.ts` do exactly that, so tiles are written in TMS row
    order (y increasing north) to match.
    """
    from PIL import Image

    counts: dict[str, int] = {}
    src = Image.fromarray(image)  # RGB or RGBA; the mode carries through the pyramid
    for level in range(max_level + 1):
        nx, ny = 2 ** (level + 1), 2**level
        full = src.resize((nx * tile_size, ny * tile_size), Image.LANCZOS)
        arr = np.asarray(full)
        for x in range(nx):
            for y in range(ny):
                # Row 0 of the array is north; TMS y = 0 is south.
                top = (ny - 1 - y) * tile_size
                crop = arr[top : top + tile_size, x * tile_size : (x + 1) * tile_size]
                out = root / str(level) / str(x)
                out.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop).save(out / f"{y}.png", optimize=True)
        counts[str(level)] = nx * ny
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("../ishtar-globe/public/tiles"))
    ap.add_argument("--max-level", type=int, default=5)
    ap.add_argument("--dem-height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi", nargs="*", default=["maxwell"], choices=sorted(ROIS) + [],
                    help="regions to carry to --roi-level; pass none to skip")
    ap.add_argument("--roi-level", type=int, default=7)
    a = ap.parse_args()

    dem = planet_dem(a.dem_height, a.dem_height * 2, a.seed)
    world = TileBounds(-180.0, -90.0, 180.0, 90.0)
    print(f"synthetic planet DEM {dem.shape}, {dem.min():.0f}..{dem.max():.0f} m")

    a.out.mkdir(parents=True, exist_ok=True)
    counts = build_pyramid(dem, world, a.out / "terrain_ishtar", max_level=a.max_level)
    print(f"terrain_ishtar: {sum(counts.values())} tiles, levels 0-{a.max_level}")

    # The GTDR baseline terrain: the same surface seen through a coarse altimeter. This
    # is the comparison the swipe tool exists for — what the model adds over altimetry.
    from PIL import Image

    src = Image.fromarray(dem)
    coarse = np.asarray(
        src.resize((dem.shape[1] // 8, dem.shape[0] // 8), Image.BOX)
           .resize((dem.shape[1], dem.shape[0]), Image.BICUBIC)
    )
    counts = build_pyramid(coarse, world, a.out / "terrain_gtdr", max_level=a.max_level)
    print(f"terrain_gtdr:   {sum(counts.values())} tiles")

    # Regions of interest: deeper terrain over a few boxes, from the same field. The real
    # product does this at 75 m inside ~10 boxes and 225 m globally.
    for roi in a.roi:
        box = ROIS[roi]
        counts = build_pyramid(dem, world, a.out / "terrain_ishtar",
                               min_level=a.max_level + 1, max_level=a.roi_level, box=box)
        print(f"terrain_ishtar ROI {roi}: {sum(counts.values())} tiles, "
              f"levels {a.max_level + 1}-{a.roi_level}")

    for name, image in (
        ("sar_left", np.repeat(sar_like(dem)[..., None], 3, axis=2)),
        ("sar_right", np.repeat(sar_like(dem, seed=11)[..., None], 3, axis=2)),
        ("colour_relief", colour_relief(dem)),
        ("hillshade", np.repeat(hillshade(dem)[..., None], 3, axis=2)),
        ("uncertainty", uncertainty_like(dem)),
        ("stereo_coverage", stereo_coverage_like(dem)),
        ("emissivity", emissivity_like(dem)),
    ):
        n = write_imagery_pyramid(image, a.out / name, a.max_level)
        print(f"{name:14s}: {sum(n.values())} tiles, levels 0-{a.max_level}")

    n = write_graticule_pyramid(a.out / "graticule", a.max_level)
    print(f"{'graticule':14s}: {sum(n.values())} tiles (alignment check)")

    # The globe reads this to clamp each layer's maximumLevel. Without it, Cesium keeps
    # requesting levels the pyramid does not have and the console fills with 404s.
    import json

    (a.out / "manifest.json").write_text(json.dumps({
        "synthetic": True,
        "maxLevel": {
            k: a.max_level for k in (
                "sar_left", "sar_right", "colour_relief", "hillshade",
                "uncertainty", "stereo_coverage", "emissivity", "graticule",
            )
        },
        "terrainMaxLevel": {
            "terrain_ishtar": a.roi_level if a.roi else a.max_level,
            "terrain_gtdr": a.max_level,
        },
        "roi": {r: [ROIS[r].west, ROIS[r].south, ROIS[r].east, ROIS[r].north] for r in a.roi},
        "note": "Generated by export/demo_tiles.py from synthetic terrain. Not Magellan data.",
    }, indent=2))

    total = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file())
    print(f"\ntotal {total / 1e6:.1f} MB in {a.out}")
    print("run `npm run dev` in ishtar-globe to see it")


if __name__ == "__main__":
    main()
