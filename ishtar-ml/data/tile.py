"""Turn the raw Magellan rasters into the training tile store.

Reads the FMAP mosaics, GTDR, GSDR, GEDR and the Herrick stereo DEM; decodes DNs to dB;
rasterises the viewing geometry; cuts tiles on a common grid; writes Zarr.

    python -m data.tile --regions ovda alpha mead --out data_tiles/venus.zarr

Two rules this module exists to enforce, both from `CLAUDE.md`:

* Decode `DN -> RV(dB)` at ingest. Nothing downstream should ever see a raw DN.
* Cut tiles with a **context margin**. The altimeter footprint is 10 x 20 km and a bare
  512 px tile is 38 km, so `L_alt` on a marginless tile measures the padding, not the
  terrain (see `model.losses.loss_alt`). Tiles are therefore written at
  `core + 2 * margin` and the loss masks back to the core.

`rasterio` and `zarr` are imported lazily so the model and the test suite do not depend
on the geo stack.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from model import physics

from . import geometry

# Native FMAP posting, and the derived grid constants.
FMAP_RES_M = 75.0
GTDR_RES_M = 4641.0
GTDR_STRIDE_PX = int(round(GTDR_RES_M / FMAP_RES_M))  # 62
GTDR_NODATA = -32768


@dataclass
class TileSpec:
    core_px: int = 512
    """Pixels the network is scored on. 512 at 75 m = 38.4 km."""

    margin_px: int = 384
    """Context ring, ~3 sigma of the altimeter footprint at 75 m (28.8 km).

    Tiles are cut at `core_px + 2 * margin_px` so the footprint convolution in `L_alt`
    sees real terrain rather than padding out to 3 sigma from every core pixel. This is
    the single most expensive choice in the pipeline — it multiplies tile area by ~6 —
    so if disk is tight, shrink it and raise `alt_edge_margin_px` to match, accepting a
    weaker altimetry anchor rather than a silently wrong one.
    """

    pixel_size_m: float = FMAP_RES_M
    max_abs_lat_deg: float = 80.0
    stereo_trusted_scale_m: float = 1000.0

    @property
    def full_px(self) -> int:
        return self.core_px + 2 * self.margin_px

    @property
    def core_slice(self) -> slice:
        return slice(self.margin_px, self.margin_px + self.core_px)


@dataclass
class TileArrays:
    """One tile, in the layout `data.dataset.build_batch` expects."""

    dn_left: np.ndarray
    dn_right: np.ndarray
    dn_stereo: np.ndarray
    gtdr_up: np.ndarray
    gtdr_valid: np.ndarray
    stereo_dem: np.ndarray
    stereo_valid: np.ndarray
    emissivity: np.ndarray
    rms_slope: np.ndarray
    theta_left: float
    theta_right: float
    theta_stereo: float
    look_left: np.ndarray
    look_right: np.ndarray
    look_stereo: np.ndarray
    has_left: float
    has_right: float
    has_stereo: float
    has_stereo_dem: float
    lat_deg: float
    quad: str = ""


def decode_fmap(dn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`DN -> (RV in dB, valid)`, the numpy twin of `physics.rv_from_dn`."""
    valid = dn != physics.DN_NODATA
    rv = (dn.astype(np.float32) - 1.0) / physics.DN_SCALE - physics.DN_OFFSET
    return np.where(valid, rv, 0.0).astype(np.float32), valid


def decode_gtdr(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = raw != GTDR_NODATA
    return np.where(valid, raw, 0).astype(np.float32), valid


def upsample_posts(posts, out_shape: tuple[int, int], stride_px: int, offset_px: int | None = None):
    """Resample a post grid onto the fine grid with the posts landing where they belong.

    `torch.nn.functional.interpolate` stretches the input to fill the output, which places
    post `j` at `(j + 0.5) * H / P - 0.5`. The altimetry loss samples the fine grid at
    `offset + j * stride`, and those two agree only when `H` happens to equal `P * stride`.
    At 75 m with a 62 px stride they do not, and the last post of a 512 px tile lands
    ~6 px away from where the loss looks for it — so `L_alt`, the anchor that is supposed
    to keep the surface from drifting, compares the model at one place against altimetry
    interpolated from another.

    This maps output pixel `i` to post coordinate `(i - offset) / stride` explicitly, and
    clamps outside the post extent rather than extrapolating.
    """
    import torch
    import torch.nn.functional as F

    t = posts if torch.is_tensor(posts) else torch.from_numpy(np.asarray(posts)).float()
    t = t.float()
    while t.dim() < 4:
        t = t[None]
    _, _, ph, pw = t.shape
    h, w = out_shape
    off = stride_px // 2 if offset_px is None else offset_px

    def axis(n_out, n_post):
        idx = (torch.arange(n_out, dtype=torch.float32) - off) / stride_px
        idx = idx.clamp(0, n_post - 1)
        # grid_sample wants normalised coordinates with align_corners=True semantics.
        return 2.0 * idx / max(n_post - 1, 1) - 1.0

    gx = axis(w, pw)[None, :].expand(h, w)
    gy = axis(h, ph)[:, None].expand(h, w)
    grid = torch.stack([gx, gy], dim=-1)[None]
    out = F.grid_sample(t, grid, mode="bicubic", padding_mode="border", align_corners=True)
    return out[0, 0]


def upsample_gtdr(gtdr: np.ndarray, out_shape: tuple[int, int],
                  stride_px: int = GTDR_STRIDE_PX) -> np.ndarray:
    """Upsample the 4641 m posts to the 75 m grid, with the posts correctly placed.

    This becomes the base the network predicts a residual over, so it must be smooth and
    artefact-free: ringing here is elevation the model has to cancel rather than model.
    """
    return upsample_posts(gtdr, out_shape, stride_px).numpy()


def quad_id(lat_deg: float, lon_deg: float) -> str:
    """The 12 x 12 degree FMAP quadrangle a tile falls in.

    Splits are by whole quadrangle, never by tile: neighbouring tiles share terrain, and
    a random split would leak the validation set into training through the overlap.
    """
    q_lat = int(np.floor((lat_deg + 90.0) / 12.0))
    q_lon = int(np.floor((lon_deg % 360.0) / 12.0))
    return f"q{q_lat:02d}_{q_lon:02d}"


HELD_OUT_QUADS: tuple[str, ...] = ()
"""Validation quadrangles, filled in once the real mosaics are tiled.

Section 2.4 asks for a tessera region (Ovda), a plains region with small volcanoes, and a
crater field. Maxwell and Maat stay in *training* — they are demo regions you will look
at constantly, and a metric you have been eyeballing for weeks is not a held-out metric.
"""


def build_tile(
    sources: dict[str, np.ndarray],
    window: tuple[int, int, int, int],
    lat_deg: float,
    lon_deg: float,
    spec: TileSpec,
) -> TileArrays:
    """Cut one tile from open rasters. `sources` maps product key to a windowed array."""
    r0, c0, h, w = window
    shape = (h, w)

    gtdr_raw, gtdr_ok = decode_gtdr(sources["gtdr"])
    gtdr_up = upsample_gtdr(gtdr_raw, shape)
    # Carry the nodata mask forward. GTDR nodata decodes to 0 m, and Venus has no sea
    # level, so a dropped mask would anchor `L_alt` to zero elevation wherever the
    # altimeter has a gap — a silent, confident error over exactly the ~2% of the planet
    # it never measured.
    gtdr_valid = upsample_gtdr(gtdr_ok.astype(np.float32), shape) > 0.99

    look = {k: geometry.look_vector(k) for k in ("left", "right", "stereo")}
    theta = {k: float(geometry.INCIDENCE_MODELS[k].theta_rad(lat_deg)) for k in look}

    stereo = sources.get("stereo_dem")
    has_stereo_dem = float(stereo is not None)
    if stereo is None:
        stereo = np.zeros(shape, np.float32)
        stereo_valid = np.zeros(shape, np.float32)
    else:
        stereo_valid = np.isfinite(stereo).astype(np.float32)
        stereo = np.nan_to_num(stereo).astype(np.float32)

    return TileArrays(
        dn_left=sources["fmap_left"].astype(np.float32),
        dn_right=sources.get("fmap_right", np.zeros(shape, np.float32)).astype(np.float32),
        dn_stereo=sources.get("fmap_stereo", np.zeros(shape, np.float32)).astype(np.float32),
        gtdr_up=gtdr_up,
        gtdr_valid=gtdr_valid.astype(np.float32),
        stereo_dem=stereo,
        stereo_valid=stereo_valid,
        emissivity=sources.get("gedr", np.full(shape, 0.85, np.float32)).astype(np.float32),
        rms_slope=np.deg2rad(sources.get("gsdr", np.full(shape, 2.0, np.float32))).astype(np.float32),
        theta_left=theta["left"], theta_right=theta["right"], theta_stereo=theta["stereo"],
        look_left=look["left"], look_right=look["right"], look_stereo=look["stereo"],
        has_left=float((sources["fmap_left"] != 0).mean() > 0.5),
        has_right=float("fmap_right" in sources and (sources["fmap_right"] != 0).mean() > 0.5),
        has_stereo=float("fmap_stereo" in sources and (sources["fmap_stereo"] != 0).mean() > 0.5),
        has_stereo_dem=has_stereo_dem,
        lat_deg=lat_deg,
        quad=quad_id(lat_deg, lon_deg),
    )


def open_store(path: Path, spec: TileSpec, n_tiles: int):
    """Create the Zarr store with one array per channel, chunked one tile per chunk."""
    import zarr

    root = zarr.open_group(str(path), mode="a")
    n = spec.full_px
    raster = dict(shape=(n_tiles, n, n), chunks=(1, n, n), dtype="float32")
    for name in ("dn_left", "dn_right", "dn_stereo", "gtdr_up", "gtdr_valid", "stereo_dem",
                 "stereo_valid", "emissivity", "rms_slope"):
        root.require_dataset(name, **raster)
    for name in ("theta_left", "theta_right", "theta_stereo", "has_left", "has_right",
                 "has_stereo", "has_stereo_dem", "lat_deg"):
        root.require_dataset(name, shape=(n_tiles,), chunks=(4096,), dtype="float32")
    for name in ("look_left", "look_right", "look_stereo"):
        root.require_dataset(name, shape=(n_tiles, 2), chunks=(4096, 2), dtype="float32")
    root.require_dataset("quad", shape=(n_tiles,), chunks=(4096,), dtype="<U12")
    root.attrs.update({
        "pixel_size_m": spec.pixel_size_m,
        "core_px": spec.core_px,
        "margin_px": spec.margin_px,
        "gtdr_stride_px": GTDR_STRIDE_PX,
        "venus_radius_m": geometry.VENUS_RADIUS_M,
        "dn_encoding": "RV_dB = (DN - 1) / 5 - 20; DN 0 = nodata",
        "look_vec_convention": "down-range (away from radar), (east, north)",
    })
    return root


@dataclass(frozen=True)
class MosaicGrid:
    """The simple cylindrical grid the FMAP mosaics are on.

    Row 0 is the north edge, column 0 is longitude 0. Everything in the pipeline that
    converts between pixels and degrees goes through here, so there is exactly one place
    to be wrong.
    """

    width: int
    height: int
    lon_min_deg: float = 0.0
    lat_max_deg: float = 90.0

    @property
    def deg_per_px_lon(self) -> float:
        return 360.0 / self.width

    @property
    def deg_per_px_lat(self) -> float:
        return 180.0 / self.height

    def lon_lat(self, row: float, col: float) -> tuple[float, float]:
        """Centre of pixel `(row, col)` in degrees east and planetocentric latitude."""
        lon = self.lon_min_deg + (col + 0.5) * self.deg_per_px_lon
        lat = self.lat_max_deg - (row + 0.5) * self.deg_per_px_lat
        return lon % 360.0, lat

    def row_col(self, lon_deg: float, lat_deg: float) -> tuple[float, float]:
        col = ((lon_deg - self.lon_min_deg) % 360.0) / self.deg_per_px_lon - 0.5
        row = (self.lat_max_deg - lat_deg) / self.deg_per_px_lat - 0.5
        return row, col

    def pixel_size_m(self, lat_deg: float = 0.0) -> float:
        """Ground spacing in the longitude direction, which shrinks as cos(lat).

        Latitude spacing stays at the equatorial value on a cylindrical grid — which is
        exactly why training stops at 80 degrees and the caps are re-tiled.
        """
        import math

        return math.radians(self.deg_per_px_lon) * geometry.VENUS_RADIUS_M * math.cos(math.radians(lat_deg))


def iter_windows(
    grid: MosaicGrid,
    spec: TileSpec,
    bbox_deg: tuple[float, float, float, float],
    stride_px: int | None = None,
):
    """Yield `(row, col, lon, lat)` for every full tile inside `bbox_deg`.

    Windows are the *full* extent (`core + 2 * margin`) but step by the core size, so
    cores tile the region exactly once and margins overlap. Windows that would run off
    the top or bottom of the mosaic are skipped: padding a tile against the pole is the
    boundary condition the margin exists to avoid.

    Longitude wraps; latitude does not.
    """
    west, south, east, north = bbox_deg
    step = stride_px or spec.core_px
    n = spec.full_px

    row_start, _ = grid.row_col(west, north)
    row_end, _ = grid.row_col(west, south)
    r0 = max(0, int(row_start) - spec.margin_px)
    r1 = min(grid.height - n, int(row_end) - spec.margin_px)

    _, c_start = grid.row_col(west, north)
    _, c_end = grid.row_col(east, north)
    span = int((c_end - c_start) % grid.width) or grid.width

    for row in range(r0, r1 + 1, step):
        for offset in range(0, max(1, span - spec.core_px + 1), step):
            col = int(c_start - spec.margin_px + offset) % grid.width
            lon, lat = grid.lon_lat(row + n / 2, col + n / 2)
            if abs(lat) > spec.max_abs_lat_deg:
                continue
            yield row, col, lon, lat


def read_window(src, row: int, col: int, size: int, width: int) -> np.ndarray:
    """Read one window from an open rasterio dataset, wrapping in longitude.

    The mosaics are global and cylindrical, so a window near longitude 0 straddles the
    seam. rasterio will not wrap for you; it returns nodata past the edge, which produces
    a stripe of dead tiles at the prime meridian that is easy to miss.
    """
    from rasterio.windows import Window

    if col + size <= width:
        return src.read(1, window=Window(col, row, size, size))
    first = width - col
    left = src.read(1, window=Window(col, row, first, size))
    right = src.read(1, window=Window(0, row, size - first, size))
    return np.concatenate([left, right], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data_raw"))
    ap.add_argument("--out", type=Path, default=Path("data_tiles/venus.zarr"))
    ap.add_argument("--regions", nargs="*", default=["ovda"])
    ap.add_argument("--core-px", type=int, default=512)
    ap.add_argument("--margin-px", type=int, default=384)
    a = ap.parse_args()

    spec = TileSpec(core_px=a.core_px, margin_px=a.margin_px)
    missing = [p for p in ("fmap_left", "gtdr") if not any(a.raw.glob(f"*{p}*"))]
    print(f"tiles are {spec.full_px} px ({spec.full_px * spec.pixel_size_m / 1000:.1f} km), "
          f"core {spec.core_px} px, margin {spec.margin_px} px")
    if missing:
        raise SystemExit(
            f"missing required products in {a.raw}: {', '.join(missing)}\n"
            f"run: python -m data.download --list"
        )
    raise SystemExit(
        "Windowed reads need rasterio and the downloaded mosaics. `MosaicGrid`, "
        "`iter_windows`, `read_window` and `build_tile` are the pieces; wire them into a "
        "loop over --regions once the rasters are in place. Until then, `data.synthetic` "
        "provides the same tile layout."
    )


if __name__ == "__main__":
    # Run as a module (`python -m data.tile`), not as a script: this file uses
    # package-relative imports, which are unavailable when Python treats it as __main__
    # in its own directory.
    main()
