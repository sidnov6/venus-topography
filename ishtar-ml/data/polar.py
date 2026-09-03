"""Polar stereographic re-tiling for the caps.

Training stays equatorward of 80 degrees: a simple cylindrical grid oversamples longitude
by `1 / cos(lat)`, which is 5.8x at 80 degrees and unbounded at the pole, so a tile there
is a smeared strip the network has never seen anything like. Inference still has to cover
the caps, so they are re-projected to polar stereographic — conformal, so a tile looks
locally like the equatorial tiles the model was trained on — run, and projected back.

Formulae for a sphere of radius R, north pole case, with `k0 = 1` (tangent plane at the
pole rather than a secant plane; the scale error at 70 degrees is then 3%, which is well
inside the model's own uncertainty):

    rho = 2 R tan(pi/4 - lat/2)
    x   = rho sin(lon),  y = -rho cos(lon)

The south pole is the same with `lat -> -lat` and `y` flipped, so both are handled by one
`sign` factor rather than two code paths that can drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import VENUS_RADIUS_M


@dataclass(frozen=True)
class PolarGrid:
    """A square polar stereographic grid centred on a pole."""

    size: int
    pixel_size_m: float
    north: bool = True
    radius_m: float = VENUS_RADIUS_M

    @property
    def sign(self) -> float:
        return 1.0 if self.north else -1.0

    @property
    def half_extent_m(self) -> float:
        return self.size * self.pixel_size_m / 2.0

    @property
    def megapixels(self) -> float:
        return self.size**2 / 1e6

    def min_latitude_deg(self) -> float:
        """The latitude reached at the grid's edge midpoint — how far down the cap goes."""
        rho = self.half_extent_m
        lat = 90.0 - 2.0 * np.degrees(np.arctan(rho / (2.0 * self.radius_m)))
        return self.sign * lat

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        a = (np.arange(self.size) - (self.size - 1) / 2.0) * self.pixel_size_m
        return np.meshgrid(a, a, indexing="xy")

    def lon_lat(self) -> tuple[np.ndarray, np.ndarray]:
        """Degrees east and planetocentric latitude at every grid cell."""
        x, y = self.xy()
        rho = np.hypot(x, y)
        lat = self.sign * (90.0 - 2.0 * np.degrees(np.arctan(rho / (2.0 * self.radius_m))))
        lon = np.degrees(np.arctan2(x, -self.sign * y)) % 360.0
        return lon, lat

    def from_lon_lat(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse: degrees to grid column and row (float, for interpolation)."""
        lat = self.sign * np.asarray(lat_deg, dtype=np.float64)
        rho = 2.0 * self.radius_m * np.tan(np.radians(45.0 - lat / 2.0))
        lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
        x = rho * np.sin(lon)
        y = -self.sign * rho * np.cos(lon)
        col = x / self.pixel_size_m + (self.size - 1) / 2.0
        row = y / self.pixel_size_m + (self.size - 1) / 2.0
        return col, row

    def scale_factor(self) -> np.ndarray:
        """Local scale distortion `k = (1 + sin|lat|) / (2 ...)`; 1 at the pole.

        Stereographic projection is conformal — shapes are locally correct at every point,
        which is the property that matters here — but not equal-area. This is how much a
        metre on the grid differs from a metre on the ground.
        """
        _, lat = self.lon_lat()
        return 2.0 / (1.0 + np.sin(np.radians(np.abs(lat))))


def _bilinear(src: np.ndarray, col: np.ndarray, row: np.ndarray, wrap_columns: bool = False,
              fill: float = np.nan, clamp_rows: bool = False) -> np.ndarray:
    """Bilinear sample of `src[row, col]`, optionally wrapping in the column direction.

    `clamp_rows` extends the top and bottom rows by half a pixel. A global cylindrical
    raster's first row is centred at 89.5 degrees, not 90, so without it every polar grid
    cell within half a pixel of the pole comes back as nodata — a hole exactly where the
    cap projection exists to provide coverage.
    """
    h, w = src.shape
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    fc = col - c0
    fr = row - r0

    if wrap_columns:
        c0m, c1m = c0 % w, (c0 + 1) % w
        inside = np.ones_like(row, dtype=bool) if clamp_rows else (row >= 0) & (row <= h - 1)
    else:
        c0m, c1m = np.clip(c0, 0, w - 1), np.clip(c0 + 1, 0, w - 1)
        inside = (row >= 0) & (row <= h - 1) & (col >= 0) & (col <= w - 1)
    r0m, r1m = np.clip(r0, 0, h - 1), np.clip(r0 + 1, 0, h - 1)
    if clamp_rows:
        fr = np.where((row < 0) | (row > h - 1), 0.0, fr)

    top = src[r0m, c0m] * (1 - fc) + src[r0m, c1m] * fc
    bot = src[r1m, c0m] * (1 - fc) + src[r1m, c1m] * fc
    out = top * (1 - fr) + bot * fr
    return np.where(inside, out, fill)


def cylindrical_to_polar(raster: np.ndarray, grid: PolarGrid, lat_top_deg: float = 90.0,
                         lon_left_deg: float = 0.0, fill: float = np.nan) -> np.ndarray:
    """Resample a global north-up cylindrical raster onto the polar grid.

    Longitude wraps; latitude does not. A polar tile always straddles every meridian, so
    forgetting the wrap leaves a wedge of nodata that looks like missing data rather than
    an indexing bug.
    """
    h, w = raster.shape
    lon, lat = grid.lon_lat()
    col = ((lon - lon_left_deg) % 360.0) / (360.0 / w) - 0.5
    row = (lat_top_deg - lat) / (180.0 / h) - 0.5
    return _bilinear(raster.astype(np.float64), col, row, wrap_columns=True, fill=fill,
                     clamp_rows=True).astype(np.float32)


def polar_to_cylindrical(polar: np.ndarray, grid: PolarGrid, out_shape: tuple[int, int],
                         lat_top_deg: float = 90.0, lon_left_deg: float = 0.0,
                         fill: float = np.nan) -> np.ndarray:
    """Project a polar-grid result back onto the cylindrical output raster.

    Only cells inside the cap get a value; everything else comes back as `fill`, so the
    caller can blend it against the equatorial inference without guessing at coverage.
    """
    h, w = out_shape
    lat = lat_top_deg - (np.arange(h) + 0.5) * (180.0 / h)
    lon = lon_left_deg + (np.arange(w) + 0.5) * (360.0 / w)
    lon_g, lat_g = np.meshgrid(lon, lat)
    col, row = grid.from_lon_lat(lon_g, lat_g)
    return _bilinear(polar.astype(np.float64), col, row, wrap_columns=False, fill=fill).astype(np.float32)


def cap_grid(pixel_size_m: float = 75.0, min_abs_lat_deg: float = 75.0, north: bool = True,
             radius_m: float = VENUS_RADIUS_M) -> PolarGrid:
    """A polar grid that reaches down to `min_abs_lat_deg`, with a margin of overlap.

    The overlap is deliberate: the cap result is feathered against the cylindrical
    inference between 75 and 80 degrees rather than butt-joined at a hard latitude.

    **This grid describes an extent, not an array.** A 75 m cap reaching 75 degrees is
    42,500 px square — 1.8 gigapixels, 7 GB in float32 — so it must be run tile by tile
    through `infer_global.run_tiled` like everything else. `megapixels` is there to make
    that obvious before something tries to allocate it. Only the coarse products (GTDR at
    4641 m is a 690 px cap) fit in memory whole.
    """
    rho = 2.0 * radius_m * np.tan(np.radians(45.0 - min_abs_lat_deg / 2.0))
    size = int(np.ceil(2.0 * rho / pixel_size_m))
    return PolarGrid(size=size + size % 2, pixel_size_m=pixel_size_m, north=north, radius_m=radius_m)


def blend_weight(lat_deg: np.ndarray, inner_deg: float = 80.0, outer_deg: float = 75.0) -> np.ndarray:
    """Raised-cosine weight for the cap, 1 poleward of `inner_deg` and 0 equatorward of
    `outer_deg`. The complement weights the cylindrical inference."""
    a = np.abs(np.asarray(lat_deg, dtype=np.float64))
    t = np.clip((a - outer_deg) / max(inner_deg - outer_deg, 1e-9), 0.0, 1.0)
    return (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)
