"""Real Magellan products, sampled by longitude and latitude.

Every product here is on a cylindrical projection of the 6051.8 km sphere, so `x = R*lon`
and `y = R*lat` and no reprojection is needed — but they do *not* share a grid. They
differ in resolution (75 m, 225 m, 600 m, 4641 m), in origin, and in extent: the SAR
mosaics stop at about +/-80 degrees while the altimetry is global, and the stereo DEM is a
raw PDS array covering one irregular block of the planet.

So the one thing this module exists to do is take a lon/lat box and return each product
resampled onto that same box, with an explicit validity mask. `data/tile.py` never sees a
pixel index.

Two of the products are awkward in ways worth stating:

* The 75 m mosaics are 507k x 231k pixel JPEG-compressed COGs, 117 GB uncompressed and
  17 GB as stored. They are tiled 256 x 256, so a window read over HTTP fetches only the
  tiles it touches. Nothing here ever downloads one whole.
* The stereo DEM is a headerless float32 `.img` with its geometry in a detached PDS4 XML
  label, and its values are radii offset by 6040000 m rather than elevations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VENUS_RADIUS_M = 6_051_800.0
M_PER_DEG = math.radians(1.0) * VENUS_RADIUS_M  # 105632.3 m at the equator

S3 = "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic"
PDS_STEREO = ("https://pds-geosciences.wustl.edu/mgn/"
              "urn-nasa-pds-magellan_stereo_topography/data")


@dataclass(frozen=True)
class Product:
    """One Magellan raster, local or remote."""

    key: str
    filename: str
    remote: str
    kind: str          # sar | topography | slope | emissivity
    resolution_m: float
    gb: float
    note: str

    def path(self, root: Path) -> Path:
        return root / self.filename

    def uri(self, root: Path, prefer_local: bool = True) -> str:
        """A path rasterio can open: the local file if present, else a `/vsicurl/` URL.

        Falling back to the remote is what makes the 75 m mosaics usable at all — they
        are read by window and never stored.
        """
        p = self.path(root)
        if prefer_local and p.exists():
            return str(p)
        return f"/vsicurl/{self.remote}"


PRODUCTS: dict[str, Product] = {
    "sar_left_75m": Product(
        "sar_left_75m", "Venus_Magellan_LeftLook_mosaic_global_75m_jpeg.tif",
        f"{S3}/Venus_Magellan_LeftLook_mosaic_global_75m_jpeg.tif",
        "sar", 75.0, 16.77,
        "JPEG-in-TIFF: same 75 m as the 117 GB original, 7x smaller. Read by window.",
    ),
    "sar_right_75m": Product(
        "sar_right_75m", "Venus_Magellan_RightLook_mosaic_global_75m_jpeg.tif",
        f"{S3}/Venus_Magellan_RightLook_mosaic_global_75m_jpeg.tif",
        "sar", 75.0, 7.15, "~17% coverage. The free cross-look constraint.",
    ),
    "sar_stereo_75m": Product(
        "sar_stereo_75m", "Venus_Magellan_StereoLook_mosaic_global_75m.tif",
        f"{S3}/Venus_Magellan_StereoLook_mosaic_global_75m.tif",
        "sar", 75.0, 87.86,
        "Cycle 3, different incidence. No JPEG variant exists, so window reads only.",
    ),
    "sar_left_225m": Product(
        "sar_left_225m", "Venus_Magellan_LeftLook_mosaic_global_225m.tif",
        f"{S3}/Venus_Magellan_LeftLook_mosaic_global_225m.tif",
        "sar", 225.0, 13.01, "Whole planet at the shipped product's posting.",
    ),
    "gtdr": Product(
        "gtdr", "Venus_Magellan_Topography_Global_4641m_v02.tif",
        f"{S3}/Venus_Magellan_Topography_Global_4641m_v02.tif",
        "topography", 4641.06, 0.07, "int16 metres, nodata -32768. Footprint ~10 x 20 km.",
    ),
    "gsdr": Product(
        "gsdr", "Venus_Magellan_MeterScaleSlope_Global_4641m.tif",
        f"{S3}/Venus_Magellan_MeterScaleSlope_Global_4641m.tif",
        "slope", 4641.06, 0.03, "Metre-scale RMS slope; the weak roughness target.",
    ),
    "gedr": Product(
        "gedr", "Venus_Magellan_MicrowaveEmissivity_Global_4641m.tif",
        f"{S3}/Venus_Magellan_MicrowaveEmissivity_Global_4641m.tif",
        "emissivity", 4641.06, 0.07, "Optional input channel; highlands read differently.",
    ),
}


# --------------------------------------------------------------------------------------
# Cylindrical sampling
# --------------------------------------------------------------------------------------
def lonlat_to_xy(lon_deg, lat_deg) -> tuple[np.ndarray, np.ndarray]:
    """Cylindrical projected metres on the Venus sphere. Longitudes wrap to -180..180."""
    lon = np.asarray(lon_deg, dtype=np.float64)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon * M_PER_DEG, np.asarray(lat_deg, dtype=np.float64) * M_PER_DEG


@dataclass
class Window:
    """A lon/lat box and the pixel grid it will be sampled onto."""

    west: float
    south: float
    east: float
    north: float
    pixel_size_m: float

    @property
    def width(self) -> int:
        return max(1, int(round((self.east - self.west) * M_PER_DEG / self.pixel_size_m)))

    @property
    def height(self) -> int:
        return max(1, int(round((self.north - self.south) * M_PER_DEG / self.pixel_size_m)))

    def lon_lat_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre longitude and latitude, north-up."""
        lon = self.west + (np.arange(self.width) + 0.5) * (self.east - self.west) / self.width
        lat = self.north - (np.arange(self.height) + 0.5) * (self.north - self.south) / self.height
        return np.meshgrid(lon, lat)

    @classmethod
    def centred(cls, lon: float, lat: float, span_km: float, pixel_size_m: float) -> "Window":
        half = (span_km * 1000.0 / M_PER_DEG) / 2.0
        return cls(lon - half, lat - half, lon + half, lat + half, pixel_size_m)


def sample(uri: str, window: Window, resampling: str = "bilinear",
           nodata_out: float = np.nan) -> tuple[np.ndarray, np.ndarray]:
    """Read `uri` over `window`, returning `(values, valid)` on the window's grid.

    The read is a single windowed request with `out_shape` set, so GDAL decodes only the
    source tiles that intersect and resamples them in one step — the whole reason a 507k
    pixel-wide mosaic is usable without downloading it.

    Pixels outside the product's own extent come back as `nodata_out` and False, which is
    the normal case: the SAR mosaics stop near +/-80 degrees and the stereo DEM covers one
    block of the planet.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    with rasterio.open(uri) as ds:
        x0, y0 = lonlat_to_xy(window.west, window.south)
        x1, y1 = lonlat_to_xy(window.east, window.north)
        # A box spanning the seam would wrap to a negative width; callers cut tiles that
        # do not straddle it, and this makes the violation loud instead of silent.
        if x1 <= x0:
            raise ValueError(
                f"window {window.west}..{window.east} straddles the +/-180 seam; "
                "split it in two before sampling"
            )

        left, bottom, right, top = ds.bounds
        if x1 <= left or x0 >= right or y1 <= bottom or y0 >= top:
            empty = np.full((window.height, window.width), nodata_out, np.float32)
            return empty, np.zeros_like(empty, bool)

        win = from_bounds(x0, y0, x1, y1, ds.transform)
        data = ds.read(
            1, window=win, out_shape=(window.height, window.width),
            resampling=getattr(Resampling, resampling), boundless=True,
            fill_value=ds.nodata if ds.nodata is not None else 0,
        ).astype(np.float32)

        valid = np.isfinite(data)
        if ds.nodata is not None:
            valid &= data != ds.nodata

        # Apply the band's scale and offset. `ds.read` does not: it returns raw storage
        # values, so emissivity would arrive as ~8500 instead of 0.85 and the network
        # would be handed an input channel four orders of magnitude off, with nothing to
        # show for it but a worse model.
        scale = ds.scales[0] if ds.scales else 1.0
        offset = ds.offsets[0] if ds.offsets else 0.0
        if scale != 1.0 or offset != 0.0:
            data = data * scale + offset
        # Anything outside the product's latitude range is nodata regardless.
        _, lat_grid = window.lon_lat_grids()
        y = lat_grid * M_PER_DEG
        valid &= (y > bottom) & (y < top)

        return np.where(valid, data, nodata_out).astype(np.float32), valid


# --------------------------------------------------------------------------------------
# The stereo DEM, which is a raw PDS array rather than a GeoTIFF
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StereoDEM:
    """Herrick et al. (2012) stereo topography, `mosaic_allstereo.img`.

    A headerless little-endian float32 array whose geometry lives in a detached PDS4
    label. Values are planetary *radii* offset by `value_offset`; elevation relative to
    the 6051.8 km datum is `raw + value_offset - VENUS_RADIUS_M`.

    Geometry is read from the label rather than hard-coded, because these numbers are
    exactly the sort that get transcribed wrong once and then quietly govern every
    stereo-supervised pixel.
    """

    path: Path
    width: int
    height: int
    pixel_size_m: float
    upperleft_x: float
    upperleft_y: float
    value_offset: float

    @classmethod
    def from_label(cls, img_path: Path, xml_path: Path | None = None) -> "StereoDEM":
        import re

        xml = xml_path or img_path.with_suffix(".xml")
        s = xml.read_text(errors="ignore")

        def one(tag: str, cast=float):
            m = re.search(rf"<(?:\w+:)?{tag}[^>]*>\s*([^<]+?)\s*</(?:\w+:)?{tag}>", s, re.I)
            if not m:
                raise ValueError(f"{xml.name}: no <{tag}>")
            return cast(m.group(1))

        elements = [int(x) for x in re.findall(r"<elements>\s*(\d+)\s*</elements>", s)]
        if len(elements) < 2:
            raise ValueError(f"{xml.name}: expected two Axis_Array elements")
        return cls(
            path=img_path,
            height=elements[0],           # Line
            width=elements[1],            # Sample
            pixel_size_m=one("pixel_resolution_x"),
            upperleft_x=one("upperleft_corner_x"),
            upperleft_y=one("upperleft_corner_y"),
            value_offset=one("value_offset"),
        )

    @property
    def expected_bytes(self) -> int:
        return self.width * self.height * 4

    def is_complete(self) -> bool:
        """A partial download memmaps to the declared shape and reads zeros past the end,
        which would enter training as a flat sea-level plain rather than as missing data."""
        return self.path.exists() and self.path.stat().st_size >= self.expected_bytes

    @property
    def available_rows(self) -> int:
        """Rows actually present on disk.

        The array is row-major and the rows run north to south, so a partial download is
        a complete northern band rather than a corrupt file. Treating everything past it
        as nodata makes the data usable while it is still arriving, and — more
        importantly — keeps the boundary honest instead of feeding zeros to `L_stereo` as
        if Venus had a sea-level plain there.
        """
        if not self.path.exists():
            return 0
        return min(self.height, self.path.stat().st_size // (self.width * 4))

    def available_south_deg(self) -> float:
        return (self.upperleft_y - self.available_rows * self.pixel_size_m) / M_PER_DEG

    @property
    def bounds_deg(self) -> tuple[float, float, float, float]:
        west = self.upperleft_x / M_PER_DEG
        north = self.upperleft_y / M_PER_DEG
        east = (self.upperleft_x + self.width * self.pixel_size_m) / M_PER_DEG
        south = (self.upperleft_y - self.height * self.pixel_size_m) / M_PER_DEG
        return west, south, east, north

    def read(self, window: Window, nodata_out: float = np.nan) -> tuple[np.ndarray, np.ndarray]:
        """Sample elevations relative to the 6051.8 km sphere onto the window's grid."""
        lon, lat = window.lon_lat_grids()
        x, y = lonlat_to_xy(lon, lat)
        col = (x - self.upperleft_x) / self.pixel_size_m
        row = (self.upperleft_y - y) / self.pixel_size_m

        rows = self.available_rows
        inside = (col >= 0) & (col < self.width - 1) & (row >= 0) & (row < rows - 1)
        out = np.full(lon.shape, nodata_out, np.float32)
        if not inside.any():
            return out, np.zeros(lon.shape, bool)

        arr = np.memmap(self.path, dtype="<f4", mode="r", shape=(rows, self.width))
        c0 = np.clip(np.floor(col).astype(np.int64), 0, self.width - 2)
        r0 = np.clip(np.floor(row).astype(np.int64), 0, max(rows - 2, 0))
        fc, fr = col - c0, row - r0

        p00 = arr[r0, c0]; p01 = arr[r0, c0 + 1]
        p10 = arr[r0 + 1, c0]; p11 = arr[r0 + 1, c0 + 1]
        corners = np.stack([p00, p01, p10, p11])
        # The array marks gaps with a sentinel far below any real radius; a bilinear blend
        # that touched one would smear it into the surrounding terrain.
        good = np.all(np.isfinite(corners) & (corners > -1e6), axis=0) & inside

        top = p00 * (1 - fc) + p01 * fc
        bot = p10 * (1 - fc) + p11 * fc
        radius = top * (1 - fr) + bot * fr
        elev = radius + self.value_offset - VENUS_RADIUS_M

        out = np.where(good, elev, nodata_out).astype(np.float32)
        return out, good
