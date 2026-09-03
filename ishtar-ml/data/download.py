"""Fetch the Magellan products.

The mosaics are large (109 GB for the left-look FMAP alone), so the default here is the
USGS Map-A-Planet 2 clipping service: pull the six regions of interest first, get Phase 2
metrics, and only mirror the global GeoTIFF when you actually start Phase 3.

    python -m data.download --list
    python -m data.download --product gtdr --out data_raw/
    python -m data.download --product fmap_left --region ovda --out data_raw/

Nothing here is imported by the model or the tests, so `requests` and `rasterio` are
optional dependencies of this module alone.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

VENUS_RADIUS_M = 6_051_800.0


@dataclass(frozen=True)
class Product:
    key: str
    name: str
    url: str
    approx_gb: float
    resolution_m: float
    notes: str


# URLs are the USGS Astrogeology / PDS landing points. They are deliberately not
# hard-coded S3 keys: USGS re-lays-out its bucket periodically, and a stale key fails
# silently as a 403 halfway through a 100 GB transfer.
PRODUCTS: dict[str, Product] = {
    "fmap_left": Product(
        "fmap_left", "Magellan FMAP left-look global SAR mosaic",
        "https://astrogeology.usgs.gov/search/map/venus_magellan_c3_mdir_left_look_global_mosaic_75m",
        109.0, 75.0, "Global (~92-96%). 8-bit DN of Muhleman-flattened backscatter.",
    ),
    "fmap_right": Product(
        "fmap_right", "Magellan FMAP right-look mosaic",
        "https://astrogeology.usgs.gov/search/map/venus_magellan_c3_mdir_right_look_global_mosaic_75m",
        110.0, 75.0, "~17% coverage. Opposite look: the free cross-look constraint.",
    ),
    "fmap_stereo": Product(
        "fmap_stereo", "Magellan FMAP stereo-look mosaic (Cycle 3)",
        "https://astrogeology.usgs.gov/search/map/venus_magellan_c3_mdir_stereo_look_global_mosaic_75m",
        82.0, 75.0, "~17% coverage at a deliberately different incidence angle.",
    ),
    "gtdr": Product(
        "gtdr", "Magellan global topography (GTDR) v02",
        "https://astrogeology.usgs.gov/search/map/venus_magellan_global_topography_4641m",
        0.4, 4641.0, "int16 metres, -2951..11687, nodata -32768. Footprint ~10 x 20 km.",
    ),
    "gsdr": Product(
        "gsdr", "Magellan RMS slope (GSDR)",
        "https://pds-geosciences.wustl.edu/missions/magellan/gxdr/", 0.4, 4641.0,
        "RMS slope at 4.6 km; the weak roughness target for L_rms.",
    ),
    "gedr": Product(
        "gedr", "Magellan emissivity (GEDR)",
        "https://pds-geosciences.wustl.edu/missions/magellan/gxdr/", 0.4, 4641.0,
        "Emissivity; optional input channel for dielectric/roughness context.",
    ),
    "stereo_dem": Product(
        "stereo_dem", "Herrick et al. (2012) stereo-derived DEM",
        "https://www.gi.alaska.edu/~rherrick/venus.html", 2.0, 1500.0,
        "~20% coverage, 50-100 m vertical. Has seam and radar-dark artefacts: mask them.",
    ),
    "gazetteer": Product(
        "gazetteer", "IAU planetary nomenclature, Venus",
        "https://planetarynames.wr.usgs.gov/", 0.01, 0.0,
        "Feature names and centres for the globe's fly-to search.",
    ),
}

# Section 2.4: spatial splits. Ovda and the plains/crater quads are held out; Maxwell and
# Maat are demo regions you will stare at constantly, so they must not be test data.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    # name: (west, south, east, north) in degrees east / planetocentric latitude
    "ovda": (75.0, -10.0, 100.0, 5.0),          # tessera, held out
    "alpha": (0.0, -30.0, 15.0, -18.0),         # tessera + pancake domes
    "maxwell": (0.0, 60.0, 20.0, 70.0),         # demo: highest relief on the planet
    "maat": (190.0, 5.0, 200.0, 15.0),          # demo: large shield volcano
    "artemis": (125.0, -45.0, 150.0, -25.0),    # corona rim
    "mead": (52.0, 8.0, 62.0, 17.0),            # large impact crater
    "guinevere": (330.0, 15.0, 350.0, 30.0),    # smooth plains, low relief
    "lakshmi": (330.0, 55.0, 20.0, 70.0),       # plateau, crosses the prime meridian
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def map_a_planet_url(product: str, region: str) -> str:
    """Clip request for USGS Map-A-Planet 2.

    Returns the parameterised endpoint rather than firing it: the service is
    asynchronous (it emails or polls for a job), so the caller decides how to wait.
    """
    w, s, e, n = REGIONS[region]
    return (
        "https://astrocloud.wr.usgs.gov/index.php?view=map"
        f"&product={PRODUCTS[product].key}&minlon={w}&minlat={s}&maxlon={e}&maxlat={n}"
        "&format=GeoTIFF"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--product", choices=sorted(PRODUCTS))
    ap.add_argument("--region", choices=sorted(REGIONS))
    ap.add_argument("--out", type=Path, default=Path("data_raw"))
    a = ap.parse_args()

    if a.list or not a.product:
        total = sum(p.approx_gb for p in PRODUCTS.values())
        print(f"{'key':12s} {'GB':>6s} {'res m':>7s}  source")
        for p in PRODUCTS.values():
            print(f"{p.key:12s} {p.approx_gb:6.1f} {p.resolution_m:7.0f}  {p.url}\n{'':28s}{p.notes}")
        print(f"\ntotal if you mirror everything: ~{total:.0f} GB")
        print(f"regions available for clipping: {', '.join(sorted(REGIONS))}")
        return

    p = PRODUCTS[a.product]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.region:
        print(f"Clip request for {p.name} over {a.region}:\n  {map_a_planet_url(a.product, a.region)}")
    else:
        print(f"{p.name}: ~{p.approx_gb:.0f} GB\n  {p.url}")
    print(f"\nDownload to {a.out.resolve()}, then verify with data/download.py's sha256() and\n"
          f"run data/tile.py to build the Zarr store.")


if __name__ == "__main__":
    main()
