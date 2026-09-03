"""Zarr -> Cloud-Optimised GeoTIFF, hillshade, colour relief, and tile pyramids.

The commands are here as a reference implementation because the projection arguments are
where a Venus product goes wrong silently: GDAL will happily write a Venus raster with an
Earth datum, and everything downstream looks fine until distances are 5% off.

    python export/to_cog.py --print-commands
"""

from __future__ import annotations

import argparse

VENUS_RADIUS_M = 6_051_800.0

# A sphere, not an ellipsoid: equal semi-major and semi-minor axes, zero flattening.
VENUS_2000_WKT = (
    'GEOGCS["Venus 2000",'
    'DATUM["D_Venus_2000",SPHEROID["Venus_2000_IAU_IAG",6051800.0,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)

COMMANDS: list[tuple[str, str]] = [
    (
        "global elevation COG at 225 m",
        "gdal_translate -of COG -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BLOCKSIZE=512 "
        "-a_srs venus2000.wkt -ot Int16 -a_nodata -32768 "
        "outputs/venus_dem_v1_225m.tif outputs/venus_dem_v1.tif",
    ),
    (
        "uncertainty COG",
        "gdal_translate -of COG -co COMPRESS=DEFLATE -a_srs venus2000.wkt -ot Int16 "
        "outputs/venus_sigma_225m.tif outputs/venus_dem_v1_sigma.tif",
    ),
    (
        "hillshade at 450 m",
        "gdaldem hillshade -z 2 -az 315 -alt 45 -compute_edges "
        "outputs/venus_dem_v1.tif outputs/hillshade_450m.tif",
    ),
    (
        "colour relief",
        "gdaldem color-relief outputs/venus_dem_v1.tif export/venus_ramp.txt "
        "outputs/colour_relief.tif",
    ),
    (
        "imagery tiles (geodetic, NOT web mercator)",
        "gdal2tiles.py --profile=geodetic --zoom=0-9 --xyz --resampling=average "
        "outputs/colour_relief.tif ishtar-globe/public/tiles/colour_relief",
    ),
    (
        "ROI imagery to level 12",
        "gdal2tiles.py --profile=geodetic --zoom=10-12 --xyz "
        "outputs/roi_maxwell_75m.tif ishtar-globe/public/tiles/sar_left",
    ),
]

# Hypsometric ramp in metres relative to the 6051.8 km sphere. Venus spans roughly
# -3 km (Diana Chasma) to +11 km (Maxwell Montes).
COLOUR_RAMP = """\
-3000  20  30  60
-1000  40  80 110
    0  90 120 100
 1000 150 140  90
 3000 190 150  80
 6000 220 180 120
11000 250 245 235
   nv   0   0   0   0
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print-commands", action="store_true")
    ap.add_argument("--write-aux", action="store_true", help="write venus2000.wkt and venus_ramp.txt")
    a = ap.parse_args()

    if a.write_aux:
        from pathlib import Path

        Path("venus2000.wkt").write_text(VENUS_2000_WKT + "\n")
        Path("export/venus_ramp.txt").write_text(COLOUR_RAMP)
        print("wrote venus2000.wkt and export/venus_ramp.txt")
        return

    print(f"Venus 2000 SRS (sphere, R = {VENUS_RADIUS_M:.1f} m):\n  {VENUS_2000_WKT}\n")
    for title, cmd in COMMANDS:
        print(f"# {title}\n{cmd}\n")
    print("# Terrain tiles: see export/terrain_tiles.py -- do NOT use a WGS84 tiler "
          "without reading the note there first.")


if __name__ == "__main__":
    main()
