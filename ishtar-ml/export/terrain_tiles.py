"""Quantized-mesh terrain for a non-WGS84 body.

**Read this before running any terrain tiler.** Cesium Terrain Builder and its forks
hard-code WGS84 when they compute each tile's bounding sphere and horizon occlusion
point. Those fields are culling metadata, not geometry: with wrong values the heights
decode correctly and the mesh is right, but tiles disappear at the limb or refuse to
refine, and the failure looks like a rendering bug rather than a data bug.

Route A is implemented, in `export/quantized_mesh.py`: it encodes tiles on the Venus
sphere, computing the bounding sphere and the horizon occlusion point from the 6051.8 km
radius. `tests/test_quantized_mesh.py` covers the wire format round-trip, the winding,
the vertex ordering the index encoding requires, and the two planet-dependent quantities.
This module remains the planning view — how many tiles, at what resolution, and what
Route B would cost if you fell back to it.

Route B (fallback): keep a WGS84 tiler and pre-scale elevations by
`EARTH_RADIUS / VENUS_RADIUS = 1.0527` so relative relief is preserved on an Earth-sized
globe. The globe is then 5% too large and absolute elevations are wrong by the same
factor. It is a legitimate stopgap **only if the UI says so**.

    python export/terrain_tiles.py --plan
"""

from __future__ import annotations

import argparse
import math

VENUS_RADIUS_M = 6_051_800.0
EARTH_RADIUS_M = 6_371_000.0
ROUTE_B_SCALE = EARTH_RADIUS_M / VENUS_RADIUS_M


def metres_per_vertex(level: int, vertices_per_tile: int = 65) -> float:
    """Ground spacing of quantized-mesh vertices at the equator for a geodetic pyramid."""
    degrees_per_tile = 180.0 / 2**level
    return math.radians(degrees_per_tile) * VENUS_RADIUS_M / (vertices_per_tile - 1)


def tile_count(level: int) -> int:
    """Geodetic pyramid: 2 x 1 at level 0."""
    return 2 ** (2 * level + 1)


def horizon_occlusion_point(centre_ecef: tuple[float, float, float], radius_m: float
                            ) -> tuple[float, float, float]:
    """The occlusion point on a sphere, in the scaled space Cesium expects.

    On a sphere all three ellipsoid radii are equal, so the usual scaling by
    `1 / radii` is a single division and the whole computation collapses to a scale of
    the tile centre direction. This is the piece a WGS84 tiler gets wrong.
    """
    x, y, z = (c / radius_m for c in centre_ecef)
    magnitude = math.sqrt(x * x + y * y + z * z)
    return (x / magnitude, y / magnitude, z / magnitude)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true")
    ap.parse_args()

    print(f"{'level':>5s} {'m/vertex':>10s} {'tiles':>12s}   note")
    for lvl in range(0, 14):
        note = ""
        if lvl == 9:
            note = "global terrain stops here (~580 m/vertex)"
        if lvl == 12:
            note = "ROIs only (~73 m/vertex); global would be ~33M tiles"
        print(f"{lvl:5d} {metres_per_vertex(lvl):10.1f} {tile_count(lvl):12,d}   {note}")
    print(f"\nRoute A is implemented: export/quantized_mesh.py encodes on the Venus sphere.")
    print(f"Route B elevation scale, if you fall back to a WGS84 tiler: {ROUTE_B_SCALE:.4f}")
    print("Label it in the UI. The globe is then 5% too big and heights 5% too large.")


if __name__ == "__main__":
    main()
