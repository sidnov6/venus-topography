"""The Venus terrain tiler.

The reason this exists rather than a WGS84 tiler is that bounding spheres and horizon
occlusion points are computed on the wrong body by every off-the-shelf option, so those
two are tested hardest.
"""

import numpy as np
import pytest

from export.quantized_mesh import (
    QUANTIZED_MAX,
    reorder_for_encoding,
    VENUS_RADIUS_M,
    TileBounds,
    bounding_sphere,
    build_tile,
    decode_indices,
    decode_tile,
    dequantize_heights,
    encode_indices,
    encode_tile,
    geodetic_to_ecef,
    horizon_occlusion_point,
    layer_json,
    zigzag_decode,
    zigzag_encode,
)


def terrain(n=33, relief=800.0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    return (relief * (np.sin(3 * x) * np.cos(2 * y) + 0.1 * rng.normal(size=(n, n))))


# --- primitives -----------------------------------------------------------------------
def test_zigzag_round_trips_over_the_signed_range():
    v = np.array([-32768, -1, 0, 1, 32767], np.int32)
    assert np.array_equal(zigzag_decode(zigzag_encode(v)), v)


def test_index_high_water_mark_round_trips():
    idx = np.array([0, 1, 2, 1, 3, 2, 0, 4, 5], np.int64)
    assert np.array_equal(decode_indices(encode_indices(idx)), idx)


def test_index_encoding_stays_small():
    """The point of high-water-mark coding: values stay near zero, so 16 bits suffice
    even for a mesh whose absolute indices run into the tens of thousands."""
    idx = np.repeat(np.arange(20000), 3)
    assert encode_indices(idx).max() < 65536


# --- geometry -------------------------------------------------------------------------
def test_geodetic_to_ecef_puts_the_surface_at_the_venus_radius():
    p = geodetic_to_ecef(np.array([0.0, 90.0, 180.0]), np.array([0.0, 45.0, -60.0]), np.zeros(3))
    assert np.allclose(np.linalg.norm(p, axis=1), VENUS_RADIUS_M)


def test_geodetic_to_ecef_is_not_wgs84():
    """The regression guard: an Earth ellipsoid would put this point ~319 km further out
    and 21 km off the sphere between pole and equator."""
    p = geodetic_to_ecef(0.0, 45.0, 0.0)
    assert abs(float(np.linalg.norm(p)) - VENUS_RADIUS_M) < 1.0
    assert abs(float(np.linalg.norm(p)) - 6_371_000.0) > 300_000.0


def test_bounding_sphere_contains_every_point():
    rng = np.random.default_rng(0)
    for seed in range(5):
        pts = rng.normal(size=(400, 3)) * 1000 + np.array([VENUS_RADIUS_M, 0, 0])
        centre, radius = bounding_sphere(pts)
        assert np.max(np.linalg.norm(pts - centre, axis=1)) <= radius + 1e-6


def test_bounding_sphere_is_not_wildly_loose():
    pts = geodetic_to_ecef(
        *np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20)), np.zeros((20, 20))
    ).reshape(-1, 3)
    _, radius = bounding_sphere(pts)
    extent = np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))
    assert radius < 1.6 * extent


def test_horizon_occlusion_point_is_along_the_tile_direction():
    tile = build_tile(terrain(), TileBounds(0.0, 0.0, 2.0, 2.0))
    d = tile.centre_ecef / np.linalg.norm(tile.centre_ecef)
    o = tile.occlusion_point
    assert np.allclose(o / np.linalg.norm(o), d, atol=1e-6)


def test_horizon_occlusion_point_is_scaled_by_the_venus_radius():
    """It lives in ellipsoid-scaled space, so its magnitude is order 1, not order 6e6.
    A WGS84 tiler scales by Earth's radii and lands ~5% off — enough to cull real tiles."""
    tile = build_tile(terrain(), TileBounds(30.0, 10.0, 32.0, 12.0))
    mag = float(np.linalg.norm(tile.occlusion_point))
    assert 0.9 < mag < 1.2


def test_high_terrain_pushes_the_occlusion_point_further_out():
    flat = build_tile(np.zeros((33, 33)), TileBounds(0.0, 0.0, 2.0, 2.0))
    tall = build_tile(np.full((33, 33), 11_000.0), TileBounds(0.0, 0.0, 2.0, 2.0))
    assert np.linalg.norm(tall.occlusion_point) > np.linalg.norm(flat.occlusion_point)


# --- tile ------------------------------------------------------------------------------
def test_tile_round_trips_through_the_wire_format():
    heights = terrain(n=17)
    tile = build_tile(heights, TileBounds(10.0, -5.0, 12.0, -3.0))
    got = decode_tile(encode_tile(tile))

    assert got["vertex_count"] == 17 * 17
    assert np.array_equal(got["u"], tile.u)
    assert np.array_equal(got["v"], tile.v)
    assert np.array_equal(got["h"], tile.h)
    assert np.array_equal(got["indices"], tile.indices)
    assert got["min_height"] == pytest.approx(heights.min(), rel=1e-5)
    assert got["max_height"] == pytest.approx(heights.max(), rel=1e-5)


def test_decoded_heights_match_within_the_quantisation_step():
    heights = terrain(n=17, relief=11_000.0)
    tile = build_tile(heights, TileBounds(0.0, 0.0, 1.0, 1.0))
    got = decode_tile(encode_tile(tile))
    back = dequantize_heights(got["h"], got["min_height"], got["max_height"])
    # Vertices are stored in first-use order; grid_order maps them back to the raster.
    regrid = np.empty(17 * 17)
    regrid[tile.grid_order] = back
    step = (heights.max() - heights.min()) / QUANTIZED_MAX
    assert np.max(np.abs(regrid.reshape(17, 17) - heights[::-1])) <= step


def test_height_quantisation_is_finer_than_the_model_uncertainty():
    """Heights are 16-bit *within each tile's own range*, so the step is set by local
    relief. Even a tile spanning Venus's entire -3 km to +11 km range quantises to 0.43 m,
    two orders of magnitude below the model's own sigma — the format is never the limit.
    """
    whole_planet = np.linspace(-3000.0, 11_000.0, 9 * 9).reshape(9, 9)
    tile = build_tile(whole_planet, TileBounds(0.0, 0.0, 1.0, 1.0))
    step = (tile.max_height - tile.min_height) / QUANTIZED_MAX
    assert step < 0.5

    flat_plain = np.linspace(0.0, 40.0, 9 * 9).reshape(9, 9)
    tile2 = build_tile(flat_plain, TileBounds(0.0, 0.0, 1.0, 1.0))
    assert (tile2.max_height - tile2.min_height) / QUANTIZED_MAX < 0.002


def test_v_increases_northward():
    """The rasters in this repo are north-up; quantized-mesh v is south-up. The flip
    happens exactly once, in build_tile."""
    heights = np.zeros((5, 5))
    heights[0] = 1000.0  # north edge of a north-up raster
    tile = build_tile(heights, TileBounds(0.0, 0.0, 1.0, 1.0))
    assert tile.h[tile.v == QUANTIZED_MAX].max() == QUANTIZED_MAX
    assert tile.h[tile.v == 0].max() == 0


def test_edge_indices_lie_on_their_edges():
    n = 9
    tile = build_tile(terrain(n=n), TileBounds(0.0, 0.0, 1.0, 1.0))
    assert np.all(tile.u[tile.west_indices] == 0)
    assert np.all(tile.u[tile.east_indices] == QUANTIZED_MAX)
    assert np.all(tile.v[tile.south_indices] == 0)
    assert np.all(tile.v[tile.north_indices] == QUANTIZED_MAX)
    assert tile.west_indices.size == n


def test_every_vertex_is_used_by_a_triangle():
    n = 9
    tile = build_tile(terrain(n=n), TileBounds(0.0, 0.0, 1.0, 1.0))
    assert tile.indices.size == (n - 1) ** 2 * 6
    assert set(np.unique(tile.indices)) == set(range(n * n))


def test_triangles_wind_counter_clockwise_seen_from_outside():
    n = 5
    bounds = TileBounds(0.0, 0.0, 1.0, 1.0)
    heights = np.zeros((n, n))
    tile = build_tile(heights, bounds)
    lon = np.tile(np.linspace(bounds.west, bounds.east, n), n)[tile.grid_order]
    lat = np.repeat(np.linspace(bounds.south, bounds.north, n), n)[tile.grid_order]
    p = geodetic_to_ecef(lon, lat, np.zeros(n * n))
    tri = tile.indices.reshape(-1, 3)
    normals = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    outward = np.sum(normals * p[tri[:, 0]], axis=1)
    assert np.all(outward > 0), "triangles face inward: the terrain will render black"


def test_layer_json_declares_the_geodetic_scheme():
    lj = layer_json(9)
    assert lj["format"] == "quantized-mesh-1.0"
    assert lj["projection"] == "EPSG:4326"  # geodetic 2x1, not web mercator
    assert lj["bounds"] == [-180, -90, 180, 90]
    assert len(lj["available"]) == 10
    assert lj["available"][0][0]["endX"] == 1 and lj["available"][0][0]["endY"] == 0


def test_reorder_makes_first_use_order_increasing():
    idx = np.array([0, 17, 1, 1, 17, 18], np.int64)
    perm, remapped = reorder_for_encoding(idx, 19)
    seen: set[int] = set()
    highest = -1
    for v in remapped:
        if v not in seen:
            assert v == highest + 1, "a new vertex must be the next id"
            highest = v
            seen.add(int(v))
    assert perm.size == 19


def test_encode_indices_refuses_out_of_order_input():
    """The silent-corruption guard: unordered indices would wrap in the unsigned buffer."""
    with pytest.raises(ValueError, match="first-use order"):
        encode_indices(np.array([0, 5, 1], np.int64))


def test_sample_dem_returns_none_outside_the_product():
    """A regional product makes a sparse pyramid; a missing tile is normal, not an error."""
    from export.quantized_mesh import sample_dem

    dem = np.zeros((16, 16))
    bounds = TileBounds(0.0, 0.0, 10.0, 10.0)
    assert sample_dem(dem, bounds, TileBounds(2.0, 2.0, 4.0, 4.0)) is not None
    assert sample_dem(dem, bounds, TileBounds(50.0, 50.0, 60.0, 60.0)) is None


def test_sample_dem_preserves_north_up_orientation():
    from export.quantized_mesh import sample_dem

    dem = np.zeros((32, 32))
    dem[0] = 1000.0  # north edge
    bounds = TileBounds(0.0, 0.0, 10.0, 10.0)
    out = sample_dem(dem, bounds, bounds, vertices=9)
    assert out[0].max() > 900.0, "row 0 of the sampled grid must still be the north edge"
    assert out[-1].max() == pytest.approx(0.0)


def test_tile_extent_covers_the_planet_at_every_level():
    from export.quantized_mesh import tile_extent

    for level in range(4):
        west = tile_extent(level, 0, 0).west
        east = tile_extent(level, 2 ** (level + 1) - 1, 0).east
        south = tile_extent(level, 0, 0).south
        north = tile_extent(level, 0, 2**level - 1).north
        assert (west, east, south, north) == (-180.0, 180.0, -90.0, 90.0)


def test_build_pyramid_writes_a_loadable_tree(tmp_path):
    from export.quantized_mesh import build_pyramid
    import gzip
    import json

    dem = terrain(n=64, relief=1500.0)
    counts = build_pyramid(dem, TileBounds(-10.0, -10.0, 10.0, 10.0), tmp_path, max_level=3)

    assert counts["0"] == 2, "both level-0 tiles overlap a 20-degree product"
    assert counts["3"] > counts["0"]
    assert (tmp_path / "layer.json").exists()
    lj = json.loads((tmp_path / "layer.json").read_text())
    assert lj["projection"] == "EPSG:4326"

    some = next(tmp_path.glob("3/*/*.terrain"))
    got = decode_tile(gzip.decompress(some.read_bytes()))
    assert got["vertex_count"] == 65 * 65
    assert np.isfinite(got["sphere_radius"]) and got["sphere_radius"] > 0


def test_roi_pyramid_writes_only_the_region(tmp_path):
    """Global to one level, deeper only inside a box — how the real product ships. Level 12
    globally would be 33 million tiles; the Maxwell box is 105 thousand."""
    from export.quantized_mesh import build_pyramid, tiles_in

    dem = terrain(n=64, relief=1500.0)
    world = TileBounds(-180, -90, 180, 90)
    box = TileBounds(-6.0, 60.0, 14.0, 70.0)

    counts = build_pyramid(dem, world, tmp_path, min_level=5, max_level=6, box=box)
    assert 0 < counts["5"] < 2 ** (2 * 5 + 1)
    assert 0 < counts["6"] < 2 ** (2 * 6 + 1)
    assert counts["6"] > counts["5"]

    # Everything written must intersect the box.
    for path in tmp_path.glob("6/*/*.terrain"):
        x, y = int(path.parent.name), int(path.stem)
        span = 180.0 / 2**6
        west, south = -180.0 + x * span, -90.0 + y * span
        assert west < box.east and west + span > box.west
        assert south < box.north and south + span > box.south


def test_tiles_in_covers_the_box_and_nothing_far_from_it():
    from export.quantized_mesh import tiles_in, tile_extent

    box = TileBounds(52.0, 8.0, 62.0, 17.0)  # Mead crater
    for level in (3, 6, 9):
        got = list(tiles_in(level, box))
        assert got, f"level {level} produced no tiles"
        span = 180.0 / 2**level
        for x, y in got:
            b = tile_extent(level, x, y)
            assert b.west <= box.east + span and b.east >= box.west - span
        # And it is a small fraction of the level, which is the point.
        assert len(got) < 2 ** (2 * level + 1)


def test_tiles_in_without_a_box_is_the_whole_level():
    from export.quantized_mesh import tiles_in

    assert len(list(tiles_in(3, None))) == 2 ** (2 * 3 + 1)
