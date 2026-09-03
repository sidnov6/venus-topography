"""Polar re-tiling. The caps are 4% of the planet's area but 100% of the places where a
cylindrical grid stops meaning anything."""

import numpy as np
import pytest

from data.polar import (
    PolarGrid,
    blend_weight,
    cap_grid,
    cylindrical_to_polar,
    polar_to_cylindrical,
)


def smooth_global(h=180, w=360):
    """A global field with structure at every longitude, so a wrap bug is visible."""
    lat = np.linspace(90, -90, h)[:, None]
    lon = np.linspace(0, 360, w, endpoint=False)[None, :]
    return (1000 * np.sin(np.radians(lat)) + 400 * np.cos(np.radians(3 * lon))
            * np.cos(np.radians(lat))).astype(np.float32)


def test_grid_centre_is_the_pole():
    for north in (True, False):
        g = PolarGrid(size=65, pixel_size_m=20_000.0, north=north)
        lon, lat = g.lon_lat()
        assert abs(float(lat[32, 32])) == pytest.approx(90.0, abs=0.05)


def test_latitude_falls_away_monotonically_from_the_pole():
    g = PolarGrid(size=101, pixel_size_m=20_000.0)
    _, lat = g.lon_lat()
    row = lat[50, 50:]
    assert np.all(np.diff(row) < 0)


def test_forward_and_inverse_projection_agree():
    g = PolarGrid(size=257, pixel_size_m=10_000.0)
    lon, lat = g.lon_lat()
    col, row = g.from_lon_lat(lon, lat)
    ii, jj = np.meshgrid(np.arange(257), np.arange(257), indexing="xy")
    assert np.max(np.abs(col - ii)) < 1e-6
    assert np.max(np.abs(row - jj)) < 1e-6


@pytest.mark.parametrize("north", [True, False])
def test_both_poles_use_the_same_code_path(north):
    g = PolarGrid(size=65, pixel_size_m=20_000.0, north=north)
    _, lat = g.lon_lat()
    assert np.sign(np.nanmean(lat)) == (1 if north else -1)
    assert abs(g.min_latitude_deg()) < 90.0


def test_stereographic_scale_is_one_at_the_pole_and_bounded_over_the_cap():
    """Conformality is the reason for this projection: shapes stay locally right, and the
    scale error over a cap reaching 75 degrees stays inside the model's own uncertainty."""
    g = cap_grid(4641.0, 75.0)
    k = g.scale_factor()
    assert k.min() == pytest.approx(1.0, abs=1e-4)
    assert k.max() < 1.08


def test_resampling_round_trips_a_smooth_field():
    src = smooth_global()
    g = PolarGrid(size=192, pixel_size_m=12_000.0)
    polar = cylindrical_to_polar(src, g)
    back = polar_to_cylindrical(polar, g, src.shape)

    covered = np.isfinite(back)
    lat = np.linspace(90, -90, src.shape[0])[:, None] * np.ones((1, src.shape[1]))
    deep = covered & (np.abs(lat) > 82.0)
    assert deep.sum() > 100
    assert np.max(np.abs(back[deep] - src[deep])) < 0.02 * float(np.ptp(src))


def test_polar_resampling_wraps_in_longitude():
    """A polar tile straddles every meridian. Forgetting the wrap leaves a wedge of nodata
    that reads as missing data rather than as an indexing bug."""
    src = smooth_global()
    g = PolarGrid(size=128, pixel_size_m=12_000.0)
    polar = cylindrical_to_polar(src, g)
    lon, lat = g.lon_lat()
    inside = np.abs(lat) > 80.0
    assert np.isfinite(polar[inside]).all()
    # And the values really do vary with longitude, i.e. we are not sampling one column.
    assert float(np.nanstd(polar[inside])) > 1.0


def test_cap_outside_the_grid_is_nodata_not_zero():
    src = smooth_global()
    g = PolarGrid(size=64, pixel_size_m=12_000.0)
    back = polar_to_cylindrical(cylindrical_to_polar(src, g), g, src.shape)
    assert np.isnan(back[90]).all(), "the equator is nowhere near the north cap"


def test_blend_weight_is_a_smooth_partition_with_the_cylindrical_inference():
    lat = np.linspace(60, 90, 301)
    w = blend_weight(lat, inner_deg=80.0, outer_deg=75.0)
    assert w[lat <= 75].max() == 0.0
    assert w[lat >= 80].min() == 1.0
    assert np.all(np.diff(w) >= -1e-9)
    assert 0.4 < float(w[np.argmin(np.abs(lat - 77.5))]) < 0.6


def test_cap_grid_reaches_the_requested_latitude():
    for px in (225.0, 4641.0):
        g = cap_grid(px, 75.0)
        assert g.min_latitude_deg() == pytest.approx(75.0, abs=0.1)


def test_native_resolution_cap_is_too_large_to_hold_in_memory():
    """Documented in `cap_grid`: the grid is an extent, not an array. If this ever
    shrinks below a gigapixel, someone has changed the geometry."""
    g = cap_grid(75.0, 75.0)
    assert g.megapixels > 1000
    assert g.size * g.pixel_size_m / 1000 == pytest.approx(3187, rel=0.02)
