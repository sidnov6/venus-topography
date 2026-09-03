"""The demo tile generator. It is not part of the science, but it is what makes the globe
verifiable before any download, so its conventions have to match the real pipeline's."""

import json

import numpy as np
import pytest

from export.demo_tiles import (
    SITES,
    colour_relief,
    emissivity_like,
    hillshade,
    planet_dem,
    sar_like,
    stereo_coverage_like,
    uncertainty_like,
    write_graticule_pyramid,
    write_imagery_pyramid,
)
from export.quantized_mesh import TileBounds, build_pyramid


def test_planet_dem_has_venus_like_hypsometry():
    dem = planet_dem(128, 256)
    assert dem.shape == (128, 256)
    # Venus spans roughly -3 km to +11 km, with most of the surface near the mean.
    assert -4000 < dem.min() < 0
    assert 500 < dem.max() < 12_000
    assert float(np.percentile(np.abs(dem - np.median(dem)), 80)) < 2000


@pytest.mark.parametrize("fn,channels", [
    (colour_relief, 3), (emissivity_like, 3),
    (uncertainty_like, 4), (stereo_coverage_like, 4),
])
def test_layer_images_have_the_right_shape_and_alpha(fn, channels):
    dem = planet_dem(64, 128)
    img = fn(dem)
    assert img.shape == (64, 128, channels)
    assert img.dtype == np.uint8
    if channels == 4:
        assert 0 < img[..., 3].mean() < 255, "an overlay that is fully opaque or fully clear is useless"


def test_hillshade_and_sar_are_single_channel_uint8():
    dem = planet_dem(64, 128)
    for img in (hillshade(dem), sar_like(dem)):
        assert img.shape == (64, 128) and img.dtype == np.uint8


def test_sites_are_inside_the_valid_coordinate_ranges():
    for name, lon, lat in SITES:
        assert 0.0 <= lon < 360.0, name
        assert -90.0 < lat < 90.0, name
    assert any(n == "Mead crater" for n, _, _ in SITES), "the alignment check needs Mead"


def test_imagery_pyramid_is_geodetic_two_by_one(tmp_path):
    img = np.zeros((32, 64, 3), np.uint8)
    counts = write_imagery_pyramid(img, tmp_path, max_level=2, tile_size=32)
    assert counts == {"0": 2, "1": 8, "2": 32}
    for level in range(3):
        assert (tmp_path / str(level) / "0" / "0.png").exists()


def test_imagery_pyramid_is_written_in_tms_row_order(tmp_path):
    """The layer URLs use `{reverseY}`, which is Cesium's TMS ordering. Writing XYZ rows
    instead flips the planet north for south, and every feature is still plausibly
    somewhere."""
    from PIL import Image

    img = np.zeros((64, 128, 3), np.uint8)
    img[:32] = 255  # north half white
    write_imagery_pyramid(img, tmp_path, max_level=1, tile_size=32)
    north = np.asarray(Image.open(tmp_path / "1" / "0" / "1.png"))
    south = np.asarray(Image.open(tmp_path / "1" / "0" / "0.png"))
    assert north.mean() > 200, "TMS y=1 must be the northern tile"
    assert south.mean() < 55


def test_graticule_marks_the_named_sites(tmp_path):
    from PIL import Image

    write_graticule_pyramid(tmp_path, max_level=0, tile_size=128)
    left = np.asarray(Image.open(tmp_path / "0" / "0" / "0.png"))
    right = np.asarray(Image.open(tmp_path / "0" / "1" / "0.png"))
    marked = np.concatenate([left.reshape(-1, 4), right.reshape(-1, 4)])
    red = (marked[:, 0] > 200) & (marked[:, 1] < 140) & (marked[:, 3] > 200)
    assert red.sum() >= len(SITES) * 8, "every site should leave a visible marker"


def test_demo_pyramid_and_manifest_are_consistent(tmp_path):
    dem = planet_dem(64, 128)
    counts = build_pyramid(dem, TileBounds(-180, -90, 180, 90), tmp_path / "terrain",
                           max_level=2, vertices=17)
    assert counts == {"0": 2, "1": 8, "2": 32}
    manifest = {"synthetic": True, "maxLevel": {"sar_left": 2}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert json.loads((tmp_path / "manifest.json").read_text())["synthetic"] is True
