"""Grid arithmetic. Every pixel-to-degrees conversion in the pipeline goes through
`MosaicGrid`, so an error here is an error everywhere at once."""

import numpy as np
import pytest

from data.tile import GTDR_STRIDE_PX, MosaicGrid, TileSpec, decode_fmap, decode_gtdr, iter_windows, quad_id

# The full-resolution FMAP mosaic: 75 m at the equator on a 6051.8 km sphere.
FMAP = MosaicGrid(width=506_880, height=253_440)


def test_grid_resolution_is_75_m_at_the_equator():
    assert FMAP.pixel_size_m(0.0) == pytest.approx(75.0, rel=0.01)
    assert FMAP.pixel_size_m(60.0) == pytest.approx(37.5, rel=0.02)


def test_row_col_round_trips_through_lon_lat():
    for lon, lat in ((0.1, 0.0), (57.2, 12.5), (194.6, 0.5), (359.9, -60.0), (3.3, 65.2)):
        r, c = FMAP.row_col(lon, lat)
        got_lon, got_lat = FMAP.lon_lat(r, c)
        assert got_lon == pytest.approx(lon, abs=1e-3)
        assert got_lat == pytest.approx(lat, abs=1e-3)


def test_longitude_wraps_and_latitude_does_not():
    lon, _ = FMAP.lon_lat(0, FMAP.width - 1)
    assert 359.0 < lon < 360.0
    _, lat_top = FMAP.lon_lat(0, 0)
    _, lat_bot = FMAP.lon_lat(FMAP.height - 1, 0)
    assert lat_top == pytest.approx(90.0, abs=1e-3)
    assert lat_bot == pytest.approx(-90.0, abs=1e-3)


def test_gtdr_stride_matches_the_product_resolution():
    assert GTDR_STRIDE_PX == 62
    assert 4641.0 / 75.0 == pytest.approx(GTDR_STRIDE_PX, abs=0.5)


def test_tile_spec_margin_covers_three_sigma_of_the_footprint():
    """The margin exists so the altimeter footprint convolution sees terrain, not padding."""
    spec = TileSpec()
    assert spec.margin_px * spec.pixel_size_m >= 3.0 * 8000.0 * 0.95
    assert spec.full_px == spec.core_px + 2 * spec.margin_px
    assert spec.core_slice == slice(spec.margin_px, spec.margin_px + spec.core_px)


def test_iter_windows_cores_tile_the_region_once():
    grid = MosaicGrid(width=4096, height=2048)
    spec = TileSpec(core_px=64, margin_px=32, max_abs_lat_deg=80.0)
    windows = list(iter_windows(grid, spec, (10.0, -5.0, 20.0, 5.0)))
    assert windows, "a 10 x 10 degree box must contain tiles"
    origins = {(r, c) for r, c, _, _ in windows}
    assert len(origins) == len(windows), "no duplicate windows"
    for _, _, lon, lat in windows:
        assert abs(lat) <= spec.max_abs_lat_deg


def test_iter_windows_skips_the_poles():
    grid = MosaicGrid(width=4096, height=2048)
    spec = TileSpec(core_px=64, margin_px=32, max_abs_lat_deg=80.0)
    assert list(iter_windows(grid, spec, (0.0, 84.0, 20.0, 89.0))) == []


def test_quad_id_is_stable_and_12_degrees_wide():
    # Quad edges fall at lat = -90 + 12k, so the band containing the equator is -6..+6.
    assert quad_id(0.0, 0.0) == quad_id(5.9, 11.9)
    assert quad_id(0.0, 0.0) != quad_id(6.1, 0.0)
    assert quad_id(0.0, 0.0) != quad_id(0.0, 12.1)
    # Longitude wraps, so a negative longitude lands in the same quad as its 0..360 form.
    assert quad_id(0.0, 359.0) == quad_id(0.0, -1.0)


def test_numpy_decoders_match_the_documented_encoding():
    dn = np.array([[0, 1, 101, 251]], np.uint8)
    rv, valid = decode_fmap(dn)
    assert not valid[0, 0] and valid[0, 1:].all()
    assert rv[0, 1] == pytest.approx(-20.0)
    assert rv[0, 2] == pytest.approx(0.0)
    assert rv[0, 3] == pytest.approx(30.0)

    raw = np.array([[-32768, -2951, 0, 11687]], np.int16)
    z, ok = decode_gtdr(raw)
    assert not ok[0, 0] and ok[0, 1:].all()
    assert z[0, 3] == pytest.approx(11687.0)


def test_build_tile_produces_exactly_the_dataset_contract():
    """Exercise the real ingest path with in-memory rasters.

    `build_tile` is what turns Magellan products into the tile layout everything else
    consumes, and its output has to match `data.synthetic` key for key — otherwise the
    first real training run discovers the mismatch after a 300 GB download.
    """
    import numpy as np
    import torch
    from torch.utils.data import default_collate

    from data.dataset import BatchSpec, build_batch
    from data.synthetic import SyntheticConfig, SyntheticVenus
    from data.tile import build_tile

    n = 128
    rng = np.random.default_rng(0)
    sources = {
        "fmap_left": rng.integers(1, 252, (n, n)).astype(np.uint8),
        "fmap_right": rng.integers(1, 252, (n, n)).astype(np.uint8),
        "gtdr": rng.integers(-500, 500, (n, n)).astype(np.int16),
        "gedr": np.full((n, n), 0.85, np.float32),
        "gsdr": np.full((n, n), 2.0, np.float32),
        "stereo_dem": rng.normal(0, 200, (n, n)).astype(np.float32),
    }
    tile = build_tile(sources, window=(0, 0, n, n), lat_deg=12.5, lon_deg=57.2, spec=TileSpec())

    reference = SyntheticVenus(1, SyntheticConfig(size=n), seed=0)[0]
    produced = {k: v for k, v in vars(tile).items() if k != "quad"}
    assert set(produced) | {"z_true", "brightness_true_db", "gtdr_posts"} >= set(reference)

    # And the result actually feeds the model.
    batch = {}
    for k in reference:
        if k in produced:
            v = produced[k]
            batch[k] = torch.as_tensor(np.asarray(v, dtype=np.float32))
    batch = default_collate([batch])
    out = build_batch(batch, BatchSpec(augment=False), np.random.default_rng(0))
    assert out["x"].shape[1:] == (18, n, n)
    assert torch.isfinite(out["x"]).all()


def test_read_window_wraps_at_the_prime_meridian():
    """A window straddling longitude 0 must wrap. rasterio returns nodata past the edge
    instead, which puts a stripe of dead tiles down the prime meridian."""
    import numpy as np

    from data.tile import read_window

    width = 32

    class FakeSrc:
        def __init__(self, data):
            self.data = data

        def read(self, _band, window):
            return self.data[window.row_off : window.row_off + window.height,
                             window.col_off : window.col_off + window.width]

    class Window:
        def __init__(self, col_off, row_off, width, height):
            self.col_off, self.row_off, self.width, self.height = col_off, row_off, width, height

    import data.tile as tile_mod

    data = np.arange(width * 8).reshape(8, width)
    src = FakeSrc(data)

    import sys
    import types

    fake = types.ModuleType("rasterio.windows")
    fake.Window = Window
    sys.modules.setdefault("rasterio", types.ModuleType("rasterio"))
    sys.modules["rasterio.windows"] = fake
    try:
        got = read_window(src, row=0, col=width - 3, size=6, width=width)
        assert got.shape == (6, 6)
        assert np.array_equal(got[0], np.concatenate([data[0, -3:], data[0, :3]]))
    finally:
        del sys.modules["rasterio.windows"]
    _ = tile_mod


def test_upsampled_posts_land_on_the_lattice_the_loss_samples():
    """`L_alt` samples the fine grid at `offset + j * stride`. `F.interpolate` puts post j
    at `(j + 0.5) * H / P - 0.5` instead, which agrees only when H happens to equal
    P * stride. At 75 m with a 62 px stride the last post of a 512 px tile lands ~6 px
    away, so the anchor compares the model at one place against altimetry from another.
    """
    import numpy as np
    import torch

    from data.tile import GTDR_STRIDE_PX, upsample_posts

    posts = torch.tensor([[[[0.0, 100.0, 300.0, 250.0]]]])
    s = GTDR_STRIDE_PX
    up = upsample_posts(posts, (1, 512), s)
    off = s // 2
    got = [float(up[0, off + j * s]) for j in range(4)]
    assert np.allclose(got, [0.0, 100.0, 300.0, 250.0], atol=1e-3)


def test_upsampled_posts_are_smooth_and_do_not_extrapolate():
    import numpy as np
    import torch

    from data.tile import upsample_posts

    posts = torch.tensor([[[[0.0, 100.0, 300.0, 250.0]]]])
    up = upsample_posts(posts, (1, 512), 62)[0].numpy()
    assert np.isfinite(up).all()
    # Bicubic overshoot is bounded, and beyond the last post the value is held, not
    # extrapolated off toward infinity.
    assert up.min() > -60 and up.max() < 360
    assert up[-1] == pytest.approx(250.0, abs=1.0)


def test_synthetic_gtdr_agrees_with_its_own_posts():
    """The generator and the loss have to share one definition of where a post is."""
    import numpy as np
    import torch

    from data.synthetic import SyntheticConfig, SyntheticVenus

    cfg = SyntheticConfig(size=256)
    t = SyntheticVenus(1, cfg, seed=3)[0]
    up = t["gtdr_up"].numpy()
    posts = t["gtdr_posts"].numpy()
    s = cfg.gtdr_stride_px
    off = s // 2
    sampled = up[off :: s, off :: s][: posts.shape[0], : posts.shape[1]]
    assert np.allclose(sampled, posts[: sampled.shape[0], : sampled.shape[1]], atol=1e-2)
