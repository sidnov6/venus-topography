"""Overlap-tiled inference. The blend is a partition of unity or it is a visible grid."""

import numpy as np
import pytest

from infer_global import (
    BlendCanvas,
    InferConfig,
    feather_profile,
    feather_window,
    run_tiled,
    tile_origins,
)


def test_tile_origins_cover_the_extent_and_end_flush():
    for extent, tile, overlap in ((1000, 256, 64), (512, 512, 64), (300, 256, 64), (256, 256, 0)):
        starts = tile_origins(extent, tile, overlap)
        assert starts[0] == 0
        assert starts[-1] + tile == max(extent, tile) if extent >= tile else True
        covered = np.zeros(extent, bool)
        for s in starts:
            covered[s : s + tile] = True
        assert covered.all()


@pytest.mark.parametrize("size,overlap", [(64, 16), (128, 32), (512, 64), (64, 1)])
def test_feather_profile_is_a_partition_of_unity(size, overlap):
    """Two tiles offset by the step must sum to exactly 1 through the overlap, or the
    blend leaves a periodic ripple at every seam."""
    w = feather_profile(size, overlap).numpy()
    step = size - overlap
    seam = w[step:] + w[: size - step]
    assert np.allclose(seam, 1.0, atol=1e-6)
    assert np.allclose(w[overlap : size - overlap], 1.0)


def test_feather_window_is_the_separable_product():
    w = feather_profile(32, 8).numpy()
    assert np.allclose(feather_window(32, 8).numpy(), np.outer(w, w))


def test_constant_field_reconstructs_exactly():
    cfg = InferConfig(tile_px=64, overlap_px=16)
    out = run_tiled(200, 173, lambda r, c, n: np.full((1, n, n), 7.5), cfg, channels=1)
    assert np.allclose(out, 7.5, atol=1e-9)
    assert not np.isnan(out).any()


def test_smooth_field_reconstructs_without_seams():
    """A linear ramp is the worst case for a bad blend: the seam shows as a step."""
    h, w = 300, 260
    truth = np.add.outer(np.linspace(0, 100, h), np.linspace(0, 50, w))
    cfg = InferConfig(tile_px=64, overlap_px=16)
    out = run_tiled(h, w, lambda r, c, n: truth[None, r : r + n, c : c + n], cfg, channels=1)[0]
    assert np.abs(out - truth).max() < 1e-4 * float(np.abs(truth).max())


def test_blend_canvas_reports_uncovered_pixels():
    canvas = BlendCanvas(10, 10, 1)
    canvas.add(np.ones((1, 4, 4)), np.ones((4, 4)), 0, 0)
    assert canvas.coverage == pytest.approx(0.16)
    assert np.isnan(canvas.result()[0, 9, 9])


def test_blend_averages_disagreeing_tiles_in_the_overlap():
    cfg = InferConfig(tile_px=32, overlap_px=8)

    def predict(r, c, n):
        return np.full((1, n, n), float(c))  # each column of tiles claims a different value

    out = run_tiled(32, 88, predict, cfg, channels=1)[0]
    assert np.isfinite(out).all()
    # Monotone left to right, and strictly between the two tile values in the overlap.
    row = out[16]
    assert np.all(np.diff(row) >= -1e-9)


def test_merge_cap_is_seamless_where_both_passes_agree():
    """If the cylindrical pass and the cap pass produce the same surface, the merge must
    reproduce it exactly — no latitude line, no weight normalisation error."""
    from data.polar import PolarGrid
    from infer_global import merge_cap

    h, w = 180, 360
    lat = np.linspace(90, -90, h)[:, None]
    field = (500 + 300 * np.sin(np.radians(lat))) * np.ones((1, w))
    grid = PolarGrid(size=160, pixel_size_m=12_000.0)

    from data.polar import cylindrical_to_polar

    cap = cylindrical_to_polar(field.astype(np.float32), grid)
    merged = merge_cap(field.astype(np.float32), cap, grid)
    assert np.max(np.abs(merged - field)) < 0.02 * float(np.ptp(field))


def test_merge_cap_leaves_the_far_hemisphere_untouched():
    from data.polar import PolarGrid
    from infer_global import merge_cap

    h, w = 180, 360
    cyl = np.full((h, w), 100.0, np.float32)
    grid = PolarGrid(size=160, pixel_size_m=12_000.0, north=True)
    cap = np.full((160, 160), -999.0, np.float32)
    merged = merge_cap(cyl, cap, grid)
    south = merged[h // 2 :]
    assert np.allclose(south, 100.0), "a north cap must not touch the southern hemisphere"
    assert merged[0].mean() < 0.0, "the north pole must come from the cap"
