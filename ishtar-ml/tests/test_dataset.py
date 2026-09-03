"""Batch assembly. The channel stack and the conditioning vector are the contract between
the dataset and the network, and a silent reordering there is a bug no loss curve shows."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import BatchSpec, build_batch, drop_second_looks
from data.masks import radar_dark_mask, seam_mask, unsupervised_mask
from data.synthetic import SyntheticConfig, SyntheticVenus
from model.unet import COND_FEATURES, INPUT_CHANNELS, UNetConfig, build_model


def batch(n=4, size=64, augment=False, seed=0):
    ds = SyntheticVenus(n, SyntheticConfig(size=size), seed=seed)
    tiles = next(iter(DataLoader(ds, batch_size=n)))
    return build_batch(tiles, BatchSpec(augment=augment), np.random.default_rng(seed))


def test_channel_stack_matches_the_model_contract():
    b = batch()
    assert b["x"].shape[1] == len(INPUT_CHANNELS) == UNetConfig().in_channels
    assert b["cond"].shape[1] == len(COND_FEATURES) == UNetConfig().cond_dim


def test_batch_feeds_the_model_without_reshaping():
    b = batch()
    out = build_model(UNetConfig())(b["x"], b["cond"], b["gtdr_up"])
    assert out["z_hat"].shape == b["gtdr_up"].shape
    assert torch.isfinite(out["z_hat"]).all()


def test_prediction_is_a_residual_over_gtdr():
    """The architecture's central choice: at initialisation the model *is* the baseline,
    because the residual head is zero-initialised."""
    b = batch()
    out = build_model(UNetConfig())(b["x"], b["cond"], b["gtdr_up"])
    assert torch.allclose(out["residual"], torch.zeros_like(out["residual"]), atol=1e-6)
    assert torch.allclose(out["z_hat"], b["gtdr_up"], atol=1e-4)


def test_no_channel_carries_absolute_planetary_elevation():
    """The GTDR channel is referenced to the tile mean, so the network cannot memorise
    'this is the 3 km terrain' from an absolute offset."""
    b = batch()
    gtdr_ch = b["x"][:, INPUT_CHANNELS.index("gtdr_up")]
    assert float(gtdr_ch.mean().abs()) < 1e-4
    assert float(b["gtdr_up"].mean().abs()) > 1.0  # the target still carries it


def test_drop_second_looks_zeroes_exactly_four_channels():
    b = batch()
    x = b["x"]
    dropped = drop_second_looks(x)
    changed = [c for i, c in enumerate(INPUT_CHANNELS) if not torch.equal(x[:, i], dropped[:, i])]
    assert set(changed) <= {"sar_right_db", "mask_right", "sar_stereo_db", "mask_stereo"}
    assert torch.equal(x[:, INPUT_CHANNELS.index("sar_left_db")],
                       dropped[:, INPUT_CHANNELS.index("sar_left_db")])


def test_masks_are_disjoint_where_they_must_be():
    b = batch()
    # A pixel cannot be both trusted stereo and unsupervised.
    assert not bool((b["stereo_trust"] & b["unsupervised"]).any())


def test_augmentation_changes_the_tile_but_keeps_the_contract():
    plain = batch(augment=False, seed=3)
    aug = batch(augment=True, seed=3)
    assert plain["x"].shape == aug["x"].shape
    assert not torch.allclose(plain["x"], aug["x"])
    assert torch.allclose(aug["look_left"].norm(dim=-1), torch.ones(aug["look_left"].shape[0]), atol=1e-5)


def test_look_dropout_clears_the_mask_with_the_image():
    """A dropped look must lose its mask too; a nonzero mask over a zeroed image tells the
    network the terrain is flat there rather than unobserved."""
    for seed in range(6):
        b = batch(n=8, augment=True, seed=seed)
        for name in ("right", "stereo"):
            zeroed = b[f"rv_{name}"].flatten(1).abs().sum(dim=1) == 0
            masked = b[f"valid_{name}"].flatten(1).any(dim=1)
            assert not bool((zeroed & masked).any())


def test_seam_mask_finds_an_injected_discontinuity():
    dem = torch.zeros(1, 1, 64, 64)
    dem[..., :, 30:33] += 400.0
    m = seam_mask(dem)
    assert not bool(m[..., :, 30:33].any()), "the seam itself must be excluded"
    assert bool(m[..., :, :20].all()), "clean terrain must survive"


def test_radar_dark_mask_excludes_dark_patches_and_a_halo():
    sar = torch.zeros(1, 1, 64, 64)
    sar[..., 20:24, 20:24] = -20.0
    valid = torch.ones_like(sar, dtype=torch.bool)
    m = radar_dark_mask(sar, valid, threshold_db=-12.0, dilation=3)
    assert not bool(m[..., 20:24, 20:24].any())
    assert not bool(m[..., 18, 22])  # the dilated halo
    assert bool(m[..., 0, 0])


def test_unsupervised_mask_is_where_nothing_constrains_the_model():
    shape = (1, 1, 8, 8)
    has_stereo = torch.zeros(shape, dtype=torch.bool)
    right = torch.zeros(shape, dtype=torch.bool)
    stereo_look = torch.zeros(shape, dtype=torch.bool)
    assert bool(unsupervised_mask(has_stereo, right, stereo_look).all())
    right[..., :4, :] = True
    assert not bool(unsupervised_mask(has_stereo, right, stereo_look)[..., :4, :].any())


def test_gtdr_nodata_is_carried_into_the_altimetry_mask():
    """GTDR nodata decodes to 0 m, and Venus has no sea level. A dropped mask would let
    `L_alt` anchor the surface to zero elevation wherever the altimeter has a gap — over
    the ~2% of the planet it never measured, confidently and silently.

    The tile is 256 px so that several 62 px post lattice points actually fall inside it;
    on a 64 px tile only one post is sampled and the test proves nothing.
    """
    from model import losses as L
    from model import physics as P

    ds = SyntheticVenus(2, SyntheticConfig(size=256), seed=0)
    tiles = next(iter(DataLoader(ds, batch_size=2)))
    tiles["gtdr_valid"] = tiles["gtdr_valid"].clone()
    tiles["gtdr_valid"][:, :64, :] = 0.0  # covers the post row at index 31

    b = build_batch(tiles, BatchSpec(augment=False), np.random.default_rng(0))
    assert not bool(b["gtdr_valid"][:, :, :64, :].any())
    assert bool(b["gtdr_valid"][:, :, 64:, :].all())

    # Corrupt the *altimetry* where it is flagged nodata. The loss reads GTDR only at the
    # sampled posts, so a working mask makes this invisible; a dropped one turns the
    # nodata sentinel into a target.
    spec, z = P.FootprintSpec(), b["gtdr_up"]
    clean = float(L.loss_alt(z, b["gtdr_up"], b["gtdr_valid"], spec, 75.0, 62, scales=L.UNIT_SCALES))
    corrupt = b["gtdr_up"].clone()
    corrupt[:, :, :64, :] = -32768.0
    masked = float(L.loss_alt(z, corrupt, b["gtdr_valid"], spec, 75.0, 62, scales=L.UNIT_SCALES))
    unmasked = float(L.loss_alt(z, corrupt, torch.ones_like(b["gtdr_valid"]), spec, 75.0, 62,
                                scales=L.UNIT_SCALES))
    assert masked == pytest.approx(clean, abs=1e-3)
    assert unmasked > 1000.0, "without the mask, nodata would become a target"


def test_gtdr_validity_is_flipped_with_the_rasters():
    ds = SyntheticVenus(2, SyntheticConfig(size=64), seed=0)
    tiles = next(iter(DataLoader(ds, batch_size=2)))
    tiles["gtdr_valid"] = tiles["gtdr_valid"].clone()
    tiles["gtdr_valid"][:, :8, :] = 0.0
    seen = set()
    for seed in range(8):
        b = build_batch(tiles, BatchSpec(augment=True), np.random.default_rng(seed))
        frac = float(b["gtdr_valid"].float().mean())
        seen.add(round(frac, 4))
        assert frac == pytest.approx(1 - 8 / 64, abs=0.02), "the mask must move, not vanish"
    assert len(seen) == 1


def test_gain_offsets_differ_between_tiles_in_a_batch():
    """Gain striping is what the model must learn to ignore. One offset shared across the
    batch could be absorbed by a batch statistic instead of by the brightness head."""
    ds = SyntheticVenus(8, SyntheticConfig(size=64), seed=2)
    tiles = next(iter(DataLoader(ds, batch_size=8)))
    b = build_batch(tiles, BatchSpec(augment=True), np.random.default_rng(0))
    valid = b["valid_left"]
    means = [float(b["rv_left"][i][valid[i]].mean()) for i in range(8)]
    assert len(set(round(m, 4) for m in means)) > 4, "offsets look shared across the batch"
