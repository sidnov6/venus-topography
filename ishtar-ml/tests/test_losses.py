"""Each loss term must be minimised by the truth, and must survive its supervision being
entirely absent from a batch — which on Venus is the normal case, not the edge case."""

import math

import numpy as np
import pytest
import torch

from data.synthetic import SyntheticConfig, SyntheticVenus, fractal_field
from model import losses as L
from model import physics as P

PX = 75.0

# The losses normalise by each observation's uncertainty so that Section 5.7's weights
# are meaningful. These tests assert physical magnitudes, so they ask for raw units.
RAW = L.UNIT_SCALES


def terrain(seed=0, size=128, amp=200.0):
    z = fractal_field(size, 3.2, np.random.default_rng(seed))
    return torch.from_numpy(z).float()[None, None] * amp


def test_masked_mean_of_empty_mask_is_zero_not_nan():
    x = torch.randn(1, 1, 8, 8)
    out = L._masked_mean(x, torch.zeros(1, 1, 8, 8, dtype=torch.bool))
    assert float(out) == 0.0 and torch.isfinite(out)


def test_every_term_is_finite_with_no_supervision_at_all():
    """A left-look-only tile with no stereo: most terms have nothing to look at."""
    z = terrain()
    empty = torch.zeros_like(z, dtype=torch.bool)
    assert torch.isfinite(L.loss_stereo(z, torch.zeros_like(z), empty, PX))
    assert torch.isfinite(L.loss_phys(z, torch.zeros_like(z), empty, torch.tensor([[1.0, 0.0]]), torch.tensor([0.7]), PX)[0])
    assert torch.isfinite(L.loss_nll(z, torch.zeros_like(z), torch.zeros_like(z), empty, pixel_size=PX))


def test_stereo_loss_is_minimised_by_the_truth():
    z = terrain()
    valid = torch.ones_like(z, dtype=torch.bool)
    exact = float(L.loss_stereo(z, z, valid, PX))
    off = float(L.loss_stereo(z + 50.0, z, valid, PX))
    assert exact < off
    assert exact == pytest.approx(0.0, abs=1e-2)


def _consistent_gtdr(z):
    """GTDR as `data.synthetic` produces it: the same padded footprint operator the loss
    applies, over the same extent, so the pair is self-consistent at the tile edges."""
    return P.footprint_blur(z, P.FootprintSpec(), PX)


def test_alt_loss_is_zero_at_the_truth():
    z = terrain(size=256, amp=300.0)
    valid = torch.ones_like(z, dtype=torch.bool)
    assert float(L.loss_alt(z, _consistent_gtdr(z), valid, P.FootprintSpec(), PX, 62,
                            scales=RAW)) == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize("size,margin_px,bound_m", [(512, 186, 0.5), (1024, 373, 0.01)])
def test_alt_loss_ignores_detail_below_the_footprint(size, margin_px, bound_m):
    """The altimetry anchor constrains long wavelengths and nothing else.

    A 1.5 km ripple is deep in the footprint's stop band, which is exactly why the
    100 m - 10 km band is left free for the network to fill in: `L_alt` has no opinion
    there, and the physics and stereo terms decide it instead.

    It only has no opinion once the boundary is excluded, though. The same 60 m ripple
    leaks 4.5 m into the loss on a bare 256 px tile, 0.11 m on 512 px with a margin, and
    0.0007 m on 1024 px with a margin — which is the case for cutting tiles with context.
    """
    z = terrain(size=size, amp=300.0)
    gtdr = _consistent_gtdr(z)
    x = torch.arange(size, dtype=torch.float32)
    ripple = (60.0 * torch.sin(2 * math.pi * x * PX / 1500.0)).view(1, 1, 1, -1).expand_as(z).contiguous()
    valid = torch.ones_like(z, dtype=torch.bool)
    spec = P.FootprintSpec()

    assert float(L.loss_alt(z, gtdr, valid, spec, PX, 62, edge_margin_px=margin_px,
                            scales=RAW)) == pytest.approx(0.0, abs=1e-3)
    assert float(L.loss_alt(z + ripple, gtdr, valid, spec, PX, 62, edge_margin_px=margin_px,
                            scales=RAW)) < bound_m


def test_alt_loss_edge_contamination_shrinks_with_context():
    """The quantitative version of the tiling rule in `loss_alt`'s docstring."""
    spec = P.FootprintSpec()
    leak = []
    for size, margin in ((256, 0), (512, 186), (1024, 373)):
        z = terrain(size=size, amp=300.0)
        x = torch.arange(size, dtype=torch.float32)
        ripple = (60.0 * torch.sin(2 * math.pi * x * PX / 1500.0)).view(1, 1, 1, -1).expand_as(z).contiguous()
        valid = torch.ones_like(z, dtype=torch.bool)
        leak.append(float(L.loss_alt(z + ripple, _consistent_gtdr(z), valid, spec, PX, 62,
                                     edge_margin_px=margin, scales=RAW)))
    assert leak[0] > 1.0, "a bare tile really is dominated by its boundary"
    assert leak[0] > leak[1] > leak[2]


def test_alt_loss_catches_a_long_wavelength_offset():
    z = terrain(size=256, amp=300.0)
    valid = torch.ones_like(z, dtype=torch.bool)
    got = float(L.loss_alt(z + 100.0, _consistent_gtdr(z), valid, P.FootprintSpec(), PX, 62, scales=RAW))
    assert got == pytest.approx(100.0, rel=0.02)


def test_alt_edge_margin_drops_boundary_posts():
    z = terrain(size=512, amp=300.0)
    gtdr = _consistent_gtdr(z)
    valid = torch.ones_like(z, dtype=torch.bool)
    # A bump confined to the tile border must be invisible once the margin is applied.
    bump = torch.zeros_like(z)
    bump[..., :64, :] = 400.0
    spec = P.FootprintSpec()
    assert float(L.loss_alt(z + bump, gtdr, valid, spec, PX, 62, scales=RAW)) > 5.0
    assert float(L.loss_alt(z + bump, gtdr, valid, spec, PX, 62, edge_margin_px=186, scales=RAW)) < 1.0


def test_phys_loss_is_minimised_by_the_true_dem():
    """The whole premise of radarclinometry: the true DEM explains the image better than
    a flat one or a wrong-sign one."""
    z = terrain(amp=150.0)
    lv = torch.tensor([[1.0, 0.0]])
    theta = torch.tensor([0.7])
    rv_obs = P.render_rv(z, lv, theta, PX)["rv_db"]
    valid = torch.ones_like(z, dtype=torch.bool)

    at_truth, diag = L.loss_phys(z, rv_obs, valid, lv, theta, PX)
    at_flat, _ = L.loss_phys(torch.zeros_like(z), rv_obs, valid, lv, theta, PX)
    at_inverted, _ = L.loss_phys(-z, rv_obs, valid, lv, theta, PX)

    # Exact at the truth, at every pyramid level: the pyramid averages the residual, not
    # the DEM, so it introduces no Jensen bias of its own.
    assert float(at_truth) == pytest.approx(0.0, abs=1e-4)
    assert float(at_flat) > 10 * float(at_truth) + 1e-3
    assert float(at_inverted) > 10 * float(at_truth) + 1e-3
    assert float(diag["phys_resid_db"]) == pytest.approx(0.0, abs=1e-2)


def test_phys_loss_gradient_is_finite_and_nonzero():
    """Phase 0 acceptance: L_phys must actually push on the DEM, with a sane magnitude."""
    z = terrain(amp=150.0)
    lv = torch.tensor([[1.0, 0.0]])
    theta = torch.tensor([0.7])
    rv_obs = P.render_rv(z, lv, theta, PX)["rv_db"]
    valid = torch.ones_like(z, dtype=torch.bool)

    guess = torch.zeros_like(z, requires_grad=True)
    loss, _ = L.loss_phys(guess, rv_obs, valid, lv, theta, PX)
    loss.backward()
    g = guess.grad
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) > 0
    # dB per metre of elevation over a 75 m cell: order 1e-2 to 1e-4, not 1e3 or 1e-12.
    assert 1e-7 < float(g.abs().mean()) < 1e-1


def test_phys_loss_uses_the_brightness_field():
    z = terrain(amp=100.0)
    lv, theta = torch.tensor([[1.0, 0.0]]), torch.tensor([0.7])
    valid = torch.ones_like(z, dtype=torch.bool)
    b = torch.full_like(z, 2.0)
    rv_obs = P.render_rv(z, lv, theta, PX, brightness=b)["rv_db"]
    without = float(L.loss_phys(z, rv_obs, valid, lv, theta, PX)[0])
    with_b = float(L.loss_phys(z, rv_obs, valid, lv, theta, PX, brightness=b)[0])
    assert with_b < without


def test_nll_prefers_large_sigma_where_the_error_is_large():
    z = terrain()
    valid = torch.ones_like(z, dtype=torch.bool)
    target = z + 100.0
    tight = torch.full_like(z, 2.0)   # sigma ~ 2.7 m
    loose = torch.full_like(z, 9.2)   # sigma ~ 100 m
    assert float(L.loss_nll(z, loose, target, valid, pixel_size=PX)) < float(
        L.loss_nll(z, tight, target, valid, pixel_size=PX)
    )


def test_rms_loss_penalises_oversmoothing():
    z = terrain(amp=200.0)
    valid = torch.ones_like(z, dtype=torch.bool)
    cell = max(1, int(round(4641.0 / PX)))
    target = P.rms_slope(z, PX, cell)
    smooth = P.gaussian_downsample(z, 4)
    smooth = torch.nn.functional.interpolate(smooth, size=z.shape[-2:], mode="bilinear", align_corners=False)
    assert float(L.loss_rms(z, target, valid, PX)) < float(L.loss_rms(smooth, target, valid, PX))


def test_total_loss_ramp_scales_only_the_physics_terms():
    terms = {k: torch.tensor(1.0) for k in ("stereo", "alt", "phys", "cross", "rms", "nll", "reg")}
    w = L.LossWeights()
    hot = float(L.total_loss(terms, w, phys_ramp=1.0))
    cold = float(L.total_loss(terms, w, phys_ramp=0.0))
    assert hot - cold == pytest.approx(w.phys + w.cross, rel=1e-6)


def test_synthetic_tile_is_self_consistent():
    """The generator renders through the same code the loss inverts, so the true DEM must
    explain the noiseless image essentially perfectly."""
    ds = SyntheticVenus(1, SyntheticConfig(size=128, looks=10_000, brightness_sigma_db=0.0,
                                           stripe_amplitude_db=0.0), seed=7)
    t = ds[0]
    z = t["z_true"][None, None]
    rv, valid = P.rv_from_dn(t["dn_left"][None, None])
    loss, diag = L.loss_phys(z, rv, valid, t["look_left"][None], t["theta_left"].view(1), PX)
    assert float(diag["phys_resid_db"]) < 0.15


def test_nll_hinge_penalises_overconfidence_where_nothing_supervises():
    """Where no stereo and no second look constrain the prediction, the model may not
    claim to be more certain than it is where data exists."""
    z = terrain()
    target = z + 40.0
    supervised = torch.zeros_like(z, dtype=torch.bool)
    supervised[..., :64, :] = True
    unsupervised = ~supervised

    logvar = torch.full_like(z, 6.0)          # sigma ~20 m everywhere
    overconfident = logvar.clone()
    overconfident[unsupervised] = 1.0          # sigma ~1.6 m where nothing is known

    honest = float(L.loss_nll(z, logvar, target, supervised,
                              unsupervised_mask=unsupervised, pixel_size=PX))
    bluffing = float(L.loss_nll(z, overconfident, target, supervised,
                                unsupervised_mask=unsupervised, pixel_size=PX))
    assert bluffing > honest


def test_nll_hinge_does_not_reward_inflating_sigma():
    """The old formulation subtracted the mean log-variance, which paid the model to be
    vague. The hinge is silent once the bar is met."""
    z = terrain()
    target = z + 40.0
    supervised = torch.zeros_like(z, dtype=torch.bool)
    supervised[..., :64, :] = True
    unsupervised = ~supervised

    at_floor = torch.full_like(z, 6.0)
    inflated = at_floor.clone()
    inflated[unsupervised] = 11.0

    a = float(L.loss_nll(z, at_floor, target, supervised, unsupervised_mask=unsupervised, pixel_size=PX))
    b = float(L.loss_nll(z, inflated, target, supervised, unsupervised_mask=unsupervised, pixel_size=PX))
    # Inflating past the floor must never reduce the loss. (It rises slightly here only
    # because the 1 km comparison averages variance across the region boundary.)
    assert b >= a - 1e-6, "inflating sigma past the floor must not pay"
