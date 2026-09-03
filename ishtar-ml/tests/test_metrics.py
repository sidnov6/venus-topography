import numpy as np
import pytest
import torch

from data.synthetic import fractal_field
from eval import metrics as M
from model import physics as P

PX = 75.0


def terrain(seed=0, size=128, amp=200.0):
    return torch.from_numpy(fractal_field(size, 3.2, np.random.default_rng(seed))).float()[None, None] * amp


def test_cross_look_psnr_rewards_the_true_dem():
    """The metric must prefer a DEM that predicts an unseen look over one that does not."""
    z = terrain(amp=150.0)
    look_other, theta = torch.tensor([[0.0, 1.0]]), torch.tensor([0.7])
    rv_other = P.render_rv(z, look_other, theta, PX)["rv_db"]
    valid = torch.ones_like(z, dtype=torch.bool)

    good = M.cross_look_psnr(z, rv_other, valid, look_other, theta, PX)
    flat = M.cross_look_psnr(torch.zeros_like(z), rv_other, valid, look_other, theta, PX)
    assert good > flat + 10


def test_power_spectrum_distinguishes_real_detail_from_noise():
    """A noise-injected DEM must show excess power at short wavelengths — the failure mode
    the spectrum exists to catch."""
    z = terrain(size=256, amp=200.0)
    noisy = z + torch.from_numpy(np.random.default_rng(0).normal(0, 20, z.shape).astype(np.float32))
    wl, p_true = M.radial_power_spectrum(z, PX)
    _, p_noisy = M.radial_power_spectrum(noisy, PX)
    ok = np.isfinite(p_true) & np.isfinite(p_noisy)
    ratio = np.where(ok, p_noisy / p_true, np.nan)

    # Indistinguishable where the terrain has power; a growing excess where it does not.
    long = ok & (wl > 3000)
    assert np.allclose(ratio[long], 1.0, atol=0.05)
    shortest = ok & (wl < 200)
    assert ratio[shortest].min() > 3.0, "20 m of white noise must be visible at 150 m"
    assert np.nanmax(ratio[wl < 400]) > np.nanmax(ratio[(wl > 400) & (wl < 2000)])


def test_uncertainty_calibration_reads_68_percent_on_gaussian_errors():
    rng = np.random.default_rng(0)
    sigma_true = 40.0
    target = torch.zeros(1, 1, 256, 256)
    pred = torch.from_numpy(rng.normal(0, sigma_true, target.shape).astype(np.float32))
    out = M.uncertainty_calibration(pred, torch.full_like(pred, sigma_true), target,
                                    torch.ones_like(pred, dtype=torch.bool))
    assert out["coverage_1sigma"] == pytest.approx(0.68, abs=0.02)
    assert out["coverage_2sigma"] == pytest.approx(0.95, abs=0.02)


def test_altimetry_residual_is_zero_for_a_consistent_pair():
    z = terrain(size=256, amp=300.0)
    g = P.footprint_blur(z, P.FootprintSpec(), PX)
    v = torch.ones_like(z, dtype=torch.bool)
    assert M.altimetry_residual(z, g, v, PX, 62) == pytest.approx(0.0, abs=1e-2)


def test_temperature_scaling_multiplies_sigma():
    logvar = torch.zeros(1, 1, 4, 4)  # sigma = 1
    assert float(torch.exp(0.5 * M.temperature_scale(logvar, 3.0)).mean()) == pytest.approx(3.0, rel=1e-5)


def test_physics_residual_is_gain_invariant():
    """The network predicts an intrinsic-brightness field and the baselines do not, so a
    residual that counted absolute gain would hand it an advantage that is not about
    shape. Adding a constant dB offset to the observation must not change the score."""
    z = terrain(amp=150.0)
    lv, theta = torch.tensor([[1.0, 0.0]]), torch.tensor([0.7])
    rv = P.render_rv(z, lv, theta, PX)["rv_db"]
    valid = torch.ones_like(z, dtype=torch.bool)

    base = M.physics_residual_db(z, rv, valid, lv, theta, PX)
    shifted = M.physics_residual_db(z, rv + 3.0, valid, lv, theta, PX)
    assert base == pytest.approx(shifted, abs=1e-4)
    assert base == pytest.approx(0.0, abs=1e-4)

    # And with the allowance switched off it does move, so the flag is doing something.
    assert M.physics_residual_db(z, rv + 3.0, valid, lv, theta, PX, fit_offset=False) > 2.5


def test_physics_residual_still_penalises_the_wrong_shape():
    z = terrain(amp=150.0)
    lv, theta = torch.tensor([[1.0, 0.0]]), torch.tensor([0.7])
    rv = P.render_rv(z, lv, theta, PX)["rv_db"]
    valid = torch.ones_like(z, dtype=torch.bool)
    assert M.physics_residual_db(torch.zeros_like(z), rv, valid, lv, theta, PX) > 1.0


def test_cross_look_psnr_is_gain_invariant_too():
    z = terrain(amp=150.0)
    lv, theta = torch.tensor([[0.0, 1.0]]), torch.tensor([0.7])
    rv = P.render_rv(z, lv, theta, PX)["rv_db"]
    valid = torch.ones_like(z, dtype=torch.bool)
    a = M.cross_look_psnr(z, rv, valid, lv, theta, PX)
    b = M.cross_look_psnr(z, rv + 2.5, valid, lv, theta, PX)
    assert a == pytest.approx(b, rel=1e-4)
