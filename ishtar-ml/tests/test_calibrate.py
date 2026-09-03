"""Phase 4. The uncertainty map is what makes a model-derived DEM honest, so the
calibration has to be right and has to be checked on data it was not fitted to."""

import numpy as np
import pytest
import torch

from calibrate import fit_temperature
from eval.metrics import temperature_scale, uncertainty_calibration


def test_temperature_recovers_a_known_miscalibration():
    rng = np.random.default_rng(0)
    true_sigma = 40.0
    err = torch.from_numpy(np.abs(rng.normal(0, true_sigma, 200_000)).astype(np.float32))
    claimed = torch.full_like(err, true_sigma / 3.0)  # three times overconfident
    assert fit_temperature(err, claimed) == pytest.approx(3.0, rel=0.05)


def test_temperature_is_one_when_already_calibrated():
    rng = np.random.default_rng(1)
    sigma = 25.0
    err = torch.from_numpy(np.abs(rng.normal(0, sigma, 200_000)).astype(np.float32))
    assert fit_temperature(err, torch.full_like(err, sigma)) == pytest.approx(1.0, rel=0.05)


def test_calibration_hits_the_target_coverage():
    rng = np.random.default_rng(2)
    err = torch.from_numpy(np.abs(rng.normal(0, 60.0, 100_000)).astype(np.float32))
    claimed = torch.full_like(err, 10.0)
    t = fit_temperature(err, claimed)
    after = uncertainty_calibration(err, t * claimed, torch.zeros_like(err),
                                    torch.ones_like(err, dtype=torch.bool))
    assert after["coverage_1sigma"] == pytest.approx(0.683, abs=0.01)


def test_temperature_scaling_is_the_same_operation_in_log_variance_space():
    """`calibrate.py` fits a multiplier on sigma; `metrics.temperature_scale` applies it to
    the log-variance the model actually stores. They must agree."""
    logvar = torch.tensor([[[[0.0, 2.0, -3.0]]]])
    t = 2.5
    assert torch.allclose(torch.exp(0.5 * temperature_scale(logvar, t)),
                          t * torch.exp(0.5 * logvar), rtol=1e-5)


def test_a_single_scalar_cannot_fix_a_heavy_tail():
    """Matching 1-sigma coverage does not guarantee 2-sigma coverage: a heavier-than-
    Gaussian error distribution stays under-covered at 2 sigma, and the report has to say
    so rather than claiming calibration on one number."""
    rng = np.random.default_rng(3)
    err = torch.from_numpy(np.abs(rng.standard_t(df=3, size=200_000) * 30).astype(np.float32))
    claimed = torch.full_like(err, 10.0)
    t = fit_temperature(err, claimed)
    after = uncertainty_calibration(err, t * claimed, torch.zeros_like(err),
                                    torch.ones_like(err, dtype=torch.bool))
    assert after["coverage_1sigma"] == pytest.approx(0.683, abs=0.01)
    assert after["coverage_2sigma"] < 0.94
