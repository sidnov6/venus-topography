"""Baseline (b) is the control that decides whether the network is earning its keep, so
it has to be a real inversion rather than a strawman."""

import numpy as np
import torch

from data.synthetic import fractal_field
from eval import baselines
from model import physics as P

PX = 75.0


def scene(size=96, rms_slope_deg=4.0, seed=0):
    """A tile with a known DEM, its rendered image, and its footprint-blurred altimetry."""
    z = torch.from_numpy(fractal_field(size, 3.2, np.random.default_rng(seed))).float()[None, None]
    de, dn = P.sobel_gradient(z, PX)
    z = z * (np.tan(np.deg2rad(rms_slope_deg)) / float(torch.sqrt((de**2 + dn**2).mean())))
    look = torch.tensor([[1.0, 0.0]])
    theta = torch.tensor([0.7])
    rv = P.render_rv(z, look, theta, PX)["rv_db"]
    gtdr = P.footprint_blur(z, P.FootprintSpec(), PX)
    return z, rv, look, theta, gtdr


def test_classical_radarclinometry_reduces_the_image_residual():
    """It need not recover the DEM — the cross-track slope is unconstrained — but it must
    explain the image better than the altimetry alone, or the forward model is wrong."""
    z, rv, look, theta, gtdr = scene()
    valid = torch.ones_like(z, dtype=torch.bool)
    est = baselines.classical_radarclinometry(rv, valid, look, theta, gtdr, PX, steps=120, lr=30.0)

    def resid(dem):
        r = P.render_rv(dem, look, theta, PX)
        m = valid & r["valid"]
        return float((r["rv_db"] - rv).abs()[m].mean())

    assert resid(est) < 0.6 * resid(gtdr)


def test_classical_radarclinometry_stays_anchored_to_altimetry():
    """Unanchored shape-from-shading drifts: the along-track integration of slope has an
    unconstrained additive mode. The altimetry term is what stops it."""
    z, rv, look, theta, gtdr = scene()
    valid = torch.ones_like(z, dtype=torch.bool)
    est = baselines.classical_radarclinometry(rv, valid, look, theta, gtdr, PX, steps=120, lr=30.0)
    drift = float(P.footprint_blur(est - gtdr, P.FootprintSpec(), PX).abs().mean())
    assert drift < 30.0, f"altimetry drift {drift:.1f} m exceeds the Section 7 budget"


def test_bicubic_gtdr_baseline_is_the_input_unchanged():
    g = torch.randn(2, 1, 16, 16)
    assert torch.equal(baselines.bicubic_gtdr(g), g)
