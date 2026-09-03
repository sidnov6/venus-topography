"""The eight terms have to be commensurable before Section 5.7's weights mean anything.

This is the test that would have caught the original imbalance: with raw units, the
radarclinometry term — the only source of sub-kilometre detail over 80% of Venus —
contributes well under 1% of the objective, and nothing about the loss curve says so.
"""

import numpy as np
import pytest
import torch

from data.dataset import BatchSpec, build_batch
from data.synthetic import SyntheticConfig, SyntheticVenus
from model import losses as L
from model import physics as P
from torch.utils.data import DataLoader

PX = 75.0


def realistic_terms(scales: L.LossScales) -> dict[str, torch.Tensor]:
    """Every term evaluated on a real synthetic batch, at the bicubic-GTDR baseline —
    the state the model actually starts in, since the residual head is zero-initialised."""
    ds = SyntheticVenus(8, SyntheticConfig(size=128), seed=5)
    tiles = next(iter(DataLoader(ds, batch_size=8)))
    b = build_batch(tiles, BatchSpec(augment=False), np.random.default_rng(0))
    z = b["gtdr_up"]
    brightness = torch.zeros_like(z)

    terms = {
        "stereo": L.loss_stereo(z, b["stereo_dem"], b["stereo_trust"], PX, scales=scales),
        "alt": L.loss_alt(z, b["gtdr_up"], b["gtdr_valid"], P.FootprintSpec(), PX, 62, scales=scales),
        "phys": L.loss_phys(z, b["rv_left"], b["valid_left"], b["look_left"], b["theta_left"],
                            PX, brightness=brightness, scales=scales)[0],
        "rms": L.loss_rms(z, b["rms_slope"], b["rms_valid"], PX, scales=scales),
    }
    return terms


def contributions(terms: dict[str, torch.Tensor], w: L.LossWeights) -> dict[str, float]:
    weighted = {k: getattr(w, k) * float(v) for k, v in terms.items()}
    total = sum(weighted.values())
    return {k: v / total for k, v in weighted.items()}


def test_raw_units_starve_the_physics_term():
    """The failure this normalisation exists to prevent, asserted so it cannot come back."""
    share = contributions(realistic_terms(L.UNIT_SCALES), L.LossWeights())
    assert share["phys"] < 0.02, "if this ever passes, the imbalance has changed shape"
    assert share["stereo"] > 0.7


def test_normalised_terms_are_within_an_order_of_magnitude_of_each_other():
    terms = realistic_terms(L.SCALES)
    values = [float(v) for v in terms.values() if float(v) > 0]
    assert max(values) / min(values) < 30, {k: round(float(v), 3) for k, v in terms.items()}


def test_every_term_gets_a_meaningful_share_of_the_objective():
    share = contributions(realistic_terms(L.SCALES), L.LossWeights())
    for name, frac in share.items():
        assert frac > 0.005, f"{name} contributes {frac:.3%} of the loss: it is decorative"
    assert share["phys"] > 0.02


def test_scales_carry_their_documented_meaning():
    """Each scale is the observation's own uncertainty, so a normalised term reads as
    sigmas of disagreement. A stereo DEM off by its own 75 m accuracy scores ~1."""
    z = torch.zeros(1, 1, 128, 128)
    target = torch.full_like(z, L.SCALES.stereo_m)
    valid = torch.ones_like(z, dtype=torch.bool)
    assert float(L.loss_stereo(z, target, valid, PX)) == pytest.approx(1.0, rel=0.05)

    gtdr = torch.full_like(z, L.SCALES.alt_m)
    assert float(L.loss_alt(z, gtdr, valid, P.FootprintSpec(), PX, 62)) == pytest.approx(1.0, rel=0.05)
