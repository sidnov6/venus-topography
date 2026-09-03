"""The Earth-to-Magellan degradation is a domain bridge, so it is worth testing that it
lands in the right units rather than merely running."""

import numpy as np
import pytest
import torch

from data.earth import DegradeConfig, degrade, flatten_to_muhleman, resample_to_75m
from model import physics as P


def test_flattening_is_the_inverse_of_the_muhleman_law():
    """A surface obeying Muhleman exactly must flatten to 0 dB at every incidence — which
    is the definition of the FMAP's RV."""
    theta = torch.deg2rad(torch.linspace(20, 50, 32)).view(1, 1, 1, -1)
    sigma0_db = 10 * torch.log10(P.muhleman_sigma0(theta))
    assert torch.allclose(flatten_to_muhleman(sigma0_db, theta), torch.zeros_like(theta), atol=1e-5)


def test_degrade_lands_on_the_magellan_dn_lattice():
    rv = torch.zeros(1, 1, 64, 64)
    out = degrade(rv, DegradeConfig(), np.random.default_rng(0))
    dn = P.dn_from_rv(out)
    assert torch.allclose(dn, torch.round(dn))
    assert float(dn.min()) >= 1 and float(dn.max()) <= 255


def test_degrade_adds_speckle_and_striping_but_not_bias():
    rv = torch.zeros(1, 1, 256, 256)
    cfg = DegradeConfig(gain_jitter_db=0.0, quantise_to_dn=False)
    out = degrade(rv, cfg, np.random.default_rng(1))
    assert float(out.std()) > 1.0, "speckle must dominate a flat scene"
    # Speckle is unbiased in power even though it is biased in dB.
    assert float((10 ** (out / 10.0)).mean()) == pytest.approx(1.0, rel=0.05)


def test_resample_to_75m_preserves_the_mean():
    x = torch.full((1, 1, 256, 256), 3.0)
    assert float(resample_to_75m(x, 10.0, DegradeConfig()).mean()) == pytest.approx(3.0, rel=1e-4)
