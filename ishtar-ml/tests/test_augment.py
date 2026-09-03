"""Augmentation must move the geometry with the pixels."""

import numpy as np
import pytest
import torch

from data.augment import apply_dihedral, dihedral_look_vec, gain_offset, speckle
from data.geometry import look_vector
from model.physics import render_rv, slope_toward_radar

DIHEDRALS = [(k, fh, fv) for k in range(4) for fh in (False, True) for fv in (False, True)]


@pytest.mark.parametrize("k,fh,fv", DIHEDRALS)
def test_dihedral_preserves_radar_geometry(k, fh, fv):
    """Transforming the DEM and the look vector together must leave the slope toward the
    radar unchanged (up to the same transform). This is the augmentation bug the
    architecture note calls out: forget the look vector and you train inverted physics
    while every loss curve still looks healthy."""
    rng = np.random.default_rng(3)
    z = torch.from_numpy(rng.normal(size=(1, 1, 64, 64)).astype(np.float32)) * 200
    lv = torch.from_numpy(np.array([look_vector("left")]))

    base = slope_toward_radar(z, lv, 75.0)
    moved = slope_toward_radar(apply_dihedral(z, k, fh, fv), dihedral_look_vec(lv, k, fh, fv), 75.0)
    assert torch.allclose(apply_dihedral(base, k, fh, fv), moved, atol=1e-5)


@pytest.mark.parametrize("k,fh,fv", DIHEDRALS)
def test_dihedral_preserves_rendered_image(k, fh, fv):
    """Compared where the render is actually used — outside layover.

A 90-degree rotation reorders the Sobel arithmetic, and float32 rounding of order 1e-7
    is amplified to ~5e-4 dB where the Muhleman law is steepest, near the layover
    boundary. The tolerance is set from what the data can express, not from float32
    exactness: FMAP stores RV in 0.2 dB steps, so 1e-3 dB is two hundred times finer than
    anything the product records. `test_a_wrong_look_vector_is_caught` shows the test
    still has all the sensitivity it needs.
    """
    rng = np.random.default_rng(4)
    z = torch.from_numpy(rng.normal(size=(1, 1, 48, 48)).astype(np.float32)) * 150
    lv = torch.from_numpy(np.array([look_vector("right")]))
    theta = torch.tensor([0.7])

    base = render_rv(z, lv, theta, 75.0)
    moved = render_rv(apply_dihedral(z, k, fh, fv), dihedral_look_vec(lv, k, fh, fv), theta, 75.0)
    ok = apply_dihedral(base["valid"], k, fh, fv) & moved["valid"]
    assert ok.any(), "the test terrain is entirely in layover"
    diff = (apply_dihedral(base["rv_db"], k, fh, fv) - moved["rv_db"])[ok]
    assert float(diff.abs().max()) < 1e-3


def test_dihedral_look_vec_stays_unit_length():
    lv = torch.from_numpy(np.array([look_vector("left"), look_vector("right")]))
    for k, fh, fv in DIHEDRALS:
        v = dihedral_look_vec(lv, k, fh, fv)
        assert torch.allclose(v.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_left_and_right_looks_are_opposite():
    assert np.allclose(look_vector("left"), -look_vector("right"))


def test_gain_offset_is_spatially_constant():
    rv = torch.zeros(1, 1, 8, 8)
    out = gain_offset(rv, 3.0, np.random.default_rng(0))
    assert float(out.std()) == pytest.approx(0.0, abs=1e-6)
    assert abs(float(out.mean())) <= 3.0


def test_speckle_is_multiplicative_and_unbiased_in_power():
    rv = torch.zeros(1, 1, 256, 256)  # 0 dB = unit power
    out = speckle(rv, 6.0, np.random.default_rng(0))
    power = 10 ** (out / 10.0)
    assert float(power.mean()) == pytest.approx(1.0, rel=0.02)
    assert float(out.mean()) < 0.0, "log of a unit-mean gamma is biased low, as on real SAR"


def test_a_wrong_look_vector_is_caught():
    """The tolerance above is loose enough to ignore float32 dust; this shows it is still
    far tighter than any real convention error. Rotating the raster without rotating the
    look vector — the exact bug the augmentation is written to prevent — disagrees by
    whole decibels, four orders of magnitude above the threshold."""
    rng = np.random.default_rng(4)
    z = torch.from_numpy(rng.normal(size=(1, 1, 48, 48)).astype(np.float32)) * 150
    lv = torch.from_numpy(np.array([look_vector("right")]))
    theta = torch.tensor([0.7])

    base = render_rv(z, lv, theta, 75.0)
    forgot = render_rv(apply_dihedral(z, 1, False, False), lv, theta, 75.0)
    ok = apply_dihedral(base["valid"], 1, False, False) & forgot["valid"]
    diff = (apply_dihedral(base["rv_db"], 1, False, False) - forgot["rv_db"])[ok].abs()
    assert float(diff.max()) > 1.0
