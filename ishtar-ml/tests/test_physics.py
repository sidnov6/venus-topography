"""The physics tests are the ones that matter. Everything else in ISHTAR can be wrong in
a way that shows up as a bad number; the sign conventions can be wrong in a way that
shows up as a beautiful, confident, upside-down planet."""

import math

import numpy as np
import pytest
import torch

from model import physics as P


def ramp(size=64, slope=0.1, axis="east", pixel_size=75.0):
    """A plane rising at `slope` (m/m) toward the east or the north."""
    a = torch.arange(size, dtype=torch.float32) * pixel_size * slope
    z = a.view(1, 1, 1, -1).expand(1, 1, size, size).contiguous()
    if axis == "north":
        z = torch.rot90(z, 1, dims=(-2, -1)).contiguous()
    return z


def test_dn_roundtrip():
    """DN 1..251 spans the documented -20..+30 dB range; 252..255 decode above it and
    are not round-trippable, which is a property of the product, not a bug here."""
    dn = torch.arange(1, 252, dtype=torch.float32).view(1, 1, 1, -1)
    rv, valid = P.rv_from_dn(dn)
    assert valid.all()
    assert torch.allclose(P.dn_from_rv(rv), dn)
    assert rv.min() == pytest.approx(P.RV_MIN_DB)
    assert rv.max() == pytest.approx(P.RV_MAX_DB)


def test_dn_zero_is_nodata():
    _, valid = P.rv_from_dn(torch.zeros(1, 1, 4, 4))
    assert not valid.any()


def test_muhleman_decreases_with_incidence():
    th = torch.deg2rad(torch.linspace(5, 80, 40))
    s = P.muhleman_sigma0(th)
    assert (s[1:] < s[:-1]).all(), "backscatter must fall monotonically with incidence"


def test_flat_terrain_renders_zero_db():
    z = torch.zeros(1, 1, 32, 32)
    r = P.render_rv(z, torch.tensor([[1.0, 0.0]]), torch.tensor([math.radians(45)]), 75.0)
    assert torch.allclose(r["rv_db"], torch.zeros_like(r["rv_db"]), atol=1e-5)


@pytest.mark.parametrize("axis,downrange", [("east", [1.0, 0.0]), ("north", [0.0, 1.0])])
def test_slope_toward_radar_sign(axis, downrange):
    """The convention test. A ramp rising along the down-range direction faces the radar,
    so alpha is positive and the return is brighter than the same ramp seen from the
    opposite side. Get this backwards and ISHTAR produces a confident inverted planet."""
    z = ramp(axis=axis, slope=0.2)
    lv_facing = torch.tensor([downrange])   # beam travels up the ramp: slope faces radar
    lv_backing = -lv_facing                 # radar on the other side: slope faces away
    theta = torch.tensor([math.radians(45)])

    a_facing = P.slope_toward_radar(z, lv_facing, 75.0)[..., 4:-4, 4:-4]
    a_backing = P.slope_toward_radar(z, lv_backing, 75.0)[..., 4:-4, 4:-4]

    assert float(a_facing.mean()) > 0
    assert float(a_backing.mean()) < 0
    assert float(a_facing.mean()) == pytest.approx(-float(a_backing.mean()), rel=1e-5)
    assert float(a_facing.mean()) == pytest.approx(math.atan(0.2), rel=1e-3)

    rv_facing = P.render_rv(z, lv_facing, theta, 75.0)["rv_db"][..., 4:-4, 4:-4]
    rv_backing = P.render_rv(z, lv_backing, theta, 75.0)["rv_db"][..., 4:-4, 4:-4]
    assert float(rv_facing.mean()) > float(rv_backing.mean()), "slopes facing the radar are brighter"


def test_flipping_look_vec_flips_slope_sign():
    """The Phase 0 acceptance check from the architecture note, at the physics level."""
    rng = np.random.default_rng(0)
    z = torch.from_numpy(rng.normal(size=(2, 1, 64, 64)).astype(np.float32)) * 100
    lv = torch.tensor([[0.8, 0.6], [-0.3, 0.95]])
    a = P.slope_toward_radar(z, lv, 75.0)
    a_flipped = P.slope_toward_radar(z, -lv, 75.0)
    assert torch.allclose(a, -a_flipped, atol=1e-6)


def test_render_is_differentiable_in_z():
    z = torch.zeros(1, 1, 32, 32, requires_grad=True)
    r = P.render_rv(z + 0.0, torch.tensor([[1.0, 0.0]]), torch.tensor([math.radians(40)]), 75.0)
    r["rv_db"].sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_layover_mask_triggers_on_steep_slopes():
    theta = math.radians(30)
    gentle = ramp(slope=0.05)
    steep = ramp(slope=2.0)  # ~63 deg, far past layover at 30 deg incidence
    lv = torch.tensor([[1.0, 0.0]])  # beam travels east, up the ramp: it faces the radar
    assert P.render_rv(gentle, lv, torch.tensor([theta]), 75.0)["valid"].all()
    assert not P.render_rv(steep, lv, torch.tensor([theta]), 75.0)["valid"].any()


def test_gradient_magnitude_on_known_ramp():
    z = ramp(slope=0.1)
    de, dn = P.sobel_gradient(z, 75.0)
    assert float(de[..., 4:-4, 4:-4].mean()) == pytest.approx(0.1, rel=1e-4)
    assert float(dn[..., 4:-4, 4:-4].abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_footprint_blur_matches_analytic_transfer_function():
    """A Gaussian footprint of sigma s attenuates wavelength L by exp(-2 pi^2 s^2 / L^2).

    The multiscale implementation must reproduce that to within a few percent at every
    wavelength the footprint actually passes, or the altimetry loss is anchoring to the
    wrong surface.
    """
    n, rows, px = 8192, 128, 75.0  # 614 km wide: several periods of every wavelength tested
    spec = P.FootprintSpec(4000.0, 8000.0, 0.0)
    x = torch.arange(n, dtype=torch.float32)
    for wl in (25_000.0, 50_000.0, 100_000.0):
        sig = torch.sin(2 * math.pi * x * px / wl).view(1, 1, 1, -1).expand(1, 1, rows, -1).contiguous()
        blurred = P.footprint_blur(sig, spec, px)
        # Peak amplitude well away from the replicate-padded edges.
        got = float(blurred[0, 0, rows // 2, n // 8 : -n // 8].abs().max())
        want = math.exp(-0.5 * (2 * math.pi * spec.sigma_cross_m / wl) ** 2)
        assert got == pytest.approx(want, rel=0.05), f"lambda={wl / 1000:.0f} km: {got} vs {want}"


def test_footprint_blur_preserves_the_mean():
    z = torch.full((1, 1, 128, 128), 300.0)
    assert float(P.footprint_blur(z, P.FootprintSpec(), 75.0).mean()) == pytest.approx(300.0, rel=1e-4)


def test_gaussian_downsample_is_registration_preserving():
    """Decimation must sample cell centres: an off-by-half-cell here becomes an
    f/2-pixel shift the moment the result is upsampled again."""
    n, f = 128, 8
    x = torch.arange(n, dtype=torch.float32).view(1, 1, 1, -1).expand(1, 1, n, -1).contiguous()
    down = P.gaussian_downsample(x, f)
    up = torch.nn.functional.interpolate(down, size=(n, n), mode="bilinear", align_corners=False)
    interior = slice(2 * f, -2 * f)
    # An even decimation factor leaves an irreducible half-pixel offset (the cell centre
    # falls between samples); what must not survive is the f/2-pixel error that
    # decimating from index 0 produces.
    assert float((up - x)[..., interior, interior].abs().max()) < 0.75


def test_rms_slope_on_known_ramp():
    z = ramp(size=128, slope=0.1)
    rs = P.rms_slope(z, 75.0, 32)
    assert float(torch.rad2deg(rs).mean()) == pytest.approx(math.degrees(math.atan(0.1)), abs=0.2)


def test_render_gradient_matches_finite_differences():
    """The renderer is only useful because it is differentiable in the DEM, so the
    analytic gradient is checked against a numerical one. `torch.autograd.gradcheck` in
    float64 is the strongest statement available: it perturbs every input element."""
    torch.manual_seed(0)
    z = (torch.randn(1, 1, 9, 9, dtype=torch.float64) * 20).requires_grad_(True)
    lv = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    theta = torch.tensor([0.7], dtype=torch.float64)

    def render(zz):
        return P.render_rv(zz, lv, theta, 75.0)["rv_db"]

    assert torch.autograd.gradcheck(render, (z,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_footprint_blur_gradient_matches_finite_differences():
    """`L_alt` back-propagates through the multiscale footprint, which is three chained
    operations with a hand-picked decimation factor — exactly the kind of thing that can
    be forward-correct and backward-wrong."""
    torch.manual_seed(1)
    z = (torch.randn(1, 1, 24, 24, dtype=torch.float64) * 100).requires_grad_(True)
    spec = P.FootprintSpec(600.0, 900.0, 0.0)

    def blur(zz):
        return P.footprint_blur(zz, spec, 75.0)

    assert torch.autograd.gradcheck(blur, (z,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_render_gradient_tilts_the_facet_toward_the_radar():
    """Sign check on the gradient, not just its magnitude.

    With `look_vec` pointing down-range (east), the radar is in the west, and a facet
    faces it by rising eastward. So to brighten one pixel, the loss must want the ground
    raised on its east side and lowered on its west side. The opposite pattern is what an
    inverted convention produces, and gradcheck would pass either way.
    """
    z = torch.zeros(1, 1, 32, 32, requires_grad=True)
    lv = torch.tensor([[1.0, 0.0]])
    out = P.render_rv(z, lv, torch.tensor([0.7]), 75.0)["rv_db"]
    out[0, 0, 16, 16].backward()
    g = z.grad[0, 0]
    assert float(g[16, 17]) > 0, "raise the ground east of the pixel"
    assert float(g[16, 15]) < 0, "lower the ground west of it"
    assert float(g[15, 16]) == pytest.approx(0.0, abs=1e-9), "no north-south sensitivity"
