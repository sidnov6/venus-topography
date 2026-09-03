"""Radar physics for Venus: Muhleman backscatter, the differentiable renderer that
turns a DEM into a Magellan-like flattened-backscatter image, and the footprint /
statistics operators the altimetry and RMS-slope losses need.

Conventions used everywhere in this file:

* Elevation `z` is metres on a regular grid of spacing `pixel_size` metres, shaped
  `(B, 1, H, W)`. Row index increases southward (north-up rasters), column index
  increases eastward.
* `look_vec` is `(B, 2)` as `(east, north)`: the horizontal **ground-range direction of
  the beam**, pointing from near range to far range, i.e. *away* from the radar. With
  that convention `alpha = atan(grad(z) . look_vec)` is positive when a facet tilts
  toward the radar, which lowers the local incidence angle and brightens the return.

  The sign is easy to get backwards and the symptom is a plausible-looking inverted
  planet, so it is pinned by `tests/test_physics.py::test_slope_toward_radar_sign`:
  a radar in the west illuminates eastward, and a slope rising eastward faces it.
* Angles are radians. Backscatter is `RV`, the Muhleman-flattened ratio in dB that
  Magellan FMAP DNs encode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# Muhleman (1964) law as used for the FMAP flattening.
MUHLEMAN_A = 0.0118
MUHLEMAN_C = 0.111

# FMAP 8-bit DN encoding of RV.
DN_SCALE = 5.0
DN_OFFSET = 20.0
RV_MIN_DB = -20.0
RV_MAX_DB = 30.0
DN_NODATA = 0

_EPS = 1e-8


# --------------------------------------------------------------------------------------
# DN <-> dB
# --------------------------------------------------------------------------------------
def rv_from_dn(dn: Tensor) -> tuple[Tensor, Tensor]:
    """Decode FMAP DNs to RV in dB. Returns `(rv_db, valid)` where `valid` is False on
    nodata (DN == 0). RV is zeroed on invalid pixels so it never poisons a reduction."""
    valid = dn != DN_NODATA
    rv = (dn.to(torch.float32) - 1.0) / DN_SCALE - DN_OFFSET
    return torch.where(valid, rv, torch.zeros_like(rv)), valid


def dn_from_rv(rv_db: Tensor) -> Tensor:
    """Encode RV in dB back to FMAP DNs (used to degrade Earth SAR to Magellan-like)."""
    rv = rv_db.clamp(RV_MIN_DB, RV_MAX_DB)
    return torch.round(DN_SCALE * (rv + DN_OFFSET) + 1.0).clamp(1, 255)


# --------------------------------------------------------------------------------------
# Muhleman law
# --------------------------------------------------------------------------------------
def muhleman_sigma0(theta: Tensor) -> Tensor:
    """Muhleman backscatter cross-section for incidence angle `theta` (radians).

    `sigma0 = a * cos(theta) / (sin(theta) + c * cos(theta))**3`

    `theta` is clamped away from 0 and pi/2 so the expression stays finite and
    differentiable at grazing and nadir geometry.
    """
    t = theta.clamp(1e-3, math.pi / 2 - 1e-3)
    ct, st = torch.cos(t), torch.sin(t)
    return MUHLEMAN_A * ct / (st + MUHLEMAN_C * ct).pow(3).clamp_min(_EPS)


def muhleman_rv_db(theta_local: Tensor, theta_nominal: Tensor) -> Tensor:
    """Flattened backscatter in dB: what the FMAP encodes for a facet whose local
    incidence is `theta_local` under a mosaic flattened at `theta_nominal`."""
    ratio = muhleman_sigma0(theta_local) / muhleman_sigma0(theta_nominal).clamp_min(_EPS)
    return 10.0 * torch.log10(ratio.clamp_min(_EPS))


# --------------------------------------------------------------------------------------
# Slope
# --------------------------------------------------------------------------------------
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0


def sobel_gradient(z: Tensor, pixel_size: float) -> tuple[Tensor, Tensor]:
    """Terrain gradient as `(dz_deast, dz_dnorth)`, dimensionless (m/m).

    Rows run north to south, so the north derivative is the negative of the row
    derivative. Edges use replicate padding.
    """
    kx = _SOBEL_X.to(z.device, z.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(z.device, z.dtype).view(1, 1, 3, 3)
    zp = F.pad(z, (1, 1, 1, 1), mode="replicate")
    dz_de = F.conv2d(zp, kx) / pixel_size
    dz_drow = F.conv2d(zp, ky) / pixel_size
    return dz_de, -dz_drow


def slope_toward_radar(z: Tensor, look_vec: Tensor, pixel_size: float) -> Tensor:
    """Facet tilt toward the radar, `alpha` in radians, shape `(B, 1, H, W)`.

    Positive means tilted toward the radar (brighter, lower local incidence). `look_vec`
    points down-range, away from the radar, so terrain rising along it faces the radar.
    """
    dz_de, dz_dn = sobel_gradient(z, pixel_size)
    lv = look_vec.to(z.dtype).view(-1, 2, 1, 1)
    along = dz_de * lv[:, 0:1] + dz_dn * lv[:, 1:2]
    return torch.atan(along)


def local_incidence(alpha: Tensor, theta_nominal: Tensor) -> Tensor:
    """Small-slope local incidence angle in the range plane: `theta - alpha`."""
    return theta_nominal - alpha


# --------------------------------------------------------------------------------------
# Differentiable renderer
# --------------------------------------------------------------------------------------
def render_rv(
    z: Tensor,
    look_vec: Tensor,
    theta_nominal: Tensor,
    pixel_size: float,
    brightness: Tensor | None = None,
    layover_margin_deg: float = 5.0,
) -> dict[str, Tensor]:
    """Render a Magellan-like flattened-backscatter image from a DEM.

    This is the forward model of radarclinometry and the whole point of `L_phys`:
    DEM in, RV image out, differentiable in `z`.

    Args:
        z: `(B, 1, H, W)` elevation in metres.
        look_vec: `(B, 2)` unit ground-range vector (east, north), pointing away from
            the radar (near range to far range).
        theta_nominal: nominal incidence in radians, broadcastable to `(B, 1, H, W)`.
        pixel_size: grid spacing in metres.
        brightness: optional `b(x)` intrinsic-brightness field in dB, broadcastable to
            `(B, 1, H, W)`. This is the nuisance term for roughness and dielectric.
        layover_margin_deg: pixels with `alpha > theta_nominal - margin` are in or near
            layover, where the small-slope model breaks down; they are flagged invalid.

    Returns:
        `rv_db`, `alpha`, `theta_local`, and `valid` (True where the small-slope model
        is trustworthy).
    """
    theta_nom = torch.as_tensor(theta_nominal, dtype=z.dtype, device=z.device)
    theta_nom = theta_nom.expand_as(z) if theta_nom.dim() == 4 else theta_nom.view(-1, 1, 1, 1).expand_as(z)

    alpha = slope_toward_radar(z, look_vec, pixel_size)
    theta_loc = local_incidence(alpha, theta_nom)
    rv = muhleman_rv_db(theta_loc, theta_nom)
    if brightness is not None:
        rv = rv + brightness

    margin = math.radians(layover_margin_deg)
    valid = (alpha < theta_nom - margin) & (alpha > -(math.pi / 2 - margin))
    return {"rv_db": rv, "alpha": alpha, "theta_local": theta_loc, "valid": valid}


# --------------------------------------------------------------------------------------
# Footprint and statistics operators
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FootprintSpec:
    """Magellan altimeter footprint: an anisotropic Gaussian, ~10 x 20 km, elongated
    cross-track. `azimuth_rad` is the along-track direction clockwise from north."""

    sigma_along_m: float = 4000.0
    sigma_cross_m: float = 8000.0
    azimuth_rad: float = 0.0


def gaussian_footprint_kernel(
    sigma_along_m: float,
    sigma_cross_m: float,
    orbit_azimuth_rad: float,
    pixel_size: float,
    truncate: float = 2.5,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """Anisotropic rotated Gaussian on a grid of spacing `pixel_size`.

    Returns a normalised `(1, 1, k, k)` kernel. At 75 m this kernel is ~535 px wide, so
    do not convolve with it directly — use `footprint_blur`, which builds it at a coarse
    resolution instead.
    """
    sigma_max = max(sigma_along_m, sigma_cross_m)
    half = max(1, int(math.ceil(truncate * sigma_max / pixel_size)))
    ax = torch.arange(-half, half + 1, device=device, dtype=dtype) * pixel_size
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")  # yy south-positive, xx east-positive
    ca, sa = math.cos(orbit_azimuth_rad), math.sin(orbit_azimuth_rad)
    along = -yy * ca + xx * sa
    cross = yy * sa + xx * ca
    k = torch.exp(-0.5 * ((along / sigma_along_m) ** 2 + (cross / sigma_cross_m) ** 2))
    return (k / k.sum().clamp_min(_EPS)).view(1, 1, *k.shape)


def apply_footprint(z: Tensor, kernel: Tensor) -> Tensor:
    """Direct convolution with a footprint kernel, replicate-padded. Only sane for small
    kernels; `footprint_blur` is what training should call."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(F.pad(z, (pad, pad, pad, pad), mode="replicate"), kernel.to(z.dtype))


def footprint_blur(
    z: Tensor,
    spec: FootprintSpec,
    pixel_size: float,
    truncate: float = 3.5,
    max_kernel_px: int = 96,
) -> Tensor:
    """Convolve a DEM with the altimeter footprint, cheaply and differentiably.

    A 10 x 20 km Gaussian on a 75 m grid is a ~535 px kernel; convolving with it directly
    costs ~2e10 multiply-adds per 256 px tile and dominates the whole training step. The
    footprint is far wider than the grid, so instead we

      1. anti-alias downsample by `f`, which itself applies a Gaussian of
         `sigma_ds = 0.5 * f` pixels,
      2. convolve at the coarse resolution with the *residual* Gaussian
         `sqrt(sigma^2 - sigma_ds^2)`, so the composition has exactly the target width,
      3. bilinearly upsample back to the input grid.

    Step 3 loses nothing: the field is band-limited far below the fine grid by then.
    """
    sigma_min = min(spec.sigma_along_m, spec.sigma_cross_m)
    f = max(1, int(sigma_min / (4.0 * pixel_size)))
    # Keep the coarse kernel bounded even for extreme aspect ratios.
    while f > 1 and (truncate * max(spec.sigma_along_m, spec.sigma_cross_m) / (pixel_size * f)) > max_kernel_px:
        f += 1

    coarse_px = pixel_size * f
    # A 0.5-pixel prefilter leaves enough aliasing to show up as a several-metre bias in
    # the altimetry residual, which is the one number that must stay clean; 1.0 costs
    # nothing here because the residual Gaussian absorbs it exactly.
    prefilter_scale = 1.0
    sigma_ds_m = prefilter_scale * f * pixel_size if f > 1 else 0.0
    resid = lambda s: math.sqrt(max(s * s - sigma_ds_m * sigma_ds_m, (0.5 * coarse_px) ** 2))

    zc = gaussian_downsample(z, f, sigma_scale=prefilter_scale)
    k = gaussian_footprint_kernel(
        resid(spec.sigma_along_m), resid(spec.sigma_cross_m), spec.azimuth_rad,
        coarse_px, truncate=truncate, device=z.device, dtype=z.dtype,
    )
    blurred = apply_footprint(zc, k)
    if f == 1:
        return blurred
    return F.interpolate(blurred, size=z.shape[-2:], mode="bilinear", align_corners=False)


def gaussian_downsample(z: Tensor, factor: int, sigma_scale: float = 0.5) -> Tensor:
    """Anti-aliased downsample by an integer factor: Gaussian blur then strided sample.

    Used to bring `z_hat` to the scale at which the stereo DEM is actually trusted
    (~1 km) before comparing. `sigma_scale` is the prefilter width in output pixels;
    0.5 is the usual mild default, 1.0 is what `footprint_blur` needs to keep aliasing
    out of the altimetry residual.
    """
    if factor <= 1:
        return z
    sigma = sigma_scale * factor
    half = int(math.ceil(2.5 * sigma))
    ax = torch.arange(-half, half + 1, device=z.device, dtype=z.dtype)
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    zk = F.pad(z, (half, half, 0, 0), mode="replicate")
    zk = F.conv2d(zk, g.view(1, 1, 1, -1))
    zk = F.pad(zk, (0, 0, half, half), mode="replicate")
    zk = F.conv2d(zk, g.view(1, 1, -1, 1))
    # Sample at cell centres, not cell corners. Decimating from index 0 puts the coarse
    # samples half a coarse cell off from where `align_corners=False` interpolation
    # expects them, which shows up as an f/2-pixel registration shift the moment the
    # result is upsampled again.
    off = factor // 2
    return zk[..., off::factor, off::factor]


def rms_slope(z: Tensor, pixel_size: float, cell_px: int) -> Tensor:
    """RMS of the terrain slope magnitude inside `cell_px` blocks, in radians.

    This is the quantity the Magellan GSDR reports at 4.6 km, so `cell_px` is
    `4641 / pixel_size` on the native grid.
    """
    dz_de, dz_dn = sobel_gradient(z, pixel_size)
    s2 = dz_de.pow(2) + dz_dn.pow(2)
    return torch.sqrt(F.avg_pool2d(s2, cell_px).clamp_min(_EPS)).atan()
