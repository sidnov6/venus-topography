"""The eight loss terms of Section 5.

Every term returns a scalar and is safe when its supervision is entirely absent from a
batch (it returns 0 with no gradient rather than a NaN), because Venus supervision is
patchy by construction: stereo covers ~20% of the planet and a second look ~17%.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from . import physics

_EPS = 1e-8


@dataclass(frozen=True)
class LossScales:
    """Per-term normalisers, so the weights in `LossWeights` mean what they say.

    The eight terms are dimensionally incommensurable: `L_stereo` and `L_alt` are metres
    of elevation (order 100), `L_phys` is decibels (order 2), `L_rms` is radians (order
    0.05). Applying Section 5.7's weights to the raw values makes `w_p = 0.3` on a term
    of size 2 contribute about half a percent of the objective, while `w_s = 1.0` on a
    term of size 100 contributes most of it — so the radarclinometry term, which is the
    only source of sub-kilometre detail over 80% of the planet, is effectively switched
    off. The loss curve looks fine throughout.

    Each scale is the observation's own uncertainty, which makes the normalised terms
    read as "sigmas of disagreement" and the weights read as relative trust:

    * `stereo_m`  : Herrick et al. quote 50-100 m vertical.
    * `alt_m`     : the Section 7 drift budget, and roughly GTDR's own accuracy.
    * `earth_m`   : Copernicus GLO-30 vertical accuracy, ~4 m absolute but ~30 m in the
                    steep unvegetated terrain the Earth set is drawn from.
    * `phys_db`   : the speckle-limited residual floor at 4-8 looks.
    * `slope_rad` : ~1 degree, the scale at which a slope error starts to matter.
    * `rms_rad`   : ditto, for the GSDR roughness target.
    """

    stereo_m: float = 75.0
    alt_m: float = 30.0
    earth_m: float = 30.0
    phys_db: float = 2.0
    slope_rad: float = 0.0175
    rms_rad: float = 0.0175


SCALES = LossScales()

UNIT_SCALES = LossScales(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
"""All-ones scales, so a loss can be read back in its own physical units (metres, dB).

Used by the tests, and useful when you want a diagnostic in units you can reason about
rather than in sigmas.
"""


@dataclass
class LossWeights:
    """Section 5.7 starting point. `w_phys` is ramped from 0 by the trainer: the physics
    loss is unstable while the DEM is still garbage."""

    earth: float = 1.0
    stereo: float = 1.0
    alt: float = 2.0
    phys: float = 0.3
    cross: float = 0.3
    rms: float = 0.05
    nll: float = 0.1
    reg: float = 0.01


def _masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    """Mean of `x` over True pixels of `mask`, returning 0 when the mask is empty.

    Deliberately branch-free: an `if mask.sum() == 0` here would force a device sync on
    every loss term of every step.
    """
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def charbonnier(x: Tensor, eps: float = 1e-3) -> Tensor:
    """Smooth L1 that keeps a gradient at zero; robust to the stereo DEM's outliers."""
    return torch.sqrt(x * x + eps * eps)


def gradient_l1(pred: Tensor, target: Tensor, pixel_size: float, mask: Tensor | None = None) -> Tensor:
    """L1 on the terrain gradient — the term that makes relief right, not just heights."""
    pe, pn = physics.sobel_gradient(pred, pixel_size)
    te, tn = physics.sobel_gradient(target, pixel_size)
    return _masked_mean((pe - te).abs() + (pn - tn).abs(), mask)


# --------------------------------------------------------------------------------------
# 5.0 Earth stage
# --------------------------------------------------------------------------------------
def loss_earth(z_hat: Tensor, z_true: Tensor, pixel_size: float, mask: Tensor | None = None,
               scales: LossScales = SCALES) -> Tensor:
    """L1 + gradient L1 against GLO-30. Only used in the Earth pretraining phase, where
    a real label exists."""
    height = _masked_mean((z_hat - z_true).abs(), mask) / scales.earth_m
    slope = gradient_l1(z_hat, z_true, pixel_size, mask) / scales.slope_rad
    return height + slope


# --------------------------------------------------------------------------------------
# 5.1 Stereo
# --------------------------------------------------------------------------------------
def loss_stereo(
    z_hat: Tensor,
    stereo_dem: Tensor,
    valid: Tensor,
    pixel_size: float,
    trusted_scale_m: float = 1000.0,
    grad_weight: float = 1.0,
    scales: LossScales = SCALES,
) -> Tensor:
    """Compare at the scale the stereo DEM is actually trusted at (~1 km), not at 75 m.

    `valid` must already exclude mosaic seams and radar-dark patches (see
    `data.masks.stereo_artefact_mask`).
    """
    factor = max(1, int(round(trusted_scale_m / pixel_size)))
    zp = physics.gaussian_downsample(z_hat, factor)
    zt = physics.gaussian_downsample(stereo_dem * valid.to(z_hat.dtype), factor)
    w = physics.gaussian_downsample(valid.to(z_hat.dtype), factor)
    keep = w > 0.5
    zt = zt / w.clamp_min(_EPS)
    height = _masked_mean(charbonnier(zp - zt), keep) / scales.stereo_m
    slope = gradient_l1(zp, zt, pixel_size * factor, keep) / scales.slope_rad
    return height + grad_weight * slope


# --------------------------------------------------------------------------------------
# 5.2 Altimetry
# --------------------------------------------------------------------------------------
def loss_alt(
    z_hat: Tensor,
    gtdr: Tensor,
    valid: Tensor,
    footprint: physics.FootprintSpec,
    pixel_size: float,
    gtdr_stride_px: int,
    edge_margin_px: int = 0,
    scales: LossScales = SCALES,
) -> Tensor:
    """Anchor the surface to Magellan altimetry: `z_hat` seen through the altimeter
    footprint must reproduce the GTDR posts. This is what stops the global surface
    drifting.

    Only posts on the `gtdr_stride_px` lattice are compared, so the loss cannot reward
    fitting the artefacts of whatever upsampling produced `gtdr` in between.

    **Tiles must carry a context margin.** The footprint is ~10 x 20 km; a 512 px tile at
    75 m is 38 km. Convolving with a kernel comparable to the tile means the padding at
    the tile boundary propagates ~3 sigma inward — roughly 25 km — so on a bare 512 px
    tile essentially every post is contaminated by the boundary condition rather than by
    the terrain. Two things follow, and both are required:

      1. cut tiles with a context margin (predict on the interior, convolve over the
         full extent), and
      2. set `edge_margin_px` to drop posts within the contaminated band.

    Note that comparing `blur(z_hat)` against `blur(gtdr)` instead would cancel the edge
    effect but is *wrong*: GTDR already is the footprint-averaged surface, so blurring it
    again anchors `z_hat` to a doubly smoothed target and biases the whole model smooth.

    `edge_margin_px = 0` (the default) is correct only when `gtdr` was produced with the
    same padded operator on the same extent — which is the case for `data.synthetic`, and
    is why Phase 0 can use it as-is.
    """
    smoothed = physics.footprint_blur(z_hat, footprint, pixel_size)
    s = gtdr_stride_px
    off = s // 2
    zp = smoothed[..., off::s, off::s]
    gp = gtdr[..., off::s, off::s]
    vp = valid[..., off::s, off::s]

    if edge_margin_px > 0:
        m = max(1, edge_margin_px // s)
        keep = torch.zeros_like(vp, dtype=torch.bool)
        if zp.shape[-2] > 2 * m and zp.shape[-1] > 2 * m:
            keep[..., m:-m, m:-m] = True
        vp = vp & keep

    return _masked_mean((zp - gp).abs(), vp) / scales.alt_m


# --------------------------------------------------------------------------------------
# 5.3 Radarclinometry
# --------------------------------------------------------------------------------------
def loss_phys(
    z_hat: Tensor,
    rv_obs: Tensor,
    valid_obs: Tensor,
    look_vec: Tensor,
    theta_nominal: Tensor,
    pixel_size: float,
    brightness: Tensor | None = None,
    huber_delta: float = 1.0,
    pyramid_levels: int = 2,
    layover_margin_deg: float = 5.0,
    scales: LossScales = SCALES,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Huber on rendered-minus-observed RV, at full resolution and on a 4x pyramid.

    The coarse level is what stabilises early training: at 75 m the residual is dominated
    by speckle, and the gradient toward large-scale shape is buried in it.

    The pyramid averages the *residual image*, not the DEM. Rendering from a downsampled
    DEM would be cheaper by one render, but biased: RV is nonlinear in slope, so the
    render of a block-mean slope is not the block mean of the render, and the gap grows
    with subgrid roughness. That bias would land as a systematic roughness error in
    exactly the terrain (tessera) where it is hardest to spot. Averaging the residual
    instead is unbiased and kills the speckle just as well.

    Returns `(loss, diagnostics)`; diagnostics carry the mean absolute dB residual and
    the fraction of pixels surviving the layover mask, both worth logging every step.
    """
    r = physics.render_rv(
        z_hat, look_vec, theta_nominal, pixel_size, brightness=brightness,
        layover_margin_deg=layover_margin_deg,
    )
    ok = valid_obs & r["valid"]
    resid_full = (r["rv_db"] - rv_obs) * ok.to(z_hat.dtype)

    diag: dict[str, Tensor] = {
        "phys_resid_db": _masked_mean(resid_full.abs(), ok).detach(),
        "phys_valid_frac": ok.to(z_hat.dtype).mean().detach(),
    }
    total = z_hat.new_zeros(())
    for level in range(max(1, pyramid_levels)):
        f = 4**level
        if f == 1:
            resid, m = resid_full, ok
        else:
            w = physics.gaussian_downsample(ok.to(z_hat.dtype), f)
            resid = physics.gaussian_downsample(resid_full, f) / w.clamp_min(_EPS)
            m = w > 0.9
        total = total + _masked_mean(
            F.huber_loss(resid, torch.zeros_like(resid), reduction="none", delta=huber_delta), m
        )
    return total / (max(1, pyramid_levels) * scales.phys_db), diag


# --------------------------------------------------------------------------------------
# 5.4 Cross-look
# --------------------------------------------------------------------------------------
def loss_cross(
    z_hat_left_only: Tensor,
    rv_obs_other: Tensor,
    valid_other: Tensor,
    look_vec_other: Tensor,
    theta_other: Tensor,
    pixel_size: float,
    brightness: Tensor | None = None,
    **kwargs,
) -> Tensor:
    """Render the *other* look from a prediction made without it.

    This is the term that trains the left-look-only pathway, which is what 80% of the
    planet will run through at inference, and it is the closest thing to a free stereo
    constraint the dataset contains.
    """
    loss, _ = loss_phys(
        z_hat_left_only, rv_obs_other, valid_other, look_vec_other, theta_other,
        pixel_size, brightness=brightness, **kwargs,
    )
    return loss


# --------------------------------------------------------------------------------------
# 5.5 RMS slope
# --------------------------------------------------------------------------------------
def loss_rms(
    z_hat: Tensor, gsdr_rad: Tensor, valid: Tensor, pixel_size: float, cell_m: float = 4641.0,
    scales: LossScales = SCALES,
) -> Tensor:
    """Keep the predicted roughness in the right ballpark. Weak by design: it is a
    guard against uniformly too-smooth or too-rough terrain, not a target."""
    cell_px = max(1, int(round(cell_m / pixel_size)))
    pred = physics.rms_slope(z_hat, pixel_size, cell_px)
    tgt = F.adaptive_avg_pool2d(gsdr_rad, pred.shape[-2:])
    v = F.adaptive_max_pool2d(valid.to(z_hat.dtype), pred.shape[-2:]) > 0.5
    return _masked_mean((pred - tgt).abs(), v) / scales.rms_rad


# --------------------------------------------------------------------------------------
# 5.6 Uncertainty
# --------------------------------------------------------------------------------------
def loss_nll(
    z_hat: Tensor,
    logvar: Tensor,
    target: Tensor,
    valid: Tensor,
    unsupervised_mask: Tensor | None = None,
    unsupervised_weight: float = 0.5,
    trusted_scale_m: float = 1000.0,
    pixel_size: float = 75.0,
) -> Tensor:
    """Gaussian NLL against the stereo DEM at its trusted scale, plus a hinge that stops
    the model claiming confidence where nothing supervises it.

    The uncertainty map is the part of ISHTAR that makes the 75 m product honest, so the
    two failure modes matter equally: overconfidence where there is data (the NLL term
    handles that) and overconfidence where there is none.

    The obvious second term — subtract the mean `logvar` over unsupervised pixels — has an
    unbounded incentive to inflate sigma, and saturates against `logvar_range` regardless
    of what the model actually knows. This is a hinge against the *supervised* sigma
    instead: where nothing constrains the prediction, the model may not claim to be more
    confident than it is where something does. It says nothing at all once that bar is
    met, so it cannot manufacture uncertainty.
    """
    factor = max(1, int(round(trusted_scale_m / pixel_size)))
    zp = physics.gaussian_downsample(z_hat, factor)
    # Average the variance, not the log of it: the coarse cell's predicted spread is the
    # mean of the fine-scale variances, and averaging logs would report the geometric
    # mean instead — systematically overconfident wherever the field is heterogeneous.
    lv = torch.log(physics.gaussian_downsample(torch.exp(logvar), factor).clamp_min(_EPS))
    zt = physics.gaussian_downsample(target * valid.to(z_hat.dtype), factor)
    w = physics.gaussian_downsample(valid.to(z_hat.dtype), factor)
    keep = w > 0.5
    zt = zt / w.clamp_min(_EPS)

    nll = 0.5 * (lv + (zp - zt).pow(2) / torch.exp(lv).clamp_min(_EPS))
    loss = _masked_mean(nll, keep)

    if unsupervised_mask is not None:
        floor = _masked_mean(logvar, valid).detach()
        loss = loss + unsupervised_weight * _masked_mean(
            F.relu(floor - logvar), unsupervised_mask
        )
    return loss


# --------------------------------------------------------------------------------------
# 5.7 Regularisers
# --------------------------------------------------------------------------------------
def loss_reg(brightness_lr: Tensor, z_hat: Tensor, pixel_size: float) -> Tensor:
    """Smoothness on `b(x)` so it cannot absorb slope information, plus curvature TV on
    `z_hat` so the residual head does not paint high-frequency noise."""
    b = brightness_lr
    b_tv = (b[..., 1:, :] - b[..., :-1, :]).abs().mean() + (b[..., :, 1:] - b[..., :, :-1]).abs().mean()

    # Curvature in metres of relief per pixel-cell, scaled so it is comparable to the
    # brightness TV term rather than to a physical 1/m^2 curvature.
    lap = (
        z_hat[..., 2:, 1:-1] + z_hat[..., :-2, 1:-1] + z_hat[..., 1:-1, 2:] + z_hat[..., 1:-1, :-2]
        - 4.0 * z_hat[..., 1:-1, 1:-1]
    )
    return b_tv + 1e-3 * lap.abs().mean()


# --------------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------------
def total_loss(terms: dict[str, Tensor], weights: LossWeights, phys_ramp: float = 1.0) -> Tensor:
    """Weighted sum. `phys_ramp` in [0, 1] scales the two radar-physics terms together;
    the trainer ramps it over the first 5k Venus steps."""
    w = weights
    scale = {
        "earth": w.earth, "stereo": w.stereo, "alt": w.alt,
        "phys": w.phys * phys_ramp, "cross": w.cross * phys_ramp,
        "rms": w.rms, "nll": w.nll, "reg": w.reg,
    }
    out = None
    for k, v in terms.items():
        if k not in scale:
            continue
        contrib = scale[k] * v
        out = contrib if out is None else out + contrib
    return out if out is not None else torch.zeros(())
