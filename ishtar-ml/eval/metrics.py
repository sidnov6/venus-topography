"""The Section 7 metric table.

A single RMSE is not an honest summary of super-resolution without labels, so every one
of these is reported. Two of them carry most of the weight:

* **Cross-look PSNR** — render the right-look image from a DEM predicted using only the
  left look, and compare with the observed right-look. It is the one metric that tests
  75 m detail against data the model never saw, and it is available over the 17% of the
  planet with two looks.
* **Radially averaged power spectrum** — tells you whether the added high frequencies are
  terrain or noise. A model that hallucinates texture scores well on RMSE and badly here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from model import physics


def _masked(x: Tensor, m: Tensor | None) -> Tensor:
    return x if m is None else x[m]


def mae(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    return float(_masked((pred - target).abs(), mask).mean())


def rmse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    return float(_masked((pred - target).pow(2), mask).mean().sqrt())


def at_scale(x: Tensor, pixel_size: float, scale_m: float) -> Tensor:
    """Bring a field to the scale a product is actually trusted at."""
    return physics.gaussian_downsample(x, max(1, int(round(scale_m / pixel_size))))


def stereo_metrics(pred: Tensor, stereo: Tensor, mask: Tensor, pixel_size: float,
                   scale_m: float = 1000.0) -> dict[str, float]:
    """MAE, RMSE and slope MAE against the stereo DEM, at 1 km where it is trustworthy."""
    p, s = at_scale(pred, pixel_size, scale_m), at_scale(stereo, pixel_size, scale_m)
    w = at_scale(mask.float(), pixel_size, scale_m) > 0.9
    px = pixel_size * max(1, int(round(scale_m / pixel_size)))

    pe, pn = physics.sobel_gradient(p, px)
    se, sn = physics.sobel_gradient(s, px)
    slope_err = torch.rad2deg(
        torch.atan(torch.sqrt(pe**2 + pn**2)) - torch.atan(torch.sqrt(se**2 + sn**2))
    ).abs()
    return {
        "stereo_mae_m": mae(p, s, w),
        "stereo_rmse_m": rmse(p, s, w),
        "stereo_slope_mae_deg": float(_masked(slope_err, w).mean()),
    }


def altimetry_residual(pred: Tensor, gtdr: Tensor, valid: Tensor, pixel_size: float,
                       stride_px: int, footprint: physics.FootprintSpec | None = None,
                       edge_margin_px: int = 0) -> float:
    """Drift check. Section 7 wants this under 30 m."""
    fp = footprint or physics.FootprintSpec()
    blurred = physics.footprint_blur(pred, fp, pixel_size)
    off = stride_px // 2
    p, g, v = blurred[..., off::stride_px, off::stride_px], gtdr[..., off::stride_px, off::stride_px], valid[..., off::stride_px, off::stride_px]
    if edge_margin_px > 0:
        m = max(1, edge_margin_px // stride_px)
        keep = torch.zeros_like(v, dtype=torch.bool)
        if p.shape[-2] > 2 * m and p.shape[-1] > 2 * m:
            keep[..., m:-m, m:-m] = True
        v = v & keep
    return mae(p, g, v)


def physics_residual_db(pred: Tensor, rv_obs: Tensor, valid: Tensor, look_vec: Tensor,
                        theta: Tensor, pixel_size: float, brightness: Tensor | None = None,
                        fit_offset: bool = True) -> float:
    """How much of the image the DEM actually explains, in dB.

    `fit_offset` removes the best constant brightness per tile before measuring. Without
    it the metric is not a fair comparison: the network predicts an intrinsic-brightness
    field and the baselines do not, so part of any advantage it shows would be gain
    matching rather than shape. Removing a scalar leaves the quantity that matters —
    whether the *structure* of the image follows from the DEM — and applies the same
    allowance to every candidate.
    """
    r = physics.render_rv(pred, look_vec, theta, pixel_size, brightness=brightness)
    m = valid & r["valid"]
    resid = r["rv_db"] - rv_obs
    if fit_offset:
        sel = _masked(resid, m)
        if sel.numel():
            resid = resid - sel.mean()
    return float(_masked(resid.abs(), m).mean())


def cross_look_psnr(pred_from_left_only: Tensor, rv_other: Tensor, valid_other: Tensor,
                    look_other: Tensor, theta_other: Tensor, pixel_size: float,
                    brightness: Tensor | None = None, peak_db: float = 50.0,
                    fit_offset: bool = True) -> float:
    """The honest test of 75 m detail: predict a DEM from one look, render the *other*
    look from it, and score against an observation the model never saw.

    Gain-invariant for the same reason as `physics_residual_db` — the question is whether
    the DEM predicts the *structure* of an unseen image, not whether it matched its
    calibration.
    """
    r = physics.render_rv(pred_from_left_only, look_other, theta_other, pixel_size, brightness=brightness)
    m = valid_other & r["valid"]
    resid = r["rv_db"] - rv_other
    if fit_offset:
        sel = _masked(resid, m)
        if sel.numel():
            resid = resid - sel.mean()
    mse = float(_masked(resid.pow(2), m).mean())
    return 10.0 * math.log10(peak_db**2 / max(mse, 1e-9))


def radial_power_spectrum(z: Tensor, pixel_size: float, n_bins: int = 64
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged 2-D power spectrum. Returns `(wavelength_m, power)`.

    Compare the model's curve against the stereo DEM's: matching amplitude at 1-10 km and
    a plausible continuation below it is the signature of real detail. A curve that is
    flat and elevated at short wavelengths is invented texture.
    """
    x = z - z.mean(dim=(-2, -1), keepdim=True)
    n = x.shape[-1]
    win = torch.hann_window(n, periodic=False, device=x.device, dtype=x.dtype)
    x = x * win.view(1, 1, -1, 1) * win.view(1, 1, 1, -1)

    p = torch.fft.fft2(x).abs().pow(2).mean(dim=(0, 1)).cpu().numpy()
    fy = np.fft.fftfreq(x.shape[-2], d=pixel_size)[:, None]
    fx = np.fft.fftfreq(n, d=pixel_size)[None, :]
    f = np.sqrt(fy**2 + fx**2).ravel()
    p = p.ravel()

    keep = f > 0
    f, p = f[keep], p[keep]
    edges = np.logspace(np.log10(f.min()), np.log10(f.max()), n_bins + 1)
    idx = np.digitize(f, edges) - 1
    power = np.array([p[idx == i].mean() if np.any(idx == i) else np.nan for i in range(n_bins)])
    centres = np.sqrt(edges[:-1] * edges[1:])
    return 1.0 / centres, power


def uncertainty_calibration(pred: Tensor, sigma: Tensor, target: Tensor, mask: Tensor
                            ) -> dict[str, float]:
    """Coverage at 1 and 2 sigma. A calibrated heteroscedastic head gives ~68% and ~95%;
    anything far below means the uncertainty map is decorative."""
    err = _masked((pred - target).abs(), mask)
    s = _masked(sigma, mask)
    return {
        "coverage_1sigma": float((err <= s).float().mean()),
        "coverage_2sigma": float((err <= 2 * s).float().mean()),
        "mean_sigma_m": float(s.mean()),
        "rms_error_m": float(err.pow(2).mean().sqrt()),
    }


@dataclass
class Baselines:
    """Section 7's three reference points. A model that does not beat (a) is not a model."""

    bicubic_gtdr: float | None = None
    classical_radarclinometry: float | None = None
    earth_only_no_finetune: float | None = None


def temperature_scale(logvar: Tensor, t: float) -> Tensor:
    """Post-hoc calibration knob fitted in `calibrate.py`: sigma -> t * sigma."""
    return logvar + 2.0 * math.log(max(t, 1e-6))
