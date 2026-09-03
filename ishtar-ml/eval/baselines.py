"""The three baselines of Section 7. A model that does not beat (a) is not a model.

(a) Bicubic GTDR      -- the input the network predicts a residual over. Free.
(b) Classical radarclinometry with a fixed brightness -- what you get from the physics
    alone, with no learned prior. This is the honest "is the network doing anything"
    control, and it is the one people will ask about.
(c) Earth-only, no Venus fine-tune -- isolates how much of the result is transfer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from model import physics


def bicubic_gtdr(gtdr_up: Tensor) -> Tensor:
    """Baseline (a): the upsampled altimetry, unchanged."""
    return gtdr_up


def classical_radarclinometry(
    rv_obs: Tensor,
    valid: Tensor,
    look_vec: Tensor,
    theta_nominal: Tensor,
    gtdr_up: Tensor,
    pixel_size: float,
    steps: int = 300,
    lr: float = 20.0,
    smooth_weight: float = 0.02,
    footprint: physics.FootprintSpec | None = None,
    alt_weight: float = 1.0,
) -> Tensor:
    """Baseline (b): invert the same forward model by gradient descent on the DEM alone.

    No network, no learned prior, and a constant brightness field — so the cross-track
    slope is unconstrained and only the altimetry anchor and a smoothness prior keep it
    from wandering. That is exactly the weakness the learned prior is meant to fix, which
    is why this is the baseline that matters.

    Optimises the residual over `gtdr_up`, like the network does, so the two are
    comparable rather than differing by their starting point.
    """
    fp = footprint or physics.FootprintSpec()
    residual = torch.zeros_like(gtdr_up, requires_grad=True)
    brightness = torch.zeros(gtdr_up.shape[0], 1, 1, 1, device=gtdr_up.device,
                             dtype=gtdr_up.dtype, requires_grad=True)
    opt = torch.optim.Adam([residual, brightness], lr=lr)

    for _ in range(steps):
        z = gtdr_up + residual
        r = physics.render_rv(z, look_vec, theta_nominal, pixel_size, brightness=brightness)
        m = (valid & r["valid"]).to(z.dtype)
        data = (((r["rv_db"] - rv_obs) ** 2) * m).sum() / m.sum().clamp_min(1.0)

        de, dn = physics.sobel_gradient(residual, pixel_size)
        smooth = (de.pow(2) + dn.pow(2)).mean() * pixel_size**2

        anchor = physics.footprint_blur(residual, fp, pixel_size).abs().mean()

        opt.zero_grad(set_to_none=True)
        (data + smooth_weight * smooth + alt_weight * anchor).backward()
        opt.step()

    return (gtdr_up + residual).detach()


def earth_only(model_earth, batch: dict[str, Tensor]) -> Tensor:
    """Baseline (c): the Phase 1 checkpoint, run on Venus with no fine-tuning."""
    with torch.no_grad():
        return model_earth(batch["x"], batch["cond"], batch["gtdr_up"])["z_hat"]
