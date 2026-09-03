"""Validity masks. The Venus supervision is patchy and partly wrong; these decide which
pixels a loss is allowed to look at."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dilate(mask: Tensor, radius: int) -> Tensor:
    """Binary dilation by a square structuring element."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask.to(torch.float32), k, stride=1, padding=radius) > 0.5


def radar_dark_mask(sar_db: Tensor, valid: Tensor, threshold_db: float = -12.0, dilation: int = 3) -> Tensor:
    """Pixels the stereo DEM cannot be trusted on.

    Radar-dark patches defeat stereo correlation and come back spuriously low in the
    Herrick DEM, so the stereo loss must not chase them. Threshold on the left-look dB
    and dilate, because the correlation window drags the error outward.
    """
    dark = (sar_db < threshold_db) & valid
    return ~dilate(dark, dilation)


def seam_mask(stereo_dem: Tensor, jump_m: float = 150.0, dilation: int = 4) -> Tensor:
    """Mosaic-seam ("noodle") misregistration in the stereo DEM shows up as a
    one-pixel-wide elevation discontinuity. Detect it as an implausible neighbour jump
    and drop a band around it."""
    dx = (stereo_dem[..., :, 1:] - stereo_dem[..., :, :-1]).abs()
    dy = (stereo_dem[..., 1:, :] - stereo_dem[..., :-1, :]).abs()
    bad = torch.zeros_like(stereo_dem, dtype=torch.bool)
    bad[..., :, 1:] |= dx > jump_m
    bad[..., :, :-1] |= dx > jump_m
    bad[..., 1:, :] |= dy > jump_m
    bad[..., :-1, :] |= dy > jump_m
    return ~dilate(bad, dilation)


def stereo_trust_mask(
    stereo_dem: Tensor, stereo_valid: Tensor, sar_db: Tensor, sar_valid: Tensor
) -> Tensor:
    """Everything the stereo loss is allowed to see: shipped validity, minus seams,
    minus radar-dark patches."""
    return stereo_valid.bool() & seam_mask(stereo_dem) & radar_dark_mask(sar_db, sar_valid.bool())


def unsupervised_mask(has_stereo: Tensor, mask_right: Tensor, mask_stereo_look: Tensor) -> Tensor:
    """Pixels with neither a stereo DEM nor a second look: nothing constrains the
    cross-track slope there, and the uncertainty head must say so."""
    return ~(has_stereo.bool() | mask_right.bool() | mask_stereo_look.bool())
