"""Conditional residual U-Net with a heteroscedastic uncertainty head.

The network never predicts absolute elevation. It predicts a residual over the
bicubically upsampled GTDR altimetry, so the low frequencies are correct by
construction and the capacity goes into the 100 m - 10 km band that Magellan
altimetry cannot see.

Three heads:
  h_res  : residual elevation, metres, tanh-bounded
  h_logv : log variance of elevation (heteroscedastic uncertainty)
  h_b    : intrinsic brightness nuisance field at 1/16 resolution, dB

Incidence angle and look direction are fed twice: as input channels, and as FiLM
conditioning at every decoder stage. They are near-global variables per tile and the
network learns them far too slowly from image channels alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Input channel layout. Kept as an explicit list so the dataset, the augmentation code
# and the model cannot silently disagree about ordering.
INPUT_CHANNELS: tuple[str, ...] = (
    "sar_left_db",
    "mask_left",
    "sar_right_db",
    "mask_right",
    "sar_stereo_db",
    "mask_stereo",
    "gtdr_up",
    "emissivity",
    "theta_left_sin",
    "theta_left_cos",
    "theta_right_sin",
    "theta_right_cos",
    "theta_stereo_sin",
    "theta_stereo_cos",
    "look_east",
    "look_north",
    "lat_sin",
    "lat_cos",
)

# Global conditioning vector fed to FiLM (per tile, not per pixel).
COND_FEATURES: tuple[str, ...] = (
    "theta_left_sin",
    "theta_left_cos",
    "theta_right_sin",
    "theta_right_cos",
    "theta_stereo_sin",
    "theta_stereo_cos",
    "look_east",
    "look_north",
    "has_right",
    "has_stereo",
)


@dataclass
class UNetConfig:
    in_channels: int = len(INPUT_CHANNELS)
    cond_dim: int = len(COND_FEATURES)
    widths: tuple[int, ...] = (64, 128, 256, 512, 512)
    depths: tuple[int, ...] = (2, 2, 3, 3, 3)
    decoder_width: int = 192
    bottleneck_dilations: tuple[int, ...] = (1, 2, 4, 8)
    residual_scale_m: float = 1500.0
    brightness_downscale: int = 16
    logvar_range: tuple[float, float] = (-2.0, 12.0)  # ~0.37 m to ~400 m of sigma


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise 7x7, GroupNorm, pointwise expand/contract, GELU,
    layer scale, residual. Cheap receptive field growth without attention."""

    def __init__(self, dim: int, expansion: int = 4, layer_scale: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, 1)
        self.pw2 = nn.Conv2d(dim * expansion, dim, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        h = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
        return x + self.gamma * h


class FiLM(nn.Module):
    """Feature-wise linear modulation from the global geometry vector."""

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.Linear(cond_dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 2 * dim)
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        return x * (1.0 + scale[..., None, None]) + shift[..., None, None]


class Encoder(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.widths[0], 3, padding=1),
            nn.GroupNorm(1, cfg.widths[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, (w, d) in enumerate(zip(cfg.widths, cfg.depths)):
            if i > 0:
                self.downs.append(
                    nn.Sequential(nn.GroupNorm(1, cfg.widths[i - 1]), nn.Conv2d(cfg.widths[i - 1], w, 2, stride=2))
                )
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(w) for _ in range(d)]))

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = []
        h = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                h = self.downs[i - 1](h)
            h = stage(h)
            feats.append(h)
        return feats


class DilatedBottleneck(nn.Module):
    """Widens the receptive field past the 20 km altimeter footprint (~270 px at 75 m).

    A 5-level encoder alone tops out below that, and the altimetry loss then cannot be
    satisfied by anything the decoder can see.
    """

    def __init__(self, dim: int, dilations: tuple[int, ...]):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(1, dim),
                    nn.Conv2d(dim, dim, 3, padding=d, dilation=d),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, 1),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for b in self.blocks:
            x = x + b(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, cfg: UNetConfig, in_dim: int, skip_dim: int, out_dim: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim + skip_dim, out_dim, 1)
        self.block = ConvNeXtBlock(out_dim)
        self.film = FiLM(cfg.cond_dim, out_dim)

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(torch.cat([x, skip], dim=1))
        return self.block(self.film(x, cond))


class IshtarUNet(nn.Module):
    def __init__(self, cfg: UNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or UNetConfig()
        c = self.cfg
        self.encoder = Encoder(c)
        self.bottleneck = DilatedBottleneck(c.widths[-1], c.bottleneck_dilations)
        self.bottleneck_film = FiLM(c.cond_dim, c.widths[-1])

        dims = list(c.widths)
        self.decoder = nn.ModuleList()
        in_dim = dims[-1]
        for i in range(len(dims) - 2, -1, -1):
            self.decoder.append(DecoderStage(c, in_dim, dims[i], c.decoder_width))
            in_dim = c.decoder_width

        self.head_res = nn.Sequential(
            nn.GroupNorm(1, c.decoder_width), nn.Conv2d(c.decoder_width, c.decoder_width, 3, padding=1),
            nn.GELU(), nn.Conv2d(c.decoder_width, 1, 1),
        )
        self.head_logv = nn.Sequential(
            nn.GroupNorm(1, c.decoder_width), nn.Conv2d(c.decoder_width, 64, 3, padding=1),
            nn.GELU(), nn.Conv2d(64, 1, 1),
        )
        # Brightness rides off the bottleneck, so it is structurally low-resolution and
        # cannot absorb the slope signal even before the smoothness penalty.
        self.head_b = nn.Sequential(
            nn.GroupNorm(1, c.widths[-1]), nn.Conv2d(c.widths[-1], 64, 3, padding=1),
            nn.GELU(), nn.Conv2d(64, 1, 1),
        )
        nn.init.zeros_(self.head_res[-1].weight)
        nn.init.zeros_(self.head_res[-1].bias)
        nn.init.zeros_(self.head_b[-1].weight)
        nn.init.zeros_(self.head_b[-1].bias)

    def forward(self, x: Tensor, cond: Tensor, gtdr_up: Tensor) -> dict[str, Tensor]:
        """Args:
            x: `(B, C, H, W)` stacked input channels in `INPUT_CHANNELS` order.
            cond: `(B, cond_dim)` global geometry vector in `COND_FEATURES` order.
            gtdr_up: `(B, 1, H, W)` upsampled altimetry in metres — the base the
                residual is added to.
        Returns `z_hat` (m), `residual` (m), `logvar`, `sigma` (m), `brightness` (dB),
        and `brightness_lr` (the raw low-resolution field, for the smoothness penalty).
        """
        feats = self.encoder(x)
        h = self.bottleneck_film(self.bottleneck(feats[-1]), cond)

        b_lr = self.head_b(h)
        target_lr = max(1, x.shape[-1] // self.cfg.brightness_downscale)
        if b_lr.shape[-1] != target_lr:
            b_lr = F.interpolate(b_lr, size=(target_lr, target_lr), mode="bilinear", align_corners=False)

        for i, stage in enumerate(self.decoder):
            skip = feats[len(feats) - 2 - i]
            h = stage(h, skip, cond)

        residual = torch.tanh(self.head_res(h)) * self.cfg.residual_scale_m
        lo, hi = self.cfg.logvar_range
        logvar = lo + (hi - lo) * torch.sigmoid(self.head_logv(h))
        brightness = F.interpolate(b_lr, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "z_hat": gtdr_up + residual,
            "residual": residual,
            "logvar": logvar,
            "sigma": torch.exp(0.5 * logvar),
            "brightness": brightness,
            "brightness_lr": b_lr,
        }


def build_model(cfg: UNetConfig | None = None) -> IshtarUNet:
    return IshtarUNet(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
