"""A self-consistent synthetic Venus, for Phase 0 and for CI.

Real ISHTAR training needs ~300 GB of Magellan mosaics. Everything upstream of that
download — the renderer, the losses, the sign conventions, the augmentation, the
training loop, the metrics — can and should be validated on terrain we generated
ourselves, because here we know the answer exactly.

The generator makes fractal (1/f^beta) terrain, renders it through the *same*
`model.physics.render_rv` the loss uses, then degrades the result the way Magellan
degrades it: multiplicative speckle, a slowly varying brightness field standing in for
roughness and dielectric variation, low-frequency gain striping, and 8-bit DN
quantisation. It also produces a footprint-blurred, decimated GTDR and a coarse
"stereo" DEM so every loss term has something to consume.

A model that cannot fit this cannot fit Venus, and the failure is much cheaper to find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from model import physics

from . import geometry
from .tile import upsample_posts


@dataclass
class SyntheticConfig:
    size: int = 256
    pixel_size_m: float = 75.0
    beta: float = 3.2               # 2-D PSD exponent; ~3.2 matches terrestrial volcanic terrain
    rms_slope_deg: tuple[float, float] = (1.5, 6.0)
    """Target RMS slope at the native posting, sampled per tile.

    Amplitude is set from this rather than from an elevation std, because RMS slope is
    what actually controls the radar physics (and it is a real Magellan observable:
    plains sit near 1-3 deg, tessera near 5-10 deg). Fixing an elevation std instead
    silently produces 50-deg terrain that is pure layover and teaches the model nothing.
    """
    regional_relief_m: tuple[float, float] = (100.0, 1200.0)
    """Peak-to-peak of an extra very-smooth (beta=4.5) component, sampled per tile.

    The slope-normalised fractal above is nearly flat at tile scale, so without this
    there is no long-wavelength signal for the altimetry loss to anchor and the GTDR
    channel carries no information.
    """
    looks: int = 6                  # speckle looks (gamma shape)
    brightness_sigma_db: float = 1.2
    brightness_scale_px: float = 64.0
    stripe_amplitude_db: float = 0.8
    gtdr_stride_px: int = 62        # 4641 m / 75 m
    stereo_scale_m: float = 1000.0
    stereo_noise_m: float = 60.0
    stereo_coverage: float = 0.5    # fraction of tiles that carry a stereo DEM
    second_look_coverage: float = 0.35


def fractal_field(size: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean unit-variance 1/f^beta field via spectral synthesis."""
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    f = np.sqrt(fy**2 + fx**2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    spec = amp * np.exp(1j * phase)
    field = np.real(np.fft.ifft2(spec))
    return (field - field.mean()) / (field.std() + 1e-8)


def _smooth_noise(size: int, scale_px: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise field, used for the intrinsic-brightness nuisance term."""
    small = max(2, int(size / max(1.0, scale_px)))
    coarse = rng.normal(size=(small, small))
    t = torch.from_numpy(coarse).float()[None, None]
    up = torch.nn.functional.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    a = up[0, 0].numpy()
    return (a - a.mean()) / (a.std() + 1e-8)


def make_tile(cfg: SyntheticConfig, rng: np.random.Generator, lat_deg: float | None = None) -> dict[str, np.ndarray]:
    """One synthetic tile with every field the real dataset provides."""
    n = cfg.size
    lat = float(rng.uniform(-60, 60)) if lat_deg is None else lat_deg

    z = fractal_field(n, cfg.beta, rng)
    zt0 = torch.from_numpy(z).float()[None, None]
    de, dn_ = physics.sobel_gradient(zt0, cfg.pixel_size_m)
    unit_slope = float(torch.sqrt((de**2 + dn_**2).mean()).clamp_min(1e-9))
    target = math.tan(math.radians(float(rng.uniform(*cfg.rms_slope_deg))))
    z = (z - z.mean()) * (target / unit_slope)

    regional = fractal_field(n, 4.5, rng)
    regional = regional - regional.mean()
    ptp = float(regional.max() - regional.min())
    z = z + regional * (float(rng.uniform(*cfg.regional_relief_m)) / max(ptp, 1e-6))
    zt = torch.from_numpy(z).float()[None, None]

    tile: dict[str, np.ndarray] = {"z_true": z.astype(np.float32), "lat_deg": np.float32(lat)}

    brightness = cfg.brightness_sigma_db * _smooth_noise(n, cfg.brightness_scale_px, rng)
    stripe = cfg.stripe_amplitude_db * np.sin(
        2 * np.pi * np.arange(n)[None, :] / rng.uniform(40, 200) + rng.uniform(0, 6.28)
    )
    b_total = torch.from_numpy((brightness + stripe).astype(np.float32))[None, None]
    tile["brightness_true_db"] = (brightness + stripe).astype(np.float32)

    have = {"left": True,
            "right": rng.random() < cfg.second_look_coverage,
            "stereo": rng.random() < cfg.second_look_coverage}
    for look in ("left", "right", "stereo"):
        theta = float(geometry.INCIDENCE_MODELS[look].theta_rad(lat))
        lv = torch.from_numpy(geometry.look_vector(look))[None]
        r = physics.render_rv(zt, lv, torch.tensor([theta]), cfg.pixel_size_m, brightness=b_total)
        rv = r["rv_db"][0, 0].numpy()

        if have[look]:
            # Multiplicative speckle in linear power, then back to dB.
            lin = 10 ** (rv / 10.0)
            lin = lin * rng.gamma(cfg.looks, 1.0 / cfg.looks, size=lin.shape)
            rv_obs = 10 * np.log10(np.maximum(lin, 1e-6))
            dn = physics.dn_from_rv(torch.from_numpy(rv_obs).float()).numpy()
            # Layover pixels are where the real mosaic goes to mush; drop a few outright.
            dn = np.where(~r["valid"][0, 0].numpy() & (rng.random(dn.shape) < 0.5), 0, dn)
        else:
            dn = np.zeros((n, n), dtype=np.float32)

        tile[f"dn_{look}"] = dn.astype(np.float32)
        tile[f"theta_{look}"] = np.float32(theta)
        tile[f"look_{look}"] = geometry.look_vector(look)
        tile[f"has_{look}"] = np.float32(have[look])

    # GTDR: footprint-blurred, decimated, quantised to integer metres, then held at the
    # posts (the real product is a 4641 m grid we bicubically upsample).
    blur = physics.footprint_blur(zt, physics.FootprintSpec(), cfg.pixel_size_m)
    s = cfg.gtdr_stride_px
    posts = blur[..., s // 2 :: s, s // 2 :: s]
    posts = torch.round(posts + torch.from_numpy(rng.normal(0, 20.0, posts.shape).astype(np.float32)))
    # Placed on the same lattice the altimetry loss samples — see tile.upsample_posts.
    up = upsample_posts(posts, (n, n), s)
    tile["gtdr_up"] = up.numpy().astype(np.float32)
    tile["gtdr_valid"] = np.ones((n, n), dtype=np.float32)
    tile["gtdr_posts"] = posts[0, 0].numpy().astype(np.float32)

    # Stereo DEM: trusted only at ~1 km, with noise and a seam artefact.
    if rng.random() < cfg.stereo_coverage:
        f = max(1, int(round(cfg.stereo_scale_m / cfg.pixel_size_m)))
        coarse = physics.gaussian_downsample(zt, f)
        coarse = coarse + torch.from_numpy(rng.normal(0, cfg.stereo_noise_m, coarse.shape).astype(np.float32))
        st = torch.nn.functional.interpolate(coarse, size=(n, n), mode="bilinear", align_corners=False)
        valid = np.ones((n, n), dtype=np.float32)
        col = rng.integers(0, n)  # a mosaic seam ("noodle") the artefact mask must catch
        lo, hi = max(0, col - 3), min(n, col + 3)
        st[..., :, lo:hi] += float(rng.normal(0, 250))
        valid[:, lo:hi] = 0.0
        tile["stereo_dem"] = st[0, 0].numpy().astype(np.float32)
        tile["stereo_valid"] = valid
        tile["has_stereo_dem"] = np.float32(1.0)
    else:
        tile["stereo_dem"] = np.zeros((n, n), dtype=np.float32)
        tile["stereo_valid"] = np.zeros((n, n), dtype=np.float32)
        tile["has_stereo_dem"] = np.float32(0.0)

    # GSDR-like RMS slope at 4.6 km, and a stand-in emissivity channel.
    cell = max(1, int(round(4641.0 / cfg.pixel_size_m)))
    tile["rms_slope"] = physics.rms_slope(zt, cfg.pixel_size_m, cell)[0, 0].numpy().astype(np.float32)
    tile["emissivity"] = (0.85 + 0.02 * _smooth_noise(n, 96.0, rng)).astype(np.float32)
    return tile


class SyntheticVenus(torch.utils.data.Dataset):
    """A fixed-size, deterministic synthetic set. Tiles are cached so Phase 0 really is
    an overfitting test rather than an infinite-data test."""

    def __init__(self, n_tiles: int = 200, cfg: SyntheticConfig | None = None, seed: int = 0):
        self.cfg = cfg or SyntheticConfig()
        self.seed = seed
        self.n_tiles = n_tiles
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.n_tiles

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx not in self._cache:
            self._cache[idx] = make_tile(self.cfg, np.random.default_rng(self.seed * 100003 + idx))
        t = self._cache[idx]
        return {k: torch.as_tensor(v) for k, v in t.items()}
