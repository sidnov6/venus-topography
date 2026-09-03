"""Earth pretraining set: Sentinel-1 GRD paired with Copernicus GLO-30, degraded to look
like Magellan.

The point of this stage is that Earth is the only place where a *correct* SAR-to-DEM
label exists. It supplies the prior that no amount of Venus data can: what terrain looks
like in radar at 75 m.

Two rules that are easy to get wrong and fatal if you do:

* **Do not use RTC (radiometrically terrain-corrected) products.** RTC divides out the
  local illuminated area, which is exactly the slope signal the network is here to learn.
  A model trained on RTC learns that brightness is uninformative — the one thing that
  must not transfer.
* **Pick unvegetated terrain.** Volume scattering from canopy breaks the
  backscatter-slope relation outright, and Venus has no vegetation to teach it about.

`sentinelsat` / `pystac_client` / `rasterio` are imported lazily; this module is not
needed to run the model or the tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from model import physics

# Section 2.2: unvegetated volcanic and tectonic terrain, spanning the roughness regimes
# Venus actually presents — young lava plains, tessera-like deformed terrain, big shields.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    # name: (west, south, east, north)
    "iceland": (-24.5, 63.3, -13.5, 66.6),
    "hawaii": (-156.1, 18.9, -154.8, 20.3),
    "afar": (39.0, 8.5, 42.5, 14.5),
    "atacama": (-70.5, -26.5, -67.0, -22.0),
    "tibet": (80.0, 30.0, 92.0, 36.0),
    "kamchatka": (156.0, 51.0, 162.0, 57.0),
    "canaries": (-18.2, 27.6, -13.4, 29.5),
    "ethiopian_rift": (37.0, 6.0, 40.5, 9.5),
    "nevada": (-119.5, 36.5, -114.0, 41.5),
}


@dataclass
class DegradeConfig:
    """Make Sentinel-1 look like a Magellan FMAP tile."""

    target_res_m: float = 75.0
    looks: float = 6.0
    """Speckle looks. Magellan F-BIDRs were roughly 4-8 look after mosaicking."""

    stripe_amplitude_db: float = 0.8
    stripe_wavelength_px: tuple[float, float] = (40.0, 200.0)
    """Low-frequency gain striping, imitating orbit-to-orbit calibration differences in
    the mosaic. Without it the network learns to trust absolute brightness."""

    gain_jitter_db: float = 1.5
    quantise_to_dn: bool = True
    """Round-trip through the 8-bit DN encoding, so the network sees Magellan's 0.2 dB
    quantisation step rather than float radiometry."""


def flatten_to_muhleman(sigma0_db: torch.Tensor, incidence_rad: torch.Tensor) -> torch.Tensor:
    """Convert calibrated sigma0 (dB) to the Muhleman-flattened `RV` that FMAP encodes.

    This is the step that puts Earth and Venus in the same units. Sentinel-1 gives
    sigma0; Magellan gives sigma0 divided by the Muhleman prediction for the nominal
    incidence. Flatten with the *true* per-pixel incidence, which for a GRD product is
    the near-to-far range ramp across the swath.
    """
    ref = 10.0 * torch.log10(physics.muhleman_sigma0(incidence_rad))
    return sigma0_db - ref


def degrade(
    rv_db: torch.Tensor, cfg: DegradeConfig, rng: np.random.Generator
) -> torch.Tensor:
    """Speckle, stripe, jitter and quantise a flattened Earth tile into Magellan's units."""
    from .augment import gain_offset, speckle

    out = speckle(rv_db, cfg.looks, rng)

    n = out.shape[-1]
    wl = float(rng.uniform(*cfg.stripe_wavelength_px))
    phase = float(rng.uniform(0, 2 * np.pi))
    x = torch.arange(n, dtype=out.dtype, device=out.device)
    stripe = cfg.stripe_amplitude_db * torch.sin(2 * np.pi * x / wl + phase)
    out = out + stripe.view(1, 1, 1, -1)

    out = gain_offset(out, cfg.gain_jitter_db, rng)
    if cfg.quantise_to_dn:
        out, _ = physics.rv_from_dn(physics.dn_from_rv(out))
    return out


def resample_to_75m(x: torch.Tensor, src_res_m: float, cfg: DegradeConfig) -> torch.Tensor:
    """Area-average Sentinel-1 (10-20 m) or GLO-30 (30 m) onto the Magellan grid."""
    factor = max(1, int(round(cfg.target_res_m / src_res_m)))
    return physics.gaussian_downsample(x, factor)


STAC_ENDPOINTS = {
    "sentinel1": "https://earth-search.aws.element84.com/v1",
    "copernicus_dem": "https://earth-search.aws.element84.com/v1",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", nargs="*", default=sorted(REGIONS))
    ap.add_argument("--tiles-per-region", type=int, default=4000)
    ap.add_argument("--out", default="data_tiles/earth.zarr")
    a = ap.parse_args()

    total = len(a.regions) * a.tiles_per_region
    print(f"target: {total} tiles of 512 px at 75 m over {len(a.regions)} regions")
    print("regions:", ", ".join(a.regions))
    print("\nProduct requirements:")
    print("  SAR : Sentinel-1 GRD IW, VV, NOT terrain-corrected (RTC removes the signal)")
    print("  DEM : Copernicus GLO-30, resampled to 75 m")
    print("  keep the per-pixel incidence angle band; the model is conditioned on it")
    raise SystemExit(
        f"Staging from {STAC_ENDPOINTS['sentinel1']} needs pystac_client + rasterio and a "
        "few hundred GB of scratch. `flatten_to_muhleman` and `degrade` are the parts "
        "specific to this project and are testable without any download."
    )


if __name__ == "__main__":
    # Run as a module (`python -m data.earth`), not as a script: this file uses
    # package-relative imports, which are unavailable when Python treats it as __main__
    # in its own directory.
    main()
