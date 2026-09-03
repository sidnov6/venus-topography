"""Phase 4: calibrate the uncertainty head.

The heteroscedastic head is trained by NLL against a stereo DEM that is itself noisy at
50-100 m, so its raw sigma is not a calibrated 1-sigma of the true error. This fits a
single temperature on held-out validation quadrangles such that 1-sigma coverage lands
near 68%, and reports coverage before and after.

    python calibrate.py --ckpt runs/venus_global/last.pt

Calibrating on the validation quads and reporting on them is circular, so the fit uses
half the quads and the report uses the other half.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import BatchSpec, build_batch
from data.synthetic import SyntheticConfig, SyntheticVenus
from eval.metrics import uncertainty_calibration
from model.unet import UNetConfig, build_model
from train import load_weights, pick_device


def collect(model, loader, device, spec: BatchSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute error and predicted sigma over every supervised pixel."""
    errs, sigmas = [], []
    model.eval()
    with torch.no_grad():
        for tiles in loader:
            tiles = {k: v.to(device) for k, v in tiles.items()}
            b = build_batch(tiles, spec, np.random.default_rng(0))
            out = model(b["x"], b["cond"], b["gtdr_up"])
            target = b["z_true"] if "z_true" in b else b["stereo_dem"]
            mask = torch.ones_like(target, dtype=torch.bool) if "z_true" in b else b["stereo_trust"]
            errs.append((out["z_hat"] - target)[mask].abs().flatten().cpu())
            sigmas.append(out["sigma"][mask].flatten().cpu())
    return torch.cat(errs), torch.cat(sigmas)


def fit_temperature(err: torch.Tensor, sigma: torch.Tensor, target_coverage: float = 0.6827
                    ) -> float:
    """Smallest `t` with `P(err <= t * sigma) >= target`, by bisection.

    Coverage is monotone in `t`, so bisection is exact and needs no gradient. A single
    scalar is the right amount of freedom here: anything richer starts fitting the
    validation set's terrain rather than the model's overconfidence.
    """
    lo, hi = 1e-3, 1e3
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if float((err <= mid * sigma).float().mean()) < target_coverage:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path("runs/sanity/last.pt"))
    ap.add_argument("--tile-size", type=int, default=128)
    ap.add_argument("--n-tiles", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    device = pick_device(a.device)
    model = build_model(UNetConfig()).to(device)
    used = load_weights(a.ckpt, model, device)
    print(f"loaded {used} weights from {a.ckpt}")

    spec = BatchSpec(augment=False)
    scfg = SyntheticConfig(size=a.tile_size)
    # Disjoint seeds stand in for disjoint quadrangles: fit on one half, report on the other.
    fit_dl = DataLoader(SyntheticVenus(a.n_tiles, scfg, seed=4242), batch_size=a.batch_size)
    rep_dl = DataLoader(SyntheticVenus(a.n_tiles, scfg, seed=9911), batch_size=a.batch_size)

    err_f, sig_f = collect(model, fit_dl, device, spec)
    t = fit_temperature(err_f, sig_f)

    err_r, sig_r = collect(model, rep_dl, device, spec)
    before = uncertainty_calibration(err_r, sig_r, torch.zeros_like(err_r), torch.ones_like(err_r, dtype=torch.bool))
    after = uncertainty_calibration(err_r, t * sig_r, torch.zeros_like(err_r), torch.ones_like(err_r, dtype=torch.bool))

    print(f"temperature t = {t:.3f}   (sigma -> {t:.3f} * sigma)")
    print(f"{'':22s}{'before':>10s}{'after':>10s}   target")
    for k, tgt in (("coverage_1sigma", 0.683), ("coverage_2sigma", 0.954)):
        print(f"  {k:20s}{before[k]:10.3f}{after[k]:10.3f}{tgt:9.3f}")
    print(f"  {'mean_sigma_m':20s}{before['mean_sigma_m']:10.2f}{after['mean_sigma_m']:10.2f}")
    print(f"  {'rms_error_m':20s}{before['rms_error_m']:10.2f}{after['rms_error_m']:10.2f}")

    out = a.out or a.ckpt.with_name("calibration.json")
    out.write_text(json.dumps({"temperature": t, "before": before, "after": after}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
