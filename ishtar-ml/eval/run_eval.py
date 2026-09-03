"""Produce the Section 7 metric table.

    python -m eval.run_eval --ckpt runs/sanity/last.pt --n-tiles 32

Reports the model against all three baselines on held-out tiles. Every metric in the
architecture note is here, because on a super-resolution-without-labels problem a single
RMSE is not a summary — it is a way of not looking at the failure modes.

On the synthetic set the true DEM is known, so this also reports the metrics you will
never have on Venus (`mae_vs_truth`). Those are the ones that tell you whether the
proxy metrics you *will* have are worth trusting.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import BatchSpec, build_batch, drop_second_looks  # noqa: E402
from data.synthetic import SyntheticConfig, SyntheticVenus  # noqa: E402
from eval import baselines, metrics  # noqa: E402
from model import physics  # noqa: E402
from model.unet import UNetConfig, build_model  # noqa: E402
from train import load_weights, pick_device  # noqa: E402


@dataclass
class Row:
    name: str
    mae_vs_truth: float = float("nan")
    rmse_vs_truth: float = float("nan")
    slope_mae_deg: float = float("nan")
    stereo_mae_m: float = float("nan")
    alt_resid_m: float = float("nan")
    phys_resid_db: float = float("nan")
    cross_look_psnr: float = float("nan")
    cov_1sigma: float = float("nan")


def evaluate_dem(z: torch.Tensor, b: dict[str, torch.Tensor], px: int, spec: BatchSpec,
                 sigma: torch.Tensor | None = None, brightness: torch.Tensor | None = None,
                 ) -> dict[str, float]:
    out: dict[str, float] = {}
    if "z_true" in b:
        truth = b["z_true"]
        out["mae_vs_truth"] = metrics.mae(z, truth)
        out["rmse_vs_truth"] = metrics.rmse(z, truth)
        pe, pn = physics.sobel_gradient(z, spec.pixel_size_m)
        te, tn = physics.sobel_gradient(truth, spec.pixel_size_m)
        out["slope_mae_deg"] = float(
            torch.rad2deg(
                torch.atan(torch.sqrt(pe**2 + pn**2)) - torch.atan(torch.sqrt(te**2 + tn**2))
            ).abs().mean()
        )
    if b["stereo_trust"].any():
        out.update({k: v for k, v in metrics.stereo_metrics(
            z, b["stereo_dem"], b["stereo_trust"], spec.pixel_size_m).items()
            if k == "stereo_mae_m"})
    out["alt_resid_m"] = metrics.altimetry_residual(
        z, b["gtdr_up"], b["gtdr_valid"], spec.pixel_size_m, spec.gtdr_stride_px,
        edge_margin_px=spec.alt_edge_margin_px,
    )
    out["phys_resid_db"] = metrics.physics_residual_db(
        z, b["rv_left"], b["valid_left"], b["look_left"], b["theta_left"],
        spec.pixel_size_m, brightness=brightness,
    )
    if sigma is not None and "z_true" in b:
        cal = metrics.uncertainty_calibration(
            z, sigma, b["z_true"], torch.ones_like(z, dtype=torch.bool))
        out["cov_1sigma"] = cal["coverage_1sigma"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path("runs/sanity/last.pt"))
    ap.add_argument("--tile-size", type=int, default=128)
    ap.add_argument("--n-tiles", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-classical", action="store_true",
                    help="the radarclinometry baseline runs 300 optimiser steps per batch")
    ap.add_argument("--classical-steps", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval.json"))
    a = ap.parse_args()

    device = pick_device(a.device)
    spec = BatchSpec(augment=False)
    ds = SyntheticVenus(a.n_tiles, SyntheticConfig(size=a.tile_size), seed=9911)
    dl = DataLoader(ds, batch_size=a.batch_size)

    model = build_model(UNetConfig()).to(device).eval()
    used = load_weights(a.ckpt, model, device)
    print(f"loaded {used} weights from {a.ckpt}")

    acc: dict[str, dict[str, list[float]]] = {}

    def add(name: str, vals: dict[str, float]) -> None:
        d = acc.setdefault(name, {})
        for k, v in vals.items():
            d.setdefault(k, []).append(v)

    for tiles in dl:
        tiles = {k: v.to(device) for k, v in tiles.items()}
        b = build_batch(tiles, spec, np.random.default_rng(0))

        with torch.no_grad():
            out = model(b["x"], b["cond"], b["gtdr_up"])
            out_lo = model(drop_second_looks(b["x"]), b["cond"], b["gtdr_up"])

        add("ISHTAR", evaluate_dem(out["z_hat"], b, a.tile_size, spec,
                                   sigma=out["sigma"], brightness=out["brightness"]))
        add("(a) bicubic GTDR", evaluate_dem(baselines.bicubic_gtdr(b["gtdr_up"]), b, a.tile_size, spec))

        # Cross-look PSNR: the DEM predicted without the second look, scored against it.
        if bool(b["valid_right"].any()):
            psnr = metrics.cross_look_psnr(
                out_lo["z_hat"], b["rv_right"], b["valid_right"], b["look_right"],
                b["theta_right"], spec.pixel_size_m, brightness=out_lo["brightness"])
            psnr_base = metrics.cross_look_psnr(
                b["gtdr_up"], b["rv_right"], b["valid_right"], b["look_right"],
                b["theta_right"], spec.pixel_size_m)
            add("ISHTAR", {"cross_look_psnr": psnr})
            add("(a) bicubic GTDR", {"cross_look_psnr": psnr_base})

        if not a.skip_classical:
            z_cl = baselines.classical_radarclinometry(
                b["rv_left"], b["valid_left"], b["look_left"], b["theta_left"],
                b["gtdr_up"], spec.pixel_size_m, steps=a.classical_steps)
            add("(b) classical radarclinometry", evaluate_dem(z_cl, b, a.tile_size, spec))
            if bool(b["valid_right"].any()):
                # The classical inversion only ever saw the left look, so this is the
                # same held-out test the network gets.
                add("(b) classical radarclinometry", {"cross_look_psnr": metrics.cross_look_psnr(
                    z_cl, b["rv_right"], b["valid_right"], b["look_right"],
                    b["theta_right"], spec.pixel_size_m)})

    rows = {name: {k: float(np.mean(v)) for k, v in d.items()} for name, d in acc.items()}
    print(f"\n{a.n_tiles} held-out tiles of {a.tile_size} px, checkpoint {a.ckpt}")

    cols = ["mae_vs_truth", "rmse_vs_truth", "slope_mae_deg", "stereo_mae_m",
            "alt_resid_m", "phys_resid_db", "cross_look_psnr", "cov_1sigma"]
    width = max(len(n) for n in rows) + 2
    print(f"\n{'':{width}s}" + "".join(f"{c:>16s}" for c in cols))
    for name, r in rows.items():
        print(f"{name:{width}s}" + "".join(
            f"{r.get(c, float('nan')):16.3f}" for c in cols))

    ishtar = rows.get("ISHTAR", {})
    print()
    for name, ref in (("(a) bicubic GTDR", rows.get("(a) bicubic GTDR")),
                      ("(b) classical radarclinometry", rows.get("(b) classical radarclinometry"))):
        if ref and "mae_vs_truth" in ishtar and "mae_vs_truth" in ref:
            skill = 1.0 - ishtar["mae_vs_truth"] / max(ref["mae_vs_truth"], 1e-9)
            verdict = "PASS" if skill > 0 else "FAIL"
            print(f"skill vs {name} (MAE): {skill:+6.1%}  [{verdict}]")
    print("\nBaseline (b) is the one that matters: it inverts the same forward model with "
          "no\nlearned prior, so beating it is what justifies the network at all.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
