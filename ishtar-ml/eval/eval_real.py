"""Score checkpoints on held-out *real* Venus tiles.

    python -m eval.eval_real runs/real_cold runs/real_warm --data data_tiles/venus_256

There is no ground truth, so every column here is a proxy the architecture note asks for,
and each says something different:

* **altimetry residual** — the drift check. Under 30 m is the budget. It is the one metric
  a model can satisfy by doing nothing, since the prediction starts as the altimetry.
* **physics residual** — how much of the radar image the DEM explains, in dB, with the
  best constant brightness removed so a model with a brightness head is not flattered.
* **cross-look PSNR** — the honest test of 75 m detail. Predict a DEM from the left look
  alone, render the *right* look from it, and score against an image the model never saw.
  Only the ~31% of tiles with right-look coverage contribute.
* **relief over GTDR** — how much the model actually added. A model that scores well
  everywhere else with 2 m of relief has simply declined to answer.

The bicubic-GTDR baseline is the row to beat: it is what the model is handed as input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import BatchSpec, build_batch, drop_second_looks  # noqa: E402
from data.real import RealVenus  # noqa: E402
from eval import metrics  # noqa: E402
from model import physics  # noqa: E402
from model.unet import UNetConfig, build_model  # noqa: E402
from train import load_weights, pick_device  # noqa: E402

COLS = [("alt_resid_m", "alt m", 2), ("phys_resid_db", "phys dB", 3),
        ("cross_look_psnr", "x-look dB", 2), ("stereo_mae_m", "stereo m", 1),
        ("relief_m", "relief m", 1), ("sigma_m", "sigma m", 1)]


def score(z, sigma, b, spec, brightness=None, z_left_only=None):
    px = spec.pixel_size_m
    out = {
        "alt_resid_m": metrics.altimetry_residual(
            z, b["gtdr_up"], b["gtdr_valid"], px, spec.gtdr_stride_px),
        "phys_resid_db": metrics.physics_residual_db(
            z, b["rv_left"], b["valid_left"], b["look_left"], b["theta_left"], px,
            brightness=brightness),
        "relief_m": float((z - b["gtdr_up"]).std()),
    }
    if sigma is not None:
        out["sigma_m"] = float(sigma.mean())
    if bool(b["stereo_trust"].any()):
        out["stereo_mae_m"] = metrics.stereo_metrics(
            z, b["stereo_dem"], b["stereo_trust"], px)["stereo_mae_m"]
    if bool(b["valid_right"].any()):
        out["cross_look_psnr"] = metrics.cross_look_psnr(
            z_left_only if z_left_only is not None else z,
            b["rv_right"], b["valid_right"], b["look_right"], b["theta_right"], px,
            brightness=brightness)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--data", default="data_tiles/venus_256")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tile-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("outputs/eval_real.json"))
    a = ap.parse_args()

    device = pick_device(a.device)
    spec = BatchSpec(augment=False)
    ds = RealVenus(a.data, a.split, crop_px=a.tile_size)
    print(ds.summary())
    dl = DataLoader(ds, batch_size=a.batch_size)

    models = {}
    for run in a.runs:
        ck = run if run.suffix == ".pt" else run / "last.pt"
        if not ck.exists():
            print(f"  skipping {run}: no checkpoint")
            continue
        m = build_model(UNetConfig()).to(device).eval()
        load_weights(ck, m, device)
        models[run.name] = m

    acc: dict[str, dict[str, list[float]]] = {}

    def add(name, vals):
        d = acc.setdefault(name, {})
        for k, v in vals.items():
            d.setdefault(k, []).append(v)

    for tiles in dl:
        b = build_batch({k: v.to(device) for k, v in tiles.items()}, spec,
                        np.random.default_rng(0))
        add("(a) bicubic GTDR", score(b["gtdr_up"], None, b, spec))
        for name, m in models.items():
            with torch.no_grad():
                o = m(b["x"], b["cond"], b["gtdr_up"])
                o_lo = m(drop_second_looks(b["x"]), b["cond"], b["gtdr_up"])
            add(name, score(o["z_hat"], o["sigma"], b, spec,
                            brightness=o["brightness"], z_left_only=o_lo["z_hat"]))

    rows = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in acc.items()}
    w = max(len(n) for n in rows) + 2
    print(f"\n{len(ds)} held-out real tiles ({a.split})\n")
    print(f"{'':{w}s}" + "".join(f"{lab:>11s}" for _, lab, _ in COLS))
    for n, r in rows.items():
        print(f"{n:{w}s}" + "".join(
            f"{r[k]:11.{d}f}" if k in r else f"{'—':>11s}" for k, _, d in COLS))

    base = rows["(a) bicubic GTDR"]
    print()
    for n, r in rows.items():
        if n.startswith("(a)"):
            continue
        dphys = base["phys_resid_db"] - r["phys_resid_db"]
        dx = r.get("cross_look_psnr", float("nan")) - base.get("cross_look_psnr", float("nan"))
        print(f"  {n}: explains the image {dphys:+.3f} dB better than the altimetry it "
              f"started from; cross-look {dx:+.2f} dB")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
