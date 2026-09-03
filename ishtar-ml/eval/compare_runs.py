"""Score several checkpoints on one held-out set and print the ablation table.

    python -m eval.compare_runs runs/sanity runs/sanity_pretrained runs/pretrain

Everything is evaluated on the *same* tiles with the same baselines, which is the only
way the comparison means anything: a 16-tile synthetic draw moves MAE by several percent
on its own, so numbers taken from separate runs' own validation logs are not comparable.
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
from data.synthetic import SyntheticConfig, SyntheticVenus  # noqa: E402
from eval import baselines, metrics  # noqa: E402
from eval.run_eval import evaluate_dem  # noqa: E402
from model.unet import UNetConfig, build_model  # noqa: E402
from train import load_weights, pick_device  # noqa: E402

COLUMNS = [
    ("mae_vs_truth", "MAE m"),
    ("rmse_vs_truth", "RMSE m"),
    ("slope_mae_deg", "slope deg"),
    ("stereo_mae_m", "stereo m"),
    ("alt_resid_m", "alt m"),
    ("phys_resid_db", "phys dB"),
    ("cross_look_psnr", "xlook dB"),
    ("cov_1sigma", "1sig cov"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--tile-size", type=int, default=128)
    ap.add_argument("--n-tiles", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--classical-steps", type=int, default=120)
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/ablation.json"))
    a = ap.parse_args()

    device = pick_device(a.device)
    spec = BatchSpec(augment=False)
    ds = SyntheticVenus(a.n_tiles, SyntheticConfig(size=a.tile_size), seed=9911)
    dl = DataLoader(ds, batch_size=a.batch_size)

    acc: dict[str, dict[str, list[float]]] = {}

    def add(name: str, vals: dict[str, float]) -> None:
        d = acc.setdefault(name, {})
        for k, v in vals.items():
            d.setdefault(k, []).append(v)

    models = {}
    for run in a.runs:
        ckpt = run if run.suffix == ".pt" else run / "last.pt"
        if not ckpt.exists():
            print(f"skipping {run}: no checkpoint", file=sys.stderr)
            continue
        m = build_model(UNetConfig()).to(device).eval()
        used = load_weights(ckpt, m, device)
        print(f"{run.name}: {used} weights")
        models[run.name] = m

    for tiles in dl:
        tiles = {k: v.to(device) for k, v in tiles.items()}
        b = build_batch(tiles, spec, np.random.default_rng(0))
        has_right = bool(b["valid_right"].any())

        add("(a) bicubic GTDR", evaluate_dem(b["gtdr_up"], b, a.tile_size, spec))
        if has_right:
            add("(a) bicubic GTDR", {"cross_look_psnr": metrics.cross_look_psnr(
                b["gtdr_up"], b["rv_right"], b["valid_right"], b["look_right"],
                b["theta_right"], spec.pixel_size_m)})

        if not a.skip_classical:
            z_cl = baselines.classical_radarclinometry(
                b["rv_left"], b["valid_left"], b["look_left"], b["theta_left"],
                b["gtdr_up"], spec.pixel_size_m, steps=a.classical_steps)
            add("(b) classical", evaluate_dem(z_cl, b, a.tile_size, spec))
            if has_right:
                add("(b) classical", {"cross_look_psnr": metrics.cross_look_psnr(
                    z_cl, b["rv_right"], b["valid_right"], b["look_right"],
                    b["theta_right"], spec.pixel_size_m)})

        for name, m in models.items():
            with torch.no_grad():
                out = m(b["x"], b["cond"], b["gtdr_up"])
                out_lo = m(drop_second_looks(b["x"]), b["cond"], b["gtdr_up"])
            add(name, evaluate_dem(out["z_hat"], b, a.tile_size, spec,
                                   sigma=out["sigma"], brightness=out["brightness"]))
            if has_right:
                add(name, {"cross_look_psnr": metrics.cross_look_psnr(
                    out_lo["z_hat"], b["rv_right"], b["valid_right"], b["look_right"],
                    b["theta_right"], spec.pixel_size_m, brightness=out_lo["brightness"])})

    rows = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in acc.items()}

    width = max(len(n) for n in rows) + 2
    print(f"\n{a.n_tiles} held-out tiles of {a.tile_size} px\n")
    print(f"{'':{width}s}" + "".join(f"{lbl:>11s}" for _, lbl in COLUMNS) + f"{'skill':>9s}")
    base = rows["(a) bicubic GTDR"]["mae_vs_truth"]
    for name, r in rows.items():
        skill = 1.0 - r["mae_vs_truth"] / base
        print(f"{name:{width}s}" + "".join(f"{r.get(k, float('nan')):11.2f}" for k, _ in COLUMNS)
              + f"{skill:+8.1%}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
