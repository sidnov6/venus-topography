"""Qualitative panels for the write-up.

Section 7 asks for SAR, GTDR, prediction, uncertainty and stereo side by side over
Maxwell Montes, Maat Mons, Alpha Regio, Artemis, Mead and a pancake-dome field. The same
function draws the Phase 0 panel from synthetic tiles, which is what you actually look at
while debugging.

    python -m eval.panels --ckpt runs/sanity/last.pt --out outputs/panel_sanity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PANEL_SITES = ["Maxwell Montes", "Maat Mons", "Alpha Regio", "Artemis Corona",
               "Mead crater", "Alpha/Eistla domes"]


def draw_panel(fields: dict[str, np.ndarray], out: Path, title: str = "") -> Path:
    """One row per field, shared extent. Elevation panels share a colour scale so the
    prediction and the reference are actually comparable by eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)

    # Scale every elevation panel to the *reference* field's range, not to the pooled
    # range of all of them. Pooling lets the widest-range panel set the scale, which
    # flattens the predictions into featureless green and hides exactly the comparison
    # the figure exists to make.
    elev_keys = [k for k in fields if k in ("GTDR", "ISHTAR", "classical", "stereo DEM", "truth")]
    reference = next((k for k in ("truth", "stereo DEM") if k in fields), None)
    if reference:
        vmin, vmax = np.nanpercentile(fields[reference], [1, 99])
    elif elev_keys:
        vmin, vmax = np.nanpercentile(np.concatenate([fields[k].ravel() for k in elev_keys]), [2, 98])
    else:
        vmin = vmax = None

    for ax, (name, arr) in zip(axes, fields.items()):
        if name in ("SAR (dB)",):
            im = ax.imshow(arr, cmap="gray", vmin=-8, vmax=8)
        elif name in ("1 sigma (m)",):
            im = ax.imshow(arr, cmap="magma")
        else:
            im = ax.imshow(arr, cmap="terrain", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)

    if title:
        fig.suptitle(title, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def spectrum_figure(curves: dict[str, np.ndarray], wavelengths: np.ndarray, out: Path,
                    title: str = "") -> Path:
    """Radially averaged power spectra on log-log axes.

    This is the figure that catches invented texture. A model that hallucinates detail
    matches the reference at long wavelengths, where the altimetry anchor pins it, and
    then runs high and flat below a kilometre — plausible to the eye, wrong in the data.
    Excess power at short wavelengths is the signature; matching power is the goal.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for name, power in curves.items():
        ok = np.isfinite(power) & (power > 0)
        style = {"linewidth": 2.0} if name.startswith("ISHTAR") else {"linewidth": 1.2, "alpha": 0.85}
        ax.loglog(wavelengths[ok] / 1000.0, power[ok], label=name, **style)

    ax.set_xlabel("wavelength (km)")
    ax.set_ylabel("radially averaged power")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.2)
    ax.axvspan(0.075, 1.0, color="0.9", zorder=0)
    ax.text(0.9, 0.05, "band the model must invent", transform=ax.transAxes,
            ha="right", fontsize=8, color="0.4")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.dataset import BatchSpec, build_batch
    from data.synthetic import SyntheticConfig, SyntheticVenus
    from model.physics import rv_from_dn
    from model.unet import UNetConfig, build_model
    from train import load_weights, pick_device

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path("runs/sanity/last.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/panel_sanity.png"))
    ap.add_argument("--tile-size", type=int, default=128)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--classical-steps", type=int, default=150)
    a = ap.parse_args()

    device = pick_device(a.device)
    ds = SyntheticVenus(8, SyntheticConfig(size=a.tile_size), seed=9911)
    tiles = {k: v[None].to(device) for k, v in ds[a.index].items()}
    b = build_batch(tiles, BatchSpec(augment=False), np.random.default_rng(0))

    model = build_model(UNetConfig()).to(device).eval()
    used = load_weights(a.ckpt, model, device)
    print(f"loaded {used} weights from {a.ckpt}")
    with torch.no_grad():
        out = model(b["x"], b["cond"], b["gtdr_up"])

    from eval.baselines import classical_radarclinometry

    z_classical = classical_radarclinometry(
        b["rv_left"], b["valid_left"], b["look_left"], b["theta_left"],
        b["gtdr_up"], BatchSpec().pixel_size_m, steps=a.classical_steps,
    )

    sar, _ = rv_from_dn(tiles["dn_left"][:, None])
    fields = {
        "SAR (dB)": sar[0, 0].cpu().numpy(),
        "GTDR": b["gtdr_up"][0, 0].cpu().numpy(),
        "classical": z_classical[0, 0].cpu().numpy(),
        "ISHTAR": out["z_hat"][0, 0].cpu().numpy(),
        "1 sigma (m)": out["sigma"][0, 0].cpu().numpy(),
        "truth": b["z_true"][0, 0].cpu().numpy(),
    }
    err = float(np.abs(fields["ISHTAR"] - fields["truth"]).mean())
    base = float(np.abs(fields["GTDR"] - fields["truth"]).mean())
    path = draw_panel(fields, a.out, title=f"synthetic tile {a.index} — MAE {err:.1f} m vs bicubic {base:.1f} m")
    print(f"wrote {path}")

    from eval.metrics import radial_power_spectrum

    curves = {}
    for name, key in (("truth", "truth"), ("ISHTAR", "ISHTAR"),
                      ("(b) classical", "classical"), ("(a) bicubic GTDR", "GTDR")):
        wl, power = radial_power_spectrum(
            torch.from_numpy(fields[key]).float()[None, None], BatchSpec().pixel_size_m)
        curves[name] = power
    spec_path = a.out.with_name(a.out.stem + "_spectrum.png")
    print(f"wrote {spectrum_figure(curves, wl, spec_path, title=f'tile {a.index}')}")


if __name__ == "__main__":
    main()
