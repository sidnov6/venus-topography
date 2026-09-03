"""Assemble the ISHTAR write-up into a single self-contained HTML file.

    python docs/writeup/build.py

Figures are inlined as data URIs, so the output is one file with no external assets. The
numbers live in the tables at the top of this script and come from `docs/RESULTS.md`;
the ablation table is read straight from `ishtar-ml/outputs/ablation.json`, so it cannot
drift from the run that produced it.

The build output (`ishtar.html`, ~2 MB of base64) is gitignored — rebuild it rather than
committing it.
"""
import base64, json, pathlib, sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]

# --- numbers, filled from the runs -----------------------------------------------------
# Mean absolute error in metres against the synthetic truth, on the trainer's validation
# set. Plotted as metres rather than skill: the raw arm's mid-training excursion to 244 m
# would otherwise compress the whole normalised curve into the top of the axis.
BASELINE_MAE = 132.46
CONVERGENCE = {
    "raw":        [(250, 132.47), (500, 244.25), (750, 123.69), (900, 123.49)],
    "normalised": [(250, 112.51), (500, 92.54), (750, 89.31), (900, 90.90)],
}
FINAL = {
    "raw":        {"mae": 123.49, "rmse": 161.69, "slope": 7.75, "alt": 10.87, "cov": 81.0, "skill": 6.8},
    "normalised": {"mae": 90.90, "rmse": 119.12, "slope": 5.24, "alt": 10.67, "cov": 68.3, "skill": 31.4},
    # pretrain -> fine-tune, the staged arm
    "staged":     {"mae": 82.81, "rmse": 108.17, "slope": 3.91, "alt": 11.66, "cov": 70.4, "skill": 37.5},
    "pretrain":   {"mae": 85.79, "rmse": 111.20, "slope": 3.38, "alt": 22.37, "cov": 66.5, "skill": 35.2},
}
ABLATION_JSON = ROOT / "ishtar-ml/outputs/ablation.json"
CALIBRATION = {"t": 1.044, "cov1_before": 0.693, "cov1_after": 0.709,
               "cov2_before": 0.912, "cov2_after": 0.922}

# Committed copies, so the write-up rebuilds from a fresh clone. `ishtar-ml/outputs/` is
# gitignored; re-run `scripts/phase0.sh` and copy the new figures here to refresh them.
FIGURES = {
    "spectrum": HERE / "spectrum.png",
    "panel":    HERE / "panel.png",
    # Produced by ishtar-globe/scripts/smoke.mjs or a manual capture; see the README.
    "globe":    HERE / "globe.png",
}
FALLBACK = {
    "spectrum": ROOT / "ishtar-ml/outputs/panel_sanity_0_spectrum.png",
    "panel":    ROOT / "ishtar-ml/outputs/panel_sanity_0.png",
}


def data_uri(key):
    p = FIGURES[key]
    if not p.exists():
        p = FALLBACK.get(key)
    if p is None or not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def fmt(v, digits=2, suffix=""):
    return "&mdash;" if v is None else f"{v:.{digits}f}{suffix}"


def pct(v):
    return "&mdash;" if v is None else f"{v:+.1f}%"


def main():
    head = (HERE / "head.html").read_text()
    body = (HERE / "body.html").read_text()

    ablation = json.loads(ABLATION_JSON.read_text()) if ABLATION_JSON.exists() else {}

    body = body.replace("{{CONVERGENCE_JSON}}",
                        json.dumps({"series": CONVERGENCE, "baseline": BASELINE_MAE}))
    for key in ("spectrum", "panel", "globe"):
        uri = data_uri(key)
        body = body.replace("{{IMG_" + key.upper() + "}}", uri or "")
    body = body.replace("{{ABLATION_ROWS}}", ablation_rows(ablation))
    for name, d in FINAL.items():
        for k, v in d.items():
            digits = 1 if k in ("slope", "cov", "skill") else 2
            body = body.replace("{{%s_%s}}" % (name.upper(), k.upper()), fmt(v, digits))
    for name, pts in CONVERGENCE.items():
        for step, v in pts:
            body = body.replace("{{%s_%d}}" % (name.upper(), step), fmt(v, 1, " m"))
            body = body.replace("{{%s_%d_SKILL}}" % (name.upper(), step),
                                pct(None if v is None else 100 * (1 - v / BASELINE_MAE)))

    out = HERE / "ishtar.html"
    out.write_text(head + "\n" + body)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")


ROW_ORDER = ["(a) bicubic GTDR", "(b) classical", "sanity_rawscale_v2", "sanity",
             "sanity_pretrained", "pretrain", "overfit"]
LABELS = {
    "sanity": "Weakly supervised (normalised losses)",
    "sanity_pretrained": "Pretrained, then weakly supervised",
    "sanity_rawscale_v2": "Weakly supervised (raw loss units)",
    "pretrain": "Supervised pretrain only",
    "overfit": "Supervised control (8 tiles)",
    "(a) bicubic GTDR": "(a) Bicubic GTDR &mdash; the baseline",
    "(b) classical": "(b) Classical radarclinometry",
}
COLS = [("mae_vs_truth", 2), ("rmse_vs_truth", 2), ("slope_mae_deg", 2),
        ("stereo_mae_m", 1), ("alt_resid_m", 1), ("phys_resid_db", 2),
        ("cross_look_psnr", 1), ("cov_1sigma", 3)]


def ablation_rows(rows):
    if not rows:
        return ('<tr><td colspan="9" style="color:var(--muted)">'
                'The ablation run had not finished when this page was built.</td></tr>')
    base = rows.get("(a) bicubic GTDR", {}).get("mae_vs_truth")
    out = []
    for key in ROW_ORDER:
        r = rows.get(key)
        if not r:
            continue
        cells = "".join(
            f'<td class="num">{fmt(r.get(c), d)}</td>' for c, d in COLS
        )
        skill = ""
        if base and r.get("mae_vs_truth") is not None:
            s = 1 - r["mae_vs_truth"] / base
            cls = "good" if s > 0.01 else ("bad" if s < -0.01 else "")
            skill = f'<td class="num {cls}">{s:+.1%}</td>'
        else:
            skill = '<td class="num">&mdash;</td>'
        hl = ' class="highlight"' if key == "sanity" else ""
        out.append(f"<tr{hl}><th scope=\"row\">{LABELS.get(key, key)}</th>{cells}{skill}</tr>")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
