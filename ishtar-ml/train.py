"""Training driver for phases 0-3.

    python train.py --phase overfit         # Phase 0 control: fit z_true directly
    python train.py --phase sanity          # Phase 0: weakly supervised, 200 tiles
    python train.py --phase earth  --config configs/config.yaml
    python train.py --phase venus_stereo
    python train.py --phase venus_global

Phase 0 runs entirely on synthetic Venus (`data.synthetic`) and needs no downloads, so
it is also the CI test that the renderer, the losses and the sign conventions still
agree with each other.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import BatchSpec, build_batch, drop_second_looks
from data.synthetic import SyntheticConfig, SyntheticVenus
from eval import metrics as eval_metrics
from model import losses as L
from model import physics
from model.unet import UNetConfig, build_model, count_parameters


@dataclass
class TrainConfig:
    phase: str = "sanity"
    steps: int = 2000
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    phys_ramp_steps: int = 5000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 25
    val_every: int = 250
    ckpt_dir: str = "runs/sanity"
    seed: int = 0
    tile_size: int = 256
    n_tiles: int = 200
    device: str = "auto"
    compile: bool = False
    init_from: str | None = None
    """Checkpoint to start from, as Phase 2 starts from Phase 1.

    EMA weights are preferred when present, matching what inference uses.
    """

    supervised_only: bool = False
    """Fit `z_true` directly with `L_earth` and switch every Venus term off.

    This is the control for Phase 0. The weakly supervised objective can fail for two
    completely different reasons — the network cannot represent or optimise the target,
    or the Venus supervision does not determine it — and they call for opposite fixes.
    Running the same architecture against a real label separates them in a few minutes.
    """
    weights: L.LossWeights = field(default_factory=L.LossWeights)
    loss_scales: L.LossScales = field(default_factory=lambda: L.SCALES)
    """Per-term normalisers. The default is right; `--raw-loss-scales` sets them all to 1
    so the pre-normalisation behaviour can be reproduced on identical code, which is the
    only way the A/B in docs/RESULTS.md means anything."""
    batch_spec: BatchSpec = field(default_factory=BatchSpec)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of weights; inference always uses these.

    The decay is warmed up: `decay_t = min(decay, (1 + t) / (10 + t))`. Without it, a
    fixed 0.999 keeps `0.999^t` of the *initial* weights, which after 900 steps is 41% —
    so a short run saves an average that is nearly half untrained, and every consumer
    (evaluation, calibration, global inference) silently scores a model that never
    existed. It is invisible in the training log, which reports the live weights.

    Measured on the Phase 0 run before the fix: +29.5% skill from the live weights,
    +4.0% from the saved EMA of the same run.
    """

    def __init__(self, model: torch.nn.Module, decay: float, warmup: float = 10.0):
        self.decay = decay
        self.warmup = warmup
        self.steps = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def current_decay(self) -> float:
        return min(self.decay, (1.0 + self.steps) / (self.warmup + self.steps))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.current_decay()
        self.steps += 1
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone().float()

    def state_dict(self) -> dict:
        return self.shadow


def load_weights(path, model: torch.nn.Module, device, prefer: str = "ema") -> str:
    """Load a checkpoint into `model`, and say out loud which weights were used.

    Inference should use the EMA, but a checkpoint written before the decay warmup
    existed holds an average that is partly the initialisation. `ema_steps` records how
    many updates went into it; without that key, or with too few, the raw weights are the
    honest choice and the caller is told.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    want = prefer if prefer in ck else "model"
    if want == "ema":
        steps = ck.get("ema_steps")
        if steps is None:
            want = "model"
            print(f"  {path}: EMA has no step count (pre-warmup checkpoint) — using raw weights")
        elif steps < 50:
            want = "model"
            print(f"  {path}: EMA saw only {steps} updates — using raw weights")
    model.load_state_dict(ck[want])
    return want


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def compute_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
    phys_ramp: float,
    venus: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """One forward pass and every loss term of Section 5."""
    px = cfg.batch_spec.pixel_size_m
    out = model(batch["x"], batch["cond"], batch["gtdr_up"])
    z = out["z_hat"]
    terms: dict[str, torch.Tensor] = {}
    diag: dict[str, torch.Tensor] = {}

    if not venus or cfg.supervised_only:
        terms["earth"] = L.loss_earth(z, batch["z_true"], px, scales=cfg.loss_scales)
    if cfg.supervised_only:
        # The uncertainty head still needs a target, and it is the one head whose
        # calibration is meaningful even in the supervised control.
        terms["nll"] = L.loss_nll(
            z, out["logvar"], batch["z_true"],
            torch.ones_like(batch["z_true"], dtype=torch.bool), pixel_size=px,
        )
        total = L.total_loss(terms, cfg.weights, phys_ramp)
        return total, terms, {"z_std": z.std().detach(), "sigma_mean": out["sigma"].mean().detach()}

    terms["stereo"] = L.loss_stereo(z, batch["stereo_dem"], batch["stereo_trust"], px,
                                    scales=cfg.loss_scales)
    terms["alt"] = L.loss_alt(
        z, batch["gtdr_up"], batch["gtdr_valid"], physics.FootprintSpec(), px,
        cfg.batch_spec.gtdr_stride_px, edge_margin_px=cfg.batch_spec.alt_edge_margin_px,
        scales=cfg.loss_scales,
    )

    phys_total = z.new_zeros(())
    n_looks = 0
    for name in ("left", "right", "stereo"):
        if not bool(batch[f"valid_{name}"].any()):
            continue
        lp, d = L.loss_phys(
            z, batch[f"rv_{name}"], batch[f"valid_{name}"], batch[f"look_{name}"],
            batch[f"theta_{name}"], px, brightness=out["brightness"],
            scales=cfg.loss_scales,
        )
        phys_total = phys_total + lp
        n_looks += 1
        if name == "left":
            diag.update({f"{k}_left": v for k, v in d.items()})
    terms["phys"] = phys_total / max(1, n_looks)

    # Cross-look: predict without the second looks, render them, compare. This is what
    # trains the pathway the other 80% of Venus runs through.
    if bool(batch["valid_right"].any()):
        out_lo = model(drop_second_looks(batch["x"]), batch["cond"], batch["gtdr_up"])
        terms["cross"] = L.loss_cross(
            out_lo["z_hat"], batch["rv_right"], batch["valid_right"], batch["look_right"],
            batch["theta_right"], px, brightness=out_lo["brightness"],
            scales=cfg.loss_scales,
        )

    terms["rms"] = L.loss_rms(z, batch["rms_slope"], batch["rms_valid"], px,
                              cfg.batch_spec.gsdr_cell_m, scales=cfg.loss_scales)
    terms["nll"] = L.loss_nll(
        z, out["logvar"], batch["stereo_dem"], batch["stereo_trust"],
        unsupervised_mask=batch["unsupervised"], pixel_size=px,
    )
    terms["reg"] = L.loss_reg(out["brightness_lr"], z, px)

    total = L.total_loss(terms, cfg.weights, phys_ramp)
    return total, terms, {**diag, **{"z_std": z.std().detach(), "sigma_mean": out["sigma"].mean().detach()}}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> dict[str, float]:
    """Metrics against the synthetic ground truth, which exists only in Phase 0.

    On real Venus tiles there is no `z_true`; the honest metric table lives in
    `eval/run_eval.py`, which scores against stereo, altimetry, the radiometric residual
    and cross-look prediction instead. This function reports nothing rather than
    inventing a substitute.
    """
    model.eval()
    spec = BatchSpec(**{**asdict(cfg.batch_spec), "augment": False})
    px = spec.pixel_size_m
    acc = {"mae_m": 0.0, "rmse_m": 0.0, "gtdr_mae_m": 0.0, "slope_mae_deg": 0.0,
           "alt_resid_m": 0.0, "sigma_mean_m": 0.0, "cov_1sigma": 0.0, "n": 0.0}
    for tiles in loader:
        tiles = {k: v.to(device) for k, v in tiles.items()}
        b = build_batch(tiles, spec, np.random.default_rng(0))
        if "z_true" not in b:
            model.train()
            return {}
        out = model(b["x"], b["cond"], b["gtdr_up"])
        z, zt = out["z_hat"], b["z_true"]
        err = z - zt
        pe, pn = physics.sobel_gradient(z, px)
        te, tn = physics.sobel_gradient(zt, px)
        slope_err = (torch.atan(torch.sqrt(pe**2 + pn**2)) - torch.atan(torch.sqrt(te**2 + tn**2))).abs()
        n = float(z.shape[0])
        acc["mae_m"] += float(err.abs().mean()) * n
        acc["rmse_m"] += float(err.pow(2).mean().sqrt()) * n
        acc["gtdr_mae_m"] += float((b["gtdr_up"] - zt).abs().mean()) * n
        acc["slope_mae_deg"] += float(torch.rad2deg(slope_err).mean()) * n
        # Same definition the metric table uses — against the altimetry posts, not
        # against a blurred truth that only exists in the synthetic set. A number in the
        # log that means something slightly different from the same-named number in the
        # evaluation table is how two reports quietly stop being comparable.
        acc["alt_resid_m"] += eval_metrics.altimetry_residual(
            z, b["gtdr_up"], b["gtdr_valid"], px, spec.gtdr_stride_px,
            edge_margin_px=spec.alt_edge_margin_px,
        ) * n
        acc["sigma_mean_m"] += float(out["sigma"].mean()) * n
        acc["cov_1sigma"] += float((err.abs() <= out["sigma"]).float().mean()) * n
        acc["n"] += n
    model.train()
    n = acc.pop("n")
    if n == 0:
        return {}
    return {k: v / n for k, v in acc.items()}


def train(cfg: TrainConfig) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    ckpt = Path(cfg.ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    scfg = SyntheticConfig(size=cfg.tile_size, pixel_size_m=cfg.batch_spec.pixel_size_m)
    train_ds = SyntheticVenus(cfg.n_tiles, scfg, seed=cfg.seed)
    val_ds = SyntheticVenus(max(8, cfg.batch_size * 2), scfg, seed=cfg.seed + 991)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)
    # A fixed slice of the *training* set, scored the same way. Without it, a flat metric
    # curve is ambiguous: the model may be underfitting or may be fitting and failing to
    # generalise, and those call for opposite responses.
    seen_dl = DataLoader(
        torch.utils.data.Subset(train_ds, range(min(len(train_ds), max(8, cfg.batch_size * 2)))),
        batch_size=cfg.batch_size,
    )

    model = build_model(UNetConfig()).to(device)
    if cfg.init_from:
        ck = torch.load(cfg.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("ema") or ck["model"])
        print(f"initialised from {cfg.init_from} "
              f"({'EMA' if 'ema' in ck else 'raw'} weights)")
    if cfg.compile:
        model = torch.compile(model)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"device={device}  params={count_parameters(model)/1e6:.1f}M  "
          f"tiles={len(train_ds)}  steps={cfg.steps}")

    step, t0, history = 0, time.time(), []
    # Per-term running means. The raw per-step numbers are close to unreadable: whether a
    # batch contains any stereo-covered tiles swings `L_stereo` between 0 and ~150, so an
    # instantaneous loss says more about the sampler than about the model.
    smooth: dict[str, float] = {}
    smooth_beta = 0.98
    while step < cfg.steps:
        for tiles in train_dl:
            if step >= cfg.steps:
                break
            tiles = {k: v.to(device) for k, v in tiles.items()}
            batch = build_batch(tiles, cfg.batch_spec, rng)

            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            ramp = min(1.0, (step + 1) / max(1, cfg.phys_ramp_steps))

            total, terms, diag = compute_losses(model, batch, cfg, ramp)
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)

            for k, v in list(terms.items()) + [("total", total)]:
                smooth[k] = smooth_beta * smooth.get(k, float(v.detach())) + (1 - smooth_beta) * float(v.detach())

            if step % cfg.log_every == 0:
                parts = " ".join(f"{k}={smooth[k]:.3f}" for k in terms)
                print(f"step {step:5d} lr={lr_at(step,cfg):.2e} loss={smooth['total']:8.3f} {parts} "
                      f"|g|={float(gnorm):.1f} phys_dB={float(diag.get('phys_resid_db_left', 0)):.2f} "
                      f"({(time.time()-t0)/max(step,1):.2f}s/step)")
            if cfg.val_every and step > 0 and step % cfg.val_every == 0 and (
                    m := evaluate(model, val_dl, cfg, device)):
                seen = evaluate(model, seen_dl, cfg, device)
                print(f"  val  @ {step}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
                print(f"  seen @ {step}: mae_m={seen['mae_m']:.3f} "
                      f"(bicubic {seen['gtdr_mae_m']:.3f}, skill {1 - seen['mae_m']/max(seen['gtdr_mae_m'],1e-9):+.1%})")
                history.append({"step": step, **m, "seen_mae_m": seen["mae_m"],
                                "seen_gtdr_mae_m": seen["gtdr_mae_m"]})
            step += 1

    metrics = evaluate(model, val_dl, cfg, device)
    if metrics:
        print("final val: " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    # Score the EMA too. Inference loads it, so a gap here is a gap in what ships — and
    # it is exactly the gap an un-warmed EMA opens without any other symptom.
    ema_model = build_model(UNetConfig()).to(device)
    ema_model.load_state_dict(ema.state_dict())
    ema_metrics = evaluate(ema_model, val_dl, cfg, device)
    if ema_metrics:
        print(f"final val (EMA, decay {ema.current_decay():.4f}): "
              + " ".join(f"{k}={v:.3f}" for k, v in ema_metrics.items()))
    if metrics and cfg.phase in ("sanity", "overfit", "pretrain"):
        skill = 1.0 - metrics["mae_m"] / max(metrics["gtdr_mae_m"], 1e-6)
        verdict = "PASS" if skill > 0.0 else "FAIL"
        label = "supervised control" if cfg.supervised_only else "weakly supervised"
        print(f"\nPhase 0 acceptance ({label}) -- baseline (a), bicubic GTDR:")
        print(f"  bicubic GTDR MAE : {metrics['gtdr_mae_m']:7.2f} m")
        print(f"  ISHTAR MAE       : {metrics['mae_m']:7.2f} m")
        print(f"  skill vs baseline: {skill:+7.1%}   [{verdict}]")
        print(f"  1-sigma coverage : {metrics['cov_1sigma']:7.1%}  (well-calibrated is ~68%)")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "ema_steps": ema.steps, "ema_decay": ema.current_decay(),
                "cfg": asdict(cfg), "metrics": metrics}, ckpt / "last.pt")
    (ckpt / "history.json").write_text(json.dumps({"history": history, "final": metrics}, indent=2))
    return metrics


PHASES = {
    "overfit":      dict(steps=600,   n_tiles=8,     lr=3e-4, ckpt_dir="runs/overfit",
                         supervised_only=True, val_every=100),
    # The Earth stage in miniature: a real label exists, so the network learns the prior
    # "what terrain looks like in radar" that no amount of Venus supervision can teach.
    # On the real pipeline this is Sentinel-1 + GLO-30; here it is synthetic z_true.
    "pretrain":     dict(steps=900,   n_tiles=200,   lr=3e-4, ckpt_dir="runs/pretrain",
                         supervised_only=True, val_every=300),
    "sanity":       dict(steps=2000,  n_tiles=200,   lr=3e-4, ckpt_dir="runs/sanity"),
    "earth":        dict(steps=80000, n_tiles=30000, lr=3e-4, ckpt_dir="runs/earth"),
    "venus_stereo": dict(steps=30000, n_tiles=20000, lr=1e-4, ckpt_dir="runs/venus_stereo"),
    "venus_global": dict(steps=50000, n_tiles=60000, lr=1e-4, ckpt_dir="runs/venus_global"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="sanity", choices=sorted(PHASES))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--tile-size", type=int)
    ap.add_argument("--n-tiles", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--init-from", help="checkpoint to warm-start from, as Phase 2 does")
    ap.add_argument("--raw-loss-scales", action="store_true",
                    help="disable the per-term uncertainty normalisation (for the A/B only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = TrainConfig(phase=a.phase, seed=a.seed, device=a.device, **PHASES[a.phase])
    for k in ("steps", "batch_size", "tile_size", "n_tiles", "ckpt_dir", "init_from"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.raw_loss_scales:
        cfg.loss_scales = L.UNIT_SCALES
        cfg.ckpt_dir = cfg.ckpt_dir + "_rawscale" if not a.ckpt_dir else cfg.ckpt_dir
        print("raw loss scales: terms are metres, decibels and radians, unnormalised")
    if a.phase not in ("sanity", "overfit", "pretrain"):
        raise SystemExit(
            f"Phase {a.phase!r} needs real tiles: run data/download.py then data/tile.py "
            "and point the dataset at the Zarr store. Only --phase sanity runs on the "
            "synthetic set."
        )
    train(cfg)


if __name__ == "__main__":
    main()
