"""The training driver. Cheap structural checks — the expensive question (does it learn)
is answered by `scripts/phase0.sh`, not by the test suite."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import BatchSpec, build_batch
from data.synthetic import SyntheticConfig, SyntheticVenus
from model.unet import UNetConfig, build_model
from train import EMA, PHASES, TrainConfig, compute_losses, lr_at, pick_device


def small_batch(size=64, n=2, device="cpu", seed=1):
    ds = SyntheticVenus(n, SyntheticConfig(size=size), seed=seed)
    tiles = {k: v.to(device) for k, v in next(iter(DataLoader(ds, batch_size=n))).items()}
    return build_batch(tiles, BatchSpec(augment=False), np.random.default_rng(0))


def test_every_declared_phase_is_runnable_or_explicitly_gated():
    for name, cfg in PHASES.items():
        assert "steps" in cfg and "ckpt_dir" in cfg
        assert cfg["steps"] > 0
    # The phases that need the real 300 GB of tiles must say so rather than silently
    # training on the synthetic set and producing meaningless numbers.
    assert {"earth", "venus_stereo", "venus_global"} <= set(PHASES)


def test_lr_schedule_warms_up_then_decays_to_zero():
    cfg = TrainConfig(steps=1000, warmup_steps=100, lr=3e-4)
    assert lr_at(0, cfg) < cfg.lr / 10
    assert lr_at(99, cfg) == pytest.approx(cfg.lr, rel=1e-6)
    assert lr_at(500, cfg) < cfg.lr
    assert lr_at(999, cfg) < cfg.lr / 100
    assert lr_at(2000, cfg) == pytest.approx(0.0, abs=1e-9)


def test_ema_tracks_the_weights_without_being_them():
    model = build_model(UNetConfig())
    ema = EMA(model, decay=0.9)
    for _ in range(50):  # past the warmup, where the decay reaches its nominal value
        ema.update(model)
    before = {k: v.clone() for k, v in ema.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ema.update(model)
    key = next(k for k, v in model.state_dict().items() if v.dtype.is_floating_point)
    assert not torch.allclose(ema.state_dict()[key], before[key])
    assert not torch.allclose(ema.state_dict()[key], model.state_dict()[key])


def test_ema_decay_warms_up():
    """A fixed 0.999 keeps 0.999^t of the *initial* weights. At 900 steps that is 41%, so
    a short run saves an average that is nearly half untrained — and the training log,
    which reports the live weights, shows nothing. Measured before this fix: +29.5% skill
    live, +4.0% from the saved EMA of the same run."""
    ema = EMA(build_model(UNetConfig()), decay=0.999, warmup=10.0)
    assert ema.current_decay() < 0.2, "the first updates must track the model, not the init"
    for _ in range(900):
        ema.steps += 1
    assert ema.current_decay() < 0.995, "at 900 steps the horizon must be far shorter than 1000"
    for _ in range(100_000):
        ema.steps += 1
    assert ema.current_decay() == pytest.approx(0.999), "and must reach the nominal decay"


def test_ema_of_a_static_model_converges_to_it():
    """The property the warmup exists to give: after a realistic number of steps the EMA
    of an unchanging model is that model, not a blend with its initialisation."""
    model = build_model(UNetConfig())
    ema = EMA(model, decay=0.999)
    key = next(k for k, v in model.state_dict().items()
               if v.dtype.is_floating_point and v.numel() > 100)
    with torch.no_grad():
        model.state_dict()[key].add_(1.0)
    for _ in range(900):
        ema.update(model)
    err = (ema.state_dict()[key] - model.state_dict()[key]).abs().max()
    assert float(err) < 1e-3, f"EMA is still {float(err):.3f} away from the model it averaged"


def test_all_eight_terms_are_present_and_finite_in_a_venus_step():
    cfg = TrainConfig(batch_size=2, tile_size=64)
    model = build_model(UNetConfig())
    total, terms, diag = compute_losses(model, small_batch(), cfg, phys_ramp=1.0)
    assert torch.isfinite(total)
    assert {"stereo", "alt", "phys", "rms", "nll", "reg"} <= set(terms)
    for name, v in terms.items():
        assert torch.isfinite(v), name


def test_supervised_control_switches_off_the_venus_terms():
    """The control has to be a clean comparison: if a Venus term leaked in, a difference
    between the two runs would not mean what the ablation says it means."""
    cfg = TrainConfig(batch_size=2, tile_size=64, supervised_only=True)
    model = build_model(UNetConfig())
    _, terms, _ = compute_losses(model, small_batch(), cfg, phys_ramp=1.0)
    assert set(terms) == {"earth", "nll"}


def test_a_training_step_moves_the_weights():
    cfg = TrainConfig(batch_size=2, tile_size=64)
    model = build_model(UNetConfig())
    before = next(model.head_res[-1].parameters()).clone()
    total, _, _ = compute_losses(model, small_batch(), cfg, phys_ramp=1.0)
    total.backward()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert not torch.allclose(next(model.head_res[-1].parameters()), before)


def test_phys_ramp_is_applied_at_step_zero():
    """`L_phys` is unstable while the DEM is still garbage, so the ramp must actually be
    near zero at the start rather than nominally present."""
    cfg = TrainConfig(batch_size=2, tile_size=64, phys_ramp_steps=5000)
    model = build_model(UNetConfig())
    batch = small_batch()
    hot, _, _ = compute_losses(model, batch, cfg, phys_ramp=1.0)
    cold, _, _ = compute_losses(model, batch, cfg, phys_ramp=1.0 / 5000)
    assert float(hot) > float(cold)


def test_pick_device_returns_something_usable():
    d = pick_device("auto")
    assert d.type in ("cuda", "mps", "cpu")
    assert torch.zeros(2, device=d).sum().item() == 0.0


def test_every_cli_entry_point_starts():
    """Each module with a `main()` must at least parse its arguments. Import-time errors
    in the data-pipeline modules are easy to introduce (they use package-relative imports
    and are run with `python -m`) and never show up in the rest of the suite."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    invocations = [
        ["-m", "data.download", "--list"],
        ["-m", "data.tile", "--help"],
        ["-m", "data.earth", "--help"],
        ["-m", "export.terrain_tiles", "--plan"],
        ["-m", "export.to_cog", "--print-commands"],
        ["-m", "export.demo_tiles", "--help"],
        ["-m", "eval.run_eval", "--help"],
        ["-m", "eval.compare_runs", "--help"],
        ["-m", "eval.panels", "--help"],
        ["train.py", "--help"],
        ["calibrate.py", "--help"],
        ["infer_global.py", "--help"],
    ]
    failures = []
    for argv in invocations:
        r = subprocess.run([sys.executable, *argv], cwd=root, capture_output=True, timeout=120)
        if r.returncode != 0:
            failures.append(f"{' '.join(argv)}: {r.stderr.decode()[-200:]}")
    assert not failures, "\n".join(failures)


def test_raw_loss_scales_reproduce_the_unnormalised_objective():
    """The A/B in the results only means something if both arms run on identical code.
    This is the switch that makes that possible, and it must actually change the balance.

    Eight tiles, so at least one carries a stereo DEM — a two-tile batch often carries
    none, and every metre-valued term is then zero.
    """
    from model.losses import UNIT_SCALES

    model = build_model(UNetConfig())
    batch = small_batch(n=8)
    normalised = TrainConfig(batch_size=8, tile_size=64)
    raw = TrainConfig(batch_size=8, tile_size=64)
    raw.loss_scales = UNIT_SCALES

    _, terms_n, _ = compute_losses(model, batch, normalised, phys_ramp=1.0)
    _, terms_r, _ = compute_losses(model, batch, raw, phys_ramp=1.0)
    assert set(terms_n) == set(terms_r)

    def ratio(terms, a, b):
        return float(terms[a].detach()) / max(float(terms[b].detach()), 1e-9)

    # Roughness is radians and the radiometric residual is decibels. Raw, the roughness
    # term is an order of magnitude below the physics term and contributes nothing;
    # normalised, the two are comparable.
    assert ratio(terms_r, "rms", "phys") < 0.2
    assert ratio(terms_n, "rms", "phys") > 2.0

    # And where stereo exists, metres dwarf decibels until they are normalised.
    if float(terms_r["stereo"].detach()) > 0:
        assert ratio(terms_r, "stereo", "phys") > ratio(terms_n, "stereo", "phys") * 5


def test_the_default_scales_are_the_normalised_ones():
    assert TrainConfig().loss_scales is not None
    from model.losses import SCALES

    assert TrainConfig().loss_scales == SCALES
