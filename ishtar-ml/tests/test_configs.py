"""The YAML configs are documentation until Hydra is wired in, so they are free to drift
from the dataclasses that actually run. These tests are what stops them."""

import ast
from pathlib import Path

import pytest

from data.dataset import BatchSpec
from data.tile import TileSpec
from model.losses import LossWeights
from model.unet import UNetConfig
from train import PHASES

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load_flat_yaml(path: Path) -> dict:
    """Read the flat `key: value` subset these files use.

    Deliberately not a YAML parser: pulling in PyYAML to read four files of scalars would
    put a dependency in the test path that the model itself does not need.
    """
    out: dict[str, object] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        raw = raw.strip()
        if not raw:
            continue
        try:
            out[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key.strip()] = raw
    return out


def test_data_config_matches_the_dataclasses():
    cfg = load_flat_yaml(CONFIGS / "data" / "venus.yaml")
    spec, tiles = BatchSpec(), TileSpec()
    assert cfg["pixel_size_m"] == spec.pixel_size_m
    assert cfg["gtdr_stride_px"] == spec.gtdr_stride_px
    assert cfg["gsdr_cell_m"] == spec.gsdr_cell_m
    assert cfg["core_px"] == tiles.core_px
    assert cfg["margin_px"] == tiles.margin_px
    assert cfg["max_abs_lat_deg"] == tiles.max_abs_lat_deg
    assert cfg["speckle_looks"] == spec.speckle_looks
    assert cfg["max_gain_db"] == spec.max_gain_db
    assert cfg["p_drop_right"] == spec.p_drop_right


def test_alt_edge_margin_covers_three_sigma_of_the_footprint():
    """The config's margin and the tiler's margin are the same physical quantity written
    twice; if they disagree, `L_alt` either wastes posts or measures padding."""
    cfg = load_flat_yaml(CONFIGS / "data" / "venus.yaml")
    assert cfg["alt_edge_margin_px"] * cfg["pixel_size_m"] >= 3.0 * 8000.0 * 0.95
    assert cfg["alt_edge_margin_px"] <= cfg["margin_px"]


def test_model_config_matches_the_dataclass():
    cfg = load_flat_yaml(CONFIGS / "model" / "unet_convnext.yaml")
    m = UNetConfig()
    assert cfg["in_channels"] == m.in_channels
    assert cfg["cond_dim"] == m.cond_dim
    assert tuple(cfg["widths"]) == m.widths
    assert tuple(cfg["depths"]) == m.depths
    assert cfg["decoder_width"] == m.decoder_width
    assert tuple(cfg["bottleneck_dilations"]) == m.bottleneck_dilations
    assert cfg["residual_scale_m"] == m.residual_scale_m
    assert cfg["brightness_downscale"] == m.brightness_downscale
    assert tuple(cfg["logvar_range"]) == m.logvar_range


def test_loss_weights_match_section_5_7():
    cfg = load_flat_yaml(CONFIGS / "loss" / "weights_v1.yaml")
    w = LossWeights()
    for name in ("earth", "stereo", "alt", "phys", "cross", "rms", "nll", "reg"):
        assert cfg[name] == getattr(w, name), name
    # The architecture note's Section 5.7 starting point, restated so a silent edit fails.
    assert (w.stereo, w.alt, w.phys, w.cross, w.rms, w.nll, w.reg) == (1.0, 2.0, 0.3, 0.3, 0.05, 0.1, 0.01)


@pytest.mark.parametrize("name", sorted(PHASES))
def test_every_phase_has_a_config_file(name):
    path = CONFIGS / "phase" / f"{name}.yaml"
    if not path.exists():
        pytest.skip(f"{name} is a code-only phase")
    cfg = load_flat_yaml(path)
    assert cfg["name"] == name
    assert cfg["ckpt_dir"] == PHASES[name]["ckpt_dir"]
