# Configs

Hydra-shaped, and the values here are the documented ones — but `train.py` reads plain
dataclasses (`TrainConfig`, `BatchSpec`, `LossWeights`, `UNetConfig`), not these files.
Hydra is an optional dependency and the model has to run without it.

That means these files can drift from the code, so `tests/test_configs.py` asserts they
agree. If you change a default in the dataclass, the test tells you to change it here
too. Wiring Hydra properly is the fix; until then this is the guard.

| file | mirrors |
|---|---|
| `data/venus.yaml` | `data.dataset.BatchSpec`, `data.tile.TileSpec` |
| `model/unet_convnext.yaml` | `model.unet.UNetConfig` |
| `loss/weights_v1.yaml` | `model.losses.LossWeights` |
| `phase/*.yaml` | `train.PHASES` |
