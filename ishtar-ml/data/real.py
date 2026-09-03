"""Training dataset over real Magellan tiles cut by `data.ingest`.

Returns exactly the keys `data.synthetic` returns, minus `z_true` — because on Venus
there isn't one. Everything downstream already handles its absence: `train.evaluate`
reports nothing rather than inventing a substitute, and the honest metrics live in
`eval.run_eval`, which scores against stereo, altimetry, the radiometric residual and
cross-look prediction instead.

Splits are by region, not by tile. Neighbouring tiles share terrain and the ingest cuts
them on a regular lattice, so a random split would put a tile's neighbour in the training
set and flatter every number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .ingest import HELD_OUT, RASTER_KEYS, SCALAR_KEYS, VECTOR_KEYS


def to_memmap_store(npz_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Expand a compressed `.npz` into a directory of `.npy` files.

    `np.savez_compressed` stores each array as a single deflate stream, so *any* element
    access decompresses the whole array. Indexing one tile out of a 278 MB raster per
    sample turned a 2.8 s training step into 13.8 s. Separate `.npy` files can be
    memory-mapped, so a sample reads only the bytes it needs.

    Costs about 4x the disk of the compressed form, which is the right trade for data
    that is read thousands of times.
    """
    npz_path = Path(npz_path)
    out = Path(out_dir) if out_dir else npz_path.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    stamp = out / "_complete"
    if stamp.exists():
        return out
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            np.save(out / f"{k}.npy", z[k])
    stamp.write_text("ok\n")
    return out


class _Store:
    """Memory-mapped view over a directory of `.npy` files, with an `npz` fallback."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.suffix == ".npz":
            path = to_memmap_store(path)
        self.dir = path
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = np.load(self.dir / f"{key}.npy", mmap_mode="r")
        return self._cache[key]


class RealVenus(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, split: str = "train",
                 held_out: tuple[str, ...] = HELD_OUT, crop_px: int | None = None,
                 seed: int = 0):
        self.store = _Store(path)
        regions = np.asarray(self.store["region"])
        held = np.isin(regions, np.asarray(held_out, dtype=regions.dtype))
        keep = held if split == "val" else ~held
        self.index = np.flatnonzero(keep)
        if self.index.size == 0:
            raise ValueError(
                f"split {split!r} is empty; regions present are {sorted(set(regions.tolist()))}"
            )
        self.crop_px = crop_px
        self.seed = seed
        self.split = split
        self.regions = regions

    def __len__(self) -> int:
        return int(self.index.size)

    def summary(self) -> str:
        r = self.regions[self.index]
        uniq, counts = np.unique(r, return_counts=True)
        parts = ", ".join(f"{u} {c}" for u, c in zip(uniq, counts))
        right = float(self.store["has_right"][self.index].mean())
        stereo = float(self.store["has_stereo_dem"][self.index].mean())
        return (f"{self.split}: {len(self)} tiles ({parts}); "
                f"right-look {right:.0%}, stereo DEM {stereo:.0%}")

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        j = int(self.index[i])
        out: dict[str, torch.Tensor] = {}
        # A deterministic crop per (tile, epoch-seed) keeps validation reproducible while
        # still letting training see different parts of a tile across runs.
        r0 = c0 = 0
        n = self.store[RASTER_KEYS[0]].shape[-1]
        if self.crop_px and self.crop_px < n:
            rng = np.random.default_rng(self.seed * 1_000_003 + j)
            r0 = int(rng.integers(0, n - self.crop_px + 1))
            c0 = int(rng.integers(0, n - self.crop_px + 1))
        sl = (slice(r0, r0 + self.crop_px), slice(c0, c0 + self.crop_px)) if self.crop_px else (
            slice(None), slice(None))

        for k in RASTER_KEYS:
            # `.copy()` because the memmap is read-only and torch will not wrap a
            # non-writable buffer without warning about undefined behaviour.
            out[k] = torch.from_numpy(np.array(self.store[k][j][sl], dtype=np.float32))
        for k in SCALAR_KEYS:
            out[k] = torch.tensor(float(self.store[k][j]), dtype=torch.float32)
        for k in VECTOR_KEYS:
            out[k] = torch.from_numpy(np.array(self.store[k][j], dtype=np.float32))
        return out


def coverage_report(path: str | Path) -> dict[str, float]:  # noqa: D401
    """What supervision the store actually contains — worth printing before a run,
    because a loss term with no data in it is silently zero."""
    s = _Store(path)
    n = len(s["lat"])
    left = (s["dn_left"] > 0).mean(axis=(1, 2))
    return {
        "tiles": n,
        "mean_left_look_valid": float(left.mean()),
        "frac_with_right_look": float(s["has_right"].mean()),
        "frac_with_stereo_look": float(s["has_stereo"].mean()),
        "frac_with_stereo_dem": float(s["has_stereo_dem"].mean()),
        "lat_min": float(s["lat"].min()), "lat_max": float(s["lat"].max()),
    }
