# Status

What is built and verified, what is stubbed and why, and what the next person should do.

## Built and tested

| | state |
|---|---|
| Radar physics: Muhleman law, differentiable renderer, footprint operators | ✅ 208 tests |
| Model: ConvNeXt encoder, FiLM decoder, residual/logvar/brightness heads (31 M) | ✅ |
| All eight loss terms of Section 5, dimensionally normalised | ✅ |
| Augmentation that moves `look_vec` with the rasters | ✅ all 8 dihedrals |
| Synthetic Venus: self-consistent, rendered through the same forward model | ✅ |
| Training driver, phases 0–3, warm-started EMA, warm-start, train/val reporting | ✅ |
| A/B switch for the loss normalisation, so the comparison runs on one commit | ✅ |
| Phase 4 uncertainty calibration | ✅ 1σ 47.6% → 69.2% |
| Section 7 metric table, three baselines, power spectra, qualitative panels | ✅ |
| Overlap-feathered global inference; polar cap re-tiling and merge | ✅ |
| Quantized-mesh terrain tiler on the Venus sphere (Route A), global + ROI pyramids | ✅ |
| CesiumJS globe: layers, terrain switch, swipe, profile, gazetteer, tour, URLs | ✅ builds + smoke test |
| Globe unit tests under `node --test`, no framework, no build step | ✅ 21 tests |
| Synthetic demo tiles: 8 layers, global + region-of-interest terrain pyramids | ✅ |
| Imagery alignment check (Mead crater, 1.5 px) | ✅ automated |

## Real data: what was actually needed

The headline number in the architecture note is ~300 GB. The real figure is **3.1 GB**,
because the USGS mosaics are tiled Cloud-Optimised GeoTIFFs and a window read fetches
only the 256 x 256 internal tiles it touches.

| | size | how |
|---|---|---|
| 75 m SAR, left + right look | **~130 MB** | 1 059 windowed reads from S3; never downloaded |
| GTDR + GSDR + GEDR, global | 0.18 GB | downloaded whole; they are small |
| Herrick stereo DEM | 2.97 GB | the only bulk item worth having |

Two further economies worth knowing. The `_jpeg` mosaic variants are the *same* 75 m
resolution with JPEG compression inside the GeoTIFF — 117 GB becomes 16.8 GB — so even a
full download is a seventh of the naive figure. And the 225 m global mosaic (13 GB) buys
a whole-planet product if you want one; it was dropped here only because the link
measured about 1 MB/s.

`data/sources.py` holds the products and the lon/lat sampling; `data/ingest.py` cuts
tiles; `data/real.py` is the training dataset. `data/tile.py` remains the idealised-grid
version and its tests, superseded for real work by `sources.py`.

**Earth pretraining** (`data/earth.py`) is still stubbed: Sentinel-1 staging needs a STAC
client and a few hundred GB of scratch.

**Phases 1–3.** They need the real tiles and a CUDA machine; MPS is ~30× worse per pixel
above 128 px tiles. `train.py` refuses these phases rather than silently training on the
synthetic set and reporting numbers that mean nothing.

**Hydra.** The YAML configs are documentation; the dataclasses run. `tests/test_configs.py`
asserts they agree so they cannot drift while that is true.

**Latitude weighting.** Section 2.3 asks for losses weighted by `cos(latitude)`, to undo
a cylindrical grid's over-representation of high latitudes. `geometry.cos_lat_weight`
exists; it is not wired into `build_batch` yet, and deliberately so — the synthetic
generator assigns each tile a latitude but does not distort it, so applying the weight
there would correct for a problem the data does not have. Wire it in with the first real
tiles.

**Per-tile geometric augmentation.** One dihedral transform is drawn per batch rather than
per tile. Correctness is unaffected — the look vectors move with the rasters either way —
but a batch of eight sees one orientation. Worth revisiting if the model turns out to be
orientation-sensitive on real mosaics, where the gain striping is directional. The
radiometric offset *is* per tile, because a shared one could be absorbed by a batch
statistic rather than by the brightness head.

## What to do next, in order

0. **Get the incidence angles from the F-BIDR labels.** The placeholder profile is still
   in use and the stereo-based calibration cannot replace it (DECISIONS §15). This is now
   the highest-value open item, because the physics loss reads those angles directly.

1. **Finish the stereo DEM download.** It is the only source of signal for `L_nll`, so
   without it the uncertainty head trains on nothing — and the uncertainty map is what
   makes a model-derived DEM honest. `data/sources.py` handles a partial file by treating
   rows past the downloaded extent as nodata, so it is usable while it arrives.
2. **Re-ingest with stereo** (`python -m data.ingest`), which turns on `L_stereo` and
   `L_nll` over the ~20% with coverage.
3. **Scale up the ingest.** 1 059 tiles is what fits an evening; `--per-region` caps it
   and the regions hold 27 000 at 512 px. Transfer is about 130 kB per tile.
4. **Stage the Earth set** and run Phase 1 on CUDA. Not RTC products. Not vegetated
   terrain. This is the one part of the recommended framing still untested.
5. **Phases 2 and 3**, then calibrate, then global inference.
6. **Watch the EMA.** `train.py` now scores the moving average alongside the live weights
   at the end of every run, and the two should agree closely. If they diverge, the average
   is not what the log describes — that is how a checkpoint that was 41% initialisation
   went unnoticed here (see [DECISIONS.md](DECISIONS.md) §8).
7. **Watch the brightness head.** `h_b` is deliberately 1/16 resolution so it cannot
   absorb slope. The measured cost on synthetic data is ~13% more elevation error on
   tiles with more intrinsic-brightness variation. On real mosaics, with gain striping and
   genuine dielectric contrasts, expect it to be worse, and check the correlation between
   elevation error and the emissivity channel before adding capacity there.
8. **Re-check the loss balance** on real data. The normalisers in `losses.LossScales` are
   each observation's uncertainty; the Herrick DEM's real error is what it is, not the
   75 m the literature quotes, and `tests/test_loss_balance.py` will tell you if a term
   has gone decorative again.

## Known limits of the Phase 0 evidence

The synthetic set has the right physics and roughly the right statistics, and nothing
else. It has no mosaic seams, no orbit-to-orbit gain steps, no real layover, and no
terrain type that a fractal does not produce. It cannot tell you whether the Earth prior
transfers, because there is no Earth stage in it. See [RESULTS.md](RESULTS.md) for what
the numbers do and do not support.
