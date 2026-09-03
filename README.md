# ISHTAR

Learned high-resolution topography of Venus from Magellan SAR, served as an interactive
CesiumJS globe.

| | |
|---|---|
| The specification | [docs/ISHTAR_ARCHITECTURE.md](docs/ISHTAR_ARCHITECTURE.md) |
| What is built, what is stubbed, what to do next | [docs/STATUS.md](docs/STATUS.md) |
| Phase 0 numbers and what they do not support | [docs/RESULTS.md](docs/RESULTS.md) |
| Findings that changed the code | [docs/DECISIONS.md](docs/DECISIONS.md) |
| Constraints for agents | [CLAUDE.md](CLAUDE.md) |

Nobody has 75 m ground-truth elevation for Venus, so this is not regression to a label.
It is weakly supervised super-resolution: the 75 m signal comes from radar physics
(radarclinometry) and an Earth-trained prior, while Magellan altimetry pins the long
wavelengths and the Herrick stereo DEM pins the middle ones. Every product ships with a
per-pixel uncertainty map, and the 75 m output is a **model-derived candidate, not a
measurement**.

---

## Where this actually stands

The pipeline runs end to end on real Magellan data, and **the model does not yet produce
Venus topography.** On held-out tiles it adds 0.19 m of relief over the altimetry it was
given, and explains the radar image entirely through the nuisance brightness field rather
than through terrain — remove `b(x)` and its physics residual is the baseline's, to three
decimals. It reports a mean 1σ of 100 m alongside that, which is the uncertainty head
correctly saying it does not know.

The cause is measurable: resolvable topographic slope explains about **1% of the variance**
in 75 m Magellan backscatter. The architecture note assumes the radiometric residual is "to
first order, exactly the slope signal you want to invert"; against real data it mostly is
not. See [docs/RESULTS.md](docs/RESULTS.md) for the numbers and what to try next.

Phase 0 on synthetic data reaches +28.5% skill over the bicubic baseline, and that is not
evidence about Venus — the synthetic generator renders through the same forward model the
losses invert.

## What runs today

The physics, the model, the losses, the augmentation, the metrics and the training loop
are implemented and tested end to end against a synthetic Venus that is rendered through
the same forward model the loss inverts. No downloads are needed to run any of it.

```bash
cd ishtar-ml
python -m pytest tests/ -q          # 208 tests, ~60 s
python train.py --phase sanity      # Phase 0: overfit synthetic tiles
python -m data.download --list      # what the real pipeline needs, and how big
python export/terrain_tiles.py --plan
```

```bash
cd ishtar-ml
python -m export.demo_tiles --out ../ishtar-globe/public/tiles \
    --max-level 5 --roi maxwell mead --roi-level 8

cd ../ishtar-globe
npm install && npm run dev          # a real Venus globe, from synthetic terrain
```

The demo tile set is generated, not measured — it exists so the globe, the tiling
conventions and the alignment check can be verified end to end before any download. The
UI says so on screen.

The data ingest (`data/tile.py`, `data/earth.py`) and global inference
(`infer_global.py`) are written against the real products but stop with an explicit
message until the rasters are downloaded — they do not silently produce nothing.

## Reproducing the numbers

```bash
cd ishtar-ml
./scripts/phase0.sh            # ~3 h on an M-series Mac; two controls, two candidates,
                               # the A/B, then the table, calibration and panels
```

Every arm is the same commit — the unnormalised-loss arm differs only by
`--raw-loss-scales` — so the comparison in [docs/RESULTS.md](docs/RESULTS.md) is not
confounded by anything else. Checkpoints are ~250 MB each (model plus EMA at 31 M
parameters) and land in `runs/`, which is gitignored.

## What the sanity phase checks

Phase 0 exists to catch the failures that are invisible in a loss curve:

| Check | Where |
|---|---|
| A slope facing the radar is brighter, and flipping `look_vec` flips `alpha` | `tests/test_physics.py::test_slope_toward_radar_sign` |
| All eight dihedral augmentations move the DEM and `look_vec` together | `tests/test_augment.py::test_dihedral_preserves_radar_geometry` |
| `L_phys` is minimised by the true DEM, and beaten by flat *and* by inverted terrain | `tests/test_losses.py::test_phys_loss_is_minimised_by_the_true_dem` |
| `L_phys` gradients are O(1e-3) dB/m, not exploding or vanishing | `tests/test_losses.py::test_phys_loss_gradient_is_finite_and_nonzero` |
| The footprint operator matches the analytic Gaussian transfer function | `tests/test_physics.py::test_footprint_blur_matches_analytic_transfer_function` |
| The power spectrum separates real detail from invented texture | `tests/test_metrics.py::test_power_spectrum_distinguishes_real_detail_from_noise` |
| Every loss term gets a meaningful share of the objective | `tests/test_loss_balance.py` |
| The EMA of a static model converges to that model, not to its init | `tests/test_train.py::test_ema_of_a_static_model_converges_to_it` |
| Altimetry posts land on the lattice the loss samples | `tests/test_tile.py::test_upsampled_posts_land_on_the_lattice_the_loss_samples` |
| GTDR nodata never becomes a target | `tests/test_dataset.py::test_gtdr_nodata_is_carried_into_the_altimetry_mask` |
| The inference blend is a partition of unity (no tile seams) | `tests/test_infer.py::test_feather_profile_is_a_partition_of_unity` |
| The model beats the bicubic-GTDR baseline | printed at the end of `train.py --phase sanity` |

## Five things that were wrong on the first pass

Recorded because they are the failure modes this codebase is shaped around, and all
three produce a plausible-looking planet rather than an obvious error.

**The look-vector sign.** The spec defines `alpha = atan(grad(z) . look_vec)` as positive
for facets tilted toward the radar. That only holds if `look_vec` points *down-range*,
away from the radar — a radar in the west illuminates eastward, and a slope rising
eastward faces it. Defining it as ground-to-radar, which reads more naturally, inverts
every slope in the model. It is now pinned by a test and stated in `CLAUDE.md`.

**The altimetry footprint is comparable to the tile.** The Magellan altimeter footprint
is ~10 x 20 km; a 512 px tile at 75 m is 38 km. Convolving with a kernel that size means
the boundary condition propagates ~3 sigma (about 25 km) inward, so on a bare tile
essentially every GTDR post measures the padding rather than the terrain. Measured: a
60 m ripple leaks 4.5 m into `L_alt` on a 256 px tile, 0.11 m on 512 px with a margin,
and 0.0007 m on 1024 px with a margin. Tiles are therefore cut at `core + 2 * margin`
(`data/tile.py::TileSpec`) and `loss_alt` masks posts near the border.

Blurring GTDR to match, which cancels the edge effect cleanly, is *wrong*: GTDR already
is the footprint-averaged surface, so blurring it again anchors the model to a doubly
smoothed target and biases the whole planet smooth.

**The eight loss terms are not in the same units.** `L_stereo` and `L_alt` are metres,
`L_phys` is decibels, `L_rms` is radians. Applying Section 5.7's weights to the raw
values gives the radarclinometry term — the only source of sub-kilometre detail over the
80% of Venus with one look and no stereo — 0.5% of the objective, and the roughness term
0.0%. Nothing in the loss curve says so; it simply converges to a smooth planet. Each
term is now divided by its own observational uncertainty, so the weights read as relative
trust. `tests/test_loss_balance.py` pins both the fix and the original failure.

**The saved model was 41% untrained.** `train.py` logs metrics from the live weights;
every downstream tool loads the EMA. A fixed decay of 0.999 keeps `0.999^t` of the
*initial* weights, which after 900 steps is 41% — so the same checkpoint scored +29.5%
live and +4.0% from its own EMA, and the training log showed only the first. The decay now
warms up, the trainer scores both at the end of every run, and `load_weights` refuses an
under-warmed EMA out loud.

**Pyramiding the physics loss biases it.** Rendering from a downsampled DEM is a render
of the block-mean slope, which is not the block mean of the render, because `RV` is
nonlinear in slope. The gap grows with subgrid roughness, so it would land as a
systematic error in tessera — the terrain where it is hardest to notice. The pyramid now
averages the *residual image* instead, which is unbiased and kills speckle just as well.

## Layout

```
ishtar-ml/
  configs/            hydra: data, model, losses, phases
  data/
    download.py       product inventory, sizes, Map-A-Planet 2 clip URLs
    tile.py           windowed reads -> Zarr; enforces DN decode and context margin
    earth.py          Sentinel-1 + GLO-30, degraded to Magellan radiometry
    geometry.py       incidence-vs-latitude models, look vectors, cos(lat) weights
    polar.py          polar stereographic re-tiling for the caps, and the blend weight
    store.py          Zarr tile dataset with spatial (whole-quadrangle) splits
    synthetic.py      self-consistent fake Venus; Phase 0 and CI run on this
    augment.py        dihedral transforms that move look_vec with the rasters
    masks.py          seam, radar-dark, layover and unsupervised masks
    dataset.py        tile dict -> input stack, conditioning vector, targets
  model/
    physics.py        Muhleman law, differentiable renderer, footprint operators
    unet.py           ConvNeXt encoder, FiLM decoder, residual/logvar/brightness heads
    losses.py         the eight terms of Section 5
  train.py            phases 0-3
  calibrate.py        phase 4, temperature scaling of the uncertainty head
  infer_global.py     phase 5, overlapped feathered inference
  eval/
    metrics.py        the Section 7 table, including cross-look PSNR and power spectra
    baselines.py      bicubic GTDR, classical radarclinometry, Earth-only
    run_eval.py       one checkpoint against all three baselines
    compare_runs.py   several checkpoints on one held-out set: the ablation table
    panels.py         qualitative figures
  export/
    quantized_mesh.py Venus-sphere terrain tiler (Route A) and the pyramid driver
    terrain_tiles.py  level/resolution/tile-count planning, and the Route B fallback
    to_cog.py         COG, hillshade, colour relief, the Venus 2000 SRS
    demo_tiles.py     a runnable globe from synthetic Venus, before any download
  scripts/            phase0.sh, demo_globe.sh
  tests/              208 tests, no network and no geo dependencies

ishtar-globe/
  src/venus.ts        the ellipsoid, the geodetic tiling scheme, the sites
  src/layers.ts       imagery layers; no Earth assets anywhere
  src/terrain.ts      non-WGS84 terrain, Route A and the Route B fallback
  src/App.tsx         viewer, layer panel, terrain switch, exaggeration, fly-to
```

## Dependencies

The model, losses and tests need only `torch` and `numpy`. The geo stack (`rasterio`,
`zarr`, `pyproj`, `dask`) is imported lazily inside the data-pipeline modules, so the
test suite runs anywhere.

## Compute notes

Measured on an M-series Mac (MPS), batch of 8 at 128 px: 1.7-2.1 s/step. The same GPU at
256 px collapses to 8.3 s per tile — a 29x per-pixel regression, so MPS is fine for
sanity runs at small tile sizes and useless for the real phases. The architecture note's
budget of 3-5 GPU-days on an A100/H100 stands; plan on CUDA for Phases 1-3.

## Status

- [x] Physics, model, losses, augmentation, metrics, training loop, tests
- [x] Synthetic Venus for Phase 0 and CI
- [x] Globe: Venus ellipsoid, geodetic tiling, layers, terrain switch, swipe, profile,
      gazetteer search, tour, shareable URLs
- [x] Quantized-mesh terrain tiler on the Venus sphere (Route A), with a pyramid driver
- [x] Synthetic demo tile set, so the globe runs before any download
- [x] Headless smoke test including an imagery-alignment check (1.5 px at Mead crater)
- [x] Polar stereographic cap re-tiling and the feathered merge
- [ ] Real data ingest (needs the ~300 GB of USGS/PDS products)
- [ ] Phases 1-3 on a CUDA machine
- [x] Gazetteer search, elevation profile, swipe compare, tour mode, shareable URLs
