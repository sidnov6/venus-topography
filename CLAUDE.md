# ISHTAR — constraints for agents

Learned high-resolution topography of Venus from Magellan SAR, served as a CesiumJS globe.
Full spec: `docs/ISHTAR_ARCHITECTURE.md`. Read it before changing anything structural.

## Planetary constants — never substitute Earth values

- Venus is a **sphere**, radius **6051800 m**. No flattening, no WGS84.
- Magellan FMAP native posting: **75 m/px**. GTDR altimetry: **4641 m/px**.
- Venus has no ocean and no sea level: elevations are referenced to the 6051.8 km sphere.

## Globe rules (`ishtar-globe/`)

- Set `Ellipsoid.default` to the Venus sphere **before** constructing the `Viewer`.
- Pass `ellipsoid: VENUS` to every API that accepts one — `GeographicTilingScheme`,
  `CesiumTerrainProvider.fromUrl`, `Cartesian3.fromDegrees`, camera helpers.
- **No Earth assets.** No Bing imagery, no Cesium World Terrain, no Ion default token
  paths, no `createWorldTerrainAsync`. If a snippet from an Earth tutorial pulls one in,
  it is wrong here.
- Imagery tiling scheme is **geodetic** (2x1 root), not Web Mercator. Venus tiles are
  produced with `gdal2tiles.py --profile=geodetic`.

## SAR radiometry rules (`ishtar-ml/`)

- FMAP pixels are 8-bit DNs of Muhleman-flattened backscatter, **not** raw sigma0:
  `RV_dB = (DN - 1) / 5 - 20`, valid range -20..30 dB, `DN == 0` is nodata.
  Decode to dB at ingest; never train on raw DNs.
- Do not apply radiometric terrain correction to Earth pretraining SAR. RTC removes the
  slope signal the model exists to learn.
- `look_vec` is the horizontal **down-range** direction of the beam — pointing *away*
  from the radar — so `alpha = atan(grad(z) . look_vec)` is positive for facets tilted
  toward the radar. A radar in the west illuminates eastward; a slope rising eastward
  faces it.
  Any flip/rotation augmentation must transform `look_vec` jointly with the rasters.
  Getting this sign wrong trains inverted physics and still looks plausible.

## Modelling rules

- The network predicts a **residual** over upsampled GTDR, never absolute elevation.
- Altimetry is compared through an anisotropic Gaussian **footprint** (~10 x 20 km),
  never as a 4.6 km box average.
- **Every loss term is divided by its own observational uncertainty**
  (`losses.LossScales`) before Section 5.7's weights are applied. The terms are metres,
  decibels and radians; on raw values `w_p = 0.3` gives the radarclinometry term 0.5% of
  the objective and the model quietly converges to a smooth planet.
  `tests/test_loss_balance.py` pins this.
- **Tiles carry a context margin.** The altimeter footprint is comparable to a 512 px
  tile, so `L_alt` on a bare tile measures the padding. Cut at `core + 2 * margin` and
  set `alt_edge_margin_px`. Never "fix" this by blurring GTDR to match — GTDR already is
  the footprint-averaged surface, and blurring it again biases the planet smooth.
- **Inference uses the EMA, so score the EMA.** The training log reports the live
  weights; a fixed-decay EMA of a short run is partly the initialisation and nothing else
  shows it. `train.py` prints both at the end of a run and `train.load_weights` refuses an
  under-warmed average.
- Splits are **spatial** (whole quadrangles). Never split tiles randomly.
- Maxwell Montes and Maat Mons are demo regions, not test regions.

## House style

- PyTorch + numpy core; heavy geo deps (`rasterio`, `zarr`, `pyproj`) are imported lazily
  inside the data-pipeline modules so the model and tests run without them. Modules that
  use package-relative imports run as `python -m data.tile`, not as scripts.
- Tensors are `(B, C, H, W)`, float32, metres for elevation and dB for backscatter.
- In the globe, framework-free logic stays out of the Cesium-importing modules: pure
  functions in `gazetteer.ts` / `profile.ts` / `urlState.ts`, bindings in the `*.cesium.ts`
  siblings. That split is what lets `npm test` run the source under `node --test` with no
  build step and no runner.
- Relative imports in `ishtar-globe/src` carry explicit `.ts` extensions, for the same
  reason.

## Before you claim something works

The three checks that catch what looks fine:

- `python -m pytest tests/ -q` in `ishtar-ml` — 208 tests, no network, no geo stack.
- `npm test && npm run typecheck && npm run build` in `ishtar-globe`.
- `npm run smoke` against a running dev server — verifies the Venus ellipsoid is the
  default, terrain tiles parse, nothing 404s, and the imagery is aligned (a marker drawn
  at Mead crater's coordinates lands 1.5 px from a camera placed at the same coordinates).
