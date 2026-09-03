# ISHTAR
### Learned high-resolution topography of Venus from Magellan SAR, served as an interactive CesiumJS globe

Version 0.1 · September 2026 · Architecture and training document

---

## 0. Read this first: what is actually possible, and the recommended framing

**The honest constraint.** Nobody has 75 m ground-truth elevation for Venus. The best you can get is:

| Product | Horizontal res. | Vertical res. | Coverage | Source |
|---|---|---|---|---|
| Magellan SAR FMAP left-look mosaic | ~75 m/px | n/a (backscatter) | ~92-96% | USGS Astrogeology (109 GB GeoTIFF) |
| Magellan SAR FMAP right-look mosaic | ~75 m/px | n/a | ~17% | USGS Astrogeology (110 GB) |
| Magellan SAR FMAP stereo-look mosaic (Cycle 3, steeper incidence) | ~75 m/px | n/a | ~17% | USGS Astrogeology (82 GB) |
| Magellan altimetry GTDR | 4641 m/px grid, footprint ~10 x 20 km | ~50-100 m | ~98% | USGS `Venus_Magellan_Topography_Global_4641m_v02` |
| Herrick et al. stereo-derived DEM | ~1-2 km | ~50-100 m | ~20% | R. Herrick (UAF) public download |
| Magellan GSDR (RMS slope) and GEDR (emissivity) | ~4.6 km | n/a | ~98% | PDS Geosciences |

So the task is not "regression to a label". It is **weakly supervised super-resolution of topography**, where the 75 m signal must come from physics (radar shape-from-shading, called radarclinometry) and from cross-domain priors, while altimetry and stereo pin down the low and mid frequencies.

**Recommended framing (this is the "better idea" you asked for):**

> Train a single-image SAR-to-DEM network on **Earth**, where perfect labels exist (Sentinel-1 backscatter paired with Copernicus GLO-30 DEM), then **fine-tune on Venus** using three Venus-native supervision signals: altimetry consistency at 4.6 km, stereo DEMs at 1-2 km where they exist, and a differentiable radar backscatter model at 75 m. Output a global 75 m elevation estimate *with a per-pixel uncertainty map*, and be explicit that it is a model-derived candidate, not a measurement.

Why this framing is stronger than "just train on Magellan":

1. Earth pretraining gives the network a real prior on how terrain looks in SAR, which pure Venus data cannot provide at 75 m.
2. The uncertainty map is what makes it scientifically honest and visually interesting on the globe (you can show *where* the model is confident).
3. The timing is good. VERITAS (NASA) and EnVision (ESA) are both slated for the early 2030s and will return the first new Venus SAR since 1992. A "what can we squeeze out of the 1992 data before then" story is a strong portfolio narrative for Frankfurt-era data science work.

**Two alternatives, in case you want a different risk profile:**

- **Mars instead of Venus (lower risk, benchmarkable).** Mars has CTX imagery at 6 m, HRSC DTMs at 50-100 m, MOLA at 463 m, and HiRISE DTMs at 1 m. There is published prior art (MADNet, Tao et al. 2021, single-image DTM estimation). You would get real quantitative metrics. Downside: it is optical, not radar, and it has been done.
- **Classical stereo as ground truth (Venus, medium risk).** Run a modern stereo matcher (NASA Ames Stereo Pipeline supports Magellan in principle via ISIS) on the 17% stereo coverage to build your own ~300-500 m DEM, then train the net to reproduce it from single-look SAR and extrapolate to the other 80%. This is a legitimate paper-shaped project on its own. It is folded into the plan below as an optional "Phase 1b".

The rest of this document assumes the recommended framing.

---

## 1. Goals, non-goals, deliverables

**Goal.** A global Venus elevation model at 75-225 m posting with calibrated uncertainty, derived from Magellan SAR, consistent with Magellan altimetry, validated against held-out stereo DEMs, and rendered as an interactive 3D globe.

**Non-goals (v1).** Sub-100 m absolute accuracy anywhere. Full 75 m global terrain tiles (too large; see Section 8). Radiometric re-calibration of the F-BIDRs.

**Deliverables.**
1. `ishtar-ml/` : data pipeline, model, training, evaluation, global inference.
2. `venus_dem_v1.tif` : global COG at 225 m (3x downsample of native output) plus 75 m COGs for ~10 regions of interest.
3. `venus_dem_v1_sigma.tif` : per-pixel 1-sigma uncertainty.
4. `ishtar-globe/` : CesiumJS app (Vite + React + TS) with terrain, SAR imagery, uncertainty layer, compare swipe, feature fly-to, profile tool.
5. A short technical write-up with the evaluation tables in Section 7.

---

## 2. Data

### 2.1 Inputs

**SAR mosaics (USGS Astrogeology, S3-hosted GeoTIFFs).** Three FMAP products at 75 m: left-look (global), right-look (17%), stereo-look (17%). Pixel values are 8-bit DNs encoding backscatter *relative to the Muhleman law*:

```
DN = 5 * (clamp(RV, -20, 30) + 20) + 1
RV = 10*log10( sigma0_measured / sigma0_Muhleman(theta) )   [dB]
sigma0_Muhleman(theta) = 0.0118 * cos(theta) / (sin(theta) + 0.111*cos(theta))^3
```

This matters enormously for the physics loss: **the image is already flattened for the nominal incidence angle assuming a flat surface.** A tilted facet changes the local incidence angle, so the residual you see in RV is (to first order) exactly the slope signal you want to invert. Decode DN back to RV in dB before anything else: `RV = (DN - 1)/5 - 20`. DN = 0 is nodata.

**Altimetry.** GTDR at 4641 m/px, int16 metres, range -2951 to 11687 m, nodata -32768. Effective footprint is 10 x 20 km-ish and varies with latitude, so do *not* treat it as a 4.6 km box average. Model it as a Gaussian footprint (Section 5.3).

**Stereo DEM.** Herrick et al. (2012), 1-2 km horizontal, 50-100 m vertical, ~20% of the planet. Known artefacts: mosaic seam ("noodle") misregistration and radar-dark patches that appear spuriously low. Use a robust loss and a validity mask.

**Auxiliary.** GSDR (RMS slope at 4.6 km) as a slope-statistics target; GEDR (emissivity) as an optional input channel that tells the net about surface dielectric properties (high-emissivity-anomaly highlands behave differently); IAU nomenclature GeoJSON for feature names.

**Geometry metadata.** You need, per pixel, the nominal incidence angle and the look direction. Cycle 1 incidence varied with latitude (roughly 45 deg near periapsis around 10 deg N, decreasing toward the poles to under 20 deg). Cycle 3 stereo passes used a deliberately different incidence. Recover these from the F-BIDR / mosaic labels rather than from memory; if a pixel-level angle map is not shipped with the mosaic, fit a smooth latitude-dependent model per cycle and store it as a raster. The look direction is roughly east-west because the orbit was near-polar; store it as a unit vector per tile.

### 2.2 Earth pretraining set

| Component | Choice | Notes |
|---|---|---|
| SAR | Sentinel-1 GRD IW, VV, terrain-*un*corrected (GRD in ground range, or use SLC-derived sigma0 without RTC) | RTC would remove exactly the slope signal you want to learn. C-band 5.4 GHz vs Magellan S-band 2.4 GHz: different wavelength, so this is a prior, not a twin. |
| DEM | Copernicus GLO-30 | Downsample to 75 m |
| Regions | Unvegetated, volcanic/tectonic terrain: Iceland, Hawaii, Afar, Atacama, Tibet, Kamchatka, Canary Islands, Ethiopian Rift, Nevada basin-and-range | Vegetation breaks the backscatter-slope relation |
| Incidence | Sentinel-1 IW covers 29-46 deg; keep all, condition the model on it | Overlaps most of Magellan Cycle 1 |
| Volume | 20-40k tiles of 512 x 512 at 75 m | Enough for a strong prior |

Degrade Earth SAR to look like Magellan: resample to 75 m, add multiplicative speckle (gamma distributed, ~4-8 looks), quantise to Magellan's 8-bit DN encoding after applying the Muhleman flattening with the true incidence, and add low-frequency radiometric striping to imitate orbit-to-orbit gain differences.

### 2.3 Tiling and storage

- Reproject everything to a common equal-area-friendly working grid. The FMAP mosaics are cylindrical; keep that for tiling, but weight losses by cos(latitude) and avoid training tiles above ~80 deg latitude where distortion is severe (handle poles at inference with polar stereographic re-tiling).
- Tile size 512 x 512 px at 75 m = 38.4 km square. Global count is roughly 300k tiles; you will not need all of them for training.
- Store as Zarr (chunks 512 x 512, one array per channel) or as a directory of COGs. Zarr on local NVMe is the pragmatic choice.
- Every tile carries: `sar_left`, `sar_right` (+mask), `sar_stereo` (+mask), `gtdr_up` (bicubic upsample to 75 m), `stereo_dem` (+mask, resampled to 75 m but only trusted at 1 km), `theta_left`, `theta_right`, `theta_stereo`, `look_vec`, `emissivity`, `rms_slope`, `lat`.

### 2.4 Splits

Split **spatially**, never randomly by tile. Hold out whole 12 x 12 deg FMAP quadrangles that contain stereo coverage. Suggested held-out validation regions: parts of Ovda Regio (tessera), a plains region with small volcanoes, and one crater field. Keep Maxwell Montes and Maat Mons as *demo* regions, not test regions, since you will look at them constantly.

---

## 3. Physical model (the part that makes this work)

For a facet with slope angle `alpha` in the range (look) direction, positive when tilted toward the radar, the local incidence angle is approximately

```
theta_local = theta_nominal - alpha_range
```

and the expected flattened backscatter is

```
RV_expected(dB) = 10*log10( M(theta_local) / M(theta_nominal) ) + b(x)
```

where `M` is the Muhleman law above and `b(x)` is a smooth, slowly varying intrinsic-brightness term (roughness, dielectric constant). This is a **differentiable renderer**: DEM in, SAR-like image out. It is exactly the forward model of radarclinometry.

Design consequences:

- The slope along the look direction is well constrained by the image; the slope *across* the look direction is not. That is why the coarse DEM, the stereo DEM, and the learned Earth prior are needed: they supply the cross-track component.
- `b(x)` is a nuisance field. Let the network predict it as a second low-resolution output head (say 1/16 resolution, bilinearly upsampled) with a strong smoothness penalty, so it cannot absorb slope information.
- Layover and foreshortening at steep slopes (Maxwell Montes) violate the small-slope assumption. Mask the physics loss where `|alpha| > theta_nominal - 5 deg` and let the stereo/altimetry terms carry those pixels.
- When two looks exist (left + right), the slope sign flips between them. This gives a *cross-look consistency* loss that is close to a stereo constraint in strength, and it is free.

---

## 4. Model

### 4.1 Architecture: conditional residual U-Net with uncertainty head

Predict the **residual** over the upsampled altimetry, not absolute elevation:

```
z_hat(x) = Up(GTDR)(x) + f_theta(inputs)(x)
```

This keeps the low frequencies right by construction and lets the net focus on the 100 m to 10 km band.

```
Inputs  (C x 512 x 512):
  sar_left_dB, mask_left
  sar_right_dB, mask_right
  sar_stereo_dB, mask_stereo
  gtdr_up (normalised), emissivity (normalised)
  theta_left, theta_right, theta_stereo  (as sin, cos)
  look_vec (2 channels, constant per tile)
  lat encoding (2 channels)

Encoder : ConvNeXt-T or Swin-T backbone (ImageNet init is fine; the Earth stage re-learns it)
Decoder : U-Net decoder with skip connections, GroupNorm, GELU
Conditioning : FiLM on incidence angle and look direction at every decoder stage
                (angle is a global-ish variable; feeding it only as an image channel
                 makes the net learn it slowly)

Heads:
  h_res   : residual elevation, 1 x 512 x 512, metres, tanh-scaled to +/- 1500 m
  h_logv  : log variance of elevation, 1 x 512 x 512  (heteroscedastic uncertainty)
  h_b     : intrinsic brightness field, 1 x 32 x 32, bilinear upsampled

Parameters : ~30-60 M
```

Receptive field must exceed the altimetry footprint (20 km = ~270 px), which a 5-level U-Net with a transformer-ish encoder gives you. If it does not, add a dilated bottleneck.

### 4.2 Why not a diffusion / flow-matching super-resolver (v2 option)

A conditional diffusion model produces sharper, more plausible terrain and gives you samples for uncertainty for free. But it hallucinates confidently, it is slower for 300k tiles, and the physics loss is harder to inject cleanly. Ship the deterministic U-Net first, then consider a flow-matching refiner conditioned on the U-Net output as v2. Keep the same losses; use them as guidance terms at sampling time.

### 4.3 Augmentation rules (these are not optional)

- Flips and 90-degree rotations must be applied *jointly* to the SAR tile, the DEM, the incidence rasters, **and** the look vector. A horizontal flip negates the range-direction slope sign; if you forget to flip `look_vec`, you train the net on inverted physics.
- Random additive dB offset (+/- 3 dB) and multiplicative speckle to make the net robust to gain striping.
- Random dropout of the right/stereo look channels (set to zero + mask) so the net works on left-only pixels, which is 80% of the planet.

---

## 5. Losses

Total loss:

```
L = w_e * L_earth          (Earth stage only; L1 + gradient L1 to GLO-30)
  + w_s * L_stereo         (Venus, where stereo DEM exists)
  + w_a * L_alt            (Venus, everywhere)
  + w_p * L_phys           (Venus, everywhere, masked)
  + w_x * L_cross          (Venus, where two looks exist)
  + w_r * L_rms            (Venus, everywhere)
  + w_u * L_nll            (uncertainty calibration)
  + w_t * L_reg            (smoothness of h_b, curvature TV on z_hat)
```

### 5.1 `L_stereo`

Downsample `z_hat` to the stereo DEM's trusted scale (1 km, Gaussian) and compare with Charbonnier loss plus a gradient (slope) term. Mask seams and radar-dark artefacts (derive the mask from the left-look SAR: pixels below a dB threshold and their 3-pixel dilation).

### 5.2 `L_alt`

Convolve `z_hat` with an anisotropic Gaussian approximating the altimeter footprint (sigma_x ~ 4 km along-track direction, sigma_y ~ 8 km cross-track, rotated to the orbit direction), sample at GTDR posts, L1 against GTDR. This is the anchor that prevents drift.

### 5.3 `L_phys` (radarclinometry)

For each available look `k`:

```
alpha_k     = atan( grad(z_hat) . look_vec_k )          # slope toward radar
theta_loc_k = theta_k - alpha_k
RV_pred_k   = 10*log10( M(theta_loc_k) / M(theta_k) ) + Up(h_b)
L_phys_k    = mean( mask_k * valid_k * Huber( RV_pred_k - RV_obs_k ) )
```

where `valid_k` removes layover pixels and nodata. Compute gradients with a Sobel operator on the 75 m grid; `M` is implemented in PyTorch and is differentiable in `theta`. Do this loss at full resolution and also on a 4x downsampled pyramid level to stabilise early training.

### 5.4 `L_cross`

Where left and right looks coexist, `L_phys_left + L_phys_right` share one `z_hat`, which already couples them. Add an explicit term: render the right-look image from the left-look-only prediction (run the net with right channels dropped) and compare to the observed right-look. This trains the left-only pathway, which is what the other 80% of the planet will use.

### 5.5 `L_rms`

RMS slope of `z_hat` inside each 4.6 km GSDR cell versus the GSDR value. Weak (w_r small); it mostly stops the net from producing terrain that is uniformly too smooth or too rough.

### 5.6 `L_nll`

Gaussian negative log-likelihood of the stereo DEM under `N(z_hat, exp(h_logv))`, applied only where stereo exists, plus a small penalty pushing `h_logv` up where no stereo and no second look exist. After training, calibrate with temperature scaling on the validation quads and report coverage at 1 and 2 sigma.

### 5.7 Suggested weights (starting point)

`w_s = 1.0, w_a = 2.0, w_p = 0.3, w_x = 0.3, w_r = 0.05, w_u = 0.1, w_t = 0.01`. Ramp `w_p` from 0 to 0.3 over the first 5k Venus steps; the physics loss is unstable when the DEM is still garbage.

---

## 6. Training plan

| Phase | Data | Steps | Notes |
|---|---|---|---|
| 0. Sanity | 200 Earth tiles | 2k | Overfit; verify that flipping `look_vec` flips predicted slope sign. Verify `L_phys` gradient magnitudes. |
| 1. Earth pretrain | 20-40k Earth tiles | 60-100k | AdamW, lr 3e-4 cosine, bf16, batch 16 x 512^2 on one 80 GB GPU or batch 8 on a 24 GB card. |
| 1b. (optional) ASP stereo | Own stereo DEM from Cycle 1 + Cycle 3 | | Adds a second, sharper Venus target. Skip in v1 if time is short. |
| 2. Venus fine-tune, stereo regions | Tiles with stereo DEM | 30k | All losses on. Lower lr (1e-4). |
| 3. Venus fine-tune, global | Random 60k tiles worldwide | 50k | Stereo loss only where available; physics + altimetry everywhere. |
| 4. Calibrate | Validation quads | | Temperature scaling of `h_logv`. |
| 5. Global inference | ~300k tiles, 64 px overlap, feathered blend | ~6-12 h on one GPU | Write directly into a Zarr, then GDAL to COG. |

Tooling: PyTorch 2.x, `torch.compile`, Lightning or a plain loop, Hydra configs, Weights & Biases, `rasterio`/`rioxarray`, `zarr`, `dask` for the tiling pipeline. EMA of weights for inference.

Compute budget: the whole thing fits in roughly 3-5 GPU-days on an A100/H100, or a week on a 4090. Fine-tuning phases are cheap; the Earth stage and global inference dominate.

Data budget: ~300 GB of raw mosaics plus ~150 GB of tiled Zarr. Use the USGS Map-A-Planet 2 clipping service to pull regions first and only download the full global mosaic when you start Phase 3.

---

## 7. Evaluation

Report all of these on the held-out quads; a single "RMSE" number is not enough for a super-resolution-without-labels problem.

| Metric | What it tells you |
|---|---|
| MAE / RMSE vs stereo DEM at 1 km | Mid-frequency accuracy |
| Slope MAE vs stereo DEM | Whether relief is right, not just heights |
| Altimetry residual at 4.6 km | Drift check (should be < 30 m) |
| Physics residual (dB) on held-out tiles | How much of the image the DEM explains |
| Cross-look PSNR (left-only prediction rendered as right-look) | The honest test of 75 m detail: does the DEM predict an image it never saw? |
| Radially averaged power spectrum of `z_hat` vs stereo DEM | Are you adding real high frequencies or noise? |
| Uncertainty calibration (1-sigma coverage) | Should be ~68% on stereo pixels |
| Baselines | (a) bicubic GTDR, (b) classical radarclinometry with a fixed `b`, (c) Earth-only model with no Venus fine-tune |

Qualitative panels for the write-up: Maxwell Montes, Maat Mons, Alpha Regio tessera, Artemis Corona rim, Mead crater, a pancake-dome field in Alpha/Eistla. Show SAR, GTDR, prediction, uncertainty, and stereo DEM side by side.

---

## 8. Products for the globe

Native 75 m global output is ~1.3e11 pixels (~250 GB int16). You do not want to tile that as terrain. Plan:

- **Global**: 225 m elevation COG (~28 GB) and 225 m uncertainty COG, plus a 450 m hillshade and a colour-relief raster for imagery layers.
- **Regions of interest** (10-15 boxes, ~500 x 500 km each): 75 m elevation COGs.
- **Terrain tiles** (quantized-mesh): global to level 9 (~580 m per vertex at the equator), ROIs to level 12 (~73 m per vertex). Level 12 globally would be ~33 million tiles; do not.
- **Imagery tiles** (XYZ, geodetic profile, PNG/WebP): left-look SAR to level 12 in ROIs and level 9 globally; hillshade, colour relief, uncertainty, and the stereo-coverage mask to level 9. `gdal2tiles.py --profile=geodetic`.
- Host on any static object store or a local server; Cesium ion is optional for imagery but see the ellipsoid caveat below before uploading terrain there.

---

## 9. The Venus globe in CesiumJS (built with Claude Code + Cesium skills)

### 9.1 Setup

1. `npm create vite@latest ishtar-globe -- --template react-ts`, add `cesium` and `vite-plugin-cesium`.
2. Install Cesium's official agent skills into the project: clone `github.com/CesiumGS/cesiumjs-skills` and copy into `.claude/skills/`. Cesium also publishes a "Build a CesiumJS App with AI" tutorial with the exact prompts and setup for Claude Code; follow it for the skeleton, then diverge.
3. Optional: the community `cesium-skill` MCP server (Jastman) for live ion asset checks, and Cesium's `cesium-ai-integrations` repo for reference patterns.

### 9.2 Non-negotiable Venus specifics

**Ellipsoid.** Venus is effectively a sphere of radius 6051.8 km. Set the default ellipsoid *before* creating the Viewer:

```ts
import { Ellipsoid, Cartesian3, Viewer, CesiumTerrainProvider, UrlTemplateImageryProvider, GeographicTilingScheme } from "cesium";

const VENUS = Ellipsoid.fromCartesian3(new Cartesian3(6051800, 6051800, 6051800));
Ellipsoid.default = VENUS;   // CesiumJS supports non-Earth defaults (MOON and MARS ship as presets)
```

Everything downstream (`Cartesian3.fromDegrees`, camera, terrain, imagery tiling scheme) must be given `ellipsoid: VENUS` where the API takes it.

**Terrain on a non-WGS84 body: verify this early.** Most quantized-mesh tilers (Cesium Terrain Builder and forks) compute tile bounding spheres and horizon-occlusion points assuming WGS84. `CesiumTerrainProvider.fromUrl(url, { ellipsoid: VENUS })` will decode heights correctly, but mis-sized bounding volumes can cause culling glitches. Two routes:
- *Route A (correct):* write a small Python tiler using the `quantized-mesh-tile` package or your own encoder, computing bounds on the Venus sphere. A day of work; Claude Code can do most of it against the quantized-mesh spec.
- *Route B (pragmatic fallback):* keep WGS84 in the tiler, scale elevation values by `6371.0 / 6051.8` so relative relief is preserved on an Earth-sized globe, and label it clearly. The globe will be 5% too big and nobody will notice visually. Try A, keep B in your pocket.

**Look and feel.** Turn off the Earth defaults: no Bing, no Cesium World Terrain, `skyAtmosphere` either off or tinted with a pale yellow hue and high brightness shift to suggest the sulphuric cloud deck, `globe.enableLighting = true` with a fixed light direction (Venus rotates once per 243 days; a real sun position is boring), `scene.verticalExaggeration` slider defaulting to 1.0 (Venus relief is real enough at Maxwell; do not fake it by default).

### 9.3 Feature list (v1)

| Feature | Cesium mechanism |
|---|---|
| Base imagery: left-look SAR | `UrlTemplateImageryProvider` with `GeographicTilingScheme({ ellipsoid: VENUS })` |
| Layer toggles: right-look SAR, colour relief, hillshade, uncertainty, stereo-coverage mask, emissivity | `ImageryLayerCollection`, alpha sliders |
| Before/after swipe: bicubic GTDR terrain vs learned terrain | Two terrain providers is awkward; instead swipe *imagery* hillshades via `ImageryLayer.splitDirection` + `scene.splitPosition`, and swap `terrainProvider` with a toggle |
| Fly to named features | IAU gazetteer GeoJSON -> searchable list -> `camera.flyTo` with `Rectangle` on VENUS ellipsoid |
| Elevation profile tool | Click two points, `sampleTerrainMostDetailed`, draw with a small chart component |
| Uncertainty "confidence fog" | Uncertainty raster as a semi-transparent imagery layer, ramp from clear (low sigma) to grey (high sigma) |
| Terrain exaggeration | `scene.verticalExaggeration` |
| Cinematic tour | Bookmarked camera paths through 6-8 sites, `camera.flyTo` chain |
| Share links | Camera state and active layers in the URL hash |

### 9.4 Claude Code work plan (each item is one session)

1. Scaffold Vite + React + TS + Cesium; render a blank Venus sphere with `Ellipsoid.default` set, no Earth assets, dark space background. Acceptance: `Cartesian3.fromDegrees(0, 90)` has magnitude 6051.8 km.
2. Add the SAR imagery layer from local XYZ tiles; verify alignment against a known feature (Mead crater at 12.5 N, 57.2 E).
3. Terrain: implement or adapt a quantized-mesh tiler for the Venus sphere (Route A). Load level 0-9 globally; check no culling artefacts at the limb.
4. ROI high-res terrain and imagery; test Maxwell Montes.
5. Layer panel, opacity, uncertainty layer, swipe compare.
6. Gazetteer search and fly-to; profile tool.
7. Tour mode, URL state, polish per the frontend-design skill; deploy as static site.

Prompting tip for Claude Code on this project: put the ellipsoid constant, the tiling scheme, and the "no Earth assets" rule in `CLAUDE.md` at the repo root. Those are the three things an agent trained on Earth-centric Cesium examples will otherwise regress on.

---

## 10. Repository layout

```
ishtar/
  CLAUDE.md                     # constraints for agents: ellipsoid, projections, no Earth assets
  ishtar-ml/
    configs/                    # hydra: data, model, losses, phases
    data/
      download.py               # wget from USGS S3, checksum
      tile.py                   # reproject, decode DN->dB, build zarr
      earth.py                  # Sentinel-1 + GLO-30 tiles, Magellan-style degradation
      geometry.py               # incidence angle and look-vector rasters
    model/
      unet.py                   # ConvNeXt/Swin encoder, FiLM decoder, three heads
      physics.py                # Muhleman law, differentiable renderer, footprint kernels
      losses.py
    train.py                    # phases 0-3
    calibrate.py                # phase 4
    infer_global.py             # overlap-tiled inference into zarr
    eval/                       # metrics, spectra, baselines, qualitative panels
    export/                     # zarr -> COG, hillshade, colour relief, terrain tiles
  ishtar-globe/
    .claude/skills/cesiumjs-skills/
    src/
      venus.ts                  # ellipsoid, tiling scheme, constants
      layers.ts
      terrain.ts
      features/                 # search, profile, swipe, tour
    public/tiles/               # or an object-store URL
```

---

## 11. Risks and how the plan handles them

| Risk | Mitigation |
|---|---|
| Net hallucinates plausible-looking but wrong 75 m relief | Cross-look PSNR metric; uncertainty head; power-spectrum comparison; report at 225 m as the headline product and 75 m as "candidate" |
| Earth prior transfers badly (wavelength, no vegetation on Venus, different roughness regime) | Emissivity input channel; `h_b` nuisance head; Venus fine-tune phases; ablate Earth pretraining explicitly |
| Mosaic seams and gain striping leak into topography | dB-offset augmentation; seam mask from SAR; `h_b` absorbs low-frequency brightness |
| Altimetry footprint mismodelled | Treat footprint sigma as a tunable, check the 4.6 km residual map for striping in the orbit direction |
| Polar distortion | Train equatorward of 80 deg; re-tile polar caps in polar stereographic for inference |
| Terrain tiling on a non-WGS84 ellipsoid | Verify in week 1 of globe work; Route B fallback |
| Data volume | Start with Map-A-Planet 2 clips of 5-6 regions; only go global once Phase 2 metrics look right |

---

## 12. Milestones

| Week | Milestone |
|---|---|
| 1-2 | Data pipeline for 6 regions (3 with stereo), Earth set for 5 regions, Phase 0 sanity passing |
| 3-4 | Earth pretrain done; first Venus fine-tune on stereo regions; first metric table |
| 5 | Globe scaffold with Venus ellipsoid, SAR imagery, GTDR terrain at level 8 |
| 6-7 | Phase 3 global fine-tune; global inference at 225 m; ROIs at 75 m |
| 8 | Terrain and imagery tiles exported; learned terrain in the globe; swipe compare |
| 9 | Uncertainty layer, gazetteer, profile tool, tour; write-up and evaluation panels |
| 10 | Deploy; portfolio page; optional v2 (flow-matching refiner or ASP stereo target) |

---

## 13. Key references and data links

- USGS Astrogeology, Venus Magellan SAR FMAP Left/Right/Stereo Look Global Mosaics 75 m (S3 GeoTIFFs, DN encoding documented on the product pages)
- USGS Astrogeology, Venus Magellan Global Topography 4641 m v02 (GTDR)
- PDS Geosciences Node, Magellan GxDR products (GTDR, GSDR, GEDR)
- Herrick, Stahlke, Sharpton (2012), "Fine-scale Venusian topography from Magellan stereo data", Eos 93(12); data via R. Herrick's UAF page
- Muhleman (1964) backscatter law; Ford and Pettengill (1992) on Magellan altimetry and radiometry
- Tao et al. (2021), MADNet: single-image DTM estimation for Mars (closest ML prior art)
- CesiumGS/cesiumjs-skills (agent skills), Cesium "Build a CesiumJS App with AI" tutorial, Cesium `Ellipsoid.default` API
- Cesium quantized-mesh terrain format specification
