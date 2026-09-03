# Decisions and findings

Things that were not obvious from the architecture note, discovered while building
against it. Each one changed the code.

## 1. `look_vec` points down-range, away from the radar

The note defines `alpha = atan(grad(z) . look_vec)` as the tilt *toward* the radar. That
identity only holds when `look_vec` is the ground-range direction of the beam — pointing
away from the radar, near range to far range.

The natural reading, "the vector from the ground to the radar", inverts every slope in
the model. Consider a radar in the west illuminating eastward: a hillside that faces it
rises toward the east, so `dz/deast > 0`, and `alpha` must be positive. With a
ground-to-radar vector (pointing west) the dot product is negative instead.

Nothing about the failure is loud. The rendered image still looks like SAR, the losses
still decrease, and the DEM comes out with every slope reversed.

Pinned by `tests/test_physics.py::test_slope_toward_radar_sign` and stated in `CLAUDE.md`.
`data/geometry.py::look_vector` returns the down-range convention.

## 2. The altimeter footprint is comparable to the tile

The Magellan altimeter footprint is ~10 × 20 km. A 512 px tile at 75 m is 38.4 km. So
`L_alt` convolves with a kernel whose support is close to the tile's own size, and the
boundary condition — whatever padding the convolution uses — propagates about 3σ, roughly
25 km, inward from every edge. On a bare tile that is the entire tile.

Measured with a 60 m, 1.5 km ripple, which the footprint should annihilate completely:

| tile | edge margin | leakage into `L_alt` |
|---|---|---|
| 256 px (19 km) | none | 4.54 m |
| 512 px (38 km) | 186 px | 0.113 m |
| 1024 px (77 km) | 373 px | 0.0007 m |

So tiles are cut at `core + 2 × margin` (`data/tile.py::TileSpec`, margin 384 px ≈ 28.8 km)
and `loss_alt` takes an `edge_margin_px` that drops the contaminated posts. It costs
about 6× the tile area on disk, which is the most expensive single choice in the pipeline
and is called out as such in the config.

**The cheap fix is wrong.** Comparing `blur(z_hat)` against `blur(gtdr)` cancels the edge
effect exactly, and it is tempting. But GTDR *already is* the footprint-averaged surface,
so blurring it again anchors the model to a doubly smoothed target — it biases the entire
planet smooth, in the one term whose whole job is to be the trustworthy anchor.

## 3. Pyramiding the physics loss on the DEM biases it

`L_phys` runs at full resolution and on a 4× pyramid level, because at 75 m the residual
is dominated by speckle. The obvious implementation downsamples the DEM and renders from
it, which costs one render instead of two.

It is biased. `RV` is nonlinear in slope, so the render of a block-mean slope is not the
block mean of the render, and the size of the gap scales with subgrid roughness. The
error would therefore be systematic in exactly the terrain where it is hardest to
notice — tessera, which is both the roughest terrain and the most scientifically
interesting. At the true DEM the biased version bottoms out at a loss of 0.27 instead of 0.

The pyramid now averages the *residual image*. Unbiased, kills speckle just as well, one
extra render.

## 4. The eight loss terms are dimensionally incommensurable

Section 5.7 gives weights `w_s = 1.0, w_a = 2.0, w_p = 0.3, w_r = 0.05`. Applied to the
raw values, those weights do not mean what they look like, because the terms are in
different units:

| term | units | typical value at the bicubic baseline | share of the objective |
|---|---|---|---|
| `L_stereo` | metres | 62.6 | 73.8% |
| `L_alt` | metres | 10.9 | 25.7% |
| `L_phys` | decibels | 1.35 | **0.5%** |
| `L_rms` | radians | 0.12 | 0.0% |

So the radarclinometry term — the only source of sub-kilometre detail over the 80% of
Venus with a single look and no stereo DEM — contributes half a percent of the gradient,
and the roughness term contributes nothing at all. The loss curve gives no sign of this;
it just converges to a smooth planet.

Each term is now divided by its own observational uncertainty (`losses.LossScales`), so a
normalised term reads as *sigmas of disagreement* and the weights read as relative trust:
75 m for the Herrick stereo DEM, 30 m for altimetry, 2 dB for the speckle-limited
radiometric residual, 1 degree for slope and roughness. After normalisation:

| term | value | share |
|---|---|---|
| `L_stereo` | 3.84 | 75.1% |
| `L_alt` | 0.36 | 14.2% |
| `L_phys` | 0.68 | **4.0%** |
| `L_rms` | 6.93 | 6.8% |

Pinned by `tests/test_loss_balance.py`, including a test that asserts the *old* imbalance
still holds under `UNIT_SCALES` — so if someone removes the normalisation, the test that
fails is the one that explains why.

## 5. Synthetic terrain has to be parameterised by slope, not by relief

The first synthetic generator fixed an elevation standard deviation. At 900 m over a
19 km tile with a realistic spectral slope that produces RMS slopes of 50–80°, which is
pure layover: the physics mask rejects everything and the tiles teach nothing.

RMS slope is the right knob because it is what the radar physics actually responds to,
and because it is a real Magellan observable (GSDR): plains sit near 1–3°, tessera near
5–10°. Tiles are now generated at a target RMS slope with a separate very-smooth regional
component for the altimetry loss to anchor.

## 6. `gaussian_downsample` must sample cell centres

Decimating from index 0 puts the coarse samples half a coarse cell away from where
`align_corners=False` interpolation expects them. Invisible until the result is upsampled
again, at which point it is an `f/2`-pixel registration shift — 4 pixels at the factor
the footprint operator uses.

## 7. MPS collapses above 128 px tiles

Measured on an M-series Mac with the 31 M-parameter model:

| device | tile | batch | s/step | ms/tile |
|---|---|---|---|---|
| MPS | 128 | 8 | 2.29 | 286 |
| CPU | 128 | 8 | 7.91 | 989 |
| MPS | 256 | 4 | 33.28 | 8320 |

A 29× per-pixel regression between 128 and 256. Fine for sanity runs at small tile sizes,
useless for the real phases — plan on CUDA for Phases 1–3, as the architecture note
assumes.


## 8. The saved model was 41% untrained, and the log said otherwise

`train.py` reports metrics from the live weights. Everything downstream — evaluation,
calibration, the qualitative panels, global inference — loads the EMA, because that is
what you want at inference time. With a fixed decay of 0.999 the EMA retains `0.999^t` of
the *initial* weights: after 900 steps that is 41%.

Measured on the same Phase 0 checkpoint:

| weights | MAE | skill vs bicubic | mean \|residual\| |
|---|---|---|---|
| live model | 92.87 m | **+29.5%** | 68.1 m |
| saved EMA | 126.34 m | **+4.0%** | 10.3 m |

The residual column is the tell: the EMA's predicted correction over the altimetry is a
seventh of the model's, because the residual head is zero-initialised and the average is
still mostly holding that zero. Nothing in the training log shows it — the log is
reporting the other set of weights — and the discrepancy surfaced only because a
standalone evaluation of the same checkpoint disagreed with the number the trainer had
printed twenty minutes earlier.

Two changes. The decay now warms up, `decay_t = min(decay, (1 + t) / (10 + t))`, so the
average tracks the model early and slows to the nominal horizon later — standard practice,
and it makes short runs usable at all. And `train.py` now scores the EMA as well as the
live weights at the end of every run, so the two can never disagree silently again.
`train.load_weights` refuses an EMA with too few updates behind it and says so.

## 9. The altimetry posts were not where the loss looked for them

`L_alt` samples the fine grid at `offset + j * stride` — 31, 93, 155, 217 at 75 m with a
62 px stride. `torch.nn.functional.interpolate` stretches its input to fill the output,
which puts post `j` at `(j + 0.5) * H / P - 0.5` instead. Those agree only when the tile
height happens to equal `posts x stride`, which at these numbers it does not: on a 512 px
tile the last post lands about 6 pixels — 450 m — from where the loss reads it.

So the anchor was comparing the model at one location against altimetry interpolated from
another. On a surface varying by ~100 m over 4.6 km that is a systematic error of order
10 m in the one term whose job is to stop the surface drifting.

`data.tile.upsample_posts` now maps each output pixel to its post coordinate explicitly
and clamps rather than extrapolating past the edge. It is shared by the real ingest and
the synthetic generator, so the two cannot disagree about where a post is —
`tests/test_tile.py` asserts a post round-trips to its own lattice position, and that the
synthetic tiles agree with their own recorded posts.

## 10. GTDR nodata decoded to zero elevation

`decode_gtdr` returns a validity mask; `build_tile` was dropping it. GTDR nodata is
`-32768`, which decodes to 0 m, and Venus has no sea level — so over the ~2% of the planet
the altimeter never measured, `L_alt` would have anchored the surface to zero and the
uncertainty head would have had no reason to object. The mask is now carried into the tile
store, through augmentation, and into the loss.

Both of these are real-pipeline bugs found without any real data, by keeping the synthetic
generator and the ingest path on one shared definition and testing that definition.

## 11. Two globe failures that look like renderer bugs

Both were found by building the thing and looking at it, and neither produces an error
message.

**A physical sun leaves Venus black.** Venus rotates once per 243 Earth days, so a sun
placed by the clock is static, and from most camera positions it is on the far side. With
`globe.enableLighting = true` the first working build rendered a correctly tiled,
correctly projected, entirely black planet. The light is now a `DirectionalLight` pinned
to the camera with a fixed offset — which is also what the architecture note asks for,
for the same reason.

**Quantized-mesh tiles need `Content-Encoding: gzip`.** The format expects gzipped tiles
and CesiumJS does not decompress them itself; it relies on the transport. A static file
server that does not set the header hands Cesium compressed bytes, every tile fails to
parse, and the console shows a generic terrain error with no hint of the cause.
`vite.config.ts` sets it; any production host must too.

## 12. Alignment is testable, and worth testing

The globe work plan asks for the imagery to be verified against a known feature. That is
easy to do by eye and easy to get subtly wrong, so it is automated instead: the demo tile
generator writes a `graticule` layer with a marker drawn at each site's published
coordinates, and `ishtar-globe/scripts/smoke.mjs` puts the camera exactly overhead at
Mead crater and reads the framebuffer. The marker lands 1.5 px from centre.

It catches the whole class of projection mistakes at once — a Web Mercator tiling scheme,
a TMS/XYZ row-order flip, a 0-360 longitude that was never wrapped — each of which
renders a perfectly plausible globe with everything in the wrong place.


## 11. The 300 GB was never necessary

The architecture note budgets ~300 GB of Magellan mosaics. Three properties of the actual
products reduce that to about 3 GB.

**They are tiled COGs.** The 75 m left-look mosaic is 506 928 x 230 948 pixels, internally
tiled 256 x 256. A `rasterio` window read over `/vsicurl/` fetches only the tiles the
window touches, so cutting a 512 px training tile moves roughly 130 kB. 1 059 tiles across
ten regions cost about 130 MB and four minutes, against 117 GB for the file.

**There are JPEG variants.** `Venus_Magellan_LeftLook_mosaic_global_75m_jpeg.tif` is the
same 75 m resolution with JPEG compression inside the GeoTIFF: 117 GB becomes 16.8 GB, and
the right-look becomes 7.2 GB. Even a full mirror is a seventh of the naive figure. There
is no JPEG variant of the Cycle 3 stereo-look mosaic, which is why that look is off by
default in the ingest.

**The auxiliaries are tiny.** GTDR, GSDR and GEDR together are 0.18 GB at 4641 m.

Only the Herrick stereo DEM genuinely has to be fetched whole (2.97 GB), because it is a
headerless PDS `.img` rather than a COG. It is also the one product without which `L_nll`
has no signal at all.

## 12. Four things only real data could tell us

**`rasterio.read()` does not apply band scale and offset.** Emissivity is stored as int16
with a scale of 1e-4, so it arrived as ~8500 instead of 0.85 — four orders of magnitude
off, as a network input, with nothing to show for it but a worse model. Confirmed fixed by
reading 0.850 on the plains and 0.452 at Maxwell Montes, which is the known
high-emissivity-anomaly signature.

**A partial `.img` reads as terrain.** `np.memmap` on a headerless array happily maps to
the declared shape; rows past the downloaded extent come back as zeros, which on Venus is
a sea-level plain rather than missing data. `StereoDEM.available_rows` bounds the map to
what is actually on disk, and because the array runs north to south a partial download is
an honest northern band rather than a corrupt file.

**The stereo DEM's datum is off by ~795 m.** Regressed over 50 patches against the
altimetry: `stereo = 0.982 * GTDR - 795 m`, correlation 0.9981, residual scatter 71 m. The
scatter is the Herrick DEM's own quoted 50-100 m vertical accuracy, so the shape is right
and the datum is not. Stereo gives relative heights and the absolute tie has to come from
altimetry, so `data/ingest.py` removes the median difference per tile. That also leaves
`L_stereo` teaching exactly what it should — the departure from GTDR in the 100 m - 10 km
band — while `L_alt` keeps the level.

**`np.savez_compressed` decompresses the whole array per access.** Each array is one
deflate stream, so indexing a single tile out of a 278 MB raster decompresses all of it.
Training went from 2.8 s/step on synthetic to 13.8 s/step on real tiles for that reason
alone. `data/real.py` expands the store into memory-mapped `.npy` files once: 1 ms per
sample, at about 4x the disk.

## 13. Without the stereo DEM the uncertainty head trains on nothing

The first real-data training step reported `nll=0.000`, and it stayed there. `L_nll` is
computed against the stereo DEM, and its unsupervised hinge is measured relative to the
supervised sigma — so with no stereo coverage at all, both halves are identically zero and
the variance head receives no gradient.

The DEM still trains: altimetry, radarclinometry, cross-look and roughness are all
unaffected. But the uncertainty map is the thing that makes a model-derived product honest
rather than merely plausible, and it is worth being explicit that it depends entirely on
the one product that has to be downloaded whole.


## 14. The look direction was wrong, and only real data could show it

`data/geometry.py` assumed a northward ground track, which puts the left-look beam's
down-range direction **west**. Magellan imaged on the *descending* leg of each orbit,
running north to south, so a left-looking beam illuminates **east**.

Measured against the real mosaics — correlation between stereo-derived slope toward the
radar and observed flattened backscatter, both smoothed to 2.4 km:

| assumed down-range | correlation |
|---|---|
| east (descending pass, correct) | **+0.091** |
| west (as shipped) | −0.091 |
| north | +0.007 |
| south | −0.007 |

Physics requires a positive correlation. The orthogonal directions sitting at zero is what
confirms the signal is real rather than an artefact of the test.

**This is the failure mode the whole repository is built to prevent, and its own tests
could not catch it.** `tests/test_augment.py` and `tests/test_physics.py` verify that the
renderer and the losses agree about the convention, and they do — the synthetic generator
renders *with the same convention the loss inverts*, so a wrong one is perfectly
self-consistent and every test passes. Synthetic data can prove internal consistency. It
cannot tell you which way the radar was pointing.

## 15. The incidence angle cannot be recovered from the stereo DEM

`data/calibrate_geometry.py` was written to replace the placeholder incidence profile by
fitting theta where the stereo DEM makes the slope known. It does not work, and the script
now says so instead of returning a number.

The fit slides to whatever theta the model is least sensitive to. A +/-3 degree slope swing
moves the predicted RV by 3.64 dB at theta=15 and only 1.42 dB at theta=53, so when the
observation does not track slope, least squares rails against the top of the search grid:
64% of tiles pin there, and the quadratic profile through them peaks at latitude -264
degrees, which is not a latitude.

The cause is upstream. The stereo DEM's 71 m vertical accuracy becomes ~9.5 degrees of
slope noise at its 600 m posting, against real Venus slopes of a few degrees. Smoothing to
2.4 km cuts the noise to 2.4 degrees and the correlation still only reaches +0.09.

That number is worth sitting with. The architecture note says the residual in RV is "to
first order, exactly the slope signal you want to invert". Against real data, resolvable
topographic slope explains about **1% of the variance** in 75 m Magellan backscatter; the
rest is the `b(x)` nuisance field — roughness and dielectric contrast. The physics loss is
real but far weaker on Venus than the synthetic experiments implied, and this is the
central risk to the approach that Phase 0 could not have surfaced.

Recovering the angles needs the F-BIDR labels, which is what the note specified first.
