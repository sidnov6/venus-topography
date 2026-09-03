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
