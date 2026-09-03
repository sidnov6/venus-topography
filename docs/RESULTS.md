# Results

> **Read this first.** Everything below the "Real Venus" section is Phase 0 on synthetic
> data, where the pipeline works. On real Magellan data it does not yet: the model
> collapses to a degenerate solution. That result is at the top because it is the one that
> matters.

## Real Venus: the model declines to produce topography

1 059 real Magellan tiles at 75 m, ten regions, 848 train / 211 held out (Ovda and
Guinevere). Corrected look direction, stereo supervision on 42% of tiles, uncertainty head
active. Scored on the held-out tiles:

| | alt m | phys dB | x-look dB | stereo m | relief m | sigma m |
|---|---|---|---|---|---|---|
| bicubic GTDR (the input) | 34.10 | 2.498 | 20.95 | 140.2 | 0.0 | — |
| ISHTAR on real data | 34.12 | 2.131 | 22.31 | 140.2 | **0.1** | 100.8 |

The `relief` column is the finding: **the model added 0.188 m of topography.** Its output
is the altimetry it was given. Its stereo error is identical to the baseline's to one
decimal, and so is its altimetry residual.

The apparent win on the physics residual is not topography either:

| physics residual | dB |
|---|---|
| model, with its brightness field | 2.131 |
| model, with `b(x)` removed | **2.498** |
| plain bicubic GTDR | **2.498** |

Strip the nuisance field and the model *is* the baseline. It explained the radar image
entirely through `b(x)` — a 1.15 dB brightness field — and left the terrain alone.

This is the exact failure the architecture note names: "`b(x)` is a nuisance field. Let the
network predict it as a second low-resolution output head ... with a strong smoothness
penalty, so it cannot absorb slope information." Against real data the penalty is not
strong enough, and the reason is quantitative: resolvable slope explains about **1% of the
variance** in 75 m Magellan backscatter (DECISIONS §15), while a free low-resolution
brightness field can explain considerably more. The optimiser takes the cheaper route.

Two things are worth saying in the model's defence. Reporting a mean 1σ of **100.8 m**
alongside 0.2 m of relief is the uncertainty head working exactly as designed — the model
is saying it cannot recover the topography, and saying so honestly. And the earlier run
without the uncertainty term *did* produce 30.7 m of relief, which suggests `L_nll` is what
tips it into the degenerate solution: with σ free, the negative log-likelihood is minimised
by predicting the conditional mean and declaring high uncertainty.

**What to try next**, in the order I would try it:

1. **Constrain `b(x)` much harder** — lower resolution than 1/16, a far stronger TV
   penalty, or a bounded amplitude. It currently has enough freedom to explain the image
   without help from the terrain.
2. **Get the real incidence angles** from the F-BIDR labels. The physics term is running on
   a placeholder profile, so its gradient is pointing somewhere slightly wrong everywhere.
3. **Raise `w_stereo` and clamp σ** during early training, so the model cannot buy its way
   out by claiming ignorance before it has tried.
4. **Then** the Earth pretraining stage, which is the one part of the recommended framing
   still untested and the most likely source of a prior strong enough to beat 1%.


Everything here is Phase 0: the weakly supervised objective run against a synthetic Venus
that is rendered through the same forward model the losses invert. It is not a result
about Venus. It is the check that the pipeline can recover terrain from the supervision
the architecture note says will be available, before spending 300 GB and several GPU-days
finding out.

All numbers below are from the **live weights**, which is what `train.py` reports. Until
the EMA decay warmup landed (see [DECISIONS.md](DECISIONS.md) §8) the saved EMA was
partly the initialisation, so any figure taken from a checkpoint written before that fix
is not comparable — those runs are being re-measured.

Run on an M-series Mac (MPS), 128 px tiles, batch 8. The architecture note's Phase 0 is
512 px for 2000 steps; MPS collapses above 128 px (see [DECISIONS.md](DECISIONS.md) §7),
so these runs are shorter and smaller than the note specifies, and are labelled as such.

## The Section 7 table

48 held-out synthetic tiles, 128 px. Every row is the same commit, scored from EMA
weights. This is the number to quote: it is a larger and fully independent sample than
the 16-tile set the trainer prints during a run, and the two differ by several points.

| | MAE m | RMSE m | slope ° | stereo m | alt m | phys dB | x-look dB | 1σ cov | skill |
|---|---|---|---|---|---|---|---|---|---|
| (a) bicubic GTDR | 116.24 | 158.12 | 7.48 | 91.10 | 13.15 | 2.30 | 24.09 | — | — |
| (b) classical radarclinometry | 101.44 | 136.86 | 4.04 | 81.58 | 13.27 | **1.40** | 24.69 | — | +12.7% |
| weakly supervised, raw loss units | 115.12 | 155.32 | 6.75 | 90.51 | **7.61** | 2.33 | 24.01 | 0.88 | +1.0% |
| weakly supervised | 92.55 | 123.85 | 4.70 | 77.80 | 10.16 | 2.32 | 23.70 | 0.67 | +20.4% |
| **pretrain → fine-tune** | **83.08** | **112.85** | 3.43 | **66.76** | 10.35 | 1.90 | **24.78** | 0.69 | **+28.5%** |
| supervised pretrain | 81.71 | 111.01 | **2.86** | 67.86 | 17.24 | 1.74 | 25.30 | 0.69 | +29.7% |
| supervised control (8 tiles) | 104.79 | 140.24 | 3.55 | 85.56 | 28.24 | 1.84 | 25.07 | 0.13 | +9.9% |

Four things in this table are worth more than the MAE column.

**The raw-units row is the loss-balance finding, stated by the model itself.** It scores
+1.0% overall and yet holds the *best altimetry residual in the table*, 7.61 m against the
normalised run's 10.16 m. That is exactly what the diagnosis predicts: with raw scaling the
metre-valued terms dominate the objective, so the model becomes excellent at the one thing
they measure and mediocre at everything else. Its slope error is 6.75°, barely better than
the bicubic baseline's 7.48°, and its radiometric residual (2.33 dB) is *worse* than the
baseline's. It optimised the altimeter and ignored the radar.

**Cross-look PSNR is the honest test of 75 m detail** — predict a DEM from one look, render
the *other* look from it, score against an image the model never saw. The staged run
(24.78 dB) beats classical (24.69) and bicubic (24.09). The cold-start run (**23.70**) is
*worse than bicubic*: without the pretrained prior it adds detail that does not predict the
other look, while its MAE improves by 20%. Nothing else in the table shows that.

**The physics residual is lowest for classical** (1.40 dB), which optimises it directly and
per tile at test time. Every learned run trades some of that for everything else — and the
two runs that never see the physics term with any real weight (raw units, 2.33 dB; cold
start, 2.32 dB) sit at the bicubic baseline's 2.30 dB, i.e. they explain the image no
better than a surface with no detail in it at all.

**Coverage separates the honest from the overconfident.** 0.67–0.69 for the three
well-behaved runs, 0.88 for the raw-units arm (over-cautious), 0.13 for the memorising
control.

## The loss rebalancing is measurable, and it is about stability as much as score

Same commit, same architecture, same data, same schedule, same seed. The only difference
is `--raw-loss-scales`, which switches off the per-term normalisation. MAE against the
synthetic truth on the trainer's validation set; the bicubic baseline is 132.5 m.

| step | raw units | normalised |
|---|---|---|
| 250 | 132.5 m (−0.0%) | 112.5 m (+15.1%) |
| 500 | **244.3 m (−84.4%)** | 92.5 m (+30.1%) |
| 750 | 123.7 m (+6.6%) | 89.3 m (+32.6%) |
| 900 | 123.5 m (**+6.8%**) | 90.9 m (**+31.4%**) |

Four and a half times the skill at the end, but step 500 is the more telling row. The
raw-units arm does not merely converge more slowly — it **destabilises**, wandering to
244 m, nearly twice the error of the baseline it started from, with a 147 m altimetry
residual, and recovering only as the learning rate decays. Gradient norms there run
1 000–7 000 against 25–150 for the normalised arm.

| final | raw units | normalised |
|---|---|---|
| MAE | 123.49 m | **90.90 m** |
| RMSE | 161.69 m | **119.12 m** |
| slope MAE | 7.75° | **5.24°** |
| altimetry residual | 10.87 m | 10.67 m |
| mean 1σ | 254.5 m | 108.7 m |
| 1σ coverage | 81.0% | **68.3%** |
| **skill vs bicubic GTDR** | **+6.8%** | **+31.4%** |

It also ends up badly over-cautious — a mean 1σ of 255 m and 81% coverage against a 68%
target. The uncertainty term is one of the few that was already dimensionless, so raw
scaling swamps it along with everything in decibels and radians.

The altimetry residual is the one column where raw units do fine, and for a reason: that
term is already in metres, which is the unit the raw weighting happens to suit.

Note that these are the trainer's own 16-tile validation numbers, which run several points
above the 48-tile table at the top. Use them for shape, not for quoting.

## The EMA agrees with the live weights again

| weights | MAE | slope | 1σ cov |
|---|---|---|---|
| live | 90.90 m | 5.24° | 68.3% |
| EMA (decay 0.9901) | 90.56 m | 5.25° | 68.6% |

Before the decay warmup landed, the same comparison read 126.34 m against 92.87 m. See
[DECISIONS.md](DECISIONS.md) §8.

## Calibration needs almost nothing, and still shows a heavy tail

`calibrate.py` on the weakly supervised run, fitted on one set of held-out tiles and
reported on a disjoint one:

| | before | after | target |
|---|---|---|---|
| 1σ coverage | 0.693 | 0.709 | 0.683 |
| 2σ coverage | 0.912 | 0.922 | 0.954 |
| mean σ | 108.9 m | 113.7 m | |

The fitted temperature is **1.044** — the uncertainty head is essentially calibrated as
trained, which is the one thing that was right from the start. The remaining 3-point gap
at 2σ is informative rather than a failure of the method: a single scalar cannot match
both, and the shortfall says the elevation-error distribution has heavier tails than
Gaussian. Reporting "calibrated" from the 1σ number alone would understate how often the
model is badly wrong.

The contrast is the memorising control, whose 1σ coverage is 0.13. That is what
temperature scaling is actually for, and why it is a phase rather than an afterthought.

## The power spectrum is the informative figure

`outputs/panel_sanity_0_spectrum.png`, from `eval/panels.py`. Radially averaged power
against wavelength, with the 75 m – 1 km band the model exists to invent shaded:

- **Bicubic GTDR** falls off a cliff below 1 km. It has no information there; that is the
  whole problem statement in one line.
- **Classical radarclinometry** tracks the truth down to about 400 m, then turns sharply
  upward below 200 m. That bump is speckle being integrated into the DEM as terrain — the
  classic failure of shape-from-shading, and RMSE cannot see it.
- **ISHTAR** follows the truth from 10 km to about 700 m, runs modestly above it through
  the few-hundred-metre band, and rolls off below 200 m. It over-produces a little in the
  middle of the invented band and under-produces at the finest scales, with none of
  classical's noise floor.

Two different failures needing two different fixes, which is why Section 7 asks for the
spectrum instead of another scalar.

## What staging actually buys, on synthetic data

Measured on the 48-tile set, not the trainer's 16:

| | cold start | pretrain only | pretrain → fine-tune |
|---|---|---|---|
| MAE | 92.55 m | **81.71 m** | 83.08 m |
| slope MAE | 4.70° | **2.86°** | 3.43° |
| altimetry residual | **10.16 m** | 17.24 m | 10.35 m |
| cross-look PSNR | 23.70 dB | **25.30 dB** | 24.78 dB |
| skill | +20.4% | **+29.7%** | +28.5% |

**The supervised pretrain alone is the best arm on almost every column, and that is not a
result about staging — it is an artefact of the setup.** On synthetic data the pretraining
label *is* the test answer, drawn from the same generator; nothing can beat training
directly on it, and the Venus terms are a proxy that necessarily pulls away from it. On
Venus there is no such label anywhere, which is the entire premise.

What the comparison does support:

- Fine-tuning **recovers the anchoring** the supervised objective throws away — 10.35 m
  against 17.24 m — for about 1.4 m of MAE. On the real problem that trade is the whole
  point: an unanchored surface drifts globally and no ground truth exists to catch it.
- Fine-tuning **retains most of the prior**. It keeps 3.43° of slope error against the
  cold start's 4.70°, and 24.78 dB of cross-look against 23.70 dB. The prior survives
  contact with the weak objective rather than being trained away.
- The cold-start run's cross-look score is **below the bicubic baseline** (23.70 against
  24.09). Left to the Venus terms alone at this budget, the model adds detail that does
  not predict the other look. That is the sharpest thing in the table, and MAE hides it
  completely.

## The brightness nuisance field costs something, but not much

Looking at individual tiles, the model and the classical inversion sometimes agree with
each other and disagree with the truth — which suggests both are converting the synthetic
scene's intrinsic-brightness variation into relief, since that is the one thing the image
contains that the terrain does not explain. Measured over the 48 held-out tiles:

| | |
|---|---|
| correlation of elevation error with the true brightness field, per tile | mean \|r\| = 0.21 |
| MAE on the smoother-brightness half | 86.9 m |
| MAE on the rougher-brightness half | 98.2 m |

So it is real and secondary: about 13% more error on tiles whose intrinsic brightness
varies more, not the dominant term. The `h_b` head runs at 1/16 resolution by design, so
that it cannot absorb slope information — this is the cost of that choice, and it is the
first place to look if the physics residual stops improving. Do not read the per-tile
panels as evidence of more than this; one tile is not a result.

## What Phase 0 does not tell you

- Nothing about **Venus**. The synthetic set has the right physics and roughly the right
  statistics, but real Magellan mosaics have seams, gain striping, layover, and terrain
  types no fractal reproduces.
- Nothing about the **Earth prior**, which is the whole premise of the recommended
  framing and only exists from Phase 1 onward.
- The MAE numbers are large in absolute terms (122 m) because the synthetic tiles carry
  100–1200 m of regional relief that the simulated GTDR only partly resolves. The number
  that matters is the ratio to the baseline, not the metres.

## Terrain products

Measured from `export/quantized_mesh.py` on a level-9 tile of plausible Venus relief:

| | |
|---|---|
| vertices / triangles per tile | 65 × 65 / 8192 |
| resolution at level 9 | 580 m per vertex |
| tile size | 73.4 kB raw, 8.6 kB gzipped |
| global pyramid, levels 0–9 | ~6.2 GB gzipped |
| bounding sphere at level 0 | 6051.8 km — exactly the planet radius, i.e. optimal |
| horizon occlusion point magnitude | 1.00001 at level 9 (WGS84 would give ~1.05) |

Height quantisation is 16-bit *within each tile's own range*: 0.43 m for a tile spanning
Venus's entire −3 km to +11 km, and 0.001 m for a flat plains tile. The format is never
the limiting factor on this product.
