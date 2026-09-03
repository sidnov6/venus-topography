#!/usr/bin/env bash
# Phase 0 end to end: the two controls, the two candidate runs, then the Section 7 table,
# the uncertainty calibration and the qualitative panels.
#
#   ./scripts/phase0.sh [tile_px] [steps]
#
# Defaults are sized for an M-series Mac and take about three hours. On CUDA, use 512 px
# tiles and 2000 steps as the architecture note specifies — MPS collapses above 128 px
# (docs/DECISIONS.md §7).
#
# Every run is the same commit. The raw-loss-units arm differs only by a flag, which is
# what makes the comparison in docs/RESULTS.md mean anything.
set -euo pipefail
cd "$(dirname "$0")/.."

TILE=${1:-128}
STEPS=${2:-900}

echo "== control: fit z_true directly, 8 tiles =="
python -u train.py --phase overfit --tile-size "$TILE" --steps 600 --batch-size 8 \
    --ckpt-dir runs/overfit

echo; echo "== candidate: weakly supervised =="
python -u train.py --phase sanity --tile-size "$TILE" --steps "$STEPS" --batch-size 8 \
    --n-tiles 200 --ckpt-dir runs/sanity

echo; echo "== A/B: the same run without the loss normalisation =="
python -u train.py --phase sanity --tile-size "$TILE" --steps "$STEPS" --batch-size 8 \
    --n-tiles 200 --raw-loss-scales --ckpt-dir runs/sanity_rawscale

echo; echo "== the Earth stage in miniature: supervised pretrain =="
python -u train.py --phase pretrain --tile-size "$TILE" --steps "$STEPS" --batch-size 8 \
    --ckpt-dir runs/pretrain

echo; echo "== candidate: weakly supervised, warm-started from the pretrain =="
python -u train.py --phase sanity --tile-size "$TILE" --steps "$STEPS" --batch-size 8 \
    --n-tiles 200 --init-from runs/pretrain/last.pt --ckpt-dir runs/sanity_pretrained

echo; echo "== Section 7 table =="
python -m eval.compare_runs runs/sanity runs/sanity_pretrained runs/sanity_rawscale \
    runs/pretrain runs/overfit \
    --tile-size "$TILE" --n-tiles 48 --batch-size 8 --classical-steps 120 --device cpu \
    --out outputs/ablation.json

echo; echo "== uncertainty calibration =="
python calibrate.py --ckpt runs/sanity/last.pt --tile-size "$TILE" --n-tiles 32 --device cpu

echo; echo "== qualitative panels and power spectra =="
for i in 0 1 2; do
    python -m eval.panels --ckpt runs/sanity/last.pt --tile-size "$TILE" --index "$i" \
        --device cpu --out "outputs/panel_sanity_$i.png"
done
