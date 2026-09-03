#!/usr/bin/env bash
# Build a runnable demo globe from synthetic Venus, then serve it.
# Everything it renders is synthetic; the real pipeline needs ~300 GB of Magellan data.
set -euo pipefail
cd "$(dirname "$0")/.."

LEVEL=${1:-5}
ROI_LEVEL=${2:-8}
python -m export.demo_tiles --out ../ishtar-globe/public/tiles \
    --max-level "$LEVEL" --roi maxwell mead --roi-level "$ROI_LEVEL"

cd ../ishtar-globe
[ -d node_modules ] || npm install
npm run dev
