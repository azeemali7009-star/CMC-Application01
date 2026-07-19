#!/usr/bin/env bash
set -euo pipefail
# Optional Neper physical-mesh generator. Requires Neper on PATH.
# The generated tessellation is a physical mesh M; the article's Python
# preprocessing constructs the incidence-resolved computational complex K.
SEED=${1:-64}
OUT=${2:-data/neper}
mkdir -p "$OUT"
neper -T -n "$SEED" -id 1 -domain 'cube(1,1,1)' -morpho voronoi -o "$OUT/tumour"
neper -M "$OUT/tumour.tess" -elttype hex -order 1 -o "$OUT/tumour_hex"
echo "Generated $OUT/tumour.tess and mesh files."
