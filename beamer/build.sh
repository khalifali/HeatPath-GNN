#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

pdflatex -interaction=nonstopmode -halt-on-error heatpath_gnn.tex
pdflatex -interaction=nonstopmode -halt-on-error heatpath_gnn.tex

cp heatpath_gnn.tex heatpath_gnn_notes.tex
sed -i '/show notes on second screen=right/s/^% //' heatpath_gnn_notes.tex

pdflatex -interaction=nonstopmode -halt-on-error heatpath_gnn_notes.tex
pdflatex -interaction=nonstopmode -halt-on-error heatpath_gnn_notes.tex
