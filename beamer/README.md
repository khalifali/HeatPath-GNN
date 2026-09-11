# HeatPath-GNN presentation

The 24-slide Beamer deck follows the supplied PoreAccess-CO2 template and
presents the completed 30-packing campaign, from motivation to conclusions.

- `heatpath_gnn.tex` and `heatpath_gnn.pdf`: presentation source and slides.
- `heatpath_gnn_notes.tex` and `heatpath_gnn_notes.pdf`: each slide beside its
  presenter transcript, on a wide page suitable for a monitor or printing.
- `presenter_notes.md`: a convenient plain-text copy of the transcript.
  Edit the notes TeX for the compiled handout, and keep this copy consistent.
- `figures/`: six supplied vector result plots and the supplied TUM title assets.
- `data/`: compact supplied result tables, figure definitions, and campaign manifest.
  This is supporting presentation evidence, not the full simulation archive.

## Build

From the repository root:

```bash
bash beamer/build.sh
```

Requires `pdflatex` with Beamer, TikZ, Latin Modern, geometry and graphicx
(usually texlive-latex-recommended, texlive-latex-extra, texlive-pictures and
lmodern on Ubuntu). The notes source includes pages from the compiled deck,
so compile the deck before the notes. No Python or simulation run is required.
The PDF figures can also be reused independently in a paper.

## Scientific scope

The figures are supplied campaign results, not synthetic or smoke-test results.
Gain and recovery are computed per packing and then averaged. The search is a
fixed-budget best-found reference, not a global optimum. Scoring timings exclude
baseline and verification solves. Contact-size and physical-validity limitations
are discussed explicitly. No new simulations were performed for these slides.
The campaign source identities are recorded in `data/campaign_manifest.json`.

Mechanical model reference: https://docs.lammps.org/pair_granular.html
Thermal formulation and implementation audit: ../docs/contact_model_audit.md
Campaign procedure: ../docs/campaign30.md

Only presentation-related files live here. Simulation and analysis programs
remain in `code_work/`.
