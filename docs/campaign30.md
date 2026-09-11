# Run the corrected 30-packing campaign

This workflow generates **30 new DEM packings** and then runs the complete
thermal, allocation and graph-learning analysis. It does not reuse the old
ten-packing results. The first ten realization seeds are retained for continuity,
but their geometry is regenerated with the corrected DEM properties.

## 1. Clone and prepare Python

```bash
git clone git@github.com:khalifali/HeatPath-GNN.git
cd HeatPath-GNN
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r code_work/requirements_campaign.txt
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Python 3.12 is recommended. CPU execution is the default; PyTorch Geometric is
not needed. For CUDA, install a compatible PyTorch build using the official
instructions at https://pytorch.org/get-started/locally/ and pass `--device cuda`.
Recorded CPU and GPU timings must not be pooled.

## 2. LAMMPS

Use a native LAMMPS executable with the GRANULAR package, preferably
`stable_22Jul2025_update5`, and the MPI launcher belonging to that build.
An existing installation is sufficient. If it is called `lmp_mpi` instead of
`lmp`, pass `--lammps lmp_mpi`. An absolute executable path also works.

If you need to build it, load your server's C++ compiler, CMake and MPI modules
first, then build outside the HeatPath-GNN repository:

```bash
git clone --depth 1 --branch stable_22Jul2025_update5 git@github.com:lammps/lammps.git
cmake -S lammps/cmake -B lammps/build \
  -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -D PKG_GRANULAR=ON
cmake --build lammps/build -j 8
```

The resulting executable is `lammps/build/lmp`. Use its absolute path below.
Do not mix an MPI-linked LAMMPS build with another MPI implementation.

## 3. Check the installation

From the HeatPath-GNN repository root, with the Python environment active:

```bash
bash code_work/Allrun30 --check --lammps lmp --mpi-ranks 8
```

This checks the Python dependencies, GRANULAR availability, analytical thermal
tests and a **zero-step LAMMPS two-sphere Hertz-force test**. It does not generate
the thirty packings. `--mpi-ranks 1` runs LAMMPS directly without an MPI launcher.
The static check runs one process; the full campaign uses the requested MPI ranks.

Inspect the complete settings without running anything:

```bash
bash code_work/Allrun30 --dry-run
```

## 4. Launch the full campaign

```bash
nohup bash code_work/Allrun30 --lammps lmp --mpi-ranks 8 > campaign30.log 2>&1 &
```

Monitor progress:

```bash
tail -f campaign30.log
```

The program executes one packing job at a time, with eight MPI ranks for DEM.
Thermal analysis and CPU training use one thread to avoid oversubscription.
On a scheduled cluster, run the same command inside an allocated job instead of
launching it on a login node. Allocate eight CPU cores; no GPU is required.

Default output: `runs/campaign30/`. To store results elsewhere:

```bash
bash code_work/Allrun30 --lammps /absolute/path/to/lmp --mpi-ranks 8 \
  --output /absolute/path/to/heatpath_results
```

The program prints each stage and its log location. On error it stops, prints
the last log lines and leaves completed stages intact. There is no promised
runtime: DEM and search dominate, and hardware determines elapsed time.

## 5. Resume after interruption

Run **the same launch command again**, in the same environment and output location.
Completed stages are verified using SHA-256 hashes and skipped. An incomplete
stage reruns from its beginning. ML checkpoints are per held-out packing; at
most that packing's model fits need to rerun after interruption.

Do not edit code, change settings or replace dependencies during a campaign.
The manifest rejects such changes to prevent mixing results. Start a new output
directory for a changed experiment. Never delete the manifest alone to bypass
the provenance checks. Completed data that was manually changed must be restored,
or the campaign must be started in a new directory. A file lock prevents two
processes from writing the same campaign simultaneously.

## Scientific settings

| Quantity | Setting |
|---|---|
| Independent packings | 30 fixed realization seeds, listed in run_campaign30.py |
| Particles per packing | 500 |
| Diameter classes | 1.5 / 2.0 / 2.5 mm |
| Counts | 100 / 300 / 100 |
| Density | 2500 kg/m3 |
| Nominal porosity | 0.38, identical by fixed solid and box volumes |
| Boundaries | Periodic x/y; fixed z walls; no gravity |
| DEM normal/tangential model | Hertz material / Mindlin rescale; Tsuji damping |
| Young's modulus | 50 MPa, explicitly a softened packing parameter |
| Poisson ratio / restitution / friction | 0.25 / 0.30 / 0.50 |
| Growth / final relaxation | 300000 / 200000 steps; dt = 2e-7 s |
| Final translational KE acceptance | <= 1e-12 J; change only with a documented reason |
| Thermal model | G_ij = 4 a k_i k_j/(k_i+k_j); infinite-wall G = 4 a k_i |
| Thermal boundaries | 301 / 300 K |
| Material conductivities | 1 / 10 W/(m K) |
| Selected conductive particles | 10 / 30 / 10 by size class |
| Random allocations | 100 per packing |
| Search | Five restarts, 3000 same-size swap proposals each |
| MLP/GNN | Width 64; GNN has three message-passing layers |
| Evaluation | 30 outer folds: 28 training + 1 validation + 1 test packing |
| Training | Three repeats per model; maximum 400 epochs; patience 60 |
| Inference | Mean within-type ranks, then exact size quotas |

The current study remains contact-only. A higher nominal modulus alone is not
a validated cure for large overlaps in a fixed-volume packing. This workflow
reports contact-size distributions and numerical overlap-floor sensitivity;
it does not claim experimental validation or physical stiffness sensitivity.

## Generated outputs

| Location below the output directory | Contents |
|---|---|
| campaign_manifest.json | Seeds, settings, source hashes and software versions |
| source_snapshot/ | Exact scripts used for the campaign |
| logs/ and checkpoints/ | Commands, execution logs, elapsed times and verified output hashes |
| packing_seed_*/ | Final particles, contacts, data, DEM restart and packing diagnostics |
| packing_seed_*/allocation_final/ | Random/ranking/search results and all restart ID lists |
| packing_seed_*/thermal_baseline/ | Homogeneous contact-only solution |
| packing_seed_*/thermal_optimized/ | Independent verification of the best allocation |
| dataset/ | Graph features, targets and manifest |
| ml/packing_seed_*/ | Fold results, predicted IDs and model checkpoints with scalers/splits |
| analysis/packing_seed_*/ | Mechanisms, VTP visualization files, contact sizes and floor sensitivity |
| report/report.pdf | Automatic results report |
| report/report.html and report.md | Browser-readable and editable reports |
| report/*.csv | Combined per-packing, ML, mechanism and sensitivity data |
| report/figures/ | Six figures, each in vector PDF/SVG and 600-dpi PNG |
| report/figure_captions.md | Figure captions and definitions |

Figures show conductivity, relative allocation gain, held-out ML recovery,
mechanisms across all packings, contact-size distributions and numerical
overlap-floor sensitivity. Scatter points retain the individual packing results;
summary error bars are sample standard deviations, not confidence intervals.
The report states the observed outcomes even if GNN performance is disappointing.
It never fabricates a successful outcome or reports R-squared for allocation.

The standalone PDF/SVG plots are intended for publication use. Review the final
report and figures after the real campaign; numerical ranges cannot be known in
advance. Raw run directories are ignored by Git. Keep them on the server and
later select the report, figures and compact reference data for version control.

## Optional execution tests

Full pipeline with three **new full-duration DEM packings**, tiny search budgets
and two ML epochs (writes to runs/smoke_test/):

```bash
bash code_work/Allrun30 --smoke-test --lammps lmp --mpi-ranks 8
```

Its reports and figures are marked **SMOKE TEST - NOT SCIENTIFIC RESULTS**.
It is optional and does not replace the 30-packing run.

Python-only integration test using three tracked legacy geometries:

```bash
python3 code_work/test_campaign_workflow.py
```

This recalculates thermal/search/ML results with tiny budgets, builds the report,
and tests resume and tamper detection. It does not test DEM generation.

## Historical data

The old ten-packing data and manuscript are retained as a historical snapshot.
Their absolute conductivities use the earlier prefactor and their DEM parameter
provenance needs care. They should not be combined with the new campaign.
The analytical audit described in contact_model_audit.md has now been corrected
in the production solver and generator. The new regression tests must pass.

## Growth contact-search correction

The initial campaign input omitted an explicit final-size granular cutoff.
LAMMPS initializes its automatic cutoff from the initial radii; those spheres
start at only 5% of final diameter. The input now sets `pair_style granular ${d3}`
(the largest final diameter, 0.0025 m) and rebuilds neighbors every step, including
when radii grow without appreciable center motion. Contact forces still act only
on actual overlapping spheres. See the cutoff discussion in
https://docs.lammps.org/pair_granular.html.

This fixes a contact-search defect implicated in the seed 46549 lost-atom failure.
A full rerun on the server is still required to establish dynamic stability.
The static Hertz check alone does not test diameter growth or MPI neighbors.
Preserve the earlier campaign for diagnostics and start a separate output:

```bash
git pull --ff-only
bash code_work/Allrun30 --lammps lmp --mpi-ranks 8 --output "$(pwd)/runs/campaign30_growthfix"
```

Do not reuse completed packings from the earlier growth implementation in this
corrected ensemble, even if their final kinetic-energy checks passed.
