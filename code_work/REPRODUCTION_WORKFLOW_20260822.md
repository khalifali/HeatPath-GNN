# Reproduction workflow: DEM packing, thermal network and conductive-particle allocation

This file records the workflow used up to the controlled fixed-volume allocation
study. Run all commands from `code_work/`, the directory containing the scripts and
the `packing_seed_*` case folders.

## 1. Software and Python dependencies

- LAMMPS: `22 Jul 2025 - Update 5`, compiled with the GRANULAR package.
- Python 3 with NumPy and SciPy.
- ParaView for inspecting VTP output.

Verify the Python thermal solver before processing project cases:

```bash
python3 test_heat_transfer_solver.py
```

Expected terminal message: `heat-transfer solver self-test: PASS`.

## 2. Files and their roles

| File | Role |
|---|---|
| `in.generate_equal_porosity_psd.lammps` | Generate one mechanically relaxed packing with corrected contact IDs |
| `packing_dump_to_vtp.py` | Convert a selected particle dump to VTP for ParaView |
| `solve_packing_heat_transfer.py` | Build and solve the steady thermal contact network |
| `test_heat_transfer_solver.py` | Deterministic thermal-solver verification test |
| `run_allocation_pilot_stratified_v2.py` | Controlled random, ranked and fixed-budget swap allocation |
| `run_all_cases_stratified_allocation_v1.sh` | Run the final controlled protocol and verification on all ten packings |
| `analyze_allocation_mechanisms.py` | Compare random, heat-path and optimized network mechanisms and write CSV/VTP output |
| `build_gnn_allocation_dataset.py` | Build one physics-guided graph file per packing for held-out-packing GNN tests |
| `train_evaluate_gnn_allocation_loocv.py` | Compare a same-feature MLP and edge-aware GNN by leave-one-packing-out thermal evaluation |
| `plot_gnn_loocv_recovered_gain.py` | Generate the paper's fold-wise MLP/GNN recovered-gain figure as PDF and PNG |
| `clean_all_packing_cases.sh` | Preview or explicitly remove all `packing_seed_<digits>` folders |

## 3. Generate the ten DEM packings

The accepted seed list is:

```text
18427 29173 37811 46549 55291 64109 72869 81647 90439 99223
```

For each seed, run:

```bash
mpirun -np 8 lmp -var seed_realization 18427 \
  -in in.generate_equal_porosity_psd.lammps
```

Change only `seed_realization`. Keep all mechanical parameters, particle counts,
diameters and the target porosity unchanged. Each case must contain:

```text
particles_final.dump
contacts_final.dump
packing_final.data
packing_final.restart
packing_summary.dat
```

The controlled number-based PSD is 100/300/100 particles with diameters
1.5/2.0/2.5 mm. There are 500 particles and the nominal porosity is 0.38.

### Contact-ID formatting

The uploaded `in.generate_equal_porosity_psd.lammps` contains the corrected
contact-ID formatting (floating-point values printed with zero decimals).
Older notes called this generator v5. The old v4 generator and restart-repair
input are not part of this repository. Use the supplied final dumps or the
current generator.

## 4. Check packing equivalence and visualize

For every `packing_summary.dat`, confirm:

- 500 particles;
- counts 100/300/100 for types 1/2/3;
- nominal porosity 0.38;
- identical total solid and box volumes;
- sufficiently low final kinetic energy;
- reasonable pressure and overlaps;
- no lost atoms.

Convert final particle dumps without rerunning DEM:

```bash
python3 packing_dump_to_vtp.py \
  --root . \
  --pattern 'packing_seed_*/particles_final.dump' \
  --output-name packing_final.vtp
```

Open each `packing_final.vtp` in ParaView and apply a Sphere Glyph using
`diameter` with scale factor 1.

## 5. Homogeneous contact-only thermal baseline

For one case:

```bash
python3 solve_packing_heat_transfer.py packing_seed_18427 \
  --output thermal_results \
  --k-low 1.0 \
  --k-high 10.0 \
  --k-fluid 0.0 \
  --t-hot 301 \
  --t-cold 300
```

Because no `--high-ids` file is provided, all particles use `k-low`. Check in
`thermal_summary.json`:

- at least one hot-to-cold spanning component;
- hot and cold wall links greater than zero;
- relative energy-balance error below `1e-8`;
- solved temperatures within 300--301 K;
- any unanchored particles are explicitly reported.

For the ten homogeneous packings, the observed mean was
`0.0932867 +/- 0.0035123 W/(m K)` with a CV of 3.77%. All ten contained one
boundary-spanning component and the maximum energy-balance error was
`1.31e-12`.

## 6. Numerical scaling verification

For `packing_seed_18427`, three calculations verified the linear solver:

1. `k-low=1`, temperature difference 1 K;
2. `k-low=2`, temperature difference 1 K;
3. `k-low=1`, temperature difference 10 K.

Doubling particle conductivity doubled heat rate and effective conductivity.
Increasing the temperature difference tenfold increased heat rate tenfold but
left effective conductivity unchanged. Energy-balance errors were near machine
precision.

## 7. Stagnant-air sensitivity diagnostic

The optional pairwise fluid-gap model was tested only as a diagnostic with
`k-fluid=0.026 W/(m K)`. At regularization ratio 0.001, cutoff ratios
0.05/0.10/0.20 produced 245/427/784 fluid edges and effective conductivities
0.10578/0.11825/0.14498 W/(m K). This cutoff dependence is too strong for a
validated physical model. Therefore:

- do not use the provisional fluid-gap model in the allocation study;
- use `--k-fluid 0.0` for the current controlled results;
- later replace it with a geometry-defined pore-network model.

## 8. Controlled fixed-volume allocation protocol

The final pilot fixes both the number and volume/PSD of high-conductivity
particles:

```text
type 1: 10 of 100 particles
type 2: 30 of 300 particles
type 3: 10 of 100 particles
total:  50 of 500 particles
```

Run one case with:

```bash
python3 run_allocation_pilot_stratified_v2.py packing_seed_18427 \
  --output packing_seed_18427/allocation_stratified_p3000 \
  --k-low 1.0 \
  --k-high 10.0 \
  --high-count 50 \
  --type-quotas 1:10,2:30,3:10 \
  --random-samples 100 \
  --swap-proposals 3000 \
  --optimizer-restarts 5 \
  --seed 12345
```

The script verifies the quotas and identical selected particle volume. It
compares 100 controlled random allocations, degree ranking, heat-path ranking
and five restarts of one fixed-budget same-size swap search. ``Same-size swap''
describes the proposed move; ``fixed budget'' means that every restart stops
after exactly 3000 proposals. They are not two different algorithms, and the
best result found is not a claimed global optimum.

For seed 18427, the controlled results were:

| Method | Effective conductivity [W/(m K)] | Improvement over random mean |
|---|---:|---:|
| Random mean | 0.11123 +/- 0.00194 | reference |
| Degree ranking | 0.11362 | 2.1% |
| Heat-path ranking | 0.12186 | 9.6% |
| Fixed-budget swap search, five-restart mean | 0.17474 +/- 0.00259 | 57.1% |
| Fixed-budget swap search, best of five | 0.178948 | 60.9% |

## 9. Independent verification of the best assignment

```bash
python3 solve_packing_heat_transfer.py packing_seed_18427 \
  --output thermal_results_stratified_optimized_p3000 \
  --high-ids packing_seed_18427/allocation_stratified_p3000/high_ids_optimized.txt \
  --k-low 1.0 \
  --k-high 10.0 \
  --k-fluid 0.0 \
  --t-hot 301 \
  --t-cold 300
```

The independently reproduced pilot value was
`0.1789481826350723 W/(m K)` with 50 high-conductivity particles and a relative
energy-balance error of `2.93e-12`.

## 10. Run all ten cases

```bash
bash run_all_cases_stratified_allocation_v1.sh
```

The batch script skips a case if both its allocation and independent thermal
summaries already exist. Use `--force` only when intentional replacement is
required. Use `--dry-run` to preview commands or `--start-seed 29173` to begin
at a particular seed.

Final per-case outputs are placed inside each packing directory:

```text
allocation_stratified_p3000/
thermal_results_stratified_optimized_p3000/
```

The project-level combined result is:

```text
allocation_all_packings_summary.csv
```

## 11. Mechanism analysis and ParaView files

For a first representative comparison, run:

```bash
python3 analyze_allocation_mechanisms.py packing_seed_18427 \
  --allocation-dir packing_seed_18427/allocation_stratified_p3000 \
  --output mechanism_analysis_seed18427
```

This does not repeat DEM or optimization. It compares a controlled random
allocation closest to the recorded random mean, the heat-path allocation and
the best swap-search allocation. The main output is:

```text
mechanism_analysis_seed18427/mechanism_comparison.csv
```

Particle and contact CSV/VTP files are also written for all three allocations.
The reported minimum-path resistance is a descriptive single-path quantity and
must not be interpreted as the total resistance of the parallel contact network.

### Generate the held-out GNN visualization

Evaluate the leave-one-packing-out GNN ensemble ID list for seed 18427 with the
unchanged thermal solver:

```bash
python3 solve_packing_heat_transfer.py packing_seed_18427 \
  --output thermal_results_gnn_loocv_ensemble \
  --high-ids gnn_loocv_results_rank_ensemble/predicted_ids/packing_seed_18427_gnn_ensemble.txt \
  --k-low 1.0 \
  --k-high 10.0 \
  --k-fluid 0.0 \
  --t-hot 301 \
  --t-cold 300
```

Check that the result contains 50 high-conductivity particles, its relative
energy-balance error is below `1e-8`, and its effective conductivity agrees with
fold 1 of `gnn_loocv_results_rank_ensemble/loocv_results.csv`.

### Files for the four paper subfigures

| Panel | Allocation | Particle VTP file |
|---|---|---|
| (a) | Random realization closest to the random mean | `mechanism_analysis_seed18427/particles_random_near_mean.vtp` |
| (b) | Heat-path ranking | `mechanism_analysis_seed18427/particles_heat_path.vtp` |
| (c) | Held-out GNN rank ensemble | `packing_seed_18427/thermal_results_gnn_loocv_ensemble/thermal_nodes.vtp` |
| (d) | Best fixed-budget same-size swap search | `mechanism_analysis_seed18427/particles_swap_optimized.vtp` |

The associated contact/edge files are:

```text
mechanism_analysis_seed18427/contacts_random_near_mean.vtp
mechanism_analysis_seed18427/contacts_heat_path.vtp
packing_seed_18427/thermal_results_gnn_loocv_ensemble/thermal_edges.vtp
mechanism_analysis_seed18427/contacts_swap_optimized.vtp
```

For panels (a), (b) and (d), colour particles by `high_conductivity` and use
`radius_m` for glyph scaling. For panel (c), colour by `conductivity` and use
`radius`; values 1 and 10 W/(m K) identify the low- and high-conductivity
materials. Use the same camera, projection, image size, crop, background, glyph
scale and low/high colours in all panels. If contacts are shown, also keep their
line width and heat-rate colour range fixed. Save the ParaView state after the
first view so that it can be applied to the remaining cases.

## 12. Build the graph-allocation dataset

### Purpose

The first graph-learning experiment is supervised node ranking, not
reinforcement learning. For each unseen packing, the model will assign one score
to every particle. At inference, particles are ranked separately within each
size class and the fixed quotas 10/30/10 are applied.

The fixed-budget same-size swap search is the teacher. To reduce dependence on a single
stochastic best allocation, the primary soft target is the fraction of the five
final optimizer restarts that selected each particle. A separate binary target
records membership in the best allocation.

### Required inputs for every packing

Each `packing_seed_*` directory must contain:

```text
particles_final.dump
contacts_final.dump
allocation_stratified_p3000/allocation_summary.json
allocation_stratified_p3000/high_ids_optimized.txt
allocation_stratified_p3000/high_ids_optimized_restart_1.txt
allocation_stratified_p3000/high_ids_optimized_restart_2.txt
allocation_stratified_p3000/high_ids_optimized_restart_3.txt
allocation_stratified_p3000/high_ids_optimized_restart_4.txt
allocation_stratified_p3000/high_ids_optimized_restart_5.txt
```

### Build all ten graphs

```bash
python3 build_gnn_allocation_dataset.py \
  --root . \
  --pattern 'packing_seed_*' \
  --allocation-template '{case}/allocation_stratified_p3000' \
  --output gnn_allocation_dataset
```

If allocation folders use another naming convention, change only the template.
To test one case before processing all packings:

```bash
python3 build_gnn_allocation_dataset.py \
  --root . \
  --case packing_seed_18427 \
  --allocation-template '{case}/allocation_stratified_p3000' \
  --output gnn_dataset_test
```

### Graph contents

Each `graph_packing_seed_<seed>.npz` contains:

| Array | Meaning |
|---|---|
| `node_ids` | Original LAMMPS particle IDs |
| `node_types` | Particle size classes 1/2/3 |
| `node_features` | Eight geometry and homogeneous-physics node features |
| `edge_index` | Directed contact pairs with shape `[2, 2E]` |
| `edge_features` | Six contact geometry and baseline-heat features |
| `target_selection_frequency` | Fraction of optimizer restarts selecting each particle |
| `target_best_selection` | Binary membership in the best recorded allocation |
| `box_lo_m`, `box_hi_m` | Simulation-box limits |

The node features, in order, are:

1. particle diameter divided by the reference diameter;
2. normalized vertical coordinate;
3. contact degree;
4. hot-wall contact flag;
5. cold-wall contact flag;
6. thermally anchored flag;
7. homogeneous-baseline normalized temperature;
8. homogeneous-baseline normalized heat-path score.

The directed edge features, in order, are:

1. contact radius divided by the reference diameter;
2. minimum-image `dx/dref`;
3. minimum-image `dy/dref`;
4. `dz/dref`;
5. absolute vertical alignment `|dz|/distance`;
6. normalized absolute baseline contact heat rate.

No optimized-allocation information is used as an input feature. The optimizer
outputs appear only in the training targets.

### Required checks

After a successful ten-case build, confirm:

- `dataset_manifest.csv` contains ten rows;
- every graph contains 500 nodes;
- directed edge count is twice the DEM contact count;
- the soft target sums to 10/30/10 within types 1/2/3;
- the binary best target contains exactly 50 selected particles;
- every graph has the same ordered feature names;
- no node or edge feature contains `NaN` or infinity.

The builder performs the quota and restart-file checks automatically and writes
per-graph JSON metadata plus `dataset_metadata.json`.

### Data-leakage rule

The ten packings are the independent samples. Do not randomly split particles
or contacts. Use leave-one-packing-out cross-validation: one complete graph is
the test case, one of the other graphs is reserved for validation and early
stopping, and the remaining eight graphs are used for training.

The builder performs only dimensionless and within-packing normalization.
Dataset-wide feature means and standard deviations must be fitted on the eight
training graphs inside each fold and then applied unchanged to the validation
and held-out test graphs.

## 13. Train and evaluate the allocation models

### Additional dependency

The training script requires PyTorch but not PyTorch Geometric. Install the
appropriate CPU or CUDA build using the command provided for the local machine
at <https://pytorch.org/get-started/locally/>. Confirm the installation with:

```bash
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

### Models and split

The experiment compares:

- a non-graph multilayer perceptron (`mlp`) using each particle's eight node
  features independently;
- an edge-aware message-passing graph neural network (`gnn`) using the same node
  features plus the six contact-edge features.

The outer loop holds out one complete packing for testing. One of the remaining
nine packings is held out for early stopping, leaving eight training graphs.
This is repeated until every packing has served as the test graph. Three
training repeats are converted to normalized ranks separately within each
particle type and those ranks are averaged before applying the 10/30/10 quotas.
Raw logits are not averaged because independently trained ranking models can
produce different score offsets and scales.

### Run the final leave-one-packing-out experiment

```bash
python3 train_evaluate_gnn_allocation_loocv.py \
  --dataset gnn_allocation_dataset \
  --project-root . \
  --output gnn_loocv_results_rank_ensemble \
  --models mlp,gnn \
  --repeats 3 \
  --epochs 400 \
  --patience 60 \
  --hidden 64 \
  --message-layers 3 \
  --seed 20260822 \
  --device cpu
```

Use `--device cpu` to force CPU execution or `--device cuda` to require a GPU.
For an initial pipeline check, reduce the calculation to:

```bash
python3 train_evaluate_gnn_allocation_loocv.py \
  --dataset gnn_allocation_dataset \
  --project-root . \
  --output gnn_loocv_smoke_test \
  --models mlp,gnn \
  --repeats 1 \
  --epochs 5 \
  --patience 5 \
  --device cpu
```

The smoke test checks execution and file compatibility only; its model results
have no scientific meaning.

### Outputs

```text
gnn_loocv_results_rank_ensemble/
  loocv_results.csv
  loocv_summary.json
  predicted_ids/
    packing_seed_<seed>_mlp_ensemble.txt
    packing_seed_<seed>_gnn_ensemble.txt
    ... individual repeat ID files ...
```

For every held-out packing and model, the script reports:

- predicted effective conductivity from the original physical solver;
- improvement relative to that packing's controlled random mean;
- difference from heat-path ranking;
- difference from the best fixed-budget teacher allocation;
- recovered optimizer-gain fraction;
- overlap with the teacher's best selected IDs;
- training and inference times.

The primary quantity is

```text
(k_model - k_random_mean) / (k_teacher_best - k_random_mean)
```

A value of zero means that the model adds nothing beyond random placement. A
value of one means that it recovers the complete improvement of the recorded
teacher allocation. Particle-label accuracy is secondary because different ID
sets can have similar thermal performance.

### Final corrected ML result

The three-repeat within-type rank ensemble produced:

| Model | Mean effective conductivity [W/(m K)] | Recovered search gain | Folds above heat-path ranking |
|---|---:|---:|---:|
| Same-feature MLP | 0.131373 +/- 0.005831 | 0.3361 +/- 0.0429 | 10/10 |
| Edge-aware GNN | 0.141895 +/- 0.007306 | 0.5072 +/- 0.0767 | 10/10 |

The GNN exceeded the MLP in all ten held-out packings. These are the final
reported transfer results for unseen stochastic realizations of the specified
500-particle bed. They do not establish extrapolation to different PSDs,
porosities, particle counts, conductivity ratios or boundary conditions.

### Computational-cost summary

The field `unique_thermal_evaluations` in each
`allocation_stratified_p3000/allocation_summary.json` counts distinct thermal
allocations actually solved during teacher generation; cached repeated
allocations are not counted twice. Across the ten packings:

```text
cases = 10
mean = 14828.9
standard deviation = 45.2
minimum = 14750
maximum = 14874
```

The corrected LOOCV table records the following CPU times for the three-repeat
ensembles:

| Model | Offline training per fold [s] | Inference per held-out packing [s] |
|---|---:|---:|
| MLP | 9.93 +/- 1.55 | 0.000343 +/- 0.000013 |
| GNN | 30.70 +/- 9.19 | 0.00396 +/- 0.00013 |

For deployment on a new packing, heat-path, MLP and GNN allocations require one
homogeneous thermal solve to obtain baseline physics and one final solve to
verify the selected ID list. Degree ranking requires only the final solve. The
full teacher-generation workflow required approximately 14829 distinct thermal
evaluations per packing. Thus, a trained GNN uses two online thermal solves plus
about 4 ms inference, roughly 7400 times fewer thermal evaluations. This is an
online deployment comparison; teacher generation and neural training remain
offline costs.

## 14. Generate the paper's fold-wise ML figure

Run from the project root after completing the corrected LOOCV calculation:

```bash
python3 plot_gnn_loocv_recovered_gain.py \
  --input gnn_loocv_results_rank_ensemble/loocv_results.csv \
  --output figures/gnn_loocv_recovered_gain.pdf
```

The script reads only rows with `repeat=ensemble` and writes:

```text
figures/gnn_loocv_recovered_gain.pdf
figures/gnn_loocv_recovered_gain.png
```

The PDF is the vector figure included by the LaTeX paper. The PNG is a 300-dpi
preview. The LaTeX source shows a placeholder instead of failing when the PDF
has not yet been generated.

## 15. Interpretation limits and next work package

The current conclusions apply to a stationary packing, contact-only conduction,
the chosen conductivity ratio and a 10% size-stratified allocation. The current
best assignment is the best found under the declared search budget, not a proven
global optimum. After all ten cases are complete, compare improvements relative
to each packing's own controlled random distribution. Only after cross-packing
generalization is established should a GNN placement policy be considered.

The ten-graph leave-one-packing-out calculation is the final transfer analysis
for stochastic realizations of the defined bed. The GNN is retained because it
outperformed heat-path ranking and the same-feature MLP in all held-out
packings, while producing its allocation without thousands of new swap-search
evaluations. Additional packings or operating conditions form a later scope
extension rather than a requirement for reproducing the present paper.
