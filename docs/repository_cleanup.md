# Repository cleanup — 2026-09-11

## Basis

Read the current manuscript `paper/conductive_particle_allocation_packed_bed_paper_draft_v2.tex`
and traced the reproduction workflow and Python imports. The current study is
the ten-packing, contact-only, 10/30/10 allocation experiment, followed by
three-repeat within-type rank ensembles for MLP/GNN evaluation.

Original import commit: `8a37ede1ed48eff83697d13f7bcadd83e5be6d8c`. Removed files remain recoverable
from that commit. No scientific model or manuscript text was changed.

## Active code retained

| File in code_work | Purpose |
|---|---|
| in.generate_equal_porosity_psd.lammps | Corrected DEM generator; older notes called it v5 |
| run_all | Generate all ten packings |
| solve_packing_heat_transfer.py | Shared physical evaluator |
| test_heat_transfer_solver.py | Existing deterministic verification |
| run_heattransfer_all.sh | Batch homogeneous thermal baseline |
| combine_therm_comp.sh | Combine thermal reference summaries |
| run_allocation_pilot_stratified_v2.py | Current size-constrained search and shared evaluation helpers |
| run_all_cases_stratified_allocation_v1.sh | Current batch search and independent verification |
| build_gnn_allocation_dataset.py | Graph features and search-frequency targets |
| train_evaluate_gnn_allocation_loocv.py | Same-feature MLP and edge-aware GNN evaluation |
| analyze_allocation_mechanisms.py | Paper mechanism comparison and visualization data |
| plot_gnn_loocv_recovered_gain.py | Paper rank-ensemble comparison figure |
| packing_dump_to_vtp.py | Particle visualization utility |
| clean_all_packing_cases.sh | Explicit case-removal utility; preview by default |

The v1/v2 suffixes do not indicate obsolescence: the batch runner uses the
stratified v2 script, which is also imported by dataset construction, learning
and mechanism analysis.

## Removed groups

| Group | Reason |
|---|---|
| run_allocation_pilot.py | Earlier count-only allocation; lacks the paper's size-class budget; no active imports |
| allocation_pilot_seed18427/ | Superseded unconstrained pilot |
| allocation_pilot_stratified_seed18427/ | Earlier search budget |
| allocation_pilot_stratified_seed18427_p1200/ | Earlier 1200-proposal search |
| allocation_pilot_stratified_seed18427_p3000/ | All 12 files have identical Git blob hashes to the retained per-case allocation_stratified_p3000 folder |
| gnn_loocv_results/ | Superseded ML ensemble outputs; current manuscript matches gnn_loocv_results_rank_ensemble |
| gnn_loocv_smoke_test/ | Pipeline execution check, not scientific results |
| packing_seed_*/growth_*.dump | Intermediate DEM snapshots; final geometry, contacts, data, restart and diagnostics retained |
| __pycache__/ and run_all~ | Python bytecode and editor backup |
| LaTeX auxiliary files | Rebuildable compilation output; source, PDFs and figures retained |

Total removed: **320 files**, **19.44 MB**
of uncompressed tracked content. Git history still contains the original files;
this does not shrink existing Git history.

## Scientific records retained

- All ten final packing datasets and DEM restart/data files.
- All per-case final search results, five restart ID lists and histories.
- Graph dataset and corrected rank-ensemble results, including predicted IDs.
- Homogeneous and optimized thermal results.
- Conductivity/temperature scaling checks and exploratory air-gap sensitivity
  results: these document verification and a limitation explicitly discussed
  in the manuscript.
- Mechanism analysis and ParaView data/state.
- Both paper drafts, concept and proposal sources, PDFs and figures. Older
  document drafts are left for the later content review.

## Small maintenance changes

- Added the root README and .gitignore.
- Replaced the misleading packing README (which repeated the thermal README).
- Updated the solver guide to the current contact-only allocation workflow.
- Corrected references to the actual DEM filename and removed instructions for
  an absent legacy contact-repair script.
- Made the default search output CASE/allocation_stratified_p<swap-proposals> and the default
  ML output gnn_loocv_results_rank_ensemble. Explicit output paths still work.

## Verification

The existing thermal solver self-test passed. Independent solver calls on all
ten saved best allocations reproduced their saved conductivities with maximum
absolute difference 9.89e-14 W/(m K); maximum relative energy imbalance was
2.82e-12. Python source compilation passed. DEM generation and full neural
training were not rerun. PyTorch is unavailable in this environment, so the
ML command could not be runtime-tested; its only change is the output-directory default.

## Exact removal inventory

- `code_work/__pycache__/run_allocation_pilot_stratified_v2.cpython-312.pyc`
- `code_work/__pycache__/solve_packing_heat_transfer.cpython-312.pyc`
- `code_work/allocation_pilot_seed18427/allocation_results.csv`
- `code_work/allocation_pilot_seed18427/allocation_summary.json`
- `code_work/allocation_pilot_seed18427/high_ids_best_random.txt`
- `code_work/allocation_pilot_seed18427/high_ids_degree.txt`
- `code_work/allocation_pilot_seed18427/high_ids_heat_path.txt`
- `code_work/allocation_pilot_seed18427/high_ids_optimized.txt`
- `code_work/allocation_pilot_seed18427/optimization_history.csv`
- `code_work/allocation_pilot_stratified_seed18427/allocation_results.csv`
- `code_work/allocation_pilot_stratified_seed18427/allocation_summary.json`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_best_random.txt`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_degree.txt`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_heat_path.txt`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_optimized.txt`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_optimized_restart_1.txt`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_optimized_restart_2.txt`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_optimized_restart_3.txt`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_optimized_restart_4.txt`
- `code_work/allocation_pilot_stratified_seed18427/high_ids_optimized_restart_5.txt`
- `code_work/allocation_pilot_stratified_seed18427/optimization_history.csv`
- `code_work/allocation_pilot_stratified_seed18427_p1200/allocation_results.csv`
- `code_work/allocation_pilot_stratified_seed18427_p1200/allocation_summary.json`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_best_random.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_degree.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_heat_path.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_optimized.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_optimized_restart_1.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_optimized_restart_2.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_optimized_restart_3.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_optimized_restart_4.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/high_ids_optimized_restart_5.txt`
- `code_work/allocation_pilot_stratified_seed18427_p1200/optimization_history.csv`
- `code_work/allocation_pilot_stratified_seed18427_p3000/allocation_results.csv`
- `code_work/allocation_pilot_stratified_seed18427_p3000/allocation_summary.json`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_best_random.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_degree.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_heat_path.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_optimized.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_optimized_restart_1.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_optimized_restart_2.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_optimized_restart_3.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_optimized_restart_4.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/high_ids_optimized_restart_5.txt`
- `code_work/allocation_pilot_stratified_seed18427_p3000/optimization_history.csv`
- `code_work/gnn_loocv_results/loocv_results.csv`
- `code_work/gnn_loocv_results/loocv_summary.json`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_18427_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_18427_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_18427_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_18427_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_18427_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_18427_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_18427_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_18427_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_29173_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_29173_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_29173_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_29173_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_29173_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_29173_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_29173_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_29173_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_37811_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_37811_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_37811_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_37811_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_37811_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_37811_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_37811_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_37811_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_46549_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_46549_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_46549_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_46549_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_46549_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_46549_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_46549_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_46549_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_55291_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_55291_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_55291_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_55291_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_55291_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_55291_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_55291_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_55291_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_64109_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_64109_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_64109_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_64109_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_64109_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_64109_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_64109_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_64109_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_72869_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_72869_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_72869_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_72869_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_72869_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_72869_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_72869_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_72869_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_81647_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_81647_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_81647_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_81647_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_81647_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_81647_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_81647_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_81647_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_90439_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_90439_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_90439_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_90439_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_90439_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_90439_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_90439_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_90439_mlp_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_99223_gnn_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_99223_gnn_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_99223_gnn_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_99223_gnn_repeat3.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_99223_mlp_ensemble.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_99223_mlp_repeat1.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_99223_mlp_repeat2.txt`
- `code_work/gnn_loocv_results/predicted_ids/packing_seed_99223_mlp_repeat3.txt`
- `code_work/gnn_loocv_smoke_test/loocv_results.csv`
- `code_work/gnn_loocv_smoke_test/loocv_summary.json`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_18427_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_18427_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_18427_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_18427_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_29173_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_29173_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_29173_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_29173_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_37811_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_37811_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_37811_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_37811_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_46549_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_46549_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_46549_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_46549_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_55291_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_55291_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_55291_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_55291_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_64109_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_64109_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_64109_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_64109_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_72869_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_72869_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_72869_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_72869_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_81647_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_81647_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_81647_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_81647_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_90439_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_90439_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_90439_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_90439_mlp_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_99223_gnn_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_99223_gnn_repeat1.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_99223_mlp_ensemble.txt`
- `code_work/gnn_loocv_smoke_test/predicted_ids/packing_seed_99223_mlp_repeat1.txt`
- `code_work/packing_seed_18427/growth_0.dump`
- `code_work/packing_seed_18427/growth_100000.dump`
- `code_work/packing_seed_18427/growth_125000.dump`
- `code_work/packing_seed_18427/growth_150000.dump`
- `code_work/packing_seed_18427/growth_175000.dump`
- `code_work/packing_seed_18427/growth_200000.dump`
- `code_work/packing_seed_18427/growth_225000.dump`
- `code_work/packing_seed_18427/growth_25000.dump`
- `code_work/packing_seed_18427/growth_250000.dump`
- `code_work/packing_seed_18427/growth_275000.dump`
- `code_work/packing_seed_18427/growth_300000.dump`
- `code_work/packing_seed_18427/growth_50000.dump`
- `code_work/packing_seed_18427/growth_75000.dump`
- `code_work/packing_seed_29173/growth_0.dump`
- `code_work/packing_seed_29173/growth_100000.dump`
- `code_work/packing_seed_29173/growth_125000.dump`
- `code_work/packing_seed_29173/growth_150000.dump`
- `code_work/packing_seed_29173/growth_175000.dump`
- `code_work/packing_seed_29173/growth_200000.dump`
- `code_work/packing_seed_29173/growth_225000.dump`
- `code_work/packing_seed_29173/growth_25000.dump`
- `code_work/packing_seed_29173/growth_250000.dump`
- `code_work/packing_seed_29173/growth_275000.dump`
- `code_work/packing_seed_29173/growth_300000.dump`
- `code_work/packing_seed_29173/growth_50000.dump`
- `code_work/packing_seed_29173/growth_75000.dump`
- `code_work/packing_seed_37811/growth_0.dump`
- `code_work/packing_seed_37811/growth_100000.dump`
- `code_work/packing_seed_37811/growth_125000.dump`
- `code_work/packing_seed_37811/growth_150000.dump`
- `code_work/packing_seed_37811/growth_175000.dump`
- `code_work/packing_seed_37811/growth_200000.dump`
- `code_work/packing_seed_37811/growth_225000.dump`
- `code_work/packing_seed_37811/growth_25000.dump`
- `code_work/packing_seed_37811/growth_250000.dump`
- `code_work/packing_seed_37811/growth_275000.dump`
- `code_work/packing_seed_37811/growth_300000.dump`
- `code_work/packing_seed_37811/growth_50000.dump`
- `code_work/packing_seed_37811/growth_75000.dump`
- `code_work/packing_seed_46549/growth_0.dump`
- `code_work/packing_seed_46549/growth_100000.dump`
- `code_work/packing_seed_46549/growth_125000.dump`
- `code_work/packing_seed_46549/growth_150000.dump`
- `code_work/packing_seed_46549/growth_175000.dump`
- `code_work/packing_seed_46549/growth_200000.dump`
- `code_work/packing_seed_46549/growth_225000.dump`
- `code_work/packing_seed_46549/growth_25000.dump`
- `code_work/packing_seed_46549/growth_250000.dump`
- `code_work/packing_seed_46549/growth_275000.dump`
- `code_work/packing_seed_46549/growth_300000.dump`
- `code_work/packing_seed_46549/growth_50000.dump`
- `code_work/packing_seed_46549/growth_75000.dump`
- `code_work/packing_seed_55291/growth_0.dump`
- `code_work/packing_seed_55291/growth_100000.dump`
- `code_work/packing_seed_55291/growth_125000.dump`
- `code_work/packing_seed_55291/growth_150000.dump`
- `code_work/packing_seed_55291/growth_175000.dump`
- `code_work/packing_seed_55291/growth_200000.dump`
- `code_work/packing_seed_55291/growth_225000.dump`
- `code_work/packing_seed_55291/growth_25000.dump`
- `code_work/packing_seed_55291/growth_250000.dump`
- `code_work/packing_seed_55291/growth_275000.dump`
- `code_work/packing_seed_55291/growth_300000.dump`
- `code_work/packing_seed_55291/growth_50000.dump`
- `code_work/packing_seed_55291/growth_75000.dump`
- `code_work/packing_seed_64109/growth_0.dump`
- `code_work/packing_seed_64109/growth_100000.dump`
- `code_work/packing_seed_64109/growth_125000.dump`
- `code_work/packing_seed_64109/growth_150000.dump`
- `code_work/packing_seed_64109/growth_175000.dump`
- `code_work/packing_seed_64109/growth_200000.dump`
- `code_work/packing_seed_64109/growth_225000.dump`
- `code_work/packing_seed_64109/growth_25000.dump`
- `code_work/packing_seed_64109/growth_250000.dump`
- `code_work/packing_seed_64109/growth_275000.dump`
- `code_work/packing_seed_64109/growth_300000.dump`
- `code_work/packing_seed_64109/growth_50000.dump`
- `code_work/packing_seed_64109/growth_75000.dump`
- `code_work/packing_seed_72869/growth_0.dump`
- `code_work/packing_seed_72869/growth_100000.dump`
- `code_work/packing_seed_72869/growth_125000.dump`
- `code_work/packing_seed_72869/growth_150000.dump`
- `code_work/packing_seed_72869/growth_175000.dump`
- `code_work/packing_seed_72869/growth_200000.dump`
- `code_work/packing_seed_72869/growth_225000.dump`
- `code_work/packing_seed_72869/growth_25000.dump`
- `code_work/packing_seed_72869/growth_250000.dump`
- `code_work/packing_seed_72869/growth_275000.dump`
- `code_work/packing_seed_72869/growth_300000.dump`
- `code_work/packing_seed_72869/growth_50000.dump`
- `code_work/packing_seed_72869/growth_75000.dump`
- `code_work/packing_seed_81647/growth_0.dump`
- `code_work/packing_seed_81647/growth_100000.dump`
- `code_work/packing_seed_81647/growth_125000.dump`
- `code_work/packing_seed_81647/growth_150000.dump`
- `code_work/packing_seed_81647/growth_175000.dump`
- `code_work/packing_seed_81647/growth_200000.dump`
- `code_work/packing_seed_81647/growth_225000.dump`
- `code_work/packing_seed_81647/growth_25000.dump`
- `code_work/packing_seed_81647/growth_250000.dump`
- `code_work/packing_seed_81647/growth_275000.dump`
- `code_work/packing_seed_81647/growth_300000.dump`
- `code_work/packing_seed_81647/growth_50000.dump`
- `code_work/packing_seed_81647/growth_75000.dump`
- `code_work/packing_seed_90439/growth_0.dump`
- `code_work/packing_seed_90439/growth_100000.dump`
- `code_work/packing_seed_90439/growth_125000.dump`
- `code_work/packing_seed_90439/growth_150000.dump`
- `code_work/packing_seed_90439/growth_175000.dump`
- `code_work/packing_seed_90439/growth_200000.dump`
- `code_work/packing_seed_90439/growth_225000.dump`
- `code_work/packing_seed_90439/growth_25000.dump`
- `code_work/packing_seed_90439/growth_250000.dump`
- `code_work/packing_seed_90439/growth_275000.dump`
- `code_work/packing_seed_90439/growth_300000.dump`
- `code_work/packing_seed_90439/growth_50000.dump`
- `code_work/packing_seed_90439/growth_75000.dump`
- `code_work/packing_seed_99223/growth_0.dump`
- `code_work/packing_seed_99223/growth_100000.dump`
- `code_work/packing_seed_99223/growth_125000.dump`
- `code_work/packing_seed_99223/growth_150000.dump`
- `code_work/packing_seed_99223/growth_175000.dump`
- `code_work/packing_seed_99223/growth_200000.dump`
- `code_work/packing_seed_99223/growth_225000.dump`
- `code_work/packing_seed_99223/growth_25000.dump`
- `code_work/packing_seed_99223/growth_250000.dump`
- `code_work/packing_seed_99223/growth_275000.dump`
- `code_work/packing_seed_99223/growth_300000.dump`
- `code_work/packing_seed_99223/growth_50000.dump`
- `code_work/packing_seed_99223/growth_75000.dump`
- `code_work/run_allocation_pilot.py`
- `code_work/run_all~`
- `concept/conductive_particle_allocation_concept.aux`
- `concept/conductive_particle_allocation_concept.log`
- `concept/conductive_particle_allocation_concept.out`
- `concept/conductive_particle_allocation_concept.synctex.gz`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v1.aux`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v1.log`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v1.out`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v1.synctex.gz`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v2.aux`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v2.log`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v2.nav`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v2.out`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v2.snm`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v2.synctex.gz`
- `paper/conductive_particle_allocation_packed_bed_paper_draft_v2.toc`
- `proposal/hybrid_dem_graph_thermal_allocation_proposal.aux`
- `proposal/hybrid_dem_graph_thermal_allocation_proposal.log`
- `proposal/hybrid_dem_graph_thermal_allocation_proposal.out`
- `proposal/hybrid_dem_graph_thermal_allocation_proposal.synctex.gz`
