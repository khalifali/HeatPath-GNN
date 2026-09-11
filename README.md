# HeatPath-GNN

Physics-guided allocation of conductive particles in fixed DEM packings, with
a steady thermal network and MLP/GNN allocation models.

## Start here

- [Paper, current draft v2](paper/conductive_particle_allocation_packed_bed_paper_draft_v2.pdf)
- [Complete reproduction workflow](code_work/REPRODUCTION_WORKFLOW_20260822.md)
- [Cleanup inventory](docs/repository_cleanup.md)
- [Concept](concept/conductive_particle_allocation_concept.pdf)
- [Proposal](proposal/hybrid_dem_graph_thermal_allocation_proposal.pdf)

Run the computational commands from `code_work/`. Python requires NumPy and
SciPy, with PyTorch for learning and Matplotlib for plotting. DEM generation
uses LAMMPS with the GRANULAR package; existing final packings allow thermal
analysis without rerunning DEM.

```bash
cd code_work
python3 test_heat_transfer_solver.py
```

The paper studies ten 500-particle packings and contact-only conduction
(`--k-fluid 0.0`). Every allocation uses the same 10/30/10 size-class quota.
The final teacher results are in each `packing_seed_*/allocation_stratified_p3000/`;
the final learned-model results are in `gnn_loocv_results_rank_ensemble/`.

## Repository contents

| Location | Contents |
|---|---|
| `code_work/` | DEM input, Python scripts, run helpers, final packings and reference results |
| `paper/` | Current v2 manuscript, earlier v1 draft, PDFs and figures |
| `concept/` | Short concept source and PDF |
| `proposal/` | Proposal source and PDFs |
| `state.pvsm` | ParaView state; reassign local data paths when opening on another machine |

Final data and scientific reference results are deliberately tracked.
Intermediate growth snapshots, Python caches and LaTeX compilation auxiliaries
are ignored. Git history preserves files removed during cleanup.
