# Steady packing heat-transfer solver

Run from `code_work/`. The solver needs NumPy, SciPy and the case files
`particles_final.dump` and `contacts_final.dump`, produced by
`in.generate_equal_porosity_psd.lammps`.

## Homogeneous contact-only baseline

```bash
python3 solve_packing_heat_transfer.py packing_seed_18427 \
  --output thermal_results_contact --k-low 1.0 --k-fluid 0.0 \
  --t-hot 301 --t-cold 300
```

## Evaluate the final searched allocation

```bash
python3 solve_packing_heat_transfer.py packing_seed_18427 \
  --output thermal_results_stratified_optimized_p3000 \
  --high-ids packing_seed_18427/allocation_stratified_p3000/high_ids_optimized.txt \
  --k-low 1.0 --k-high 10.0 --k-fluid 0.0 --t-hot 301 --t-cold 300
```

Outputs are written inside the case directory: a thermal summary, particle
temperatures, contact heat rates and particle/contact VTP files for ParaView.
Check energy balance, wall-spanning connectivity and the temperature bounds.
Open `thermal_nodes.vtp` and apply Sphere Glyphs using the diameter array.

```bash
python3 test_heat_transfer_solver.py
```

The existing test checks a connected three-particle column plus an isolated
particle, thermal connectivity, conservation and output generation.

## Scope

The paper uses contact-only conduction. The optional `--k-fluid` gap model is
an exploratory diagnostic with cutoff sensitivity; its saved sensitivity
results are retained, but it is not used for the paper's allocation results.
Contact radii depend on DEM overlap and therefore on the mechanical model.

See [the reproduction workflow](REPRODUCTION_WORKFLOW_20260822.md) for the
complete verification, allocation and learning commands.
