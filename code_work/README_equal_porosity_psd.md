# Equal-porosity DEM packings

The current generator is `in.generate_equal_porosity_psd.lammps`.
It contains the corrected contact-ID formatting formerly described as v5.

Run from `code_work/`:

```bash
mpirun -np 8 lmp -var seed_realization 18427 -in in.generate_equal_porosity_psd.lammps
```

Use `bash run_all` to regenerate all ten packings. This writes into the existing
case folders, so use it only when regeneration is intended.

Each packing contains 100/300/100 spheres of diameter 1.5/2.0/2.5 mm,
with nominal porosity 0.38, periodic x/y boundaries and fixed z walls.
Only the realization seed changes between cases.

Keep `particles_final.dump`, `contacts_final.dump`, `packing_summary.dat`,
`packing_final.data` and `packing_final.restart`. The final particle/contact
dumps are the thermal solver inputs. Intermediate `growth_*.dump` snapshots
are not needed for the paper's thermal or learning pipeline.

See [the reproduction workflow](REPRODUCTION_WORKFLOW_20260822.md) for the
seed list, mechanical checks and subsequent analysis.
