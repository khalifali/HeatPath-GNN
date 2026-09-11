# DEM packing heat-transfer solver

This is the first physical solver for the conductive-particle allocation study.
It processes **one packing case at a time** and does not rerun LAMMPS.

## Required case files

Place the solver beside folders such as `packing_seed_18427/`. Each case must
contain:

- `particles_final.dump`
- `contacts_final.dump`

These are already produced by `in.generate_equal_porosity_psd_v4.lammps`.

The Python environment needs NumPy and SciPy.

## Step 1: homogeneous solid-contact test

Start with all particles at the same conductivity and disable stagnant-fluid
gap conduction:

```bash
python3 solve_packing_heat_transfer.py packing_seed_18427 \
  --k-low 1.0 \
  --k-high 10.0 \
  --k-fluid 0.0 \
  --t-hot 301 \
  --t-cold 300
```

Because no `--high-ids` file is supplied, every particle uses `--k-low`.
Results are written only to:

```text
packing_seed_18427/thermal_results/
```

Important files:

- `thermal_summary.json`: heat rates, effective conductivity and verification checks
- `node_temperatures.csv`: temperature and conductivity of every particle
- `edge_heat_flux.csv`: conductance and heat flow of every thermal edge
- `thermal_nodes.vtp`: ParaView particle field
- `thermal_edges.vtp`: ParaView contact/gap lines

The most important initial checks in `thermal_summary.json` are:

```text
hot_wall_links > 0
cold_wall_links > 0
spanning_component_count > 0
relative_energy_balance_error < 1e-8
temperature_min_K >= 300
temperature_max_K <= 301
```

## Step 2: include stagnant air

For a first diagnostic air-conduction calculation:

```bash
python3 solve_packing_heat_transfer.py packing_seed_18427 \
  --k-low 1.0 \
  --k-fluid 0.026 \
  --gap-cutoff-ratio 0.10 \
  --gap-regularization-ratio 0.001
```

This assumes stationary fluid, `u = 0`. It adds proximity edges only between
non-contacting particle surfaces separated by at most
`gap_cutoff_ratio * median_diameter`. The implemented lens approximation is:

```text
G_gap = k_fluid * A_eff / (surface_gap + gap_regularization)
A_eff = pi * effective_radius * gap_cutoff
```

This is a controlled reduced-order model, not CFD. The final study must report
sensitivity to `gap-cutoff-ratio` and `gap-regularization-ratio`.

## Step 3: assign selected conductive particles

Create a text file containing exactly one particle ID per line, for example:

```text
7
18
42
```

Then run:

```bash
python3 solve_packing_heat_transfer.py packing_seed_18427 \
  --high-ids selected_high_ids.txt \
  --k-low 1.0 \
  --k-high 10.0 \
  --k-fluid 0.026
```

The optimization and GNN will later generate these ID lists. They are not part
of this first solver-validation step.

## ParaView

Open `thermal_nodes.vtp`, apply **Glyph -> Sphere**, choose `diameter` as the
scale array, set scale factor to `1`, and color by `temperature` or
`conductivity`.

Open `thermal_edges.vtp` and color by:

- `edge_kind`: 0 = solid contact, 1 = stagnant-fluid gap
- `heat_i_to_j`: signed edge heat flow
- `conductance`: total edge conductance

## Built-in verification test

Run:

```bash
python3 test_heat_transfer_solver.py
```

The test creates a temporary three-particle column, checks wall/contact
connections, solves the temperature field, verifies energy conservation, and
checks the CSV/VTP outputs.

## Modeling cautions

- Solid contact area is calculated from DEM overlap using a Hertzian contact
  radius. Therefore, stiffness and packing pressure affect the thermal result.
- `overlap-floor-ratio` only regularizes zero overlaps caused by output
  precision; it must not be used to create unrealistically large contacts.
- The stagnant-fluid model contains no forced convection, natural convection,
  or radiation.
- A fluid-flowing packed bed would require an additional convective model and
  is outside this solver.
- The first ten DEM cases should initially be run with identical homogeneous
  thermal parameters. Material-allocation experiments begin only after all ten
  satisfy the same numerical checks.
