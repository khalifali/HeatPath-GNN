# Contact-model audit — 2026-09-11

**Update:** The 30-packing campaign corrects the thermal prefactor and DEM
argument order. All nine regression tests now pass. The findings below describe
the pre-correction model; old stored results have not been regenerated. See
[campaign30.md](campaign30.md) for the new workflow.

Run from code_work:

```bash
python3 test_contact_model.py
```

This is an independent reference audit. On the current production model it
reports **9 tests, 3 passes, 6 failures** and exits nonzero. Failures are
deliberately visible; they are not marked expectedFailure or skipped.
The prior conservation test can pass despite these discrepancies.

## 1. Thermal prefactor

For an ideal circular isothermal contact of radius a, the spreading
resistances of the two half-spaces in series are

R = 1/(4 k_i a) + 1/(4 k_j a),
G = 4 a k_i k_j/(k_i+k_j).

For equal conductivities this becomes G=2ka; for an infinitely conducting
wall it becomes G=4ka on the particle side. The implementation instead
uses 2 a k_i k_j/(k_i+k_j), and 2ka for the infinite wall.
These conductances are half the reference values.

LAMMPS's documented heat-radius model gives Q=2 k_s a deltaT.
The same expression was checked in its stable_22Jul2025_update5 source
documentation, the version named by the paper:
https://github.com/lammps/lammps/blob/stable_22Jul2025_update5/doc/src/pair_granular.rst
https://docs.lammps.org/pair_granular.html

The dissimilar-material and wall references in the tests are derived from
the two spreading resistances, rather than copied from the solver formula.
They assume the ideal constriction model, without roughness, interfacial
resistance or finite-sphere corrections.

The absolute two-particle-column benchmark produces
2.61971658966e-5 W instead of 5.23943317932e-5 W for deltaT=1 K,
while still conserving heat.

For fixed geometry and contact-only conduction, doubling ALL particle and
wall conductances doubles heat rates and effective conductivity and leaves
temperatures and relative allocation gains unchanged mathematically.
This is not a statement that regenerating DEM packings leaves results unchanged.
Existing result files and paper equations have not been updated in this audit.

## 2. DEM argument order

The version-specific LAMMPS syntax is:
hertz/material E restitution Poisson_ratio (when using damping tsuji).

Both particle and wall definitions in the uploaded input instead pass:
hertz/material ${young} ${poisson} ${restitution}.

Thus this input requests restitution 0.25 and Poisson ratio 0.30, opposite
to the named variables and paper description (0.30 and 0.25 respectively).
This establishes a discrepancy in the supplied generator; historical run
provenance should be checked before asserting that every saved packing was
generated with this exact input. Correcting future generation can change
packing dynamics and contact geometry.

## 3. Geometry and physical validity

The Hertz radius a=sqrt(R_eff*overlap) passes the independent force-radius
identity test, including unequal spheres. The periodic contact geometry
test also passes.

Across the ten saved particle/contact datasets:
- 13,483 particle-particle contacts.
- 36 at or below the default overlap floor.
- a/min(R_i,R_j): median 0.2018, 95th percentile 0.3024, maximum 0.4642.

These relatively large contacts warrant checking the small-contact
approximation and sensitivity to packing mechanics. Passing analytical
tests will not experimentally validate the packed-bed thermal model.

The solver clamps negative reconstructed overlaps to zero and then applies
the floor to listed contacts; a future robustness check should distinguish
roundoff from inconsistent contact lists. Wall detection likewise permits
the configured small positive gap tolerance.

LIGGGHTS documents an overlap-area correction for softened Young's moduli:
https://www.cfdem.com/media/DEM/docu/fix_heat_gran_conduction.html
For Hertz contacts, that area scaling is (E*_soft/E*_physical)^(2/3).
A fixed-load correction should not be interpreted as equivalent to
regenerating this fixed-volume bed at a different stiffness.

## Tests and proposed correction check

Tests cover equal/dissimilar materials, finite/infinite wall conductivity,
zero radius and symmetry, Hertz radius, periodic geometry, an absolute
series-column heat rate and both DEM argument lists.

A temporary copy with the thermal prefactors changed from 2 to 4 and both
DEM argument lists reordered passed all nine tests. No production solver,
DEM generator, manuscript or scientific output was changed.
No LAMMPS simulation or full ML training was run.

Recommended sequence: correct the thermal prefactor and update affected
absolute results; resolve the historical DEM parameters; then regenerate
new packings with explicit verified settings and assess contact sensitivity.

