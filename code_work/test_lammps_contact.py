#!/usr/bin/env python3
"""Run a zero-step Hertz contact check with the actual server LAMMPS binary."""
import argparse
import math
from pathlib import Path
import subprocess
import tempfile

INPUT = '''clear
units si
dimension 3
boundary f f f
atom_style sphere
atom_modify map array
newton off
comm_modify vel yes
region domain block 0 0.02 0 0.02 0 0.02 units box
create_box 1 domain
create_atoms 1 single 0.005 0.005 0.005 units box
create_atoms 1 single 0.005 0.005 0.006999 units box
set group all diameter 0.002
set group all density 2500
pair_style granular
pair_coeff * * hertz/material 5.0e7 0.30 0.25 tangential mindlin_rescale NULL 1.0 0.50 damping tsuji
neighbor 0.00025 bin
neigh_modify delay 0 every 1 check yes
fix integration all nve/sphere
timestep 2e-7
run 0
write_dump all custom forces.dump id fx fy fz modify sort id format line "%d %.17g %.17g %.17g"
'''


def check(executable):
    with tempfile.TemporaryDirectory(prefix="heatpath_lammps_check_") as folder:
        root = Path(folder)
        (root/"check.in").write_text(INPUT)
        result = subprocess.run([executable,"-in","check.in"],cwd=root,
                                text=True,capture_output=True)
        if result.returncode:
            raise RuntimeError("LAMMPS contact check failed:\n"+result.stdout+result.stderr)
        lines=(root/"forces.dump").read_text().splitlines()
        start=next(i for i,line in enumerate(lines) if line.startswith("ITEM: ATOMS"))+1
        rows=[list(map(float,line.split())) for line in lines[start:] if line.strip()]
        expected=(4/3)*(5e7/(2*(1-0.25**2)))*math.sqrt(0.0005)*(1e-6)**1.5
        assert len(rows)==2
        assert math.isclose(rows[0][3],-expected,rel_tol=1e-8), (rows[0],expected)
        assert math.isclose(rows[1][3],expected,rel_tol=1e-8), (rows[1],expected)
        assert all(abs(row[axis])<1e-12 for row in rows for axis in [1,2])
    print("LAMMPS static Hertz force check: PASS")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps",default="lmp")
    args=parser.parse_args()
    check(args.lammps)
