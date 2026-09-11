#!/usr/bin/env python3
"""Small deterministic verification test for solve_packing_heat_transfer.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


PARTICLES = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp ff
0.0 3.0
0.0 1.0
0.0 2.96
ITEM: ATOMS id type x y z radius mass fx fy fz
1 1 0.5 0.5 0.49 0.5 1.0 0 0 0
2 1 0.5 0.5 1.48 0.5 1.0 0 0 0
3 1 0.5 0.5 2.47 0.5 1.0 0 0 0
4 1 2.0 0.5 1.48 0.1 0.01 0 0 0
"""

CONTACTS = """ITEM: TIMESTEP
1
ITEM: NUMBER OF ENTRIES
2
ITEM: BOX BOUNDS pp pp ff
0.0 1.0
0.0 1.0
0.0 2.96
ITEM: ENTRIES index c_pairIDs[1] c_pairIDs[2] c_pairIDs[3] c_pairIDs[4] c_pairState[1]
1 1 2 1 1 0.99
2 2 3 1 1 0.99
"""


def main() -> None:
    solver = Path(__file__).resolve().with_name("solve_packing_heat_transfer.py")
    with tempfile.TemporaryDirectory(prefix="thermal_solver_test_") as tmp:
        case = Path(tmp)
        (case / "particles_final.dump").write_text(PARTICLES, encoding="utf-8")
        (case / "contacts_final.dump").write_text(CONTACTS, encoding="utf-8")
        subprocess.run(
            [sys.executable, str(solver), str(case), "--t-hot", "1", "--t-cold", "0"],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads((case / "thermal_results" / "thermal_summary.json").read_text())
        assert summary["particles"] == 4
        assert summary["thermally_solved_particles"] == 3
        assert summary["thermally_unanchored_particles"] == 1
        assert summary["graph_diagnostics"]["unanchored_component_sizes"] == [1]
        assert summary["contact_edges"] == 2
        assert summary["hot_wall_contact_links"] == 1
        assert summary["cold_wall_contact_links"] == 1
        assert summary["relative_energy_balance_error"] < 1.0e-10
        assert summary["effective_conductivity_W_mK"] > 0.0
        node_rows = (case / "thermal_results" / "node_temperatures.csv").read_text().splitlines()
        assert len(node_rows) == 5
        assert (case / "thermal_results" / "thermal_nodes.vtp").exists()
        assert (case / "thermal_results" / "thermal_edges.vtp").exists()
    print("heat-transfer solver self-test: PASS")


if __name__ == "__main__":
    main()
