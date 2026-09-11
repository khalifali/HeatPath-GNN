#!/usr/bin/env python3
"""Convert final LAMMPS packing dumps to ParaView VTP point datasets.

The default mode scans packing_seed_*/particles_final.dump below the current
directory.  It uses only Python's standard library.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def read_last_snapshot(path: Path) -> Tuple[List[str], List[List[float]]]:
    """Return column names and numeric rows from the last dump snapshot."""
    lines = path.read_text(encoding="utf-8").splitlines()
    snapshots: List[Tuple[List[str], List[List[float]]]] = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != "ITEM: TIMESTEP":
            i += 1
            continue

        if i + 8 >= len(lines):
            raise ValueError(f"Incomplete LAMMPS snapshot in {path}")

        i += 2  # skip ITEM: TIMESTEP and its value
        if lines[i].strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError(f"Missing atom-count header in {path}")
        natoms = int(lines[i + 1].strip())
        i += 2

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Missing box-bounds header in {path}")
        i += 4  # bounds header plus x/y/z bound lines

        if not lines[i].startswith("ITEM: ATOMS "):
            raise ValueError(f"Missing atom-data header in {path}")
        columns = lines[i].split()[2:]
        i += 1

        rows: List[List[float]] = []
        for _ in range(natoms):
            if i >= len(lines):
                raise ValueError(f"Incomplete atom table in {path}")
            values = [float(value) for value in lines[i].split()]
            if len(values) != len(columns):
                raise ValueError(
                    f"Column mismatch in {path}: expected {len(columns)}, "
                    f"found {len(values)}"
                )
            rows.append(values)
            i += 1

        snapshots.append((columns, rows))

    if not snapshots:
        raise ValueError(f"No LAMMPS snapshots found in {path}")
    return snapshots[-1]


def vtk_type(name: str) -> str:
    return "Int64" if name in {"id", "type"} else "Float64"


def scalar_array(name: str, values: List[float]) -> str:
    formatted = " ".join(
        str(int(value)) if vtk_type(name) == "Int64" else f"{value:.16e}"
        for value in values
    )
    return (
        f'        <DataArray type="{vtk_type(name)}" Name="{name}" '
        f'format="ascii">\n          {formatted}\n        </DataArray>\n'
    )


def vector_array(name: str, vectors: List[Tuple[float, float, float]]) -> str:
    formatted = " ".join(
        f"{x:.16e} {y:.16e} {z:.16e}" for x, y, z in vectors
    )
    return (
        f'        <DataArray type="Float64" Name="{name}" '
        f'NumberOfComponents="3" format="ascii">\n'
        f"          {formatted}\n"
        "        </DataArray>\n"
    )


def write_vtp(source: Path, destination: Path) -> int:
    columns, rows = read_last_snapshot(source)
    index: Dict[str, int] = {name: position for position, name in enumerate(columns)}

    required = {"id", "type", "x", "y", "z", "radius"}
    missing = sorted(required.difference(index))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")

    points = [(row[index["x"]], row[index["y"]], row[index["z"]]) for row in rows]
    npoints = len(points)

    point_data = ""
    for name in ("id", "type", "radius", "mass"):
        if name in index:
            point_data += scalar_array(name, [row[index[name]] for row in rows])

    point_data += scalar_array("diameter", [2.0 * row[index["radius"]] for row in rows])

    if {"fx", "fy", "fz"}.issubset(index):
        forces = [
            (row[index["fx"]], row[index["fy"]], row[index["fz"]]) for row in rows
        ]
        point_data += vector_array("force", forces)

    point_text = " ".join(f"{x:.16e} {y:.16e} {z:.16e}" for x, y, z in points)
    connectivity = " ".join(str(value) for value in range(npoints))
    offsets = " ".join(str(value) for value in range(1, npoints + 1))

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{npoints}" NumberOfVerts="{npoints}" '
        'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n'
        "      <PointData>\n"
        f"{point_data}"
        "      </PointData>\n"
        "      <CellData/>\n"
        "      <Points>\n"
        '        <DataArray type="Float64" NumberOfComponents="3" '
        'Name="Points" format="ascii">\n'
        f"          {point_text}\n"
        "        </DataArray>\n"
        "      </Points>\n"
        "      <Verts>\n"
        '        <DataArray type="Int64" Name="connectivity" format="ascii">\n'
        f"          {connectivity}\n"
        "        </DataArray>\n"
        '        <DataArray type="Int64" Name="offsets" format="ascii">\n'
        f"          {offsets}\n"
        "        </DataArray>\n"
        "      </Verts>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )

    destination.write_text(xml, encoding="utf-8")
    return npoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS packing particle dumps to ParaView VTP files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Directory containing packing_seed_* folders (default: current directory).",
    )
    parser.add_argument(
        "--pattern",
        default="packing_seed_*/particles_final.dump",
        help="Glob pattern below --root.",
    )
    parser.add_argument(
        "--output-name",
        default="packing_final.vtp",
        help="VTP filename written next to each source dump.",
    )
    args = parser.parse_args()

    sources = sorted(args.root.glob(args.pattern))
    if not sources:
        raise SystemExit(
            f"No files matched {args.pattern!r} below {args.root.resolve()}"
        )

    for source in sources:
        destination = source.with_name(args.output_name)
        count = write_vtp(source, destination)
        print(f"{source} -> {destination} ({count} particles)")


if __name__ == "__main__":
    main()
