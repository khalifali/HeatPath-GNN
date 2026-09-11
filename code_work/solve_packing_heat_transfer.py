#!/usr/bin/env python3
"""Steady heat-transfer network solver for the DEM packing project.

The program reads one LAMMPS case directory containing
``particles_final.dump`` and ``contacts_final.dump``.  It constructs a sparse
thermal network, applies fixed temperatures to the z walls, solves the nodal
temperatures, and writes CSV/VTP results plus a JSON summary.

Model levels
------------
1. Solid contact conduction from DEM contacts and Hertzian contact radii.
2. Optional stagnant-fluid conduction between nearby, non-contacting spheres.

No fluid velocity, forced convection, natural convection, or radiation is
solved.  The optional gap model is deliberately parameterized so its
sensitivity can be reported rather than hidden.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Particle:
    atom_id: int
    atom_type: int
    x: float
    y: float
    z: float
    radius: float
    mass: float


@dataclass(frozen=True)
class Box:
    lo: np.ndarray
    hi: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def area_z(self) -> float:
        return float(self.lengths[0] * self.lengths[1])

    @property
    def height(self) -> float:
        return float(self.lengths[2])


@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    kind: str
    distance: float
    surface_gap: float
    overlap: float
    contact_radius: float
    g_contact: float
    g_gap: float

    @property
    def conductance(self) -> float:
        return self.g_contact + self.g_gap


def _read_last_lammps_frame(path: Path) -> tuple[list[str], np.ndarray, Box]:
    """Return column names, numeric data, and box from the final dump frame."""
    lines = path.read_text(encoding="utf-8").splitlines()
    frame_starts = [i for i, line in enumerate(lines) if line.startswith("ITEM: TIMESTEP")]
    if not frame_starts:
        raise ValueError(f"No LAMMPS dump frame found in {path}")
    start = frame_starts[-1]
    try:
        n_header = start + 2
        if lines[n_header] != "ITEM: NUMBER OF ATOMS" and lines[n_header] != "ITEM: NUMBER OF ENTRIES":
            raise ValueError
        n = int(lines[n_header + 1])
        box_header = n_header + 2
        if not lines[box_header].startswith("ITEM: BOX BOUNDS"):
            raise ValueError
        bounds = []
        for line in lines[box_header + 1 : box_header + 4]:
            values = [float(v) for v in line.split()[:2]]
            bounds.append(values)
        data_header = box_header + 4
        if not (lines[data_header].startswith("ITEM: ATOMS") or lines[data_header].startswith("ITEM: ENTRIES")):
            raise ValueError
        columns = lines[data_header].split()[2:]
        rows = lines[data_header + 1 : data_header + 1 + n]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected LAMMPS dump structure in {path}") from exc
    if len(rows) != n:
        raise ValueError(f"Expected {n} rows in final frame of {path}, found {len(rows)}")
    data = np.asarray([[float(value) for value in row.split()] for row in rows], dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"Column mismatch in {path}: header has {len(columns)}, data has {data.shape}")
    bounds_array = np.asarray(bounds, dtype=float)
    box = Box(lo=bounds_array[:, 0], hi=bounds_array[:, 1])
    return columns, data, box


def read_particles(path: Path) -> tuple[list[Particle], Box]:
    columns, data, box = _read_last_lammps_frame(path)
    required = ("id", "type", "x", "y", "z", "radius", "mass")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Missing particle columns in {path}: {missing}; available={columns}")
    col = {name: columns.index(name) for name in required}
    particles = [
        Particle(
            atom_id=int(row[col["id"]]),
            atom_type=int(row[col["type"]]),
            x=float(row[col["x"]]),
            y=float(row[col["y"]]),
            z=float(row[col["z"]]),
            radius=float(row[col["radius"]]),
            mass=float(row[col["mass"]]),
        )
        for row in data
    ]
    particles.sort(key=lambda p: p.atom_id)
    ids = [p.atom_id for p in particles]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate particle IDs found")
    return particles, box


def read_contact_ids(path: Path) -> list[tuple[int, int]]:
    columns, data, _ = _read_last_lammps_frame(path)
    # The generator writes index, atom-ID i, atom-ID j, type i, type j, ...
    # LAMMPS preserves the compute names in the dump header.  Prefer positions
    # 1 and 2 because those are unambiguous for this project output.
    if data.shape[1] < 3:
        raise ValueError(f"Contact dump {path} has fewer than three columns")
    pairs: set[tuple[int, int]] = set()
    for row in data:
        i, j = int(row[1]), int(row[2])
        if i == j:
            continue
        pairs.add((min(i, j), max(i, j)))
    if not pairs:
        raise ValueError(f"No particle contacts found in {path}; columns={columns}")
    return sorted(pairs)


def minimum_image_vector(a: np.ndarray, b: np.ndarray, box: Box) -> np.ndarray:
    """b-a with periodic minimum image in x/y and fixed z."""
    delta = b - a
    lengths = box.lengths
    for axis in (0, 1):
        delta[axis] -= lengths[axis] * np.rint(delta[axis] / lengths[axis])
    return delta


def harmonic_contact_conductance(k_i: float, k_j: float, a: float) -> float:
    if min(k_i, k_j, a) <= 0.0:
        return 0.0
    # Two circular-contact spreading resistances in series: 1/(4*k*a) each.
    return 4.0 * a * k_i * k_j / (k_i + k_j)


def wall_contact_conductance(k_particle: float, k_wall: float, a: float) -> float:
    if min(k_particle, a) <= 0.0:
        return 0.0
    if math.isinf(k_wall):
        return 4.0 * a * k_particle
    return harmonic_contact_conductance(k_particle, k_wall, a)


def build_contact_edges(
    particles: list[Particle],
    box: Box,
    contact_pairs: Iterable[tuple[int, int]],
    conductivity: np.ndarray,
    overlap_floor: float,
) -> tuple[list[Edge], set[tuple[int, int]]]:
    id_to_index = {p.atom_id: idx for idx, p in enumerate(particles)}
    xyz = np.asarray([(p.x, p.y, p.z) for p in particles])
    radii = np.asarray([p.radius for p in particles])
    edges: list[Edge] = []
    index_pairs: set[tuple[int, int]] = set()
    for atom_i, atom_j in contact_pairs:
        if atom_i not in id_to_index or atom_j not in id_to_index:
            raise ValueError(f"Contact ({atom_i}, {atom_j}) refers to a missing particle")
        i, j = id_to_index[atom_i], id_to_index[atom_j]
        if i > j:
            i, j = j, i
        vector = minimum_image_vector(xyz[i], xyz[j], box)
        distance = float(np.linalg.norm(vector))
        overlap = max(float(radii[i] + radii[j] - distance), 0.0)
        effective_radius = radii[i] * radii[j] / (radii[i] + radii[j])
        a = math.sqrt(effective_radius * max(overlap, overlap_floor))
        g_contact = harmonic_contact_conductance(conductivity[i], conductivity[j], a)
        edges.append(Edge(i, j, "contact", distance, -overlap, overlap, a, g_contact, 0.0))
        index_pairs.add((i, j))
    return edges, index_pairs


def build_gap_edges(
    particles: list[Particle],
    box: Box,
    contact_index_pairs: set[tuple[int, int]],
    fluid_k: float,
    gap_cutoff: float,
    gap_regularization: float,
) -> list[Edge]:
    """Construct non-contact stagnant-fluid proximity edges.

    A local lens approximation is used:
        G_gap = k_f A_eff / (s + s_0),
        A_eff = pi R* gap_cutoff.
    Here s is surface separation and s_0 regularizes unresolved roughness.
    The result must be subjected to gap-cutoff and s_0 sensitivity tests.
    """
    if fluid_k <= 0.0 or gap_cutoff <= 0.0:
        return []
    xyz = np.asarray([(p.x, p.y, p.z) for p in particles])
    radii = np.asarray([p.radius for p in particles])
    lengths = box.lengths
    shifted = xyz - box.lo
    # cKDTree supports a fully periodic box.  Make z artificially periodic with
    # a large padded period so no top-bottom pair is introduced.
    z_span = lengths[2] + 4.0 * (2.0 * radii.max() + gap_cutoff)
    tree_box = np.asarray([lengths[0], lengths[1], z_span])
    tree_points = shifted.copy()
    tree_points[:, 2] += 0.5 * (z_span - lengths[2])
    search_radius = float(2.0 * radii.max() + gap_cutoff)
    candidate_pairs = cKDTree(tree_points, boxsize=tree_box).query_pairs(search_radius)
    edges: list[Edge] = []
    for raw_i, raw_j in candidate_pairs:
        i, j = min(raw_i, raw_j), max(raw_i, raw_j)
        if (i, j) in contact_index_pairs:
            continue
        vector = minimum_image_vector(xyz[i], xyz[j], box)
        distance = float(np.linalg.norm(vector))
        surface_gap = distance - radii[i] - radii[j]
        if surface_gap <= 0.0 or surface_gap > gap_cutoff:
            continue
        effective_radius = radii[i] * radii[j] / (radii[i] + radii[j])
        area = math.pi * effective_radius * gap_cutoff
        g_gap = fluid_k * area / (surface_gap + gap_regularization)
        edges.append(Edge(i, j, "fluid_gap", distance, surface_gap, 0.0, 0.0, 0.0, g_gap))
    return edges


def build_wall_links(
    particles: list[Particle],
    box: Box,
    conductivity: np.ndarray,
    wall_k: float,
    wall_tolerance: float,
    overlap_floor: float,
    fluid_k: float,
    wall_gap_cutoff: float,
    gap_regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(particles)
    g_hot = np.zeros(n)
    g_cold = np.zeros(n)
    hot_mode = np.zeros(n, dtype=np.int8)  # 0 none, 1 contact, 2 gap
    cold_mode = np.zeros(n, dtype=np.int8)
    z_lo, z_hi = box.lo[2], box.hi[2]
    for i, p in enumerate(particles):
        bottom_gap = p.z - p.radius - z_lo
        top_gap = z_hi - (p.z + p.radius)
        if bottom_gap <= wall_tolerance:
            overlap = max(-bottom_gap, overlap_floor)
            a = math.sqrt(p.radius * overlap)
            g_hot[i] = wall_contact_conductance(conductivity[i], wall_k, a)
            hot_mode[i] = 1
        elif fluid_k > 0.0 and bottom_gap <= wall_gap_cutoff:
            area = math.pi * p.radius * wall_gap_cutoff
            g_hot[i] = fluid_k * area / (bottom_gap + gap_regularization)
            hot_mode[i] = 2
        if top_gap <= wall_tolerance:
            overlap = max(-top_gap, overlap_floor)
            a = math.sqrt(p.radius * overlap)
            g_cold[i] = wall_contact_conductance(conductivity[i], wall_k, a)
            cold_mode[i] = 1
        elif fluid_k > 0.0 and top_gap <= wall_gap_cutoff:
            area = math.pi * p.radius * wall_gap_cutoff
            g_cold[i] = fluid_k * area / (top_gap + gap_regularization)
            cold_mode[i] = 2
    return g_hot, g_cold, hot_mode, cold_mode


def solve_network(
    n: int,
    edges: list[Edge],
    g_hot: np.ndarray,
    g_cold: np.ndarray,
    t_hot: float,
    t_cold: float,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, csr_matrix]:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diagonal = g_hot + g_cold
    for edge in edges:
        g = edge.conductance
        if not np.isfinite(g) or g <= 0.0:
            raise ValueError(f"Non-positive or non-finite edge conductance: {edge}")
        i, j = edge.i, edge.j
        diagonal[i] += g
        diagonal[j] += g
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((-g, -g))
    rows.extend(range(n))
    cols.extend(range(n))
    vals.extend(diagonal.tolist())
    full_matrix = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    full_rhs = g_hot * t_hot + g_cold * t_cold
    matrix = full_matrix[active_mask][:, active_mask]
    rhs = full_rhs[active_mask]
    solved_temperature = np.asarray(spsolve(matrix, rhs), dtype=float)
    if not np.all(np.isfinite(solved_temperature)):
        raise RuntimeError("Thermal solve produced non-finite temperatures; graph may be disconnected from a boundary")
    temperature = np.full(n, np.nan)
    temperature[active_mask] = solved_temperature
    q_hot_particles = np.zeros(n)
    q_cold_particles = np.zeros(n)
    q_hot_particles[active_mask] = g_hot[active_mask] * (t_hot - solved_temperature)
    q_cold_particles[active_mask] = g_cold[active_mask] * (solved_temperature - t_cold)
    q_hot = float(q_hot_particles.sum())
    q_cold = float(q_cold_particles.sum())
    denominator = max(abs(q_hot), abs(q_cold), np.finfo(float).tiny)
    balance_error = abs(q_hot - q_cold) / denominator
    return temperature, q_hot_particles, q_hot, q_cold, balance_error, matrix


def thermally_anchored_mask(
    n: int, edges: list[Edge], g_hot: np.ndarray, g_cold: np.ndarray
) -> np.ndarray:
    """Select components connected to at least one prescribed-temperature wall."""
    rows, cols = [], []
    for edge in edges:
        rows.extend((edge.i, edge.j))
        cols.extend((edge.j, edge.i))
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    component_count, labels = connected_components(adjacency, directed=False)
    boundary_nodes = np.flatnonzero((g_hot > 0.0) | (g_cold > 0.0))
    anchored_components = np.zeros(component_count, dtype=bool)
    anchored_components[labels[boundary_nodes]] = True
    return anchored_components[labels]


def graph_diagnostics(n: int, edges: list[Edge], g_hot: np.ndarray, g_cold: np.ndarray) -> dict:
    rows, cols = [], []
    for edge in edges:
        rows.extend((edge.i, edge.j))
        cols.extend((edge.j, edge.i))
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    component_count, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels, minlength=component_count)
    hot_components = set(labels[np.flatnonzero(g_hot > 0.0)].tolist())
    cold_components = set(labels[np.flatnonzero(g_cold > 0.0)].tolist())
    spanning_components = sorted(hot_components & cold_components)
    anchored_components = hot_components | cold_components
    unanchored_components = sorted(set(range(component_count)) - anchored_components)
    return {
        "component_count": int(component_count),
        "largest_component_size": int(component_sizes.max()),
        "spanning_component_count": len(spanning_components),
        "spanning_component_sizes": [int(component_sizes[c]) for c in spanning_components],
        "unanchored_component_count": len(unanchored_components),
        "unanchored_component_sizes": [int(component_sizes[c]) for c in unanchored_components],
    }


def write_csv_outputs(
    output_dir: Path,
    particles: list[Particle],
    conductivity: np.ndarray,
    temperature: np.ndarray,
    g_hot: np.ndarray,
    g_cold: np.ndarray,
    hot_mode: np.ndarray,
    cold_mode: np.ndarray,
    edges: list[Edge],
) -> None:
    mode_name = {0: "none", 1: "contact", 2: "fluid_gap"}
    node_path = output_dir / "node_temperatures.csv"
    with node_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "id", "type", "x_m", "y_m", "z_m", "radius_m", "conductivity_W_mK",
            "temperature_K", "g_hot_W_K", "g_cold_W_K", "hot_link", "cold_link",
        ])
        for i, p in enumerate(particles):
            writer.writerow([
                p.atom_id, p.atom_type, p.x, p.y, p.z, p.radius, conductivity[i],
                temperature[i], g_hot[i], g_cold[i], mode_name[int(hot_mode[i])],
                mode_name[int(cold_mode[i])],
            ])
    edge_path = output_dir / "edge_heat_flux.csv"
    with edge_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "id_i", "id_j", "edge_kind", "distance_m", "surface_gap_m", "overlap_m",
            "contact_radius_m", "g_contact_W_K", "g_gap_W_K", "g_total_W_K",
            "T_i_K", "T_j_K", "heat_i_to_j_W",
        ])
        for edge in edges:
            q_ij = edge.conductance * (temperature[edge.i] - temperature[edge.j])
            writer.writerow([
                particles[edge.i].atom_id, particles[edge.j].atom_id, edge.kind,
                edge.distance, edge.surface_gap, edge.overlap, edge.contact_radius,
                edge.g_contact, edge.g_gap, edge.conductance,
                temperature[edge.i], temperature[edge.j], q_ij,
            ])


def _xml_data_array(name: str, values: Iterable, vtk_type: str = "Float64", components: int = 1) -> str:
    attribute = f' NumberOfComponents="{components}"' if components > 1 else ""
    text = " ".join(str(value) for value in values)
    return f'<DataArray type="{vtk_type}" Name="{name}" format="ascii"{attribute}>{text}</DataArray>'


def write_vtp_outputs(
    output_dir: Path,
    particles: list[Particle],
    box: Box,
    conductivity: np.ndarray,
    temperature: np.ndarray,
    g_hot: np.ndarray,
    g_cold: np.ndarray,
    edges: list[Edge],
) -> None:
    n = len(particles)
    xyz = np.asarray([(p.x, p.y, p.z) for p in particles])
    point_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
        '<PolyData>',
        f'<Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">',
        '<PointData>',
        _xml_data_array("id", [p.atom_id for p in particles], "Int64"),
        _xml_data_array("type", [p.atom_type for p in particles], "Int32"),
        _xml_data_array("radius", [p.radius for p in particles]),
        _xml_data_array("diameter", [2.0 * p.radius for p in particles]),
        _xml_data_array("conductivity", conductivity),
        _xml_data_array("temperature", temperature),
        _xml_data_array("g_hot", g_hot),
        _xml_data_array("g_cold", g_cold),
        '</PointData><Points>',
        _xml_data_array("Points", xyz.ravel(), components=3),
        '</Points><Verts>',
        _xml_data_array("connectivity", range(n), "Int64"),
        _xml_data_array("offsets", range(1, n + 1), "Int64"),
        '</Verts></Piece></PolyData></VTKFile>',
    ]
    (output_dir / "thermal_nodes.vtp").write_text("\n".join(point_lines), encoding="utf-8")

    # Duplicate periodic endpoints using the minimum-image vector so ParaView
    # does not draw long lines across the entire cell.
    edge_points: list[np.ndarray] = []
    connectivity: list[int] = []
    offsets: list[int] = []
    heat_flux: list[float] = []
    for edge_index, edge in enumerate(edges):
        p_i = xyz[edge.i]
        vector = minimum_image_vector(p_i, xyz[edge.j], box)
        edge_points.extend((p_i, p_i + vector))
        connectivity.extend((2 * edge_index, 2 * edge_index + 1))
        offsets.append(2 * edge_index + 2)
        heat_flux.append(edge.conductance * (temperature[edge.i] - temperature[edge.j]))
    kind_code = [0 if edge.kind == "contact" else 1 for edge in edges]
    edge_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
        '<PolyData>',
        f'<Piece NumberOfPoints="{2 * len(edges)}" NumberOfVerts="0" NumberOfLines="{len(edges)}" NumberOfStrips="0" NumberOfPolys="0">',
        '<CellData>',
        _xml_data_array("edge_kind", kind_code, "Int32"),
        _xml_data_array("conductance", [edge.conductance for edge in edges]),
        _xml_data_array("g_contact", [edge.g_contact for edge in edges]),
        _xml_data_array("g_gap", [edge.g_gap for edge in edges]),
        _xml_data_array("heat_i_to_j", heat_flux),
        '</CellData><Points>',
        _xml_data_array("Points", np.asarray(edge_points).ravel(), components=3),
        '</Points><Lines>',
        _xml_data_array("connectivity", connectivity, "Int64"),
        _xml_data_array("offsets", offsets, "Int64"),
        '</Lines></Piece></PolyData></VTKFile>',
    ]
    (output_dir / "thermal_edges.vtp").write_text("\n".join(edge_lines), encoding="utf-8")


def read_high_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    ids: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            ids.add(int(line.split()[0]))
    return ids


def run(args: argparse.Namespace) -> dict:
    case_dir = args.case.resolve()
    particle_path = case_dir / args.particles
    contact_path = case_dir / args.contacts
    output_dir = case_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    particles, box = read_particles(particle_path)
    contact_pairs = read_contact_ids(contact_path)
    n = len(particles)
    high_ids = read_high_ids(args.high_ids)
    known_ids = {p.atom_id for p in particles}
    unknown = high_ids - known_ids
    if unknown:
        raise ValueError(f"High-conductivity ID file contains unknown IDs: {sorted(unknown)[:10]}")
    conductivity = np.asarray([
        args.k_high if p.atom_id in high_ids else args.k_low for p in particles
    ], dtype=float)

    radii = np.asarray([p.radius for p in particles])
    d_ref = float(2.0 * np.median(radii))
    overlap_floor = args.overlap_floor_ratio * d_ref
    gap_cutoff = args.gap_cutoff_ratio * d_ref
    gap_regularization = args.gap_regularization_ratio * d_ref
    wall_tolerance = args.wall_tolerance_ratio * d_ref

    contact_edges, contact_index_pairs = build_contact_edges(
        particles, box, contact_pairs, conductivity, overlap_floor
    )
    gap_edges = build_gap_edges(
        particles, box, contact_index_pairs, args.k_fluid, gap_cutoff,
        gap_regularization,
    )
    edges = contact_edges + gap_edges
    g_hot, g_cold, hot_mode, cold_mode = build_wall_links(
        particles, box, conductivity, args.k_wall, wall_tolerance,
        overlap_floor, args.k_fluid, gap_cutoff, gap_regularization,
    )
    if not np.any(g_hot > 0.0):
        raise RuntimeError("No particle is thermally linked to the hot z wall. Check wall geometry/tolerance.")
    if not np.any(g_cold > 0.0):
        raise RuntimeError("No particle is thermally linked to the cold z wall. Check wall geometry/tolerance.")

    diagnostics = graph_diagnostics(n, edges, g_hot, g_cold)
    if diagnostics["spanning_component_count"] == 0:
        raise RuntimeError("No thermal-network component connects both z boundaries")
    active_mask = thermally_anchored_mask(n, edges, g_hot, g_cold)
    temperature, q_hot_particles, q_hot, q_cold, balance_error, matrix = solve_network(
        n, edges, g_hot, g_cold, args.t_hot, args.t_cold, active_mask
    )
    delta_t = args.t_hot - args.t_cold
    if delta_t <= 0.0:
        raise ValueError("t-hot must be greater than t-cold")
    k_eff = q_hot * box.height / (box.area_z * delta_t)
    solved_min = float(np.nanmin(temperature))
    solved_max = float(np.nanmax(temperature))
    if solved_min < args.t_cold - 1.0e-9 or solved_max > args.t_hot + 1.0e-9:
        raise RuntimeError("Solved temperatures violate the imposed boundary bounds")

    write_csv_outputs(
        output_dir, particles, conductivity, temperature, g_hot, g_cold,
        hot_mode, cold_mode, edges,
    )
    write_vtp_outputs(output_dir, particles, box, conductivity, temperature, g_hot, g_cold, edges)
    summary = {
        "thermal_model": "circular_constriction_4a_v2",
        "case_directory": str(case_dir),
        "particles": n,
        "thermally_solved_particles": int(np.count_nonzero(active_mask)),
        "thermally_unanchored_particles": int(np.count_nonzero(~active_mask)),
        "high_conductivity_particles": len(high_ids),
        "contact_edges": len(contact_edges),
        "fluid_gap_edges": len(gap_edges),
        "hot_wall_links": int(np.count_nonzero(g_hot)),
        "cold_wall_links": int(np.count_nonzero(g_cold)),
        "hot_wall_contact_links": int(np.count_nonzero(hot_mode == 1)),
        "cold_wall_contact_links": int(np.count_nonzero(cold_mode == 1)),
        "hot_wall_fluid_links": int(np.count_nonzero(hot_mode == 2)),
        "cold_wall_fluid_links": int(np.count_nonzero(cold_mode == 2)),
        "box_height_m": box.height,
        "cross_section_area_m2": box.area_z,
        "reference_diameter_m": d_ref,
        "temperature_hot_K": args.t_hot,
        "temperature_cold_K": args.t_cold,
        "conductivity_low_W_mK": args.k_low,
        "conductivity_high_W_mK": args.k_high,
        "conductivity_fluid_W_mK": args.k_fluid,
        "heat_rate_hot_W": q_hot,
        "heat_rate_cold_W": q_cold,
        "relative_energy_balance_error": balance_error,
        "effective_conductivity_W_mK": float(k_eff),
        "temperature_min_K": solved_min,
        "temperature_max_K": solved_max,
        "matrix_size": list(matrix.shape),
        "matrix_nonzeros": int(matrix.nnz),
        "model_parameters": {
            "overlap_floor_ratio": args.overlap_floor_ratio,
            "gap_cutoff_ratio": args.gap_cutoff_ratio,
            "gap_regularization_ratio": args.gap_regularization_ratio,
            "wall_tolerance_ratio": args.wall_tolerance_ratio,
            "wall_conductivity_W_mK": "infinite" if math.isinf(args.k_wall) else args.k_wall,
        },
        "graph_diagnostics": diagnostics,
    }
    (output_dir / "thermal_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="One packing_seed_* case directory")
    parser.add_argument("--particles", default="particles_final.dump")
    parser.add_argument("--contacts", default="contacts_final.dump")
    parser.add_argument("--output", default="thermal_results")
    parser.add_argument("--high-ids", type=Path, default=None,
                        help="Optional text file with one high-conductivity particle ID per line")
    parser.add_argument("--k-low", type=float, default=1.0, help="Low particle conductivity [W/(m K)]")
    parser.add_argument("--k-high", type=float, default=10.0, help="High particle conductivity [W/(m K)]")
    parser.add_argument("--k-fluid", type=float, default=0.0,
                        help="Stagnant-fluid conductivity [W/(m K)]; 0 disables fluid-gap edges")
    parser.add_argument("--k-wall", type=float, default=math.inf,
                        help="Wall conductivity [W/(m K)]; default infinite")
    parser.add_argument("--t-hot", type=float, default=301.0, help="Bottom-wall temperature [K]")
    parser.add_argument("--t-cold", type=float, default=300.0, help="Top-wall temperature [K]")
    parser.add_argument("--overlap-floor-ratio", type=float, default=1.0e-8,
                        help="Minimum overlap/contact-radius regularization divided by median diameter")
    parser.add_argument("--gap-cutoff-ratio", type=float, default=0.10,
                        help="Maximum fluid gap divided by median diameter")
    parser.add_argument("--gap-regularization-ratio", type=float, default=1.0e-3,
                        help="Fluid-gap roughness regularization divided by median diameter")
    parser.add_argument("--wall-tolerance-ratio", type=float, default=1.0e-8,
                        help="Geometric wall-contact tolerance divided by median diameter")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:  # concise CLI failure while preserving traceback on request
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

