#!/usr/bin/env python3
"""Explain why conductive-particle allocations change packed-bed conductivity.

The script reuses the contact-only thermal model and compares three controlled
allocations on one fixed DEM packing:

1. a stratified random realization closest to the recorded random mean;
2. the heat-path ranking allocation; and
3. the best fixed-budget single-swap allocation.

It writes a compact comparison CSV, per-particle and per-contact CSV files, and
ParaView-ready VTP files.  No DEM simulation is repeated.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np

import run_allocation_pilot_stratified_v2 as pilot


def read_ids(path: Path) -> frozenset[int]:
    ids = frozenset(
        int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not ids:
        raise ValueError(f"No particle IDs found in {path}")
    return ids


def components(nodes: set[int], adjacency: dict[int, set[int]]) -> list[set[int]]:
    remaining = set(nodes)
    result: list[set[int]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component = {root}
        remaining.remove(root)
        while stack:
            current = stack.pop()
            for neighbour in adjacency.get(current, set()):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        result.append(component)
    return sorted(result, key=lambda item: (-len(item), min(item)))


def shortest_resistance_path(
    n: int, edges, g_hot: np.ndarray, g_cold: np.ndarray
) -> tuple[float, list[int]]:
    """Return minimum series resistance and its particle-index path."""
    source, sink = n, n + 1
    graph: list[list[tuple[int, float]]] = [[] for _ in range(n + 2)]
    for edge in edges:
        if edge.conductance > 0.0:
            resistance = 1.0 / edge.conductance
            graph[edge.i].append((edge.j, resistance))
            graph[edge.j].append((edge.i, resistance))
    for i, conductance in enumerate(g_hot):
        if conductance > 0.0:
            resistance = 1.0 / conductance
            graph[source].append((i, resistance))
            graph[i].append((source, resistance))
    for i, conductance in enumerate(g_cold):
        if conductance > 0.0:
            resistance = 1.0 / conductance
            graph[i].append((sink, resistance))
            graph[sink].append((i, resistance))

    distance = [math.inf] * (n + 2)
    previous = [-1] * (n + 2)
    distance[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        if node == sink:
            break
        for neighbour, resistance in graph[node]:
            trial = current_distance + resistance
            if trial < distance[neighbour]:
                distance[neighbour] = trial
                previous[neighbour] = node
                heapq.heappush(queue, (trial, neighbour))
    if not math.isfinite(distance[sink]):
        return math.inf, []
    path = []
    node = sink
    while node != -1:
        path.append(node)
        node = previous[node]
    path.reverse()
    return distance[sink], [node for node in path if node < n]


def xml_array(name: str, values: Iterable, dtype: str = "Float64",
              components: int = 1) -> str:
    text = " ".join(str(value) for value in values)
    return (
        f'<DataArray type="{dtype}" Name="{escape(name)}" '
        f'NumberOfComponents="{components}" format="ascii">{text}</DataArray>'
    )


def write_particle_vtp(path: Path, evaluator, high_ids: set[int], result,
                       node_throughput: np.ndarray, degree: np.ndarray,
                       hot_link: np.ndarray, cold_link: np.ndarray,
                       shortest_path: set[int]) -> None:
    particles = evaluator.particles
    xyz = [coordinate for p in particles for coordinate in (p.x, p.y, p.z)]
    ids = [p.atom_id for p in particles]
    content = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
        '<PolyData>',
        f'<Piece NumberOfPoints="{len(particles)}" NumberOfVerts="{len(particles)}" '
        'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">',
        '<PointData>',
        xml_array("atom_id", ids, "Int64"),
        xml_array("atom_type", [p.atom_type for p in particles], "Int32"),
        xml_array("radius_m", [p.radius for p in particles]),
        xml_array("high_conductivity", [int(i in high_ids) for i in ids], "Int32"),
        xml_array("conductivity_W_mK", [
            evaluator.k_high if i in high_ids else evaluator.k_low for i in ids
        ]),
        xml_array("temperature_K", result.temperature),
        xml_array("contact_degree", degree, "Int32"),
        xml_array("absolute_heat_throughput_W", node_throughput),
        xml_array("hot_wall_link", [int(v > 0.0) for v in hot_link], "Int32"),
        xml_array("cold_wall_link", [int(v > 0.0) for v in cold_link], "Int32"),
        xml_array("on_minimum_resistance_path", [int(i in shortest_path)
                  for i in range(len(particles))], "Int32"),
        '</PointData>',
        '<Points>', xml_array("Points", xyz, components=3), '</Points>',
        '<Verts>',
        xml_array("connectivity", range(len(particles)), "Int64"),
        xml_array("offsets", range(1, len(particles) + 1), "Int64"),
        '</Verts>', '</Piece>', '</PolyData>', '</VTKFile>',
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def write_contact_vtp(path: Path, evaluator, high_ids: set[int], result) -> None:
    xyz = np.asarray([(p.x, p.y, p.z) for p in evaluator.particles], dtype=float)
    ids = np.asarray([p.atom_id for p in evaluator.particles], dtype=int)
    points: list[float] = []
    heat_rates, conductances, high_counts, vertical_alignment = [], [], [], []
    atom_i_values, atom_j_values = [], []
    for edge in result.edges:
        start = xyz[edge.i]
        vector = evaluator.solver.minimum_image_vector(
            start, xyz[edge.j], evaluator.box
        )
        end = start + vector
        points.extend(start.tolist())
        points.extend(end.tolist())
        heat_rates.append(edge.conductance * (
            result.temperature[edge.i] - result.temperature[edge.j]
        ))
        conductances.append(edge.conductance)
        atom_i_values.append(int(ids[edge.i]))
        atom_j_values.append(int(ids[edge.j]))
        high_counts.append(int(ids[edge.i] in high_ids) + int(ids[edge.j] in high_ids))
        vertical_alignment.append(abs(vector[2]) / edge.distance if edge.distance > 0 else 0.0)
    n_edges = len(result.edges)
    content = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
        '<PolyData>',
        f'<Piece NumberOfPoints="{2*n_edges}" NumberOfVerts="0" '
        f'NumberOfLines="{n_edges}" NumberOfStrips="0" NumberOfPolys="0">',
        '<CellData>',
        xml_array("atom_id_i", atom_i_values, "Int64"),
        xml_array("atom_id_j", atom_j_values, "Int64"),
        xml_array("conductance_W_K", conductances),
        xml_array("signed_heat_rate_i_to_j_W", heat_rates),
        xml_array("absolute_heat_rate_W", np.abs(heat_rates)),
        xml_array("high_endpoint_count", high_counts, "Int32"),
        xml_array("vertical_alignment", vertical_alignment),
        '</CellData>',
        '<Points>', xml_array("Points", points, components=3), '</Points>',
        '<Lines>',
        xml_array("connectivity", range(2*n_edges), "Int64"),
        xml_array("offsets", range(2, 2*n_edges + 1, 2), "Int64"),
        '</Lines>', '</Piece>', '</PolyData>', '</VTKFile>',
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def analyse(method: str, evaluator, high_ids: frozenset[int], output: Path) -> dict:
    high = set(high_ids)
    pilot.validate_quota(evaluator, high, analyse.quotas)
    result = evaluator.evaluate(high)
    n = len(evaluator.particles)
    ids = np.asarray([p.atom_id for p in evaluator.particles], dtype=int)
    xyz = np.asarray([(p.x, p.y, p.z) for p in evaluator.particles], dtype=float)
    radii = np.asarray([p.radius for p in evaluator.particles], dtype=float)
    degree = np.zeros(n, dtype=int)
    node_throughput = np.zeros(n, dtype=float)
    adjacency: dict[int, set[int]] = {i: set() for i in high}
    total_edge_heat = high_incident_heat = high_high_heat = 0.0
    high_high_edges = 0
    high_high_alignments, high_high_weighted = [], []
    contact_rows = []

    for edge in result.edges:
        atom_i, atom_j = int(ids[edge.i]), int(ids[edge.j])
        degree[edge.i] += 1
        degree[edge.j] += 1
        vector = evaluator.solver.minimum_image_vector(xyz[edge.i], xyz[edge.j], evaluator.box)
        alignment = abs(vector[2]) / edge.distance if edge.distance > 0 else 0.0
        signed_heat = edge.conductance * (
            result.temperature[edge.i] - result.temperature[edge.j]
        )
        absolute_heat = abs(signed_heat)
        node_throughput[edge.i] += absolute_heat
        node_throughput[edge.j] += absolute_heat
        total_edge_heat += absolute_heat
        high_count = int(atom_i in high) + int(atom_j in high)
        if high_count:
            high_incident_heat += absolute_heat
        if high_count == 2:
            high_high_edges += 1
            high_high_heat += absolute_heat
            high_high_alignments.append(alignment)
            high_high_weighted.append((alignment, absolute_heat))
            adjacency[atom_i].add(atom_j)
            adjacency[atom_j].add(atom_i)
        contact_rows.append({
            "atom_id_i": atom_i, "atom_id_j": atom_j,
            "high_endpoint_count": high_count,
            "conductance_W_K": edge.conductance,
            "signed_heat_rate_i_to_j_W": signed_heat,
            "absolute_heat_rate_W": absolute_heat,
            "vertical_alignment": alignment,
        })

    g_hot, g_cold, _, _ = evaluator.solver.build_wall_links(
        evaluator.particles, evaluator.box,
        np.asarray([evaluator.k_high if int(i) in high else evaluator.k_low for i in ids]),
        math.inf, evaluator.wall_tolerance, evaluator.overlap_floor, 0.0, 0.0, 0.0,
    )
    high_hot = {int(ids[i]) for i in np.flatnonzero(g_hot > 0.0) if int(ids[i]) in high}
    high_cold = {int(ids[i]) for i in np.flatnonzero(g_cold > 0.0) if int(ids[i]) in high}
    high_components = components(high, adjacency)
    largest = high_components[0]
    spanning = [c for c in high_components if c & high_hot and c & high_cold]
    largest_indices = [evaluator.id_to_index[i] for i in largest]
    vertical_extent = (
        max(xyz[i, 2] + radii[i] for i in largest_indices)
        - min(xyz[i, 2] - radii[i] for i in largest_indices)
    )
    resistance, shortest_path = shortest_resistance_path(n, result.edges, g_hot, g_cold)
    path_high_count = sum(int(ids[i] in high) for i in shortest_path)
    path_ids = [int(ids[i]) for i in shortest_path]

    particle_rows = []
    shortest_set = set(shortest_path)
    for i, particle in enumerate(evaluator.particles):
        particle_rows.append({
            "atom_id": particle.atom_id,
            "atom_type": particle.atom_type,
            "x_m": particle.x, "y_m": particle.y, "z_m": particle.z,
            "radius_m": particle.radius,
            "high_conductivity": int(particle.atom_id in high),
            "temperature_K": result.temperature[i],
            "contact_degree": degree[i],
            "absolute_heat_throughput_W": node_throughput[i],
            "hot_wall_link": int(g_hot[i] > 0.0),
            "cold_wall_link": int(g_cold[i] > 0.0),
            "on_minimum_resistance_path": int(i in shortest_set),
        })

    for filename, rows in [
        (f"particles_{method}.csv", particle_rows),
        (f"contacts_{method}.csv", contact_rows),
    ]:
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_particle_vtp(
        output / f"particles_{method}.vtp", evaluator, high, result,
        node_throughput, degree, g_hot, g_cold, shortest_set,
    )
    write_contact_vtp(output / f"contacts_{method}.vtp", evaluator, high, result)

    heat_weight_sum = sum(weight for _, weight in high_high_weighted)
    return {
        "method": method,
        "effective_conductivity_W_mK": result.k_eff,
        "high_count": len(high),
        "high_high_contact_count": high_high_edges,
        "high_cluster_count": len(high_components),
        "largest_high_cluster_size": len(largest),
        "largest_high_cluster_fraction": len(largest) / len(high),
        "spanning_high_cluster_count": len(spanning),
        "largest_high_cluster_vertical_extent_ratio": vertical_extent / evaluator.box.height,
        "high_particles_linked_to_hot_wall": len(high_hot),
        "high_particles_linked_to_cold_wall": len(high_cold),
        "high_incident_contact_heat_fraction": (
            high_incident_heat / total_edge_heat if total_edge_heat else 0.0
        ),
        "high_high_contact_heat_fraction": (
            high_high_heat / total_edge_heat if total_edge_heat else 0.0
        ),
        "mean_vertical_alignment_high_high": (
            float(np.mean(high_high_alignments)) if high_high_alignments else 0.0
        ),
        "heat_weighted_vertical_alignment_high_high": (
            sum(a*w for a, w in high_high_weighted) / heat_weight_sum
            if heat_weight_sum else 0.0
        ),
        "minimum_series_path_resistance_K_W": resistance,
        "minimum_series_path_conductance_W_K": 1.0 / resistance,
        "minimum_resistance_path_particle_count": len(shortest_path),
        "minimum_resistance_path_high_count": path_high_count,
        "minimum_resistance_path_high_fraction": path_high_count / len(shortest_path),
        "minimum_resistance_path_atom_ids": " ".join(map(str, path_ids)),
        "relative_energy_balance_error": result.balance_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--allocation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("mechanism_analysis"))
    parser.add_argument("--solver", type=Path,
                        default=Path(__file__).with_name("solve_packing_heat_transfer.py"))
    args = parser.parse_args()

    allocation_dir = args.allocation_dir.resolve()
    summary_path = allocation_dir / "allocation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    quotas = {int(key): int(value) for key, value in summary["type_quotas"].items()}
    analyse.quotas = quotas
    solver = pilot.load_solver(args.solver.resolve())
    evaluator = pilot.ContactOnlyEvaluator(
        solver, args.case, summary["k_low_W_mK"], summary["k_high_W_mK"],
        301.0, 300.0, 1.0e-8, 1.0e-8,
    )

    rng = np.random.default_rng(int(summary["seed"]))
    random_allocations, random_results = [], []
    for _ in range(int(summary["random_samples"])):
        ids = pilot.random_stratified(evaluator, quotas, rng)
        random_allocations.append(ids)
        random_results.append(evaluator.evaluate(ids))
    target = float(summary["random_mean_k_eff_W_mK"])
    random_index = min(
        range(len(random_results)), key=lambda i: abs(random_results[i].k_eff - target)
    )
    representative_random = random_allocations[random_index]

    allocations = {
        "random_near_mean": representative_random,
        "heat_path": read_ids(allocation_dir / "high_ids_heat_path.txt"),
        "swap_optimized": read_ids(allocation_dir / "high_ids_optimized.txt"),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [analyse(method, evaluator, ids, output)
            for method, ids in allocations.items()]
    with (output / "mechanism_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "case": str(args.case.resolve()),
        "allocation_directory": str(allocation_dir),
        "representative_random_replicate": random_index + 1,
        "representative_random_k_eff_W_mK": random_results[random_index].k_eff,
        "recorded_random_mean_k_eff_W_mK": target,
        "absolute_difference_from_random_mean_W_mK": abs(
            random_results[random_index].k_eff - target
        ),
        "methods": list(allocations),
        "notes": {
            "vertical_alignment": "absolute contact direction cosine |dz|/distance",
            "minimum_resistance_path": "shortest series-resistance path; descriptive, not total network resistance",
            "heat_fraction_denominator": "sum of absolute heat rates over all particle-particle contacts",
        },
    }
    (output / "mechanism_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metadata": metadata, "comparison": rows}, indent=2))


if __name__ == "__main__":
    main()
