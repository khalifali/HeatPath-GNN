#!/usr/bin/env python3
"""Build graph files for physics-guided conductive-particle allocation.

Each DEM packing becomes one compressed NumPy ``.npz`` file.  Input features
are computed only from the fixed geometry and the homogeneous low-conductivity
thermal solution.  Targets are derived from the final allocations of multiple
fixed-budget optimizer restarts.

The builder deliberately does not perform dataset-wide standardization.  Such
statistics must be fitted on the training packings inside each cross-validation
fold to avoid information leakage from the held-out packing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

import run_allocation_pilot_stratified_v2 as pilot


NODE_FEATURES = [
    "diameter_over_dref",
    "z_normalized",
    "contact_degree",
    "hot_wall_contact",
    "cold_wall_contact",
    "thermally_anchored",
    "baseline_temperature_normalized",
    "baseline_heat_score_normalized",
]

EDGE_FEATURES = [
    "contact_radius_over_dref",
    "dx_over_dref",
    "dy_over_dref",
    "dz_over_dref",
    "absolute_vertical_alignment",
    "baseline_absolute_heat_rate_normalized",
]


def read_ids(path: Path) -> frozenset[int]:
    ids = frozenset(
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not ids:
        raise ValueError(f"No particle IDs found in {path}")
    return ids


def case_label(case: Path) -> tuple[str, int | None]:
    match = re.fullmatch(r"packing_seed_(\d+)", case.name)
    return case.name, int(match.group(1)) if match else None


def resolve_allocation_dir(root: Path, case: Path, template: str) -> Path:
    _, seed = case_label(case)
    relative = template.format(case=case.name, seed="" if seed is None else seed)
    path = Path(relative)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def discover_cases(root: Path, pattern: str, explicit: list[Path]) -> list[Path]:
    cases = [path.resolve() for path in explicit]
    if not cases:
        cases = sorted(
            (path.resolve() for path in root.glob(pattern) if path.is_dir()),
            key=lambda path: case_label(path)[1] or path.name,
        )
    if not cases:
        raise FileNotFoundError(
            f"No packing cases found below {root} with pattern {pattern!r}"
        )
    return cases


def teacher_targets(evaluator, allocation_dir: Path, quotas: dict[int, int],
                    expected_restarts: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    restart_paths = sorted(
        allocation_dir.glob("high_ids_optimized_restart_*.txt"),
        key=lambda path: int(re.search(r"_(\d+)\.txt$", path.name).group(1)),
    )
    if len(restart_paths) != expected_restarts:
        raise ValueError(
            f"Expected {expected_restarts} restart ID files in {allocation_dir}, "
            f"found {len(restart_paths)}"
        )
    counts = np.zeros(len(evaluator.ids), dtype=float)
    for path in restart_paths:
        ids = read_ids(path)
        pilot.validate_quota(evaluator, ids, quotas)
        counts += np.isin(evaluator.ids, list(ids)).astype(float)
    soft = counts / len(restart_paths)

    best_ids = read_ids(allocation_dir / "high_ids_optimized.txt")
    pilot.validate_quota(evaluator, best_ids, quotas)
    best = np.isin(evaluator.ids, list(best_ids)).astype(np.float32)
    return soft.astype(np.float32), best, [path.name for path in restart_paths]


def build_graph(solver, case: Path, allocation_dir: Path) -> tuple[dict, dict]:
    summary_path = allocation_dir / "allocation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("contact_only", True):
        raise ValueError(f"Allocation in {allocation_dir} is not contact-only")
    quotas = {int(key): int(value) for key, value in summary["type_quotas"].items()}
    t_hot = float(summary.get("temperature_hot_K", 301.0))
    t_cold = float(summary.get("temperature_cold_K", 300.0))
    if t_hot <= t_cold:
        raise ValueError("Require temperature_hot_K > temperature_cold_K")

    evaluator = pilot.ContactOnlyEvaluator(
        solver,
        case,
        float(summary["k_low_W_mK"]),
        float(summary["k_high_W_mK"]),
        t_hot,
        t_cold,
        float(summary.get("overlap_floor_ratio", 1.0e-8)),
        float(summary.get("wall_tolerance_ratio", 1.0e-8)),
    )
    homogeneous = evaluator.evaluate(frozenset())
    n = len(evaluator.particles)
    xyz = np.asarray([(p.x, p.y, p.z) for p in evaluator.particles], dtype=float)
    diameter = 2.0 * evaluator.radii
    degree = pilot.particle_degrees(evaluator).astype(float)
    heat_score = pilot.heat_path_scores(evaluator, homogeneous)
    heat_scale = float(np.max(heat_score))
    heat_score_normalized = heat_score / heat_scale if heat_scale > 0.0 else heat_score

    conductivity = np.full(n, evaluator.k_low, dtype=float)
    g_hot, g_cold, _, _ = solver.build_wall_links(
        evaluator.particles,
        evaluator.box,
        conductivity,
        math.inf,
        evaluator.wall_tolerance,
        evaluator.overlap_floor,
        0.0,
        0.0,
        0.0,
    )
    anchored = solver.thermally_anchored_mask(
        n, homogeneous.edges, g_hot, g_cold
    )
    temperature_normalized = (
        homogeneous.temperature - t_cold
    ) / (t_hot - t_cold)
    temperature_normalized = np.where(
        np.isfinite(temperature_normalized), temperature_normalized, 0.0
    )
    z_normalized = (xyz[:, 2] - evaluator.box.lo[2]) / evaluator.box.height
    node_features = np.column_stack([
        diameter / evaluator.d_ref,
        z_normalized,
        degree,
        (g_hot > 0.0).astype(float),
        (g_cold > 0.0).astype(float),
        anchored.astype(float),
        temperature_normalized,
        heat_score_normalized,
    ]).astype(np.float32)

    directed_pairs: list[tuple[int, int]] = []
    directed_features: list[list[float]] = []
    baseline_edge_heat = []
    for edge in homogeneous.edges:
        temperature_i = homogeneous.temperature[edge.i]
        temperature_j = homogeneous.temperature[edge.j]
        # Contacts in a component connected to neither thermal wall have no
        # uniquely defined steady temperature.  The solver marks those nodes
        # with NaN and excludes them from the linear system.  Such a component
        # carries no boundary-to-boundary heat in the contact-only model, so its
        # baseline contact heat feature must be zero rather than NaN.
        if np.isfinite(temperature_i) and np.isfinite(temperature_j):
            absolute_heat = abs(
                edge.conductance * (temperature_i - temperature_j)
            )
        else:
            absolute_heat = 0.0
        baseline_edge_heat.append(float(absolute_heat))
    edge_heat_scale = max(baseline_edge_heat, default=0.0)

    for edge, absolute_heat in zip(homogeneous.edges, baseline_edge_heat):
        vector = solver.minimum_image_vector(xyz[edge.i], xyz[edge.j], evaluator.box)
        alignment = abs(vector[2]) / edge.distance if edge.distance > 0.0 else 0.0
        heat_normalized = absolute_heat / edge_heat_scale if edge_heat_scale > 0.0 else 0.0
        base = [
            edge.contact_radius / evaluator.d_ref,
            vector[0] / evaluator.d_ref,
            vector[1] / evaluator.d_ref,
            vector[2] / evaluator.d_ref,
            alignment,
            heat_normalized,
        ]
        reverse = [base[0], -base[1], -base[2], -base[3], base[4], base[5]]
        directed_pairs.extend([(edge.i, edge.j), (edge.j, edge.i)])
        directed_features.extend([base, reverse])

    edge_index = np.asarray(directed_pairs, dtype=np.int64).T
    edge_features = np.asarray(directed_features, dtype=np.float32)
    soft_target, best_target, restart_files = teacher_targets(
        evaluator,
        allocation_dir,
        quotas,
        int(summary["optimizer_restarts"]),
    )

    observed_soft_sums = {
        str(atom_type): float(soft_target[evaluator.types == atom_type].sum())
        for atom_type in sorted(quotas)
    }
    for atom_type, quota in quotas.items():
        if not np.isclose(observed_soft_sums[str(atom_type)], quota, atol=1.0e-6):
            raise RuntimeError(
                f"Soft target violates type {atom_type} quota: "
                f"{observed_soft_sums[str(atom_type)]} != {quota}"
            )

    name, seed = case_label(case)
    arrays = {
        "node_ids": evaluator.ids.astype(np.int64),
        "node_types": evaluator.types.astype(np.int64),
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "target_selection_frequency": soft_target,
        "target_best_selection": best_target,
        "box_lo_m": evaluator.box.lo.astype(np.float64),
        "box_hi_m": evaluator.box.hi.astype(np.float64),
    }
    for array_name in ("node_features", "edge_features"):
        if not np.isfinite(arrays[array_name]).all():
            bad = np.argwhere(~np.isfinite(arrays[array_name]))
            raise RuntimeError(
                f"Non-finite values remain in {array_name} for {name}; "
                f"first indices={bad[:5].tolist()}"
            )
    metadata = {
        "case": name,
        "seed": seed,
        "case_directory": str(case),
        "allocation_directory": str(allocation_dir),
        "particles": n,
        "undirected_contact_edges": len(homogeneous.edges),
        "directed_graph_edges": int(edge_index.shape[1]),
        "node_feature_names": NODE_FEATURES,
        "edge_feature_names": EDGE_FEATURES,
        "target_names": ["target_selection_frequency", "target_best_selection"],
        "type_quotas": {str(key): value for key, value in sorted(quotas.items())},
        "soft_target_sum_by_type": observed_soft_sums,
        "optimizer_restart_files": restart_files,
        "homogeneous_k_eff_W_mK": homogeneous.k_eff,
        "teacher_best_k_eff_W_mK": float(summary["optimized_best_k_eff_W_mK"]),
        "random_mean_k_eff_W_mK": float(summary["random_mean_k_eff_W_mK"]),
        "heat_path_k_eff_W_mK": float(summary["heat_path_k_eff_W_mK"]),
        "k_low_W_mK": evaluator.k_low,
        "k_high_W_mK": evaluator.k_high,
        "temperature_hot_K": t_hot,
        "temperature_cold_K": t_cold,
        "contact_only": True,
        "normalization_note": (
            "Only dimensionless geometry and within-graph baseline heat quantities "
            "are formed here. Fit any dataset-wide feature scaler on training "
            "packings only inside each cross-validation fold."
        ),
    }
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pattern", default="packing_seed_*")
    parser.add_argument(
        "--case", type=Path, action="append", default=[],
        help="Explicit case directory; repeat for multiple cases. Overrides --pattern.",
    )
    parser.add_argument(
        "--allocation-template",
        default="{case}/allocation_stratified_p3000",
        help="Path below root; supports {case} and {seed} placeholders.",
    )
    parser.add_argument("--output", type=Path, default=Path("gnn_allocation_dataset"))
    parser.add_argument(
        "--solver", type=Path,
        default=Path(__file__).with_name("solve_packing_heat_transfer.py"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(root, args.pattern, args.case)
    solver = pilot.load_solver(args.solver.resolve())

    manifest_rows = []
    for case in cases:
        allocation_dir = resolve_allocation_dir(
            root, case, args.allocation_template
        )
        arrays, metadata = build_graph(solver, case, allocation_dir)
        graph_name = f"graph_{case.name}.npz"
        metadata_name = f"graph_{case.name}.json"
        np.savez_compressed(output / graph_name, **arrays)
        (output / metadata_name).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        manifest_rows.append({
            "case": metadata["case"],
            "seed": metadata["seed"],
            "graph_file": graph_name,
            "metadata_file": metadata_name,
            "particles": metadata["particles"],
            "undirected_contact_edges": metadata["undirected_contact_edges"],
            "homogeneous_k_eff_W_mK": metadata["homogeneous_k_eff_W_mK"],
            "random_mean_k_eff_W_mK": metadata["random_mean_k_eff_W_mK"],
            "heat_path_k_eff_W_mK": metadata["heat_path_k_eff_W_mK"],
            "teacher_best_k_eff_W_mK": metadata["teacher_best_k_eff_W_mK"],
        })
        print(f"wrote {graph_name}")

    with (output / "dataset_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    dataset_metadata = {
        "graphs": len(manifest_rows),
        "node_feature_names": NODE_FEATURES,
        "edge_feature_names": EDGE_FEATURES,
        "target_selection_frequency": (
            "Fraction of final optimizer restarts selecting each particle"
        ),
        "target_best_selection": (
            "Binary membership in the best fixed-budget allocation"
        ),
        "recommended_split": "Leave one complete packing out; never split nodes.",
        "recommended_inference": (
            "Rank node scores separately within each particle type and apply "
            "the recorded type quotas."
        ),
        "leakage_warning": (
            "Fit feature standardization on training graphs only. Do not use "
            "teacher-allocation features as model inputs."
        ),
    }
    (output / "dataset_metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(dataset_metadata, indent=2))


if __name__ == "__main__":
    main()
