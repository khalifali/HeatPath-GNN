#!/usr/bin/env python3
"""Fixed-budget conductivity-allocation pilot for one DEM packing.

Compares 100 random allocations with degree ranking, a physics-based heat-path
ranking, and a fixed-budget stochastic swap optimization.  The script imports
the existing solve_packing_heat_transfer.py and deliberately uses contact-only
conduction (no provisional fluid-gap edges).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Evaluation:
    k_eff: float
    q_hot: float
    q_cold: float
    balance_error: float
    solved_particles: int
    temperature: np.ndarray
    edges: list


def load_solver(path: Path):
    spec = importlib.util.spec_from_file_location("packing_heat_solver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load thermal solver from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContactOnlyEvaluator:
    def __init__(self, solver, case: Path, k_low: float, k_high: float,
                 t_hot: float, t_cold: float, overlap_floor_ratio: float,
                 wall_tolerance_ratio: float):
        self.solver = solver
        self.case = case.resolve()
        self.k_low = float(k_low)
        self.k_high = float(k_high)
        self.t_hot = float(t_hot)
        self.t_cold = float(t_cold)
        self.particles, self.box = solver.read_particles(
            self.case / "particles_final.dump"
        )
        self.contact_pairs = solver.read_contact_ids(
            self.case / "contacts_final.dump"
        )
        self.ids = np.asarray([p.atom_id for p in self.particles], dtype=int)
        self.id_to_index = {int(atom_id): i for i, atom_id in enumerate(self.ids)}
        self.known_ids = set(int(i) for i in self.ids)
        radii = np.asarray([p.radius for p in self.particles], dtype=float)
        self.d_ref = float(2.0 * np.median(radii))
        self.overlap_floor = overlap_floor_ratio * self.d_ref
        self.wall_tolerance = wall_tolerance_ratio * self.d_ref
        self.cache: dict[frozenset[int], Evaluation] = {}

    def evaluate(self, high_ids: Iterable[int]) -> Evaluation:
        key = frozenset(int(i) for i in high_ids)
        if key in self.cache:
            return self.cache[key]
        unknown = key - self.known_ids
        if unknown:
            raise ValueError(f"Unknown particle IDs: {sorted(unknown)[:10]}")
        conductivity = np.asarray([
            self.k_high if int(atom_id) in key else self.k_low
            for atom_id in self.ids
        ])
        edges, _ = self.solver.build_contact_edges(
            self.particles, self.box, self.contact_pairs, conductivity,
            self.overlap_floor,
        )
        g_hot, g_cold, _, _ = self.solver.build_wall_links(
            self.particles, self.box, conductivity, math.inf,
            self.wall_tolerance, self.overlap_floor, 0.0, 0.0, 0.0,
        )
        diagnostics = self.solver.graph_diagnostics(
            len(self.particles), edges, g_hot, g_cold
        )
        if diagnostics["spanning_component_count"] < 1:
            raise RuntimeError("No contact component spans the two thermal walls")
        active = self.solver.thermally_anchored_mask(
            len(self.particles), edges, g_hot, g_cold
        )
        temperature, _, q_hot, q_cold, error, _ = self.solver.solve_network(
            len(self.particles), edges, g_hot, g_cold,
            self.t_hot, self.t_cold, active,
        )
        k_eff = q_hot * self.box.height / (
            self.box.area_z * (self.t_hot - self.t_cold)
        )
        result = Evaluation(
            float(k_eff), float(q_hot), float(q_cold), float(error),
            int(active.sum()), temperature, edges,
        )
        self.cache[key] = result
        return result


def particle_degrees(evaluator: ContactOnlyEvaluator) -> np.ndarray:
    degree = np.zeros(len(evaluator.particles), dtype=int)
    for atom_i, atom_j in evaluator.contact_pairs:
        degree[evaluator.id_to_index[atom_i]] += 1
        degree[evaluator.id_to_index[atom_j]] += 1
    return degree


def heat_path_scores(evaluator: ContactOnlyEvaluator,
                     baseline: Evaluation) -> np.ndarray:
    """Baseline heat throughput: sum of |G_ij(T_i-T_j)| at each particle."""
    score = np.zeros(len(evaluator.particles), dtype=float)
    for edge in baseline.edges:
        ti, tj = baseline.temperature[edge.i], baseline.temperature[edge.j]
        if np.isfinite(ti) and np.isfinite(tj):
            current = abs(edge.conductance * (ti - tj))
            score[edge.i] += current
            score[edge.j] += current
    return score


def select_top(ids: np.ndarray, score: np.ndarray, count: int) -> frozenset[int]:
    order = np.lexsort((ids, -score))
    return frozenset(int(i) for i in ids[order[:count]])


def optimize_swaps(evaluator: ContactOnlyEvaluator, initial: frozenset[int],
                   proposals: int, rng: np.random.Generator):
    selected = set(initial)
    current = evaluator.evaluate(selected)
    history = [(0, current.k_eff, 0)]
    accepted = 0
    for proposal in range(1, proposals + 1):
        remove_id = int(rng.choice(np.asarray(sorted(selected))))
        unselected = evaluator.known_ids - selected
        add_id = int(rng.choice(np.asarray(sorted(unselected))))
        candidate = (selected - {remove_id}) | {add_id}
        trial = evaluator.evaluate(candidate)
        if trial.k_eff > current.k_eff * (1.0 + 1.0e-12):
            selected, current = candidate, trial
            accepted += 1
        history.append((proposal, current.k_eff, accepted))
    return frozenset(selected), current, history


def write_ids(path: Path, ids: Iterable[int]) -> None:
    path.write_text("".join(f"{i}\n" for i in sorted(ids)), encoding="utf-8")


def result_row(method: str, replicate: int, high_ids: Iterable[int],
               result: Evaluation, homogeneous_k: float,
               random_mean: float) -> dict:
    return {
        "method": method,
        "replicate": replicate,
        "high_count": len(set(high_ids)),
        "effective_conductivity_W_mK": result.k_eff,
        "improvement_vs_homogeneous_percent":
            100.0 * (result.k_eff / homogeneous_k - 1.0),
        "improvement_vs_random_mean_percent":
            100.0 * (result.k_eff / random_mean - 1.0),
        "heat_rate_hot_W": result.q_hot,
        "relative_energy_balance_error": result.balance_error,
        "thermally_solved_particles": result.solved_particles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--solver", type=Path,
                        default=Path(__file__).with_name("solve_packing_heat_transfer.py"))
    parser.add_argument("--output", type=Path,
                        default=Path("allocation_pilot_seed18427"))
    parser.add_argument("--k-low", type=float, default=1.0)
    parser.add_argument("--k-high", type=float, default=10.0)
    parser.add_argument("--high-count", type=int, default=50)
    parser.add_argument("--random-samples", type=int, default=100)
    parser.add_argument("--swap-proposals", type=int, default=400)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--t-hot", type=float, default=301.0)
    parser.add_argument("--t-cold", type=float, default=300.0)
    parser.add_argument("--overlap-floor-ratio", type=float, default=1.0e-8)
    parser.add_argument("--wall-tolerance-ratio", type=float, default=1.0e-8)
    args = parser.parse_args()

    if not (0.0 < args.k_low < args.k_high):
        parser.error("require 0 < k-low < k-high")
    if args.t_hot <= args.t_cold:
        parser.error("require t-hot > t-cold")
    if args.random_samples < 2 or args.swap_proposals < 0:
        parser.error("require random-samples >= 2 and swap-proposals >= 0")

    solver = load_solver(args.solver.resolve())
    evaluator = ContactOnlyEvaluator(
        solver, args.case, args.k_low, args.k_high, args.t_hot, args.t_cold,
        args.overlap_floor_ratio, args.wall_tolerance_ratio,
    )
    n = len(evaluator.particles)
    if not 1 <= args.high_count < n:
        parser.error(f"high-count must be between 1 and {n - 1}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    homogeneous_ids: frozenset[int] = frozenset()
    homogeneous = evaluator.evaluate(homogeneous_ids)

    random_ids, random_results = [], []
    for _ in range(args.random_samples):
        allocation = frozenset(int(i) for i in rng.choice(
            evaluator.ids, size=args.high_count, replace=False
        ))
        if len(allocation) != args.high_count:
            raise RuntimeError("Random allocation violated the fixed budget")
        random_ids.append(allocation)
        random_results.append(evaluator.evaluate(allocation))
    random_values = np.asarray([r.k_eff for r in random_results])
    random_mean = float(random_values.mean())
    random_std = float(random_values.std(ddof=1))

    degree_ids = select_top(
        evaluator.ids, particle_degrees(evaluator).astype(float), args.high_count
    )
    degree_result = evaluator.evaluate(degree_ids)
    path_ids = select_top(
        evaluator.ids, heat_path_scores(evaluator, homogeneous), args.high_count
    )
    path_result = evaluator.evaluate(path_ids)
    initial = degree_ids if degree_result.k_eff >= path_result.k_eff else path_ids
    optimized_ids, optimized_result, history = optimize_swaps(
        evaluator, initial, args.swap_proposals, rng
    )
    if len(optimized_ids) != args.high_count:
        raise RuntimeError("Optimization violated the fixed budget")

    rows = [result_row(
        "homogeneous", 0, homogeneous_ids, homogeneous,
        homogeneous.k_eff, random_mean,
    )]
    rows.extend(result_row(
        "random", i, ids, result, homogeneous.k_eff, random_mean
    ) for i, (ids, result) in enumerate(zip(random_ids, random_results), 1))
    rows.extend([
        result_row("degree", 1, degree_ids, degree_result,
                   homogeneous.k_eff, random_mean),
        result_row("heat_path", 1, path_ids, path_result,
                   homogeneous.k_eff, random_mean),
        result_row("swap_optimized", 1, optimized_ids, optimized_result,
                   homogeneous.k_eff, random_mean),
    ])
    with (output / "allocation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "optimization_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["proposal", "best_k_eff_W_mK", "accepted_swaps"])
        writer.writerows(history)

    best_random = int(np.argmax(random_values))
    write_ids(output / "high_ids_best_random.txt", random_ids[best_random])
    write_ids(output / "high_ids_degree.txt", degree_ids)
    write_ids(output / "high_ids_heat_path.txt", path_ids)
    write_ids(output / "high_ids_optimized.txt", optimized_ids)

    summary = {
        "case": str(evaluator.case),
        "particles": n,
        "high_count": args.high_count,
        "high_fraction": args.high_count / n,
        "k_low_W_mK": args.k_low,
        "k_high_W_mK": args.k_high,
        "contact_only": True,
        "random_samples": args.random_samples,
        "swap_proposals": args.swap_proposals,
        "seed": args.seed,
        "homogeneous_k_eff_W_mK": homogeneous.k_eff,
        "random_mean_k_eff_W_mK": random_mean,
        "random_std_k_eff_W_mK": random_std,
        "random_cv_percent": 100.0 * random_std / random_mean,
        "random_min_k_eff_W_mK": float(random_values.min()),
        "random_max_k_eff_W_mK": float(random_values.max()),
        "degree_k_eff_W_mK": degree_result.k_eff,
        "heat_path_k_eff_W_mK": path_result.k_eff,
        "optimized_k_eff_W_mK": optimized_result.k_eff,
        "optimized_improvement_vs_random_mean_percent":
            100.0 * (optimized_result.k_eff / random_mean - 1.0),
        "optimized_improvement_vs_homogeneous_percent":
            100.0 * (optimized_result.k_eff / homogeneous.k_eff - 1.0),
        "accepted_swaps": history[-1][2],
        "unique_thermal_evaluations": len(evaluator.cache),
        "maximum_energy_balance_error": max(
            r.balance_error for r in evaluator.cache.values()
        ),
    }
    (output / "allocation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
