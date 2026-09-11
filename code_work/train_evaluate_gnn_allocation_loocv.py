#!/usr/bin/env python3
"""Leave-one-packing-out test of whether a GNN improves material allocation.

Two supervised node-ranking models are trained on exactly the same node inputs:

* ``mlp``: a non-graph particle model;
* ``gnn``: an edge-aware message-passing graph neural network.

The target is the optimizer-restart selection frequency.  For each held-out
packing, predicted scores are converted to a valid allocation by selecting the
required number of particles separately within each particle type.  Every
predicted allocation is then evaluated with the original contact-only thermal
solver.  Packings, never particles, define the train/validation/test split.

PyTorch is required; PyTorch Geometric is not used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Hide CUDA before importing PyTorch when CPU was explicitly requested.  Some
# CUDA-enabled PyTorch builds otherwise probe an incompatible driver during
# backward even though every tensor is on the CPU.
if "--device" in sys.argv:
    _device_position = sys.argv.index("--device")
    if (_device_position + 1 < len(sys.argv)
            and sys.argv[_device_position + 1].lower() == "cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised on machines without torch
    raise SystemExit(
        "PyTorch is required. Install a build suitable for your CPU/CUDA system "
        "from https://pytorch.org/get-started/locally/ and rerun this script. "
        "PyTorch Geometric is not required."
    ) from exc

import run_allocation_pilot_stratified_v2 as pilot


@dataclass
class Graph:
    case: str
    seed: int | None
    node_ids: np.ndarray
    node_types: np.ndarray
    x_raw: np.ndarray
    edge_index: np.ndarray
    edge_raw: np.ndarray
    target: np.ndarray
    target_best: np.ndarray
    metadata: dict
    x: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    edge_index_tensor: torch.Tensor | None = None
    target_tensor: torch.Tensor | None = None
    type_tensor: torch.Tensor | None = None


def set_seed(seed: int, use_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Do not probe CUDA during an explicitly CPU-only run.  On systems with an
    # incompatible NVIDIA driver/runtime, even torch.cuda.is_available() can
    # emit a warning although CUDA was not requested.
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def load_graphs(dataset: Path) -> list[Graph]:
    manifest_path = dataset / "dataset_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 3:
        raise ValueError("At least three packing graphs are required for train/validation/test")
    graphs = []
    for row in rows:
        npz_path = dataset / row["graph_file"]
        metadata_path = dataset / row["metadata_file"]
        arrays = np.load(npz_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        graph = Graph(
            case=metadata["case"],
            seed=metadata.get("seed"),
            node_ids=arrays["node_ids"].astype(np.int64),
            node_types=arrays["node_types"].astype(np.int64),
            x_raw=arrays["node_features"].astype(np.float32),
            edge_index=arrays["edge_index"].astype(np.int64),
            edge_raw=arrays["edge_features"].astype(np.float32),
            target=arrays["target_selection_frequency"].astype(np.float32),
            target_best=arrays["target_best_selection"].astype(np.float32),
            metadata=metadata,
        )
        validate_graph(graph)
        graphs.append(graph)
    return graphs


def validate_graph(graph: Graph) -> None:
    n = len(graph.node_ids)
    if graph.x_raw.shape[0] != n or graph.target.shape != (n,):
        raise ValueError(f"Node-array mismatch in {graph.case}")
    if graph.edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E] in {graph.case}")
    if graph.edge_raw.shape[0] != graph.edge_index.shape[1]:
        raise ValueError(f"Edge-array mismatch in {graph.case}")
    for name, array in [("node features", graph.x_raw),
                        ("edge features", graph.edge_raw),
                        ("soft target", graph.target)]:
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite {name} in {graph.case}")
    quotas = {int(k): int(v) for k, v in graph.metadata["type_quotas"].items()}
    for atom_type, quota in quotas.items():
        mask = graph.node_types == atom_type
        if not np.isclose(graph.target[mask].sum(), quota, atol=1.0e-5):
            raise ValueError(f"Soft target quota mismatch in {graph.case}, type {atom_type}")
        if int(graph.target_best[mask].sum()) != quota:
            raise ValueError(f"Best target quota mismatch in {graph.case}, type {atom_type}")


def scaler(training: list[Graph]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.vstack([graph.x_raw for graph in training])
    edges = np.vstack([graph.edge_raw for graph in training])
    node_mean, node_std = nodes.mean(axis=0), nodes.std(axis=0)
    edge_mean, edge_std = edges.mean(axis=0), edges.std(axis=0)
    node_std = np.where(node_std > 1.0e-8, node_std, 1.0)
    edge_std = np.where(edge_std > 1.0e-8, edge_std, 1.0)
    return node_mean, node_std, edge_mean, edge_std


def tensorize(graph: Graph, stats, device: torch.device) -> Graph:
    node_mean, node_std, edge_mean, edge_std = stats
    result = copy.copy(graph)
    result.x = torch.as_tensor(
        (graph.x_raw - node_mean) / node_std, dtype=torch.float32, device=device
    )
    result.edge_attr = torch.as_tensor(
        (graph.edge_raw - edge_mean) / edge_std,
        dtype=torch.float32, device=device,
    )
    result.edge_index_tensor = torch.as_tensor(
        graph.edge_index, dtype=torch.long, device=device
    )
    result.target_tensor = torch.as_tensor(
        graph.target, dtype=torch.float32, device=device
    )
    result.type_tensor = torch.as_tensor(
        graph.node_types, dtype=torch.long, device=device
    )
    return result


class MLPScorer(nn.Module):
    def __init__(self, node_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, graph: Graph) -> torch.Tensor:
        return self.network(graph.x).squeeze(-1)


class MessageLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        source, target = edge_index[0], edge_index[1]
        message = self.message(torch.cat([h[source], h[target], edge_attr], dim=1))
        aggregate = torch.zeros_like(h)
        aggregate.index_add_(0, target, message)
        count = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        count.index_add_(0, target, torch.ones(
            (target.numel(), 1), dtype=h.dtype, device=h.device
        ))
        aggregate = aggregate / count.clamp_min(1.0)
        return self.norm(h + self.update(torch.cat([h, aggregate], dim=1)))


class EdgeGNNScorer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int,
                 layers: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.layers = nn.ModuleList([
            MessageLayer(hidden, edge_dim, dropout) for _ in range(layers)
        ])
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, graph: Graph) -> torch.Tensor:
        h = self.encoder(graph.x)
        for layer in self.layers:
            h = layer(h, graph.edge_index_tensor, graph.edge_attr)
        return self.output(h).squeeze(-1)


def ranking_loss(scores: torch.Tensor, graph: Graph,
                 bce_weight: float) -> torch.Tensor:
    """Type-stratified listwise ranking loss plus a small soft BCE term."""
    losses = []
    for atom_type in sorted(int(v) for v in torch.unique(graph.type_tensor)):
        mask = graph.type_tensor == atom_type
        target = graph.target_tensor[mask]
        target_sum = target.sum()
        if target_sum <= 0:
            continue
        target_distribution = target / target_sum
        losses.append(-(target_distribution * torch.log_softmax(scores[mask], 0)).sum())
    listwise = torch.stack(losses).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(
        scores, graph.target_tensor
    )
    return listwise + bce_weight * bce


def train_model(model: nn.Module, training: list[Graph], validation: Graph,
                epochs: int, patience: int, learning_rate: float,
                weight_decay: float, bce_weight: float) -> tuple[nn.Module, dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = 0
    waiting = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        # Accumulate the mean gradient one graph at a time to bound memory.
        for graph in training:
            loss = ranking_loss(model(graph), graph, bce_weight) / len(training)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(ranking_loss(
                model(validation), validation, bce_weight
            ).cpu())
        if validation_loss < best_validation - 1.0e-6:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
        if waiting >= patience:
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_loss": best_validation,
        "training_time_s": time.perf_counter() - started,
    }


def predict(model: nn.Module, graph: Graph) -> tuple[np.ndarray, float]:
    model.eval()
    if graph.x.is_cuda:
        torch.cuda.synchronize(graph.x.device)
    started = time.perf_counter()
    with torch.no_grad():
        scores = model(graph).detach().cpu().numpy()
    return scores, time.perf_counter() - started


def select_by_quota(graph: Graph, scores: np.ndarray) -> frozenset[int]:
    quotas = {int(k): int(v) for k, v in graph.metadata["type_quotas"].items()}
    selected = []
    for atom_type, quota in sorted(quotas.items()):
        indices = np.flatnonzero(graph.node_types == atom_type)
        # Stable particle-ID tie break after descending score.
        order = np.lexsort((graph.node_ids[indices], -scores[indices]))
        selected.extend(int(i) for i in graph.node_ids[indices[order[:quota]]])
    return frozenset(selected)


def typewise_rank_scores(graph: Graph, scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary logits to comparable [0,1] ranks within each type.

    Independently trained ranking models can produce logits with different
    offsets and scales.  Averaging raw logits can therefore let one repeat
    dominate the ensemble.  Percentile-like rank scores preserve each model's
    ordering, which is the quantity used by quota-constrained inference.
    """
    ranked = np.zeros_like(scores, dtype=float)
    for atom_type in sorted(np.unique(graph.node_types)):
        indices = np.flatnonzero(graph.node_types == atom_type)
        # Ascending score; particle ID supplies a deterministic tie break.
        order = np.lexsort((graph.node_ids[indices], scores[indices]))
        values = np.empty(len(indices), dtype=float)
        if len(indices) == 1:
            values[order] = 1.0
        else:
            values[order] = np.arange(len(indices), dtype=float) / (len(indices) - 1)
        ranked[indices] = values
    return ranked


def write_ids(path: Path, ids: Iterable[int]) -> None:
    path.write_text("".join(f"{i}\n" for i in sorted(ids)), encoding="utf-8")


def thermal_evaluator(graph: Graph, project_root: Path, solver, cache: dict):
    case_candidate = project_root / graph.case
    case = case_candidate if case_candidate.is_dir() else Path(
        graph.metadata["case_directory"]
    )
    key = str(case.resolve())
    if key not in cache:
        cache[key] = pilot.ContactOnlyEvaluator(
            solver,
            case,
            float(graph.metadata["k_low_W_mK"]),
            float(graph.metadata["k_high_W_mK"]),
            float(graph.metadata.get("temperature_hot_K", 301.0)),
            float(graph.metadata.get("temperature_cold_K", 300.0)),
            1.0e-8,
            1.0e-8,
        )
    return cache[key]


def result_row(fold: int, graph: Graph, model_name: str, repeat: str,
               ids: frozenset[int], k_eff: float, inference_s: float,
               training_s: float, best_epoch: float) -> dict:
    random_mean = float(graph.metadata["random_mean_k_eff_W_mK"])
    teacher = float(graph.metadata["teacher_best_k_eff_W_mK"])
    heat_path = float(graph.metadata["heat_path_k_eff_W_mK"])
    denominator = teacher - random_mean
    recovered = (k_eff - random_mean) / denominator if denominator > 0 else math.nan
    best_ids = set(graph.node_ids[graph.target_best > 0.5].tolist())
    intersection = len(set(ids) & best_ids)
    union = len(set(ids) | best_ids)
    return {
        "fold": fold,
        "test_case": graph.case,
        "test_seed": graph.seed,
        "model": model_name,
        "repeat": repeat,
        "effective_conductivity_W_mK": k_eff,
        "random_mean_k_eff_W_mK": random_mean,
        "heat_path_k_eff_W_mK": heat_path,
        "teacher_best_k_eff_W_mK": teacher,
        "improvement_vs_random_percent": 100.0 * (k_eff / random_mean - 1.0),
        "recovered_optimizer_gain_fraction": recovered,
        "difference_vs_heat_path_percent": 100.0 * (k_eff / heat_path - 1.0),
        "difference_vs_teacher_percent": 100.0 * (k_eff / teacher - 1.0),
        "outperforms_heat_path": int(k_eff > heat_path),
        "selected_count": len(ids),
        "jaccard_with_teacher_best": intersection / union if union else 1.0,
        "inference_time_s": inference_s,
        "training_time_s": training_s,
        "best_epoch": best_epoch,
    }


def aggregate_summary(rows: list[dict]) -> dict:
    ensemble = [row for row in rows if row["repeat"] == "ensemble"]
    summary = {}
    for model in sorted({row["model"] for row in ensemble}):
        selected = [row for row in ensemble if row["model"] == model]
        recovered = np.asarray([
            row["recovered_optimizer_gain_fraction"] for row in selected
        ], dtype=float)
        conductivity = np.asarray([
            row["effective_conductivity_W_mK"] for row in selected
        ], dtype=float)
        summary[model] = {
            "folds": len(selected),
            "mean_effective_conductivity_W_mK": float(conductivity.mean()),
            "std_effective_conductivity_W_mK": float(conductivity.std(ddof=1)),
            "mean_recovered_optimizer_gain_fraction": float(recovered.mean()),
            "std_recovered_optimizer_gain_fraction": float(recovered.std(ddof=1)),
            "min_recovered_optimizer_gain_fraction": float(recovered.min()),
            "max_recovered_optimizer_gain_fraction": float(recovered.max()),
            "folds_outperforming_heat_path": int(sum(
                row["outperforms_heat_path"] for row in selected
            )),
            "mean_jaccard_with_teacher_best": float(np.mean([
                row["jaccard_with_teacher_best"] for row in selected
            ])),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("gnn_allocation_dataset"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("gnn_loocv_results_rank_ensemble"))
    parser.add_argument("--solver", type=Path,
                        default=Path(__file__).with_name("solve_packing_heat_transfer.py"))
    parser.add_argument("--models", default="mlp,gnn",
                        help="Comma-separated subset of mlp,gnn")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bce-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--test-case", help="Evaluate only this held-out case (campaign checkpoint unit)")
    args = parser.parse_args()

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names or any(name not in {"mlp", "gnn"} for name in model_names):
        parser.error("--models must contain mlp and/or gnn")
    if args.repeats < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("repeats, epochs and patience must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    ids_output = output / "predicted_ids"
    output.mkdir(parents=True, exist_ok=True)
    ids_output.mkdir(parents=True, exist_ok=True)
    graphs = load_graphs(dataset)
    if args.test_case and args.test_case not in {g.case for g in graphs}:
        parser.error("--test-case is not in the dataset")
    solver = pilot.load_solver(args.solver.resolve())
    evaluator_cache = {}
    rows: list[dict] = []

    for test_index, test_raw in enumerate(graphs):
        if args.test_case and test_raw.case != args.test_case:
            continue
        validation_index = (test_index - 1) % len(graphs)
        training_raw = [
            graph for index, graph in enumerate(graphs)
            if index not in {test_index, validation_index}
        ]
        validation_raw = graphs[validation_index]
        stats = scaler(training_raw)
        training = [tensorize(graph, stats, device) for graph in training_raw]
        validation = tensorize(validation_raw, stats, device)
        test = tensorize(test_raw, stats, device)
        evaluator = thermal_evaluator(
            test_raw, args.project_root.resolve(), solver, evaluator_cache
        )
        print(
            f"fold {test_index + 1}/{len(graphs)}: test={test.case}, "
            f"validation={validation.case}, training={len(training)} graphs"
        )

        for model_name in model_names:
            repeat_scores, repeat_rank_scores, repeat_info = [], [], []
            for repeat in range(1, args.repeats + 1):
                run_seed = args.seed + 100_003 * test_index + 1_009 * repeat + (
                    0 if model_name == "mlp" else 10_000_019
                )
                set_seed(run_seed, use_cuda=(device.type == "cuda"))
                if model_name == "mlp":
                    model = MLPScorer(test.x.shape[1], args.hidden, args.dropout)
                else:
                    model = EdgeGNNScorer(
                        test.x.shape[1], test.edge_attr.shape[1], args.hidden,
                        args.message_layers, args.dropout,
                    )
                model.to(device)
                model, info = train_model(
                    model, training, validation, args.epochs, args.patience,
                    args.learning_rate, args.weight_decay, args.bce_weight,
                )
                # Save weights and training-only normalization for reproducibility.
                checkpoint = output / "models" / f"{test.case}_{model_name}_repeat{repeat}.pt"
                checkpoint.parent.mkdir(exist_ok=True)
                torch.save({"state_dict": model.state_dict(),
                            "scaler": [v.tolist() for v in stats],
                            "settings": {k: str(v) if isinstance(v, Path) else v
                                         for k, v in vars(args).items()},
                            "training_cases": [g.case for g in training_raw],
                            "validation_case": validation.case,
                            "test_case": test.case, "run_seed": run_seed}, checkpoint)
                scores, inference_s = predict(model, test)
                ids = select_by_quota(test_raw, scores)
                result = evaluator.evaluate(ids)
                write_ids(
                    ids_output / f"{test.case}_{model_name}_repeat{repeat}.txt", ids
                )
                rows.append(result_row(
                    test_index + 1, test_raw, model_name, str(repeat), ids,
                    result.k_eff, inference_s, info["training_time_s"],
                    info["best_epoch"],
                ))
                repeat_scores.append(scores)
                repeat_rank_scores.append(typewise_rank_scores(test_raw, scores))
                repeat_info.append((inference_s, info))
                print(
                    f"  {model_name} repeat {repeat}: k_eff={result.k_eff:.8f}, "
                    f"recovered={rows[-1]['recovered_optimizer_gain_fraction']:.3f}"
                )

            # Ensemble rankings rather than raw logits because independently
            # trained models need not share the same score scale.
            ensemble_scores = np.mean(repeat_rank_scores, axis=0)
            ensemble_ids = select_by_quota(test_raw, ensemble_scores)
            ensemble_result = evaluator.evaluate(ensemble_ids)
            write_ids(ids_output / f"{test.case}_{model_name}_ensemble.txt", ensemble_ids)
            rows.append(result_row(
                test_index + 1, test_raw, model_name, "ensemble", ensemble_ids,
                ensemble_result.k_eff,
                float(sum(item[0] for item in repeat_info)),
                float(sum(item[1]["training_time_s"] for item in repeat_info)),
                float(np.mean([item[1]["best_epoch"] for item in repeat_info])),
            ))
            print(
                f"  {model_name} ensemble: k_eff={ensemble_result.k_eff:.8f}, "
                f"recovered={rows[-1]['recovered_optimizer_gain_fraction']:.3f}"
            )

    with (output / "loocv_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": str(dataset),
        "device": str(device),
        "graphs": len(graphs),
        "split": "leave-one-packing-out with one separate validation packing",
        "training_graphs_per_fold": len(graphs) - 2,
        "repeats": args.repeats,
        "ensemble_method": "mean of within-particle-type normalized ranks",
        "models": model_names,
        "model_summary_ensemble": aggregate_summary(rows),
        "decision_guidance": {
            "retain_gnn_if": [
                "GNN outperforms heat-path ranking on most held-out packings",
                "GNN clearly outperforms the same-feature non-graph MLP",
                "Report recovered search gain without imposing a post-hoc threshold",
                "Inference plus one thermal verification is much cheaper than swap search",
            ],
            "warning": (
                "Transfer is tested between realizations of the specified packing. "
                "Do not infer transfer to different PSDs, sizes or porosities."
            ),
        },
    }
    (output / "loocv_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

