#!/usr/bin/env python3
"""Plot fold-wise optimizer-gain recovery for the MLP and GNN ensembles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_ensemble_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["repeat"] == "ensemble"]
    by_model: dict[str, list[dict[str, str]]] = {}
    for model in ("mlp", "gnn"):
        model_rows = sorted(
            (row for row in selected if row["model"] == model),
            key=lambda row: int(row["fold"]),
        )
        if not model_rows:
            raise ValueError(f"No ensemble rows found for model {model!r} in {path}")
        by_model[model] = model_rows
    folds_mlp = [int(row["fold"]) for row in by_model["mlp"]]
    folds_gnn = [int(row["fold"]) for row in by_model["gnn"]]
    if folds_mlp != folds_gnn:
        raise ValueError("MLP and GNN ensemble folds do not match")
    return by_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("gnn_loocv_results_rank_ensemble/loocv_results.csv"),
        help="LOOCV result table produced by train_evaluate_gnn_allocation_loocv.py",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/gnn_loocv_recovered_gain.pdf"),
        help="PDF output; a PNG with the same stem is written as well",
    )
    args = parser.parse_args()

    rows = load_ensemble_rows(args.input)
    folds = [int(row["fold"]) for row in rows["mlp"]]
    mlp = [100.0 * float(row["recovered_optimizer_gain_fraction"])
           for row in rows["mlp"]]
    gnn = [100.0 * float(row["recovered_optimizer_gain_fraction"])
           for row in rows["gnn"]]

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(folds, mlp, color="#2b6cb0", marker="s", markersize=5,
            linewidth=1.8, label="Same-feature MLP")
    ax.plot(folds, gnn, color="#c53030", marker="o", markersize=5,
            linewidth=1.8, label="Edge-aware GNN")
    ax.set_xlabel("Held-out packing fold")
    ax.set_ylabel("Recovered search gain [%]")
    ax.set_xticks(folds)
    ax.set_xlim(min(folds) - 0.35, max(folds) + 0.35)
    ax.set_ylim(20, 75)
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
