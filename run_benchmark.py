# run_benchmark.py
from __future__ import annotations

import argparse
import os
import sys
sys.path.append('./evolvDM')

import random
from pathlib import Path
from typing import Optional

import numpy as np

from syran_training import SyranConfig, run_anomaly_experiment


def _set_global_seed(seed: Optional[int]) -> None:
    """Set numpy / random / PYTHONHASHSEED seeds for basic reproducibility."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _load_benchmark_dataset(data_root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from ``data_root / f"{dataset}.npz"``."""
    path = data_root / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    data = np.load(path)
    return data["x"], data["tx"], data["ty"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SYRAN anomaly detection benchmark on a single dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ACardio",
        help="Dataset name (expects file data_root/<dataset>.npz with keys x, tx, ty).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory where <dataset>.npz files are stored (default: ./data).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory where results will be stored (default: ./results).",
    )
    parser.add_argument(
        "--complexity_weight",
        type=float,
        default=0.4,
        help="Weight for the complexity penalty (default: 0.4).",
    )
    parser.add_argument(
        "--loss_bound",
        type=float,
        default=1.0,
        help="Loss2 boundary term used in the anomaly objective (default: 1.0).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of variables per chunk. Default: min(3, n_features).",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="Number of chunks to optimise. Default: 10 if n_features<=3 else 20.",
    )
    parser.add_argument(
        "--max_phase_iterations",
        type=int,
        default=30_000,
        help="Number of phase_search iterations per chunk (default: 30000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    dataset = args.dataset

    _set_global_seed(args.seed)

    train_data, test_data, test_labels = _load_benchmark_dataset(data_root, dataset)

    config = SyranConfig(
        complexity_weight=args.complexity_weight,
        chunk_size=args.chunk_size,
        num_chunks=args.num_chunks,
        loss_bound=args.loss_bound,
        max_phase_iterations=args.max_phase_iterations,
        seed=args.seed,
    )

    output_dir = (
        output_root
        / dataset
        / str(config.effective_chunk_size(train_data.shape[1]))
        / str(config.loss_bound)
        / str(config.complexity_weight)
    )

    results = run_anomaly_experiment(
        train_data=train_data,
        test_data=test_data,
        test_labels=test_labels,
        dataset_name=dataset,
        output_dir=output_dir,
        config=config,
    )

    if not results:
        print("No successful chunks were produced.")
        return

    mean_auc = float(np.mean([r.roc_auc for r in results]))
    best_auc = float(max(r.roc_auc for r in results))
    print(f"\nFinished {dataset}: mean ROC AUC over {len(results)} chunks = {mean_auc:.4f}")
    print(f"Best chunk ROC AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
