# syran_evaluation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score


# Default benchmark datasets(and hyperparameter grids for ablation)
DATASETS: List[str] = [
    "APima",
    "Abreastw",
    "Acardio",
    "AStamps",
    "ACardiotocography",
    "ALymphography",
    "APageBlocks",
    "Aglass",
    "AWaveform",
    "Aannthyroid",
    "Ayeast",
    "Apendigits",
    "AWilt",
    "AHepatitis",
    "Awine",
    "Athyroid",
    "AWBC",
    "Avowels",
    "Avertebral",
]

COMPLEXITY_WEIGHTS: List[float] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
CHUNK_SIZES: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10]
LOSS_BOUNDS: List[float] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def _results_dir(
    results_root: Union[str, Path],
    dataset: str,
    chunk_size: int,
    loss_bound: float,
    complexity_weight: float,
) -> Path:
    """Directory containing the per-chunk .npz files for a given setting.

    Layout:
        {results_root}/{dataset}/{chunk_size}/{loss_bound}/{complexity_weight}/
    """
    return (
        Path(results_root)
        / dataset
        / str(chunk_size)
        / str(loss_bound)
        / str(complexity_weight)
    )


def _load_chunk_files(
    results_root: Union[str, Path],
    dataset: str,
    chunk_size: int,
    loss_bound: float,
    complexity_weight: float,
    max_chunks: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[str]]:
    """Load all per-chunk results for a given configuration.

    Returns:
        test_scores: 2D array (n_chunks, n_samples)
        test_labels: 1D array (n_samples,)
        roc_aucs:   list of per-chunk ROC AUCs
        equations:  list of symbolic equations (strings)
    """
    directory = _results_dir(results_root, dataset, chunk_size, loss_bound, complexity_weight)
    if not directory.exists():
        raise FileNotFoundError(f"No results found in {directory}")

    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {directory}")

    if max_chunks is not None:
        files = files[:max_chunks]

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    roc_aucs: List[float] = []
    equations: List[str] = []

    for path in files:
        data = np.load(path, allow_pickle=True)
        if "test_scores" not in data or "test_labels" not in data:
            # Skip toy / incomplete runs.
            continue

        scores = np.asarray(data["test_scores"])
        labels = np.asarray(data["test_labels"])
        if scores.ndim != 1:
            raise ValueError(f"Expected 1D scores in {path}, got shape {scores.shape}")

        all_scores.append(scores)
        all_labels.append(labels)

        roc_aucs.append(float(data.get("roc_auc_chunk", np.nan)))
        equations.append(str(data.get("symbolic_equation", "")))

    if not all_scores:
        raise RuntimeError(f"No usable anomaly results found in {directory}")

    # Sanity check: all labels should be identical.
    first_labels = all_labels[0]
    for labels in all_labels[1:]:
        if not np.array_equal(first_labels, labels):
            raise ValueError(f"Inconsistent test labels across files in {directory}")

    test_scores = np.stack(all_scores, axis=0)
    test_labels = first_labels
    return test_scores, test_labels, roc_aucs, equations


def mean_roc_auc(
    results_root: Union[str, Path],
    dataset: str,
    chunk_size: int,
    loss_bound: float,
    complexity_weight: float,
    num_chunks: Union[int, str] = "max",
) -> float:
    """Compute ROC AUC from the mean anomaly score over chunks."""
    max_chunks: Optional[int]
    if isinstance(num_chunks, str):
        if num_chunks != "max":
            raise ValueError("num_chunks must be an int or 'max'")
        max_chunks = None
    else:
        max_chunks = num_chunks

    scores, labels, _, _ = _load_chunk_files(
        results_root,
        dataset,
        chunk_size,
        loss_bound,
        complexity_weight,
        max_chunks=max_chunks,
    )

    if max_chunks is not None:
        scores = scores[:max_chunks]

    mean_scores = scores.mean(axis=0)
    return float(roc_auc_score(labels, mean_scores))


def best_chunk_roc_auc(
    results_root: Union[str, Path],
    dataset: str,
    chunk_size: int,
    loss_bound: float,
    complexity_weight: float,
    num_chunks: Union[int, str] = "max",
) -> Tuple[float, str]:
    """Return ROC AUC and equation for the best-performing chunk.

    The best chunk is selected by its per-chunk ROC AUC (stored in the .npz
    files), which is a more directly relevant criterion than internal fitness.
    """
    max_chunks: Optional[int]
    if isinstance(num_chunks, str):
        if num_chunks != "max":
            raise ValueError("num_chunks must be an int or 'max'")
        max_chunks = None
    else:
        max_chunks = num_chunks

    scores, labels, roc_aucs, equations = _load_chunk_files(
        results_root,
        dataset,
        chunk_size,
        loss_bound,
        complexity_weight,
        max_chunks=max_chunks,
    )

    # If stored ROC AUCs are missing / NaN, recompute them.
    recompute = any(np.isnan(a) for a in roc_aucs)
    if recompute:
        roc_aucs = [float(roc_auc_score(labels, s)) for s in scores]

    best_idx = int(np.argmax(roc_aucs))
    best_auc = float(roc_aucs[best_idx])
    best_equation = equations[best_idx]
    return best_auc, best_equation


def load_result(
    chunk_size: int,
    complexity_weight: float,
    num_chunks: Union[int, str] = "max",
    loss_bound: float = 1.0,
    results_root: Union[str, Path] = "results",
    datasets: Sequence[str] = DATASETS,
) -> Dict[str, float]:
    """Convenience wrapper: mean ROC AUC for multiple datasets."""
    return {
        ds: mean_roc_auc(
            results_root,
            dataset=ds,
            chunk_size=chunk_size,
            loss_bound=loss_bound,
            complexity_weight=complexity_weight,
            num_chunks=num_chunks,
        )
        for ds in datasets
    }


def load_best_results(
    chunk_size: int,
    complexity_weight: float,
    num_chunks: Union[int, str] = "max",
    loss_bound: float = 1.0,
    results_root: Union[str, Path] = "results",
    datasets: Sequence[str] = DATASETS,
) -> Dict[str, Tuple[float, str]]:
    """Convenience wrapper: best-chunk ROC AUC and equation per dataset."""
    return {
        ds: best_chunk_roc_auc(
            results_root,
            dataset=ds,
            chunk_size=chunk_size,
            loss_bound=loss_bound,
            complexity_weight=complexity_weight,
            num_chunks=num_chunks,
        )
        for ds in datasets
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate SYRAN ROC AUC scores")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--chunk_size", type=int, default=2)
    parser.add_argument("--complexity_weight", type=float, default=0.4)
    parser.add_argument("--loss_bound", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--num_chunks", type=str, default="max",
                        help="Either an integer or 'max'")
    args = parser.parse_args()

    try:
        num_chunks: Union[int, str]
        num_chunks = int(args.num_chunks)
    except ValueError:
        num_chunks = args.num_chunks

    if args.dataset != "all":
        datasets = [args.dataset]
    else:
        datasets = DATASETS

    mean_results = load_result(
        chunk_size=args.chunk_size,
        complexity_weight=args.complexity_weight,
        num_chunks=num_chunks,
        loss_bound=args.loss_bound,
        results_root=args.results_root,
        datasets=datasets,
    )
    best_results = load_best_results(
        chunk_size=args.chunk_size,
        complexity_weight=args.complexity_weight,
        num_chunks=num_chunks,
        loss_bound=args.loss_bound,
        results_root=args.results_root,
        datasets=datasets,
    )

    for ds in datasets:
        mean_auc = mean_results.get(ds, float("nan"))
        best_auc, _ = best_results.get(ds, (float("nan"), ""))
        print(f"{ds}: mean={mean_auc:.4f}, best={best_auc:.4f}")

    print("Mean over datasets (mean scores):", np.nanmean(list(mean_results.values())))
    print("Mean over datasets (best chunks):",
          np.nanmean([v[0] for v in best_results.values()]))
