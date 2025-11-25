# syran_training.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from evolvDM.morga import mguess
from evolvDM.phase_search import phase_search

from syran_model import (
    Expression,
    Variable,
    ValuesDict,
    anomaly_objective,
    generate_random_chunks,
    init_random_expression,
    kepler_objective,
    make_variables,
    sigmoid,
    solve_alpha,
    solve_expression,
    update_expression,
)


@dataclass
class SyranConfig:
    """Configuration for the anomaly detection benchmark."""

    complexity_weight: float = 0.4
    chunk_size: Optional[int] = None
    num_chunks: Optional[int] = None
    loss_bound: float = 1.0
    max_phase_iterations: int = 30_000
    seed: Optional[int] = None

    def effective_chunk_size(self, num_features: int) -> int:
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if self.chunk_size is not None:
            return min(self.chunk_size, num_features)
        return min(3, num_features)

    def effective_num_chunks(self, num_features: int) -> int:
        if self.num_chunks is not None:
            return self.num_chunks
        return 10 if num_features <= 3 else 20


@dataclass
class ChunkResult:
    """Result of a single chunk optimisation run."""

    dataset_name: str
    chunk_index: int
    variables: List[str]
    symbolic_equation: str
    roc_auc: float
    error: float
    chunk_size: int
    complexity_weight: float
    loss_bound: float
    train_scores: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray
    parameters: Dict[str, float]
    path: Path


@dataclass
class ToyKeplerConfig:
    """Configuration for the toy Kepler experiment."""

    complexity_weight: float = 0.03
    chunk_size: int = 2
    num_chunks: int = 50
    max_phase_iterations: int = 30_000
    seed: Optional[int] = None
    random_margin: float = 2.0


@dataclass
class ToyChunkResult:
    """Result of a single optimisation run in the Kepler toy experiment."""

    chunk_index: int
    variables: List[str]
    symbolic_equation: str
    objective_value: float
    error: float
    chunk_size: int
    complexity_weight: float
    true_solution_objective: float
    train_scores: np.ndarray
    parameters: Dict[str, float]
    path: Path


def _build_value_dict(
    data: np.ndarray,
    index_by_name: Mapping[str, int],
    chunk: Sequence[Variable],
) -> Dict[str, np.ndarray]:
    """Map variable names to their corresponding columns for a given chunk."""
    return {
        var.name: data[:, index_by_name[var.name]]
        for var in chunk
    }


def _build_random_value_dict(
    values_dict: ValuesDict,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Draw random values in the bounding box of the observed data."""
    random_values: Dict[str, np.ndarray] = {}
    for name, values in values_dict.items():
        values = np.asarray(values)
        low, high = float(values.min()), float(values.max())
        random_values[name] = rng.uniform(low, high, size=len(values))
    return random_values


def _extract_guess_parameters(expr: Expression) -> Dict[str, float]:
    """Extract mguess parameters from an expression."""
    params: Dict[str, float] = {}

    def visitor(node: Any) -> None:
        if isinstance(node, mguess):
            params[node.name] = float(node.value)

    # morga expressions support iterate_function, but we guard just in case.
    iterate = getattr(expr, "iterate_function", None)
    if callable(iterate):
        iterate(visitor)
    return params


def run_anomaly_experiment(
    train_data: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    dataset_name: str,
    output_dir: Path,
    config: Optional[SyranConfig] = None,
) -> List[ChunkResult]:
    """Run the SYRAN anomaly detection experiment on a single dataset.

    This function is model- and data-agnostic and can be called from a thin CLI wrapper.
    """
    if config is None:
        config = SyranConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = train_data.shape[1]
    chunk_size = config.effective_chunk_size(n_features)
    num_chunks = config.effective_num_chunks(n_features)

    variables = make_variables(n_features)
    index_by_name = {var.name: idx for idx, var in enumerate(variables)}

    rng = np.random.default_rng(config.seed)
    chunk_stream = generate_random_chunks(variables, chunk_size, rng=rng)

    results: List[ChunkResult] = []

    for chunk_index in range(num_chunks):
        chunk = next(chunk_stream)
        print(
            f"[{dataset_name}] chunk {chunk_index + 1}/{num_chunks} "
            f"using variables {', '.join(v.name for v in chunk)}"
        )

        values_dict = _build_value_dict(train_data, index_by_name, chunk)
        test_values_dict = _build_value_dict(test_data, index_by_name, chunk)
        random_values_dict = _build_random_value_dict(values_dict, rng)

        def objective(expr: Expression) -> float:
            return anomaly_objective(
                expr,
                values_dict=values_dict,
                random_values_dict=random_values_dict,
                variables=chunk,
                complexity_weight=config.complexity_weight,
                loss_bound=config.loss_bound,
            )

        def updater(o1: Expression, o2: Expression) -> Expression:
            return update_expression(
                o1,
                o2,
                variables=chunk,
                values_dict=values_dict,
                init_fn=init_random_expression,
                rng=rng,
            )

        try:
            sol, error, hist, mats = phase_search(
                objective,
                updater,
                init_random_expression,
                n=config.max_phase_iterations,
            )
        except ValueError as exc:
            print(f"Skipping chunk {chunk_index} due to ValueError: {exc}")
            continue

        sol = solve_expression(sol, values_dict)
        params = _extract_guess_parameters(sol)
        variable_names = [v.name for v in chunk]

        # Training scores
        train_scores = np.abs(sol(**values_dict) - 1.0)
        mean_train_score = float(np.mean(train_scores))
        alpha = solve_alpha(mean_train_score)

        # Testing scores (vectorised over samples)
        scores_raw = np.abs(
            np.array([
                sol(**{v.name: test_values_dict[v.name][i] for v in chunk})
                for i in range(test_data.shape[0])
            ]) - 1.0
        )
        test_scores = sigmoid(alpha * scores_raw)

        if not np.isfinite(test_scores).all():
            print(f"Skipping chunk {chunk_index}: non-finite scores")
            continue

        roc_auc = float(roc_auc_score(test_labels, test_scores))

        timestamp = time()
        chunk_id = f"{chunk_index:04d}_{int(timestamp)}"
        npz_path = output_dir / f"{chunk_id}.npz"

        np.savez_compressed(
            npz_path,
            train_scores=train_scores,
            test_scores=test_scores,
            test_labels=test_labels,
            variables=np.array(variable_names),
            symbolic_equation=str(sol),
            roc_auc_chunk=roc_auc,
            error=float(error),
            chunk_size=float(chunk_size),
            complexity_weight=float(config.complexity_weight),
            loss_bound=float(config.loss_bound),
            chunk_index=int(chunk_index),
            dataset_name=str(dataset_name),
            timestamp=float(timestamp),
            **params,
        )

        results.append(
            ChunkResult(
                dataset_name=dataset_name,
                chunk_index=chunk_index,
                variables=variable_names,
                symbolic_equation=str(sol),
                roc_auc=roc_auc,
                error=float(error),
                chunk_size=chunk_size,
                complexity_weight=config.complexity_weight,
                loss_bound=config.loss_bound,
                train_scores=train_scores,
                test_scores=test_scores,
                test_labels=test_labels,
                parameters=params,
                path=npz_path,
            )
        )

    return results


def run_kepler_toy_experiment(
    data: np.ndarray,
    variable_names: Sequence[str],
    true_solution: Expression,
    output_dir: Path,
    config: Optional[ToyKeplerConfig] = None,
) -> List[ToyChunkResult]:
    """Run the toy symbolic regression experiment (Kepler's law).

    ``data`` is a 2D array with shape (n_samples, n_features).
    ``variable_names`` must have length == n_features.
    ``true_solution`` is a morga expression constructed from the same variables.
    """
    if config is None:
        config = ToyKeplerConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = data.shape[1]
    if len(variable_names) != n_features:
        raise ValueError(
            f"Expected {n_features} variable names, got {len(variable_names)}"
        )

    variables = make_variables(n_features, names=variable_names)
    index_by_name = {var.name: idx for idx, var in enumerate(variables)}

    rng = np.random.default_rng(config.seed)
    chunk_size = min(config.chunk_size, n_features)
    chunk_stream = generate_random_chunks(variables, chunk_size, rng=rng)

    results: List[ToyChunkResult] = []

    for chunk_index in range(config.num_chunks):
        chunk = next(chunk_stream)
        print(
            f"[Kepler] chunk {chunk_index + 1}/{config.num_chunks} "
            f"using variables {', '.join(v.name for v in chunk)}"
        )

        values_dict = _build_value_dict(data, index_by_name, chunk)
        random_values_dict = _build_random_value_dict(values_dict, rng)

        def objective(expr: Expression) -> float:
            return kepler_objective(
                expr,
                values_dict=values_dict,
                random_values_dict=random_values_dict,
                variables=chunk,
                complexity_weight=config.complexity_weight,
                random_margin=config.random_margin,
            )

        def updater(o1: Expression, o2: Expression) -> Expression:
            return update_expression(
                o1,
                o2,
                variables=chunk,
                values_dict=values_dict,
                init_fn=init_random_expression,
                rng=rng,
            )

        try:
            sol, error, hist, mats = phase_search(
                objective,
                updater,
                init_random_expression,
                n=config.max_phase_iterations,
            )
        except ValueError as exc:
            print(f"Skipping chunk {chunk_index} due to ValueError: {exc}")
            continue

        sol = solve_expression(sol, values_dict)
        params = _extract_guess_parameters(sol)
        variable_names_chunk = [v.name for v in chunk]

        # Training scores (no held-out test set here)
        train_scores = np.abs(sol(**values_dict) - 1.0)

        objective_value = objective(sol)
        true_solution_objective = kepler_objective(
            true_solution,
            values_dict=values_dict,
            random_values_dict=random_values_dict,
            variables=chunk,
            complexity_weight=config.complexity_weight,
            random_margin=config.random_margin,
        )

        timestamp = time()
        chunk_id = f"{chunk_index:04d}_{int(timestamp)}"
        npz_path = output_dir / f"{chunk_id}.npz"

        np.savez_compressed(
            npz_path,
            train_scores=train_scores,
            variables=np.array(variable_names_chunk),
            symbolic_equation=str(sol),
            objective_value=float(objective_value),
            error=float(error),
            chunk_size=float(chunk_size),
            complexity_weight=float(config.complexity_weight),
            true_solution_objective=float(true_solution_objective),
            right=float(true_solution_objective),  # backwards compatibility
            chunk_index=int(chunk_index),
            timestamp=float(timestamp),
            **params,
        )

        results.append(
            ToyChunkResult(
                chunk_index=chunk_index,
                variables=variable_names_chunk,
                symbolic_equation=str(sol),
                objective_value=float(objective_value),
                error=float(error),
                chunk_size=chunk_size,
                complexity_weight=config.complexity_weight,
                true_solution_objective=float(true_solution_objective),
                train_scores=train_scores,
                parameters=params,
                path=npz_path,
            )
        )

    return results
