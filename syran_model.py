# syran_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np

from evolvDM.morga import munc, mvar, mguess
from evolvDM.mconst import mconst


Expression = Any  # symbolic expression type from morga
Variable = Any    # variable node type (mvar)
ValuesDict = Mapping[str, np.ndarray]


def make_variables(num_features: int, names: Sequence[str] | None = None) -> List[Variable]:
    """Create morga variables for the given number of features.

    If names are provided, they must have length == num_features.
    Otherwise, variables are named a, b, c, ... or x0, x1, ... if num_features > 26.
    """
    if num_features <= 0:
        raise ValueError("num_features must be positive")

    if names is not None:
        if len(names) != num_features:
            raise ValueError(
                f"Expected {num_features} names, got {len(names)}: {names}"
            )
        return [mvar(name) for name in names]

    if num_features <= 26:
        var_names = [chr(ord("a") + i) for i in range(num_features)]
    else:
        var_names = [f"x{i}" for i in range(num_features)]

    return [mvar(name) for name in var_names]


def generate_random_chunks(
    variables: Sequence[Variable],
    chunk_size: int,
    rng: np.random.Generator | None = None,
) -> Iterator[List[Variable]]:
    """Yield infinite stream of random subsets of variables of size ``chunk_size``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_size > len(variables):
        raise ValueError("chunk_size cannot exceed number of variables")

    if rng is None:
        rng = np.random.default_rng()

    indices = np.arange(len(variables))
    while True:
        chosen = rng.choice(indices, size=chunk_size, replace=False)
        yield [variables[i] for i in chosen]


def is_constant(expr: Expression, variables: Sequence[Variable]) -> bool:
    """Return True if ``expr`` does not depend on any variable in ``variables``."""
    if isinstance(expr, mconst) or isinstance(expr, mguess):
        return True

    variable_names = {v.name for v in variables}

    def contains_variable(node: Any) -> bool:
        if hasattr(node, "name") and node.name in variable_names:
            return True
        children = getattr(node, "children", None)
        if callable(children):
            return any(contains_variable(child) for child in children())
        return False

    return not contains_variable(expr)


def solve_expression(expr: Expression, values_dict: ValuesDict) -> Expression:
    """Solve for free parameters in ``expr`` using morga's ``solve``."""
    try:
        return expr.solve(1, **values_dict)
    except Exception:
        # If solving fails we fall back to the unsolved expression.
        return expr


def _complexity_penalty(expr: Expression, complexity_weight: float) -> float:
    """Compute the complexity regulariser for an expression."""
    try:
        complexity = float(expr.complexity())
    except Exception:
        complexity = 0.0

    complexity = max(complexity, 0.0)
    return float(complexity_weight * np.log1p(np.log1p(complexity)))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def anomaly_objective(
    expr: Expression,
    values_dict: ValuesDict,
    random_values_dict: ValuesDict,
    variables: Sequence[Variable],
    complexity_weight: float,
    loss_bound: float,
) -> float:
    """Objective used for symbolic anomaly detection.

    - penalises deviations from 1 on the training data (loss1)
    - encourages *large* deviations on random data via a hinge (loss2)
    - adds a complexity penalty to bias towards simpler expressions
    """
    expr = expr.simplify()
    if is_constant(expr, variables):
        return 1.0e10

    try:
        values = expr(**values_dict)
        random_values = expr(**random_values_dict)
        loss1 = float(np.mean(np.abs(values - 1.0)))
        loss2 = float(np.mean(np.abs(random_values - 1.0)))
        loss = loss1 + max(0.0, loss_bound - loss2)
        loss += _complexity_penalty(expr, complexity_weight)
        return float(loss)
    except Exception:
        return 1.0e10


def kepler_objective(
    expr: Expression,
    values_dict: ValuesDict,
    random_values_dict: ValuesDict,
    variables: Sequence[Variable],
    complexity_weight: float,
    random_margin: float = 2.0,
) -> float:
    """Objective used in the toy Kepler experiment:

        loss1 + max(0, random_margin - loss2_random) + complexity_penalty
    """
    expr = expr.simplify()
    if is_constant(expr, variables):
        return 1.0e10

    try:
        values = expr(**values_dict)
        random_values = expr(**random_values_dict)
        loss1 = float(np.mean(np.abs(values - 1.0)))
        loss2_random = float(np.mean(np.abs(random_values - 1.0)))
        loss = loss1 + max(0.0, random_margin - loss2_random)
        loss += _complexity_penalty(expr, complexity_weight)
        return float(loss)
    except Exception:
        return 1.0e10


def update_expression(
    o1: Expression,
    o2: Expression,
    variables: Sequence[Variable],
    values_dict: ValuesDict,
    init_fn: Callable[[], Expression],
    rng: np.random.Generator,
) -> Expression:
    """Evolutionary update step used by phase_search.

    With some probability we restart from a fresh random expression (to avoid
    local minima); otherwise we generate an offspring from two parents.
    """
    # Restart probability grows with the complexity
    try:
        complexity = float(o1.complexity())
    except Exception:
        complexity = 1.0
    complexity = max(complexity, 1.0)

    restart_prob = 0.2 * float(sigmoid(np.log(complexity)))
    if rng.random() < restart_prob:
        return solve_expression(init_fn(), values_dict)

    # Randomly swap parents.
    if rng.random() < 0.5:
        o1, o2 = o2, o1

    try:
        offspring = o1.offspring(o2)
        return solve_expression(offspring, values_dict)
    except ValueError:
        return mconst(1.0e10)


def init_random_expression() -> Expression:
    """Sample a random expression from the morga ``munc`` universe."""
    return munc().random_function()


def solve_alpha(mean_scores_training: float) -> float:
    """Compute the scaling parameter used before the sigmoid.

        alpha = 1 / mean_scores_training  (with a small fallback value)
    """
    if mean_scores_training <= 0.0:
        return 1.0e-6
    return float(1.0 / mean_scores_training)
