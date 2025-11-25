# !/usr/local/bin/python3
# run_toy_kepler.py
from __future__ import annotations

import argparse
import os
import sys
sys.path.append('./evolvDM')

import random
from pathlib import Path
from typing import Optional

import numpy as np

from evolvDM.morga import mvar

from syran_training import ToyKeplerConfig, run_kepler_toy_experiment



def _set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _load_exoplanet_data(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Toy data file not found: {path}")
    data = np.load(path)
    return data["data"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the toy Kepler symbolic regression experiment.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="toy_data/exoplanet_data.npz",
        help="Path to the toy exoplanet .npz file (default: ./exoplanet_data.npz).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="kepler_results",
        help="Directory where toy experiment results will be stored.",
    )
    parser.add_argument(
        "--complexity_weight",
        type=float,
        default=0.03,
        help="Weight for the complexity penalty (default: 0.03).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2,
        help="Number of variables per chunk (default: 2).",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=50,
        help="Number of chunks to optimise (default: 50).",
    )
    parser.add_argument(
        "--max_phase_iterations",
        type=int,
        default=100,
        help="Number of phase_search iterations per chunk (default: 30000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_root = Path(args.output_root)

    _set_global_seed(args.seed)

    data = _load_exoplanet_data(data_path)

    n_features = data.shape[1]
    if n_features != 2:
        raise ValueError(
            f"Toy Kepler experiment expects 2 features (T, a), got {n_features}"
        )

    # Construct variables T, a and the ground-truth solution T^2 / a^3
    variables = [mvar("T"), mvar("a")]
    T, a = variables
    true_solution = T**2 / a**3  # Kepler's third law

    config = ToyKeplerConfig(
        complexity_weight=args.complexity_weight,
        chunk_size=args.chunk_size,
        num_chunks=args.num_chunks,
        max_phase_iterations=args.max_phase_iterations,
        seed=args.seed,
    )

    results = run_kepler_toy_experiment(
        data=data,
        variable_names=["T", "a"],
        true_solution=true_solution,
        output_dir=output_root,
        config=config,
    )

    if not results:
        print("No successful chunks were produced.")
        return

    mean_obj = float(np.mean([r.objective_value for r in results]))
    best_obj = float(min(r.objective_value for r in results))
    print(f"\nFinished Kepler toy experiment: mean objective over {len(results)} chunks = {mean_obj:.4f}")
    print(f"Best (lowest) objective value = {best_obj:.4f}")


if __name__ == "__main__":
    main()
