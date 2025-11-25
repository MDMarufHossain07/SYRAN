# SYRAN – SYmbolic Regression for unsupervised ANomaly detection

## File Overview

- **Model internals and objectives**: `syran_model.py`
- **Training loops and experiment logic**: `syran_training.py`
- **Evaluation**: `syran_evaluation.py`
- **Experiment entry points**: `run_benchmark.py`, `run_toy_kepler.py`

## Installation and dependencies

You need:

- Python 3.10+
- `numpy`
- `scikit-learn`
- `tqdm` (used inside `phase_search` if applicable)

## Data format

### Anomaly detection benchmark

Each dataset is stored in `data/<dataset>.npz` with keys:

- `x`: training data, shape `(n_train, n_features)`
- `tx`: test data, shape `(n_test, n_features)`
- `ty`: binary test labels, shape `(n_test,)`

### Kepler toy example

The toy data is stored in `toy_data/exoplanet_data.npz` with key:

- `data`: full dataset, shape `(n_samples, 2)`, columns `T` and `a`.


## Running experiments

Examples:

### 1. Anomaly detection benchmark (single dataset)

```bash
python run_benchmark.py \
  --dataset APima \
  --data_root data \
  --output_root results \
  --complexity_weight 0.1 \
  --loss_bound 1.0 \
  --chunk_size 2 \
  --num_chunks 20 \
  --max_phase_iterations 30000 \
  --seed 42
```

### 2. Kepler toy example

```bash
python run_toy_kepler.py \
  --data_path toy_data/exoplanet_data.npz \
  --output_root kepler_results \
  --complexity_weight 0.1 \
  --chunk_size 2 \
  --num_chunks 50 \
  --max_phase_iterations 100 \
  --seed 42
```