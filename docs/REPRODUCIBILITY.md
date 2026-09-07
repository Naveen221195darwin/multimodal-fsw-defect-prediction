# Reproducibility guide

## Environment

Python 3.11 is recommended. Install the dependencies with either:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate fsw-multimodal-prediction
```

## Recommended notebook order

1. `notebooks/01_horizon_selection/horizon_selection_study.ipynb`
2. `notebooks/02_ablation/h10_grouped_ablation.ipynb`
3. `notebooks/03_binary/h10_binary_future_weld_state.ipynb`
4. `notebooks/04_statistics/confidence_intervals.ipynb`
5. `notebooks/05_external_validation/h10_zero_shot_evaluation.ipynb`
6. `notebooks/06_diagnostics/h10_zero_shot_generalization_diagnostic.ipynb`

Launch Jupyter from the repository root. The notebooks use `Path.cwd()` as the project root.

## Included reproducibility levels

### Internal grouped evaluation

The processed frame-level H=10 dataset, fold plan, validation candidates, scaler tables,
training history, model-selection summary, out-of-fold predictions, final overall summary,
final per-class summary, consensus per-class table, and the complete final binary OOF package
are included. The full neural-network training notebooks are also included. Model checkpoints
were not saved by the study configuration (`SAVE_MODEL_CHECKPOINTS = False`).

### Independent zero-shot evaluation

The corrected 896-sample matched dataset, ground-truth table, confusion matrices,
performance summary, and figures are included. This is sufficient to reproduce the final
evaluation metrics.

The original COCO JSON and the three pre-matching prediction CSV files are not part of the
uploaded final release. Therefore, the annotation-to-prediction matching stage cannot be
repeated from raw inputs using this repository alone. The provided matched dataset is the
starting point for reproducing the released zero-shot metrics.

## Validation

```bash
python scripts/validate_results.py
python scripts/generate_internal_confusion_matrices.py
```

The validation script checks row counts, ten-seed OOF integrity, the ablation summary,
the 896-sample external evaluation, and both external confusion matrices.
