"""Validate the principal H=10 internal and external evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "outputs" / "Final_Grouped_H10_Ablation_HorizonFolds_5Fold_10Seeds"
FINAL_ANALYSIS = ABLATION / "Final_H10_Fusion_PerClass_Analysis"
BINARY_ANALYSIS = ABLATION / "Final_H10_Binary_Future_Weld_State"
EXTERNAL = ROOT / "external_test" / "FINAL_H10_ZERO_SHOT_COCO_EVALUATION_CORRECTED"
FINAL_MODEL = "Fusion-LSTM with derivatives"


def assert_close(actual: float, expected: float, label: str, atol: float = 1e-8) -> None:
    if not np.isclose(actual, expected, atol=atol):
        raise AssertionError(f"{label}: expected {expected}, obtained {actual}")


def validate_internal() -> None:
    predictions = pd.read_csv(ABLATION / "out_of_fold_predictions.csv")
    final = predictions.loc[predictions["configuration"] == FINAL_MODEL].copy()

    if len(final) != 25_810:
        raise AssertionError(f"Expected 25,810 final-model rows, obtained {len(final):,}")
    if final["seed"].nunique() != 10:
        raise AssertionError("Expected ten seeds")
    if not (final.groupby("seed").size() == 2_581).all():
        raise AssertionError("Each seed must contain 2,581 held-out predictions")

    multiclass_accuracy = []
    multiclass_macro_f1 = []

    for _, seed_data in final.groupby("seed", sort=True):
        y_true = seed_data["true_id"].to_numpy(dtype=int)
        y_pred = seed_data["pred_id"].to_numpy(dtype=int)
        multiclass_accuracy.append(100 * accuracy_score(y_true, y_pred))
        multiclass_macro_f1.append(100 * f1_score(y_true, y_pred, average="macro", zero_division=0))

    summary = pd.read_csv(ABLATION / "final_ablation_summary.csv")
    row = summary.loc[summary["configuration"] == FINAL_MODEL].iloc[0]
    assert_close(np.mean(multiclass_accuracy), row["accuracy_mean"], "Internal accuracy")
    assert_close(np.std(multiclass_accuracy, ddof=1), row["accuracy_sd"], "Internal accuracy SD")
    assert_close(np.mean(multiclass_macro_f1), row["macro_f1_mean"], "Internal macro F1")
    assert_close(np.std(multiclass_macro_f1, ddof=1), row["macro_f1_sd"], "Internal macro F1 SD")

    print("Internal evaluation: PASS")
    print(f"  H=10 sequences per seed: 2,581")
    print(f"  Ablation multiclass accuracy: {np.mean(multiclass_accuracy):.2f}%")
    print(f"  Ablation multiclass macro F1: {np.mean(multiclass_macro_f1):.2f}%")
    print("  Headline final-model values are validated from their separate result packages.")


def validate_final_reported_summary() -> None:
    summary = pd.read_csv(FINAL_ANALYSIS / "H10_overall_performance_summary.csv")
    expected = {
        "accuracy": (83.56838434715226, 3.743556613803165),
        "macro_precision": (84.24213832471476, 3.016370306488649),
        "macro_recall": (85.86317495361091, 3.316769937605331),
        "macro_f1": (84.2115329666464, 3.3388999760370006),
        "weighted_f1": (83.5148625827579, 3.6991266203043773),
    }
    if set(summary["metric"]) != set(expected):
        raise AssertionError("Unexpected metric set in final H=10 summary")
    for metric, (mean, sd) in expected.items():
        row = summary.loc[summary["metric"] == metric].iloc[0]
        assert_close(row["mean_percent"], mean, f"Final {metric} mean")
        assert_close(row["sd_percent"], sd, f"Final {metric} SD")

    per_class = pd.read_csv(FINAL_ANALYSIS / "H10_per_class_summary_10seeds.csv")
    consensus = pd.read_csv(FINAL_ANALYSIS / "H10_consensus_per_class_metrics.csv")
    if per_class["support"].sum() != 2_581 or consensus["support"].sum() != 2_581:
        raise AssertionError("Final per-class supports must sum to 2,581")
    if set(per_class["class_name"]) != {"Good", "Burr", "Flash-burr", "Surface-groove/void"}:
        raise AssertionError("Unexpected final per-class labels")

    print("Final reported ten-seed summary: PASS")
    print("  Multiclass accuracy: 83.57 ± 3.74%")
    print("  Multiclass macro F1: 84.21 ± 3.34%")


def validate_binary() -> None:
    predictions = pd.read_csv(BINARY_ANALYSIS / "H10_binary_all_10_seed_OOF_predictions.csv")
    if len(predictions) != 25_810:
        raise AssertionError(f"Expected 25,810 binary OOF rows, obtained {len(predictions):,}")
    if predictions["seed"].nunique() != 10 or not (predictions.groupby("seed").size() == 2_581).all():
        raise AssertionError("Binary OOF data must contain 2,581 rows for each of ten seeds")

    metric_rows = []
    for seed, seed_data in predictions.groupby("seed", sort=True):
        y_true = seed_data["true_binary"]
        y_pred = seed_data["pred_binary"]
        metric_rows.append(
            {
                "seed": seed,
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
        )
    recalculated = pd.DataFrame(metric_rows)
    reported = pd.read_csv(BINARY_ANALYSIS / "H10_binary_summary_10_seeds.csv").set_index("metric")
    assert_close(100 * recalculated["accuracy"].mean(), reported.loc["accuracy", "mean_percent"], "Final binary accuracy")
    assert_close(100 * recalculated["accuracy"].std(ddof=1), reported.loc["accuracy", "sd_percent"], "Final binary accuracy SD")
    assert_close(100 * recalculated["macro_f1"].mean(), reported.loc["macro_f1", "mean_percent"], "Final binary macro F1")
    assert_close(100 * recalculated["macro_f1"].std(ddof=1), reported.loc["macro_f1", "sd_percent"], "Final binary macro F1 SD")

    labels = ["Good", "Defect"]
    raw = confusion_matrix(predictions["true_binary"], predictions["pred_binary"], labels=labels)
    expected_raw = pd.read_csv(BINARY_ANALYSIS / "H10_binary_confusion_matrix_raw.csv", index_col=0).to_numpy(dtype=int)
    if not np.array_equal(raw, expected_raw):
        raise AssertionError("Final binary raw confusion matrix mismatch")

    normalized = raw / raw.sum(axis=1, keepdims=True) * 100
    expected_normalized = pd.read_csv(
        BINARY_ANALYSIS / "H10_binary_confusion_matrix_normalized_percent.csv", index_col=0
    ).to_numpy(dtype=float)
    if not np.allclose(normalized, expected_normalized):
        raise AssertionError("Final binary normalized confusion matrix mismatch")

    print("Final binary ten-seed evaluation: PASS")
    print(f"  Accuracy: {reported.loc['accuracy', 'mean_percent']:.2f} ± {reported.loc['accuracy', 'sd_percent']:.2f}%")
    print(f"  Macro F1: {reported.loc['macro_f1', 'mean_percent']:.2f} ± {reported.loc['macro_f1', 'sd_percent']:.2f}%")


def validate_external() -> None:
    matched = pd.read_csv(EXTERNAL / "FINAL_H10_ZERO_SHOT_MATCHED_896.csv")
    if len(matched) != 896:
        raise AssertionError(f"Expected 896 matched samples, obtained {len(matched)}")

    class_order = ["Good", "Burr", "Flash-burr", "Surface-groove/void"]
    binary_order = ["Good", "Defect"]
    multiclass = confusion_matrix(
        matched["true_class"], matched["pred_class"], labels=class_order
    )
    binary_pred = np.where(matched["pred_class"].eq("Good"), "Good", "Defect")
    binary = confusion_matrix(matched["binary_true"], binary_pred, labels=binary_order)

    expected_multiclass = pd.read_csv(
        EXTERNAL / "pooled_multiclass_confusion_matrix_RAW.csv", index_col=0
    ).to_numpy(dtype=int)
    expected_binary = pd.read_csv(
        EXTERNAL / "pooled_binary_confusion_matrix_RAW.csv", index_col=0
    ).to_numpy(dtype=int)

    if not np.array_equal(multiclass, expected_multiclass):
        raise AssertionError("External multiclass confusion matrix mismatch")
    if not np.array_equal(binary, expected_binary):
        raise AssertionError("External binary confusion matrix mismatch")

    summary = pd.read_csv(EXTERNAL / "FINAL_H10_ZERO_SHOT_PERFORMANCE_SUMMARY.csv")
    pooled = summary.loc[summary["experiment"] == "POOLED_EXP_12_13_14"].iloc[0]
    assert_close(100 * np.trace(multiclass) / multiclass.sum(), pooled["multiclass_accuracy"], "External multiclass accuracy")
    assert_close(100 * np.trace(binary) / binary.sum(), pooled["binary_accuracy"], "External binary accuracy")

    print("External zero-shot evaluation: PASS")
    print(f"  Matched samples: {len(matched)}")
    print(f"  Multiclass accuracy: {pooled['multiclass_accuracy']:.2f}%")
    print(f"  Multiclass macro F1: {pooled['multiclass_macro_f1']:.2f}%")
    print(f"  Binary accuracy: {pooled['binary_accuracy']:.2f}%")
    print(f"  Binary macro F1: {pooled['binary_macro_f1']:.2f}%")


if __name__ == "__main__":
    validate_internal()
    validate_final_reported_summary()
    validate_binary()
    validate_external()
    print("All repository result checks passed.")
