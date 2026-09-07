"""Generate the H=10 multiclass ablation confusion-matrix figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "outputs" / "Final_Grouped_H10_Ablation_HorizonFolds_5Fold_10Seeds"
FIGURES = ABLATION / "figures"
FINAL_MODEL = "Fusion-LSTM with derivatives"


def plot_matrix(matrix: np.ndarray, labels: list[str], title: str, output: Path) -> None:
    normalized = matrix / matrix.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    image = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=100)
    fig.colorbar(image, ax=ax, label="Row-normalized percentage (%)")
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{normalized[row, column]:.1f}%\n(n={matrix[row, column]:,})",
                ha="center",
                va="center",
                color="white" if normalized[row, column] < 35 else "black",
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(ABLATION / "out_of_fold_predictions.csv")
    data = data.loc[data["configuration"] == FINAL_MODEL]
    y_true = data["true_id"].to_numpy(dtype=int)
    y_pred = data["pred_id"].to_numpy(dtype=int)
    FIGURES.mkdir(parents=True, exist_ok=True)

    multiclass_labels = ["Good", "Burr", "Flash-burr", "Surface-groove/void"]
    multiclass = confusion_matrix(y_true, y_pred, labels=np.arange(4))
    plot_matrix(
        multiclass,
        multiclass_labels,
        "Internal H=10 grouped cross-validation - multiclass",
        FIGURES / "internal_h10_multiclass_confusion_matrix.png",
    )

    print(f"Saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
