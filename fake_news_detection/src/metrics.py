
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_proba: Optional[Sequence] = None,
    labels: Optional[List] = None,
    average: str = "weighted",
) -> Dict[str, float | list | dict]:
    """Return a dictionary of metrics.

    Parameters
    ----------
    y_true, y_pred : sequences of equal length
    y_proba        : predicted probabilities for positive class (binary).
                     If provided, ROC-AUC is computed.
    labels         : class label ordering for confusion matrix
    average        : 'weighted' | 'macro' | 'micro' | 'binary'
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
    )

    out: Dict[str, float | list | dict] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "labels": labels,
    }

    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = None

    return out


def print_metrics(metrics: Dict) -> None:
    """Pretty-print the metrics dict."""
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix:")
    for row in metrics["confusion_matrix"]:
        print("  ", row)


def confusion_plot(
    metrics: Dict,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    ax=None,
    cmap: str = "Blues",
):
    """Render a labelled confusion matrix with matplotlib."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array(metrics["confusion_matrix"])
    disp_labels = labels or metrics.get("labels") or list(range(cm.shape[0]))

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=disp_labels, yticklabels=disp_labels, ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def compare_models(results: Dict[str, Dict]) -> "pandas.DataFrame":  # noqa: F821
    """Collect metrics from multiple models into a comparison DataFrame."""
    import pandas as pd

    rows = []
    for name, m in results.items():
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m.get("roc_auc"),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
