"""Evaluation utilities for Trust & Safety analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import CHARTS_DIR

plt.style.use("seaborn-v0_8-whitegrid")


def evaluate_predictions(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def threshold_tradeoff_table(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.arange(0.1, 0.96, 0.05)
    rows: list[dict[str, float]] = []
    positive_count = int(np.sum(y_true))
    negative_count = int(len(y_true) - positive_count)

    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "false_positive_rate": fp / negative_count if negative_count else 0.0,
                "false_negative_rate": fn / positive_count if positive_count else 0.0,
                "comments_flagged_pct": float(np.mean(predictions)),
                "comments_flagged_count": int(np.sum(predictions)),
            }
        )

    return pd.DataFrame(rows)


def collect_error_examples(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    predictions = (y_prob >= threshold).astype(int)
    results = X_test.copy()
    results["actual"] = y_true.values
    results["predicted"] = predictions
    results["spam_probability"] = y_prob
    fp = results.loc[(results["actual"] == 0) & (results["predicted"] == 1)].copy()
    fp["error_type"] = "false_positive"
    fn = results.loc[(results["actual"] == 1) & (results["predicted"] == 0)].copy()
    fn["error_type"] = "false_negative"
    return pd.concat([fp, fn], ignore_index=True).sort_values(
        "spam_probability", ascending=False
    )


def error_summary_table(error_examples: pd.DataFrame) -> pd.DataFrame:
    if error_examples.empty:
        return pd.DataFrame(
            columns=["error_type", "count", "median_length", "url_share", "avg_probability"]
        )
    summary = (
        error_examples.groupby("error_type")
        .agg(
            count=("CONTENT", "size"),
            median_length=("comment_length", "median"),
            url_share=("has_url", "mean"),
            avg_probability=("spam_probability", "mean"),
        )
        .reset_index()
    )
    return summary


def save_performance_charts(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold_table: pd.DataFrame,
    threshold: float = 0.5,
    output_dir: Path | None = None,
) -> dict[str, str]:
    output_dir = output_dir or CHARTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, color="#e76f51", linewidth=2)
    ax.set_title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    pr_path = output_dir / "precision_recall_curve.png"
    fig.tight_layout()
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)
    paths["precision_recall_curve"] = str(pr_path)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, color="#457b9d", linewidth=2, label=f"ROC AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    roc_path = output_dir / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    paths["roc_curve"] = str(roc_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision", color="#1d3557")
    ax.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall", color="#e76f51")
    ax.plot(threshold_table["threshold"], threshold_table["f1"], label="F1", color="#2a9d8f")
    ax.set_title("Threshold Tradeoff Analysis")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    threshold_path = output_dir / "threshold_tradeoffs.png"
    fig.tight_layout()
    fig.savefig(threshold_path, dpi=200)
    plt.close(fig)
    paths["threshold_tradeoffs"] = str(threshold_path)

    predictions = (y_prob >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Ham", "Spam"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix @ Threshold {threshold:.2f}")
    cm_path = output_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    paths["confusion_matrix"] = str(cm_path)

    return paths
