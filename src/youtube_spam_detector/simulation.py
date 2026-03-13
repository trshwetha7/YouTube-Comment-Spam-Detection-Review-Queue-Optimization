"""Reviewer queue simulation logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_AUTO_REMOVE_THRESHOLD, DEFAULT_REVIEW_THRESHOLD


@dataclass(frozen=True)
class QueueThresholds:
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD
    auto_remove_threshold: float = DEFAULT_AUTO_REMOVE_THRESHOLD

    def __post_init__(self) -> None:
        if self.auto_remove_threshold <= self.review_threshold:
            raise ValueError("auto_remove_threshold must be greater than review_threshold.")


def assign_queue_action(probability: float, thresholds: QueueThresholds) -> str:
    if probability >= thresholds.auto_remove_threshold:
        return "auto_remove"
    if probability >= thresholds.review_threshold:
        return "human_review"
    return "allow"


def simulate_review_queue(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: QueueThresholds,
    volume_projection: int = 1000,
) -> dict[str, float]:
    actions = np.array([assign_queue_action(prob, thresholds) for prob in y_prob])
    total = len(actions)
    total_spam = int(np.sum(y_true))
    total_ham = total - total_spam

    auto_mask = actions == "auto_remove"
    review_mask = actions == "human_review"
    allow_mask = actions == "allow"

    auto_count = int(np.sum(auto_mask))
    review_count = int(np.sum(review_mask))
    allow_count = int(np.sum(allow_mask))

    auto_spam = int(np.sum(y_true[auto_mask] == 1))
    auto_ham = int(np.sum(y_true[auto_mask] == 0))
    review_spam = int(np.sum(y_true[review_mask] == 1))
    review_ham = int(np.sum(y_true[review_mask] == 0))
    allow_spam = int(np.sum(y_true[allow_mask] == 1))

    spam_caught = auto_spam + review_spam
    return {
        "review_threshold": thresholds.review_threshold,
        "auto_remove_threshold": thresholds.auto_remove_threshold,
        "auto_removed_pct": auto_count / total,
        "review_pct": review_count / total,
        "allowed_pct": allow_count / total,
        "auto_removed_count": auto_count,
        "review_count": review_count,
        "allowed_count": allow_count,
        "reviewer_workload_per_1000_comments": review_count / total * volume_projection,
        "spam_caught_pct": spam_caught / total_spam if total_spam else 0.0,
        "spam_auto_removed_pct": auto_spam / total_spam if total_spam else 0.0,
        "legitimate_comments_wrongly_escalated": review_ham,
        "wrongful_auto_removals": auto_ham,
        "spam_leaked_to_allow": allow_spam,
        "auto_remove_precision": auto_spam / auto_count if auto_count else 0.0,
        "review_queue_spam_rate": review_spam / review_count if review_count else 0.0,
        "allowed_spam_rate": allow_spam / allow_count if allow_count else 0.0,
        "total_ham": total_ham,
        "total_spam": total_spam,
    }


def scenario_grid(
    y_true: pd.Series,
    y_prob: np.ndarray,
    review_thresholds: list[float] | None = None,
    auto_remove_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    review_thresholds = review_thresholds or [0.4, 0.5, 0.55, 0.6, 0.65]
    auto_remove_thresholds = auto_remove_thresholds or [0.75, 0.8, 0.85, 0.9]
    rows: list[dict[str, float]] = []
    for review_threshold in review_thresholds:
        for auto_threshold in auto_remove_thresholds:
            if auto_threshold <= review_threshold:
                continue
            thresholds = QueueThresholds(
                review_threshold=review_threshold,
                auto_remove_threshold=auto_threshold,
            )
            rows.append(simulate_review_queue(y_true, y_prob, thresholds))
    return pd.DataFrame(rows).sort_values(
        ["spam_caught_pct", "review_pct", "wrongful_auto_removals"],
        ascending=[False, True, True],
    )
