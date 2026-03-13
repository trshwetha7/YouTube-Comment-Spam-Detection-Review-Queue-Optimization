"""Training and artifact generation entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CHARTS_DIR,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    DEFAULT_AUTO_REMOVE_THRESHOLD,
    MODEL_FILENAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SCREENSHOTS_DIR,
    TABLES_DIR,
    TEST_SIZE,
)
from .data import dataset_summary, get_model_frame, load_raw_comments, prepare_dataset
from .eda import class_balance_table, save_eda_charts, top_terms_by_label
from .evaluation import (
    collect_error_examples,
    error_summary_table,
    evaluate_predictions,
    save_performance_charts,
    threshold_tradeoff_table,
)
from .modeling import (
    build_model_specs,
    choose_deployment_model,
    cross_validate_models,
    fit_model_specs,
)
from .simulation import QueueThresholds, scenario_grid, simulate_review_queue


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))
    for directory in [PROCESSED_DATA_DIR, MODELS_DIR, CHARTS_DIR, TABLES_DIR, REPORTS_DIR, SCREENSHOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_comments()
    df = prepare_dataset(raw_df)
    model_df = get_model_frame(df)
    model_df.to_csv(PROCESSED_DATA_DIR / "youtube_spam_processed.csv", index=False)

    summary = dataset_summary(model_df)
    with open(REPORTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    train_df, test_df = train_test_split(
        model_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=model_df["CLASS"],
    )
    X_train = train_df["content_clean"]
    y_train = train_df["CLASS"]
    X_test = test_df["content_clean"]
    y_test = test_df["CLASS"]

    specs = build_model_specs()
    cv_df = cross_validate_models(specs, X_train, y_train)
    fitted_models = fit_model_specs(specs, X_train, y_train)

    evaluation_rows: list[dict[str, float | str]] = []
    probabilities: dict[str, np.ndarray] = {}
    for model_name, model in fitted_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        probabilities[model_name] = y_prob
        metrics = evaluate_predictions(y_test, y_prob, threshold=DEFAULT_DECISION_THRESHOLD)
        metrics["model"] = model_name
        evaluation_rows.append(metrics)
        joblib.dump(model, MODELS_DIR / MODEL_FILENAMES[model_name])

    evaluation_df = pd.DataFrame(evaluation_rows).sort_values(
        ["f1", "average_precision"], ascending=False
    )
    comparison_df = cv_df.merge(evaluation_df, on="model", how="left")
    comparison_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    deployment_model_name = choose_deployment_model(evaluation_df)
    deployment_model = fitted_models[deployment_model_name]
    deployment_prob = probabilities[deployment_model_name]
    joblib.dump(deployment_model, MODELS_DIR / "deployed_model.joblib")

    metadata = {
        "deployment_model": deployment_model_name,
        "decision_threshold": DEFAULT_DECISION_THRESHOLD,
        "review_threshold": DEFAULT_REVIEW_THRESHOLD,
        "auto_remove_threshold": DEFAULT_AUTO_REMOVE_THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "summary": summary,
    }
    with open(MODELS_DIR / "deployment_metadata.json", "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    threshold_df = threshold_tradeoff_table(y_test, deployment_prob)
    threshold_df.to_csv(TABLES_DIR / "threshold_tradeoffs.csv", index=False)
    prediction_export = test_df.copy()
    prediction_export["spam_probability"] = deployment_prob
    prediction_export.to_csv(TABLES_DIR / "deployment_test_predictions.csv", index=False)

    confusion = pd.DataFrame(
        [
            {
                "actual_ham_predicted_ham": int(
                    evaluation_df.loc[
                        evaluation_df["model"] == deployment_model_name, "true_negatives"
                    ].iloc[0]
                ),
                "actual_ham_predicted_spam": int(
                    evaluation_df.loc[
                        evaluation_df["model"] == deployment_model_name, "false_positives"
                    ].iloc[0]
                ),
                "actual_spam_predicted_ham": int(
                    evaluation_df.loc[
                        evaluation_df["model"] == deployment_model_name, "false_negatives"
                    ].iloc[0]
                ),
                "actual_spam_predicted_spam": int(
                    evaluation_df.loc[
                        evaluation_df["model"] == deployment_model_name, "true_positives"
                    ].iloc[0]
                ),
            }
        ]
    )
    confusion.to_csv(TABLES_DIR / "confusion_matrix.csv", index=False)

    default_queue = simulate_review_queue(
        y_test,
        deployment_prob,
        QueueThresholds(
            review_threshold=DEFAULT_REVIEW_THRESHOLD,
            auto_remove_threshold=DEFAULT_AUTO_REMOVE_THRESHOLD,
        ),
    )
    with open(REPORTS_DIR / "default_queue_simulation.json", "w", encoding="utf-8") as file_obj:
        json.dump(default_queue, file_obj, indent=2)

    scenario_df = scenario_grid(y_test, deployment_prob)
    scenario_df.to_csv(TABLES_DIR / "queue_scenarios.csv", index=False)

    errors_df = collect_error_examples(test_df, y_test, deployment_prob, DEFAULT_DECISION_THRESHOLD)
    errors_df.to_csv(TABLES_DIR / "error_analysis_examples.csv", index=False)
    error_summary_table(errors_df).to_csv(TABLES_DIR / "error_summary.csv", index=False)

    false_positives = errors_df.loc[errors_df["error_type"] == "false_positive"].copy()
    false_negatives = errors_df.loc[errors_df["error_type"] == "false_negative"].copy()
    false_positives.to_csv(TABLES_DIR / "false_positives.csv", index=False)
    false_negatives.to_csv(TABLES_DIR / "false_negatives.csv", index=False)

    top_spam_terms = top_terms_by_label(model_df, label_value=1, ngram_range=(1, 1), top_n=20)
    top_spam_phrases = top_terms_by_label(model_df, label_value=1, ngram_range=(2, 2), top_n=20)
    pd.concat([top_spam_terms, top_spam_phrases], ignore_index=True).to_csv(
        TABLES_DIR / "top_spam_language.csv",
        index=False,
    )
    class_balance_table(model_df).to_csv(TABLES_DIR / "class_balance.csv", index=False)

    save_eda_charts(model_df)
    save_performance_charts(
        y_test,
        deployment_prob,
        threshold_df,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )
    _save_scenario_chart(scenario_df)
    _create_screenshot_placeholders()

    findings = _build_key_findings(comparison_df, default_queue, errors_df, summary, deployment_model_name)
    with open(REPORTS_DIR / "key_findings.md", "w", encoding="utf-8") as file_obj:
        file_obj.write(findings)


def _save_scenario_chart(scenario_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    best = scenario_df.head(10).copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    scatter = ax.scatter(
        best["review_pct"] * 100,
        best["spam_caught_pct"] * 100,
        s=(best["auto_removed_pct"] * 1400) + 80,
        c=best["wrongful_auto_removals"],
        cmap="viridis_r",
        alpha=0.85,
    )
    for _, row in best.iterrows():
        ax.annotate(
            f"r={row['review_threshold']:.2f}\na={row['auto_remove_threshold']:.2f}",
            (row["review_pct"] * 100, row["spam_caught_pct"] * 100),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title("Queue Simulation Frontier")
    ax.set_xlabel("Comments Escalated to Human Review (%)")
    ax.set_ylabel("Spam Caught (%)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Wrongful Auto-Removals")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "queue_simulation_frontier.png", dpi=200)
    plt.close(fig)


def _create_screenshot_placeholders() -> None:
    import matplotlib.pyplot as plt

    placeholders = {
        "app_overview_placeholder.png": "Add Streamlit overview screenshot here",
        "threshold_simulator_placeholder.png": "Add threshold simulator screenshot here",
        "error_analysis_placeholder.png": "Add error analysis screenshot here",
    }
    for filename, label in placeholders.items():
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor("#f7f3ea")
        ax.set_facecolor("#16324f")
        ax.text(0.5, 0.58, label, ha="center", va="center", fontsize=20, color="white")
        ax.text(
            0.5,
            0.36,
            "Replace this placeholder with an app screenshot after local launch.",
            ha="center",
            va="center",
            fontsize=11,
            color="#d9e2ec",
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(SCREENSHOTS_DIR / filename, dpi=200)
        plt.close(fig)


def _build_key_findings(
    comparison_df: pd.DataFrame,
    default_queue: dict[str, float],
    errors_df: pd.DataFrame,
    summary: dict[str, float | int],
    deployment_model_name: str,
) -> str:
    top_model = comparison_df.sort_values("f1", ascending=False).iloc[0]
    fp_count = int((errors_df["error_type"] == "false_positive").sum())
    fn_count = int((errors_df["error_type"] == "false_negative").sum())
    return f"""# Key Findings

- Dataset coverage: {summary['rows']} comments across {summary['videos']} videos, with a spam rate of {summary['spam_rate']:.1%}.
- Best held-out F1: {top_model['model']} at {top_model['f1']:.3f}; deployed model: {deployment_model_name} for strong performance plus easier reviewer-facing explanations.
- Default queue policy (`review_threshold={default_queue['review_threshold']:.2f}`, `auto_remove_threshold={default_queue['auto_remove_threshold']:.2f}`) auto-removes {default_queue['auto_removed_pct']:.1%} of comments and sends {default_queue['review_pct']:.1%} to human review.
- Under that policy, the prototype catches {default_queue['spam_caught_pct']:.1%} of spam while keeping wrongful auto-removals to {default_queue['wrongful_auto_removals']} comments on the held-out set.
- Error analysis found {fp_count} false positives and {fn_count} false negatives at the default 0.50 threshold; borderline promotional language and short ambiguous comments remain the hardest cases.
"""


if __name__ == "__main__":
    main()
