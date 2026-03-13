"""Model definitions, fitting helpers, and text explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import RANDOM_STATE


def sparse_to_dense(matrix):
    """Convert sparse matrices for estimators that require dense arrays."""
    return matrix.toarray()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pipeline: Pipeline


def build_model_specs(random_state: int = RANDOM_STATE) -> list[ModelSpec]:
    vectorizer = dict(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
        strip_accents="unicode",
    )

    specs = [
        ModelSpec(
            name="Logistic Regression",
            pipeline=Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(**vectorizer)),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1500,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="Multinomial Naive Bayes",
            pipeline=Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(**vectorizer)),
                    ("classifier", MultinomialNB(alpha=0.5)),
                ]
            ),
        ),
        ModelSpec(
            name="Random Forest",
            pipeline=Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(**vectorizer)),
                    (
                        "densifier",
                        FunctionTransformer(sparse_to_dense, accept_sparse=True),
                    ),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=350,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]
    return specs


def cross_validate_models(
    specs: list[ModelSpec],
    X_train: pd.Series,
    y_train: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """Return comparable cross-validation metrics across models."""
    records: list[dict[str, float | str]] = []
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
    }

    for spec in specs:
        cv_results = cross_validate(
            clone(spec.pipeline),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        row: dict[str, float | str] = {"model": spec.name}
        for metric_name in scoring:
            row[f"cv_{metric_name}_mean"] = float(np.mean(cv_results[f"test_{metric_name}"]))
            row[f"cv_{metric_name}_std"] = float(np.std(cv_results[f"test_{metric_name}"]))
        records.append(row)

    return pd.DataFrame(records).sort_values("cv_f1_mean", ascending=False)


def fit_model_specs(
    specs: list[ModelSpec], X_train: pd.Series, y_train: pd.Series
) -> dict[str, Pipeline]:
    fitted: dict[str, Pipeline] = {}
    for spec in specs:
        model = clone(spec.pipeline)
        model.fit(X_train, y_train)
        fitted[spec.name] = model
    return fitted


def choose_deployment_model(evaluation_df: pd.DataFrame) -> str:
    """Prefer logistic regression when it is near the best model and easier to explain."""
    best_f1 = evaluation_df["f1"].max()
    logistic_row = evaluation_df.loc[
        evaluation_df["model"] == "Logistic Regression"
    ].iloc[0]
    if logistic_row["f1"] >= best_f1 - 0.02:
        return "Logistic Regression"
    return str(
        evaluation_df.sort_values(["f1", "average_precision"], ascending=False).iloc[0]["model"]
    )


def score_text(model: Pipeline, text: str) -> float:
    """Return spam probability for a single comment."""
    return float(model.predict_proba([text])[0, 1])


def explain_linear_prediction(
    model: Pipeline, text: str, top_n: int = 8
) -> pd.DataFrame:
    """Approximate token-level contribution for linear models."""
    classifier = model.named_steps.get("classifier")
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["term", "contribution"])

    vectorizer = model.named_steps["vectorizer"]
    matrix = vectorizer.transform([text])
    contributions = matrix.multiply(classifier.coef_[0]).toarray().ravel()
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = np.flatnonzero(contributions)
    if nonzero_indices.size == 0:
        return pd.DataFrame(columns=["term", "contribution"])

    explanation = pd.DataFrame(
        {
            "term": feature_names[nonzero_indices],
            "contribution": contributions[nonzero_indices],
        }
    )
    explanation["direction"] = np.where(
        explanation["contribution"] >= 0,
        "spam_signal",
        "ham_signal",
    )
    explanation["abs_contribution"] = explanation["contribution"].abs()
    return explanation.sort_values("abs_contribution", ascending=False).head(top_n)
