"""Exploratory data analysis helpers and chart builders."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

from .config import CHARTS_DIR

plt.style.use("seaborn-v0_8-whitegrid")


def class_balance_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("label").size().rename("count").reset_index()
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts.sort_values("count", ascending=False)


def source_balance_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["video_title", "label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["video_title", "label"])
    )
    return grouped


def top_terms_by_label(
    df: pd.DataFrame,
    label_value: int = 1,
    ngram_range: tuple[int, int] = (1, 1),
    top_n: int = 15,
) -> pd.DataFrame:
    subset = df.loc[df["CLASS"] == label_value, "content_clean"]
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(subset)
    frequencies = matrix.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    results = pd.DataFrame({"term": feature_names, "count": frequencies})
    results["label"] = "spam" if label_value == 1 else "ham"
    results["ngram"] = f"{ngram_range[0]}-{ngram_range[1]}"
    return results.sort_values("count", ascending=False).head(top_n)


def save_eda_charts(df: pd.DataFrame, output_dir: Path | None = None) -> dict[str, str]:
    """Persist portfolio-friendly EDA charts."""
    output_dir = output_dir or CHARTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, str] = {}

    balance = class_balance_table(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(balance["label"], balance["count"], color=["#1d3557", "#e76f51"])
    ax.set_title("Class Balance: Spam vs Legitimate Comments")
    ax.set_xlabel("")
    ax.set_ylabel("Comment Count")
    for idx, value in enumerate(balance["count"]):
        ax.text(idx, value + 10, f"{value}", ha="center", fontsize=10)
    balance_path = output_dir / "class_balance.png"
    fig.tight_layout()
    fig.savefig(balance_path, dpi=200)
    plt.close(fig)
    chart_paths["class_balance"] = str(balance_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        [
            df.loc[df["CLASS"] == 0, "comment_length"],
            df.loc[df["CLASS"] == 1, "comment_length"],
        ],
        bins=30,
        label=["Ham", "Spam"],
        color=["#457b9d", "#f4a261"],
        alpha=0.85,
    )
    ax.set_title("Comment Length Distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Frequency")
    ax.legend()
    length_path = output_dir / "comment_length_distribution.png"
    fig.tight_layout()
    fig.savefig(length_path, dpi=200)
    plt.close(fig)
    chart_paths["comment_length_distribution"] = str(length_path)

    spam_terms = top_terms_by_label(df, label_value=1, ngram_range=(1, 1), top_n=15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        spam_terms["term"][::-1],
        spam_terms["count"][::-1],
        color="#e63946",
    )
    ax.set_title("Top Unigrams in Spam Comments")
    ax.set_xlabel("Frequency")
    spam_terms_path = output_dir / "top_spam_terms.png"
    fig.tight_layout()
    fig.savefig(spam_terms_path, dpi=200)
    plt.close(fig)
    chart_paths["top_spam_terms"] = str(spam_terms_path)

    source_counts = source_balance_table(df)
    fig = px.bar(
        source_counts,
        x="video_title",
        y="count",
        color="label",
        barmode="group",
        color_discrete_map={"ham": "#457b9d", "spam": "#e76f51"},
        title="Label Balance by Video",
    )
    video_path = output_dir / "label_balance_by_video.html"
    fig.write_html(video_path)
    chart_paths["label_balance_by_video_html"] = str(video_path)

    return chart_paths
