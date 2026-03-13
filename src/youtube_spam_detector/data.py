"""Data loading and preprocessing utilities."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_VIDEO_MAP, RAW_DATA_DIR

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize comment text for feature extraction."""
    if text is None:
        return ""
    normalized = str(text).lower().strip()
    normalized = URL_RE.sub(" url ", normalized)
    normalized = NON_ALNUM_RE.sub(" ", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized)
    return normalized.strip()


def load_raw_comments(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and concatenate the UCI YouTube spam collection."""
    raw_dir = data_dir or RAW_DATA_DIR
    csv_paths = sorted(raw_dir.glob("Youtube*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No raw dataset files found in {raw_dir}. "
            "Download the UCI YouTube Spam Collection before training."
        )

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame["video_title"] = DATASET_VIDEO_MAP.get(path.name, path.stem)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean, analysis-ready dataframe."""
    prepared = df.copy()
    prepared["CONTENT"] = prepared["CONTENT"].fillna("")
    prepared["AUTHOR"] = prepared["AUTHOR"].fillna("Unknown")
    prepared["DATE"] = pd.to_datetime(prepared["DATE"], errors="coerce")
    prepared["content_clean"] = prepared["CONTENT"].map(clean_text)
    prepared["comment_length"] = prepared["CONTENT"].astype(str).str.len()
    prepared["token_count"] = (
        prepared["CONTENT"].astype(str).str.split().str.len().fillna(0).astype(int)
    )
    prepared["url_count"] = prepared["CONTENT"].astype(str).str.count(URL_RE)
    prepared["exclamation_count"] = prepared["CONTENT"].astype(str).str.count("!")
    prepared["uppercase_ratio"] = prepared["CONTENT"].astype(str).map(_uppercase_ratio)
    prepared["has_url"] = prepared["url_count"].gt(0).astype(int)
    prepared["label"] = prepared["CLASS"].map({0: "ham", 1: "spam"})
    return prepared


def get_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns used during modeling and analysis."""
    columns = [
        "COMMENT_ID",
        "AUTHOR",
        "DATE",
        "CONTENT",
        "CLASS",
        "label",
        "source_file",
        "video_title",
        "content_clean",
        "comment_length",
        "token_count",
        "url_count",
        "exclamation_count",
        "uppercase_ratio",
        "has_url",
    ]
    return df.loc[:, columns].copy()


def dataset_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize the corpus at a glance."""
    spam_rate = float(df["CLASS"].mean())
    return {
        "rows": int(len(df)),
        "videos": int(df["video_title"].nunique()),
        "spam_comments": int(df["CLASS"].sum()),
        "ham_comments": int((1 - df["CLASS"]).sum()),
        "spam_rate": round(spam_rate, 4),
        "mean_comment_length": round(float(df["comment_length"].mean()), 2),
        "median_comment_length": round(float(df["comment_length"].median()), 2),
        "comments_with_urls": int(df["has_url"].sum()),
    }


def _uppercase_ratio(text: str) -> float:
    letters = [char for char in str(text) if char.isalpha()]
    if not letters:
        return 0.0
    uppercase_count = sum(1 for char in letters if char.isupper())
    return uppercase_count / len(letters)

