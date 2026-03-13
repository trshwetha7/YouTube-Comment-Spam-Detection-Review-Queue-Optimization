"""Optional public YouTube Data API scoring helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
from sklearn.pipeline import Pipeline

from .data import clean_text
from .modeling import score_text

YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def fetch_public_comments(video_id: str, api_key: str, max_results: int = 50) -> pd.DataFrame:
    """Fetch public top-level comments for a video.

    Public API access does not expose moderation labels for arbitrary channels.
    This function is included for prototype scoring only.
    """
    params: dict[str, Any] = {
        "part": "snippet",
        "videoId": video_id,
        "key": api_key,
        "textFormat": "plainText",
        "maxResults": max_results,
        "order": "relevance",
    }
    response = requests.get(YOUTUBE_COMMENTS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    records = []
    for item in payload.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        records.append(
            {
                "author": snippet.get("authorDisplayName"),
                "content": snippet.get("textDisplay", ""),
                "published_at": snippet.get("publishedAt"),
                "like_count": snippet.get("likeCount", 0),
            }
        )
    return pd.DataFrame(records)


def score_public_comments(model: Pipeline, comments: pd.DataFrame) -> pd.DataFrame:
    scored = comments.copy()
    scored["content_clean"] = scored["content"].fillna("").map(clean_text)
    scored["spam_probability"] = scored["content_clean"].map(lambda text: score_text(model, text))
    return scored.sort_values("spam_probability", ascending=False)
