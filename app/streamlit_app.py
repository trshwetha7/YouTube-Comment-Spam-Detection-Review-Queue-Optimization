"""Streamlit app for YouTube spam review queue optimization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from youtube_spam_detector.modeling import explain_linear_prediction, score_text
from youtube_spam_detector.simulation import QueueThresholds, simulate_review_queue

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "youtube_spam_processed.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "deployed_model.joblib"
METADATA_FILE = PROJECT_ROOT / "models" / "deployment_metadata.json"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

PALETTE = {
    "navy": "#16324f",
    "sand": "#f7f3ea",
    "coral": "#e76f51",
    "teal": "#2a9d8f",
    "slate": "#577590",
    "gold": "#e9c46a",
}


@st.cache_data(show_spinner=False)
def load_project_data() -> dict[str, pd.DataFrame | dict]:
    if not DATA_FILE.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Artifacts are missing. Run `PYTHONPATH=src python3 -m youtube_spam_detector.train` first."
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)

    return {
        "dataset": pd.read_csv(DATA_FILE, parse_dates=["DATE"]),
        "comparison": pd.read_csv(TABLES_DIR / "model_comparison.csv"),
        "confusion": pd.read_csv(TABLES_DIR / "confusion_matrix.csv"),
        "thresholds": pd.read_csv(TABLES_DIR / "threshold_tradeoffs.csv"),
        "scenarios": pd.read_csv(TABLES_DIR / "queue_scenarios.csv"),
        "test_predictions": pd.read_csv(TABLES_DIR / "deployment_test_predictions.csv"),
        "top_language": pd.read_csv(TABLES_DIR / "top_spam_language.csv"),
        "false_positives": pd.read_csv(TABLES_DIR / "false_positives.csv"),
        "false_negatives": pd.read_csv(TABLES_DIR / "false_negatives.csv"),
        "error_summary": pd.read_csv(TABLES_DIR / "error_summary.csv"),
        "metadata": metadata,
    }


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_FILE)


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_app() -> None:
    st.set_page_config(
        page_title="YouTube Comment Spam Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, {PALETTE['sand']} 0%, #fffdf8 100%);
            }}
            .hero {{
                padding: 1.4rem 1.6rem;
                border-radius: 20px;
                background: linear-gradient(135deg, {PALETTE['navy']} 0%, #204b69 65%, {PALETTE['teal']} 100%);
                color: white;
                margin-bottom: 1rem;
                box-shadow: 0 18px 40px rgba(22, 50, 79, 0.18);
            }}
            .hero h1 {{
                margin: 0;
                font-size: 2.2rem;
            }}
            .hero p {{
                margin: 0.45rem 0 0;
                color: #edf6f9;
                font-size: 1rem;
            }}
            .metric-card {{
                background: white;
                border: 1px solid rgba(22, 50, 79, 0.08);
                border-radius: 16px;
                padding: 0.9rem 1rem;
                box-shadow: 0 8px 24px rgba(22, 50, 79, 0.08);
                min-height: 120px;
            }}
            .metric-label {{
                color: #4f5d75;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .metric-value {{
                color: {PALETTE['navy']};
                font-size: 1.9rem;
                font-weight: 700;
                margin: 0.15rem 0 0.25rem;
            }}
            .metric-help {{
                color: #6c757d;
                font-size: 0.86rem;
            }}
            .section-caption {{
                color: #4f5d75;
                margin-top: -0.35rem;
                margin-bottom: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def overview_tab(data: dict[str, pd.DataFrame | dict]) -> None:
    dataset = data["dataset"]
    comparison = data["comparison"]
    metadata = data["metadata"]
    champion = metadata["deployment_model"]
    with open(PROJECT_ROOT / "outputs" / "reports" / "default_queue_simulation.json", "r", encoding="utf-8") as file_obj:
        workload = json.load(file_obj)

    st.markdown(
        """
        <div class="hero">
            <h1>YouTube Comment Spam Detection & Review Queue Optimization</h1>
            <p>
                Prototype Trust & Safety decision-support system built on the public UCI YouTube Spam Collection.
                The goal is reviewer triage, threshold selection, and explainable moderation analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Comments", f"{len(dataset):,}", "1,956 labeled comments across 5 videos.")
    with col2:
        metric_card("Spam Rate", f"{dataset['CLASS'].mean():.1%}", "Public dataset class balance.")
    with col3:
        metric_card("Deployed Model", champion, "Selected for strong performance and operational clarity.")
    with col4:
        metric_card("Review Load", f"{workload['review_pct']:.1%}", "Share of comments routed to human review.")

    st.markdown("### Business framing")
    st.write(
        "This project is framed as a moderation triage prototype rather than a production moderation system. "
        "The objective is to balance spam capture against reviewer workload and user harm from wrongful removals."
    )

    summary_cols = st.columns([1.1, 1])
    with summary_cols[0]:
        leaderboard = comparison[
            ["model", "precision", "recall", "f1", "average_precision", "roc_auc"]
        ].sort_values("f1", ascending=False)
        st.markdown("### Model leaderboard")
        st.dataframe(
            leaderboard.style.format(
                {
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1": "{:.3f}",
                    "average_precision": "{:.3f}",
                    "roc_auc": "{:.3f}",
                }
            ),
            use_container_width=True,
        )
    with summary_cols[1]:
        by_video = (
            dataset.groupby(["video_title", "label"])
            .size()
            .rename("count")
            .reset_index()
        )
        fig = px.bar(
            by_video,
            x="video_title",
            y="count",
            color="label",
            barmode="group",
            color_discrete_map={"ham": PALETTE["slate"], "spam": PALETTE["coral"]},
            height=420,
        )
        fig.update_layout(
            title="Label balance by video",
            margin=dict(l=10, r=10, t=45, b=10),
            legend_title="Class",
        )
        st.plotly_chart(fig, use_container_width=True)


def data_exploration_tab(data: dict[str, pd.DataFrame | dict]) -> None:
    dataset = data["dataset"]
    top_language = data["top_language"]

    st.markdown("### Data exploration")
    st.caption("Comment length, source distribution, and the language patterns that make spam detection difficult.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            dataset,
            x="comment_length",
            color="label",
            nbins=35,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={"ham": PALETTE["slate"], "spam": PALETTE["coral"]},
        )
        fig.update_layout(title="Comment length distribution", margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        grouped = dataset.groupby("label").agg(
            median_length=("comment_length", "median"),
            mean_tokens=("token_count", "mean"),
            url_share=("has_url", "mean"),
        )
        st.dataframe(
            grouped.style.format(
                {"median_length": "{:.0f}", "mean_tokens": "{:.1f}", "url_share": "{:.1%}"}
            ),
            use_container_width=True,
        )

    unigram = top_language[top_language["ngram"] == "1-1"].head(15)
    bigram = top_language[top_language["ngram"] == "2-2"].head(15)
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        fig = px.bar(
            unigram.sort_values("count"),
            x="count",
            y="term",
            orientation="h",
            color_discrete_sequence=[PALETTE["coral"]],
        )
        fig.update_layout(title="Top spam unigrams", margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with lang_col2:
        fig = px.bar(
            bigram.sort_values("count"),
            x="count",
            y="term",
            orientation="h",
            color_discrete_sequence=[PALETTE["gold"]],
        )
        fig.update_layout(title="Top spam phrases", margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)


def model_performance_tab(data: dict[str, pd.DataFrame | dict]) -> None:
    comparison = data["comparison"].copy()
    thresholds = data["thresholds"]
    confusion = data["confusion"].iloc[0]

    st.markdown("### Model performance")
    st.caption("Trust & Safety metrics emphasize precision/recall tradeoffs rather than accuracy alone.")

    perf_fig = px.scatter(
        comparison,
        x="recall",
        y="precision",
        size="f1",
        color="model",
        hover_data=["average_precision", "roc_auc"],
        color_discrete_sequence=[PALETTE["navy"], PALETTE["teal"], PALETTE["coral"]],
    )
    perf_fig.update_layout(title="Held-out precision vs recall", margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(perf_fig, use_container_width=True)

    threshold_fig = go.Figure()
    for column, color in [
        ("precision", PALETTE["navy"]),
        ("recall", PALETTE["coral"]),
        ("f1", PALETTE["teal"]),
    ]:
        threshold_fig.add_trace(
            go.Scatter(
                x=thresholds["threshold"],
                y=thresholds[column],
                mode="lines+markers",
                name=column.title(),
                line=dict(color=color, width=3),
            )
        )
    threshold_fig.update_layout(
        title="Precision, recall, and F1 by threshold",
        xaxis_title="Decision threshold",
        yaxis_title="Score",
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(threshold_fig, use_container_width=True)

    confusion_matrix_fig = go.Figure(
        data=go.Heatmap(
            z=[
                [confusion["actual_ham_predicted_ham"], confusion["actual_ham_predicted_spam"]],
                [confusion["actual_spam_predicted_ham"], confusion["actual_spam_predicted_spam"]],
            ],
            x=["Predicted Ham", "Predicted Spam"],
            y=["Actual Ham", "Actual Spam"],
            colorscale="Blues",
            text=[
                [confusion["actual_ham_predicted_ham"], confusion["actual_ham_predicted_spam"]],
                [confusion["actual_spam_predicted_ham"], confusion["actual_spam_predicted_spam"]],
            ],
            texttemplate="%{text}",
        )
    )
    confusion_matrix_fig.update_layout(
        title="Confusion matrix at threshold 0.50",
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(confusion_matrix_fig, use_container_width=True)

    st.markdown("### Why precision vs recall matters")
    st.write(
        "High precision limits wrongful takedowns of legitimate users. High recall reduces spam leakage, which matters for abuse prevention and creator experience. "
        "A Trust & Safety team usually does not want a single global threshold; it wants a policy that routes the riskiest content to automation, the uncertain middle to reviewers, and low-risk comments to allow."
    )


def threshold_simulator_tab(data: dict[str, pd.DataFrame | dict]) -> None:
    scenarios = data["scenarios"]
    dataset = data["dataset"]
    heldout = data["test_predictions"]
    model = load_model()

    st.markdown("### Threshold simulator")
    st.caption("Tune triage thresholds and observe the operational tradeoff between spam capture and reviewer burden.")

    left, right = st.columns([0.9, 1.1])
    with left:
        comment = st.text_area(
            "Score a fresh comment",
            value="Check out my channel and subscribe for more giveaways!",
            height=120,
        )
        probability = score_text(model, comment)
        st.metric("Spam probability", f"{probability:.1%}")
        explanation = explain_linear_prediction(model, comment)
        if not explanation.empty:
            st.markdown("Top feature contributions")
            st.dataframe(
                explanation[["term", "direction", "contribution"]].style.format(
                    {"contribution": "{:.3f}"}
                ),
                use_container_width=True,
            )

    with right:
        review_threshold = st.slider("Review threshold", 0.2, 0.85, 0.55, 0.05)
        auto_remove_threshold = st.slider("Auto-remove threshold", 0.4, 0.99, 0.85, 0.05)
        if auto_remove_threshold <= review_threshold:
            st.error("Auto-remove threshold must be greater than review threshold.")
            return

        row = simulate_review_queue(
            heldout["CLASS"],
            heldout["spam_probability"].to_numpy(),
            QueueThresholds(
                review_threshold=review_threshold,
                auto_remove_threshold=auto_remove_threshold,
            ),
        )
        cols = st.columns(3)
        cols[0].metric("Auto-remove", f"{row['auto_removed_pct']:.1%}")
        cols[1].metric("Review queue", f"{row['review_pct']:.1%}")
        cols[2].metric("Spam caught", f"{row['spam_caught_pct']:.1%}")
        st.progress(min(float(row["spam_caught_pct"]), 1.0), text="Share of spam caught")
        st.write(
            f"Estimated reviewer workload: {row['reviewer_workload_per_1000_comments']:.0f} comments per 1,000 incoming comments. "
            f"Wrongful auto-removals on the held-out set: {int(row['wrongful_auto_removals'])}. "
            f"Legitimate comments escalated to review: {int(row['legitimate_comments_wrongly_escalated'])}."
        )


def review_queue_dashboard_tab(data: dict[str, pd.DataFrame | dict]) -> None:
    scenarios = data["scenarios"]
    metadata = data["metadata"]

    st.markdown("### Review queue dashboard")
    st.caption("Operational scenarios for routing comments into allow, review, or auto-remove buckets.")

    frontier = px.scatter(
        scenarios,
        x=scenarios["review_pct"] * 100,
        y=scenarios["spam_caught_pct"] * 100,
        size=scenarios["auto_removed_pct"] * 100,
        color="wrongful_auto_removals",
        hover_data=["review_threshold", "auto_remove_threshold", "review_queue_spam_rate"],
        color_continuous_scale="YlOrRd_r",
    )
    frontier.update_layout(
        title="Queue simulation frontier",
        xaxis_title="Comments sent to human review (%)",
        yaxis_title="Spam caught (%)",
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(frontier, use_container_width=True)

    recommendation = scenarios.iloc[0]
    rec_cols = st.columns(4)
    rec_cols[0].metric("Recommended review threshold", f"{recommendation['review_threshold']:.2f}")
    rec_cols[1].metric("Recommended auto-remove threshold", f"{recommendation['auto_remove_threshold']:.2f}")
    rec_cols[2].metric("Review queue spam rate", f"{recommendation['review_queue_spam_rate']:.1%}")
    rec_cols[3].metric("Wrongful auto-removals", f"{int(recommendation['wrongful_auto_removals'])}")

    st.write(
        f"Default policy from the training artifacts uses review threshold `{metadata['review_threshold']:.2f}` "
        f"and auto-remove threshold `{metadata['auto_remove_threshold']:.2f}`. "
        "Teams can move left on the frontier to reduce reviewer burden, or move up to catch more spam at the cost of more escalations."
    )
    st.dataframe(
        scenarios.head(12).style.format(
            {
                "auto_removed_pct": "{:.1%}",
                "review_pct": "{:.1%}",
                "allowed_pct": "{:.1%}",
                "spam_caught_pct": "{:.1%}",
                "auto_remove_precision": "{:.1%}",
                "review_queue_spam_rate": "{:.1%}",
            }
        ),
        use_container_width=True,
    )


def error_analysis_tab(data: dict[str, pd.DataFrame | dict]) -> None:
    fps = data["false_positives"]
    fns = data["false_negatives"]
    error_summary = data["error_summary"]
    dataset = data["dataset"]

    st.markdown("### Error analysis")
    st.caption("False positives and false negatives reveal the policy risks behind aggregate metrics.")

    col1, col2 = st.columns(2)
    with col1:
        metric_card("False positives", f"{len(fps)}", "Legitimate comments flagged as spam.")
    with col2:
        metric_card("False negatives", f"{len(fns)}", "Spam comments that slip through.")

    st.dataframe(
        error_summary.style.format({"median_length": "{:.0f}", "url_share": "{:.1%}", "avg_probability": "{:.2f}"}),
        use_container_width=True,
    )

    sample_cols = st.columns(2)
    with sample_cols[0]:
        st.markdown("#### False positives")
        st.dataframe(
            fps[["CONTENT", "video_title", "spam_probability"]].head(12).style.format(
                {"spam_probability": "{:.2%}"}
            ),
            use_container_width=True,
            height=420,
        )
    with sample_cols[1]:
        st.markdown("#### False negatives")
        st.dataframe(
            fns[["CONTENT", "video_title", "spam_probability"]].head(12).style.format(
                {"spam_probability": "{:.2%}"}
            ),
            use_container_width=True,
            height=420,
        )

    st.write(
        "Observed failure modes usually fall into two buckets: borderline self-promotion that resembles harmless fan engagement, and short spam comments that lack obvious giveaways such as URLs. "
        "In a real Trust & Safety workflow, these are exactly the cases worth sending to human review rather than forcing a binary model-only decision."
    )


def live_simulation_tab(data: dict[str, pd.DataFrame | dict]) -> None:
    model = load_model()
    dataset = data["dataset"]

    st.markdown("### Live comment scoring")
    st.caption("Paste a comment to get a spam probability and an explanation from the deployed model.")

    comment = st.text_area("Comment text", height=150)
    if comment:
        probability = score_text(model, comment)
        thresholds = QueueThresholds(
            review_threshold=data["metadata"]["review_threshold"],
            auto_remove_threshold=data["metadata"]["auto_remove_threshold"],
        )
        queue_decision = (
            "Auto-remove" if probability >= thresholds.auto_remove_threshold else
            "Send to human review" if probability >= thresholds.review_threshold else
            "Allow"
        )
        cols = st.columns(3)
        cols[0].metric("Spam probability", f"{probability:.1%}")
        cols[1].metric("Queue action", queue_decision)
        cols[2].metric("Benchmark spam rate", f"{dataset['CLASS'].mean():.1%}")

        explanation = explain_linear_prediction(model, comment)
        if explanation.empty:
            st.info("Feature attribution is only available for linear models.")
        else:
            fig = px.bar(
                explanation.sort_values("contribution"),
                x="contribution",
                y="term",
                color="direction",
                orientation="h",
                color_discrete_map={
                    "spam_signal": PALETTE["coral"],
                    "ham_signal": PALETTE["teal"],
                },
            )
            fig.update_layout(title="Approximate token contribution", margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    style_app()
    st.sidebar.title("Trust & Safety Prototype")
    st.sidebar.write(
        "Built on the public UCI YouTube Spam Collection. "
        "This is a reviewer triage prototype, not production YouTube moderation."
    )

    try:
        data = load_project_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    tabs = st.tabs(
        [
            "Overview",
            "Data Exploration",
            "Model Performance",
            "Threshold Simulator",
            "Review Queue Dashboard",
            "Error Analysis",
            "Live Comment Scoring",
        ]
    )

    with tabs[0]:
        overview_tab(data)
    with tabs[1]:
        data_exploration_tab(data)
    with tabs[2]:
        model_performance_tab(data)
    with tabs[3]:
        threshold_simulator_tab(data)
    with tabs[4]:
        review_queue_dashboard_tab(data)
    with tabs[5]:
        error_analysis_tab(data)
    with tabs[6]:
        live_simulation_tab(data)


if __name__ == "__main__":
    main()
