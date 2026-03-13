"""
charts.py
---------
Creates Plotly chart figures for quantitative comparison of
prompt variants: response length, sentiment polarity, token count.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.prompt_variants import get_category_color


# Shared color palette helper
def _bar_colors(df: pd.DataFrame) -> list:
    return [get_category_color(c) for c in df["Category"]]


def chart_response_length(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart: Word count per prompt variant.

    Args:
        df: Analysis DataFrame with 'Variant' and 'word_count' columns.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(go.Bar(
        x=df["Variant"],
        y=df["word_count"],
        marker_color=_bar_colors(df),
        text=df["word_count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Words: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Response Length (Word Count)",
        xaxis_title="Prompt Variant",
        yaxis_title="Word Count",
        xaxis_tickangle=-35,
        height=380,
        margin=dict(l=40, r=20, t=50, b=120),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        font=dict(size=11),
    )
    return fig


def chart_sentiment_polarity(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart: Sentiment polarity per prompt variant.
    Positive = green-ish, Negative = red-ish, encoded per value.

    Args:
        df: Analysis DataFrame with 'sentiment_polarity' column.

    Returns:
        Plotly Figure.
    """
    polarities = df["sentiment_polarity"]
    bar_colors = [
        "#10B981" if v > 0.05 else "#EF4444" if v < -0.05 else "#94A3B8"
        for v in polarities
    ]

    fig = go.Figure(go.Bar(
        x=df["Variant"],
        y=polarities,
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in polarities],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Polarity: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#64748B", line_width=1.5)
    fig.update_layout(
        title="Sentiment Polarity (–1 Negative → +1 Positive)",
        xaxis_title="Prompt Variant",
        yaxis_title="Polarity Score",
        xaxis_tickangle=-35,
        yaxis=dict(range=[-1.1, 1.1]),
        height=380,
        margin=dict(l=40, r=20, t=50, b=120),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        font=dict(size=11),
    )
    return fig


def chart_token_count(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart: Token count per prompt variant.

    Args:
        df: Analysis DataFrame with 'token_count' column.

    Returns:
        Plotly Figure.
    """
    df_sorted = df.sort_values("token_count", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted["token_count"],
        y=df_sorted["Variant"],
        orientation="h",
        marker_color=_bar_colors(df_sorted),
        text=df_sorted["token_count"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Tokens: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title="Token Count per Variant",
        xaxis_title="Token Count",
        yaxis_title="",
        height=380,
        margin=dict(l=20, r=60, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        font=dict(size=11),
    )
    return fig


def chart_subjectivity(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: Sentiment polarity vs subjectivity (bubble = word count).

    Args:
        df: Analysis DataFrame.

    Returns:
        Plotly Figure.
    """
    fig = px.scatter(
        df,
        x="sentiment_polarity",
        y="sentiment_subjectivity",
        size="word_count",
        color="Variant",
        hover_name="Variant",
        hover_data={"word_count": True, "sentiment_polarity": ":.3f",
                    "sentiment_subjectivity": ":.3f"},
        title="Polarity vs Subjectivity (bubble = word count)",
        labels={
            "sentiment_polarity":     "Polarity",
            "sentiment_subjectivity": "Subjectivity",
        },
        size_max=40,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        font=dict(size=11),
    )
    return fig
