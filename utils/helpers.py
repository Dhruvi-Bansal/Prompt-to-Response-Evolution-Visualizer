"""
helpers.py
----------
Miscellaneous utility functions used across the application.
"""

import re
from typing import List, Dict
import pandas as pd


def truncate_text(text: str, max_chars: int = 120) -> str:
    """
    Truncate a string to `max_chars` characters and append ellipsis.

    Args:
        text:      Input string.
        max_chars: Maximum character length before truncation.

    Returns:
        Possibly-truncated string.
    """
    return text[:max_chars] + "…" if len(text) > max_chars else text


def sanitize_prompt(prompt: str) -> str:
    """
    Strip leading/trailing whitespace and collapse internal whitespace runs.

    Args:
        prompt: Raw user input string.

    Returns:
        Cleaned prompt string.
    """
    return re.sub(r"\s+", " ", prompt.strip())


def category_badge_html(category: str) -> str:
    """
    Return an HTML span styled as a coloured badge for the category label.

    Args:
        category: Variant category string.

    Returns:
        HTML string for rendering in st.markdown with unsafe_allow_html=True.
    """
    color_map = {
        "original":   ("#EEF2FF", "#4F46E5"),
        "constraint": ("#ECFEFF", "#0891B2"),
        "example":    ("#ECFDF5", "#059669"),
        "tone":       ("#FFFBEB", "#D97706"),
        "detail":     ("#FEF2F2", "#DC2626"),
    }
    bg, fg = color_map.get(category, ("#F3F4F6", "#6B7280"))
    label = category.capitalize()
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:9999px;font-size:11px;font-weight:600;">'
        f'{label}</span>'
    )


def sentiment_emoji(polarity: float) -> str:
    """
    Map a sentiment polarity float to a descriptive emoji + label.

    Args:
        polarity: Float in [-1, 1].

    Returns:
        Emoji + label string.
    """
    if polarity > 0.3:
        return "😊 Very Positive"
    elif polarity > 0.05:
        return "🙂 Positive"
    elif polarity < -0.3:
        return "😠 Very Negative"
    elif polarity < -0.05:
        return "😕 Negative"
    else:
        return "😐 Neutral"


def df_to_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a user-friendly subset of the analysis DataFrame for display.

    Args:
        df: Full analysis DataFrame.

    Returns:
        Slimmed-down DataFrame with renamed columns.
    """
    return df[[
        "Variant", "word_count", "token_count",
        "sentiment_polarity", "sentiment_subjectivity", "sentence_count"
    ]].rename(columns={
        "word_count":             "Words",
        "token_count":            "Tokens",
        "sentiment_polarity":     "Polarity",
        "sentiment_subjectivity": "Subjectivity",
        "sentence_count":         "Sentences",
    })
