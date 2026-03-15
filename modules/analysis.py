"""
analysis.py
-----------
Performs NLP analysis on generated responses:
- Response length (character count)
- Token count (whitespace-based approximation)
- Sentiment polarity via TextBlob
- Subjectivity score via TextBlob
"""

import nltk
import pandas as pd
from textblob import TextBlob
from typing import List, Dict

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def analyze_response(response_text: str) -> Dict[str, float]:
    """
    Compute NLP metrics for a single response string.

    Args:
        response_text: The generated text to analyse.

    Returns:
        Dict with keys: char_count, word_count, token_count,
        sentiment_polarity, sentiment_subjectivity, sentence_count.
    """
    blob = TextBlob(response_text)

    # Token count using NLTK word tokenizer
    try:
        tokens = nltk.word_tokenize(response_text)
        token_count = len(tokens)
    except Exception:
        token_count = len(response_text.split())

    return {
        "char_count":              len(response_text),
        "word_count":              len(response_text.split()),
        "token_count":             token_count,
        "sentiment_polarity":      round(blob.sentiment.polarity, 4),
        "sentiment_subjectivity":  round(blob.sentiment.subjectivity, 4),
        "sentence_count":          len(blob.sentences),
    }


def build_analysis_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Build a tidy Pandas DataFrame from a list of variant-response dicts.

    Args:
        results: List of dicts containing at least 'label' and 'response'.

    Returns:
        DataFrame with one row per variant and analysis columns.
    """
    rows = []
    for item in results:
        metrics = analyze_response(item["response"])
        rows.append({
            "Variant":         item["label"],
            "Category":        item["category"],
            "Prompt":          item["prompt"],
            "Response":        item["response"],
            **metrics,
        })
    return pd.DataFrame(rows)
