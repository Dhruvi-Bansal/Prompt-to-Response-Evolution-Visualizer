"""
response_generator.py
---------------------
Handles loading a HuggingFace transformer model and generating
text responses for a list of prompt variants.
"""

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict


# ── Model configuration ──────────────────────────────────────────────────────
MODEL_NAME = "google/flan-t5-small"   # Lightweight, runs on CPU, good quality
MAX_NEW_TOKENS = 150
MIN_NEW_TOKENS = 20


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load and cache the HuggingFace text-generation pipeline.
    Uses FLAN-T5-small for fast, CPU-friendly inference.

    Returns:
        A HuggingFace pipeline object ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.3,
    )
    return gen_pipeline


def generate_responses(variants: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Generate a text response for each prompt variant.

    Args:
        variants: List of dicts containing 'label', 'category', and 'prompt'.

    Returns:
        The same list enriched with a 'response' key for each variant.
    """
    gen_pipeline = load_model()
    results = []

    for variant in variants:
        try:
            output = gen_pipeline(variant["prompt"])
            response_text = output[0]["generated_text"].strip()
        except Exception as e:
            response_text = f"[Generation error: {e}]"

        results.append({
            **variant,
            "response": response_text
        })

    return results
