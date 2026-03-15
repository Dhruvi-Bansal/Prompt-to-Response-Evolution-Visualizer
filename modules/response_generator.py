"""
response_generator.py
---------------------
Handles loading a HuggingFace transformer model and generating
text responses for a list of prompt variants.
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict

# ── Model configuration ──────────────────────────────────────────────────────
# Upgrading to 'base' (250M params) solves the grammar issues of 'small' (80M params)
MODEL_NAME = "google/flan-t5-base"   
MAX_NEW_TOKENS = 150

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load and cache the HuggingFace model and tokenizer directly.
    Bypasses pipeline task registry issues.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def generate_responses(variants: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Generate a text response for each prompt variant using direct inference.
    """
    # Unpack the cached assets
    tokenizer, model = load_model()
    results = []

    for variant in variants:
        try:
            # We add a task prefix to help the model understand it needs to answer
            input_text = f"answer the following: {variant['prompt']}"
            
            # Encode the input
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate output using stable parameters for better grammar
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,      # Lower temperature = more logical grammar
                top_p=0.9,            # Nucleus sampling for natural flow
                repetition_penalty=1.1 # Prevents repeating words without breaking grammar
            )
            
            # Decode the output
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
        except Exception as e:
            response_text = f"[Generation error: {e}]"

        results.append({
            **variant,
            "response": response_text
        })

    return results