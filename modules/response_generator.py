"""
response_generator.py
---------------------
Final structural update to force longer, more coherent explanations.
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict

MODEL_NAME = "google/flan-t5-base"   
MAX_NEW_TOKENS = 250 # Increased to allow for longer thoughts

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def generate_responses(variants: List[Dict[str, str]]) -> List[Dict[str, str]]:
    tokenizer, model = load_model()
    results = []

    for variant in variants:
        try:
            # We use a strict structural template to force 'Explanation' mode
            # This is the most effective way to get Seq2Seq models to provide length
            input_text = (
                f"Question: {variant['prompt']}\n\n"
                f"Detailed Answer: Provide a complete and informative explanation."
            )
            
            inputs = tokenizer(input_text, return_tensors="pt")

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=30,      # FORCES the model to keep talking
                do_sample=True,
                temperature=0.6,        # Balanced for variety and logic
                top_p=0.9,
                repetition_penalty=2.0, # Stronger penalty for circular logic
                no_repeat_ngram_size=3, 
                early_stopping=True
            )
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Clean up potential "Answer:" leftovers from the model
            response_text = response_text.replace("Detailed Answer:", "").strip()

        except Exception as e:
            response_text = f"[Generation error: {e}]"

        results.append({
            **variant,
            "response": response_text
        })

    return results