"""
app.py  ─  Prompt-to-Response Evolution Visualizer
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from modules.prompt_variants   import generate_variants
from modules.response_generator import generate_responses, load_model
from modules.analysis           import build_analysis_dataframe
from visualizations.prompt_tree import build_prompt_tree
from visualizations.charts      import (
    chart_response_length,
    chart_sentiment_polarity,
    chart_token_count,
    chart_subjectivity,
)
from utils.helpers import (
    sanitize_prompt,
    category_badge_html,
    sentiment_emoji,
    df_to_display,
    truncate_text,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Evolution Visualizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ───────────────────────────────────────────────────────────
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Evolution Visualizer</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
**How it works**
1. Enter a base prompt below
2. Click **Run Analysis**
3. The app creates 8 prompt variants
4. Explore the tree, cards, and charts

---
**Variant types**

| Symbol | Type |
|--------|------|
| 🟣 | Original |
| 🔵 | Constraint |
| 🟢 | Example |
| 🟡 | Tone |
| 🔴 | Detail |

---
""")


# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>Prompt-to-Response Evolution Visualizer</h1>
  <p>Examine how structural and tonal modifications to a single prompt alter
    the outputs of a transformer language model. Seven variants are generated
    automatically and evaluated across length, sentiment, and token metrics.</p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Base Prompt</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1], gap="medium")

with col_input:
    base_prompt_raw = st.text_area(
        label="Enter your base prompt",
        placeholder="e.g. Explain the concept of machine learning",
        height=90,
        label_visibility="collapsed",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_clicked = st.button("Run Analysis", use_container_width=True)


# ── Main Logic ────────────────────────────────────────────────────────────────
if generate_clicked:
    base_prompt = sanitize_prompt(base_prompt_raw)

    if not base_prompt:
        st.warning("Please enter a prompt before generating.")
        st.stop()

    # ── 1. Generate variants ──────────────────────────────────────────────
    with st.spinner("Building prompt variants…"):
        variants = generate_variants(base_prompt)

    # ── 2. Generate responses ─────────────────────────────────────────────
    progress_bar = st.progress(0, text="Loading model & generating responses…")
    results = []
    total = len(variants)

    # Load model once (cached)
    load_model()

    from modules.response_generator import generate_responses as _gen
    with st.spinner(f"Generating {total} responses with FLAN-T5…"):
        results = _gen(variants)
        progress_bar.progress(100, text="All responses generated!")

    # ── 3. Analysis ───────────────────────────────────────────────────────
    df = build_analysis_dataframe(results)

    # ── KPI strip ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Overview Metrics</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi_data = [
        (kpi1, str(total),                        "Variants Generated"),
        (kpi2, str(int(df["word_count"].mean())),  "Avg Words / Response"),
        (kpi3, str(int(df["token_count"].mean())), "Avg Tokens / Response"),
        (kpi4, f"{df['sentiment_polarity'].mean():+.3f}", "Avg Sentiment"),
        (kpi5, str(int(df["word_count"].max())),  "Max Response Length"),
    ]
    for col, value, label in kpi_data:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="value">{value}</div>'
                f'<div class="label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── 4. Prompt Evolution Tree ──────────────────────────────────────────
    st.markdown("""
<div class="section-header">
  Prompt Evolution Tree
  <p>This diagram shows the evolution from the base prompt to its variants and generated responses. 
  Hover over nodes to view the full text.</p>
</div>
""", unsafe_allow_html=True)
    
    tree_fig = build_prompt_tree(base_prompt, results)
    st.plotly_chart(tree_fig, use_container_width=True)

    # ── 5. Side-by-side Response Panels ──────────────────────────────────
    st.markdown('<div class="section-header">Response Comparison Panels</div>', unsafe_allow_html=True)

    cols = st.columns(2, gap="medium")
    for idx, row in df.iterrows():
        col = cols[idx % 2]
        with col:
            sentiment_str = sentiment_emoji(row["sentiment_polarity"])
            badge_html    = category_badge_html(row["Category"])
            prompt_display = (
                f'<div class="prompt-text"><b>Prompt:</b> {truncate_text(row["Prompt"], 100)}</div>'
            )
            st.markdown(
                f"""
                <div class="response-card">
                  <div class="card-header">
                    <span class="prompt-label">{row['Variant']}</span>
                    {badge_html}
                  </div>
                  {prompt_display}
                  <p class="response-text">{row['Response']}</p>
                  <div class="metrics-row">
                    <span>📏 {row['word_count']} words</span>
                    <span>🔢 {row['token_count']} tokens</span>
                    <span>{sentiment_str}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 6. Interactive Charts ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Quantitative Analysis Charts</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Response Length",
        "Sentiment Polarity",
        "Token Count",
        "Polarity vs Subjectivity",
    ])

    with tab1: st.plotly_chart(chart_response_length(df),    use_container_width=True)
    with tab2: st.plotly_chart(chart_sentiment_polarity(df), use_container_width=True)
    with tab3: st.plotly_chart(chart_token_count(df),        use_container_width=True)
    with tab4: st.plotly_chart(chart_subjectivity(df),       use_container_width=True)

    # ── 7. Raw Metrics Table (optional) ──────────────────────────────────
    st.markdown('<div class="section-header">Raw Metrics Table</div>', unsafe_allow_html=True)
    
    # Switched to Streamlit's new column_config for a cleaner, modern sentiment progress bar
    display_df = df_to_display(df)
    st.dataframe(
        display_df,
        column_config={
            "Polarity": st.column_config.ProgressColumn(
                "Polarity Score",
                help="Sentiment score from -1 to 1",
                format="%.3f",
                min_value=-1,
                max_value=1,
            ),
            "Subjectivity": st.column_config.NumberColumn(
                "Subjectivity",
                format="%.2f",
            )
        },
        hide_index=True,
        use_container_width=True,
        height=320,
    )

    # ── Download CSV ──────────────────────────────────────────────────────
    st.markdown("---")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Analysis CSV",
        data=csv_data,
        file_name="prompt_evolution_analysis.csv",
        mime="text/csv",
    )

else:
    # ── Landing / empty state ─────────────────────────────────────────────
    st.markdown("""
    <div class="empty-state">
      <div class="empty-state-title">Ready to Explore Prompt Visualizer</div>
      <p>Type a prompt above and click <b>Run Analysis</b> to see how phrasing
      changes shape AI responses — from tone shifts to detail levels.</p>
      <div class="empty-state-examples">
        <b>Example prompts to try:</b><br>
        "Explain the concept of machine learning"<br>
        "What is the impact of social media on society?"<br>
        "Describe how photosynthesis works"
      </div>
    </div>
    """, unsafe_allow_html=True)