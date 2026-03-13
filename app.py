"""
app.py  ─  Prompt-to-Response Evolution Visualizer
====================================================
Entry point for the Streamlit application.

Run with:
    streamlit run app.py
"""

import sys
import os

# Make sure sibling packages are importable regardless of working directory
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
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #0EA5E9 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.hero-banner h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 0.4rem 0; }
.hero-banner p  { font-size: 1rem; opacity: 0.88; margin: 0; }

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #4F46E5; }
.metric-card .label { font-size: 0.78rem; color: #64748B; margin-top: 2px; }

/* Response card */
.response-card {
    background: #FAFAFA;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #4F46E5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #1E293B;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1E293B;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #E2E8F0;
}

/* Prompt label pill */
.prompt-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #4F46E5;
    background: #EEF2FF;
    padding: 2px 10px;
    border-radius: 9999px;
    display: inline-block;
    margin-bottom: 6px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0F172A;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F8FAFC !important; }

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.6rem;
    font-size: 0.95rem;
    transition: opacity .2s;
}
div.stButton > button:hover { opacity: 0.88; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Evolution Visualizer")
    st.markdown("---")
    st.markdown("""
**How it works**

1. Enter a base prompt below
2. Click **Generate**
3. The app creates 8 prompt variants
4. FLAN-T5 generates responses for each
5. Explore the tree, cards, and charts

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
**Model:** `google/flan-t5-small`
*(downloads ~300 MB on first run)*

---
""")
    st.markdown("#### ⚙️ Settings")
    show_raw_table = st.checkbox("Show raw metrics table", value=False)
    show_prompts   = st.checkbox("Show full prompts in cards", value=False)


# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🔬 Prompt-to-Response Evolution Visualizer</h1>
  <p>Discover how small changes in phrasing transform AI-generated responses —
     with interactive trees, side-by-side panels, and quantitative NLP charts.</p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📝 Base Prompt</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1], gap="medium")

with col_input:
    base_prompt_raw = st.text_area(
        label="Enter your base prompt",
        placeholder="e.g.  Explain the concept of machine learning",
        height=90,
        label_visibility="collapsed",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_clicked = st.button("⚡ Generate", use_container_width=True)


# ── Pre-warm model note ───────────────────────────────────────────────────────
with st.expander("ℹ️ First-run note", expanded=False):
    st.info(
        "On the first click, the FLAN-T5-small model (~300 MB) will be "
        "downloaded from HuggingFace Hub and cached locally. "
        "Subsequent runs are instant."
    )


# ── Main Logic ────────────────────────────────────────────────────────────────
if generate_clicked:
    base_prompt = sanitize_prompt(base_prompt_raw)

    if not base_prompt:
        st.warning("⚠️  Please enter a prompt before generating.")
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
        progress_bar.progress(100, text="✅ All responses generated!")

    # ── 3. Analysis ───────────────────────────────────────────────────────
    df = build_analysis_dataframe(results)

    # ── KPI strip ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Overview Metrics</div>',
                unsafe_allow_html=True)

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

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4. Prompt Evolution Tree ──────────────────────────────────────────
    st.markdown('<div class="section-header">🌳 Prompt Evolution Tree</div>',
                unsafe_allow_html=True)

    tree_fig = build_prompt_tree(base_prompt, results)
    st.plotly_chart(tree_fig, use_container_width=True)

    # ── 5. Side-by-side Response Panels ──────────────────────────────────
    st.markdown('<div class="section-header">💬 Response Comparison Panels</div>',
                unsafe_allow_html=True)

    # Display in 2-column grid
    cols = st.columns(2, gap="medium")
    for idx, row in df.iterrows():
        col = cols[idx % 2]
        with col:
            sentiment_str = sentiment_emoji(row["sentiment_polarity"])
            badge_html    = category_badge_html(row["Category"])
            prompt_display = (
                f'<p style="font-size:0.78rem;color:#64748B;margin:4px 0 8px 0;">'
                f'<b>Prompt:</b> {truncate_text(row["Prompt"], 100)}</p>'
                if show_prompts else ""
            )
            st.markdown(
                f"""
                <div class="response-card">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span class="prompt-label">{row['Variant']}</span>
                    {badge_html}
                  </div>
                  {prompt_display}
                  <p style="margin:0 0 10px 0;">{row['Response']}</p>
                  <div style="font-size:0.75rem;color:#94A3B8;display:flex;gap:16px;flex-wrap:wrap;">
                    <span>📏 {row['word_count']} words</span>
                    <span>🔢 {row['token_count']} tokens</span>
                    <span>{sentiment_str}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 6. Interactive Charts ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Quantitative Analysis Charts</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📏 Response Length",
        "😊 Sentiment Polarity",
        "🔢 Token Count",
        "🎯 Polarity vs Subjectivity",
    ])

    with tab1:
        st.plotly_chart(chart_response_length(df),    use_container_width=True)
    with tab2:
        st.plotly_chart(chart_sentiment_polarity(df), use_container_width=True)
    with tab3:
        st.plotly_chart(chart_token_count(df),        use_container_width=True)
    with tab4:
        st.plotly_chart(chart_subjectivity(df),       use_container_width=True)

    # ── 7. Raw Metrics Table (optional) ──────────────────────────────────
    if show_raw_table:
        st.markdown('<div class="section-header">🗃️ Raw Metrics Table</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            df_to_display(df).style.background_gradient(
                subset=["Words", "Tokens"], cmap="Blues"
            ).background_gradient(
                subset=["Polarity"], cmap="RdYlGn", vmin=-1, vmax=1
            ),
            use_container_width=True,
            height=320,
        )

    # ── Download CSV ──────────────────────────────────────────────────────
    st.markdown("---")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Analysis CSV",
        data=csv_data,
        file_name="prompt_evolution_analysis.csv",
        mime="text/csv",
    )

else:
    # ── Landing / empty state ─────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#94A3B8;">
      <div style="font-size:4rem;">🔬</div>
      <h3 style="color:#64748B;">Ready to Explore Prompt Dynamics</h3>
      <p>Type a prompt above and click <b>⚡ Generate</b> to see how phrasing
      changes shape AI responses — from tone shifts to detail levels.</p>
      <br>
      <p style="font-size:0.85rem;">
        <b>Example prompts to try:</b><br>
        "Explain the concept of machine learning"<br>
        "What is the impact of social media on society?"<br>
        "Describe how photosynthesis works"
      </p>
    </div>
    """, unsafe_allow_html=True)
