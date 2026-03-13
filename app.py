"""
Prompt-to-Response Evolution Visualizer
========================================
An interactive tool for analyzing and visualizing how prompt modifications
influence Generative AI model outputs.

Tech Stack: Python · Streamlit · HuggingFace Transformers · TextBlob · Plotly · Graphviz
"""

import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import graphviz
from textblob import TextBlob
from transformers import AutoTokenizer, pipeline

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Evolution Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .score-high  { background:#06d6a0; color:#000; }
    .score-mid   { background:#ffd166; color:#000; }
    .score-low   { background:#ff6b6b; color:#fff; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
    code { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Model Loading (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading tokenizer…")
def load_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

@st.cache_resource(show_spinner="Loading text-generation model…")
def load_generator():
    return pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        pad_token_id=50256,
    )

tokenizer = load_tokenizer()
generator  = load_generator()

# ──────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ──────────────────────────────────────────────
# Prompt Variant Generator
# ──────────────────────────────────────────────
VARIANT_TEMPLATES = {
    "Original":    "{prompt}",
    "+ Constraint":"You must answer in under 100 words. {prompt}",
    "+ Example":   "{prompt}\n\nFor example, provide a specific real-world instance.",
    "Formal Tone": "In a formal academic tone, {prompt_lower}",
    "Casual Tone": "In simple everyday language, {prompt_lower}",
    "Persuasive":  "Make a compelling argument: {prompt_lower}",
}

def generate_variants(base_prompt: str) -> dict[str, str]:
    """Return a dict of {variant_label: variant_prompt}."""
    p  = base_prompt.strip()
    pl = p[0].lower() + p[1:] if p else p
    return {
        label: tpl.format(prompt=p, prompt_lower=pl)
        for label, tpl in VARIANT_TEMPLATES.items()
    }

# ──────────────────────────────────────────────
# AI Response Generation
# ──────────────────────────────────────────────
def get_ai_response(prompt_text: str) -> str:
    """Generate a response for a given prompt using the local GPT-2 model."""
    try:
        result = generator(prompt_text)
        full   = result[0]["generated_text"]
        # Strip the prompt prefix from generated text
        response = full[len(prompt_text):].strip()
        return response if response else "[No response generated]"
    except Exception as exc:
        return f"[Generation error: {exc}]"

# ──────────────────────────────────────────────
# NLP Analysis Helpers
# ──────────────────────────────────────────────
def analyze_response(text: str) -> dict:
    """Compute NLP metrics for a generated response."""
    blob      = TextBlob(text)
    sentences = re.split(r"[.!?]", text)
    return {
        "length":    len(text.split()),
        "sentences": len([s for s in sentences if s.strip()]),
        "polarity":  round(blob.sentiment.polarity,  3),
        "subjectivity": round(blob.sentiment.subjectivity, 3),
    }

# ──────────────────────────────────────────────
# Prompt Quality Analyser
# ──────────────────────────────────────────────
INSTRUCTION_VERBS = [
    "explain", "describe", "generate", "write", "summarize",
    "compare", "analyze", "list", "outline", "give", "create",
    "evaluate", "discuss", "define", "identify",
]
VAGUE_WORDS = [
    "things", "stuff", "something", "many", "various",
    "good", "bad", "better", "improve", "etc",
]
FORMAT_KEYWORDS = [
    "bullet", "steps", "table", "json", "list",
    "sections", "points", "format", "numbered",
]

def analyze_prompt_quality(prompt: str) -> dict:
    """
    Score a prompt across six quality dimensions and return diagnostic
    suggestions plus an auto-improved version.
    """
    lower      = prompt.lower()
    word_count = len(prompt.split())

    # ── 1. Instruction Clarity ──────────────────
    has_instruction   = any(k in lower for k in INSTRUCTION_VERBS)
    instruction_score = 100 if has_instruction else 40

    # ── 2. Context Depth ────────────────────────
    if word_count < 6:
        context_score = 30
    elif word_count < 15:
        context_score = 60
    else:
        context_score = 90

    # ── 3. Specificity ──────────────────────────
    vague_found       = [v for v in VAGUE_WORDS if v in lower]
    specificity_score = 40 if vague_found else 90

    # ── 4. Output Structure ─────────────────────
    has_format      = any(f in lower for f in FORMAT_KEYWORDS)
    structure_score = 90 if has_format else 50

    # ── 5. Few-Shot Examples ────────────────────
    example_score = 90 if "example" in lower else 50

    # ── 6. Efficiency ───────────────────────────
    efficiency_score = 40 if word_count > 120 else 90

    scores = {
        "Instruction": instruction_score,
        "Context":     context_score,
        "Specificity": specificity_score,
        "Structure":   structure_score,
        "Examples":    example_score,
        "Efficiency":  efficiency_score,
    }
    overall_score = int(sum(scores.values()) / len(scores))

    # ── Suggestions & Improvements ─────────────
    suggestions, improvements = [], []

    if not has_instruction:
        suggestions.append("Add a clear instruction verb (explain, analyze, compare, list, summarize).")
        improvements.append("Start with a task verb: 'Explain…', 'Compare…', or 'List…'")
    if word_count < 6:
        suggestions.append("Prompt lacks context. Add background or constraints.")
        improvements.append("Describe the topic, target audience, or goal.")
    elif word_count < 15:
        suggestions.append("Add more contextual information to guide the response.")
    if vague_found:
        suggestions.append(f"Avoid vague terms: {', '.join(vague_found)}.")
        improvements.append("Replace vague terms with precise, concrete language.")
    if not has_format:
        suggestions.append("Specify the desired output format (bullet points, table, numbered steps).")
        improvements.append("Append: 'Provide the answer in bullet points.'")
    if "example" not in lower:
        suggestions.append("Adding an example can significantly improve response quality.")
    if word_count > 120:
        suggestions.append("Prompt is very long — trim unnecessary wording.")
        improvements.append("Keep prompts concise and focused on the core task.")

    # ── Auto-Improved Prompt ────────────────────
    improved = prompt
    if not has_instruction:
        improved = "Explain: " + improved
    if not has_format:
        improved += "\n\nProvide the answer in clear, numbered bullet points."
    if word_count < 10:
        improved += "\n\nInclude relevant examples and brief explanations."

    return {
        "scores":          scores,
        "overall_score":   overall_score,
        "suggestions":     suggestions,
        "improvements":    improvements,
        "improved_prompt": improved,
    }

# ──────────────────────────────────────────────
# Colour Helper
# ──────────────────────────────────────────────
def score_color(score: int) -> str:
    if score < 40:   return "#ff6b6b"
    elif score < 70: return "#ffd166"
    else:            return "#06d6a0"

def score_badge_class(score: int) -> str:
    if score < 40:   return "score-low"
    elif score < 70: return "score-mid"
    else:            return "score-high"

# ──────────────────────────────────────────────
# Visualisation: Prompt Evolution Flow Diagram
# ──────────────────────────────────────────────
def build_evolution_flowchart(history: list[dict]) -> graphviz.Digraph:
    """Render the prompt-evolution history as a directed flow graph."""
    dot = graphviz.Digraph(graph_attr={"rankdir": "LR", "bgcolor": "#0f1117"})
    dot.attr("node", fontname="Helvetica", fontsize="11", fontcolor="white")
    dot.attr("edge", color="#6b7280", arrowsize="0.7")

    for i, item in enumerate(history):
        color = score_color(item["score"])
        label = (
            f"Iteration {i + 1}\n"
            f"Score: {item['score']}/100\n"
            f"Tokens: {item['tokens']}"
        )
        dot.node(
            str(i), label,
            style="filled", fillcolor=color,
            shape="box", color="#ffffff",
        )
        if i > 0:
            dot.edge(str(i - 1), str(i))

    return dot

# ──────────────────────────────────────────────
# Visualisation: Prompt Variant Tree
# ──────────────────────────────────────────────
def build_variant_tree(base_prompt: str, variants: list[str]) -> graphviz.Digraph:
    """Render base prompt → variant branches as a radial tree."""
    dot = graphviz.Digraph(graph_attr={"bgcolor": "#0f1117"})
    dot.attr("node", fontname="Helvetica", fontsize="10", fontcolor="white")
    dot.attr("edge", color="#6b7280")

    root_label = "Base Prompt\n" + base_prompt[:40] + ("…" if len(base_prompt) > 40 else "")
    dot.node("root", root_label, shape="ellipse", style="filled", fillcolor="#6366f1")

    for i, variant in enumerate(variants):
        nid   = f"v{i}"
        label = variant[:35] + ("…" if len(variant) > 35 else "")
        dot.node(nid, label, shape="box", style="filled", fillcolor="#1e1e2e", color="#a78bfa")
        dot.edge("root", nid)

    return dot

# ──────────────────────────────────────────────
# Visualisation: Plotly Charts
# ──────────────────────────────────────────────
def plot_quality_radar(scores: dict) -> go.Figure:
    categories = list(scores.keys())
    values     = list(scores.values()) + [list(scores.values())[0]]  # close polygon
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories_closed,
        fill="toself", fillcolor="rgba(99,102,241,0.25)",
        line=dict(color="#6366f1", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1e1e2e",
            radialaxis=dict(visible=True, range=[0, 100], color="#9ca3af"),
            angularaxis=dict(color="#d1d5db"),
        ),
        showlegend=False,
        paper_bgcolor="#0f1117",
        margin=dict(l=40, r=40, t=40, b=40),
        height=340,
    )
    return fig

def plot_response_lengths(variant_labels: list, lengths: list) -> go.Figure:
    fig = px.bar(
        x=variant_labels, y=lengths,
        labels={"x": "Prompt Variant", "y": "Response Length (words)"},
        color=lengths, color_continuous_scale="Viridis",
        title="Response Length per Prompt Variant",
    )
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e1e2e",
        font_color="#d1d5db", coloraxis_showscale=False, height=350,
    )
    return fig

def plot_sentiment_comparison(variant_labels: list, polarities: list) -> go.Figure:
    colors = ["#06d6a0" if p >= 0 else "#ff6b6b" for p in polarities]
    fig = go.Figure(go.Bar(
        x=variant_labels, y=polarities,
        marker_color=colors,
        text=[f"{p:+.2f}" for p in polarities],
        textposition="outside",
    ))
    fig.update_layout(
        title="Sentiment Polarity by Variant",
        xaxis_title="Prompt Variant", yaxis_title="Polarity",
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor="#6b7280"),
        paper_bgcolor="#0f1117", plot_bgcolor="#1e1e2e",
        font_color="#d1d5db", height=350,
    )
    return fig

def plot_token_waste(history: list[dict]) -> go.Figure:
    tokens = [h["tokens"] for h in history]
    best   = min(tokens)
    waste  = [t - best for t in tokens]
    iters  = [f"Iter {i+1}" for i in range(len(tokens))]

    fig = px.bar(
        x=iters, y=waste,
        labels={"x": "Prompt Iteration", "y": "Extra Tokens vs Best"},
        color=waste, color_continuous_scale="Reds",
        title="Token Overhead Relative to Most Efficient Prompt",
    )
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e1e2e",
        font_color="#d1d5db", coloraxis_showscale=False, height=340,
    )
    return fig

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.title("Prompt Visualizer")
    st.markdown("**Version 2.0** · Academic Edition")
    st.divider()
    st.markdown("""
    **How it works**
    1. Enter your base prompt
    2. Click **Analyse Prompt** for quality metrics
    3. Click **Generate Variants & Responses** to run the full evolution pipeline
    4. Explore charts, comparisons, and the flow diagram
    """)
    st.divider()
    cost_per_1k = st.slider("Cost per 1 k tokens (USD)", 0.0001, 0.01, 0.00015, 0.00005, format="%.5f")
    st.divider()
    if st.button("🗑 Reset Session", use_container_width=True):
        st.session_state.history = []
        st.success("Session cleared.")

# ──────────────────────────────────────────────
# Main Layout
# ──────────────────────────────────────────────
st.title("🧠 Prompt-to-Response Evolution Visualizer")
st.markdown(
    "Understand how prompt modifications influence Generative AI outputs "
    "through interactive visualizations and quantitative analysis."
)
st.divider()

# ── Base Prompt Input ────────────────────────
base_prompt = st.text_area(
    "**Enter your base prompt**",
    placeholder="e.g.  Explain the concept of machine learning to a high school student.",
    height=130,
)

col_a, col_b, col_c = st.columns([2, 2, 1])
analyse_btn  = col_a.button("🔍 Analyse Prompt",              use_container_width=True)
generate_btn = col_b.button("⚡ Generate Variants & Responses", use_container_width=True)

# ──────────────────────────────────────────────
# Prompt Quality Analysis
# ──────────────────────────────────────────────
if analyse_btn:
    if not base_prompt.strip():
        st.warning("⚠️ Please enter a prompt before analysing.")
    else:
        tokens      = tokenizer(base_prompt)["input_ids"]
        token_count = len(tokens)
        analysis    = analyze_prompt_quality(base_prompt)
        score       = analysis["overall_score"]

        # Store in history
        st.session_state.history.append({
            "prompt": base_prompt,
            "score":  score,
            "tokens": token_count,
        })

        # ── Metrics Row ──────────────────────────
        st.markdown("### 📊 Prompt Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Characters", len(base_prompt))
        m2.metric("Words",      len(base_prompt.split()))
        m3.metric("Tokens",     token_count)
        m4.metric("Sentences",  len([s for s in re.split(r"[.!?]", base_prompt) if s.strip()]))
        m5.metric("Questions",  base_prompt.count("?"))

        st.divider()

        # ── Score + Radar ────────────────────────
        st.markdown("### 🎯 Quality Score & Breakdown")
        left, right = st.columns([1, 2])

        badge_cls = score_badge_class(score)
        left.markdown(f"""
        <div class="metric-card" style="text-align:center; padding:32px 20px;">
            <div style="font-size:3rem; font-weight:800; color:{score_color(score)};">
                {score}<span style="font-size:1.2rem; color:#9ca3af;">/100</span>
            </div>
            <div class="score-badge {badge_cls}" style="margin-top:10px;">
                {"Excellent" if score >= 70 else "Needs Work" if score >= 40 else "Poor"}
            </div>
            <div style="color:#9ca3af; font-size:0.8rem; margin-top:14px;">
                Estimated cost: ${token_count / 1000 * cost_per_1k:.6f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        right.plotly_chart(plot_quality_radar(analysis["scores"]), use_container_width=True)

        # ── Score Table ──────────────────────────
        with st.expander("View Detailed Score Breakdown"):
            df_scores = pd.DataFrame(
                analysis["scores"].items(), columns=["Dimension", "Score"]
            )
            df_scores["Rating"] = df_scores["Score"].apply(
                lambda s: "✅ Good" if s >= 70 else ("⚠️ Fair" if s >= 40 else "❌ Poor")
            )
            st.dataframe(df_scores, use_container_width=True, hide_index=True)

        st.divider()

        # ── Suggestions ──────────────────────────
        if analysis["suggestions"]:
            st.markdown("### ✨ Improvement Suggestions")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**What to fix**")
                for s in analysis["suggestions"]:
                    st.markdown(f"- {s}")
            with col2:
                st.markdown("**How to fix it**")
                for imp in analysis["improvements"]:
                    st.markdown(f"- {imp}")

        st.divider()

        # ── Auto-Improved Prompt ─────────────────
        st.markdown("### 🚀 Auto-Improved Prompt")
        st.code(analysis["improved_prompt"], language="text")

# ──────────────────────────────────────────────
# Variant Generation & Response Comparison
# ──────────────────────────────────────────────
if generate_btn:
    if not base_prompt.strip():
        st.warning("⚠️ Please enter a prompt before generating variants.")
    else:
        variants_dict = generate_variants(base_prompt)
        variant_labels = list(variants_dict.keys())
        variant_prompts = list(variants_dict.values())

        st.markdown("### 🌿 Prompt Variant Tree")
        st.graphviz_chart(build_variant_tree(base_prompt, variant_prompts))
        st.divider()

        # ── Generate responses ───────────────────
        st.markdown("### ⚡ Generating AI Responses…")
        progress = st.progress(0)
        results  = []

        for idx, (label, vp) in enumerate(variants_dict.items()):
            with st.spinner(f"Running variant: **{label}**"):
                response = get_ai_response(vp)
                metrics  = analyze_response(response)
                results.append({
                    "label":    label,
                    "prompt":   vp,
                    "response": response,
                    **metrics,
                })
            progress.progress((idx + 1) / len(variants_dict))

        progress.empty()
        st.success(f"✅ Generated {len(results)} responses.")
        st.divider()

        # ── Side-by-Side Comparison ──────────────
        st.markdown("### 🔀 Side-by-Side Response Comparison")
        tab_labels = [r["label"] for r in results]
        tabs = st.tabs(tab_labels)

        for tab, res in zip(tabs, results):
            with tab:
                st.markdown(f"**Prompt Variant:** `{res['label']}`")
                st.text_area("Prompt sent to model:", res["prompt"], height=90, disabled=True)
                st.markdown("**Model Response:**")
                st.info(res["response"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Words",        res["length"])
                c2.metric("Sentences",    res["sentences"])
                c3.metric("Polarity",     res["polarity"])
                c4.metric("Subjectivity", res["subjectivity"])

        st.divider()

        # ── Quantitative Charts ──────────────────
        st.markdown("### 📈 Quantitative Analysis")
        ch1, ch2 = st.columns(2)
        ch1.plotly_chart(
            plot_response_lengths(tab_labels, [r["length"] for r in results]),
            use_container_width=True,
        )
        ch2.plotly_chart(
            plot_sentiment_comparison(tab_labels, [r["polarity"] for r in results]),
            use_container_width=True,
        )

        # ── Results Table ────────────────────────
        st.divider()
        st.markdown("### 📋 Results Summary")
        df_results = pd.DataFrame([{
            "Variant":      r["label"],
            "Words":        r["length"],
            "Sentences":    r["sentences"],
            "Polarity":     r["polarity"],
            "Subjectivity": r["subjectivity"],
        } for r in results])
        st.dataframe(df_results, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# Prompt History & Evolution
# ──────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.markdown("### 🗂 Prompt Iteration History")

    df_hist = pd.DataFrame(st.session_state.history)
    df_hist.index = df_hist.index + 1
    df_hist.index.name = "Iteration"
    st.dataframe(df_hist[["prompt", "score", "tokens"]], use_container_width=True)

    st.markdown("### 🔗 Prompt Evolution Flow Diagram")
    st.graphviz_chart(build_evolution_flowchart(st.session_state.history))

    if len(st.session_state.history) > 1:
        st.markdown("### 💸 Token Overhead Analysis")
        st.plotly_chart(plot_token_waste(st.session_state.history), use_container_width=True)