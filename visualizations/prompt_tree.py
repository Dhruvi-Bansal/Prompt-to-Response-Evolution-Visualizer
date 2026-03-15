"""
prompt_tree.py
--------------
Builds a professional, readable Plotly figure representing the prompt
evolution as a styled tree: root → variants → responses.
"""

import plotly.graph_objects as go
from typing import List, Dict
from modules.prompt_variants import get_category_color


# ── Layout constants ──────────────────────────────────────────────────────────
CANVAS_W   = 1400
CANVAS_H   = 700
ROOT_Y     = 620
VARIANT_Y  = 370
RESPONSE_Y = 120

ROOT_W,     ROOT_H     = 240, 50
VARIANT_W,  VARIANT_H  = 162, 46
RESPONSE_W, RESPONSE_H = 168, 62


def _wrap(text: str, max_chars: int = 22) -> str:
    """Wrap text at word boundaries to fit inside node boxes."""
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + " " + w).strip()
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "<br>".join(lines)


def _trunc(text: str, n: int = 28) -> str:
    return text[:n] + "…" if len(text) > n else text


def _bezier_path(x0, y0, x1, y1) -> str:
    """Cubic bezier SVG path between two points."""
    mid_y = (y0 + y1) / 2
    return f"M {x0},{y0} C {x0},{mid_y} {x1},{mid_y} {x1},{y1}"


def build_prompt_tree(base_prompt: str, results: List[Dict]) -> go.Figure:
    """
    Construct and return a professional Plotly figure of the prompt tree.

    Args:
        base_prompt: The root prompt string.
        results:     List of variant dicts (label, category, response).

    Returns:
        Plotly Figure.
    """
    variants = [r for r in results if r["category"] != "original"]
    n = len(variants)

    # ── X positions (evenly spaced) ───────────────────────────────────────────
    margin = 120
    if n > 1:
        xs = [margin + i * (CANVAS_W - 2 * margin) / (n - 1) for i in range(n)]
    else:
        xs = [CANVAS_W / 2]

    root_x = CANVAS_W / 2

    shapes      = []
    annotations = []

    # ── Helper: rounded rect shape ────────────────────────────────────────────
    def box(cx, cy, w, h, fill, border):
        return dict(
            type="rect",
            x0=cx - w / 2, y0=cy - h / 2,
            x1=cx + w / 2, y1=cy + h / 2,
            fillcolor=fill,
            line=dict(color=border, width=2),
            xref="x", yref="y",
        )

    # ── Edges (bezier curves) ─────────────────────────────────────────────────
    for vx in xs:
        # Root → Variant
        shapes.append(dict(
            type="path",
            path=_bezier_path(root_x, ROOT_Y - ROOT_H / 2,
                              vx,     VARIANT_Y + VARIANT_H / 2),
            line=dict(color="rgba(99,102,241,0.45)", width=2),
            xref="x", yref="y",
        ))
        # Variant → Response
        shapes.append(dict(
            type="path",
            path=_bezier_path(vx, VARIANT_Y - VARIANT_H / 2,
                              vx, RESPONSE_Y + RESPONSE_H / 2),
            line=dict(color="rgba(148,163,184,0.3)", width=1.5),
            xref="x", yref="y",
        ))

    # ── Level label divider lines ─────────────────────────────────────────────
    for y_line, label_text in [
        (ROOT_Y + 80,     "① BASE PROMPT"),
        (VARIANT_Y + 70,  "② PROMPT VARIANTS"),
        (RESPONSE_Y + 70, "③ AI RESPONSES"),
    ]:
        shapes.append(dict(
            type="line",
            x0=0, x1=CANVAS_W, y0=y_line, y1=y_line,
            line=dict(color="rgba(71,85,105,0.4)", width=1, dash="dot"),
            xref="x", yref="y",
        ))
        annotations.append(dict(
            x=10, y=y_line + 2,
            text=f'<span style="font-size:9px;letter-spacing:1px">{label_text}</span>',
            showarrow=False, xref="x", yref="y",
            font=dict(size=9, color="#64748B", family="Inter, sans-serif"),
            xanchor="left", yanchor="bottom",
        ))

    # ── Root node ─────────────────────────────────────────────────────────────
    shapes.append(box(root_x, ROOT_Y, ROOT_W, ROOT_H,
                      fill="#4338CA", border="#818CF8"))
    # Glow ring
    shapes.append(dict(
        type="circle",
        x0=root_x - ROOT_W / 2 - 6, y0=ROOT_Y - ROOT_H / 2 - 6,
        x1=root_x + ROOT_W / 2 + 6, y1=ROOT_Y + ROOT_H / 2 + 6,
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="rgba(129,140,248,0.35)", width=3),
        xref="x", yref="y",
    ))
    annotations.append(dict(
        x=root_x, y=ROOT_Y,
        text=f"<b>{_trunc(base_prompt, 34)}</b>",
        showarrow=False, xref="x", yref="y",
        font=dict(size=12, color="#FFFFFF", family="Inter, sans-serif"),
        align="center",
    ))

    # ── Variant + Response nodes ──────────────────────────────────────────────
    for item, vx in zip(variants, xs):
        v_color = get_category_color(item["category"])

        # Variant box
        shapes.append(box(vx, VARIANT_Y, VARIANT_W, VARIANT_H,
                          fill=v_color, border="#FFFFFF"))
        annotations.append(dict(
            x=vx, y=VARIANT_Y,
            text=f"<b>{_wrap(item['label'], 18)}</b>",
            showarrow=False, xref="x", yref="y",
            font=dict(size=10, color="#FFFFFF", family="Inter, sans-serif"),
            align="center",
        ))

        # Response box (dark card with accent left-border simulation via outline)
        shapes.append(box(vx, RESPONSE_Y, RESPONSE_W, RESPONSE_H,
                          fill="#1E293B", border=v_color))
        resp_preview = _wrap(_trunc(item["response"], 55), 24)
        annotations.append(dict(
            x=vx, y=RESPONSE_Y,
            text=resp_preview,
            showarrow=False, xref="x", yref="y",
            font=dict(size=8.5, color="#94A3B8", family="Inter, sans-serif"),
            align="center",
        ))

    # ── Hover scatter (invisible, for tooltips) ───────────────────────────────
    hx, hy, ht, hc = [], [], [], []

    hx.append(root_x); hy.append(ROOT_Y)
    ht.append(f"<b>Base Prompt</b><br><br>{base_prompt}")
    hc.append("#4F46E5")

    for item, vx in zip(variants, xs):
        hx.append(vx); hy.append(VARIANT_Y)
        ht.append(
            f"<b>{item['label']}</b> "
            f"<span style='color:#94A3B8'>[{item['category']}]</span><br><br>"
            f"<i>{item['prompt'][:140]}…</i>"
        )
        hc.append(get_category_color(item["category"]))

        hx.append(vx); hy.append(RESPONSE_Y)
        ht.append(
            f"<b>Response — {item['label']}</b><br><br>"
            f"{item['response'][:220]}…"
        )
        hc.append("#475569")

    hover_trace = go.Scatter(
        x=hx, y=hy,
        mode="markers",
        marker=dict(size=20, color=hc, opacity=0),
        hovertext=ht,
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#334155",
            font=dict(color="#E2E8F0", size=11, family="Inter, sans-serif"),
            namelength=0,
            align="left",
        ),
        showlegend=False,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=11, color="#4338CA", symbol="square",
                               line=dict(color="#818CF8", width=2)),
                   name="Base Prompt", showlegend=True),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=11, color="#D97706", symbol="square",
                               line=dict(color="#FFFFFF", width=2)),
                   name="Variant", showlegend=True),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=11, color="#1E293B", symbol="square",
                               line=dict(color="#D97706", width=2)),
                   name="Response", showlegend=True),
    ]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = go.Figure(
        data=[hover_trace, *legend_traces],
        layout=go.Layout(
            title=dict(
                text="<b>Prompt Evolution Tree</b>  "
                     "<span style='font-size:13px;color:#64748B'>"
                     "— hover over any node for full text</span>",
                x=0.5, xanchor="center",
                font=dict(size=17, color="#F1F5F9",
                          family="Inter, sans-serif"),
            ),
            shapes=shapes,
            annotations=annotations,
            xaxis=dict(range=[-20, CANVAS_W + 20],
                       showgrid=False, zeroline=False,
                       showticklabels=False, fixedrange=True),
            yaxis=dict(range=[RESPONSE_Y - 90, ROOT_Y + 100],
                       showgrid=False, zeroline=False,
                       showticklabels=False, fixedrange=True),
            margin=dict(l=10, r=10, t=65, b=45),
            height=630,
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            hovermode="closest",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.07,
                xanchor="center", x=0.5,
                font=dict(size=11, color="#CBD5E1"),
                bgcolor="rgba(0,0,0,0)",
                tracegroupgap=10,
            ),
        )
    )
    return fig
