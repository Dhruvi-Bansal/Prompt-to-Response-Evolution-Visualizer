"""
prompt_tree.py
--------------
Builds an interactive Plotly figure representing the prompt
evolution as a tree: root → variants → responses.
Uses NetworkX to compute node positions via a simple hierarchy layout.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict
from modules.prompt_variants import get_category_color


def _hierarchy_pos(G: nx.DiGraph, root: str, width: float = 2.0, vert_gap: float = 0.4,
                   vert_loc: float = 0, xcenter: float = 0.5):
    """
    Compute (x, y) positions for a tree rooted at `root`.
    Recursively assigns positions so leaves fan out horizontally.
    """
    def _recurse(G, node, left, right, vert_loc, pos):
        pos[node] = ((left + right) / 2, vert_loc)
        children = list(G.successors(node))
        if children:
            dx = (right - left) / len(children)
            nextx = left
            for child in children:
                _recurse(G, child, nextx, nextx + dx, vert_loc - vert_gap, pos)
                nextx += dx
        return pos

    return _recurse(G, root, 0, width, vert_loc, {})


def build_prompt_tree(base_prompt: str, results: List[Dict]) -> go.Figure:
    """
    Construct and return a Plotly figure of the prompt evolution tree.

    Args:
        base_prompt: The root prompt string.
        results:     List of variant dicts with 'label', 'category', 'response'.

    Returns:
        A Plotly Figure object ready to pass to st.plotly_chart.
    """
    G = nx.DiGraph()

    # Truncate helper for node labels
    def trunc(text: str, n: int = 30) -> str:
        return text[:n] + "…" if len(text) > n else text

    root_id   = "ROOT"
    root_label = trunc(base_prompt, 35)
    G.add_node(root_id, label=root_label, kind="root", color="#4F46E5")

    for i, item in enumerate(results):
        if item["category"] == "original":
            continue  # skip duplicate of base

        variant_id  = f"V{i}"
        response_id = f"R{i}"
        v_color = get_category_color(item["category"])

        G.add_node(variant_id,  label=item["label"],               kind="variant",  color=v_color)
        G.add_node(response_id, label=trunc(item["response"], 40), kind="response", color="#6B7280")
        G.add_edge(root_id,    variant_id)
        G.add_edge(variant_id, response_id)

    pos = _hierarchy_pos(G, root_id, width=2.5, vert_gap=0.55)

    # ── Build Plotly traces ──────────────────────────────────────────────────

    # Edges
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#CBD5E1"),
        hoverinfo="none",
    )

    # Nodes grouped by kind for different marker styles
    node_traces = []
    kind_styles = {
        "root":     dict(size=22, symbol="circle"),
        "variant":  dict(size=16, symbol="diamond"),
        "response": dict(size=12, symbol="square"),
    }

    for kind, style in kind_styles.items():
        nx_data = [(n, d) for n, d in G.nodes(data=True) if d.get("kind") == kind]
        if not nx_data:
            continue

        node_ids    = [n for n, _ in nx_data]
        node_labels = [d["label"] for _, d in nx_data]
        node_colors = [d["color"] for _, d in nx_data]
        xs = [pos[n][0] for n in node_ids]
        ys = [pos[n][1] for n in node_ids]

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            name=kind.capitalize(),
            text=node_labels,
            textposition="top center",
            textfont=dict(size=9, color="#1E293B"),
            marker=dict(
                size=style["size"],
                symbol=style["symbol"],
                color=node_colors,
                line=dict(width=1.5, color="#FFFFFF"),
            ),
            hovertext=node_labels,
            hoverinfo="text",
        ))

    fig = go.Figure(
        data=[edge_trace, *node_traces],
        layout=go.Layout(
            title=dict(
                text="Prompt Evolution Tree",
                font=dict(size=16, color="#1E293B"),
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right",  x=1,
                font=dict(size=10),
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=60, b=20),
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="closest",
        )
    )
    return fig
