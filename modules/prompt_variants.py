"""
prompt_variants.py
------------------
Generates multiple variants of a base prompt by applying
different transformation strategies (tone, length, detail, etc.).
"""

from typing import List, Dict


def generate_variants(base_prompt: str) -> List[Dict[str, str]]:
    """
    Generate a list of prompt variants from a base prompt.

    Args:
        base_prompt: The original user-provided prompt string.

    Returns:
        A list of dicts, each with 'label' and 'prompt' keys.
    """
    variants = []

    # 1. Base (unchanged)
    variants.append({
        "label": "Base Prompt",
        "category": "original",
        "prompt": base_prompt
    })

    # 2. Add word constraint
    variants.append({
        "label": "Constrained (≤50 words)",
        "category": "constraint",
        "prompt": f"{base_prompt} Answer in 50 words or less."
    })

    # 3. Add an example instruction
    variants.append({
        "label": "With Example",
        "category": "example",
        "prompt": f"{base_prompt} For example, provide a concrete real-world illustration."
    })

    # 4. Formal tone
    variants.append({
        "label": "Formal Tone",
        "category": "tone",
        "prompt": f"In a formal and academic tone, {base_prompt.lower()}"
    })

    # 5. Casual tone
    variants.append({
        "label": "Casual Tone",
        "category": "tone",
        "prompt": f"Hey, can you explain this in a super casual, friendly way? {base_prompt}"
    })

    # 6. Persuasive tone
    variants.append({
        "label": "Persuasive Tone",
        "category": "tone",
        "prompt": f"Write a persuasive and compelling response to: {base_prompt}"
    })

    # 7. Simplified explanation
    variants.append({
        "label": "Simplified (ELI5)",
        "category": "detail",
        "prompt": f"Explain this as simply as possible, like I'm 5 years old: {base_prompt}"
    })

    # 8. Detailed explanation
    variants.append({
        "label": "Detailed Explanation",
        "category": "detail",
        "prompt": f"Provide a thorough, detailed, and comprehensive explanation: {base_prompt}"
    })

    return variants


def get_category_color(category: str) -> str:
    """
    Return a color hex code associated with each variant category.

    Args:
        category: The category string (e.g., 'tone', 'detail').

    Returns:
        A hex color string.
    """
    color_map = {
        "original":   "#4F46E5",  # Indigo
        "constraint": "#0891B2",  # Cyan
        "example":    "#059669",  # Emerald
        "tone":       "#D97706",  # Amber
        "detail":     "#DC2626",  # Red
    }
    return color_map.get(category, "#6B7280")
