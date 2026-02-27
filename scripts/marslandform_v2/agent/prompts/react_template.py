from __future__ import annotations

from typing import Iterable


REACT_TEMPLATE = """
Follow this exact ReACT format.

Thought: <your concise reasoning about what evidence is missing or sufficient>
Action: <one of: classify, query_rag, analyze_mola, zoom_tile, regional_context>
Action Input: <valid JSON object for the selected tool>
Observation: <filled by environment after tool execution>

Repeat Thought/Action/Action Input/Observation until ready.

When ready to decide, return:
Final Answer:
Class: <LDA|LVF|CCF|GLF|BACKGROUND>
Confidence: <0.00-1.00>
Reasoning: <2-4 sentence evidence-based summary>
""".strip()


FINAL_ANSWER_TEMPLATE = """
Final Answer:
Class: {landform_class}
Confidence: {confidence}
Reasoning: {reasoning_summary}
""".strip()


TOOL_DESCRIPTIONS = """
Available tools and expected inputs:

1) classify
   - Purpose: Run MIL classifier on an image.
   - Input JSON: {"image_id": "ESP_012345_1234"}
   - Returns: class, confidence, per_class_probs, top_3_tiles, attention_summary.

2) query_rag
   - Purpose: Retrieve relevant scientific excerpts from Mars landform corpus.
   - Input JSON: {"query": "diagnostic criteria for LVF", "class_filter": "LVF"}
   - class_filter is optional.
   - Returns: relevant_excerpts, sources, class_tags.

3) analyze_mola
   - Purpose: Read precomputed MOLA terrain features and interpret them.
   - Input JSON: {"image_id": "ESP_012345_1234"}
   - Returns: elevation, slope_mean, slope_std, TPI, TRI, roughness, lobateness, curvature, interpretation.

4) zoom_tile
   - Purpose: Inspect selected high-value tiles using image statistics and embedding similarity.
   - Input JSON examples:
     {"image_id": "ESP_012345_1234", "tile_indices": "top_attention"}
     {"image_id": "ESP_012345_1234", "tile_indices": [0, 7, 14]}
   - Returns: tile_descriptions, tile_paths, attention_weights.

5) regional_context
   - Purpose: Analyze nearby image predictions and latitude priors.
   - Input JSON: {"image_id": "ESP_012345_1234", "radius_km": 100}
   - radius_km is optional.
   - Returns: neighbors, cluster_stats, latitude_context.
""".strip()


def format_react_prompt(
    image_id: str,
    tool_names: Iterable[str],
    react_template: str = REACT_TEMPLATE,
    scratchpad: str = "",
) -> str:
    tool_list = ", ".join(tool_names)
    return (
        f"Classify image_id={image_id}.\\n"
        f"You may use these tools: {tool_list}.\\n\\n"
        f"{TOOL_DESCRIPTIONS}\\n\\n"
        f"{react_template}\\n\\n"
        "Current trace:\\n"
        f"{scratchpad.strip() if scratchpad else '(no prior observations)'}\\n"
        "Respond with the next Thought/Action block or Final Answer."
    )
