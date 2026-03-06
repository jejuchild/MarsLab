#!/usr/bin/env python3
"""
Export landing site report as a styled PDF with embedded figures.

Reads:  landing_site_final_report.md + figures/*.png
Writes: landing_site_final_report.pdf

Usage:
  cd backend && python -m analysis.integration.export_pdf
"""

import base64
import os
import re

import markdown
from weasyprint import HTML, CSS

_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_DIR, "figures")
_MD_PATH = os.path.join(_DIR, "landing_site_final_report.md")
_PDF_PATH = os.path.join(_DIR, "landing_site_final_report.pdf")


# ═══════════════════════════════════════════════════════════════
# CSS for the PDF — dark Mars theme, print-optimized
# ═══════════════════════════════════════════════════════════════

CSS_STYLE = """
@page {
    size: A4;
    margin: 2cm 1.8cm;
    @top-center {
        content: "Mars Landing Site Selection — MarsLab v2.0";
        font-size: 8pt;
        color: #666;
    }
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 8pt;
        color: #666;
    }
}

body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #1a1a1a;
    background: #ffffff;
}

h1 {
    font-size: 22pt;
    color: #b71c1c;
    border-bottom: 3px solid #b71c1c;
    padding-bottom: 8px;
    margin-top: 0;
    page-break-before: avoid;
}

h2 {
    font-size: 16pt;
    color: #1565c0;
    border-bottom: 2px solid #1565c0;
    padding-bottom: 5px;
    margin-top: 24px;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    color: #2e7d32;
    margin-top: 18px;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    color: #4527a0;
    margin-top: 14px;
    page-break-after: avoid;
}

p {
    margin-bottom: 8px;
    text-align: justify;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

th {
    background: #1565c0;
    color: white;
    padding: 6px 8px;
    text-align: left;
    font-weight: bold;
    border: 1px solid #0d47a1;
}

td {
    padding: 5px 8px;
    border: 1px solid #ddd;
}

tr:nth-child(even) {
    background: #f5f5f5;
}

tr:nth-child(odd) {
    background: #ffffff;
}

strong {
    color: #b71c1c;
}

blockquote {
    border-left: 4px solid #b71c1c;
    background: #fff3e0;
    padding: 8px 12px;
    margin: 12px 0;
    font-style: italic;
    color: #555;
}

code {
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9pt;
}

hr {
    border: none;
    border-top: 2px solid #ddd;
    margin: 20px 0;
}

ul, ol {
    margin-left: 12px;
    padding-left: 8px;
}

li {
    margin-bottom: 4px;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.figure-caption {
    text-align: center;
    font-size: 9pt;
    color: #666;
    font-style: italic;
    margin-top: -10px;
    margin-bottom: 16px;
}

em {
    color: #555;
}

/* Print optimization */
h2, h3, h4 {
    page-break-after: avoid;
}

table, img {
    page-break-inside: avoid;
}
"""


def embed_image_as_base64(img_path: str) -> str:
    """Convert image file to base64 data URI."""
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(img_path)[1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}.get(ext.strip("."), "image/png")
    return f"data:{mime};base64,{data}"


def build_html():
    """Convert markdown report to styled HTML with embedded figures."""
    
    # Read markdown
    with open(_MD_PATH, "r") as f:
        md_text = f.read()
    
    # ═══ Insert figure references into the markdown ═══
    # We'll add figures at strategic points in the report
    
    figure_insertions = {
        "## Executive Summary": "",  # After title, before exec summary
        
        "### High-Resolution Refinement (Phase 2)": f"""
![Composite Score Heatmap]({embed_image_as_base64(os.path.join(_FIG_DIR, 'fig1_composite_heatmap.png'))})
<p class="figure-caption">Figure 1. Composite landing site score heatmap over Arcadia Planitia (493 points, 0.5° grid). Green star marks the optimal site at 42.0°N, 176.0°E.</p>

![SWIM Ice Consistency]({embed_image_as_base64(os.path.join(_FIG_DIR, 'fig2_swim_ice_heatmap.png'))})
<p class="figure-caption">Figure 2. SWIM subsurface ice consistency map. The anomalous hot spot at 42°N, 176°E shows SWIM avg=0.829 — far exceeding surrounding points.</p>

""",
        
        "**Interpretation**: SpaceX optimizes": f"""
![SpaceX Comparison Map]({embed_image_as_base64(os.path.join(_FIG_DIR, 'fig4_spacex_comparison.png'))})
<p class="figure-caption">Figure 3. MarsLab optimal site vs SpaceX/Golombek (2021) 7 downselected candidates. Red dashed line = SpaceX's &lt;40°N latitude constraint. Our site at 42°N trades solar margin for dramatically higher ice availability.</p>

""",
        
        "### Key Risks and Mitigations": f"""
![Terrain Analysis]({embed_image_as_base64(os.path.join(_FIG_DIR, 'fig3_elevation_contour.png'))})
<p class="figure-caption">Figure 4. MOLA elevation (left) and terrain slope (right) across the Arcadia refinement grid. Star marks optimal site at -4,035m elevation, 0.13° slope.</p>

![SWIM Depth Breakdown]({embed_image_as_base64(os.path.join(_FIG_DIR, 'fig5_swim_depth_breakdown.png'))})
<p class="figure-caption">Figure 5. Left: SWIM ice consistency by depth layer for top 6 sites. Site #1 has anomalous 1-5m and 5m+ readings near 1.0. Right: Multi-criteria radar profile for the optimal site vs top-20 average.</p>

""",
        
        "## Phase 1: Quantitative Rankings": f"""
![Regional Overview]({embed_image_as_base64(os.path.join(_FIG_DIR, 'fig6_regional_overview.png'))})
<p class="figure-caption">Figure 6. Phase 1 regional analysis. Left: Final composite scores for 5 viable candidates (out of 55). Right: Multi-metric comparison across Landing Scorer, SWIM Ice, ISRU Access, and Climate Resilience.</p>

""",
    }
    
    # Insert figures before their anchor points
    for anchor, fig_html in figure_insertions.items():
        if fig_html and anchor in md_text:
            md_text = md_text.replace(anchor, fig_html + anchor)
    
    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'nl2br'],
    )
    
    # Wrap in full HTML document
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Mars Landing Site Selection — Final Report</title>
</head>
<body>
{html_body}
</body>
</html>"""
    
    return html_doc


def export_pdf():
    """Generate the final PDF."""
    print("Building HTML from markdown + figures...")
    html_content = build_html()
    
    # Save intermediate HTML for debugging
    html_path = os.path.join(_DIR, "landing_site_final_report.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"  HTML saved to {html_path}")
    
    print("Rendering PDF with WeasyPrint...")
    html = HTML(string=html_content, base_url=_DIR)
    css = CSS(string=CSS_STYLE)
    html.write_pdf(_PDF_PATH, stylesheets=[css])
    
    # Report file size
    size_mb = os.path.getsize(_PDF_PATH) / (1024 * 1024)
    print(f"  PDF saved to {_PDF_PATH} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    export_pdf()
