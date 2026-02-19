#!/usr/bin/env python3
"""Convert the Arcadia Planitia markdown report to a styled PDF using WeasyPrint."""

import pathlib, markdown
from weasyprint import HTML

REPORT_DIR = pathlib.Path(__file__).parent
MD_FILE = REPORT_DIR / "arcadia_planitia_comprehensive_report.md"
PDF_FILE = REPORT_DIR / "arcadia_planitia_comprehensive_report.pdf"

CSS = """
@page {
    size: A4;
    margin: 2cm 1.8cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9px;
        color: #888;
    }
}
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 11px;
    line-height: 1.5;
    color: #222;
}
h1 {
    font-size: 22px;
    color: #b33000;
    border-bottom: 2px solid #b33000;
    padding-bottom: 6px;
    margin-top: 0;
}
h2 {
    font-size: 16px;
    color: #1a5276;
    border-bottom: 1px solid #aed6f1;
    padding-bottom: 4px;
    margin-top: 28px;
    page-break-after: avoid;
}
h3 {
    font-size: 13px;
    color: #2e4053;
    margin-top: 18px;
    page-break-after: avoid;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 16px 0;
    font-size: 10px;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #ccc;
    padding: 4px 8px;
    text-align: left;
}
th {
    background-color: #1a5276;
    color: white;
    font-weight: 600;
}
tr:nth-child(even) { background-color: #f4f6f7; }
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12px auto;
    page-break-inside: avoid;
}
p { margin: 6px 0; }
strong { color: #1a5276; }
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 20px 0;
}
em { color: #555; }
"""

md_text = MD_FILE.read_text()

# Convert markdown to HTML
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code"],
)

# Wrap in full HTML doc with CSS
html_doc = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body>
</html>
"""

# Generate PDF — base_url points to report dir so images resolve
HTML(string=html_doc, base_url=str(REPORT_DIR)).write_pdf(str(PDF_FILE))
print(f"PDF written: {PDF_FILE}")
print(f"Size: {PDF_FILE.stat().st_size / 1024:.0f} KB")
