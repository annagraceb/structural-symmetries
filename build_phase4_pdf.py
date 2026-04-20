"""Build PHASE4_REPORT.pdf from PHASE4_REPORT.md.

Mirrors build_pdf.py (TMLR paper builder) but targets the Phase 4 report
and Phase 4 figures with their own captions.
"""

import os
import subprocess
import sys
import base64

import markdown


INPUT_MD = "PHASE4_REPORT.md"
HTML_PATH = "/tmp/phase4_build.html"
PDF_PATH = "PHASE4_REPORT.pdf"
FIG_DIR = "figures"


CSS = """
<style>
  body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.45;
    max-width: 820px;
    margin: 2.2em auto;
    padding: 0 1.5em;
    color: #222;
  }
  h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 0.3em; margin-top: 0.4em; }
  h2 { font-size: 14pt; margin-top: 1.6em; border-bottom: 1px solid #888; padding-bottom: 0.2em; page-break-before: avoid; }
  h3 { font-size: 12pt; margin-top: 1.2em; color: #333; }
  h4 { font-size: 10.5pt; margin-top: 1.0em; color: #444; }
  p  { text-align: justify; margin: 0.4em 0; }
  code {
    font-family: 'Courier New', monospace;
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9.5pt;
  }
  pre code {
    display: block;
    padding: 0.6em;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.25;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }
  th, td {
    border: 1px solid #bbb;
    padding: 0.3em 0.5em;
    text-align: left;
    vertical-align: top;
  }
  th { background: #eee; font-weight: bold; }
  blockquote {
    border-left: 4px solid #888;
    padding-left: 1em;
    margin: 1em 0;
    color: #555;
    font-style: italic;
  }
  img {
    max-width: 100%;
    margin: 1em auto;
    display: block;
    page-break-inside: avoid;
  }
  hr { border: none; border-top: 1px solid #ccc; margin: 1.6em 0; }
  .caption {
    text-align: center;
    font-size: 9pt;
    color: #555;
    margin-top: -0.4em;
  }
  ul, ol { margin: 0.3em 0 0.4em 1.2em; padding: 0; }
  li { margin: 0.2em 0; }
</style>
"""


# Appended to the bottom of the PDF as a figure appendix
FIGURE_APPENDIX = [
    ("fig_phase4_composite.png",
     "Fig. 1 — Phase 4 composite. (a) A4 structured-vs-random complement ablation at three 8L sites; (b) stranded universality at N-1 across depths; (c) H-A6 complement-CKA task specificity; (d) linear probe at N-1 (shared, complement, full, random); (e) unembed-nullspace fractions; (f) raw-residual CKA depth-invariance."),
    ("fig_null_sites.png",
     "Fig. 2 — Null-site specificity. Primary site 8L L7 result-0 passes the pre-reg confirm threshold (AUC 0.36 ≥ 0.12); two task-destroyed nulls (layer 7 position 9, permuted-label zoo n=10) pass the null threshold (AUC 0.000 < 0.03); layer 1 null fails as a null (0.62) — disclosed as a pre-reg design error."),
    ("fig9_universality_vs_causality.png",
     "Fig. 3 — Universality-vs-causality dissociation at N-1 across depths. Shared-CKA rises 0.81→0.85 from 4L to 8L while shared ablation drop falls 0.27→0.00. The untrained 8L-seed baseline (dashed) shows shared-CKA ≈ 0.33, placing the trained 0.85 at ~6.5σ above architectural prior."),
    ("fig_r11_defensive.png",
     "Fig. 4 — R11 defensive-evidence figure. (a) Cross-digit resample test: shared silent across all three ablation conditions (drop ≤0.004); complement cross-digit drop exceeds same-digit by +0.07, positively localizing digit-specific causal information in the complement. (b) Untrained (8 seeds, 28 pairs) vs trained (3 seeds) CKA baselines: training adds +0.52 to shared-CKA."),
]


def append_figures(html_body: str) -> str:
    """Embed figures as base64 data URIs at the end of the doc."""
    parts = [html_body, "<hr/>", "<h2>Figure appendix</h2>"]
    for fig_name, caption in FIGURE_APPENDIX:
        fpath = os.path.join(FIG_DIR, fig_name)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        parts.append(f'<div><img src="{data_uri}" alt="{fig_name}" /></div>')
        parts.append(f'<div class="caption">{caption}</div>')
    return "\n".join(parts)


def main():
    with open(INPUT_MD) as f:
        md = f.read()
    html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    html_body = append_figures(html_body)
    html = f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'>{CSS}</head><body>\n{html_body}\n</body></html>"
    with open(HTML_PATH, "w") as f:
        f.write(html)

    cmd = [
        "wkhtmltopdf",
        "--enable-local-file-access",
        "--page-size", "A4",
        "--margin-top", "16mm",
        "--margin-bottom", "16mm",
        "--margin-left", "16mm",
        "--margin-right", "16mm",
        "--encoding", "UTF-8",
        HTML_PATH,
        PDF_PATH,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("wkhtmltopdf failed:")
        print(res.stderr)
        sys.exit(1)
    print(f"Saved {PDF_PATH} ({os.path.getsize(PDF_PATH)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
