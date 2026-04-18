"""Build the paper PDF from PAPER_DRAFT.md via markdown → HTML → wkhtmltopdf."""

import os
import subprocess
import sys
import base64
import re

import markdown


INPUT_MD = "PAPER_DRAFT.md"
HTML_PATH = "/tmp/paper_build.html"
PDF_PATH = "PAPER.pdf"
FIG_DIR = "figures"


CSS = """
<style>
  body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.5;
    max-width: 780px;
    margin: 2.2em auto;
    padding: 0 1.5em;
    color: #222;
  }
  h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 0.3em; margin-top: 1.2em; }
  h2 { font-size: 15pt; margin-top: 1.8em; border-bottom: 1px solid #888; padding-bottom: 0.2em; }
  h3 { font-size: 12pt; margin-top: 1.4em; color: #333; }
  h4 { font-size: 11pt; margin-top: 1.2em; color: #444; }
  p  { text-align: justify; margin: 0.55em 0; }
  code {
    font-family: 'Courier New', monospace;
    background: #f4f4f4;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 9.5pt;
  }
  pre code {
    display: block;
    padding: 0.8em;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.3;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
  }
  th, td {
    border: 1px solid #bbb;
    padding: 0.4em 0.6em;
    text-align: left;
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
  hr { border: none; border-top: 1px solid #ccc; margin: 2em 0; }
  .caption {
    text-align: center;
    font-size: 9pt;
    color: #555;
    margin-top: -0.5em;
  }
</style>
"""


FIGURE_CAPTIONS = {
    "fig1_joint_ablation.png": "Fig. 1 — Joint ablation reveals hidden load invisible to single-subspace ablation (main zoo, k=8, 33 models per site).",
    "fig2_layer_dependence.png": "Fig. 2 — Single-subspace drops by layer in the main 4-layer zoo.",
    "fig3_modp_vs_main.png": "Fig. 3 — Hidden load across sites, main zoo vs mod-23 replication.",
    "fig4_unit_norm_vs_matched.png": "Fig. 4 — A1 (PCA restriction) flips the Step-7 sign at all three primary sites.",
    "fig5_deep_layer_sweep.png": "Fig. 5 — 6-layer zoo layer sweep: shared/complement/joint drops and hidden-load profile.",
    "fig6_hidden_load_vs_depth.png": "Fig. 6 — Hidden load at the N-1 layer shrinks with depth (4-layer, 6-layer, 8-layer zoos).",
}


def inline_images(html: str) -> str:
    """Embed figures as data URIs so wkhtmltopdf doesn't need filesystem access."""
    out = html
    for fig_name in FIGURE_CAPTIONS:
        fpath = os.path.join(FIG_DIR, fig_name)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        caption = FIGURE_CAPTIONS[fig_name]
        img_block = (
            f'\n<div><img src="{data_uri}" alt="{fig_name}" /></div>'
            f'\n<div class="caption">{caption}</div>\n'
        )
        # Insert image block BEFORE the paragraph mentioning "(Fig. N"
        fig_num = re.search(r"fig(\d)", fig_name).group(1)
        pattern = rf'<p><em>\(Fig\. {fig_num}'
        out = re.sub(pattern, img_block + r'<p><em>(Fig. ' + fig_num, out, count=1)
    return out


def main():
    md = open(INPUT_MD).read()
    html_body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html_body = inline_images(html_body)
    html = f"<!DOCTYPE html>\n<html><head>{CSS}</head><body>\n{html_body}\n</body></html>"
    with open(HTML_PATH, "w") as f:
        f.write(html)

    cmd = [
        "wkhtmltopdf",
        "--enable-local-file-access",
        "--page-size", "A4",
        "--margin-top", "18mm",
        "--margin-bottom", "18mm",
        "--margin-left", "18mm",
        "--margin-right", "18mm",
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
