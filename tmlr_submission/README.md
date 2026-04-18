# TMLR submission package

This directory contains the TMLR-formatted LaTeX source for submission to
[Transactions on Machine Learning Research](https://jmlr.org/tmlr/).

## Contents

| File | Purpose |
|------|---------|
| `main.tex` | Wrapper: `\documentclass`, title, author, bibliography hooks. Uses `[preprint]` option for a self-authored non-anonymous build. |
| `main_body.tex` | Paper body (auto-generated from `../PAPER_DRAFT.md` via `md_to_tex.py`, then hand-patched). |
| `refs.bib` | Bibliography database (24 entries). |
| `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex` | TMLR style package (upstream: https://github.com/JmlrOrg/tmlr-style-file). |
| `figures/` | Six PNG figures (fig1-6) referenced from `main_body.tex`. |
| `md_to_tex.py` | Narrow markdown → LaTeX converter used to generate `main_body.tex` initially. |
| `patch_citations.py` | Second-pass converter that rewrites author-year mentions to `\citet{}`/`\citep{}`. |

## To regenerate `main_body.tex` from scratch

```
python3 md_to_tex.py        # markdown → main_body.tex (mechanical conversion)
python3 patch_citations.py  # inline author-year → \citet / \citep
# Then hand-patch figure insertions (already present in committed main_body.tex)
```

## To compile locally (requires texlive)

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Output: `main.pdf`.

## To compile via Overleaf (no local install)

1. Create a new project on [overleaf.com](https://overleaf.com)
2. Upload all files in this directory (including `figures/`)
3. Set `main.tex` as the main document
4. Compile with pdfLaTeX

## To compile for TMLR submission

TMLR's OpenReview uses pdfLaTeX server-side. Upload the entire
directory contents (source + style + figures + bib) via
[OpenReview](https://openreview.net/).

### Submission mode switches

In `main.tex`:

- **Initial (anonymous) submission**: change `\usepackage[preprint]{tmlr}` to `\usepackage{tmlr}`
- **Non-anonymous preprint / arXiv**: keep `\usepackage[preprint]{tmlr}` (current setting)
- **Camera-ready for accepted paper**: change to `\usepackage[accepted]{tmlr}`, set `\month` and `\year`, fill `\openreview` URL

## Author info

`\author{\name Anna Bentley \email mbentley@nttglobal.net \\ \addr Independent Researcher}`

## Known pending tasks before submission

1. Double-check abstract word count (TMLR recommends $\leq$ 300-400 words; current is ~450, may need a trim).
2. The A1/A2/A3 paragraph in the abstract uses `\texttt{C\_shared ...}` — acceptable but `$C_\mathrm{shared}$` would be cleaner. Style choice.
3. Limitations and reproducibility statements are present in-line in the body; TMLR does not require them as standalone sections but a reviewer may ask.
4. Figures are 85-95% text-width; some are dense — consider splitting Fig. 5 into two subfigures if the reviewer asks.
5. `\cite*` calls have not been rebalanced between `\citet` (in-flow) and `\citep` (parenthetical) in every case. Scan main_body.tex for `(\citet{...}` and change to `(\citep{...}` where parenthetical.
