"""Markdown → LaTeX converter for PAPER_DRAFT.md → main_body.tex.

Narrow-scope converter built for the specific markdown idioms used in
PAPER_DRAFT.md. NOT a general-purpose tool. Writes main_body.tex which
is \\input{} from main.tex (main.tex holds \\documentclass, \\title,
\\author, bibliography, etc.).

Invariants it preserves:
  - Sections / subsections from ##/###
  - **bold** / *italic* → \\textbf / \\emph
  - `code` → \\texttt (escapes underscore)
  - Inline math candidates (C_shared, lambda, v^T ...) wrapped in $...$
  - Markdown pipe tables → tabular environments
  - Horizontal rules '---' → dropped (section breaks handle spacing)
  - Em-dashes '—' → '---'
  - Figure image markdown → \\includegraphics with caption
  - Lists (- item / 1. item) → itemize / enumerate
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parent.parent
SRC = ROOT / "PAPER_DRAFT.md"
OUT = Path(__file__).parent / "main_body.tex"


def escape_tex(s: str) -> str:
    """Escape LaTeX special chars in text runs."""
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("$", r"\$")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("~", r"\textasciitilde{}")
         .replace("^", r"\textasciicircum{}")
    )


def inline_code_to_tex(s: str) -> str:
    """Convert `foo` → \\texttt{foo} with underscore escaping inside."""
    def repl(m):
        inner = m.group(1)
        # Inside \texttt we still need to escape specials but underscores
        # can be kept as \_
        safe = inner.replace("\\", r"\textbackslash{}").replace("_", r"\_") \
                    .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#") \
                    .replace("{", r"\{").replace("}", r"\}")
        return r"\texttt{" + safe + "}"
    return re.sub(r"`([^`]+)`", repl, s)


def _unicode_sub(line: str) -> str:
    """Unicode → LaTeX single-char replacements (safe in math or text mode)."""
    repl = {
        "—": "---", "–": "--", "…": r"\ldots{}",
        "λ": r"$\lambda$", "α": r"$\alpha$", "β": r"$\beta$",
        "μ": r"$\mu$", "σ": r"$\sigma$", "Δ": r"$\Delta$",
        "ε": r"$\varepsilon$", "ρ": r"$\rho$", "π": r"$\pi$",
        "Σ": r"$\Sigma$", "θ": r"$\theta$", "φ": r"$\phi$",
        "∪": r"$\cup$", "∩": r"$\cap$", "≈": r"$\approx$",
        "≤": r"$\leq$", "≥": r"$\geq$", "±": r"$\pm$",
        "×": r"$\times$", "→": r"$\to$", "⊤": r"$\top$",
        "·": r"$\cdot$", "∥": r"$\|$",
        "≫": r"$\gg$", "≪": r"$\ll$",
        "−": "-",
        "\u00a0": "~",
        "\u201c": "``", "\u201d": "''",
        "\u2018": "`", "\u2019": "'",
    }
    for k, v in repl.items():
        line = line.replace(k, v)
    return line


def _escape_text_segment(seg: str) -> str:
    return (seg.replace("%", r"\%")
                 .replace("_", r"\_")
                 .replace("#", r"\#")
                 .replace("&", r"\&")
                 .replace("~", r"\textasciitilde{}")
                 .replace("^", r"\textasciicircum{}"))


def escape_text_specials(line: str) -> str:
    """Escape LaTeX specials in text runs. Recurses into \\textbf{}/\\emph{}
    argument groups so bold/italic content is also escaped. Leaves $...$
    math, \\texttt{} (already escaped internally), and bare commands alone.
    """
    out_chars: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "$":
            j = line.find("$", i + 1)
            if j == -1:
                out_chars.append(line[i:])
                break
            out_chars.append(line[i:j+1])
            i = j + 1
            continue
        if ch == "\\":
            m_group = re.match(r"\\(textbf|emph|textit)\{", line[i:])
            if m_group:
                cmd_len = len(m_group.group(0))
                # Find matching close brace
                depth = 1
                j = i + cmd_len
                while j < len(line) and depth > 0:
                    if line[j] == "{":
                        depth += 1
                    elif line[j] == "}":
                        depth -= 1
                    j += 1
                # inside = [i+cmd_len, j-1), closing brace at j-1
                inner = line[i+cmd_len:j-1]
                out_chars.append(line[i:i+cmd_len])
                out_chars.append(escape_text_specials(inner))  # recurse
                out_chars.append("}")
                i = j
                continue
            m_texttt = re.match(r"\\texttt\{[^{}]*\}", line[i:])
            if m_texttt:
                out_chars.append(m_texttt.group(0))
                i += len(m_texttt.group(0))
                continue
            m_cmd = re.match(r"\\[a-zA-Z]+(\{[^{}]*\})*", line[i:])
            if m_cmd:
                out_chars.append(m_cmd.group(0))
                i += len(m_cmd.group(0))
                continue
            out_chars.append(line[i:i+2])
            i += 2
            continue
        # plain text chunk until next $ or \
        j = i
        while j < len(line) and line[j] not in "$\\":
            j += 1
        out_chars.append(_escape_text_segment(line[i:j]))
        i = j
    return "".join(out_chars)


def inline_formatting(line: str) -> str:
    """Apply **bold**, *italic*, unicode, and final TeX escaping.
    Call AFTER inline_code_to_tex() (so \\texttt{} is already in place)."""
    # bold **x**
    line = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", line)
    # italic *x* (non-greedy, avoid matching **..**)
    line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\emph{\1}", line)
    # unicode
    line = _unicode_sub(line)
    # Finally escape raw LaTeX specials in remaining text runs
    line = escape_text_specials(line)
    return line


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    """Consume a markdown table starting at start. Return (tex, next_idx)."""
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1
    if len(rows) < 2:
        return "", start
    # header = rows[0], separator = rows[1], body = rows[2:]
    ncols = len(rows[0])
    col_spec = "|" + "l|" * ncols
    def fmt(cell: str) -> str:
        return inline_formatting(inline_code_to_tex(cell))
    tex = [r"\begin{center}",
           r"\begin{tabular}{" + col_spec + "}",
           r"\hline"]
    tex.append(" & ".join(r"\textbf{" + fmt(c.replace("**","")) + "}" for c in rows[0]) + r" \\")
    tex.append(r"\hline")
    for row in rows[2:]:
        padded = row + [""] * (ncols - len(row))
        tex.append(" & ".join(fmt(c) for c in padded[:ncols]) + r" \\")
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{center}")
    return "\n".join(tex) + "\n", i


FIG_PATTERN = re.compile(r"!\[([^\]]*)\]\(figures/([^)]+)\)")


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_code = False
    code_buf: list[str] = []
    in_abstract = False
    abstract_done = False

    while i < len(lines):
        line = lines[i]

        # Abstract: special-case the section header → environment
        if not abstract_done and line.strip() == "## Abstract":
            out.append(r"\begin{abstract}")
            in_abstract = True
            i += 1
            continue
        if in_abstract:
            # End abstract when next section starts
            if re.match(r"^##\s+\d", line) or line.strip() == "---":
                out.append(r"\end{abstract}")
                in_abstract = False
                abstract_done = True
                # fall through to handle current line normally
            else:
                stripped_a = line.strip()
                if not stripped_a:
                    out.append("")
                    i += 1
                    continue
                out.append(inline_formatting(inline_code_to_tex(line)))
                i += 1
                continue

        # Fenced code
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
                i += 1
                continue
            else:
                in_code = False
                out.append(r"\begin{verbatim}")
                out.extend(code_buf)
                out.append(r"\end{verbatim}")
                i += 1
                continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Top-level title / working-draft header: drop; handled in main.tex
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("**Working draft"):
            # skip until blank line
            while i < len(lines) and lines[i].strip():
                i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            i += 1
            continue

        # Section / subsection / subsubsection
        m = re.match(r"^###\s+(\S+)\s+(.*)$", line)
        if m:
            num, title = m.group(1), m.group(2)
            out.append(r"\subsection{" + inline_formatting(title) + "}")
            i += 1
            continue
        m = re.match(r"^###\s+(.*)$", line)
        if m:
            out.append(r"\subsection{" + inline_formatting(m.group(1)) + "}")
            i += 1
            continue
        m = re.match(r"^##\s+(\S+)\s+(.*)$", line)
        if m:
            title = m.group(2)
            out.append(r"\section{" + inline_formatting(title) + "}")
            i += 1
            continue
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            out.append(r"\section{" + inline_formatting(m.group(1)) + "}")
            i += 1
            continue
        m = re.match(r"^####\s+(.*)$", line)
        if m:
            out.append(r"\paragraph{" + inline_formatting(m.group(1)) + "}")
            i += 1
            continue

        # Figure
        m = FIG_PATTERN.match(stripped)
        if m:
            alt, fname = m.group(1), m.group(2)
            base = fname.rsplit(".", 1)[0]
            label = base
            out.append(r"\begin{figure}[h]")
            out.append(r"\centering")
            out.append(r"\includegraphics[width=0.85\textwidth]{figures/" + fname + "}")
            out.append(r"\caption{" + inline_formatting(alt) + "}")
            out.append(r"\label{fig:" + label + "}")
            out.append(r"\end{figure}")
            i += 1
            continue

        # Table (starts with "|")
        if stripped.startswith("|"):
            tex, ni = parse_table(lines, i)
            if tex:
                out.append(tex)
                i = ni
                continue

        # Bulleted list
        if re.match(r"^[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i]):
                item = re.sub(r"^[-*]\s+", "", lines[i])
                # collect continuation lines (indented)
                i += 1
                while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("\t")):
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            out.append(r"\begin{itemize}")
            for it in items:
                processed = inline_formatting(inline_code_to_tex(it))
                out.append(r"  \item " + processed)
            out.append(r"\end{itemize}")
            continue

        # Numbered list
        if re.match(r"^\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i]):
                item = re.sub(r"^\d+\.\s+", "", lines[i])
                i += 1
                while i < len(lines) and (lines[i].startswith("   ") or lines[i].startswith("\t")):
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            out.append(r"\begin{enumerate}")
            for it in items:
                processed = inline_formatting(inline_code_to_tex(it))
                out.append(r"  \item " + processed)
            out.append(r"\end{enumerate}")
            continue

        # Blockquote
        if line.startswith("> "):
            out.append(r"\begin{quote}")
            while i < len(lines) and lines[i].startswith("> "):
                text = lines[i][2:]
                out.append(inline_formatting(inline_code_to_tex(text)))
                i += 1
            out.append(r"\end{quote}")
            continue

        # Normal paragraph / blank
        if not stripped:
            out.append("")
            i += 1
            continue

        out.append(inline_formatting(inline_code_to_tex(line)))
        i += 1

    return "\n".join(out)


def main():
    md = SRC.read_text()
    tex = convert(md)
    OUT.write_text(tex)
    print(f"Wrote {OUT} ({len(tex)} chars, {tex.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
