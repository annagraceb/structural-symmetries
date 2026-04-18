"""Replace inline author-year mentions in main_body.tex with \\citep/\\citet."""

import re
from pathlib import Path

BODY = Path(__file__).parent / "main_body.tex"


REPL = [
    # (regex, replacement)   [use \citet for in-flow "X et al. did Y", \citep for parenthetical]
    (r"Kornblith et al\.,?\s*2019", r"\\citet{kornblith2019cka}"),
    (r"Raghu et al\.,?\s*2017", r"\\citet{raghu2017svcca}"),
    (r"Morcos et al\.,?\s*2018", r"\\citet{morcos2018pwcca}"),
    (r"Chughtai, Chan\s*\\&\s*Nanda,?\s*2023", r"\\citet{chughtai2023universality}"),
    (r"Chughtai et al\.,?\s*2023", r"\\citet{chughtai2023universality}"),
    (r"Bansal, Nakkiran\s*\\&\s*Barak,?\s*2021", r"\\citet{bansal2021stitching}"),
    (r"Bansal et al\.,?\s*2021", r"\\citet{bansal2021stitching}"),
    (r"Lenc\s*\\&\s*Vedaldi,?\s*2015", r"\\citet{lenc2015understanding}"),
    (r"Wang et al\.,?\s*2023", r"\\citet{wang2023iw}"),
    (r"McGrath et al\.,?\s*2023", r"\\citet{mcgrath2023hydra}"),
    (r"Rushing\s*\\&\s*Nanda,?\s*2024", r"\\citet{rushing2024selfrepair}"),
    (r"Makelov, Lange\s*\\&\s*Nanda,?\s*2023", r"\\citet{makelov2023subspace}"),
    (r"Makelov et al\.,?\s*2023", r"\\citet{makelov2023subspace}"),
    (r"Schiffman,?\s*2026", r"\\citet{schiffman2026cores}"),
    (r"Nanda et al\.,?\s*2023", r"\\citet{nanda2023grokking}"),
    (r"Elhage et al\.,?\s*2022", r"\\citet{elhage2022superposition}"),
    (r"Abid et al\.,?\s*2018", r"\\citet{abid2018cpca}"),
    (r"Alain\s*\\&\s*Bengio,?\s*2016", r"\\citet{alain2017probes}"),
    (r"Hewitt\s*\\&\s*Liang,?\s*2019", r"\\citet{hewitt2019designing}"),
    (r"Elazar et al\.,?\s*2021", r"\\citet{elazar2021amnesic}"),
    (r"Gurnee et al\.,?\s*2024", r"\\citet{gurnee2024universal}"),
    (r"Liu et al\.\s*``Omnigrok''", r"\\citet{liu2022omnigrok}"),
    (r"Liu et al\. ``Omnigrok''", r"\\citet{liu2022omnigrok}"),
    (r"Conmy et al\.,?\s*2023", r"\\citet{conmy2023acdc}"),
    (r"Chan et al\.,?\s*2022", r"\\citet{chan2022scrubbing}"),
    (r"Li, Yosinski et al\.", r"\\citet{li2016convergent}"),
    # loose/parenthetical names without year
    (r"Bansal\s+2021", r"\\citep{bansal2021stitching}"),
    (r"Chughtai\s+2023", r"\\citep{chughtai2023universality}"),
    (r"Wang\s+2023", r"\\citep{wang2023iw}"),
    (r"McGrath\s+2023", r"\\citep{mcgrath2023hydra}"),
    (r"Schiffman\s+2026", r"\\citep{schiffman2026cores}"),
    (r"Makelov\s+2023", r"\\citep{makelov2023subspace}"),
    (r"Rushing\s*\\&\s*Nanda\s+2024", r"\\citep{rushing2024selfrepair}"),
    # arxiv IDs in prose
    (r"\(arXiv:2602\.22600\)", r"\\citep{schiffman2026cores}"),
    (r"\(arXiv:2603\.14833\)", r"\\citep{ablaterescue2026}"),
    (r"arXiv:2311\.17030", r"\\citep{makelov2023subspace}"),
    (r"arXiv:2307\.15771", r"\\citep{mcgrath2023hydra}"),
    (r"arXiv:2401\.12181", r"\\citep{gurnee2024universal}"),
]


def main():
    txt = BODY.read_text()
    for pat, repl in REPL:
        before = txt
        txt = re.sub(pat, repl, txt)
        n = len(re.findall(pat, before))
        if n:
            print(f"{n:>3} × {pat}")
    BODY.write_text(txt)
    print(f"Wrote {BODY}")


if __name__ == "__main__":
    main()
