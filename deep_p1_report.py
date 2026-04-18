"""Analyze 6-layer zoo P1 results; apply PHASE2_PREREGISTRATION.md decision rule."""

import json
import os
import numpy as np


RESULTS_PATH = "results/deep/p1_results.json"


def bootstrap_ci(arr, n=1000, alpha=0.05):
    arr = np.asarray([x for x in arr if x is not None])
    if len(arr) == 0:
        return None, None, None
    rng = np.random.default_rng(42)
    stats = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)])
    return float(arr.mean()), float(np.quantile(stats, alpha/2)), float(np.quantile(stats, 1-alpha/2))


def main():
    d = json.load(open(RESULTS_PATH))
    primary = d["primary"]
    print("=" * 90)
    print("DEEP ZOO (6 layers) P1 — per-site results at k=8")
    print("=" * 90)

    rows = []
    for site in sorted(primary.keys()):
        sd = primary[site]
        layer = sd["layer"]
        mdls = sd["models"]
        def extract(cond):
            xs = [m[cond].get("8") or m[cond].get(8) for m in mdls.values()]
            return [x for x in xs if x is not None]
        ss = extract("shared")
        cc = extract("complement_top_k")
        jj = extract("joint")
        rr_flat = []
        for m in mdls.values():
            r = m["random"].get("8") or m["random"].get(8) or []
            rr_flat.extend(r)
        hl = [j - c for j, c in zip(jj, cc)]

        rows.append((layer, site, len(mdls),
                      bootstrap_ci(ss), bootstrap_ci(cc), bootstrap_ci(jj),
                      float(np.mean(rr_flat)) if rr_flat else 0.0,
                      bootstrap_ci(hl)))

    rows.sort()
    print(f"{'layer':>5} {'site':>22} {'n':>3}  {'shared':>24} {'comp':>24} {'joint':>24} {'rand':>7} {'hidden':>24}")
    print("-" * 145)
    def fmt(c):
        if c[0] is None: return "—"
        return f"{c[0]:.4f} [{c[1]:.3f},{c[2]:.3f}]"
    for layer, site, n, s, c, j, r, h in rows:
        print(f"{layer:>5} {site:>22} {n:>3}  {fmt(s):>24} {fmt(c):>24} {fmt(j):>24} {r:>7.4f} {fmt(h):>24}")

    # Verdict against PHASE2_PREREGISTRATION.md
    print()
    print("=" * 90)
    print("VERDICT against PHASE2_PREREGISTRATION.md")
    print("=" * 90)

    def shared_at(layer_idx):
        for r in rows:
            if r[0] == layer_idx and "result_0" in r[1]:
                return r[3]
        return None

    def hidden_at(layer_idx):
        for r in rows:
            if r[0] == layer_idx and "result_0" in r[1]:
                return r[7]
        return None

    s1 = shared_at(1)
    s3 = shared_at(3)
    s5 = shared_at(5)
    h3 = hidden_at(3)
    h5 = hidden_at(5)

    print()
    print(f"Key data points (shared single-subspace drop):")
    if s1: print(f"  layer 1: {s1[0]:.4f} [{s1[1]:.3f}, {s1[2]:.3f}]")
    if s3: print(f"  layer 3: {s3[0]:.4f} [{s3[1]:.3f}, {s3[2]:.3f}]")
    if s5: print(f"  layer 5: {s5[0]:.4f} [{s5[1]:.3f}, {s5[2]:.3f}]")
    print()
    print("Pre-registered hypotheses:")
    print("  H1 (absolute depth): expects shared near-zero at LAYER 3")
    print("  H2 (last-residual):  expects shared near-zero at LAYER 5")
    print("  H3 (normalized):     expects shared near-zero at LAYER 5")
    print()

    verdict = "INCONCLUSIVE"
    if s1 and s3 and s5:
        near_zero_3 = s3[0] < 0.005
        near_zero_5 = s5[0] < 0.005
        large_5 = s5[0] > 0.1
        large_3 = s3[0] > 0.1
        if near_zero_5 and (large_3 or s3[0] > s5[0]):
            verdict = "H2/H3 (last-residual) — shared dead at layer 5; shared non-trivial at earlier layers"
        elif near_zero_3 and (large_5 or s5[0] > s3[0]):
            verdict = "H1 (absolute depth) — shared dead at layer 3 specifically; shared non-zero at layer 5"
        elif near_zero_3 and near_zero_5:
            verdict = "BOTH layers 3 and 5 shared-dead — ambiguous between depth and last-residual; need deeper zoo"
        elif s1[0] > s3[0] > s5[0]:
            verdict = "MONOTONE GRADIENT — shared dominance decays smoothly with depth"
        else:
            verdict = "UNEXPECTED PATTERN — report as exploratory"

    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
