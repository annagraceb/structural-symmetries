"""Phase 4 — generate figures.

fig7_a4_ksweep.png:  structured vs random_a2 vs random_a2prime k-sweep at each site.
fig8_complement_cka.png: within/cross distribution for H-A6.

Reads:
  results/p1/a4_ksweep.json
  results/p1/complement_cka.json
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGS_DIR = "figures"
A4_PATH = "results/p1/a4_ksweep.json"
CKA_PATH = "results/p1/complement_cka.json"


def fig7():
    if not os.path.exists(A4_PATH):
        print(f"WARN: {A4_PATH} missing; skipping fig7")
        return
    with open(A4_PATH) as f:
        data = json.load(f)
    sites = data.get("sites", {})
    if not sites:
        print("WARN: no sites in a4_ksweep.json")
        return

    n = len(sites)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), sharey=False)
    if n == 1:
        axes = [axes]

    site_titles = {
        "deep8_layer7_result_0": "PRIMARY  8L layer 7, result-0 (N-1)",
        "deep8_layer1_result_0": "NULL early-layer  8L layer 1, result-0",
        "deep8_layer7_position9": "NULL wrong-position  8L layer 7, pos 9",
        "permuted_deep8_layer7_result_0": "NULL permuted labels  8L layer 7, result-0",
    }

    for ax, (name, site) in zip(axes, sites.items()):
        summary = site.get("summary_per_k", {})
        ks = sorted(int(k) for k in summary.keys())
        if not ks:
            ax.set_title(name + "\n[no data]")
            continue
        struct = [summary[str(k)]["structured_mean"] for k in ks]
        a2 = [summary[str(k)]["random_a2_median_mean"] for k in ks]
        a2p = [summary[str(k)]["random_a2prime_median_mean"] for k in ks]

        ax.plot(ks, struct, "-o", color="#c0392b", lw=2.0,
                 label="Structured complement")
        ax.plot(ks, a2, "--s", color="#7f8c8d", lw=1.5,
                 label="Random (A2: variance-matched)")
        ax.plot(ks, a2p, ":^", color="#2c3e50", lw=1.5,
                 label="Random (A2': task-proj matched)")
        ax.set_xlabel("k (complement directions ablated)")
        ax.set_ylabel("Accuracy drop")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
        auc = site.get("auc_a2prime", {})
        rank = site.get("ranking_structured_beats_random_a2prime", "?")
        req = site.get("ranking_required_for_primary", 6)
        title = site_titles.get(name, name)
        auc_mean = auc.get("mean", float("nan"))
        auc_lo = auc.get("ci_95", {}).get("lo", float("nan"))
        auc_hi = auc.get("ci_95", {}).get("hi", float("nan"))
        ax.set_title(f"{title}\n"
                     f"AUC(A2') = {auc_mean:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]\n"
                     f"ranking: {rank}/8 (need ≥{req})",
                     fontsize=9)

    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    out = os.path.join(FIGS_DIR, "fig7_a4_ksweep.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def fig8():
    if not os.path.exists(CKA_PATH):
        print(f"WARN: {CKA_PATH} missing; skipping fig8")
        return
    with open(CKA_PATH) as f:
        d = json.load(f)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cats = ["within\nmain 4L", "within\nmodp 4L", "cross\nmain × modp"]
    means = [d["within_main"]["mean"], d["within_modp"]["mean"], d["cross_main_modp"]["mean"]]
    lows = [d["within_main"]["ci95_lo"], d["within_modp"]["ci95_lo"], d["cross_main_modp"]["ci95_lo"]]
    highs = [d["within_main"]["ci95_hi"], d["within_modp"]["ci95_hi"], d["cross_main_modp"]["ci95_hi"]]
    err_lo = [m - l for m, l in zip(means, lows)]
    err_hi = [h - m for m, h in zip(means, highs)]

    colors = ["#2980b9", "#16a085", "#7f8c8d"]
    ax.bar(cats, means, yerr=[err_lo, err_hi], capsize=4, color=colors)
    ax.axhline(0.15, color="#c0392b", linestyle="--", lw=1.0, label="H-A6 gap threshold = 0.15")
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("Complement-CKA  (k=8)")
    verdict = d.get("verdict_H_A6", "?")
    gap = d.get("gap_within_main_minus_cross", {})
    gap_mean = gap.get("mean", float("nan"))
    gap_lo = gap.get("ci95_lo", float("nan"))
    gap_hi = gap.get("ci95_hi", float("nan"))
    ax.set_title(f"H-A6: complement universality is task-specific (verdict: {verdict})\n"
                 f"gap = within_main − cross = {gap_mean:.3f} [{gap_lo:.3f}, {gap_hi:.3f}]",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    out = os.path.join(FIGS_DIR, "fig8_complement_cka.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig7()
    fig8()
