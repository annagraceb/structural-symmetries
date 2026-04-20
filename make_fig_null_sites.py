"""Phase 4 — null-site figure showing specificity."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


A4_PATH = "results/p1/a4_ksweep.json"
OUT = "figures/fig_null_sites.png"


def main():
    with open(A4_PATH) as f:
        data = json.load(f)
    sites = data["sites"]

    ordered = [
        ("deep8_layer7_result_0", "Primary: layer 7, result-0 (N-1)", "#c0392b"),
        ("deep8_layer1_result_0", "Intended null: layer 1, result-0\n(NOT null; layer 1 is compute)", "#e67e22"),
        ("deep8_layer7_position9", "Null: layer 7, position 9\n(wrong position — operand B[3])", "#16a085"),
        ("permuted_deep8_layer7_result_0", "Null: permuted-zoo layer 7 N-1\n(per-model label permutation)", "#2c3e50"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    # Left: k-sweep curves
    for key, label, color in ordered:
        s = sites.get(key, {})
        if s.get("status") == "INSUFFICIENT_MODELS":
            continue
        sp = s.get("summary_per_k", {})
        ks = sorted(int(k) for k in sp.keys())
        if not ks:
            continue
        struct = [sp[str(k)]["structured_mean"] for k in ks]
        ax1.plot(ks, struct, "-o", color=color, lw=2.0, ms=7, label=label)

    ax1.plot([1, 8], [0, 0], ":", color="#7f8c8d", lw=1.5,
              label="Random A2/A2' (zero at all sites)")
    ax1.set_xlabel("k (complement directions ablated)", fontsize=11)
    ax1.set_ylabel("Mean accuracy drop (21 models / 4 for permuted)", fontsize=11)
    ax1.set_ylim(-0.05, 1.0)
    ax1.set_xticks(list(range(1, 9)))
    ax1.grid(alpha=0.3)
    ax1.set_title("(a) Structured k-ablation by site\n"
                  "Primary site: monotonic rise  |  Null sites: exact zero",
                  fontsize=10.5)
    ax1.legend(fontsize=8.5, loc="upper left")

    # Right: AUC bar chart with pre-reg thresholds
    names = []
    aucs = []
    los = []
    his = []
    colors = []
    for key, label, color in ordered:
        s = sites.get(key, {})
        if s.get("status") == "INSUFFICIENT_MODELS":
            continue
        auc = s.get("auc_a2prime", {})
        names.append(label.split("\n")[0])
        aucs.append(auc.get("mean", 0))
        los.append(auc.get("ci_95", {}).get("lo", 0))
        his.append(auc.get("ci_95", {}).get("hi", 0))
        colors.append(color)
    err_lo = [a - l for a, l in zip(aucs, los)]
    err_hi = [h - a for h, a in zip(his, aucs)]
    x = np.arange(len(names))
    ax2.bar(x, aucs, yerr=[err_lo, err_hi], capsize=5, color=colors, edgecolor="black", linewidth=0.6)
    ax2.axhline(0.12, color="#c0392b", linestyle="--", lw=1.2, label="Pre-reg confirm threshold = 0.12")
    ax2.axhline(0.03, color="#16a085", linestyle="--", lw=1.2, label="Pre-reg null threshold = 0.03")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax2.set_ylabel("AUC(structured − random A2') over k=1..8", fontsize=11)
    ax2.set_ylim(-0.02, 0.75)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.set_title("(b) Pre-registered AUC with decision thresholds\n"
                  "Primary passes; both task-destroyed nulls = exact zero",
                  fontsize=10.5)

    fig.suptitle("A4 structured-vs-random null: site specificity is decisive",
                  fontsize=11.5, y=1.02)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
