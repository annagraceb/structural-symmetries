"""Figure 5: 6-layer zoo layer sweep — shared, complement, joint, hidden_load vs layer."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    d = json.load(open("results/deep/p1_results.json"))

    def boot_ci(arr, n=1000, alpha=0.05):
        arr = np.asarray([x for x in arr if x is not None])
        if len(arr) == 0: return 0, 0, 0
        rng = np.random.default_rng(42)
        stats = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)])
        return float(arr.mean()), float(np.quantile(stats, alpha/2)), float(np.quantile(stats, 1-alpha/2))

    # Extract per-layer means + CIs (exclude layer5_plus control)
    layers = []
    shared = []; shared_err = []
    comp = []; comp_err = []
    joint = []; joint_err = []
    hidden = []; hidden_err = []

    for site, sd in sorted(d['primary'].items(), key=lambda x: (x[1]['layer'], x[0])):
        if "result_0" not in site:
            continue
        layer = sd['layer']
        ms = sd['models']
        def extract(cond):
            xs = [m[cond].get('8') or m[cond].get(8) for m in ms.values()]
            return [x for x in xs if x is not None]
        ss = extract("shared")
        cc = extract("complement_top_k")
        jj = extract("joint")
        hl = [j - c for j, c in zip(jj, cc)]
        layers.append(layer)
        for vals, arr, err in [(ss, shared, shared_err), (cc, comp, comp_err), (jj, joint, joint_err), (hl, hidden, hidden_err)]:
            m, lo, hi = boot_ci(vals)
            arr.append(m)
            err.append([m - lo, hi - m])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: shared, complement, joint drops by layer
    ax1.errorbar(layers, shared, yerr=np.array(shared_err).T, marker='o', markersize=10,
                  capsize=4, label="shared alone", linewidth=2, color="#1f77b4")
    ax1.errorbar(layers, comp, yerr=np.array(comp_err).T, marker='s', markersize=10,
                  capsize=4, label="complement alone", linewidth=2, color="#ff7f0e")
    ax1.errorbar(layers, joint, yerr=np.array(joint_err).T, marker='^', markersize=10,
                  capsize=4, label="joint (shared ∪ complement)", linewidth=2, color="#2ca02c")
    ax1.set_xlabel("Layer index", fontsize=11)
    ax1.set_ylabel("Mean accuracy drop (k=8, 45 models)", fontsize=11)
    ax1.set_title("6-layer zoo: shared is primary at layers 1-4,\n"
                  "DEAD at layer 5 (last-residual-before-unembed)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_xticks(list(range(6)))
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-0.05, 1.0)
    ax1.axvspan(4.5, 5.5, alpha=0.12, color="red", label="_nolegend_")
    ax1.annotate("last-residual\n(shared dead)", xy=(5, 0.0), xytext=(4.5, 0.9),
                  fontsize=9, color="darkred", ha="center",
                  arrowprops=dict(arrowstyle="->", color="darkred"))

    # Right: hidden load profile
    ax2.errorbar(layers, hidden, yerr=np.array(hidden_err).T, marker='D', markersize=10,
                  capsize=4, linewidth=2, color="purple")
    ax2.fill_between(layers, [h - e[0] for h, e in zip(hidden, hidden_err)],
                      [h + e[1] for h, e in zip(hidden, hidden_err)], alpha=0.2, color="purple")
    ax2.set_xlabel("Layer index", fontsize=11)
    ax2.set_ylabel("Hidden load (joint − complement)", fontsize=11)
    ax2.set_title("Hidden load is high in middle layers, low at both endpoints\n"
                  '"computational sandwich" structure', fontsize=11)
    ax2.set_xticks(list(range(6)))
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.5)
    for xi, yi in zip(layers, hidden):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 10),
                      ha="center", fontsize=9)

    plt.tight_layout()
    path = "figures/fig5_deep_layer_sweep.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
