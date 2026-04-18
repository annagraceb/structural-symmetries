"""Fig 6: Hidden-load at N-1 layer across three depths."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def boot_ci(arr, n=1000, alpha=0.05):
    arr = np.asarray([x for x in arr if x is not None])
    if len(arr) == 0: return 0, 0, 0
    rng = np.random.default_rng(42)
    s = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)])
    return float(arr.mean()), float(np.quantile(s, alpha/2)), float(np.quantile(s, 1-alpha/2))


def main():
    # Main zoo (4L): average hidden_load across 5 layer-3 result positions + joint ablation baseline
    joint = json.load(open("results/p1/joint_ablation.json"))
    main_hl_layer3 = []  # hidden loads at layer 3 in 4L zoo
    for site, sd in joint["per_site"].items():
        if not site.startswith("layer3_"): continue
        for m in sd["models"].values():
            main_hl_layer3.append(m["joint_drop"] - m["complement_drop"])

    # 6L zoo: layer 5
    deep = json.load(open("results/deep/p1_results.json"))
    hl_6l_l5 = []
    sd5 = deep["primary"]["layer5_result_0"]["models"]
    for m in sd5.values():
        j = m["joint"].get("8") or m["joint"].get(8)
        c = m["complement_top_k"].get("8") or m["complement_top_k"].get(8)
        if j is not None and c is not None:
            hl_6l_l5.append(j - c)

    # 8L zoo: layer 7
    deep8 = json.load(open("results/deep8/p1_results.json"))
    hl_8l_l7 = []
    sd7 = deep8["primary"]["layer7_result_0"]["models"]
    for m in sd7.values():
        j = m["joint"].get("8") or m["joint"].get(8)
        c = m["complement_top_k"].get("8") or m["complement_top_k"].get(8)
        if j is not None and c is not None:
            hl_8l_l7.append(j - c)

    depths = [4, 6, 8]
    main_ci = boot_ci(main_hl_layer3)
    six_ci = boot_ci(hl_6l_l5)
    eight_ci = boot_ci(hl_8l_l7)
    means = [main_ci[0], six_ci[0], eight_ci[0]]
    errs = np.array([
        [main_ci[0] - main_ci[1], main_ci[2] - main_ci[0]],
        [six_ci[0] - six_ci[1], six_ci[2] - six_ci[0]],
        [eight_ci[0] - eight_ci[1], eight_ci[2] - eight_ci[0]],
    ]).T

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(depths, means, yerr=errs, marker='D', markersize=14, capsize=6,
                linewidth=2, color="purple")
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.6)
    for x, y in zip(depths, means):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(12, 0),
                    fontsize=11, va="center")

    ax.set_xlabel("Architecture depth (n_layers)", fontsize=11)
    ax.set_ylabel("Hidden load at layer N−1 (joint − complement drop)", fontsize=11)
    ax.set_title("Hidden load at the readout-adjacent layer shrinks with depth\n"
                 "(main zoo, 6-layer zoo, 8-layer zoo)", fontsize=11)
    ax.set_xticks(depths)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 0.55)

    plt.tight_layout()
    path = "figures/fig6_hidden_load_vs_depth.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
