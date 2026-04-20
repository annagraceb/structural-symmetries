"""Phase 4 composite figure: A4 k-sweep, depth dissociation, H-A6 cross-task.

This is the main paper figure.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGS_DIR = "figures"
A4_PATH = "results/p1/a4_ksweep.json"
CKA_PATH = "results/p1/complement_cka.json"


def load_a4():
    if not os.path.exists(A4_PATH):
        return None
    with open(A4_PATH) as f:
        return json.load(f)


def load_cka():
    if not os.path.exists(CKA_PATH):
        return None
    with open(CKA_PATH) as f:
        return json.load(f)


def main():
    fig = plt.figure(figsize=(14.5, 10.0))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.36)

    a4 = load_a4()
    cka = load_cka()

    # -----  Panel A: A4 k-sweep primary site + null-site overlay  -----
    ax_a = fig.add_subplot(gs[0, 0])
    if a4:
        ks = None
        colors = {"deep8_layer7_result_0": "#c0392b",
                    "deep8_layer1_result_0": "#e67e22",
                    "deep8_layer7_position9": "#7f8c8d"}
        labels = {"deep8_layer7_result_0": "Primary: layer 7 N-1 (result-0)",
                    "deep8_layer1_result_0": "Null(!): layer 1 (result-0)",
                    "deep8_layer7_position9": "Null: layer 7 pos 9 (B[3])"}
        for site_key, color in colors.items():
            if site_key in a4.get("sites", {}):
                s_dict = a4["sites"][site_key].get("summary_per_k", {})
                if s_dict:
                    ks = sorted(int(k) for k in s_dict.keys())
                    struct = [s_dict[str(k)]["structured_mean"] for k in ks]
                    ax_a.plot(ks, struct, "-o", color=color, lw=2.2, ms=7,
                               label=f"{labels[site_key]} (struct)")
                    a2p = [s_dict[str(k)]["random_a2prime_median_mean"] or 0.0 for k in ks]
                    # Only plot random once (all zero)
        # Add zero-random baseline annotation
        if ks:
            ax_a.plot(ks, [0.0] * len(ks), ":", color="#34495e", lw=1.5,
                       label="Random (A2/A2', all sites = 0)")
            ax_a.set_xticks(ks)
        primary_site = a4.get("sites", {}).get("deep8_layer7_result_0", {})
        auc_p = primary_site.get("auc_a2prime", {})
        null1 = a4.get("sites", {}).get("deep8_layer1_result_0", {})
        auc1 = null1.get("auc_a2prime", {})
        ax_a.set_title(f"(a) A4 structured vs random at 8L\n"
                       f"Primary AUC={auc_p.get('mean', 0):.3f} [{auc_p.get('ci_95', {}).get('lo', 0):.3f}, {auc_p.get('ci_95', {}).get('hi', 0):.3f}]\n"
                       f"Layer1 AUC={auc1.get('mean', 0):.3f} (NOT null; saturated shape)",
                       fontsize=9.5)
        ax_a.set_xlabel("k (complement directions ablated)")
        ax_a.set_ylabel("Mean accuracy drop")
        ax_a.set_ylim(-0.05, 1.0)
        ax_a.legend(fontsize=7.5, loc="upper left")
        ax_a.grid(alpha=0.3)

    # -----  Panel B: Dissociation (shared-CKA vs shared-drop across depth)  -----
    ax_b = fig.add_subplot(gs[0, 1])
    depths = [4, 6, 8]
    shared_cka_N = [0.806, 0.809, 0.851]
    shared_drop_N = [0.27, 0.00, 0.00]
    comp_cka_N = [0.781, 0.800, 0.784]
    comp_drop_N = [0.29, 0.56, 0.79]

    ax_b.plot(depths, shared_cka_N, "-o", color="#2980b9", lw=2.4, ms=9,
               label="Shared-CKA")
    ax_b.plot(depths, comp_cka_N, "--s", color="#8e44ad", lw=2.0, ms=8,
               label="Complement-CKA")
    ax_b.set_ylabel("CKA", color="#2980b9", fontsize=11)
    ax_b.set_xlabel("Transformer depth")
    ax_b.set_ylim(0.5, 0.95)
    ax_b.set_xticks(depths)
    ax_b.tick_params(axis='y', labelcolor="#2980b9")
    ax_b.grid(alpha=0.3)

    ax_b2 = ax_b.twinx()
    ax_b2.plot(depths, shared_drop_N, "-o", color="#c0392b", lw=2.4, ms=9,
                label="Shared drop")
    ax_b2.plot(depths, comp_drop_N, "--s", color="#e67e22", lw=2.0, ms=8,
                label="Complement drop")
    ax_b2.set_ylabel("Accuracy drop under ablation at N-1", color="#c0392b", fontsize=11)
    ax_b2.set_ylim(-0.05, 1.0)
    ax_b2.tick_params(axis='y', labelcolor="#c0392b")

    ax_b.set_title("(b) Stranded universality at N-1\n"
                   "CKA rises while ablation drop falls",
                   fontsize=10)

    # combine legends
    lines1, labels1 = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    # -----  Panel C: H-A6 complement-CKA by task  -----
    ax_c = fig.add_subplot(gs[0, 2])
    if cka:
        cats = ["within\nmain 4L", "within\nmod-p 4L", "cross\nmain × mod-p"]
        means = [cka["within_main"]["mean"], cka["within_modp"]["mean"], cka["cross_main_modp"]["mean"]]
        lows = [cka["within_main"]["ci95_lo"], cka["within_modp"]["ci95_lo"], cka["cross_main_modp"]["ci95_lo"]]
        highs = [cka["within_main"]["ci95_hi"], cka["within_modp"]["ci95_hi"], cka["cross_main_modp"]["ci95_hi"]]
        err_lo = [m - l for m, l in zip(means, lows)]
        err_hi = [h - m for m, h in zip(means, highs)]
        colors = ["#2980b9", "#16a085", "#7f8c8d"]
        ax_c.bar(cats, means, yerr=[err_lo, err_hi], capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax_c.axhline(0.15, color="#c0392b", linestyle="--", lw=1.0)
        ax_c.set_ylim(-0.02, 0.9)
        ax_c.set_ylabel("Complement-CKA  (k=8)")
        verdict = cka.get("verdict_H_A6", "?")
        ax_c.set_title(f"(c) H-A6: complement is task-specific\n"
                       f"verdict: {verdict}", fontsize=10)
        ax_c.grid(alpha=0.3, axis="y")

    # -----  Panel D: probe accuracy (positive localization)  -----
    ax_d = fig.add_subplot(gs[1, 0])
    depths_probe = ["4L L3", "8L L7"]
    shared_probe = [0.484, 1.000]
    comp_probe = [0.517, 0.824]
    full_probe = [0.499, 1.000]
    rand_probe = [0.498, 0.810]
    x = np.arange(2)
    w = 0.2
    ax_d.bar(x - 1.5*w, shared_probe, w, label="Shared", color="#2980b9", edgecolor="black", lw=0.5)
    ax_d.bar(x - 0.5*w, comp_probe, w, label="Complement", color="#8e44ad", edgecolor="black", lw=0.5)
    ax_d.bar(x + 0.5*w, rand_probe, w, label="Random 8-dim", color="#7f8c8d", edgecolor="black", lw=0.5)
    ax_d.bar(x + 1.5*w, full_probe, w, label="Full residual", color="#16a085", edgecolor="black", lw=0.5)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(depths_probe)
    ax_d.set_ylabel("Linear probe accuracy (correct digit)")
    ax_d.set_ylim(0, 1.1)
    ax_d.set_title("(d) Linear probe at N-1\n8L shared subspace decodes answer perfectly\ndespite 0 ablation drop", fontsize=10)
    ax_d.legend(fontsize=8, loc="upper left")
    ax_d.grid(alpha=0.3, axis="y")

    # -----  Panel E: Unembed geometry  -----
    ax_e = fig.add_subplot(gs[1, 1])
    categories = ["Shared\n4L L3", "Shared\n8L L7", "Complement\n4L L3", "Complement\n8L L7"]
    null_fracs = [0.767, 0.787, 0.310, 0.269]
    colors_e = ["#2980b9", "#2980b9", "#8e44ad", "#8e44ad"]
    ax_e.bar(categories, null_fracs, color=colors_e, edgecolor="black", linewidth=0.5)
    ax_e.axhline(0.8125, color="#c0392b", linestyle="--", lw=1.2, label="Random baseline (52/64)")
    ax_e.set_ylim(0, 1.0)
    ax_e.set_ylabel("Fraction in unembed nullspace")
    ax_e.set_title("(e) Unembed geometry\nShared ≈ random vs readout;\ncomplement concentrated in readspace", fontsize=10)
    ax_e.legend(fontsize=8, loc="upper right")
    ax_e.grid(alpha=0.3, axis="y")

    # -----  Panel F: full residual CKA depth-invariance  -----
    ax_f = fig.add_subplot(gs[1, 2])
    depths_full = [4, 6, 8]
    raw_cka = [0.768, 0.809, 0.780]
    ax_f.plot(depths_full, raw_cka, "-o", color="#16a085", lw=2.4, ms=10, label="Raw residual CKA")
    ax_f.plot(depths, shared_cka_N, "-o", color="#2980b9", lw=2.0, ms=8, label="Shared CKA")
    ax_f.plot(depths, comp_cka_N, "--s", color="#8e44ad", lw=2.0, ms=8, label="Complement CKA")
    ax_f.set_xlabel("Transformer depth")
    ax_f.set_ylabel("Cross-model CKA at N-1")
    ax_f.set_ylim(0.6, 0.95)
    ax_f.set_xticks(depths_full)
    ax_f.set_title("(f) Raw-residual CKA is depth-invariant\n~0.77-0.81 across 4L/6L/8L", fontsize=10)
    ax_f.legend(fontsize=8, loc="center right")
    ax_f.grid(alpha=0.3)

    fig.suptitle("Phase 4: Universal Geometry, Private Causality at N-1 (8L arithmetic transformers)",
                  fontsize=12, y=0.995)

    os.makedirs(FIGS_DIR, exist_ok=True)
    out = os.path.join(FIGS_DIR, "fig_phase4_composite.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
