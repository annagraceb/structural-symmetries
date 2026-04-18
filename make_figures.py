"""Generate key figures for the paper."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_main_zoo():
    return json.load(open("results/p1/p1_results.json"))


def load_joint_ablation():
    return json.load(open("results/p1/joint_ablation.json"))


def load_modp():
    return json.load(open("results/modp/p1_results.json"))


def load_layer3_expanded():
    return json.load(open("results/p1/layer3_expanded.json"))


def load_additional_sites():
    return json.load(open("results/p1/additional_sites.json"))


def bootstrap_ci(arr, n=1000, alpha=0.05):
    arr = np.asarray(arr)
    rng = np.random.default_rng(42)
    stats = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)])
    return arr.mean(), np.quantile(stats, alpha/2), np.quantile(stats, 1-alpha/2)


def fig1_joint_ablation():
    """Bar chart: shared / complement / joint at all joint-ablation sites."""
    d = load_joint_ablation()
    sites = list(d["per_site"].keys())

    shared = []
    comp = []
    joint = []
    shared_err = []
    comp_err = []
    joint_err = []
    for s in sites:
        models = d["per_site"][s]["models"]
        ss = [m["shared_drop"] for m in models.values()]
        cc = [m["complement_drop"] for m in models.values()]
        jj = [m["joint_drop"] for m in models.values()]
        for vals, arr, err in [(ss, shared, shared_err), (cc, comp, comp_err), (jj, joint, joint_err)]:
            m, lo, hi = bootstrap_ci(vals)
            arr.append(m)
            err.append([m - lo, hi - m])

    x = np.arange(len(sites))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, shared, w, label="shared alone", yerr=np.array(shared_err).T,
            color="#1f77b4", capsize=3)
    ax.bar(x, comp, w, label="complement alone", yerr=np.array(comp_err).T,
            color="#ff7f0e", capsize=3)
    ax.bar(x + w, joint, w, label="joint (shared ∪ complement)",
            yerr=np.array(joint_err).T, color="#2ca02c", capsize=3)
    ax.set_ylabel("Accuracy drop")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in sites], fontsize=9)
    ax.set_title("Joint ablation reveals hidden load invisible to single-subspace ablation\n"
                  "(main zoo, k=8, 33 models per site)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_joint_ablation.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved {path}")


def fig2_layer_dependence():
    """Shared single-ablation drop vs layer in the main zoo."""
    # Layer 1, 2, 3 data from main zoo + expanded + additional sites.
    main = load_main_zoo()
    expanded = load_layer3_expanded()
    additional = load_additional_sites()

    entries = []  # (site, layer, shared, complement)
    def mean_drop(sd, cond, k=8):
        drops = []
        for m in sd["models"].values():
            c = m.get(cond, {})
            e = c.get(str(k))
            if isinstance(e, dict) and e.get("drop") is not None:
                drops.append(e["drop"])
        return float(np.mean(drops)) if drops else None

    for site, sd in (list(main["primary"].items()) +
                      list(additional["primary"].items()) +
                      list(expanded["primary"].items())):
        layer = sd["layer"]
        s = mean_drop(sd, "shared")
        c = mean_drop(sd, "complement_top_k")
        if s is None or c is None:
            continue
        # Exclude "dead" layer-3 positions (result_5, operands) and the control
        if "plus" in site or "operand" in site:
            continue
        if "result_5" in site:
            continue
        entries.append((site, layer, s, c))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    layers_s = [e[1] for e in entries]
    shared_vals = [e[2] for e in entries]
    comp_vals = [e[3] for e in entries]
    names = [e[0] for e in entries]

    # Scatter with jitter
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(layers_s)) - 0.5) * 0.15
    x = np.array(layers_s) + jitter

    ax.scatter(x - 0.08, shared_vals, s=60, label="shared-alone drop",
                color="#1f77b4", edgecolor="black", zorder=3)
    ax.scatter(x + 0.08, comp_vals, s=60, label="complement-alone drop",
                color="#ff7f0e", edgecolor="black", zorder=3)

    # Label the main zoo sites
    for name, xi, s, c in zip(names, x, shared_vals, comp_vals):
        short = name.replace("layer", "L").replace("_result_", ".r")
        ax.annotate(short, (xi, max(s, c) + 0.02), fontsize=7, ha="center")

    ax.set_xlabel("Layer index (main zoo, n_layers=4)")
    ax.set_ylabel("Single-subspace ablation drop at k=8")
    ax.set_title("Main-zoo layer dependence: shared is primary at layer 1,\n"
                  "redundant backup at layer 3 (single-subspace view)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(-0.02, 0.9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_layer_dependence.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved {path}")


def fig3_modp_vs_main():
    """Main zoo vs mod-p hidden-load at equivalent sites."""
    main_joint = load_joint_ablation()
    modp = load_modp()

    def hidden(models, k=8):
        hl = [m["joint_drop"] - m["complement_drop"] for m in models.values()]
        return bootstrap_ci(hl)

    def modp_hidden(sd, k=8):
        ms = sd["models"]
        jj = [m["joint"].get(str(k)) or m["joint"].get(k) for m in ms.values()]
        cc = [m["complement_top_k"].get(str(k)) or m["complement_top_k"].get(k) for m in ms.values()]
        hl = [j - c for j, c in zip(jj, cc) if j is not None and c is not None]
        return bootstrap_ci(hl) if hl else (0, 0, 0)

    # Main-zoo joint-ablation sites
    main_sites = list(main_joint["per_site"].keys())
    main_hl = [hidden(main_joint["per_site"][s]["models"]) for s in main_sites]

    modp_sites = [s for s in modp["primary"] if s != "layer3_plus"]
    modp_hl = [modp_hidden(modp["primary"][s]) for s in modp_sites]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    all_sites = main_sites + modp_sites
    all_hl = main_hl + modp_hl
    all_means = [x[0] for x in all_hl]
    all_errs = np.array([[x[0] - x[1], x[2] - x[0]] for x in all_hl]).T
    colors = ["#1f77b4"] * len(main_sites) + ["#ff7f0e"] * len(modp_sites)

    x = np.arange(len(all_sites))
    ax.bar(x, all_means, yerr=all_errs, color=colors, capsize=4, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in all_sites], fontsize=8)
    ax.set_ylabel("Hidden load (joint drop − complement drop)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Hidden load is positive at every tested site\n"
                  "(blue: main zoo, orange: mod-p replication)")
    ax.grid(axis="y", alpha=0.3)
    # Manual legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1f77b4", label="5-digit addition"),
                        Patch(facecolor="#ff7f0e", label="mod-23 arithmetic")],
              loc="upper right")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_modp_vs_main.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved {path}")


def fig4_unit_norm_vs_matched():
    """Shared/random ratio under unit-norm vs variance-matched."""
    decomp = json.load(open("results/p1/extraction_decomposition.json"))
    summary = decomp["summary"]
    sites = list(summary.keys())

    old_ratio = [summary[s]["ratios"]["old_shared/random_unit"] for s in sites]
    new_unit_ratio = [summary[s]["ratios"]["new_shared_unit/random_unit"] for s in sites]

    x = np.arange(len(sites))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, old_ratio, w, label="OLD: unrestricted eigproblem, unit-norm",
            color="#d62728", edgecolor="black")
    ax.bar(x + w/2, new_unit_ratio, w, label="A1 (PCA-restricted) + unit-norm",
            color="#2ca02c", edgecolor="black")
    ax.axhline(1, color="black", linestyle="--", alpha=0.5, label="shared = random")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in sites], fontsize=9)
    ax.set_ylabel("shared drop / random drop  (log scale)")
    ax.set_title("A1 (PCA-restriction) flips the sign of the Step-7 finding\n"
                  "(both protocols: same models, same data, same ablation hook)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_unit_norm_vs_matched.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved {path}")


def main():
    fig1_joint_ablation()
    fig2_layer_dependence()
    fig3_modp_vs_main()
    fig4_unit_norm_vs_matched()


if __name__ == "__main__":
    main()
