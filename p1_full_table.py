"""Comprehensive results table for P1: every site × every k × every condition.

Generates results/p1/p1_full_table.md, a markdown report ready to drop into
PAPER_DRAFT.md. Covers all k in K_VALUES, including saturation k=16.
"""

import json
import os
import numpy as np

DEFAULT_RESULTS = "results/p1/p1_results.json"
DEFAULT_OUT = "results/p1/p1_full_table.md"

PRIMARY_SITES = ["layer1_result_0", "layer2_equals", "layer2_result_0"]
CONTROL_SITE = "layer3_plus"


def cond_drops(site_data, cond, k):
    drops = []
    for m in site_data["models"].values():
        c = m.get(cond, {})
        e = c.get(str(k))
        if isinstance(e, dict) and e.get("drop") is not None:
            drops.append(e["drop"])
    return drops


def random_drops(site_data, k):
    out = []
    for m in site_data["models"].values():
        out.extend(m["random"].get(str(k), []))
    return out


def proj_trace(site_data, cond, k):
    vals = []
    for m in site_data["models"].values():
        v = m.get("projection_trace_variance", {}).get(cond, {}).get(str(k))
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def render(d):
    out = []
    cfg = d["config"]
    out.append(f"# P1 Full Results Table\n")
    out.append(f"_n_models = {cfg['n_models']}, k_values = {cfg['k_values']}, "
               f"n_random_trials = {cfg['n_random_trials']}, d_model = {cfg['d_model']}_\n")

    out.append("## Per-site, per-k mean ablation drop (across 33 models)\n")
    for site in PRIMARY_SITES + [CONTROL_SITE]:
        if site not in d["primary"]:
            continue
        sd = d["primary"][site]
        is_control = site == CONTROL_SITE
        label = site + (" *(CONTROL)*" if is_control else "")
        out.append(f"\n### {label}\n")
        ex = sd["extraction"]
        out.append(f"- d = {ex['d']}, K_pca = {ex['k_pca']}, "
                   f"PCA variance retained = {ex['pca_eigvals_kept_fraction']:.3f}, "
                   f"denom condition number = {ex['denom_condition_number']:.2e}, "
                   f"ridge = {ex['ridge']:.2e}\n")
        out.append(f"- Top-5 ratio eigvals (shared): {[f'{x:.4f}' for x in ex['eigenvalues_first5']]}\n")
        out.append(f"- Bottom-5 ratio eigvals (anti-shared): {[f'{x:.4f}' for x in ex['eigenvalues_last5']]}\n\n")

        rows = ["| k | shared | complement | anti_ortho | anti_raw | random | shared trace | comp trace |",
                "|---|---|---|---|---|---|---|---|"]
        for k in cfg["k_values"]:
            s = cond_drops(sd, "shared", k)
            c = cond_drops(sd, "complement_top_k", k)
            o = cond_drops(sd, "anti_shared_ortho", k)
            r = cond_drops(sd, "anti_shared_raw", k)
            rd = random_drops(sd, k)
            st = proj_trace(sd, "shared", k)
            ct = proj_trace(sd, "complement_top_k", k)
            def m(x): return f"{np.mean(x):.4f}" if x else "—"
            st_s = f"{st:.2f}" if st is not None else "—"
            ct_s = f"{ct:.2f}" if ct is not None else "—"
            rows.append(f"| {k} | {m(s)} | {m(c)} | {m(o)} | {m(r)} | {m(rd)} | "
                        f"{st_s} | {ct_s} |")
        out.append("\n".join(rows) + "\n")

        # Per-variance ordering at this site
        out.append(f"\n#### Per-variance hurt (drop / projection_trace) at primary k = 8:\n\n")
        items = []
        for cond in ["shared", "complement_top_k", "anti_shared_ortho", "anti_shared_raw"]:
            d_c = cond_drops(sd, cond, 8)
            t_c = proj_trace(sd, cond, 8)
            if d_c and t_c and t_c > 1e-10:
                items.append((cond, np.mean(d_c), t_c, np.mean(d_c) / t_c))
        items.sort(key=lambda x: x[3], reverse=True)
        for name, dr, tr, hpr in items:
            out.append(f"- **{name}**: drop = {dr:.4f}, trace = {tr:.4f}, drop/trace = {hpr:.4f}\n")

    # ε-stability section. eps_sweep entries are top-level site dicts (not
    # keyed by site name), since run_site returns per-site data directly.
    if d.get("eps_sweep"):
        out.append("\n## Tikhonov ridge stability check\n\n")
        out.append("Single site (layer1_result_0), single k = 10 (eps-sweep "
                    "config). Tests whether the eigenproblem solution is "
                    "stable under different ridge regularization scales.\n\n")
        out.append("| ε scale | shared drop | anti_shared_raw drop | comp drop | denom cond # | ridge |\n")
        out.append("|---|---|---|---|---|---|\n")
        for eps_str, sd in d["eps_sweep"].items():
            if not sd or "models" not in sd:
                continue
            s = cond_drops(sd, "shared", 10)
            b = cond_drops(sd, "anti_shared_raw", 10)
            c = cond_drops(sd, "complement_top_k", 10)
            ex = sd["extraction"]
            sm = f"{np.mean(s):.4f}" if s else "—"
            bm = f"{np.mean(b):.4f}" if b else "—"
            cm = f"{np.mean(c):.4f}" if c else "—"
            out.append(f"| {eps_str} | {sm} | {bm} | {cm} | {ex['denom_condition_number']:.2e} | {ex['ridge']:.2e} |\n")

    return "".join(out)


def main():
    d = json.load(open(DEFAULT_RESULTS))
    out_md = render(d)
    os.makedirs(os.path.dirname(DEFAULT_OUT), exist_ok=True)
    with open(DEFAULT_OUT, "w") as f:
        f.write(out_md)
    print(f"Wrote {DEFAULT_OUT} ({len(out_md)} chars)")
    print()
    print(out_md)


if __name__ == "__main__":
    main()
