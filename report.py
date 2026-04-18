"""Generate a comprehensive report of all experiment results."""

import os
import json
import numpy as np


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def report_step1():
    """Model zoo summary."""
    print("=" * 70)
    print("STEP 1: MODEL ZOO")
    print("=" * 70)

    models_dir = "models"
    if not os.path.exists(models_dir):
        print("  No models directory found")
        return

    converged = []
    failed = []

    for entry in sorted(os.listdir(models_dir)):
        meta_path = os.path.join(models_dir, entry, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["name"] = entry
            if meta["converged"]:
                converged.append(meta)
            else:
                failed.append(meta)

    print(f"\n  Converged: {len(converged)} models")
    for m in converged:
        print(f"    {m['name']}: acc={m['best_accuracy']:.4f} "
              f"step={m['converge_step']} ({m['total_time_seconds']:.0f}s)")

    if failed:
        print(f"\n  Failed to converge: {len(failed)} models")
        for m in failed:
            print(f"    {m['name']}: best_acc={m['best_accuracy']:.4f} "
                  f"({m['total_time_seconds']:.0f}s)")

    # Decision gate
    freeze_configs = set()
    for m in converged:
        if m.get("frozen_component"):
            freeze_configs.add(m["frozen_component"])
    print(f"\n  Distinct freeze configs converged: {len(freeze_configs)}")
    print(f"  Decision gate: {'PASS' if len(freeze_configs) >= 6 else 'FAIL'}")


def report_step3():
    """CKA analysis summary."""
    print("\n" + "=" * 70)
    print("STEP 3: CKA ANALYSIS")
    print("=" * 70)

    data = load_json("results/step3_cka.json")
    if not data:
        print("  No CKA results found")
        return

    # Top sites
    sites = [(k, v) for k, v in data.items()
             if isinstance(v, dict) and "cka_mean" in v]
    sites.sort(key=lambda x: x[1]["cka_mean"], reverse=True)

    print("\n  Top CKA sites:")
    for name, d in sites[:10]:
        print(f"    {name}: CKA={d['cka_mean']:.3f} "
              f"[{d['cka_ci_lower']:.3f}, {d['cka_ci_upper']:.3f}] "
              f"| no-PC1={d['cka_no_top_pc_mean']:.3f}")

    # Three-group comparison
    three_group = data.get("three_group", {})
    if three_group:
        print("\n  Three-group comparison:")
        for group, vals in three_group.items():
            print(f"    {group}: mean={vals['mean']:.3f} ± {vals['std']:.3f} "
                  f"(n={vals['n']})")

    # Within vs across
    wa = data.get("within_across", {})
    if wa:
        print("\n  Within-config vs across-config CKA:")
        wc = wa.get("within_config", {})
        ac = wa.get("across_config", {})
        if wc and ac:
            print(f"    Within:  {wc['mean']:.3f} ± {wc['std']:.3f}")
            print(f"    Across:  {ac['mean']:.3f} ± {ac['std']:.3f}")

    # Selected sites
    selected = load_json("results/selected_sites.json")
    if selected:
        print(f"\n  Selected extraction sites: {len(selected)}")
        for s in selected:
            print(f"    {s['site']}: CKA={s['cka_mean']:.3f}")


def report_step4():
    """Alignment summary."""
    print("\n" + "=" * 70)
    print("STEP 4: PROCRUSTES ALIGNMENT")
    print("=" * 70)

    data = load_json("results/step4_alignment.json")
    if not data:
        print("  No alignment results found")
        return

    for site_name, site_data in data.items():
        print(f"\n  Site: {site_name}")
        print(f"    Reference: {site_data['reference']}")
        print(f"    Random baseline residual: {site_data['random_baseline_residual']:.4f}")
        print(f"    Baseline mean residual: {site_data.get('baseline_mean_residual', 'N/A')}")
        print(f"    SVCCA fallbacks: {site_data.get('n_svcca_fallback', 0)}")

        residuals = [v["procrustes_residual"]
                      for v in site_data["models"].values()]
        random_bl = site_data['random_baseline_residual']
        if residuals:
            mean_res = np.mean(residuals)
            print(f"    Residuals: mean={mean_res:.4f}, "
                  f"std={np.std(residuals):.4f}, "
                  f"range=[{min(residuals):.4f}, {max(residuals):.4f}]")
            if mean_res > random_bl:
                print(f"    WARNING: Mean residual ({mean_res:.4f}) > random baseline "
                      f"({random_bl:.4f}). Procrustes alignment is not improving "
                      f"over random rotation. Direction-level analysis is unreliable.")


def report_step5():
    """Subspace extraction summary."""
    print("\n" + "=" * 70)
    print("STEP 5: SHARED SUBSPACE")
    print("=" * 70)

    data = load_json("results/step5_subspace.json")
    if not data:
        print("  No subspace results found")
        return

    print(f"\n  Extraction site: {data['site']}")
    print(f"  Shared eigenvalues (top 5): "
          f"{[f'{v:.4f}' for v in data['shared_eigenvalues'][:5]]}")
    print(f"  PCA cosines (shared vs PCA, top 5): "
          f"{[f'{v:.4f}' for v in data['pca_cosines'][:5]]}")

    nat = data.get("natural_validation", {})
    if nat:
        print(f"\n  Natural distribution validation:")
        for entry in nat.get("explained_ratios", []):
            print(f"    Top-{entry['k']} dirs: {entry['ratio']:.3f} variance explained")

    ref_inv = data.get("reference_invariance", {})
    if ref_inv:
        print(f"\n  Reference invariance:")
        print(f"    Second reference: {ref_inv.get('reference2', 'N/A')}")
        cross = ref_inv.get("cross_reference_cosines", [])
        if cross:
            print(f"    Cross-reference cosines: {[f'{v:.3f}' for v in cross]}")


def report_step6():
    """Probing summary."""
    print("\n" + "=" * 70)
    print("STEP 6: INTERPRETATION")
    print("=" * 70)

    data = load_json("results/step6_probing.json")
    if not data:
        print("  No probing results found")
        return

    for dir_name, dir_data in data.items():
        print(f"\n  {dir_name}:")
        # Sort by absolute correlation
        sorted_vars = sorted(
            [(k, v) for k, v in dir_data.items() if isinstance(v, dict)],
            key=lambda x: abs(x[1].get("correlation", 0)),
            reverse=True
        )
        for var_name, var_data in sorted_vars[:5]:
            parts = [f"corr={var_data.get('correlation', 0):.3f}"]
            if "mutual_info" in var_data:
                parts.append(f"MI={var_data['mutual_info']:.4f}")
            if "probe_accuracy" in var_data:
                parts.append(f"probe={var_data['probe_accuracy']:.3f}")
            print(f"    {var_name}: {', '.join(parts)}")


def report_step7():
    """Ablation summary."""
    print("\n" + "=" * 70)
    print("STEP 7: ABLATION")
    print("=" * 70)

    data = load_json("results/step7_ablation.json")
    if not data:
        print("  No ablation results found")
        return

    summary = data.get("summary", {})
    if summary:
        print("\n  Ablation results:")
        for key, s in summary.items():
            p = s['wilcoxon_pval']
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
            else: sig = "ns"
            direction = "shared < random" if s['mean_shared_drop'] < s['mean_random_drop'] else "shared > random"
            print(f"    {key}: shared_drop={s['mean_shared_drop']:.4f}, "
                  f"random_drop={s['mean_random_drop']:.4f}, "
                  f"p={p:.6f} {sig} ({direction})")


def report_step8():
    """Auxiliary loss summary."""
    print("\n" + "=" * 70)
    print("STEP 8: AUXILIARY LOSS")
    print("=" * 70)

    data = load_json("results/step8_auxiliary.json")
    if not data:
        print("  No auxiliary loss results found")
        return

    summary = data.get("summary", {})
    if summary:
        print("\n  Condition results (steps to 99% accuracy):")
        for cond, s in sorted(summary.items()):
            if s.get("mean_converge_step"):
                std = f"±{s['std_converge_step']:.0f}" if s.get("std_converge_step") else ""
                print(f"    {cond}: {s['mean_converge_step']:.0f}{std} steps "
                      f"({s['n_converged']}/{s['n_total']} converged, "
                      f"alpha={s['best_alpha']})")
            else:
                print(f"    {cond}: did not converge")


def main():
    print("\n" + "#" * 70)
    print("# SOLUTION SYMMETRY EXPLORATION — RESULTS REPORT")
    print("#" * 70)

    report_step1()
    report_step3()
    report_step4()
    report_step5()
    report_step6()
    report_step7()
    report_step8()

    print("\n" + "#" * 70)
    print("# END OF REPORT")
    print("#" * 70)


if __name__ == "__main__":
    main()
