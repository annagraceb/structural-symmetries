"""P1 report / verdict generator.

Reads results/p1/p1_results.json and applies the pre-registered decision rule
(P1_PREREGISTRATION.md), producing results/p1/p1_verdict.json.

Separate from step9_p1.py so the experiment runner has no knowledge of the
decision criteria at runtime.

Usage:
    python p1_report.py                       # print verdict to stdout
    python p1_report.py --out p1_verdict.json
"""

import argparse
import json
import os

import numpy as np

DEFAULT_RESULTS = "results/p1/p1_results.json"
DEFAULT_VERDICT = "results/p1/p1_verdict.json"
PRIMARY_SITES = ["layer1_result_0", "layer2_equals", "layer2_result_0"]
CONTROL_SITE = "layer3_plus"

# Decision thresholds (locked in pre-registration)
WIN_R = 2.0
WIN_R_CI_LOWER = 1.25
STRONG_WIN_R = 3.0
STRONG_WIN_R_CI_LOWER = 1.5
LOSS_R = 1.1
MIN_MODEL_AGREEMENT_FRAC = 0.75
MIN_SHARED_MULTIPLE = 2.0      # drop(anti) >= 2 x drop(shared)
CONTROL_MAX_DROP = 0.02        # 2% absolute
CONTROL_MAX_R = 1.5


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def bootstrap_ci(values, fn=np.mean, n_boot=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray([v for v in values if v is not None])
    if len(arr) == 0:
        return None, (None, None)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        stats[i] = fn(arr[idx])
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return float(fn(arr)), (lo, hi)


def per_model_R(site_data, variant, k):
    """For each model, compute R = drop_variant(k) / mean_random_drop(k).
    Returns list of (model_name, R, drop_variant, mean_drop_random)."""
    rows = []
    for name, m in site_data["models"].items():
        v = m.get(variant, {})
        entry = v.get(str(k)) or v.get(k)
        if not isinstance(entry, dict):
            continue
        d_v = entry.get("drop")
        if d_v is None:
            continue
        rand_drops = m["random"].get(str(k)) or m["random"].get(k) or []
        if not rand_drops:
            continue
        d_r = float(np.mean(rand_drops))
        R = d_v / d_r if d_r > 1e-10 else float("inf")
        rows.append((name, R, d_v, d_r))
    return rows


def summarize_variant_at(site_data, variant, k):
    rows = per_model_R(site_data, variant, k)
    if not rows:
        return None
    R_vals = [r[1] for r in rows]
    drop_vals = [r[2] for r in rows]
    R_mean, (R_lo, R_hi) = bootstrap_ci(R_vals)
    drop_mean, (d_lo, d_hi) = bootstrap_ci(drop_vals)
    frac_R_gt1 = float(np.mean([R > 1.0 for R in R_vals]))
    return {
        "k": k,
        "variant": variant,
        "n_models": len(rows),
        "mean_R": R_mean,
        "R_ci": [R_lo, R_hi],
        "mean_drop": drop_mean,
        "drop_ci": [d_lo, d_hi],
        "fraction_models_R_gt_1": frac_R_gt1,
    }


def summarize_shared_at(site_data, k):
    drops = []
    for m in site_data["models"].values():
        s = m.get("shared", {})
        entry = s.get(str(k)) or s.get(k)
        if isinstance(entry, dict) and entry.get("drop") is not None:
            drops.append(entry["drop"])
    if not drops:
        return None
    mean_d, (lo, hi) = bootstrap_ci(drops)
    return {"k": k, "n_models": len(drops), "mean_drop": mean_d, "drop_ci": [lo, hi]}


def classify_verdict(per_site_variant_summaries, shared_summaries, control_summary):
    """Apply the locked decision rule to a {site -> summary_at_k10} mapping for
    one variant. Returns one of: STRONG_WIN / WIN / AMBIGUOUS / LOSS."""
    # Count sites meeting each tier at k=10 (primary k)
    strong_sites = 0
    win_sites = 0
    loss_sites = 0

    for site, summ in per_site_variant_summaries.items():
        if summ is None:
            continue
        R = summ["mean_R"]
        Rlo = summ["R_ci"][0]
        frac = summ["fraction_models_R_gt_1"]
        # Check the 2x shared multiple
        shared = shared_summaries.get(site)
        meets_shared_multiple = (
            shared is not None and
            summ["mean_drop"] >= MIN_SHARED_MULTIPLE * shared["mean_drop"]
        )
        if (R is not None and R >= STRONG_WIN_R and Rlo is not None
                and Rlo > STRONG_WIN_R_CI_LOWER
                and frac >= MIN_MODEL_AGREEMENT_FRAC
                and meets_shared_multiple):
            strong_sites += 1
        elif (R is not None and R >= WIN_R and Rlo is not None
              and Rlo > WIN_R_CI_LOWER
              and frac >= MIN_MODEL_AGREEMENT_FRAC
              and shared is not None and summ["mean_drop"] > shared["mean_drop"]):
            win_sites += 1
        if R is not None and R <= LOSS_R:
            loss_sites += 1

    # Control site gate. The intent is: at a low-CKA control site, no condition
    # should cause meaningful damage. The R-ratio test is unreliable when
    # absolute drops are near zero (R blows up to inf), so we use the absolute
    # drop test as the primary control gate. The R test only applies if absolute
    # drop is non-trivial (> 0.005).
    control_fail = False
    if control_summary is not None:
        d = control_summary["mean_drop"]
        if d is not None and d > CONTROL_MAX_DROP:
            control_fail = True
        if (d is not None and d > 0.005
                and control_summary["mean_R"] is not None
                and control_summary["mean_R"] > CONTROL_MAX_R):
            control_fail = True

    n_primary = sum(1 for s in per_site_variant_summaries.values() if s is not None)
    verdict = "AMBIGUOUS"
    if strong_sites >= 2:
        verdict = "STRONG_WIN"
    elif win_sites >= 2 or (strong_sites == 1 and win_sites >= 1):
        verdict = "WIN"
    elif loss_sites == n_primary and n_primary > 0:
        verdict = "LOSS"

    if control_fail and verdict in ("STRONG_WIN", "WIN"):
        # Downgrade one tier
        verdict = "WIN" if verdict == "STRONG_WIN" else "AMBIGUOUS"

    return verdict, control_fail


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=DEFAULT_RESULTS)
    ap.add_argument("--out", default=DEFAULT_VERDICT)
    args = ap.parse_args()

    d = json.load(open(args.infile))
    primary = d["primary"]

    # Variants to evaluate under the decision rule
    variants = ["anti_shared_raw", "anti_shared_ortho", "complement_top_k"]

    # Per-site per-variant summaries at k=10 (primary k)
    per_variant_at_k10 = {v: {} for v in variants}
    shared_at_k10 = {}
    for site in PRIMARY_SITES:
        if site not in primary:
            continue
        shared_at_k10[site] = summarize_shared_at(primary[site], 8)
        for v in variants:
            per_variant_at_k10[v][site] = summarize_variant_at(primary[site], v, 8)

    # Control-site summaries
    control_summaries = {}
    if CONTROL_SITE in primary:
        for v in variants:
            control_summaries[v] = summarize_variant_at(primary[CONTROL_SITE], v, 8)

    # Verdict per variant
    verdicts = {}
    for v in variants:
        verdict, control_fail = classify_verdict(
            per_variant_at_k10[v], shared_at_k10, control_summaries.get(v))
        verdicts[v] = {
            "verdict": verdict,
            "control_gate_fail": control_fail,
            "per_site_at_k10": per_variant_at_k10[v],
            "control_at_k10": control_summaries.get(v),
        }

    # Per-variance ordering (post-hoc correction picture)
    # For each site at k=10, compute drop / projection_trace per condition
    per_variance_ordering = {}
    for site in PRIMARY_SITES + [CONTROL_SITE]:
        if site not in primary:
            continue
        sd = primary[site]
        order = {}
        for cond in ["shared", "anti_shared_raw", "anti_shared_ortho", "complement_top_k"]:
            drops = []
            traces = []
            for m in sd["models"].values():
                c = m.get(cond, {})
                entry = c.get("8") if isinstance(c, dict) else None
                pt = m.get("projection_trace_variance", {}).get(cond, {}).get("8")
                if isinstance(entry, dict) and entry.get("drop") is not None and pt is not None:
                    drops.append(entry["drop"])
                    traces.append(pt)
            if drops:
                mean_drop = float(np.mean(drops))
                mean_trace = float(np.mean(traces))
                hpr = mean_drop / mean_trace if mean_trace > 1e-10 else None
                order[cond] = {
                    "mean_drop": mean_drop,
                    "mean_proj_trace": mean_trace,
                    "drop_per_variance": hpr,
                    "n_models": len(drops),
                }
        per_variance_ordering[site] = order

    # Geometry recap (already in extraction, pull it through)
    geometry_by_site = {}
    for site in PRIMARY_SITES + [CONTROL_SITE]:
        if site in primary and "geometry" in primary[site]:
            geometry_by_site[site] = primary[site]["geometry"]

    # ε-sweep stability check
    eps_summary = {}
    for eps, edata in d.get("eps_sweep", {}).items():
        site = "layer1_result_0"
        if site in edata:
            eps_summary[eps] = {
                "shared_k10": summarize_shared_at(edata[site], 8),
                "anti_shared_raw_k10": summarize_variant_at(edata[site], "anti_shared_raw", 8),
                "complement_top_k_k10": summarize_variant_at(edata[site], "complement_top_k", 8),
                "denom_condition_number": edata[site]["extraction"]["denom_condition_number"],
            }

    out = {
        "pre_registered_verdicts": verdicts,
        "per_variance_ordering": per_variance_ordering,
        "geometry": geometry_by_site,
        "eps_sweep_stability": eps_summary,
        "config": d["config"],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Console summary
    print("=" * 70)
    print("P1 REPORT")
    print("=" * 70)
    for v, vd in verdicts.items():
        print(f"\n{v}: {vd['verdict']} (control_fail={vd['control_gate_fail']})")
        for site, summ in vd["per_site_at_k10"].items():
            if summ is None:
                print(f"  {site}: <no data>")
                continue
            print(f"  {site}: R={summ['mean_R']:.3f} [{summ['R_ci'][0]:.2f},{summ['R_ci'][1]:.2f}]  "
                  f"drop={summ['mean_drop']:.4f}  frac_R>1={summ['fraction_models_R_gt_1']:.2f}  "
                  f"n={summ['n_models']}")
        ctrl = vd["control_at_k10"]
        if ctrl:
            print(f"  {CONTROL_SITE} (control): R={ctrl['mean_R']:.3f}  drop={ctrl['mean_drop']:.4f}")

    print("\nPer-variance ordering (drop / projection_trace) at k=10:")
    for site, order in per_variance_ordering.items():
        print(f"\n  {site}:")
        ranked = sorted(
            ((cond, stats) for cond, stats in order.items() if stats.get("drop_per_variance") is not None),
            key=lambda x: x[1]["drop_per_variance"], reverse=True)
        for cond, stats in ranked:
            print(f"    {cond:>24}: drop={stats['mean_drop']:.4f}  trace={stats['mean_proj_trace']:.4f}  "
                  f"drop/trace={stats['drop_per_variance']:.4f}")

    print(f"\nSaved verdict to {args.out}")


if __name__ == "__main__":
    main()
