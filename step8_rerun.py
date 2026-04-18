"""Path 1: Rerun Step 8 with proper measurement.

Changes from original:
- eval_every=200 (not 1000)
- 20 seeds per condition (not 5)
- Only A, C2, D, D_prime (drop B and C1)
- Uses existing zoo and shared directions from run 2
"""

import os
import json
import numpy as np
import torch
from scipy import stats

from config import ModelConfig, TrainConfig
from model import ArithmeticTransformer, CarryHead
from data import generate_batch, load_eval_set
from train import set_seed, evaluate
from run_auxiliary import train_auxiliary


def main():
    model_cfg = ModelConfig()
    results_dir = "results"
    rerun_dir = "models_step8_rerun"
    eval_dir = "eval_sets"

    # Load eval set
    eval_data = load_eval_set(os.path.join(eval_dir, "convergence_eval"))
    eval_tokens = eval_data["tokens"]

    # Load shared directions
    shared_dirs = np.load(os.path.join(results_dir, "shared_dirs.npy"))
    bottom_dirs = np.load(os.path.join(results_dir, "bottom_dirs.npy"))

    # Random directions
    rng = np.random.RandomState(12345)
    d = shared_dirs.shape[1]
    n_dirs = shared_dirs.shape[0]
    random_dirs = rng.randn(n_dirs, d).astype(np.float32)
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

    # Load reference activations
    sites_path = os.path.join(results_dir, "selected_sites.json")
    selected_sites = json.load(open(sites_path))
    top_site = selected_sites[0]
    target_layer = top_site["layer"]
    target_pos = top_site["position_idx"]

    ref_act_path = None
    for entry in sorted(os.listdir("activations")):
        if "baseline" in entry and "seed0" in entry:
            ref_act_path = os.path.join("activations", entry,
                                        f"layer{target_layer}_stratified.npy")
            break
    reference_acts = np.load(ref_act_path)[:, target_pos, :]

    # CKA eval tokens
    strat_data = load_eval_set(os.path.join(eval_dir, "stratified_2000"))
    cka_eval_tokens = strat_data["tokens"]

    print(f"Target site: layer {target_layer}, position {target_pos}")
    print(f"Shared dirs: {shared_dirs.shape}, Reference acts: {reference_acts.shape}")

    # Conditions: A, C2, D, D_prime only
    # Use best alphas from sweep: C2=0.001, D=0.1, D_prime=0.001
    conditions = {
        "A":       {"shared": None,              "ref": None,           "alpha": 0.1},
        "C2":      {"shared": shared_dirs[:5],   "ref": reference_acts, "alpha": 0.001},
        "D":       {"shared": random_dirs[:5],   "ref": reference_acts, "alpha": 0.1},
        "D_prime": {"shared": bottom_dirs[:5],   "ref": reference_acts, "alpha": 0.001},
    }

    n_seeds = 20
    all_results = {}

    for cond_name, cond_data in conditions.items():
        print(f"\n{'='*60}")
        print(f"CONDITION {cond_name} (alpha={cond_data['alpha']}, {n_seeds} seeds)")
        print(f"{'='*60}")

        cond_results = []
        for seed in range(n_seeds):
            run_name = f"{cond_name}_seed{seed}"
            save_path = os.path.join(rerun_dir, run_name)

            if os.path.exists(os.path.join(save_path, "metadata.json")):
                with open(os.path.join(save_path, "metadata.json")) as f:
                    meta = json.load(f)
                print(f"  [SKIP] {run_name}: step={meta.get('converge_step', 'N/A')}")
                cond_results.append(meta)
                continue

            tcfg = TrainConfig(
                seed=seed + 500,
                eval_every=200,
                log_every=200,
            )

            print(f"  Training {run_name}...")
            meta = train_auxiliary(
                model_cfg, tcfg, eval_tokens, save_path,
                condition=cond_name,
                shared_dirs=cond_data["shared"],
                reference_acts=cond_data["ref"],
                cka_eval_tokens=cka_eval_tokens,
                alpha=cond_data["alpha"],
                target_layer=target_layer,
                target_pos=target_pos,
            )
            cond_results.append(meta)

        all_results[cond_name] = cond_results

    # Analysis
    print("\n" + "=" * 70)
    print("STEP 8 RERUN RESULTS (20 seeds, eval_every=200)")
    print("=" * 70)

    summaries = {}
    for cond_name, results in all_results.items():
        steps = [r["converge_step"] for r in results if r.get("converged")]
        n_conv = len(steps)
        if steps:
            mean_s = np.mean(steps)
            std_s = np.std(steps)
            median_s = np.median(steps)
        else:
            mean_s = std_s = median_s = None

        summaries[cond_name] = {
            "converge_steps": steps,
            "n_converged": n_conv,
            "n_total": len(results),
            "mean": mean_s,
            "std": std_s,
            "median": median_s,
        }
        if mean_s:
            print(f"  {cond_name:8}: mean={mean_s:>7.0f} ± {std_s:>6.0f}  "
                  f"median={median_s:>6.0f}  ({n_conv}/{len(results)} converged)")
        else:
            print(f"  {cond_name:8}: did not converge ({n_conv}/{len(results)})")

    # Statistical tests
    print("\nStatistical Tests (one-sided):")
    if summaries["A"]["converge_steps"] and summaries["C2"]["converge_steps"]:
        t, p = stats.ttest_ind(summaries["C2"]["converge_steps"],
                                summaries["A"]["converge_steps"],
                                alternative="less")
        print(f"  C2 < A:       t={t:.3f}, p={p:.4f}")

        # Also Mann-Whitney (non-parametric)
        u, p_mw = stats.mannwhitneyu(summaries["C2"]["converge_steps"],
                                      summaries["A"]["converge_steps"],
                                      alternative="less")
        print(f"  C2 < A (MW):  U={u:.0f}, p={p_mw:.4f}")

    if summaries["A"]["converge_steps"] and summaries["D_prime"]["converge_steps"]:
        t, p = stats.ttest_ind(summaries["D_prime"]["converge_steps"],
                                summaries["A"]["converge_steps"],
                                alternative="greater")
        print(f"  D' > A:       t={t:.3f}, p={p:.4f}")

    if summaries["C2"]["converge_steps"] and summaries["D_prime"]["converge_steps"]:
        t, p = stats.ttest_ind(summaries["C2"]["converge_steps"],
                                summaries["D_prime"]["converge_steps"],
                                alternative="less")
        print(f"  C2 < D':      t={t:.3f}, p={p:.4f}")

    if summaries["C2"]["converge_steps"] and summaries["D"]["converge_steps"]:
        t, p = stats.ttest_ind(summaries["C2"]["converge_steps"],
                                summaries["D"]["converge_steps"],
                                alternative="less")
        print(f"  C2 < D:       t={t:.3f}, p={p:.4f}")

    # Save
    save_data = {
        "config": {
            "n_seeds": n_seeds,
            "eval_every": 200,
            "conditions": {k: {"alpha": v["alpha"]} for k, v in conditions.items()},
        },
        "summaries": {k: {kk: vv for kk, vv in v.items()}
                       for k, v in summaries.items()},
    }
    with open(os.path.join(results_dir, "step8_rerun.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {results_dir}/step8_rerun.json")


if __name__ == "__main__":
    main()
