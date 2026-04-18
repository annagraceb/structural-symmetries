"""Step 10: Demonstrate the unit-norm failure mode predicted by the variance
confound (VARIANCE_CONFOUND_DERIVATION.md §6).

Re-runs the original Step 7 ablation under unit-norm conventions on the SAME
subspaces that P1's variance-matched ablation uses. Predicts:

  drop_unit(shared_top)  ≳ drop_unit(random)  ≳ drop_unit(shared_bottom)

i.e. the high-variance subspace looks more damaging than random, the
low-variance subspace looks less, and the *ratio* drop_unit(shared) / drop_unit(random)
is BELOW 1 even though variance-matched ablation gives the opposite sign.

This script demonstrates the sign-flip is reproducible and directly traceable
to the choice of normalization, holding the underlying subspaces fixed.

Outputs results/p1/unit_norm_failure_mode.json.

Usage:
    python step10_unit_norm_failure_mode.py
"""

import json
import os
import time

import numpy as np
import torch

from analysis import (
    get_model_activations,
    evaluate_with_hook,
)
from collect_activations import get_converged_models
from config import ModelConfig
from data import load_eval_set
from model import ArithmeticTransformer
from step9_p1 import (
    extract_with_max_dims,
    load_aligned_acts,
    make_ablation_hook,
    native_basis_dirs,
    K_PCA, SITE_LAYER_POS,
)

ACT_DIR = "activations"
RESULTS_DIR = "results"
OUT = os.path.join(RESULTS_DIR, "p1", "unit_norm_failure_mode.json")
SITES = ["layer1_result_0", "layer2_equals", "layer2_result_0"]
K = 10                  # primary k from pre-registration
N_RANDOM = 50           # 50 trials suffice to demonstrate the bias direction


def unit_normalize(V: np.ndarray) -> np.ndarray:
    """Renormalize each row to Euclidean unit length."""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return V / norms


def run_one_site(site_name, models, cfg, eval_tokens, device):
    layer, pos = SITE_LAYER_POS[site_name]
    print(f"\n=== Unit-norm ablation: {site_name} (layer={layer}, pos={pos}) ===")

    aligned = load_aligned_acts(site_name)
    ex = extract_with_max_dims(aligned, max_dims=K, eps_scale=1e-8, k_pca=K_PCA)

    # The unit-norm convention: renormalize the C-orthonormal eigvecs to Euclidean unit length.
    shared_unit = unit_normalize(ex["shared_dirs"])
    bottom_unit = unit_normalize(ex["bottom_dirs"])
    d = ex["d"]

    # Per-model ablation
    per_model = {}
    rng = np.random.RandomState(2026)
    for i, m in enumerate(models):
        name = m["model_name"]
        t0 = time.time()
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()

        baseline_acc = evaluate_with_hook(mdl, eval_tokens, cfg, device,
                                           hook_layer=None, hook_fn=None)
        acts = get_model_activations(mdl, eval_tokens, layer, device)
        acts_at_pos = acts[:, pos, :]

        def ablate_dirs(ref_dirs):
            native = native_basis_dirs(ref_dirs, name, site_name, d)
            t = torch.tensor(native, dtype=torch.float32, device=device)
            projs = acts_at_pos @ t.cpu().T
            mean_projs = projs.mean(dim=0).to(device)
            hook = make_ablation_hook(t, mean_projs, pos)
            return float(evaluate_with_hook(mdl, eval_tokens, cfg, device,
                                              hook_layer=layer, hook_fn=hook))

        shared_acc = ablate_dirs(shared_unit)
        bottom_acc = ablate_dirs(bottom_unit)

        rand_drops = []
        for _ in range(N_RANDOM):
            r = rng.randn(K, d).astype(np.float32)
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            rand_drops.append(baseline_acc - ablate_dirs(r))

        per_model[name] = {
            "baseline_acc": float(baseline_acc),
            "shared_unit_drop": float(baseline_acc - shared_acc),
            "bottom_unit_drop": float(baseline_acc - bottom_acc),
            "random_unit_drops": rand_drops,
            "random_unit_mean_drop": float(np.mean(rand_drops)),
        }
        del mdl
        torch.cuda.empty_cache()
        if i % 5 == 0:
            print(f"  [{i+1}/{len(models)}] {name}  "
                  f"shared={per_model[name]['shared_unit_drop']:.4f}  "
                  f"bottom={per_model[name]['bottom_unit_drop']:.4f}  "
                  f"random_mean={per_model[name]['random_unit_mean_drop']:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    return {
        "site": site_name,
        "layer": layer,
        "position_idx": pos,
        "k": K,
        "n_random_trials": N_RANDOM,
        "models": per_model,
    }


def summarize(per_site_data):
    summary = {}
    for site, sd in per_site_data.items():
        ms = sd["models"]
        s_drops = [m["shared_unit_drop"] for m in ms.values()]
        b_drops = [m["bottom_unit_drop"] for m in ms.values()]
        r_drops = [m["random_unit_mean_drop"] for m in ms.values()]
        summary[site] = {
            "mean_shared_drop": float(np.mean(s_drops)),
            "mean_bottom_drop": float(np.mean(b_drops)),
            "mean_random_drop": float(np.mean(r_drops)),
            "shared_over_random": float(np.mean(s_drops) / max(np.mean(r_drops), 1e-12)),
            "bottom_over_random": float(np.mean(b_drops) / max(np.mean(r_drops), 1e-12)),
            "shared_lt_random_pct_of_models": float(
                np.mean([s < r for s, r in zip(s_drops, r_drops)])),
        }
    return summary


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_converged_models("models")
    print(f"Found {len(models)} converged models on {device}")

    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    print(f"Eval set: {eval_tokens.shape[0]} problems")

    per_site = {}
    for site in SITES:
        per_site[site] = run_one_site(site, models, cfg, eval_tokens, device)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump({"per_site": per_site, "summary": summarize(per_site)},
                      f, indent=2)

    print("\n=== SUMMARY (unit-norm ablation, predicted failure mode) ===")
    for site, s in summarize(per_site).items():
        print(f"{site}:")
        print(f"  shared_drop={s['mean_shared_drop']:.4f}  bottom_drop={s['mean_bottom_drop']:.4f}  "
              f"random_drop={s['mean_random_drop']:.4f}")
        print(f"  shared/random ratio={s['shared_over_random']:.3f}  "
              f"({s['shared_lt_random_pct_of_models']*100:.0f}% of models show shared<random)")

    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
