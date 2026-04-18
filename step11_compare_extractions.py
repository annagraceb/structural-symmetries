"""Step 11: Disentangle the contributions of A1 (PCA restriction) and A2
(variance matching) to the Step-7 sign-flip.

Compares four protocols on the SAME models, sites, and ablation hook:
  - OLD shared dirs (no A1, no A2): unit-norm Euclidean, eigvecs of full
    generalized eigenproblem. This is original Step 7. Loaded from
    results/shared_dirs.npy.
  - NEW shared dirs + unit-norm (A1 only, no A2): unit-normalized eigvecs
    of PCA-restricted generalized eigenproblem.
  - NEW shared dirs + C-orthonormal (A1 + A2): full P1 protocol.
  - random unit-norm + random whitened: matched baselines.

Outputs results/p1/extraction_decomposition.json with per-site, per-protocol
mean ablation drops.
"""

import json
import os
import time

import numpy as np
import torch

from analysis import (
    extract_shared_subspace,
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
    whitened_random_subspace,
    K_PCA, SITE_LAYER_POS,
)

OUT = "results/p1/extraction_decomposition.json"
SITES = ["layer1_result_0", "layer2_equals", "layer2_result_0"]
K = 10
N_RANDOM = 50


def unit_normalize(V):
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def run_one_site(site_name, models, cfg, eval_tokens, device):
    layer, pos = SITE_LAYER_POS[site_name]
    print(f"\n=== Decomposition test: {site_name} ===")

    aligned = load_aligned_acts(site_name)

    # OLD: full generalized eigenproblem, no PCA restriction. Use the project's
    # original extract_shared_subspace.
    old_extraction = extract_shared_subspace(aligned)
    old_shared_unit = old_extraction["shared_dirs"][:K]   # already unit-norm

    # NEW: A1+A2 extraction
    new_ex = extract_with_max_dims(aligned, max_dims=K, eps_scale=1e-8, k_pca=K_PCA)
    new_shared_C = new_ex["shared_dirs"]                  # C-orthonormal (A2)
    new_shared_unit = unit_normalize(new_shared_C)        # A1 only (no A2)
    C_total = new_ex["C_total"]
    C_total_plus_ridge = new_ex["C_total_plus_ridge"]
    d = new_ex["d"]

    rng_unit = np.random.RandomState(7777)
    rng_white = np.random.RandomState(7778)

    per_model = {}
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

        # Four shared-protocol drops
        old_unit_acc = ablate_dirs(old_shared_unit)
        new_unit_acc = ablate_dirs(new_shared_unit)
        new_C_acc = ablate_dirs(new_shared_C)

        # Two random baselines
        unit_drops = []
        for _ in range(N_RANDOM):
            r = rng_unit.randn(K, d).astype(np.float32)
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            unit_drops.append(baseline_acc - ablate_dirs(r))

        white_drops = []
        for _ in range(N_RANDOM):
            r = whitened_random_subspace(K, d, C_total_plus_ridge, rng_white)
            white_drops.append(baseline_acc - ablate_dirs(r.astype(np.float32)))

        per_model[name] = {
            "baseline_acc": float(baseline_acc),
            "old_shared_unit_drop": float(baseline_acc - old_unit_acc),
            "new_shared_unit_drop": float(baseline_acc - new_unit_acc),
            "new_shared_C_drop": float(baseline_acc - new_C_acc),
            "random_unit_drops": unit_drops,
            "random_unit_mean_drop": float(np.mean(unit_drops)),
            "random_whitened_drops": white_drops,
            "random_whitened_mean_drop": float(np.mean(white_drops)),
        }
        del mdl
        torch.cuda.empty_cache()
        if i % 6 == 0:
            o = per_model[name]
            print(f"  [{i+1}/{len(models)}] {name:30}  "
                  f"old_unit={o['old_shared_unit_drop']:.4f}  "
                  f"new_unit={o['new_shared_unit_drop']:.4f}  "
                  f"new_C={o['new_shared_C_drop']:.4f}  "
                  f"rand_unit={o['random_unit_mean_drop']:.4f}  "
                  f"rand_white={o['random_whitened_mean_drop']:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    return {
        "site": site_name,
        "k": K,
        "n_random_trials": N_RANDOM,
        "models": per_model,
    }


def summarize(per_site):
    out = {}
    for site, sd in per_site.items():
        ms = sd["models"]
        keys = ["old_shared_unit_drop", "new_shared_unit_drop", "new_shared_C_drop",
                "random_unit_mean_drop", "random_whitened_mean_drop"]
        means = {k: float(np.mean([m[k] for m in ms.values()])) for k in keys}
        out[site] = {
            "means": means,
            "ratios": {
                "old_shared/random_unit": means["old_shared_unit_drop"] / max(means["random_unit_mean_drop"], 1e-12),
                "new_shared_unit/random_unit": means["new_shared_unit_drop"] / max(means["random_unit_mean_drop"], 1e-12),
                "new_shared_C/random_whitened": means["new_shared_C_drop"] / max(means["random_whitened_mean_drop"], 1e-12),
                "new_shared_C/random_unit": means["new_shared_C_drop"] / max(means["random_unit_mean_drop"], 1e-12),
            },
        }
    return out


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_converged_models("models")
    print(f"Found {len(models)} converged models on {device}")
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]

    per_site = {}
    for site in SITES:
        per_site[site] = run_one_site(site, models, cfg, eval_tokens, device)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump({"per_site": per_site, "summary": summarize(per_site)},
                      f, indent=2)

    s = summarize(per_site)
    print("\n=== SUMMARY: extraction-decomposition test ===")
    for site, sd in s.items():
        print(f"\n{site}:")
        for k, v in sd["means"].items():
            print(f"  {k:32}: {v:.4f}")
        print("  --- ratios ---")
        for k, v in sd["ratios"].items():
            print(f"  {k:36}: {v:.3f}")

    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
