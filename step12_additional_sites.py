"""Step 12: Replicate the P1 finding at three additional sites not in the
preselected primary set. Tests whether shared > complement > random
generalizes beyond the original 3 sites.

Sites added (chosen by top-CKA after no-PC1 filter, from step3_cka.json,
excluding the 3 primary sites already analyzed):
  layer1_result_4 (CKA=0.759, no-PC=0.454)
  layer3_result_0 (CKA=0.696, no-PC=0.556)
  layer3_result_3 (CKA=0.697, no-PC=0.578)

Outputs results/p1/additional_sites.json (same structure as p1_results.json
but for the additional sites).
"""

import json
import os
import time

import numpy as np
import torch

from collect_activations import get_converged_models
from config import ModelConfig
from data import load_eval_set
from step9_p1 import (
    run_site, K_VALUES, N_RANDOM_TRIALS, SITE_LAYER_POS,
)

OUT = "results/p1/additional_sites.json"

# Pre-locked additional sites (chosen by top-CKA, ex-primary set)
ADDITIONAL_SITES = {
    "layer1_result_4": (1, 16),       # layer 1, result digit 4
    "layer3_result_0": (3, 12),       # layer 3, result digit 0
    "layer3_result_3": (3, 15),       # layer 3, result digit 3
}


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_converged_models("models")
    print(f"Found {len(models)} converged models on {device}")
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]

    # Inject the additional sites into the global SITE_LAYER_POS so run_site
    # finds them.
    for k, v in ADDITIONAL_SITES.items():
        SITE_LAYER_POS[k] = v

    results = {
        "config": {
            "k_values": K_VALUES,
            "n_random_trials": N_RANDOM_TRIALS,
            "sites": list(ADDITIONAL_SITES.keys()),
            "n_models": len(models),
            "d_model": cfg.d_model,
            "purpose": "Replication of P1 ordering at sites not in original primary set.",
        },
        "primary": {},
    }

    for site in ADDITIONAL_SITES:
        print(f"\n=== ADDITIONAL SITE :: {site} ===")
        t0 = time.time()
        results["primary"][site] = run_site(
            models, site, cfg, eval_tokens, device,
            K_VALUES, N_RANDOM_TRIALS, max_dims_request=max(K_VALUES),
            eps_scale=1e-8,
        )
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ({time.time()-t0:.1f}s for site)")

    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
