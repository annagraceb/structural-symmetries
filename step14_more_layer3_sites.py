"""Step 14: Expand layer-3 coverage to strengthen the layer-dependence claim.

Per Codex's review ("only six handpicked sites, with strongest effect in two
final-layer result-token sites"), add more layer-3 positions:

  layer3_result_1, layer3_result_2, layer3_result_4, layer3_result_5,
  layer3_operand_a_last, layer3_operand_b_last

These span all result-token positions at layer 3 plus the two final operand
positions. If the pattern "shared is near-inert at layer 3" holds broadly,
we should see shared drop ≈ 0 and complement drop substantial across all
or most of these sites.

Outputs results/p1/layer3_expanded.json.
"""

import json
import os
import time

import numpy as np
import torch

from collect_activations import get_converged_models
from config import ModelConfig
from data import load_eval_set
from step9_p1 import run_site, K_VALUES, N_RANDOM_TRIALS, SITE_LAYER_POS


OUT = "results/p1/layer3_expanded.json"

ADDITIONAL_LAYER3_SITES = {
    "layer3_result_1": (3, 13),
    "layer3_result_2": (3, 14),
    "layer3_result_4": (3, 16),
    "layer3_result_5": (3, 17),
    "layer3_operand_a_last": (3, 4),
    "layer3_operand_b_last": (3, 10),
}


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_converged_models("models")
    print(f"Found {len(models)} models on {device}")
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]

    for k, v in ADDITIONAL_LAYER3_SITES.items():
        SITE_LAYER_POS[k] = v

    results = {
        "config": {
            "k_values": K_VALUES,
            "n_random_trials": N_RANDOM_TRIALS,
            "sites": list(ADDITIONAL_LAYER3_SITES.keys()),
            "n_models": len(models),
            "purpose": "Strengthen layer-dependence claim with more layer-3 coverage.",
        },
        "primary": {},
    }

    for site in ADDITIONAL_LAYER3_SITES:
        print(f"\n=== LAYER-3 EXPANSION :: {site} ===")
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
