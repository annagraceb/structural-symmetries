"""Step 13: Probe layer-3 shared subspace for linear readability.

If layer-3 shared directions are still linearly probeable (despite being
near-causally inert), the story is "functional universality collapse" —
shared directions remain readable late in the network but no longer drive
computation. Run linear probes for the top shared direction at each layer-3
site against hand-labeled arithmetic features (digit identity per column,
carries, sum magnitude).

Uses the same features and probing helper as analysis.py::run_probing.

Outputs results/p1/layer3_probing.json.
"""

import json
import os
import numpy as np

from analysis import run_probing
from collect_activations import get_converged_models
from config import ModelConfig
from data import load_eval_set
from step9_p1 import extract_with_max_dims, load_aligned_acts, K_PCA, SITE_LAYER_POS


OUT = "results/p1/layer3_probing.json"
LAYER3_SITES = ["layer3_result_0", "layer3_result_3"]


def main():
    cfg = ModelConfig()
    eval_data = load_eval_set(os.path.join("eval_sets", "stratified_2000"))
    eval_metadata = eval_data["metadata"]

    out = {}
    for site in LAYER3_SITES:
        print(f"\n=== Probing {site} shared directions ===")
        aligned_acts = load_aligned_acts(site)
        ex = extract_with_max_dims(aligned_acts, max_dims=10, eps_scale=1e-8, k_pca=K_PCA)
        shared = ex["shared_dirs"]          # C-orthonormal per A2
        # Re-normalize to unit Euclidean for the probe so results are comparable
        # to analysis.py's conventions.
        shared_unit = shared / np.linalg.norm(shared, axis=1, keepdims=True)

        probe_res = run_probing(shared_unit, aligned_acts, eval_metadata, cfg)

        # Extract top correlations per direction
        summary = {}
        for dir_name, dir_data in probe_res.items():
            items = []
            for var, val in dir_data.items():
                if isinstance(val, dict):
                    corr = val.get("correlation")
                    mi = val.get("mutual_info")
                    if corr is not None:
                        items.append((var, corr, mi))
            items.sort(key=lambda x: abs(x[1]), reverse=True)
            summary[dir_name] = items[:5]
        out[site] = summary

        print(f"Top correlations per direction at {site}:")
        for dn, items in summary.items():
            print(f"  {dn}:")
            for var, corr, mi in items[:3]:
                mi_s = f" MI={mi:.3f}" if mi is not None else ""
                print(f"    {var}: corr={corr:.3f}{mi_s}")

        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)

    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
