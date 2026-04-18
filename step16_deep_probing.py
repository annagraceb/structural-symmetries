"""Step 16: probe the 6-layer zoo's layer-5 shared directions for readability.

In the deep zoo, layer 5 has shared drop = 0.000 (exactly). If shared
directions are LINEARLY PROBEABLE at layer 5 despite being causally inert
under single-subspace ablation, it confirms the "functional universality
collapse" framing from the main zoo — shared directions read the task state
but don't drive computation when single-ablated.

Also probes layer 0 shared (where shared has primary single-subspace role)
for comparison.

Uses the same probing protocol as analysis.run_probing, adapted for the
deep zoo's token-position scheme.
"""

import json
import os
import numpy as np

from analysis import run_probing
from config_deep import ModelConfigDeep
from data import load_eval_set
from step9_p1_deep import SITES_DEEP
from step9_p1 import extract_with_max_dims, K_PCA


OUT = "results/deep/layer_probing.json"

# Uses step9_p1_deep's aligned activations.
RESULTS_DIR = "results/deep"


def load_aligned_deep(site_name):
    d = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(d, f))
    return out


def main():
    cfg = ModelConfigDeep()
    # The deep zoo's aligned activations were computed on convergence_eval
    # (5000 problems). Use that metadata to match.
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_metadata = eval_data["metadata"]
    print(f"Probing layer-5 shared directions (6-layer zoo)")
    print(f"  eval set: {len(eval_metadata)} problems")

    # Pick interesting sites to probe
    sites_to_probe = ["layer0_result_0", "layer3_result_0", "layer5_result_0"]

    out = {}
    for site in sites_to_probe:
        aligned = load_aligned_deep(site)
        if not aligned:
            print(f"  {site}: no aligned data, skipping")
            continue
        print(f"\n=== {site} ({len(aligned)} models) ===")
        ex = extract_with_max_dims(aligned, max_dims=10, eps_scale=1e-8, k_pca=K_PCA)
        shared = ex["shared_dirs"]
        # Unit-normalize for probe (match analysis.py convention)
        shared_u = shared / np.linalg.norm(shared, axis=1, keepdims=True)

        probe_res = run_probing(shared_u, aligned, eval_metadata, cfg)
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

        for dn, items in summary.items():
            top = items[0] if items else None
            if top:
                print(f"  {dn}: {top[0]} corr={top[1]:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
