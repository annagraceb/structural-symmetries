"""Step 17: probe 8-layer zoo shared directions at layers 3, 5, 7.

If shared directions at layer 7 (where hidden_load=0) are still readable,
that's the strongest "readable but not causal" data point in the paper.
"""

import json
import os
import numpy as np

from analysis import run_probing
from config_deep8 import ModelConfigDeep8
from data import load_eval_set
from step9_p1 import extract_with_max_dims, K_PCA


OUT = "results/deep8/layer_probing.json"
RESULTS_DIR = "results/deep8"


def load_aligned(site_name):
    d = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(d, f))
    return out


def main():
    cfg = ModelConfigDeep8()
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    meta = eval_data["metadata"]
    print(f"Probing 8-layer zoo shared directions at layers 3, 5, 7")
    print(f"  eval set: {len(meta)} problems")

    sites = ["layer3_result_0", "layer5_result_0", "layer7_result_0"]
    out = {}
    for site in sites:
        aligned = load_aligned(site)
        if not aligned:
            print(f"  {site}: no aligned data")
            continue
        print(f"\n=== {site} ({len(aligned)} models) ===")
        ex = extract_with_max_dims(aligned, max_dims=10, eps_scale=1e-8, k_pca=K_PCA)
        shared = ex["shared_dirs"]
        shared_u = shared / np.linalg.norm(shared, axis=1, keepdims=True)
        probe_res = run_probing(shared_u, aligned, meta, cfg)

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
