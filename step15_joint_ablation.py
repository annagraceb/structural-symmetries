"""Step 15: Joint ablation test (Claude paradox-hunter concern from final review).

Question: at layer-3 sites where shared ≈ 0 and complement ≈ 0.4, is the
shared subspace truly causally inert, or is it redundant (complement
compensates when shared alone is ablated)?

Disambiguation: jointly ablate (shared ∪ complement), compare to each alone.

  - If drop(shared ∪ complement) ≈ drop(complement alone):
    shared is truly causally inert. The "functional universality collapse" reading is correct.

  - If drop(shared ∪ complement) >> drop(complement alone):
    shared carries causal load that is redundantly realized in the complement;
    ablating shared alone is masked by compensation. The "functional gating"
    reading is overstated.

Run at all 5 layer-3 task-relevant sites (result digits 0-4) + one primary
layer-1 site for comparison. k=8, 33 models.

Outputs results/p1/joint_ablation.json.
"""

import json
import os
import time

import numpy as np
import torch

from analysis import get_model_activations, evaluate_with_hook
from collect_activations import get_converged_models
from config import ModelConfig
from data import load_eval_set
from model import ArithmeticTransformer
from step9_p1 import (
    extract_with_max_dims, complement_top_k, load_aligned_acts,
    align_site_on_the_fly, make_ablation_hook, native_basis_dirs,
    K_PCA, SITE_LAYER_POS,
)


OUT = "results/p1/joint_ablation.json"
K = 8

# Sites to test. Layer-3 ones = where the paradox matters most.
SITES = {
    "layer1_result_0": (1, 12),        # layer-1 comparison (shared > complement)
    "layer3_result_0": (3, 12),        # layer-3 (shared inert, complement strong)
    "layer3_result_1": (3, 13),
    "layer3_result_2": (3, 14),
    "layer3_result_3": (3, 15),
    "layer3_result_4": (3, 16),
}


def run_site(site_name, models, cfg, eval_tokens, device):
    layer, pos = SITES[site_name]
    print(f"\n=== Joint ablation: {site_name} (layer={layer}, pos={pos}) ===")

    # Load or align
    aligned_dir = os.path.join("results", f"aligned_{site_name}")
    if os.path.isdir(aligned_dir):
        aligned = load_aligned_acts(site_name)
    else:
        ref = next((m for m in models if m["frozen_component"] is None and m["seed"] == 0),
                   models[0])
        SITE_LAYER_POS[site_name] = (layer, pos)
        aligned = align_site_on_the_fly(models, site_name, ref["model_name"])

    SITE_LAYER_POS[site_name] = (layer, pos)
    ex = extract_with_max_dims(aligned, max_dims=K, eps_scale=1e-8, k_pca=K_PCA)
    shared = ex["shared_dirs"]                                          # [K, d]
    comp = complement_top_k(shared, ex["C_total"], K)                   # [K, d]
    joint = np.vstack([shared, comp])                                   # [2K, d]
    d = ex["d"]

    per_model = {}
    for i, m in enumerate(models):
        name = m["model_name"]
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()

        baseline_acc = evaluate_with_hook(mdl, eval_tokens, cfg, device,
                                           hook_layer=None, hook_fn=None)
        acts = get_model_activations(mdl, eval_tokens, layer, device)
        acts_at_pos = acts[:, pos, :]

        def ablate(ref_dirs):
            native = native_basis_dirs(ref_dirs, name, site_name, d)
            t = torch.tensor(native, dtype=torch.float32, device=device)
            projs = acts_at_pos @ t.cpu().T
            mean_projs = projs.mean(dim=0).to(device)
            hook = make_ablation_hook(t, mean_projs, pos)
            return float(evaluate_with_hook(mdl, eval_tokens, cfg, device,
                                              hook_layer=layer, hook_fn=hook))

        shared_acc = ablate(shared)
        comp_acc = ablate(comp)
        joint_acc = ablate(joint)

        per_model[name] = {
            "baseline_acc": float(baseline_acc),
            "shared_drop": float(baseline_acc - shared_acc),
            "complement_drop": float(baseline_acc - comp_acc),
            "joint_drop": float(baseline_acc - joint_acc),
        }
        del mdl
        torch.cuda.empty_cache()
        if i % 6 == 0:
            r = per_model[name]
            print(f"  [{i+1}/{len(models)}] {name:32}  "
                  f"shared={r['shared_drop']:.4f}  "
                  f"complement={r['complement_drop']:.4f}  "
                  f"joint={r['joint_drop']:.4f}")

    return {"site": site_name, "layer": layer, "position_idx": pos, "k": K,
            "models": per_model}


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_converged_models("models")
    print(f"Found {len(models)} models on {device}")
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]

    results = {"config": {"k": K, "n_models": len(models), "sites": list(SITES.keys())},
               "per_site": {}}
    for site in SITES:
        t0 = time.time()
        results["per_site"][site] = run_site(site, models, cfg, eval_tokens, device)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ({time.time()-t0:.1f}s)")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'site':>20}  {'shared':>9}  {'comp':>9}  {'joint':>9}  {'joint-c':>9}")
    summary = {}
    for site, sd in results["per_site"].items():
        ms = sd["models"]
        ss = [m["shared_drop"] for m in ms.values()]
        cc = [m["complement_drop"] for m in ms.values()]
        jj = [m["joint_drop"] for m in ms.values()]
        s = float(np.mean(ss)); c = float(np.mean(cc)); j = float(np.mean(jj))
        summary[site] = {"shared": s, "complement": c, "joint": j,
                         "joint_minus_complement": j - c,
                         "redundancy_verdict": (
                             "shared truly inert" if abs(j - c) < 0.02
                             else "shared carries hidden load" if j - c > 0.05
                             else "ambiguous")}
        print(f"{site:>20}  {s:>9.4f}  {c:>9.4f}  {j:>9.4f}  {j-c:>9.4f}  {summary[site]['redundancy_verdict']}")
    results["summary"] = summary

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
