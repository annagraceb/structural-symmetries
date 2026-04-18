"""P1 protocol for 8-layer zoo (Phase 3). Tests N-1 invariance.

Per PHASE3_PREREGISTRATION.md, runs at 3 primary sites + 1 control:
  layer3_result_0, layer5_result_0, layer7_result_0, layer7_plus.

Outputs results/deep8/p1_results.json.
"""

import json
import os
import time

import numpy as np
import torch

from analysis import procrustes_align
from config_deep8 import ModelConfigDeep8
from data import load_eval_set
from model import ArithmeticTransformer
from step9_p1 import (
    extract_with_max_dims, make_ablation_hook, complement_top_k,
    whitened_random_subspace, orthogonalize_against,
    projection_trace_variance, K_PCA,
)
from step9_p1_deep import (
    evaluate_with_hook_deep, get_model_activations_deep,
)


MODELS_DIR = "models_deep8"
ACT_DIR = "activations_deep8"
RESULTS_DIR = "results/deep8"

SITES_DEEP8 = {
    "layer3_result_0": (3, 12),
    "layer5_result_0": (5, 12),
    "layer7_result_0": (7, 12),
    "layer7_plus":     (7, 5),
}
PRIMARY_SITES = ["layer3_result_0", "layer5_result_0", "layer7_result_0"]
CONTROL_SITE = "layer7_plus"

K_VALUES = [4, 8]
K_PRIMARY = 8
N_RANDOM = 30


def get_converged_models():
    models = []
    if not os.path.isdir(MODELS_DIR):
        return models
    for entry in sorted(os.listdir(MODELS_DIR)):
        mp = os.path.join(MODELS_DIR, entry, "metadata.json")
        pp = os.path.join(MODELS_DIR, entry, "model.pt")
        if os.path.exists(mp) and os.path.exists(pp):
            with open(mp) as f:
                meta = json.load(f)
            if meta.get("converged"):
                meta["model_name"] = entry
                meta["model_path"] = pp
                models.append(meta)
    return models


def collect_activations(models, cfg, eval_tokens, device):
    os.makedirs(ACT_DIR, exist_ok=True)
    needed_layers = sorted({site[0] for site in SITES_DEEP8.values()})
    for m in models:
        name = m["model_name"]
        out_dir = os.path.join(ACT_DIR, name)
        if all(os.path.exists(os.path.join(out_dir, f"layer{l}.npy")) for l in needed_layers):
            continue
        os.makedirs(out_dir, exist_ok=True)
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()
        chunks = {l: [] for l in needed_layers}
        with torch.no_grad():
            B = 256
            for i in range(0, eval_tokens.shape[0], B):
                batch = eval_tokens[i:i+B].to(device)
                _, hiddens = mdl(batch, return_all_hiddens=True)
                for l in needed_layers:
                    chunks[l].append(hiddens[l].cpu())
        for l in needed_layers:
            arr = torch.cat(chunks[l], dim=0).numpy().astype(np.float32)
            np.save(os.path.join(out_dir, f"layer{l}.npy"), arr)
        del mdl
        torch.cuda.empty_cache()


def align_site(models, site_name, ref_name, cfg):
    layer, pos = SITES_DEEP8[site_name]
    X_ref = np.load(os.path.join(ACT_DIR, ref_name, f"layer{layer}.npy"))[:, pos, :]
    aligned = {ref_name: X_ref - X_ref.mean(axis=0)}
    os.makedirs(os.path.join(RESULTS_DIR, f"aligned_{site_name}"), exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, f"aligned_{site_name}", f"{ref_name}.npy"),
            aligned[ref_name])
    for m in models:
        name = m["model_name"]
        if name == ref_name: continue
        X_k = np.load(os.path.join(ACT_DIR, name, f"layer{layer}.npy"))[:, pos, :]
        R, _ = procrustes_align(X_ref, X_k)
        X_a = (X_k - X_k.mean(axis=0)) @ R
        aligned[name] = X_a
        np.save(os.path.join(RESULTS_DIR, f"R_{name}_{site_name}.npy"), R)
        np.save(os.path.join(RESULTS_DIR, f"aligned_{site_name}", f"{name}.npy"), X_a)
    return aligned


def native_basis(ref_dirs, model_name, site_name, d):
    R_path = os.path.join(RESULTS_DIR, f"R_{model_name}_{site_name}.npy")
    if os.path.exists(R_path):
        R = np.load(R_path)
    else:
        R = np.eye(d)
    return (R @ ref_dirs.T).T


def run_site(models, site_name, cfg, eval_tokens, device):
    layer, pos = SITES_DEEP8[site_name]
    print(f"\n=== {site_name} (layer={layer}, pos={pos}) ===")

    ref = next((m for m in models if m["frozen_component"] is None and m["seed"] == 0),
               models[0])
    aligned = align_site(models, site_name, ref["model_name"], cfg)

    ex = extract_with_max_dims(aligned, max_dims=max(K_VALUES), eps_scale=1e-8, k_pca=K_PCA)
    shared = ex["shared_dirs"]
    bottom = ex["bottom_dirs"]
    C_total = ex["C_total"]
    C_total_plus_ridge = ex["C_total_plus_ridge"]
    d = ex["d"]

    complement_per_k = {}
    for k in K_VALUES:
        if k > len(shared) or 2 * k > d: continue
        complement_per_k[k] = complement_top_k(shared[:k], C_total, k)

    per_model = {}
    rng = np.random.RandomState(1337)
    for i, m in enumerate(models):
        name = m["model_name"]
        t0 = time.time()
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()

        baseline_acc = evaluate_with_hook_deep(mdl, eval_tokens, device)
        acts = get_model_activations_deep(mdl, eval_tokens, layer, device)
        acts_at_pos = acts[:, pos, :]

        def ablate(ref_dirs):
            native = native_basis(ref_dirs, name, site_name, d)
            t = torch.tensor(native, dtype=torch.float32, device=device)
            projs = acts_at_pos @ t.cpu().T
            mean_projs = projs.mean(dim=0).to(device)
            hook = make_ablation_hook(t, mean_projs, pos)
            return float(evaluate_with_hook_deep(mdl, eval_tokens, device,
                                                   hook_layer=layer, hook_fn=hook))

        entry = {"baseline_acc": float(baseline_acc),
                 "shared": {}, "complement_top_k": {}, "joint": {}, "random": {}}
        for k in K_VALUES:
            if k > len(shared): continue
            entry["shared"][k] = float(baseline_acc - ablate(shared[:k]))
            if k in complement_per_k:
                entry["complement_top_k"][k] = float(baseline_acc - ablate(complement_per_k[k]))
                joint_dirs = np.vstack([shared[:k], complement_per_k[k]])
                entry["joint"][k] = float(baseline_acc - ablate(joint_dirs))
        for k in K_VALUES:
            if k > d: continue
            drops = []
            for _ in range(N_RANDOM):
                r = whitened_random_subspace(k, d, C_total_plus_ridge, rng).astype(np.float32)
                drops.append(float(baseline_acc - ablate(r)))
            entry["random"][k] = drops

        per_model[name] = entry
        del mdl
        torch.cuda.empty_cache()
        if i % 4 == 0:
            print(f"  [{i+1}/{len(models)}] {name} (t={time.time()-t0:.1f}s)")

    return {
        "site": site_name, "layer": layer, "position_idx": pos,
        "extraction": {
            "eigenvalues_first5": ex["eigenvalues"][:5],
            "eigenvalues_last5": ex["eigenvalues"][-5:],
            "denom_condition_number": ex["denom_condition_number"],
            "ridge": ex["ridge"], "d": ex["d"], "k_pca": ex["k_pca"],
        },
        "models": per_model,
    }


def main():
    cfg = ModelConfigDeep8()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_converged_models()
    print(f"Found {len(models)} converged 8-layer models on {device}")
    if len(models) < 6:
        print("Too few converged models; aborting.")
        return
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nCollecting activations...")
    collect_activations(models, cfg, eval_tokens, device)

    results = {"config": {"n_layers": 8, "k_values": K_VALUES,
                          "k_primary": K_PRIMARY, "n_random": N_RANDOM,
                          "sites": list(SITES_DEEP8.keys()), "n_models": len(models)},
                "primary": {}}
    for site in SITES_DEEP8:
        t0 = time.time()
        results["primary"][site] = run_site(models, site, cfg, eval_tokens, device)
        with open(os.path.join(RESULTS_DIR, "p1_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ({time.time()-t0:.1f}s)")

    # Summary at k=8
    summary = {}
    for site, sd in results["primary"].items():
        ms = sd["models"]
        def g(cond):
            return [m[cond].get(K_PRIMARY, None) for m in ms.values() if m[cond].get(K_PRIMARY) is not None]
        ss = g("shared")
        cc = g("complement_top_k")
        jj = g("joint")
        rr = []
        for m in ms.values():
            r = m["random"].get(K_PRIMARY, []) or []
            rr.extend(r)
        if ss and cc and jj:
            summary[site] = {"shared": float(np.mean(ss)), "complement": float(np.mean(cc)),
                             "joint": float(np.mean(jj)),
                             "random": float(np.mean(rr)) if rr else 0.0,
                             "hidden": float(np.mean(jj) - np.mean(cc))}
    results["summary"] = summary
    with open(os.path.join(RESULTS_DIR, "p1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (k=8) ===")
    print(f"{'site':>20} {'shared':>9} {'comp':>9} {'joint':>9} {'rand':>9} {'hidden':>9}")
    for site, s in summary.items():
        print(f"{site:>20} {s['shared']:>9.4f} {s['complement']:>9.4f} "
              f"{s['joint']:>9.4f} {s['random']:>9.4f} {s['hidden']:>9.4f}")


if __name__ == "__main__":
    main()
