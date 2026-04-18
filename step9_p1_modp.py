"""Full mod-p P1 protocol (A1 + A2 + A3): collect, align, extract, ablate, joint-ablate.

Per MOD_P_PREREGISTRATION.md, runs at 4 sites:
  - layer1_equals (position 3, layer 1)
  - layer2_equals (position 3, layer 2)
  - layer3_equals (position 3, layer 3)
  - layer3_plus   (position 1, layer 3) — CONTROL

Outputs:
  results/modp/{p1_results.json, joint_ablation.json, summary.json}
"""

import json
import os
import time

import numpy as np
import torch
from scipy import linalg

from analysis import evaluate_with_hook, get_model_activations, procrustes_align
from config_modp import ModelConfigModP
from data_modp import load_eval_set
from model import ArithmeticTransformer
from step9_p1 import (
    extract_with_max_dims,
    make_ablation_hook,
    complement_top_k,
    whitened_random_subspace,
    orthogonalize_against,
    projection_trace_variance,
    K_PCA,
)

MODELS_DIR = "models_modp"
EVAL_DIR = "eval_sets_modp"
ACT_DIR = "activations_modp"
RESULTS_DIR = "results/modp"

# Pre-registered sites (MOD_P_PREREGISTRATION.md)
SITES_MODP = {
    "layer1_equals": (1, 3),
    "layer2_equals": (2, 3),
    "layer3_equals": (3, 3),
    "layer3_plus":   (3, 1),   # CONTROL
}
PRIMARY_SITES_MODP = ["layer1_equals", "layer2_equals", "layer3_equals"]
CONTROL_SITE_MODP = "layer3_plus"

K_VALUES = [4, 8, 12, 16]
K_PRIMARY = 8
N_RANDOM = 100


def get_converged_models_modp():
    models = []
    if not os.path.isdir(MODELS_DIR):
        return models
    for entry in sorted(os.listdir(MODELS_DIR)):
        meta_path = os.path.join(MODELS_DIR, entry, "metadata.json")
        model_path = os.path.join(MODELS_DIR, entry, "model.pt")
        if os.path.exists(meta_path) and os.path.exists(model_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("converged"):
                meta["model_name"] = entry
                meta["model_path"] = model_path
                models.append(meta)
    return models


def collect_activations(models, cfg, eval_tokens, device):
    """Collect layer-l activations at position p for each model, at each requested site.
    Writes activations_modp/<model>/layer<l>.npy of shape [n_problems, seq_len, d]."""
    os.makedirs(ACT_DIR, exist_ok=True)
    needed_layers = sorted({site[0] for site in SITES_MODP.values()})

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

        with torch.no_grad():
            _, hiddens = mdl(eval_tokens.to(device), return_all_hiddens=True)
            for l in needed_layers:
                np.save(os.path.join(out_dir, f"layer{l}.npy"),
                        hiddens[l].cpu().numpy().astype(np.float32))

        del mdl
        torch.cuda.empty_cache()


def align_site(models, site_name, ref_name, cfg):
    layer, pos = SITES_MODP[site_name]
    aligned = {}
    X_ref = np.load(os.path.join(ACT_DIR, ref_name, f"layer{layer}.npy"))[:, pos, :]
    aligned[ref_name] = X_ref - X_ref.mean(axis=0)
    os.makedirs(os.path.join(RESULTS_DIR, f"aligned_{site_name}"), exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, f"aligned_{site_name}", f"{ref_name}.npy"),
            aligned[ref_name])
    for m in models:
        name = m["model_name"]
        if name == ref_name:
            continue
        X_k = np.load(os.path.join(ACT_DIR, name, f"layer{layer}.npy"))[:, pos, :]
        R, _ = procrustes_align(X_ref, X_k)
        X_a = (X_k - X_k.mean(axis=0)) @ R
        aligned[name] = X_a
        np.save(os.path.join(RESULTS_DIR, f"R_{name}_{site_name}.npy"), R)
        np.save(os.path.join(RESULTS_DIR, f"aligned_{site_name}", f"{name}.npy"), X_a)
    return aligned


def native_basis_dirs_modp(ref_dirs, model_name, site_name, d):
    """Convert reference-frame directions to native basis using saved R."""
    R_path = os.path.join(RESULTS_DIR, f"R_{model_name}_{site_name}.npy")
    if os.path.exists(R_path):
        R = np.load(R_path)
    else:
        R = np.eye(d)  # reference model
    return (R @ ref_dirs.T).T


def run_site_modp(models, site_name, cfg, eval_tokens, device, k_values, n_random):
    layer, pos = SITES_MODP[site_name]
    print(f"\n=== {site_name} (layer={layer}, pos={pos}) ===")

    # 1. Align activations
    ref = next((m for m in models if m["frozen_component"] is None and m["seed"] == 0),
               models[0])
    aligned = align_site(models, site_name, ref["model_name"], cfg)

    # 2. Extract shared + bottom (C_total-orthonormal)
    ex = extract_with_max_dims(aligned, max_dims=max(k_values), eps_scale=1e-8, k_pca=K_PCA)
    shared = ex["shared_dirs"]
    bottom = ex["bottom_dirs"]
    C_total = ex["C_total"]
    C_total_plus_ridge = ex["C_total_plus_ridge"]
    d = ex["d"]

    # Complement top-k per k
    complement_per_k = {}
    for k in k_values:
        if k > len(shared) or 2 * k > d:
            continue
        complement_per_k[k] = complement_top_k(shared[:k], C_total, k)

    # Ortho variant per k
    ortho_per_k = {}
    for k in k_values:
        if k > len(bottom) or k > len(shared) or 2 * k > d:
            continue
        oh, kept = orthogonalize_against(bottom[:k], shared[:k], C_total=C_total)
        ortho_per_k[k] = oh

    # 3. Per-model ablation
    per_model = {}
    rng = np.random.RandomState(1337)
    for i, m in enumerate(models):
        name = m["model_name"]
        t0 = time.time()
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()

        baseline_acc = evaluate_with_hook_modp(mdl, eval_tokens, cfg, device,
                                                 hook_layer=None, hook_fn=None)
        acts = get_model_activations_modp(mdl, eval_tokens, layer, device)
        acts_at_pos = acts[:, pos, :]

        def ablate(ref_dirs):
            native = native_basis_dirs_modp(ref_dirs, name, site_name, d)
            t = torch.tensor(native, dtype=torch.float32, device=device)
            projs = acts_at_pos @ t.cpu().T
            mean_projs = projs.mean(dim=0).to(device)
            hook = make_ablation_hook(t, mean_projs, pos)
            return float(evaluate_with_hook_modp(mdl, eval_tokens, cfg, device,
                                                   hook_layer=layer, hook_fn=hook))

        entry = {
            "baseline_acc": float(baseline_acc),
            "shared": {},
            "anti_shared_raw": {},
            "anti_shared_ortho": {},
            "complement_top_k": {},
            "joint": {},
            "random": {},
        }
        for k in k_values:
            if k > len(shared):
                continue
            entry["shared"][k] = float(baseline_acc - ablate(shared[:k]))
            entry["anti_shared_raw"][k] = float(baseline_acc - ablate(bottom[:k]))
            if k in ortho_per_k and ortho_per_k[k].shape[0] > 0:
                entry["anti_shared_ortho"][k] = float(baseline_acc - ablate(ortho_per_k[k]))
            if k in complement_per_k:
                entry["complement_top_k"][k] = float(baseline_acc - ablate(complement_per_k[k]))
                # Joint ablation
                joint_dirs = np.vstack([shared[:k], complement_per_k[k]])
                entry["joint"][k] = float(baseline_acc - ablate(joint_dirs))

        # Random baselines
        for k in k_values:
            if k > d:
                continue
            drops = []
            for _ in range(n_random):
                r = whitened_random_subspace(k, d, C_total_plus_ridge, rng).astype(np.float32)
                drops.append(float(baseline_acc - ablate(r)))
            entry["random"][k] = drops

        # Projection traces for reporting
        entry["projection_trace_variance"] = {
            "shared": {k: projection_trace_variance(shared[:k], C_total)
                        for k in k_values if k <= len(shared)},
            "anti_shared_raw": {k: projection_trace_variance(bottom[:k], C_total)
                                 for k in k_values if k <= len(bottom)},
            "complement_top_k": {k: projection_trace_variance(complement_per_k[k], C_total)
                                  for k in k_values if k in complement_per_k},
        }

        per_model[name] = entry
        del mdl
        torch.cuda.empty_cache()
        if i % 6 == 0:
            print(f"  [{i+1}/{len(models)}] {name}  (t={time.time()-t0:.1f}s)")

    return {
        "site": site_name,
        "layer": layer,
        "position_idx": pos,
        "extraction": {
            "eigenvalues_first5": ex["eigenvalues"][:5],
            "eigenvalues_last5": ex["eigenvalues"][-5:],
            "denom_condition_number": ex["denom_condition_number"],
            "ridge": ex["ridge"],
            "d": ex["d"],
            "k_pca": ex["k_pca"],
            "pca_eigvals_kept_fraction": ex["pca_eigvals_kept_fraction"],
        },
        "models": per_model,
    }


def evaluate_with_hook_modp(model, eval_tokens, cfg, device,
                              hook_layer=None, hook_fn=None, batch_size=512):
    """Accuracy: predict token at position 4 (result) from logits at position 3 (=)."""
    model.eval()
    if hook_layer is not None and hook_fn is not None:
        handle = model.layers[hook_layer].register_forward_hook(hook_fn)
    else:
        handle = None
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], batch_size):
            batch = eval_tokens[i:i+batch_size].to(device)
            logits = model(batch)
            pred = logits[:, 3, :].argmax(dim=-1)     # predict at '=' position
            true = batch[:, 4]                        # target is result token
            correct += (pred == true).sum().item()
            total += batch.shape[0]
    if handle is not None:
        handle.remove()
    return correct / total


def get_model_activations_modp(model, eval_tokens, layer, device, batch_size=512):
    """Return layer activations for all eval inputs, shape [N, seq_len, d_model]."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], batch_size):
            batch = eval_tokens[i:i+batch_size].to(device)
            _, hiddens = model(batch, return_all_hiddens=True)
            chunks.append(hiddens[layer].cpu())
    return torch.cat(chunks, dim=0)


def main():
    cfg = ModelConfigModP()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = get_converged_models_modp()
    print(f"Found {len(models)} converged mod-p models on {device}")
    if len(models) < 6:
        print("Too few converged models; aborting.")
        return

    eval_data = load_eval_set(os.path.join(EVAL_DIR, "full_grid"))
    eval_tokens = eval_data["tokens"]
    print(f"Full-grid eval set: {eval_tokens.shape[0]} problems")

    print("\nCollecting activations...")
    collect_activations(models, cfg, eval_tokens, device)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "config": {
            "p": cfg.p,
            "k_values": K_VALUES,
            "k_primary": K_PRIMARY,
            "n_random_trials": N_RANDOM,
            "sites": list(SITES_MODP.keys()),
            "n_models": len(models),
            "d_model": cfg.d_model,
        },
        "primary": {},
    }

    for site in SITES_MODP:
        t0 = time.time()
        results["primary"][site] = run_site_modp(
            models, site, cfg, eval_tokens, device,
            K_VALUES, N_RANDOM,
        )
        with open(os.path.join(RESULTS_DIR, "p1_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ({time.time()-t0:.1f}s for site)")

    # Summary at k=8
    summary = {}
    for site, sd in results["primary"].items():
        mdls = sd["models"]
        ss = [m["shared"].get(K_PRIMARY, None) for m in mdls.values()]
        cc = [m["complement_top_k"].get(K_PRIMARY, None) for m in mdls.values()]
        jj = [m["joint"].get(K_PRIMARY, None) for m in mdls.values()]
        rr = [np.mean(m["random"].get(K_PRIMARY, [])) for m in mdls.values() if m["random"].get(K_PRIMARY)]
        def filt(xs): return [x for x in xs if x is not None]
        ss, cc, jj = filt(ss), filt(cc), filt(jj)
        if ss and cc and jj and rr:
            summary[site] = {
                "mean_shared": float(np.mean(ss)),
                "mean_complement": float(np.mean(cc)),
                "mean_joint": float(np.mean(jj)),
                "mean_random": float(np.mean(rr)),
                "hidden_load": float(np.mean(jj) - np.mean(cc)),
            }

    results["summary"] = summary
    with open(os.path.join(RESULTS_DIR, "p1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (k=8) ===")
    print(f"{'site':>18}  {'shared':>9}  {'comp':>9}  {'joint':>9}  {'rand':>9}  {'hidden':>9}")
    for site, s in summary.items():
        print(f"{site:>18}  {s['mean_shared']:>9.4f}  {s['mean_complement']:>9.4f}  "
              f"{s['mean_joint']:>9.4f}  {s['mean_random']:>9.4f}  {s['hidden_load']:>9.4f}")

    print(f"\nSaved {os.path.join(RESULTS_DIR, 'p1_results.json')}")


if __name__ == "__main__":
    main()
