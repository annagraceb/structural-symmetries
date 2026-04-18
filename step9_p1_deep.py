"""P1 protocol for the 6-layer deep zoo (Phase 2). Same task as main zoo.

Key sites: layer1 (early), layer3 (middle), layer5 (final before unembed),
at result_0 position. If the redundancy pattern shows at layer-5 result positions
the same way it did at layer-3 in the main 4-layer zoo, it's a
'last-residual-before-unembed' effect. If the pattern shifts with depth
(e.g. appears at layer-3 but NOT layer-5), it's layer-k-dependent.

Outputs results/deep/p1_results.json.
"""

import json
import os
import time

import numpy as np
import torch

from analysis import procrustes_align
from config_deep import ModelConfigDeep
from data import load_eval_set
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


MODELS_DIR = "models_deep"
ACT_DIR = "activations_deep"
RESULTS_DIR = "results/deep"

# PHASE2_PREREGISTRATION.md: test at ALL 6 layers at result_0, plus control.
# This disambiguates absolute-depth (H1), last-residual-before-unembed (H2),
# normalized-depth (H3), and monotone-gradient hypotheses.
SITES_DEEP = {
    "layer0_result_0": (0, 12),
    "layer1_result_0": (1, 12),
    "layer2_result_0": (2, 12),
    "layer3_result_0": (3, 12),
    "layer4_result_0": (4, 12),
    "layer5_result_0": (5, 12),
    # control
    "layer5_plus": (5, 5),
}
PRIMARY_SITES_DEEP = ["layer0_result_0", "layer1_result_0", "layer2_result_0",
                      "layer3_result_0", "layer4_result_0", "layer5_result_0"]
CONTROL_SITE_DEEP = "layer5_plus"

# Trimmed from [4,8,12,16] + 100 random to speed up the 6-layer × 45-model
# sweep at all 6 layers. Primary k=8 matches main-zoo analysis.
K_VALUES = [4, 8]
K_PRIMARY = 8
N_RANDOM = 30


def get_converged_models_deep():
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
    os.makedirs(ACT_DIR, exist_ok=True)
    needed_layers = sorted({site[0] for site in SITES_DEEP.values()})
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
        chunks_by_layer = {l: [] for l in needed_layers}
        with torch.no_grad():
            B = 256
            for i in range(0, eval_tokens.shape[0], B):
                batch = eval_tokens[i:i+B].to(device)
                _, hiddens = mdl(batch, return_all_hiddens=True)
                for l in needed_layers:
                    chunks_by_layer[l].append(hiddens[l].cpu())
        for l in needed_layers:
            arr = torch.cat(chunks_by_layer[l], dim=0).numpy().astype(np.float32)
            np.save(os.path.join(out_dir, f"layer{l}.npy"), arr)
        del mdl
        torch.cuda.empty_cache()


def align_site(models, site_name, ref_name, cfg):
    layer, pos = SITES_DEEP[site_name]
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


def native_basis_dirs_deep(ref_dirs, model_name, site_name, d):
    R_path = os.path.join(RESULTS_DIR, f"R_{model_name}_{site_name}.npy")
    if os.path.exists(R_path):
        R = np.load(R_path)
    else:
        R = np.eye(d)
    return (R @ ref_dirs.T).T


def evaluate_with_hook_deep(model, eval_tokens, device, hook_layer=None,
                             hook_fn=None, batch_size=256):
    """Accuracy on 5-digit addition: all 6 result digits correct."""
    model.eval()
    handle = None
    if hook_layer is not None and hook_fn is not None:
        handle = model.layers[hook_layer].register_forward_hook(hook_fn)
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], batch_size):
            batch = eval_tokens[i:i+batch_size].to(device)
            logits = model(batch)
            pred = logits[:, 11:17, :].argmax(dim=-1)
            true = batch[:, 12:18]
            match = (pred == true).all(dim=-1)
            correct += match.sum().item()
            total += batch.shape[0]
    if handle is not None:
        handle.remove()
    return correct / total


def get_model_activations_deep(model, eval_tokens, layer, device, batch_size=256):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], batch_size):
            batch = eval_tokens[i:i+batch_size].to(device)
            _, hiddens = model(batch, return_all_hiddens=True)
            chunks.append(hiddens[layer].cpu())
    return torch.cat(chunks, dim=0)


def run_site_deep(models, site_name, cfg, eval_tokens, device, k_values, n_random):
    layer, pos = SITES_DEEP[site_name]
    print(f"\n=== {site_name} (layer={layer}, pos={pos}) ===")

    ref = next((m for m in models if m["frozen_component"] is None and m["seed"] == 0),
               models[0])
    aligned = align_site(models, site_name, ref["model_name"], cfg)

    ex = extract_with_max_dims(aligned, max_dims=max(k_values), eps_scale=1e-8, k_pca=K_PCA)
    shared = ex["shared_dirs"]
    bottom = ex["bottom_dirs"]
    C_total = ex["C_total"]
    C_total_plus_ridge = ex["C_total_plus_ridge"]
    d = ex["d"]

    complement_per_k = {}
    for k in k_values:
        if k > len(shared) or 2 * k > d:
            continue
        complement_per_k[k] = complement_top_k(shared[:k], C_total, k)

    ortho_per_k = {}
    for k in k_values:
        if k > len(bottom) or k > len(shared) or 2 * k > d:
            continue
        oh, _ = orthogonalize_against(bottom[:k], shared[:k], C_total=C_total)
        ortho_per_k[k] = oh

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
            native = native_basis_dirs_deep(ref_dirs, name, site_name, d)
            t = torch.tensor(native, dtype=torch.float32, device=device)
            projs = acts_at_pos @ t.cpu().T
            mean_projs = projs.mean(dim=0).to(device)
            hook = make_ablation_hook(t, mean_projs, pos)
            return float(evaluate_with_hook_deep(mdl, eval_tokens, device,
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
                joint_dirs = np.vstack([shared[:k], complement_per_k[k]])
                entry["joint"][k] = float(baseline_acc - ablate(joint_dirs))

        for k in k_values:
            if k > d:
                continue
            drops = []
            for _ in range(n_random):
                r = whitened_random_subspace(k, d, C_total_plus_ridge, rng).astype(np.float32)
                drops.append(float(baseline_acc - ablate(r)))
            entry["random"][k] = drops

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


def main():
    cfg = ModelConfigDeep()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_converged_models_deep()
    print(f"Found {len(models)} converged deep models on {device}")
    if len(models) < 6:
        print("Too few converged models; aborting.")
        return

    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]

    print("\nCollecting activations...")
    collect_activations(models, cfg, eval_tokens, device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "config": {"n_layers": cfg.n_layers, "k_values": K_VALUES,
                   "k_primary": K_PRIMARY, "n_random_trials": N_RANDOM,
                   "sites": list(SITES_DEEP.keys()), "n_models": len(models),
                   "d_model": cfg.d_model},
        "primary": {},
    }

    for site in SITES_DEEP:
        t0 = time.time()
        results["primary"][site] = run_site_deep(
            models, site, cfg, eval_tokens, device, K_VALUES, N_RANDOM,
        )
        with open(os.path.join(RESULTS_DIR, "p1_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ({time.time()-t0:.1f}s for site)")

    # Summary at k=8
    summary = {}
    for site, sd in results["primary"].items():
        mdls = sd["models"]
        def extract_k(cond):
            return [m[cond].get(K_PRIMARY, None) for m in mdls.values()]
        def filt(xs): return [x for x in xs if x is not None]
        ss = filt(extract_k("shared"))
        cc = filt(extract_k("complement_top_k"))
        jj = filt(extract_k("joint"))
        rr = [np.mean(m["random"].get(K_PRIMARY, [])) for m in mdls.values()
              if m["random"].get(K_PRIMARY)]
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


if __name__ == "__main__":
    main()
