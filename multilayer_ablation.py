"""Path 2: Multi-layer simultaneous ablation to test 'shared = redundant' hypothesis.

If shared directions are redundant because the network has backup pathways,
ablating them at multiple layers simultaneously should eventually overwhelm
the backups. The curve of accuracy vs number of ablated layers maps redundancy depth.
"""

import os
import json
import numpy as np
import torch
from config import ModelConfig
from model import ArithmeticTransformer
from data import load_eval_set
from collect_activations import get_converged_models
from analysis import (
    extract_shared_subspace, procrustes_align, get_model_activations,
    evaluate_with_hook
)


def run_multilayer_ablation():
    cfg = ModelConfig()
    device = "cuda"
    results_dir = "results"
    act_dir = "activations"

    # Load models and eval data
    models = get_converged_models("models")
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]

    # Load selected site info
    selected_sites = json.load(open(os.path.join(results_dir, "selected_sites.json")))
    top_site = selected_sites[0]
    target_pos = top_site["position_idx"]

    # For each layer, extract shared directions and alignment
    print("Extracting shared directions at each layer...")
    layer_shared_dirs = {}
    layer_aligned_acts = {}

    ref_model = [m for m in models if m["frozen_component"] is None and m["seed"] == 0][0]

    for layer in range(cfg.n_layers):
        print(f"  Layer {layer}...")
        # Load activations at this layer, target position
        aligned = {}
        ref_path = os.path.join(act_dir, ref_model["model_name"],
                                f"layer{layer}_stratified.npy")
        X_ref = np.load(ref_path)[:, target_pos, :]
        aligned[ref_model["model_name"]] = X_ref - X_ref.mean(axis=0)

        for m in models:
            if m["model_name"] == ref_model["model_name"]:
                continue
            X_k = np.load(os.path.join(act_dir, m["model_name"],
                                       f"layer{layer}_stratified.npy"))[:, target_pos, :]
            R, _ = procrustes_align(X_ref, X_k)
            aligned[m["model_name"]] = (X_k - X_k.mean(axis=0)) @ R

        layer_aligned_acts[layer] = aligned

        # Extract shared subspace
        sub = extract_shared_subspace(aligned, max_dims=10)
        layer_shared_dirs[layer] = sub["shared_dirs"]  # [10, d_model]
        print(f"    Top eigenvalue: {sub['shared_eigenvalues'][0]:.3f}")

    # Now run ablation experiments
    print("\nRunning multi-layer ablation...")
    n_dirs = 5  # ablate top-5 shared directions

    # Layer combinations to test
    layer_combos = [
        [],           # no ablation (baseline)
        [0],          # single layers
        [1],
        [2],
        [3],
        [0, 1],       # pairs
        [1, 2],
        [2, 3],
        [0, 1, 2],    # triples
        [1, 2, 3],
        [0, 1, 2, 3], # all layers
    ]

    results = {"shared": {}, "random": {}}

    for m_info in models[:10]:  # use first 10 models for speed
        name = m_info["model_name"]
        print(f"\n  Model: {name}")

        model = ArithmeticTransformer(cfg).to(device)
        state_dict = torch.load(m_info["model_path"], map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Baseline accuracy
        baseline_acc = evaluate_with_hook(model, eval_tokens, cfg, device)

        for combo in layer_combos:
            combo_key = str(combo)
            if combo_key not in results["shared"]:
                results["shared"][combo_key] = []
                results["random"][combo_key] = []

            if not combo:
                results["shared"][combo_key].append(baseline_acc)
                results["random"][combo_key].append(baseline_acc)
                continue

            # Get rotation matrices for this model at each layer
            R_matrices = {}
            for layer in combo:
                ref_path = os.path.join(act_dir, ref_model["model_name"],
                                        f"layer{layer}_stratified.npy")
                X_ref = np.load(ref_path)[:, target_pos, :]
                if name == ref_model["model_name"]:
                    R_matrices[layer] = np.eye(cfg.d_model)
                else:
                    X_k = np.load(os.path.join(act_dir, name,
                                               f"layer{layer}_stratified.npy"))[:, target_pos, :]
                    R, _ = procrustes_align(X_ref, X_k)
                    R_matrices[layer] = R

            # Compute mean activations for mean-ablation at each layer
            layer_means = {}
            for layer in combo:
                acts = get_model_activations(model, eval_tokens, layer, device)
                h = acts[:, target_pos, :]  # CPU

                native_dirs = (R_matrices[layer] @ layer_shared_dirs[layer][:n_dirs].T).T
                native_dirs_t = torch.tensor(native_dirs, dtype=torch.float32)
                projs = h @ native_dirs_t.T
                layer_means[layer] = {
                    "dirs": torch.tensor(native_dirs, dtype=torch.float32, device=device),
                    "mean_projs": projs.mean(dim=0).to(device),
                }

            # Shared direction ablation across all layers in combo
            handles = []
            for layer in combo:
                D = layer_means[layer]["dirs"].T  # [d, n_dirs]
                DtD_inv = torch.linalg.inv(D.T @ D)
                P = D @ DtD_inv @ D.T
                mean_component = D @ DtD_inv @ layer_means[layer]["mean_projs"]

                def make_hook(P_mat, mean_comp, pos):
                    def hook_fn(module, input, output):
                        x = output.clone()
                        h = x[:, pos, :]
                        h_proj = h @ P_mat
                        x[:, pos, :] = h - h_proj + mean_comp.unsqueeze(0)
                        return x
                    return hook_fn

                handle = model.layers[layer].register_forward_hook(
                    make_hook(P, mean_component, target_pos))
                handles.append(handle)

            shared_acc = evaluate_with_hook(model, eval_tokens, cfg, device)
            for h in handles:
                h.remove()

            results["shared"][combo_key].append(shared_acc)

            # Random direction ablation (same number of dirs, same layers)
            handles = []
            rng = np.random.RandomState(42)
            for layer in combo:
                rand_dirs = rng.randn(n_dirs, cfg.d_model).astype(np.float32)
                rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
                rand_dirs_t = torch.tensor(rand_dirs, device=device)

                acts = get_model_activations(model, eval_tokens, layer, device)
                h = acts[:, target_pos, :]
                rand_projs = h @ torch.tensor(rand_dirs, device="cpu").T
                rand_mean = rand_projs.mean(dim=0).to(device)

                D_r = rand_dirs_t.T
                DtD_inv_r = torch.linalg.inv(D_r.T @ D_r)
                P_r = D_r @ DtD_inv_r @ D_r.T
                mean_comp_r = D_r @ DtD_inv_r @ rand_mean

                def make_hook_r(P_mat, mean_comp, pos):
                    def hook_fn(module, input, output):
                        x = output.clone()
                        h = x[:, pos, :]
                        h_proj = h @ P_mat
                        x[:, pos, :] = h - h_proj + mean_comp.unsqueeze(0)
                        return x
                    return hook_fn

                handle = model.layers[layer].register_forward_hook(
                    make_hook_r(P_r, mean_comp_r, target_pos))
                handles.append(handle)

            random_acc = evaluate_with_hook(model, eval_tokens, cfg, device)
            for h in handles:
                h.remove()

            results["random"][combo_key].append(random_acc)

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-LAYER ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Layers':<20} {'Shared Drop':>12} {'Random Drop':>12} {'Ratio':>8}")
    print("-" * 55)

    for combo in layer_combos:
        combo_key = str(combo)
        shared_accs = np.array(results["shared"][combo_key])
        random_accs = np.array(results["random"][combo_key])
        baseline_accs = np.array(results["shared"]["[]"])

        shared_drop = np.mean(baseline_accs) - np.mean(shared_accs)
        random_drop = np.mean(baseline_accs) - np.mean(random_accs)
        ratio = shared_drop / random_drop if random_drop > 0.0001 else 0

        label = "none" if not combo else "+".join(f"L{l}" for l in combo)
        print(f"{label:<20} {shared_drop:>11.4f} {random_drop:>11.4f} {ratio:>7.2f}")

    # Save
    with open(os.path.join(results_dir, "multilayer_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {results_dir}/multilayer_ablation.json")


if __name__ == "__main__":
    run_multilayer_ablation()
