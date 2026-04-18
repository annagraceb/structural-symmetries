"""Steps 3-7: Full analysis pipeline.

Step 3: CKA cross-model similarity
Step 4: Procrustes alignment + SVCCA
Step 5: Shared subspace extraction
Step 6: Interpretation / probing
Step 7: Ablation experiments
"""

import os
import json
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from collections import defaultdict

from config import ModelConfig, FREEZABLE_COMPONENTS
from model import ArithmeticTransformer
from data import load_eval_set
from collect_activations import get_converged_models


# ============================================================
# Step 3: CKA
# ============================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices.

    X, Y: [n_samples, n_features]
    Returns scalar CKA similarity in [0, 1].
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtX = X.T @ X  # [d, d]
    YtY = Y.T @ Y

    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    YtX = Y.T @ X
    num = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


def linear_cka_remove_top_pc(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA after removing top-1 PC from both matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Remove top PC from X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X = X - (U[:, :1] * S[:1]) @ Vt[:1, :]

    # Remove top PC from Y
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    Y = Y - (U[:, :1] * S[:1]) @ Vt[:1, :]

    return linear_cka(X, Y)


def bootstrap_cka_ci(cka_values: list[float], n_bootstrap: int = 1000,
                      ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap confidence interval for CKA values.

    Returns (mean, lower, upper).
    """
    arr = np.array(cka_values)
    mean = arr.mean()
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.sort(boot_means)
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(mean), float(lower), float(upper)


def run_cka_analysis(act_dir: str, models: list[dict], cfg: ModelConfig,
                     results_dir: str):
    """Step 3: Compute CKA heatmaps across all model pairs."""
    os.makedirs(results_dir, exist_ok=True)
    n_models = len(models)
    model_names = [m["model_name"] for m in models]

    # Positions to analyze
    positions = {
        "operand_a_last": cfg.n_digits - 1,  # last digit of A
        "plus": cfg.n_digits,
        "operand_b_last": 2 * cfg.n_digits,
        "equals": cfg.result_start_pos - 1,
    }
    for r in range(cfg.n_result_digits):
        positions[f"result_{r}"] = cfg.result_start_pos + r

    # Compute CKA for all pairs at all sites
    results = {}

    for layer in range(cfg.n_layers):
        print(f"  CKA layer {layer}...")
        # Load all activations for this layer
        acts = {}
        for m in models:
            path = os.path.join(act_dir, m["model_name"],
                                f"layer{layer}_stratified.npy")
            acts[m["model_name"]] = np.load(path)  # [2000, seq_len, 128]

        for pos_name, pos_idx in positions.items():
            site = f"layer{layer}_{pos_name}"
            cka_vals = []
            cka_no_pc_vals = []
            pair_details = []

            for i, j in itertools.combinations(range(n_models), 2):
                X = acts[model_names[i]][:, pos_idx, :]  # [2000, 128]
                Y = acts[model_names[j]][:, pos_idx, :]

                cka = linear_cka(X, Y)
                cka_no_pc = linear_cka_remove_top_pc(X, Y)
                cka_vals.append(cka)
                cka_no_pc_vals.append(cka_no_pc)
                pair_details.append({
                    "model_i": model_names[i],
                    "model_j": model_names[j],
                    "cka": cka,
                    "cka_no_top_pc": cka_no_pc,
                })

            mean, lower, upper = bootstrap_cka_ci(cka_vals)
            mean_no_pc, lower_no_pc, upper_no_pc = bootstrap_cka_ci(cka_no_pc_vals)

            results[site] = {
                "layer": layer,
                "position": pos_name,
                "position_idx": pos_idx,
                "cka_mean": mean,
                "cka_ci_lower": lower,
                "cka_ci_upper": upper,
                "cka_no_top_pc_mean": mean_no_pc,
                "cka_no_top_pc_ci_lower": lower_no_pc,
                "cka_no_top_pc_ci_upper": upper_no_pc,
                "n_pairs": len(cka_vals),
            }

    # Three-group comparison (3b)
    group_results = compute_three_group_cka(act_dir, models, cfg)
    results["three_group"] = group_results

    # Within vs across config (3c)
    config_results = compute_within_across_cka(act_dir, models, cfg)
    results["within_across"] = config_results

    with open(os.path.join(results_dir, "step3_cka.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n  CKA Heatmap Summary (top sites):")
    sites = [(k, v) for k, v in results.items()
             if isinstance(v, dict) and "cka_mean" in v]
    sites.sort(key=lambda x: x[1]["cka_mean"], reverse=True)
    for name, data in sites[:10]:
        print(f"    {name}: CKA={data['cka_mean']:.3f} "
              f"[{data['cka_ci_lower']:.3f}, {data['cka_ci_upper']:.3f}] "
              f"| no-PC1: {data['cka_no_top_pc_mean']:.3f}")

    return results


def compute_three_group_cka(act_dir, models, cfg):
    """Step 3b: CKA for baseline-vs-baseline, freeze-vs-freeze, freeze-vs-baseline."""
    baselines = [m for m in models if m["frozen_component"] is None]
    frozen = [m for m in models if m["frozen_component"] is not None]

    # Use the best CKA site (layer 3, last result position as a reasonable default)
    # We'll compute across a few key sites
    key_sites = []
    for layer in range(cfg.n_layers):
        for r in range(cfg.n_result_digits):
            key_sites.append((layer, cfg.result_start_pos + r))

    groups = {"baseline_vs_baseline": [], "freeze_vs_freeze": [], "freeze_vs_baseline": []}

    for layer, pos in key_sites:
        acts = {}
        for m in models:
            path = os.path.join(act_dir, m["model_name"],
                                f"layer{layer}_stratified.npy")
            acts[m["model_name"]] = np.load(path)[:, pos, :]

        # Baseline vs baseline
        for i, j in itertools.combinations(range(len(baselines)), 2):
            X = acts[baselines[i]["model_name"]]
            Y = acts[baselines[j]["model_name"]]
            groups["baseline_vs_baseline"].append(linear_cka(X, Y))

        # Freeze vs freeze (different configs)
        for i, j in itertools.combinations(range(len(frozen)), 2):
            if frozen[i]["frozen_component"] != frozen[j]["frozen_component"]:
                X = acts[frozen[i]["model_name"]]
                Y = acts[frozen[j]["model_name"]]
                groups["freeze_vs_freeze"].append(linear_cka(X, Y))

        # Freeze vs baseline
        for fm in frozen:
            for bm in baselines:
                X = acts[fm["model_name"]]
                Y = acts[bm["model_name"]]
                groups["freeze_vs_baseline"].append(linear_cka(X, Y))

    result = {}
    for group_name, vals in groups.items():
        if vals:
            arr = np.array(vals)
            result[group_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "n": len(vals),
            }
    return result


def compute_within_across_cka(act_dir, models, cfg):
    """Step 3c: Within-config vs across-config CKA."""
    # Group models by frozen component
    by_config = defaultdict(list)
    for m in models:
        key = m["frozen_component"] or "baseline"
        by_config[key].append(m)

    # Use result positions at all layers — load once per layer
    within_vals = []
    across_vals = []

    for layer in range(cfg.n_layers):
        # Load all activations for this layer once
        full_acts = {}
        for m in models:
            path = os.path.join(act_dir, m["model_name"],
                                f"layer{layer}_stratified.npy")
            full_acts[m["model_name"]] = np.load(path)  # [2000, seq_len, 128]

        for r in range(cfg.n_result_digits):
            pos = cfg.result_start_pos + r
            acts = {name: arr[:, pos, :] for name, arr in full_acts.items()}

            # Within-config
            for config_name, config_models in by_config.items():
                for i, j in itertools.combinations(range(len(config_models)), 2):
                    X = acts[config_models[i]["model_name"]]
                    Y = acts[config_models[j]["model_name"]]
                    within_vals.append(linear_cka(X, Y))

            # Across-config (different frozen component, use first seed from each)
            configs = list(by_config.keys())
            for ci, cj in itertools.combinations(range(len(configs)), 2):
                m_i = by_config[configs[ci]][0]
                m_j = by_config[configs[cj]][0]
                X = acts[m_i["model_name"]]
                Y = acts[m_j["model_name"]]
                across_vals.append(linear_cka(X, Y))

        del full_acts

    return {
        "within_config": {
            "mean": float(np.mean(within_vals)),
            "std": float(np.std(within_vals)),
            "n": len(within_vals),
        },
        "across_config": {
            "mean": float(np.mean(across_vals)),
            "std": float(np.std(across_vals)),
            "n": len(across_vals),
        },
    }


def select_top_sites(cka_results: dict, cfg: ModelConfig) -> list[dict]:
    """Decision gate: select top extraction sites based on CKA criteria."""
    candidates = []
    for key, data in cka_results.items():
        if not isinstance(data, dict) or "cka_mean" not in data:
            continue
        # Decision criteria from spec v0.3
        if (data["cka_mean"] > 0.5
                and data["cka_ci_lower"] > 0.35
                and data["cka_no_top_pc_mean"] > 0.3):
            candidates.append({
                "site": key,
                "layer": data["layer"],
                "position_idx": data["position_idx"],
                "position_name": data["position"],
                "cka_mean": data["cka_mean"],
                "cka_no_pc": data["cka_no_top_pc_mean"],
            })

    candidates.sort(key=lambda x: x["cka_mean"], reverse=True)
    selected = candidates[:3]  # top 2-3 sites

    print(f"\n  Decision gate: {len(candidates)} sites pass criteria, selected top {len(selected)}")
    for s in selected:
        print(f"    {s['site']}: CKA={s['cka_mean']:.3f}, no-PC1={s['cka_no_pc']:.3f}")

    return selected


# ============================================================
# Step 4: Procrustes Alignment
# ============================================================

def procrustes_align(X_ref: np.ndarray, X_k: np.ndarray) -> tuple[np.ndarray, float]:
    """Orthogonal Procrustes alignment.

    Returns (rotation_matrix R, procrustes_residual).
    X_k @ R ≈ X_ref
    """
    X_ref_c = X_ref - X_ref.mean(axis=0)
    X_k_c = X_k - X_k.mean(axis=0)

    M = X_ref_c.T @ X_k_c  # [d, d]
    U, S, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T  # [d, d]

    aligned = X_k_c @ R
    residual = np.linalg.norm(aligned - X_ref_c, "fro") / np.linalg.norm(X_ref_c, "fro")

    return R, float(residual)


def random_rotation_baseline(X_ref: np.ndarray, X_k: np.ndarray,
                              n_samples: int = 50) -> float:
    """Null baseline: Procrustes distance to randomly rotated reference."""
    residuals = []
    rng = np.random.RandomState(42)
    d = X_ref.shape[1]

    for _ in range(n_samples):
        Q, _ = np.linalg.qr(rng.randn(d, d))
        X_rot = X_ref @ Q
        _, res = procrustes_align(X_rot, X_k)
        residuals.append(res)

    return float(np.mean(residuals))


def svcca_similarity(X: np.ndarray, Y: np.ndarray,
                      variance_threshold: float = 0.95) -> tuple[float, np.ndarray]:
    """SVCCA: SVD + CCA.

    Returns (mean canonical correlation, canonical vectors in X's space).
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    def reduce_dims(M, threshold):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        total_var = (S ** 2).sum()
        cumvar = np.cumsum(S ** 2) / total_var
        k = np.searchsorted(cumvar, threshold) + 1
        return M @ Vt[:k].T, Vt[:k]  # projected data, projection matrix

    X_red, Vx = reduce_dims(X, variance_threshold)
    Y_red, Vy = reduce_dims(Y, variance_threshold)

    # CCA
    n = X_red.shape[0]
    k = min(X_red.shape[1], Y_red.shape[1])

    Cxx = (X_red.T @ X_red) / n + 1e-8 * np.eye(X_red.shape[1])
    Cyy = (Y_red.T @ Y_red) / n + 1e-8 * np.eye(Y_red.shape[1])
    Cxy = (X_red.T @ Y_red) / n

    Cxx_inv_sqrt = linalg.sqrtm(np.linalg.inv(Cxx)).real
    Cyy_inv_sqrt = linalg.sqrtm(np.linalg.inv(Cyy)).real

    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    U, S_cca, Vt_cca = np.linalg.svd(T, full_matrices=False)

    mean_corr = float(S_cca[:k].mean())

    # Canonical vectors in original X space
    canon_vecs = Vx.T @ Cxx_inv_sqrt @ U[:, :k]

    return mean_corr, canon_vecs


def run_alignment(act_dir: str, models: list[dict], selected_sites: list[dict],
                   results_dir: str):
    """Step 4: Procrustes alignment of all models to reference."""
    os.makedirs(results_dir, exist_ok=True)

    # Reference = first baseline model (seed 0)
    ref_model = None
    for m in models:
        if m["frozen_component"] is None and m["seed"] == 0:
            ref_model = m
            break
    if ref_model is None:
        ref_model = models[0]  # fallback

    print(f"  Reference model: {ref_model['model_name']}")

    alignment_results = {}
    aligned_activations = {}

    for site in selected_sites:
        layer = site["layer"]
        pos = site["position_idx"]
        site_name = site["site"]
        print(f"\n  Aligning at {site_name}...")

        # Load reference activations
        ref_path = os.path.join(act_dir, ref_model["model_name"],
                                 f"layer{layer}_stratified.npy")
        X_ref = np.load(ref_path)[:, pos, :]  # [2000, 128]

        site_results = {
            "reference": ref_model["model_name"],
            "models": {},
        }
        site_aligned = {ref_model["model_name"]: X_ref.copy()}

        # Compute random baseline
        print(f"    Computing random rotation baseline...")
        baselines_for_null = [m for m in models
                               if m["model_name"] != ref_model["model_name"]
                               and m["frozen_component"] is None]
        if baselines_for_null:
            random_bl = random_rotation_baseline(
                X_ref,
                np.load(os.path.join(
                    act_dir, baselines_for_null[0]["model_name"],
                    f"layer{layer}_stratified.npy"
                ))[:, pos, :]
            )
        else:
            random_bl = 1.0
        site_results["random_baseline_residual"] = random_bl

        # Align each model
        baseline_residuals = []
        freeze_residuals = []
        svcca_fallback_models = []

        for m in models:
            if m["model_name"] == ref_model["model_name"]:
                continue

            X_k = np.load(os.path.join(
                act_dir, m["model_name"],
                f"layer{layer}_stratified.npy"
            ))[:, pos, :]

            R, residual = procrustes_align(X_ref, X_k)
            svcca_corr, svcca_vecs = svcca_similarity(X_ref, X_k)

            method = "procrustes"
            aligned = (X_k - X_k.mean(axis=0)) @ R

            # Track baseline vs freeze residuals
            if m["frozen_component"] is None:
                baseline_residuals.append(residual)
            else:
                freeze_residuals.append(residual)

            site_results["models"][m["model_name"]] = {
                "procrustes_residual": residual,
                "svcca_correlation": svcca_corr,
                "alignment_method": method,
                "frozen_component": m["frozen_component"],
            }
            site_aligned[m["model_name"]] = aligned

            # Save rotation matrix
            R_path = os.path.join(results_dir, f"R_{m['model_name']}_{site_name}.npy")
            np.save(R_path, R)

        # Step 4e: Check SVCCA fallback trigger
        mean_baseline_res = np.mean(baseline_residuals) if baseline_residuals else 0
        svcca_trigger = mean_baseline_res * 2.0

        for m in models:
            name = m["model_name"]
            if name == ref_model["model_name"] or name not in site_results["models"]:
                continue
            model_res = site_results["models"][name]["procrustes_residual"]
            if model_res > svcca_trigger and m["frozen_component"] is not None:
                svcca_fallback_models.append(name)
                site_results["models"][name]["alignment_method"] = "svcca_fallback"
                print(f"    SVCCA fallback for {name} "
                      f"(residual {model_res:.3f} > threshold {svcca_trigger:.3f})")

        site_results["baseline_mean_residual"] = float(mean_baseline_res)
        site_results["svcca_trigger_threshold"] = float(svcca_trigger)
        site_results["n_svcca_fallback"] = len(svcca_fallback_models)

        alignment_results[site_name] = site_results
        aligned_activations[site_name] = site_aligned

        # Print summary for this site
        all_res = [v["procrustes_residual"]
                    for v in site_results["models"].values()]
        print(f"    Residuals: mean={np.mean(all_res):.4f}, "
              f"std={np.std(all_res):.4f}, random_baseline={random_bl:.4f}")
        print(f"    Baseline residuals: {[f'{r:.4f}' for r in baseline_residuals]}")

    # Save results
    with open(os.path.join(results_dir, "step4_alignment.json"), "w") as f:
        json.dump(alignment_results, f, indent=2)

    # Save aligned activations
    for site_name, site_aligned in aligned_activations.items():
        site_dir = os.path.join(results_dir, f"aligned_{site_name}")
        os.makedirs(site_dir, exist_ok=True)
        for model_name, arr in site_aligned.items():
            np.save(os.path.join(site_dir, f"{model_name}.npy"), arr)

    return alignment_results, aligned_activations


# ============================================================
# Step 5: Shared Subspace Extraction
# ============================================================

def extract_shared_subspace(aligned_acts: dict[str, np.ndarray],
                             max_dims: int = 10) -> dict:
    """Step 5c: Extract shared subspace via cross-model agreement.

    aligned_acts: {model_name: [2000, 128]} — Procrustes-aligned activations.
    Returns dict with shared directions and diagnostics.
    """
    names = list(aligned_acts.keys())
    K = len(names)
    stacked = np.stack([aligned_acts[n] for n in names])  # [K, 2000, 128]
    n_samples, d = stacked.shape[1], stacked.shape[2]

    # 5a: Per-dimension shared/informative score (vectorized)
    W_bar = np.var(stacked, axis=1).mean(axis=0)  # mean over models of per-model input variance
    V_bar = np.var(stacked, axis=0).mean(axis=0)  # mean over inputs of per-input model variance
    epsilon = 1e-8
    dim_scores = W_bar / (V_bar + epsilon)

    # 5b: Aligned PCA (stack all models)
    all_stacked = stacked.reshape(-1, d)  # [K*2000, 128]
    all_stacked = all_stacked - all_stacked.mean(axis=0)
    U_pca, S_pca, Vt_pca = np.linalg.svd(all_stacked, full_matrices=False)
    pca_dirs = Vt_pca[:max_dims]  # [max_dims, 128]

    # 5c: Cross-model agreement (the core extraction)
    # h_bar(x) = mean across models of h_k(x)
    h_bar = stacked.mean(axis=0)  # [2000, 128]
    h_bar_centered = h_bar - h_bar.mean(axis=0)

    # C_shared = covariance of h_bar across inputs
    C_shared = (h_bar_centered.T @ h_bar_centered) / n_samples

    # C_total = mean of per-model covariances
    C_total = np.zeros((d, d))
    for k in range(K):
        h_k = stacked[k] - stacked[k].mean(axis=0)
        C_total += (h_k.T @ h_k) / n_samples
    C_total /= K

    # Ratio eigenproblem: C_shared v = lambda C_total v
    # Equivalent to solving C_total^{-1} C_shared v = lambda v
    # Use scipy for generalized eigenvalue problem
    eigenvalues, eigenvectors = linalg.eigh(C_shared, C_total + epsilon * np.eye(d))

    # eigh returns in ascending order; we want descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    shared_dirs = eigenvectors[:, :max_dims].T  # [max_dims, 128]
    shared_eigenvalues = eigenvalues[:max_dims]

    # Bottom eigenvectors (lowest shared-to-total ratio) for Step 8 D' condition
    bottom_dirs = eigenvectors[:, -max_dims:].T  # [max_dims, 128]
    bottom_eigenvalues = eigenvalues[-max_dims:]

    # Normalize directions (eigh returns C_total-orthogonal, not unit-norm)
    shared_dirs = shared_dirs / np.linalg.norm(shared_dirs, axis=1, keepdims=True)
    bottom_dirs = bottom_dirs / np.linalg.norm(bottom_dirs, axis=1, keepdims=True)

    # 5d: PCA baseline comparison (cosine similarity)
    pca_cosines = []
    for i in range(min(max_dims, len(pca_dirs), len(shared_dirs))):
        cos = abs(float(np.dot(shared_dirs[i], pca_dirs[i])))
        pca_cosines.append(min(cos, 1.0))  # clamp for numerical safety

    return {
        "shared_dirs": shared_dirs,  # [max_dims, 128]
        "shared_eigenvalues": shared_eigenvalues.tolist(),
        "bottom_dirs": bottom_dirs,  # [max_dims, 128] — for D' condition
        "bottom_eigenvalues": bottom_eigenvalues.tolist(),
        "pca_dirs": pca_dirs,
        "pca_cosines": pca_cosines,
        "dim_scores": dim_scores.tolist(),
        "W_bar": W_bar.tolist(),
        "V_bar": V_bar.tolist(),
    }


def validate_on_natural_set(shared_dirs: np.ndarray, aligned_acts_strat: dict,
                              act_dir: str, models: list[dict],
                              layer: int, pos: int,
                              ref_model_name: str,
                              results_dir: str) -> dict:
    """Step 5e: Validate shared directions on natural distribution."""
    # We need to load natural-set activations and align them using existing R matrices
    names = list(aligned_acts_strat.keys())
    K = len(names)

    stacked_nat = []
    for name in names:
        nat_path = os.path.join(act_dir, name, f"layer{layer}_natural.npy")
        X_nat = np.load(nat_path)[:, pos, :]

        if name == ref_model_name:
            X_nat_c = X_nat - X_nat.mean(axis=0)
            stacked_nat.append(X_nat_c)
        else:
            R_path = os.path.join(results_dir,
                                   f"R_{name}_layer{layer}_{list(aligned_acts_strat.keys())[0].split('_')[-1]}.npy")
            # Try to find the R matrix
            R_candidates = [f for f in os.listdir(results_dir)
                            if f.startswith(f"R_{name}_") and f.endswith(".npy")]
            if R_candidates:
                R = np.load(os.path.join(results_dir, R_candidates[0]))
                X_nat_c = (X_nat - X_nat.mean(axis=0)) @ R
                stacked_nat.append(X_nat_c)
            else:
                # Re-align using stratified set alignment
                ref_strat = aligned_acts_strat[ref_model_name]
                X_strat = np.load(os.path.join(act_dir, name,
                                                f"layer{layer}_stratified.npy"))[:, pos, :]
                R, _ = procrustes_align(ref_strat, X_strat)
                X_nat_c = (X_nat - X_nat.mean(axis=0)) @ R
                stacked_nat.append(X_nat_c)

    stacked_nat = np.stack(stacked_nat)  # [K, 2000, 128]

    # Compute shared variance on natural set using stratified directions
    h_bar_nat = stacked_nat.mean(axis=0)  # [2000, 128]
    h_bar_nat_c = h_bar_nat - h_bar_nat.mean(axis=0)

    # Variance explained by top-k shared directions on natural set
    total_var_nat = np.trace(h_bar_nat_c.T @ h_bar_nat_c)

    explained_ratios = []
    for k in [1, 3, 5, 10]:
        dirs_k = shared_dirs[:k]
        proj = h_bar_nat_c @ dirs_k.T  # [2000, k]
        var_explained = np.sum(proj ** 2)
        ratio = var_explained / total_var_nat if total_var_nat > 0 else 0
        explained_ratios.append({"k": k, "ratio": float(ratio)})

    return {
        "total_variance_natural": float(total_var_nat),
        "explained_ratios": explained_ratios,
    }


def check_reference_invariance(act_dir, models, site, results_dir):
    """Step 5f: Check if shared directions are stable under different reference."""
    # Find second baseline
    baselines = [m for m in models if m["frozen_component"] is None]
    if len(baselines) < 2:
        return {"stable": None, "reason": "Not enough baselines"}

    ref2 = baselines[1]
    layer = site["layer"]
    pos = site["position_idx"]

    # Load ref2 activations
    X_ref2 = np.load(os.path.join(
        act_dir, ref2["model_name"],
        f"layer{layer}_stratified.npy"
    ))[:, pos, :]

    # Re-align all models to ref2
    aligned2 = {ref2["model_name"]: X_ref2 - X_ref2.mean(axis=0)}
    for m in models:
        if m["model_name"] == ref2["model_name"]:
            continue
        X_k = np.load(os.path.join(
            act_dir, m["model_name"],
            f"layer{layer}_stratified.npy"
        ))[:, pos, :]
        R, _ = procrustes_align(X_ref2, X_k)
        aligned2[m["model_name"]] = (X_k - X_k.mean(axis=0)) @ R

    # Extract shared subspace under ref2
    sub2 = extract_shared_subspace(aligned2)

    # Cross-reference cosine: are shared dirs stable under different reference?
    # Need the primary shared dirs from the caller — approximate by re-extracting
    sub1 = extract_shared_subspace(
        {name: np.load(os.path.join(act_dir, name, f"layer{layer}_stratified.npy"))[:, pos, :]
         - np.load(os.path.join(act_dir, name, f"layer{layer}_stratified.npy"))[:, pos, :].mean(axis=0)
         for name in [m["model_name"] for m in models[:5]]})  # quick approx

    cross_cosines = []
    for i in range(min(5, len(sub1["shared_dirs"]), len(sub2["shared_dirs"]))):
        v1 = sub1["shared_dirs"][i]
        v2 = sub2["shared_dirs"][i]
        cos = abs(float(np.dot(v1, v2)))
        cross_cosines.append(min(cos, 1.0))

    return {
        "reference2": ref2["model_name"],
        "shared_eigenvalues_ref2": sub2["shared_eigenvalues"],
        "pca_cosines_ref2": sub2["pca_cosines"],
        "cross_reference_cosines": cross_cosines,
    }


# ============================================================
# Step 6: Interpretation
# ============================================================

def run_probing(shared_dirs: np.ndarray, aligned_acts: dict,
                eval_metadata: list[dict], cfg: ModelConfig) -> dict:
    """Step 6: Probe shared directions for task variable correlations."""
    names = list(aligned_acts.keys())
    stacked = np.stack([aligned_acts[n] for n in names])  # [K, 2000, 128]
    h_bar = stacked.mean(axis=0)  # [2000, 128]

    # Project onto shared directions
    projections = h_bar @ shared_dirs.T  # [2000, n_dirs]

    # Extract task variables
    n = len(eval_metadata)
    n_carries_arr = np.array([m["n_carries"] for m in eval_metadata])
    carries_arr = np.array([m["carries"] for m in eval_metadata])  # [2000, 5]

    # Digit values
    a_digits = np.array([
        [(m["a"] // 10**i) % 10 for i in range(cfg.n_digits)]
        for m in eval_metadata
    ])  # [2000, 5] LSB first
    b_digits = np.array([
        [(m["b"] // 10**i) % 10 for i in range(cfg.n_digits)]
        for m in eval_metadata
    ])
    r_digits = np.array([
        [(m["result"] // 10**i) % 10 for i in range(cfg.n_result_digits)]
        for m in eval_metadata
    ])

    results = {}
    n_dirs = min(5, shared_dirs.shape[0])

    for d_idx in range(n_dirs):
        proj = projections[:, d_idx]  # [2000]
        dir_results = {}

        # Correlations with task variables
        # 1. Number of carries
        corr = float(np.corrcoef(proj, n_carries_arr)[0, 1])
        mi = mutual_information_1d(proj, n_carries_arr)
        dir_results["n_carries"] = {"correlation": corr, "mutual_info": mi}

        # 2. Individual carry bits
        for col in range(cfg.n_digits):
            corr = float(np.corrcoef(proj, carries_arr[:, col])[0, 1])
            mi = mutual_information_1d(proj, carries_arr[:, col])
            dir_results[f"carry_col{col}"] = {"correlation": corr, "mutual_info": mi}

        # 3. Sum magnitude
        sums = np.array([m["result"] for m in eval_metadata])
        corr = float(np.corrcoef(proj, sums)[0, 1])
        mi = mutual_information_1d(proj, sums)
        dir_results["sum_magnitude"] = {"correlation": corr, "mutual_info": mi}

        # 4. Individual operand digits
        for col in range(cfg.n_digits):
            corr_a = float(np.corrcoef(proj, a_digits[:, col])[0, 1])
            corr_b = float(np.corrcoef(proj, b_digits[:, col])[0, 1])
            dir_results[f"a_digit_col{col}"] = {"correlation": corr_a}
            dir_results[f"b_digit_col{col}"] = {"correlation": corr_b}

        # 5. Result digits
        for col in range(cfg.n_result_digits):
            corr_r = float(np.corrcoef(proj, r_digits[:, col])[0, 1])
            dir_results[f"result_digit_col{col}"] = {"correlation": corr_r}

        # 6. Linear probe for carry (logistic regression on top-1 projection)
        from sklearn.linear_model import LogisticRegression
        for col in range(cfg.n_digits):
            y = carries_arr[:, col]
            if len(np.unique(y)) < 2:
                continue
            clf = LogisticRegression(max_iter=1000)
            clf.fit(proj.reshape(-1, 1), y)
            probe_acc = clf.score(proj.reshape(-1, 1), y)
            dir_results[f"carry_probe_col{col}"] = {"probe_accuracy": float(probe_acc)}

        results[f"direction_{d_idx}"] = dir_results

    return results


def mutual_information_1d(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Estimate mutual information between continuous x and (possibly discrete) y."""
    # Discretize x into bins
    x_binned = np.digitize(x, np.linspace(x.min(), x.max(), n_bins + 1)[1:-1])

    # If y is continuous, discretize it too
    y_unique = np.unique(y)
    if len(y_unique) > 20:
        y_binned = np.digitize(y, np.linspace(y.min(), y.max(), n_bins + 1)[1:-1])
    else:
        y_binned = y.astype(int)

    # Compute MI from contingency table
    from sklearn.metrics import mutual_info_score
    return float(mutual_info_score(x_binned, y_binned))


# ============================================================
# Step 7: Ablation
# ============================================================

def run_ablation(shared_dirs: np.ndarray, models: list[dict],
                  act_dir: str, eval_tokens: torch.Tensor,
                  eval_metadata: list[dict],
                  selected_site: dict,
                  results_dir: str, cfg: ModelConfig,
                  device: str = "cuda") -> dict:
    """Step 7: Mean-ablation of shared directions with variance-matched controls."""
    os.makedirs(results_dir, exist_ok=True)
    layer = selected_site["layer"]
    pos = selected_site["position_idx"]

    results = {"models": {}, "summary": {}}

    for m_info in models:
        name = m_info["model_name"]
        print(f"    Ablating {name}...")

        # Load model
        model = ArithmeticTransformer(cfg).to(device)
        state_dict = torch.load(m_info["model_path"], map_location=device,
                                 weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Load rotation matrix to convert shared dirs to native basis
        R_files = [f for f in os.listdir(results_dir)
                    if f.startswith(f"R_{name}_") and f.endswith(".npy")]
        if R_files:
            R = np.load(os.path.join(results_dir, R_files[0]))
        else:
            R = np.eye(cfg.d_model)  # reference model

        # Convert shared dirs to native basis: v_native = R^T @ v_shared
        # But our convention is X_k @ R ≈ X_ref, so v_native = R @ v_shared^T ... hmm
        # Actually: shared_dirs are in reference frame. To get native frame:
        # If X_k @ R = X_ref (approx), then direction v in ref frame corresponds to
        # R @ v in the model-k frame? No...
        # X_aligned = X_k @ R, so a direction v in aligned space corresponds to
        # projecting X_aligned onto v = (X_k @ R) @ v = X_k @ (R @ v)
        # So the native-basis direction is R @ v
        native_dirs = (R @ shared_dirs.T).T  # [n_dirs, 128]
        native_dirs_t = torch.tensor(native_dirs, dtype=torch.float32, device=device)

        # Baseline accuracy
        baseline_acc = evaluate_with_hook(model, eval_tokens, cfg, device,
                                           hook_layer=None, hook_fn=None)

        # Compute mean projections for mean-ablation
        # Activations are on CPU (to save GPU memory); move dirs to CPU for this computation
        acts = get_model_activations(model, eval_tokens, layer, device)  # [N, seq, 128] CPU
        acts_at_pos = acts[:, pos, :]  # [N, 128] CPU

        model_results = {"baseline_acc": baseline_acc}

        for n_dirs in [1, 3, 5, 10]:
            dirs_subset = native_dirs_t[:n_dirs]  # [n_dirs, 128] CUDA
            dirs_cpu = dirs_subset.cpu()

            # Mean projection along shared directions (computed on CPU)
            projs = acts_at_pos @ dirs_cpu.T  # [N, n_dirs]
            mean_projs = projs.mean(dim=0).to(device)  # [n_dirs] -> CUDA for hook

            # Mean-ablation hook using proper subspace projection.
            # Directions may not be orthogonal (they're C_total-orthogonal from
            # the generalized eigenproblem), so we use the full projection matrix.
            def make_ablation_hook(directions, mean_proj, target_pos):
                D = directions.T  # [d, n_dirs]
                DtD_inv = torch.linalg.inv(D.T @ D)  # [n_dirs, n_dirs]
                P = D @ DtD_inv @ D.T  # [d, d] subspace projector
                mean_component = D @ DtD_inv @ mean_proj  # [d]
                def hook_fn(module, input, output):
                    x = output.clone()
                    h = x[:, target_pos, :]  # [B, d]
                    h_proj = h @ P  # [B, d] — component in subspace
                    x[:, target_pos, :] = h - h_proj + mean_component.unsqueeze(0)
                    return x
                return hook_fn

            hook_fn = make_ablation_hook(dirs_subset, mean_projs, pos)
            abl_acc = evaluate_with_hook(model, eval_tokens, cfg, device,
                                          hook_layer=layer, hook_fn=hook_fn)

            # Variance-matched random control
            # Compute activation variance along shared directions
            shared_var = projs.var(dim=0).mean().item()

            random_accs = []
            rng = np.random.RandomState(42)
            for trial in range(10):
                # Generate random directions
                rand_dirs = rng.randn(n_dirs, cfg.d_model).astype(np.float32)
                rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
                rand_dirs_t = torch.tensor(rand_dirs, device=device)

                # Compute mean projection on CPU, then move to GPU for hook
                rand_projs = acts_at_pos @ torch.tensor(rand_dirs, device="cpu").T
                rand_mean = rand_projs.mean(dim=0).to(device)

                hook_fn_rand = make_ablation_hook(rand_dirs_t, rand_mean, pos)
                rand_acc = evaluate_with_hook(model, eval_tokens, cfg, device,
                                               hook_layer=layer, hook_fn=hook_fn_rand)
                random_accs.append(rand_acc)

            model_results[f"ablate_top{n_dirs}"] = {
                "accuracy": abl_acc,
                "accuracy_drop": baseline_acc - abl_acc,
                "random_mean_acc": float(np.mean(random_accs)),
                "random_std_acc": float(np.std(random_accs)),
                "random_mean_drop": baseline_acc - float(np.mean(random_accs)),
                "shared_activation_var": shared_var,
            }

        results["models"][name] = model_results

        del model
        torch.cuda.empty_cache()

    # Summary statistics
    for n_dirs in [1, 3, 5, 10]:
        key = f"ablate_top{n_dirs}"
        shared_drops = [r[key]["accuracy_drop"] for r in results["models"].values()
                         if key in r]
        random_drops = [r[key]["random_mean_drop"] for r in results["models"].values()
                         if key in r]
        if shared_drops and random_drops:
            from scipy import stats
            diffs = np.array(shared_drops) - np.array(random_drops)
            if np.all(diffs == 0):
                stat, pval = 0.0, 1.0  # no difference at all
            else:
                try:
                    stat, pval = stats.wilcoxon(shared_drops, random_drops)
                except ValueError:
                    stat, pval = 0.0, 1.0
            results["summary"][key] = {
                "mean_shared_drop": float(np.mean(shared_drops)),
                "mean_random_drop": float(np.mean(random_drops)),
                "wilcoxon_stat": float(stat),
                "wilcoxon_pval": float(pval),
                "significant": pval < 0.01,
            }

    with open(os.path.join(results_dir, "step7_ablation.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def get_model_activations(model, eval_tokens, layer, device, batch_size=256):
    """Get hidden states at a specific layer for all eval inputs."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], batch_size):
            batch = eval_tokens[i:i+batch_size].to(device)
            _, hiddens = model(batch, return_all_hiddens=True)
            chunks.append(hiddens[layer].cpu())
    return torch.cat(chunks, dim=0)


def evaluate_with_hook(model, eval_tokens, cfg, device,
                        hook_layer=None, hook_fn=None, batch_size=256):
    """Evaluate model accuracy, optionally with an ablation hook."""
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
            pred = logits[:, 11:17, :].argmax(dim=-1)
            true = batch[:, 12:18]
            correct += (pred == true).all(dim=-1).sum().item()
            total += batch.shape[0]

    if handle is not None:
        handle.remove()

    return correct / total


# ============================================================
# Main pipeline
# ============================================================

def main():
    cfg = ModelConfig()
    act_dir = "activations"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load converged models
    models = get_converged_models("models")
    print(f"Found {len(models)} converged models")
    if len(models) < 6:
        print("DECISION GATE FAIL: fewer than 6 converged models")
        return

    # Load eval metadata
    eval_data = load_eval_set(os.path.join("eval_sets", "stratified_2000"))
    eval_metadata = eval_data["metadata"]

    # ---- Step 3: CKA ----
    print("\n" + "=" * 60)
    print("STEP 3: CKA Analysis")
    print("=" * 60)
    cka_results = run_cka_analysis(act_dir, models, cfg, results_dir)

    # Decision gate
    selected_sites = select_top_sites(cka_results, cfg)
    if not selected_sites:
        print("\nDECISION GATE FAIL: No sites meet CKA criteria")
        # Try with relaxed criteria
        print("Trying relaxed criteria (CKA > 0.3, lower > 0.2)...")
        relaxed = []
        for key, data in cka_results.items():
            if isinstance(data, dict) and "cka_mean" in data:
                if data["cka_mean"] > 0.3 and data["cka_ci_lower"] > 0.2:
                    relaxed.append({
                        "site": key, "layer": data["layer"],
                        "position_idx": data["position_idx"],
                        "position_name": data["position"],
                        "cka_mean": data["cka_mean"],
                        "cka_no_pc": data["cka_no_top_pc_mean"],
                    })
        if relaxed:
            relaxed.sort(key=lambda x: x["cka_mean"], reverse=True)
            selected_sites = relaxed[:3]
            print(f"  Using {len(selected_sites)} sites with relaxed criteria")
        else:
            print("  No sites even with relaxed criteria. Stopping.")
            return

    with open(os.path.join(results_dir, "selected_sites.json"), "w") as f:
        json.dump(selected_sites, f, indent=2)

    # ---- Step 4: Alignment ----
    print("\n" + "=" * 60)
    print("STEP 4: Procrustes Alignment")
    print("=" * 60)
    alignment_results, aligned_acts = run_alignment(
        act_dir, models, selected_sites, results_dir)

    # ---- Step 5: Shared Subspace ----
    print("\n" + "=" * 60)
    print("STEP 5: Shared Subspace Extraction")
    print("=" * 60)

    # Use the top site
    top_site = selected_sites[0]
    site_key = top_site["site"]
    print(f"  Primary extraction site: {site_key}")

    subspace = extract_shared_subspace(aligned_acts[site_key])

    print(f"\n  Shared eigenvalues (top 10): "
          f"{[f'{v:.4f}' for v in subspace['shared_eigenvalues'][:10]]}")
    print(f"  PCA cosines (shared vs PCA): "
          f"{[f'{v:.4f}' for v in subspace['pca_cosines'][:5]]}")

    # 5e: Natural distribution validation
    print("\n  Validating on natural distribution...")
    ref_model = [m for m in models if m["frozen_component"] is None and m["seed"] == 0]
    ref_name = ref_model[0]["model_name"] if ref_model else models[0]["model_name"]
    nat_validation = validate_on_natural_set(
        subspace["shared_dirs"], aligned_acts[site_key],
        act_dir, models, top_site["layer"], top_site["position_idx"],
        ref_name, results_dir)
    print(f"  Natural set explained ratios: {nat_validation['explained_ratios']}")

    # 5f: Reference invariance
    print("\n  Checking reference invariance...")
    ref_inv = check_reference_invariance(act_dir, models, top_site, results_dir)

    # Save Step 5 results
    step5_results = {
        "site": site_key,
        "shared_eigenvalues": subspace["shared_eigenvalues"],
        "pca_cosines": subspace["pca_cosines"],
        "dim_scores_top10": sorted(enumerate(subspace["dim_scores"]),
                                     key=lambda x: x[1], reverse=True)[:10],
        "natural_validation": nat_validation,
        "reference_invariance": ref_inv,
    }
    with open(os.path.join(results_dir, "step5_subspace.json"), "w") as f:
        json.dump(step5_results, f, indent=2)

    # Save shared directions for use in Steps 7-8
    np.save(os.path.join(results_dir, "shared_dirs.npy"), subspace["shared_dirs"])
    np.save(os.path.join(results_dir, "pca_dirs.npy"), subspace["pca_dirs"])
    np.save(os.path.join(results_dir, "bottom_dirs.npy"), subspace["bottom_dirs"])

    # ---- Step 6: Interpretation ----
    print("\n" + "=" * 60)
    print("STEP 6: Interpretation")
    print("=" * 60)
    probe_results = run_probing(subspace["shared_dirs"], aligned_acts[site_key],
                                 eval_metadata, cfg)

    # Print top correlations
    for dir_name, dir_data in probe_results.items():
        print(f"\n  {dir_name}:")
        sorted_vars = sorted(dir_data.items(),
                              key=lambda x: abs(x[1].get("correlation", 0)
                                                 if isinstance(x[1], dict) else 0),
                              reverse=True)
        for var_name, var_data in sorted_vars[:5]:
            if isinstance(var_data, dict):
                corr = var_data.get("correlation", "N/A")
                mi = var_data.get("mutual_info", "N/A")
                probe = var_data.get("probe_accuracy", None)
                line = f"    {var_name}: corr={corr}"
                if mi != "N/A":
                    line += f", MI={mi:.4f}"
                if probe:
                    line += f", probe_acc={probe:.4f}"
                print(line)

    with open(os.path.join(results_dir, "step6_probing.json"), "w") as f:
        json.dump(probe_results, f, indent=2, default=str)

    # ---- Step 7: Ablation ----
    print("\n" + "=" * 60)
    print("STEP 7: Ablation")
    print("=" * 60)
    eval_convergence = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    ablation_results = run_ablation(
        subspace["shared_dirs"], models, act_dir,
        eval_convergence["tokens"], eval_convergence["metadata"],
        top_site, results_dir, cfg)

    # Print summary
    print("\n  Ablation summary:")
    for key, summary in ablation_results["summary"].items():
        print(f"    {key}: shared_drop={summary['mean_shared_drop']:.4f}, "
              f"random_drop={summary['mean_random_drop']:.4f}, "
              f"p={summary['wilcoxon_pval']:.6f} "
              f"{'***' if summary['significant'] else 'ns'}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (Steps 3-7)")
    print("=" * 60)


if __name__ == "__main__":
    main()
