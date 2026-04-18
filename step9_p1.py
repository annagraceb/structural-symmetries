"""Step 9 / P1: Inverted eigenproblem ablation experiment.

Pre-registration: P1_PREREGISTRATION.md (locked before this script runs).

Tests whether bottom-k eigenvectors of the generalized eigenproblem
(anti-shared / privileged directions) are causally more important than
norm-matched random directions — and more important than top-k shared
directions.

Outputs results/p1_results.json. Verdict computed separately by p1_report.py.

Usage:
    python step9_p1.py                 # full run: 4 sites, k in {5,10,20,30,50}
    python step9_p1.py --smoke         # quick smoke test: 1 site, k=10
    python step9_p1.py --site-only layer1_result_0   # single site
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import linalg

from analysis import (
    get_model_activations,
    evaluate_with_hook,
    procrustes_align,
)
from collect_activations import get_converged_models
from config import ModelConfig
from data import load_eval_set
from model import ArithmeticTransformer

ACT_DIR = "activations"
RESULTS_DIR = "results"
P1_DIR = os.path.join(RESULTS_DIR, "p1")

# Locked in pre-registration
# Amendment A3 (2026-04-16, before any clean run): K_VALUES capped at K_PCA/2
# so top-k and bottom-k eigenvectors are disjoint within the PCA-restricted
# subspace. With K_PCA=32 the maximum non-overlapping k is 16. Original spec
# was [5, 10, 20, 30, 50] but k>16 is geometrically meaningless after A1.
K_VALUES = [4, 8, 12, 16]
N_RANDOM_TRIALS = 100
EPS_SWEEP = [1e-5, 1e-4, 1e-3]
PRIMARY_SITES = ["layer1_result_0", "layer2_equals", "layer2_result_0"]
CONTROL_SITE = "layer3_plus"
SITE_LAYER_POS = {
    "layer1_result_0": (1, 12),
    "layer2_equals": (2, 11),
    "layer2_result_0": (2, 12),
    "layer3_plus": (3, 5),
}
# Amendment A1: restrict ratio eigenproblem to top-K_pca subspace of C_total
# to avoid dead-axis solutions. Locked at d/2 for d=64.
K_PCA = 32


# ---------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------

def extract_with_max_dims(aligned_acts: dict, max_dims: int, eps_scale: float = 1e-8,
                           k_pca: int = K_PCA):
    """Solve C_shared v = λ C_total v restricted to top-k_pca PCA subspace of
    C_total (A1), and return C_total-orthonormal (non-unit-norm) eigenvectors
    so that v^T C_total v = 1 for every returned direction (A2).

    Returns dict with:
      shared_dirs [max_dims, d]  — top eigvecs, v^T C_total v = 1
      bottom_dirs [max_dims, d]  — bottom eigvecs, v^T C_total v = 1
      C_total                    — the per-model-variance cov in full space (for whitening randoms)
      C_total_plus_ridge         — C_total + ridge*I (use for whitening to match eigproblem reg)
      eigenvalues, denom_condition_number, ridge, d, k_pca, pca_eigvals_kept_fraction
    """
    names = list(aligned_acts.keys())
    K = len(names)
    stacked = np.stack([aligned_acts[n] for n in names])
    n_samples, d = stacked.shape[1], stacked.shape[2]

    h_bar = stacked.mean(axis=0)
    h_bar_c = h_bar - h_bar.mean(axis=0)
    C_shared = (h_bar_c.T @ h_bar_c) / n_samples

    C_total = np.zeros((d, d))
    for k in range(K):
        h_k = stacked[k] - stacked[k].mean(axis=0)
        C_total += (h_k.T @ h_k) / n_samples
    C_total /= K

    # A1: restrict to top-k_pca PCA subspace of C_total
    k_pca_eff = min(k_pca, d)
    pca_eigvals, pca_eigvecs = np.linalg.eigh(C_total)
    pca_order = np.argsort(pca_eigvals)[::-1]
    pca_eigvals = pca_eigvals[pca_order]
    pca_eigvecs = pca_eigvecs[:, pca_order]
    U_k = pca_eigvecs[:, :k_pca_eff]
    C_shared_red = U_k.T @ C_shared @ U_k
    C_total_red = U_k.T @ C_total @ U_k

    tr_scale = np.trace(C_total_red) / k_pca_eff
    ridge = eps_scale * tr_scale
    eigvals, eigvecs_red = linalg.eigh(
        C_shared_red, C_total_red + ridge * np.eye(k_pca_eff))

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs_red = eigvecs_red[:, idx]

    # Pull back to full d-space. DO NOT renormalize (A2): preserve C_total-
    # orthonormality so every direction has v^T C_total v = 1.
    eigvecs_full = U_k @ eigvecs_red

    n = min(max_dims, k_pca_eff)
    # A3: ensure top-n and bottom-n eigenvectors are disjoint sets within the
    # k_pca_eff-dimensional reduced subspace. If 2n > k_pca_eff they would
    # overlap and shared/bottom would alias to the same directions. Truncate.
    if 2 * n > k_pca_eff:
        n = k_pca_eff // 2
    shared = eigvecs_full[:, :n].T   # [n, d], C_total-orthonormal
    bottom = eigvecs_full[:, -n:].T  # [n, d], C_total-orthonormal

    C_total_plus_ridge = C_total + ridge * np.eye(d)
    denom_eigvals = np.linalg.eigvalsh(C_total_red + ridge * np.eye(k_pca_eff))
    cond = float(denom_eigvals.max() / denom_eigvals.min())

    return {
        "shared_dirs": shared,
        "bottom_dirs": bottom,
        "eigenvalues": eigvals.tolist(),
        "denom_condition_number": cond,
        "ridge": float(ridge),
        "d": d,
        "k_pca": k_pca_eff,
        "pca_eigvals_kept_fraction": float(pca_eigvals[:k_pca_eff].sum() / pca_eigvals.sum()),
        "C_total": C_total,
        "C_total_plus_ridge": C_total_plus_ridge,
    }


def complement_top_k(shared_dirs_k: np.ndarray, C_total: np.ndarray, k: int,
                      C_total_plus_ridge: np.ndarray = None) -> np.ndarray:
    """Path 1: top-k eigenvectors of P_perp C_total P_perp, where P_perp projects
    orthogonal (Euclidean) to span(shared_dirs_k).

    Returns [k, d] directions that are orthogonal to shared and have maximum
    per-model variance within that complement. Normalized so v^T C_total v = 1
    (A2-compatible for variance-tracking).
    """
    d = C_total.shape[0]
    S = shared_dirs_k.T                              # [d, k_s]
    StS_inv = np.linalg.pinv(S.T @ S)
    P_shared = S @ StS_inv @ S.T                     # Euclidean projector onto shared
    P_perp = np.eye(d) - P_shared
    M = P_perp @ C_total @ P_perp
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    top = vecs[:, order[:k]].T                       # [k, d]
    # A2 normalization: v^T C_total v = 1
    vcv = np.array([top[i] @ C_total @ top[i] for i in range(k)])
    vcv = np.clip(vcv, 1e-20, None)
    return top / np.sqrt(vcv)[:, None]


def projection_trace_variance(V: np.ndarray, C: np.ndarray) -> float:
    """trace(P C P) where P is the orthogonal (Euclidean) projector onto span(V).
    This is the ACTUAL per-model activation variance removed by projection-
    ablating the subspace span(V). Basis-independent.
    """
    VVt_inv = np.linalg.pinv(V @ V.T)
    return float(np.trace(VVt_inv @ V @ C @ V.T))


def subspace_principal_angles_cos(V1: np.ndarray, V2: np.ndarray) -> list:
    """Principal angles (cosines) between span(V1) and span(V2). Uses SVD of
    Q1^T Q2 where Q_i is an orthonormal basis of span(V_i)."""
    if V1.shape[0] == 0 or V2.shape[0] == 0:
        return []
    Q1, _ = np.linalg.qr(V1.T)
    Q2, _ = np.linalg.qr(V2.T)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return [float(x) for x in np.clip(s, 0.0, 1.0)]


def whitened_random_subspace(k, d, C_total_plus_ridge, rng):
    """Sample k random directions whitened so v^T C_total_plus_ridge v = 1.

    Uses the regularized covariance (same ridge as the eigenproblem) for
    numerical stability. Returns [k, d] array.
    """
    # Eigendecomp once; reuse
    vals, vecs = np.linalg.eigh(C_total_plus_ridge)
    vals = np.clip(vals, 1e-12, None)
    C_inv_half = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T  # [d, d]

    rows = []
    for _ in range(k):
        r = rng.randn(d).astype(np.float64)
        w = C_inv_half @ r
        # Normalize so w^T C_total w = 1  (uses unregularized C for the metric)
        # Simpler: normalize under the metric we matched — use C_plus_ridge
        denom = np.sqrt(max(w @ C_total_plus_ridge @ w, 1e-20))
        rows.append(w / denom)
    return np.stack(rows)


def orthogonalize_against(target: np.ndarray, basis: np.ndarray,
                           C_total: np.ndarray = None,
                           min_residual_norm: float = 1e-6):
    """Gram-Schmidt: remove the `basis` subspace from each row of `target`.

    target: [n, d]   basis: [m, d]
    Returns (orth_dirs, kept_idx):
      orth_dirs: [n_kept, d]
      kept_idx:  list of original indices retained

    If C_total is provided (A2), each kept row is rescaled so
    v^T C_total v = 1 (variance-matched). Otherwise unit Euclidean norm.

    Rows whose residual norm falls below min_residual_norm (fully absorbed by
    the basis subspace) are dropped.
    """
    B = basis.T
    BtB_inv = np.linalg.pinv(B.T @ B)
    P = B @ BtB_inv @ B.T
    orth = target - target @ P
    eucl_norms = np.linalg.norm(orth, axis=1)
    kept = np.where(eucl_norms > min_residual_norm)[0]
    if len(kept) == 0:
        return np.zeros((0, target.shape[1])), []
    orth_kept = orth[kept]
    if C_total is not None:
        # A2: rescale so v^T C_total v = 1
        vcv = np.array([orth_kept[i] @ C_total @ orth_kept[i] for i in range(len(kept))])
        vcv = np.clip(vcv, 1e-20, None)
        orth_kept = orth_kept / np.sqrt(vcv)[:, None]
    else:
        orth_kept = orth_kept / eucl_norms[kept, None]
    return orth_kept, kept.tolist()


# ---------------------------------------------------------------
# Alignment helper for control site
# ---------------------------------------------------------------

def align_site_on_the_fly(models, site_name, ref_name):
    """Procrustes-align all models at site_name to ref_name. Returns {name: X_aligned}."""
    layer, pos = SITE_LAYER_POS[site_name]
    X_ref = np.load(os.path.join(ACT_DIR, ref_name, f"layer{layer}_stratified.npy"))[:, pos, :]
    aligned = {ref_name: X_ref - X_ref.mean(axis=0)}
    for m in models:
        if m["model_name"] == ref_name:
            continue
        X_k = np.load(os.path.join(ACT_DIR, m["model_name"], f"layer{layer}_stratified.npy"))[:, pos, :]
        R, _ = procrustes_align(X_ref, X_k)
        aligned[m["model_name"]] = (X_k - X_k.mean(axis=0)) @ R
        # Save R for ablation code path
        R_path = os.path.join(RESULTS_DIR, f"R_{m['model_name']}_{site_name}.npy")
        if not os.path.exists(R_path):
            np.save(R_path, R)
    return aligned


def load_aligned_acts(site_name):
    """Load pre-computed aligned activations for primary sites."""
    d = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(d, f))
    return out


# ---------------------------------------------------------------
# Ablation (parametric extension of analysis.py::run_ablation)
# ---------------------------------------------------------------

def make_ablation_hook(directions, mean_proj, target_pos):
    """Same as analysis.py::make_ablation_hook but importable here."""
    D = directions.T                        # [d, n_dirs]
    DtD_inv = torch.linalg.inv(D.T @ D)
    P = D @ DtD_inv @ D.T                   # [d, d]
    mean_component = D @ DtD_inv @ mean_proj

    def hook_fn(module, input, output):
        x = output.clone()
        h = x[:, target_pos, :]
        h_proj = h @ P
        x[:, target_pos, :] = h - h_proj + mean_component.unsqueeze(0)
        return x
    return hook_fn


def run_ablation_for_direction_set(
    model, eval_tokens, cfg, device,
    layer, pos, baseline_acc,
    native_dirs_t,            # [K, d] unit-norm, in native basis
    acts_at_pos_cpu,          # [N, d] cpu tensor for computing mean projections
    k_values,
):
    """Ablate top-k_i of the given direction set, for each k_i in k_values."""
    out = {}
    for k in k_values:
        if k > native_dirs_t.shape[0]:
            continue
        dirs_k = native_dirs_t[:k]
        dirs_cpu = dirs_k.cpu()
        projs = acts_at_pos_cpu @ dirs_cpu.T                   # [N, k]
        mean_projs = projs.mean(dim=0).to(device)              # [k]

        hook = make_ablation_hook(dirs_k, mean_projs, pos)
        abl_acc = evaluate_with_hook(model, eval_tokens, cfg, device,
                                      hook_layer=layer, hook_fn=hook)
        out[k] = {
            "accuracy": float(abl_acc),
            "drop": float(baseline_acc - abl_acc),
            "activation_var_along_dirs": float(projs.var(dim=0).mean().item()),
        }
    return out


def run_ablation_per_k_map(
    model, eval_tokens, cfg, device,
    layer, pos, baseline_acc,
    dirs_per_k,               # dict k -> [k', d] np.ndarray in REFERENCE basis
    model_name, site_name, d,
    acts_at_pos_cpu,
):
    """Ablate variants where the direction set changes per k (e.g. per-k ortho variant)."""
    out = {}
    for k, ref_dirs in dirs_per_k.items():
        if ref_dirs.shape[0] == 0:
            out[k] = {"accuracy": None, "drop": None, "activation_var_along_dirs": None,
                      "n_effective_dirs": 0}
            continue
        native = native_basis_dirs(ref_dirs, model_name, site_name, d)
        dirs_t = torch.tensor(native, dtype=torch.float32, device=device)
        dirs_cpu = dirs_t.cpu()
        projs = acts_at_pos_cpu @ dirs_cpu.T
        mean_projs = projs.mean(dim=0).to(device)

        hook = make_ablation_hook(dirs_t, mean_projs, pos)
        abl_acc = evaluate_with_hook(model, eval_tokens, cfg, device,
                                      hook_layer=layer, hook_fn=hook)
        out[k] = {
            "accuracy": float(abl_acc),
            "drop": float(baseline_acc - abl_acc),
            "activation_var_along_dirs": float(projs.var(dim=0).mean().item()),
            "n_effective_dirs": int(ref_dirs.shape[0]),
        }
    return out


def run_random_baseline(
    model, eval_tokens, cfg, device,
    layer, pos, baseline_acc,
    acts_at_pos_cpu, d, k_values, n_trials, rng,
    C_total_plus_ridge,
):
    """Run n_trials whitened-random subspaces per k (A2): each sampled direction
    has v^T C_total_plus_ridge v = 1, so trace(V^T C_total V) = k matches
    shared and anti-shared subspaces.
    """
    out = {k: [] for k in k_values}
    # Precompute C_inv_half once per extraction (passed via closure via rng seed)
    for k in k_values:
        for _ in range(n_trials):
            rand = whitened_random_subspace(k, d, C_total_plus_ridge, rng).astype(np.float32)
            rand_t = torch.tensor(rand, device=device)
            projs = acts_at_pos_cpu @ torch.tensor(rand, device="cpu").T
            mean_projs = projs.mean(dim=0).to(device)
            hook = make_ablation_hook(rand_t, mean_projs, pos)
            acc = evaluate_with_hook(model, eval_tokens, cfg, device,
                                      hook_layer=layer, hook_fn=hook)
            out[k].append(float(baseline_acc - acc))
    return out


def trace_cov_along_dirs(dirs: np.ndarray, C: np.ndarray) -> float:
    """trace(V C V^T) for V = dirs (rows are directions). Reports total variance
    removed by ablating the span of dirs under cov C."""
    return float(np.trace(dirs @ C @ dirs.T))


def native_basis_dirs(ref_dirs, model_name, site_name, d):
    """Convert reference-frame directions to a model's native basis.

    ref_dirs: [n, d]. Returns [n, d] in native basis.
    Uses the R matrix saved during alignment. If no R file (ref model), returns as-is.
    """
    R_path = os.path.join(RESULTS_DIR, f"R_{model_name}_{site_name}.npy")
    if os.path.exists(R_path):
        R = np.load(R_path)
    else:
        # Fall back to the existing loose matching used by run_ablation
        candidates = [f for f in os.listdir(RESULTS_DIR)
                      if f.startswith(f"R_{model_name}_") and f.endswith(".npy")]
        if candidates:
            R = np.load(os.path.join(RESULTS_DIR, candidates[0]))
        else:
            R = np.eye(d)  # reference model
    # analysis.py uses v_native = R @ v_shared^T convention
    return (R @ ref_dirs.T).T


# ---------------------------------------------------------------
# Main per-site driver
# ---------------------------------------------------------------

def run_site(models, site_name, cfg, eval_tokens, device, k_values, n_random_trials,
             max_dims_request, eps_scale):
    """Full P1 ablation at one site. Returns per-model results."""
    layer, pos = SITE_LAYER_POS[site_name]

    # 1. Load aligned activations (or compute for control site)
    aligned_dir = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    if os.path.isdir(aligned_dir):
        aligned_acts = load_aligned_acts(site_name)
        print(f"    Loaded {len(aligned_acts)} aligned activations.")
    else:
        ref = next((m for m in models if m["frozen_component"] is None and m["seed"] == 0),
                   models[0])
        print(f"    Aligning on the fly (ref={ref['model_name']})...")
        aligned_acts = align_site_on_the_fly(models, site_name, ref["model_name"])
        # Persist for reproducibility
        os.makedirs(aligned_dir, exist_ok=True)
        for name, arr in aligned_acts.items():
            np.save(os.path.join(aligned_dir, f"{name}.npy"), arr)

    # 2. Extract top-/bottom-max_dims from generalized eigenproblem
    extraction = extract_with_max_dims(aligned_acts, max_dims_request, eps_scale=eps_scale)
    shared_dirs = extraction["shared_dirs"]          # [max_dims, d] — C_total-orthonormal
    bottom_dirs = extraction["bottom_dirs"]          # [max_dims, d] — C_total-orthonormal
    C_total = extraction["C_total"]
    C_total_plus_ridge = extraction["C_total_plus_ridge"]
    d = extraction["d"]

    # Variance-match verification: each direction should have v^T C_total v ≈ 1
    shared_traces = [float(shared_dirs[i] @ C_total @ shared_dirs[i]) for i in range(min(3, len(shared_dirs)))]
    bottom_traces = [float(bottom_dirs[i] @ C_total @ bottom_dirs[i]) for i in range(min(3, len(bottom_dirs)))]
    print(f"    variance check: shared v^T C_total v first 3 = {shared_traces}")
    print(f"    variance check: bottom v^T C_total v first 3 = {bottom_traces}")

    # Ortho variant: for each k, orthogonalize bottom[:k] against shared[:k].
    # Skip k values where k + k > d (not enough null space to leave n meaningful
    # orthogonal dirs after projection).
    ortho_per_k = {}   # k -> [k', d] where k' <= k
    for k in k_values:
        if k > len(bottom_dirs) or k > len(shared_dirs):
            continue
        if 2 * k > d:
            print(f"    ortho: skipping k={k} — basis + target exceeds d={d}")
            continue
        oh, kept = orthogonalize_against(bottom_dirs[:k], shared_dirs[:k], C_total=C_total)
        ortho_per_k[k] = oh
        if len(kept) < k:
            print(f"    ortho k={k}: kept {len(kept)}/{k} after shared[:{k}] projection")

    # Path 1 — complement top-k: the top-k high-variance directions orthogonal
    # to span(shared[:k]). Different from ortho variant, which rotates bottom
    # dirs into the complement. This directly asks: "what are the max-variance
    # directions the model uses that DON'T agree across models?"
    complement_per_k = {}
    for k in k_values:
        if k > len(shared_dirs) or 2 * k > d:
            continue
        complement_per_k[k] = complement_top_k(shared_dirs[:k], C_total, k)

    # 3. Per-model ablation
    per_model = {}
    rng = np.random.RandomState(1337)  # same randoms across all models for apples-to-apples

    for i, m in enumerate(models):
        name = m["model_name"]
        t0 = time.time()
        print(f"    [{i+1}/{len(models)}] {name}...")

        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()

        baseline_acc = evaluate_with_hook(mdl, eval_tokens, cfg, device,
                                           hook_layer=None, hook_fn=None)

        acts = get_model_activations(mdl, eval_tokens, layer, device)  # [N, seq, d] cpu
        acts_at_pos_cpu = acts[:, pos, :]

        def to_native_t(ref_dirs):
            native = native_basis_dirs(ref_dirs, name, site_name, d)
            return torch.tensor(native, dtype=torch.float32, device=device)

        shared_t = to_native_t(shared_dirs)
        bottom_t = to_native_t(bottom_dirs)

        model_entry = {
            "baseline_acc": float(baseline_acc),
            "shared": run_ablation_for_direction_set(
                mdl, eval_tokens, cfg, device, layer, pos, baseline_acc,
                shared_t, acts_at_pos_cpu, k_values),
            "anti_shared_raw": run_ablation_for_direction_set(
                mdl, eval_tokens, cfg, device, layer, pos, baseline_acc,
                bottom_t, acts_at_pos_cpu, k_values),
            "anti_shared_ortho": run_ablation_per_k_map(
                mdl, eval_tokens, cfg, device, layer, pos, baseline_acc,
                ortho_per_k, name, site_name, d, acts_at_pos_cpu),
            "complement_top_k": run_ablation_per_k_map(           # Path 1
                mdl, eval_tokens, cfg, device, layer, pos, baseline_acc,
                complement_per_k, name, site_name, d, acts_at_pos_cpu),
            "random": run_random_baseline(
                mdl, eval_tokens, cfg, device, layer, pos, baseline_acc,
                acts_at_pos_cpu, d, k_values, n_random_trials, rng,
                C_total_plus_ridge),
        }
        # Path 2 — projection-trace variance (the ACTUAL variance removed by
        # ablating each subspace). Basis-independent.
        model_entry["projection_trace_variance"] = {
            "shared": {k: projection_trace_variance(shared_dirs[:k], C_total)
                        for k in k_values if k <= len(shared_dirs)},
            "anti_shared_raw": {k: projection_trace_variance(bottom_dirs[:k], C_total)
                                 for k in k_values if k <= len(bottom_dirs)},
            "anti_shared_ortho": {k: projection_trace_variance(ortho_per_k[k], C_total)
                                   for k in k_values if k in ortho_per_k and ortho_per_k[k].shape[0] > 0},
            "complement_top_k": {k: projection_trace_variance(complement_per_k[k], C_total)
                                  for k in k_values if k in complement_per_k},
        }
        per_model[name] = model_entry
        del mdl
        torch.cuda.empty_cache()
        print(f"      done in {time.time()-t0:.1f}s   baseline={baseline_acc:.4f}")

    return {
        "site": site_name,
        "layer": layer,
        "position_idx": pos,
        "extraction": {
            "eigenvalues_first5": extraction["eigenvalues"][:5],
            "eigenvalues_last5": extraction["eigenvalues"][-5:],
            "denom_condition_number": extraction["denom_condition_number"],
            "ridge": extraction["ridge"],
            "d": extraction["d"],
            "k_pca": extraction["k_pca"],
            "pca_eigvals_kept_fraction": extraction["pca_eigvals_kept_fraction"],
            "shared_trace_check_first3": shared_traces,
            "bottom_trace_check_first3": bottom_traces,
        },
        "geometry": {
            # Path 3 — subspace-level descriptors at k=10 (principal angles as cosines)
            "shared_vs_bottom_angles_cos": subspace_principal_angles_cos(
                shared_dirs[:10], bottom_dirs[:10]) if len(shared_dirs) >= 10 and len(bottom_dirs) >= 10 else [],
            "shared_vs_complement_angles_cos": subspace_principal_angles_cos(
                shared_dirs[:10], complement_per_k.get(10, np.zeros((0, d)))) if 10 in complement_per_k else [],
            "bottom_vs_complement_angles_cos": subspace_principal_angles_cos(
                bottom_dirs[:10], complement_per_k.get(10, np.zeros((0, d)))) if 10 in complement_per_k else [],
            "subspace_variance_k10": {
                "shared":            projection_trace_variance(shared_dirs[:10], C_total)  if len(shared_dirs) >= 10 else None,
                "anti_shared_raw":   projection_trace_variance(bottom_dirs[:10], C_total)  if len(bottom_dirs) >= 10 else None,
                "anti_shared_ortho": (projection_trace_variance(ortho_per_k[10], C_total)
                                      if 10 in ortho_per_k and ortho_per_k[10].shape[0] > 0 else None),
                "complement_top_k":  (projection_trace_variance(complement_per_k[10], C_total)
                                      if 10 in complement_per_k else None),
                "full_residual":     float(np.trace(C_total)),
            },
        },
        "models": per_model,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 site, k=10, 10 random trials")
    ap.add_argument("--site-only", type=str, default=None)
    ap.add_argument("--skip-eps-sweep", action="store_true")
    ap.add_argument("--out", type=str, default=os.path.join(P1_DIR, "p1_results.json"))
    args = ap.parse_args()

    os.makedirs(P1_DIR, exist_ok=True)

    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  d_model={cfg.d_model}")

    models = get_converged_models("models")
    print(f"Found {len(models)} converged models")

    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    print(f"Eval set: {eval_tokens.shape[0]} problems")

    if args.smoke:
        sites = ["layer1_result_0"]
        k_values = [10]
        n_random = 10
        max_dims = 10
        eps_sweep = [1e-8]
    else:
        sites = list(PRIMARY_SITES) + [CONTROL_SITE]
        if args.site_only:
            sites = [args.site_only]
        k_values = K_VALUES
        n_random = N_RANDOM_TRIALS
        max_dims = max(K_VALUES)
        eps_sweep = EPS_SWEEP if not args.skip_eps_sweep else [1e-8]

    all_results = {
        "config": {
            "k_values": k_values,
            "n_random_trials": n_random,
            "eps_sweep": eps_sweep,
            "max_dims": max_dims,
            "sites": sites,
            "d_model": cfg.d_model,
            "n_models": len(models),
        },
        "primary": {},     # eps=1e-8 default run, all sites
        "eps_sweep": {},   # ridge-stability check at layer1_result_0, k=10
    }

    # Primary run at default ε (matches analysis.py default)
    for site in sites:
        print(f"\n=== PRIMARY RUN :: {site} ===")
        all_results["primary"][site] = run_site(
            models, site, cfg, eval_tokens, device,
            k_values, n_random, max_dims, eps_scale=1e-8,
        )
        # Incremental save so a crash mid-run keeps partial results
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)

    # ε-sweep: small focused run at single site, single k, default random count
    if len(eps_sweep) > 1:
        print(f"\n=== ε-SWEEP :: layer1_result_0, k=10, n_random={min(n_random, 30)} ===")
        for eps in eps_sweep:
            print(f"  ε = {eps:g}")
            all_results["eps_sweep"][str(eps)] = run_site(
                models, "layer1_result_0", cfg, eval_tokens, device,
                [10], min(n_random, 30), max_dims_request=50, eps_scale=eps,
            )
            with open(args.out, "w") as f:
                json.dump(all_results, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
