"""Step 23: External K_pca criterion + P1 rerun at layer3_result_0.

Motivation (Claude paradox-hunter scale-critique, 2026-04):
    K_pca=32 was chosen by a "spectral cliff" heuristic on the eigenvalues of
    C_total — the same covariance used in the ratio eigenproblem. This is
    self-referential. The critic worry: the finding ("shared dead at N-1")
    might simply be an artifact of how we slice C_total.

This script defines K_pca by an EXTERNAL criterion that does not look at the
eigenvalue curve of C_total itself:

    K_pca_ext(model, site) = smallest K such that, using U_K = top-K PCs of
    C_total at (layer, pos), the explained variance of the unembed matrix
    ||W_U @ U_K||_F^2 / ||W_U||_F^2 exceeds 0.99.

    (Equivalently: smallest K such that 99% of what the readout projects
    onto lives in the top-K activation PCs.)

Then:
    1. Compute K_pca_ext for every model in the main zoo at layer3_result_0.
    2. Take K_pca_agg = max across models (conservative) OR
       K_pca_agg = median across models (central). Report both.
    3. Rerun step9 extraction + single-subspace ablation at layer3_result_0
       with K_pca_agg and compare findings to K_pca=32.

If the main finding (shared single-subspace ablation ~0 drop, complement large)
survives with K_pca_ext, the self-reference critique is refuted.

Outputs results/p1/kpca_external.json.
"""

import json
import os
import time

import numpy as np
import torch

from analysis import get_model_activations, evaluate_with_hook
from config import ModelConfig
from model import ArithmeticTransformer
from collect_activations import get_converged_models
from data import load_eval_set
from step9_p1 import (
    extract_with_max_dims,
    complement_top_k,
    make_ablation_hook,
    whitened_random_subspace,
    native_basis_dirs,
    projection_trace_variance,
)


RESULTS_DIR = "results"
SITE = "layer3_result_0"
LAYER = 3
POS = 12
K_ABL = 10                # ablation dim (matches K used in the paper)
N_RANDOM = 30
THRESHOLD = 0.99
OUT = os.path.join(RESULTS_DIR, "p1", "kpca_external.json")


def load_aligned(site_name):
    d = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    return {f[:-4]: np.load(os.path.join(d, f))
            for f in sorted(os.listdir(d)) if f.endswith(".npy")}


def compute_C_total_at(models, cfg, eval_tokens, device):
    """Per-model C_total at (LAYER, POS), returned as dict name -> [d, d].
    Each C_total is the mean-centered covariance of that model's activations.
    """
    out = {}
    for m in models:
        name = m["model_name"]
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, eval_tokens.shape[0], 256):
                batch = eval_tokens[i:i+256].to(device)
                _, hiddens = mdl(batch, return_all_hiddens=True)
                chunks.append(hiddens[LAYER][:, POS, :].cpu().numpy())
        h = np.concatenate(chunks, axis=0)
        hc = h - h.mean(axis=0)
        C = (hc.T @ hc) / h.shape[0]
        W_U = mdl.lm_head.weight.detach().cpu().numpy().astype(np.float64)  # [V, d]
        out[name] = {"C_total": C, "W_U": W_U}
        del mdl
        torch.cuda.empty_cache()
    return out


def k_pca_external_for(C_total, W_U, threshold):
    """Smallest K with ||W_U @ U_K||_F^2 / ||W_U||_F^2 >= threshold,
    where U_K are the top-K eigenvectors of C_total.
    """
    vals, vecs = np.linalg.eigh(C_total)
    order = np.argsort(vals)[::-1]
    U = vecs[:, order]
    total = float(np.sum(W_U ** 2))
    proj = W_U @ U                         # [V, d]  = coefficients in PCA basis
    # cumulative explained starting from top PC
    cum = np.cumsum(np.sum(proj ** 2, axis=0)) / total
    for k in range(1, len(cum) + 1):
        if cum[k-1] >= threshold:
            return k, cum.tolist()
    return len(cum), cum.tolist()


def run_ablation_single(model, eval_tokens, cfg, device, baseline_acc,
                         native_dirs_t, acts_at_pos_cpu):
    """Single-ablation: project out span(native_dirs_t). Returns (acc, drop)."""
    dirs_cpu = native_dirs_t.cpu()
    projs = acts_at_pos_cpu @ dirs_cpu.T
    mean_projs = projs.mean(dim=0).to(device)
    hook = make_ablation_hook(native_dirs_t, mean_projs, POS)
    acc = evaluate_with_hook(model, eval_tokens, cfg, device,
                              hook_layer=LAYER, hook_fn=hook)
    return float(acc), float(baseline_acc - acc)


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    N = eval_tokens.shape[0]
    print(f"Eval set: {N} problems")

    all_models = get_converged_models("models")
    print(f"Main zoo: {len(all_models)} models")

    # 1. Per-model K_pca_ext from W_U reconstruction
    t0 = time.time()
    per_model_data = compute_C_total_at(all_models, cfg, eval_tokens, device)
    print(f"Collected C_total + W_U for {len(per_model_data)} models in {time.time()-t0:.1f}s")

    kpca_results = {}
    for name, d in per_model_data.items():
        k_ext, cum = k_pca_external_for(d["C_total"], d["W_U"], THRESHOLD)
        kpca_results[name] = {"k_pca_ext": int(k_ext), "cum_explained": cum}

    ks = np.array([v["k_pca_ext"] for v in kpca_results.values()])
    k_median = int(np.median(ks))
    k_max = int(ks.max())
    k_min = int(ks.min())
    k_mean = float(ks.mean())
    print(f"\nK_pca_ext across {len(ks)} models (threshold={THRESHOLD}):")
    print(f"  min={k_min}  median={k_median}  mean={k_mean:.1f}  max={k_max}")
    print(f"  (Paper used K_pca=32; d_model=64.)")

    # 2. Rerun P1 ablation at layer3_result_0 with K_pca_ext (using the MAX
    # across models for a conservative choice — gives shared directions the
    # maximum possible subspace to live in).
    K_PCA_NEW = k_max

    aligned = load_aligned(SITE)
    ex_new = extract_with_max_dims(aligned, max_dims=K_ABL, eps_scale=1e-8, k_pca=K_PCA_NEW)
    V_S_new = ex_new["shared_dirs"][:K_ABL]
    C_total_stacked = ex_new["C_total"]
    V_C_new = complement_top_k(V_S_new, C_total_stacked, k=K_ABL)
    d = ex_new["d"]

    # Also re-extract with the paper's K_pca=32 for direct comparison
    ex_old = extract_with_max_dims(aligned, max_dims=K_ABL, eps_scale=1e-8, k_pca=32)
    V_S_old = ex_old["shared_dirs"][:K_ABL]
    V_C_old = complement_top_k(V_S_old, ex_old["C_total"], k=K_ABL)

    # Subspace angle between the two shared subspaces (how different is K_pca_ext vs 32?)
    def principal_angles_cos(V1, V2):
        Q1, _ = np.linalg.qr(V1.T)
        Q2, _ = np.linalg.qr(V2.T)
        s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
        return [float(x) for x in np.clip(s, 0.0, 1.0)]

    shared_angles = principal_angles_cos(V_S_new, V_S_old)
    print(f"\nPrincipal-angle cosines between shared(K_pca={K_PCA_NEW}) and shared(K_pca=32):")
    print(f"  {[round(x, 3) for x in shared_angles]}")
    print(f"  (1.0 = perfectly aligned; small values = orthogonal)")

    # 3. Per-model ablation drop with the NEW shared / complement subspaces
    per_model_ablation = {}
    rng = np.random.RandomState(1337)
    V_R_new = whitened_random_subspace(K_ABL, d, ex_new["C_total_plus_ridge"], rng).astype(np.float32)

    for m in all_models:
        name = m["model_name"]
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()

        base = evaluate_with_hook(mdl, eval_tokens, cfg, device,
                                    hook_layer=None, hook_fn=None)

        acts = get_model_activations(mdl, eval_tokens, LAYER, device)[:, POS, :]

        # shared
        V_S_native = native_basis_dirs(V_S_new, name, SITE, d)
        V_S_t = torch.tensor(V_S_native, dtype=torch.float32, device=device)
        acc_s, drop_s = run_ablation_single(mdl, eval_tokens, cfg, device, base, V_S_t, acts)
        # complement
        V_C_native = native_basis_dirs(V_C_new, name, SITE, d)
        V_C_t = torch.tensor(V_C_native, dtype=torch.float32, device=device)
        acc_c, drop_c = run_ablation_single(mdl, eval_tokens, cfg, device, base, V_C_t, acts)
        # joint (shared + complement)
        V_J_t = torch.cat([V_S_t, V_C_t], dim=0)
        acc_j, drop_j = run_ablation_single(mdl, eval_tokens, cfg, device, base, V_J_t, acts)
        # whitened random
        V_R_native = native_basis_dirs(V_R_new, name, SITE, d)
        V_R_t = torch.tensor(V_R_native, dtype=torch.float32, device=device)
        acc_r, drop_r = run_ablation_single(mdl, eval_tokens, cfg, device, base, V_R_t, acts)

        per_model_ablation[name] = {
            "baseline_acc": float(base),
            "shared":        {"acc": acc_s, "drop": drop_s},
            "complement":    {"acc": acc_c, "drop": drop_c},
            "joint":         {"acc": acc_j, "drop": drop_j},
            "random":        {"acc": acc_r, "drop": drop_r},
        }
        print(f"  {name:>40}  base={base:.4f}  "
              f"shared={drop_s:+.3f}  comp={drop_c:+.3f}  joint={drop_j:+.3f}  "
              f"rand={drop_r:+.3f}")
        del mdl
        torch.cuda.empty_cache()

    # 4. Summary (drops across models)
    def summarize(key):
        xs = np.array([per_model_ablation[n][key]["drop"] for n in per_model_ablation])
        return {"mean": float(xs.mean()), "median": float(np.median(xs)),
                "min": float(xs.min()), "max": float(xs.max()),
                "std": float(xs.std()), "n": int(len(xs))}

    summary = {k: summarize(k) for k in ["shared", "complement", "joint", "random"]}
    print(f"\nDrop summary (across {len(all_models)} models) at K_pca_ext={K_PCA_NEW}:")
    for k, s in summary.items():
        print(f"  {k:>11}  mean={s['mean']:+.3f}  median={s['median']:+.3f}  "
              f"min={s['min']:+.3f}  max={s['max']:+.3f}")

    # Subspace variance in C_total_stacked (ref frame) — the Path-2 metric
    var_shared = projection_trace_variance(V_S_new, C_total_stacked)
    var_comp   = projection_trace_variance(V_C_new, C_total_stacked)
    var_random = projection_trace_variance(V_R_new, C_total_stacked)
    print(f"\nSubspace projection-trace variance (stacked C_total):")
    print(f"  shared:     {var_shared:.3f}")
    print(f"  complement: {var_comp:.3f}")
    print(f"  random:     {var_random:.3f}")

    results = {
        "config": {
            "site": SITE, "layer": LAYER, "pos": POS, "k_ablate": K_ABL,
            "threshold": THRESHOLD, "d_model": d,
        },
        "k_pca_external": {
            "per_model": kpca_results,
            "min": k_min, "median": k_median, "mean": k_mean, "max": k_max,
            "used_for_rerun": K_PCA_NEW,
        },
        "subspace_comparison": {
            "principal_angles_shared_new_vs_old": shared_angles,
            "subspace_variance_new": {
                "shared": var_shared, "complement": var_comp, "random": var_random,
            },
        },
        "per_model_ablation": per_model_ablation,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")

    print("\nInterpretation guide:")
    print("  If the shared-vs-complement gap (single-subspace ablation) and")
    print("  the joint-ablation boost survive at K_pca_ext (median or max),")
    print("  the K_pca=32 choice was not driving the paper's finding.")
    print("  If the gap collapses at K_pca_ext, the self-reference critique wins.")


if __name__ == "__main__":
    main()
