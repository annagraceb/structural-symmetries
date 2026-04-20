"""Phase 4 / Step 19 — A4 k-sweep with held-out split and A2-prime matching.

Pre-registration: PHASE4_A4_PREREGISTRATION.md (locked before this script runs).

For each configured site, computes:
  - Extraction on split_A (even-indexed eval sequences)
  - Ablation on split_B (odd-indexed eval sequences)
  - k-sweep over k = 1..8:
      * structured_drop(k):      ablation of complement_top_k(shared[:k], C_total, k)
      * random_a2_drop(k):       100 random C_total-whitened subspaces (existing A2)
      * random_a2prime_drop(k):  rejection-sampled to match task-subspace projection energy

Outputs: results/p1/a4_ksweep.json, one entry per site.

Sites (locked):
  primary:                deep8_layer7_result_0  (8L, layer 7, pos 12)   [N-1, result-0]
  null_wrong_position:    deep8_layer7_position9 (8L, layer 7, pos 9)    [same layer, B[3]]
  null_early_layer:       deep8_layer1_result_0  (8L, layer 1, pos 12)   [pre-computation]
  null_permuted:          permuted_deep8_layer7_result_0 (permuted zoo)  [added after zoo trains]
"""

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import torch

from analysis import procrustes_align
from config_deep8 import ModelConfigDeep8
from data import load_eval_set
from model import ArithmeticTransformer
from step9_p1 import (
    extract_with_max_dims, complement_top_k, whitened_random_subspace,
    make_ablation_hook, K_PCA,
)
from step9_p1_deep import evaluate_with_hook_deep, get_model_activations_deep


MODELS_DIR_DEEP8 = "models_deep8"
MODELS_DIR_PERMUTED = "models_deep8_permuted"
ACT_DIR_DEEP8 = "activations_deep8"
ACT_DIR_PERMUTED = "activations_deep8_permuted"
RESULTS_DIR = "results/deep8"
P1_DIR = "results/p1"
OUT_PATH = os.path.join(P1_DIR, "a4_ksweep.json")

K_VALUES = list(range(1, 9))
N_RANDOM_A2 = 30
N_A2PRIME_POOL = 300
N_A2PRIME_DRAWS = 30
A2PRIME_WINDOW = 0.20  # +/- 20% around median of structured-projection energy

SITE_SPECS = {
    "deep8_layer7_result_0":  (MODELS_DIR_DEEP8,   ACT_DIR_DEEP8,    7, 12),
    "deep8_layer7_position9": (MODELS_DIR_DEEP8,   ACT_DIR_DEEP8,    7,  9),
    "deep8_layer1_result_0":  (MODELS_DIR_DEEP8,   ACT_DIR_DEEP8,    1, 12),
    "permuted_deep8_layer7_result_0": (MODELS_DIR_PERMUTED, ACT_DIR_PERMUTED, 7, 12),
}


def get_models(models_dir: str, require_converged: bool = True,
                min_acc: float = 0.99) -> list:
    out = []
    if not os.path.isdir(models_dir):
        return out
    for entry in sorted(os.listdir(models_dir)):
        mp = os.path.join(models_dir, entry, "metadata.json")
        pp = os.path.join(models_dir, entry, "model.pt")
        if not (os.path.exists(mp) and os.path.exists(pp)):
            continue
        with open(mp) as f:
            meta = json.load(f)
        if require_converged:
            if not meta.get("converged"):
                continue
            if meta.get("best_accuracy", meta.get("final_accuracy", 0)) < min_acc:
                continue
        else:
            # For permuted zoo: use acceptance by best_accuracy only
            if meta.get("best_accuracy", 0) < min_acc:
                continue
        meta["model_name"] = entry
        meta["model_path"] = pp
        out.append(meta)
    return out


def ensure_activations(models: list, act_dir: str, layer: int,
                        cfg: ModelConfigDeep8, eval_tokens: torch.Tensor,
                        device: str):
    """Collect and save activations[:, :, :] at `layer` if missing."""
    os.makedirs(act_dir, exist_ok=True)
    for m in models:
        name = m["model_name"]
        out_dir = os.path.join(act_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f"layer{layer}.npy")
        if os.path.exists(fp):
            continue
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()
        chunks = []
        with torch.no_grad():
            B = 256
            for i in range(0, eval_tokens.shape[0], B):
                batch = eval_tokens[i:i+B].to(device)
                _, hiddens = mdl(batch, return_all_hiddens=True)
                chunks.append(hiddens[layer].cpu())
        arr = torch.cat(chunks, dim=0).numpy().astype(np.float32)
        np.save(fp, arr)
        print(f"  collected activations: {name} layer{layer}")
        del mdl
        torch.cuda.empty_cache()


def align_site(models: list, site_name: str, act_dir: str, layer: int, pos: int,
                results_dir: str, ref_name: Optional[str] = None) -> dict:
    """Procrustes-align all models at (layer, pos), cache R_ and aligned_ files."""
    aligned_dir = os.path.join(results_dir, f"aligned_{site_name}")
    if ref_name is None:
        ref_name = models[0]["model_name"]
    os.makedirs(aligned_dir, exist_ok=True)

    X_ref = np.load(os.path.join(act_dir, ref_name, f"layer{layer}.npy"))[:, pos, :]
    X_ref_c = X_ref - X_ref.mean(axis=0)
    aligned = {ref_name: X_ref_c}
    ref_out = os.path.join(aligned_dir, f"{ref_name}.npy")
    if not os.path.exists(ref_out):
        np.save(ref_out, X_ref_c)

    for m in models:
        name = m["model_name"]
        if name == ref_name:
            continue
        ap = os.path.join(aligned_dir, f"{name}.npy")
        rp = os.path.join(results_dir, f"R_{name}_{site_name}.npy")
        if os.path.exists(ap) and os.path.exists(rp):
            aligned[name] = np.load(ap)
            continue
        X_k = np.load(os.path.join(act_dir, name, f"layer{layer}.npy"))[:, pos, :]
        R, _ = procrustes_align(X_ref, X_k)
        X_a = (X_k - X_k.mean(axis=0)) @ R
        aligned[name] = X_a
        np.save(rp, R)
        np.save(ap, X_a)
    return aligned


def native_basis_from_R(ref_dirs: np.ndarray, model_name: str, site_name: str,
                         results_dir: str, d: int) -> np.ndarray:
    R_path = os.path.join(results_dir, f"R_{model_name}_{site_name}.npy")
    if os.path.exists(R_path):
        R = np.load(R_path)
    else:
        R = np.eye(d)
    return (R @ ref_dirs.T).T


def a2prime_filter(random_pool: np.ndarray, structured: np.ndarray,
                    C_total: np.ndarray, window: float):
    """Filter random subspaces to those whose projection-energy onto the
    structured subspace lies within ±window of the median.

    random_pool: [N, k, d]
    structured:  [k_s, d]
    Energy metric: sum_i || proj_span(structured)(v_i) ||^2 in C_total norm.
    """
    S = structured.T
    StS = S.T @ S
    try:
        StS_inv = np.linalg.pinv(StS)
    except np.linalg.LinAlgError:
        StS_inv = np.linalg.pinv(StS + 1e-10 * np.eye(StS.shape[0]))
    P = S @ StS_inv @ S.T  # [d, d] Euclidean projector onto span(structured)

    energies = np.zeros(random_pool.shape[0])
    for i, V in enumerate(random_pool):
        # Projected rows: V @ P, weight each row by its own C_total norm.
        proj_rows = V @ P  # [k, d]
        energies[i] = float(np.mean(np.einsum('ij,jk,ik->i', proj_rows, C_total, proj_rows)))

    med = float(np.median(energies))
    lo, hi = med * (1.0 - window), med * (1.0 + window)
    mask = (energies >= lo) & (energies <= hi)
    return random_pool[mask], energies, med


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                  rng: Optional[np.random.RandomState] = None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.RandomState(42)
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n < 2:
        return (float("nan"), float("nan"))
    idx = rng.randint(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def run_site(site_name: str, cfg: ModelConfigDeep8, eval_tokens: torch.Tensor,
              device: str) -> dict:
    models_dir, act_dir, layer, pos = SITE_SPECS[site_name]
    if "permuted" in models_dir:
        models = get_models(models_dir, require_converged=False, min_acc=0.20)
    else:
        models = get_models(models_dir, require_converged=True, min_acc=0.99)
    if len(models) < 4:
        return {"site": site_name, "status": "INSUFFICIENT_MODELS",
                "n_models": len(models)}

    print(f"\n=== {site_name}  layer={layer} pos={pos}  n_models={len(models)} ===")

    # 1. Ensure activations for `layer` are collected.
    ensure_activations(models, act_dir, layer, cfg, eval_tokens, device)

    # 2. Held-out split: even eval-indices for extraction (split_A),
    #    odd eval-indices for ablation (split_B).
    n_eval = eval_tokens.shape[0]
    idx_A = np.arange(0, n_eval, 2)
    idx_B = np.arange(1, n_eval, 2)

    # 3. Alignment uses the FULL eval set (alignment geometry is label-blind and
    #    more stable with more samples). Only extraction/ablation are split.
    aligned_full = align_site(models, site_name, act_dir, layer, pos,
                                RESULTS_DIR, ref_name=models[0]["model_name"])

    # 4. Build split_A aligned dict (for extraction).
    aligned_A = {name: arr[idx_A] for name, arr in aligned_full.items()}

    # 5. Extraction on split_A.
    ex = extract_with_max_dims(aligned_A, max_dims=max(K_VALUES),
                                 eps_scale=1e-8, k_pca=K_PCA)
    shared = ex["shared_dirs"]            # [max_k, d]
    C_total = ex["C_total"]
    C_total_plus_ridge = ex["C_total_plus_ridge"]
    d = ex["d"]

    # 6. Per-k complement subspaces.
    complement_k = {}
    for k in K_VALUES:
        if 2 * k > d:
            continue
        complement_k[k] = complement_top_k(shared[:k], C_total, k)

    # 7. Pre-sample A2-prime random pools, one per k, with A2-prime filter.
    rng_pool = np.random.RandomState(20260417)
    random_pools = {}
    a2prime_info = {}
    for k in K_VALUES:
        if k not in complement_k:
            continue
        pool = np.stack([
            whitened_random_subspace(k, d, C_total_plus_ridge, rng_pool).astype(np.float32)
            for _ in range(N_A2PRIME_POOL)
        ])
        filtered, energies, med = a2prime_filter(pool, complement_k[k],
                                                   C_total, A2PRIME_WINDOW)
        a2prime_info[k] = {
            "pool_size": int(pool.shape[0]),
            "median_energy": med,
            "n_filtered": int(filtered.shape[0]),
            "kept_fraction": float(filtered.shape[0] / pool.shape[0]),
        }
        # Subsample to N_A2PRIME_DRAWS (or all if fewer).
        if filtered.shape[0] >= N_A2PRIME_DRAWS:
            sel = rng_pool.choice(filtered.shape[0], size=N_A2PRIME_DRAWS, replace=False)
            random_pools[k] = filtered[sel]
        else:
            random_pools[k] = filtered
        print(f"  k={k}: a2prime kept {filtered.shape[0]}/{N_A2PRIME_POOL} "
              f"(median energy={med:.4f})")

    # 8. Per-model ablation on split_B eval tokens.
    eval_tokens_B = eval_tokens[idx_B].contiguous()

    per_model = {}
    rng_a2 = np.random.RandomState(424242)
    for i, m in enumerate(models):
        name = m["model_name"]
        t0 = time.time()
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()

        baseline_acc = evaluate_with_hook_deep(mdl, eval_tokens_B, device)
        acts_B = get_model_activations_deep(mdl, eval_tokens_B, layer, device)
        acts_at_pos = acts_B[:, pos, :]

        def ablate(ref_dirs):
            native = native_basis_from_R(ref_dirs, name, site_name, RESULTS_DIR, d)
            t = torch.tensor(native, dtype=torch.float32, device=device)
            projs = acts_at_pos @ t.cpu().T
            mean_projs = projs.mean(dim=0).to(device)
            hook = make_ablation_hook(t, mean_projs, pos)
            return float(evaluate_with_hook_deep(mdl, eval_tokens_B, device,
                                                   hook_layer=layer, hook_fn=hook))

        entry = {
            "baseline_acc": float(baseline_acc),
            "structured": {},
            "random_a2": {},
            "random_a2prime": {},
        }
        for k in K_VALUES:
            if k not in complement_k:
                continue
            # Structured complement top-k
            entry["structured"][k] = float(baseline_acc - ablate(complement_k[k]))
            # A2 random nulls (unfiltered whitened random, N_RANDOM_A2 draws)
            a2_drops = []
            for _ in range(N_RANDOM_A2):
                r = whitened_random_subspace(k, d, C_total_plus_ridge, rng_a2).astype(np.float32)
                a2_drops.append(float(baseline_acc - ablate(r)))
            entry["random_a2"][k] = a2_drops
            # A2-prime nulls (pre-filtered pool)
            a2p_drops = []
            pool = random_pools.get(k)
            if pool is not None and pool.shape[0] > 0:
                for j in range(pool.shape[0]):
                    a2p_drops.append(float(baseline_acc - ablate(pool[j])))
            entry["random_a2prime"][k] = a2p_drops

        per_model[name] = entry
        del mdl
        torch.cuda.empty_cache()
        if i % 4 == 0 or i == len(models) - 1:
            print(f"  [{i+1}/{len(models)}] {name} (t={time.time()-t0:.1f}s)  "
                  f"struct@k=1={entry['structured'].get(1, 0):.3f}  "
                  f"struct@k=8={entry['structured'].get(8, 0):.3f}")

    # 9. Summary: per-k structured vs medians.
    summary = {}
    for k in K_VALUES:
        if k not in complement_k:
            continue
        s_vals = np.array([pm["structured"][k] for pm in per_model.values()
                             if k in pm["structured"]])
        # Per-model median of a2 null draws, then mean across models.
        a2_meds = np.array([float(np.median(pm["random_a2"][k]))
                              for pm in per_model.values() if k in pm["random_a2"]])
        a2p_meds = np.array([float(np.median(pm["random_a2prime"][k]))
                               for pm in per_model.values()
                               if k in pm["random_a2prime"] and len(pm["random_a2prime"][k]) > 0])

        gap_a2 = s_vals - a2_meds if len(s_vals) == len(a2_meds) else np.array([])
        gap_a2p = s_vals - a2p_meds if len(s_vals) == len(a2p_meds) else np.array([])

        summary[k] = {
            "n_models": int(len(s_vals)),
            "structured_mean": float(np.mean(s_vals)),
            "random_a2_median_mean": float(np.mean(a2_meds)) if len(a2_meds) else None,
            "random_a2prime_median_mean": float(np.mean(a2p_meds)) if len(a2p_meds) else None,
            "gap_a2_mean": float(np.mean(gap_a2)) if len(gap_a2) else None,
            "gap_a2prime_mean": float(np.mean(gap_a2p)) if len(gap_a2p) else None,
        }

    # 10. AUC over k=1..8 and bootstrap CI on gap_a2prime.
    ks_sorted = sorted(summary.keys())
    gap_a2p_per_model_per_k = {}
    for k in ks_sorted:
        gaps = []
        for pm in per_model.values():
            if k not in pm["structured"] or k not in pm["random_a2prime"]:
                continue
            if not pm["random_a2prime"][k]:
                continue
            gaps.append(pm["structured"][k] - float(np.median(pm["random_a2prime"][k])))
        gap_a2p_per_model_per_k[k] = np.asarray(gaps)

    # AUC = mean gap across k, using model-matched gaps only (same set of models).
    model_names = list(per_model.keys())
    auc_per_model = []
    for name in model_names:
        gaps_for_m = []
        for k in ks_sorted:
            pm = per_model[name]
            if k not in pm["structured"] or k not in pm["random_a2prime"]:
                continue
            a2p = pm["random_a2prime"][k]
            if not a2p:
                continue
            gaps_for_m.append(pm["structured"][k] - float(np.median(a2p)))
        if len(gaps_for_m) == len(ks_sorted):
            auc_per_model.append(float(np.mean(gaps_for_m)))
    auc_per_model_arr = np.asarray(auc_per_model)
    auc_mean = float(np.mean(auc_per_model_arr)) if auc_per_model_arr.size else float("nan")
    auc_ci = bootstrap_ci(auc_per_model_arr, n_boot=2000,
                            rng=np.random.RandomState(314159)) if auc_per_model_arr.size else (float("nan"), float("nan"))

    # Ranking sanity: for how many k is structured > random_a2prime_median (per-model mean)?
    ranking = sum(1 for k in ks_sorted
                    if summary[k]["gap_a2prime_mean"] is not None
                    and summary[k]["gap_a2prime_mean"] > 0)

    return {
        "site": site_name, "layer": layer, "position_idx": pos,
        "n_models": len(models),
        "model_names": list(per_model.keys()),
        "extraction": {
            "eigenvalues_first5": ex["eigenvalues"][:5],
            "eigenvalues_last5": ex["eigenvalues"][-5:],
            "denom_condition_number": ex["denom_condition_number"],
            "ridge": ex["ridge"], "d": ex["d"], "k_pca": ex["k_pca"],
        },
        "a2prime_info": a2prime_info,
        "per_model": per_model,
        "summary_per_k": summary,
        "auc_a2prime": {
            "mean": auc_mean,
            "ci_95": {"lo": auc_ci[0], "hi": auc_ci[1]},
            "n_models_complete": int(auc_per_model_arr.size),
        },
        "ranking_structured_beats_random_a2prime": ranking,
        "ranking_required_for_primary": 6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=None,
                         help="Subset of sites to run; default = all")
    parser.add_argument("--skip-permuted", action="store_true",
                         help="Skip permuted-zoo null site (useful if zoo not yet trained)")
    args = parser.parse_args()

    cfg = ModelConfigDeep8()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    os.makedirs(P1_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sites = args.sites or list(SITE_SPECS.keys())
    if args.skip_permuted:
        sites = [s for s in sites if "permuted" not in s]

    out = {"config": {"K_VALUES": K_VALUES, "N_RANDOM_A2": N_RANDOM_A2,
                        "N_A2PRIME_POOL": N_A2PRIME_POOL,
                        "A2PRIME_WINDOW": A2PRIME_WINDOW,
                        "ridge_eps": 1e-8,
                        "K_PCA": K_PCA},
             "sites": {}}

    # If OUT_PATH exists, merge into it.
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            prev = json.load(f)
        if "sites" in prev:
            out["sites"] = prev["sites"]

    for site in sites:
        t0 = time.time()
        res = run_site(site, cfg, eval_tokens, device)
        out["sites"][site] = res
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"[DONE] {site} ({time.time()-t0:.1f}s)")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
