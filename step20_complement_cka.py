"""Phase 4 / Step 20 — Complement-CKA universality-inversion test (H-A6).

Pre-registration: PHASE4_A4_PREREGISTRATION.md.

Computes complement-CKA across pairs of 4-layer models:
  - Within main 4L addition zoo at layer3_result_0
  - Within 4L mod-23 zoo at layer3_equals
  - Cross: 4L addition × 4L mod-23 (after alignment via projection onto matched
    complement bases and eigen-spectrum normalization)

Decision rule (from pre-reg):
  H-A6 confirmed iff
    mean_CKA(within_main_4L) - mean_CKA(main_4L × modp_4L) >= 0.15
    with 95% bootstrap CI lower bound >= 0.05.

Pure CPU analysis. Uses existing aligned activations from the published paper.

Outputs results/p1/complement_cka.json.
"""

import json
import os
import time

import numpy as np

from step9_p1 import (
    extract_with_max_dims, complement_top_k, K_PCA, load_aligned_acts,
)
from analysis import linear_cka


RESULTS_DIR_MAIN = "results"
RESULTS_DIR_MODP = "results/modp"
OUT_PATH = "results/p1/complement_cka.json"

K = 8


def load_main_aligned_acts(site_name: str) -> dict:
    return load_aligned_acts(site_name)


def load_modp_aligned_acts(site_name: str) -> dict:
    """Load aligned activations from the mod-p pipeline output."""
    d = os.path.join(RESULTS_DIR_MODP, f"aligned_{site_name}")
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(d, f))
    return out


def _best_prefix(name_map: dict, want_kind: str) -> list[str]:
    """Filter model names by kind: baseline-only is safest for cross-zoo pairs."""
    return [n for n in name_map if "baseline" in n or n.startswith("baseline")]


def complement_directions(aligned: dict, k: int = K):
    ex = extract_with_max_dims(aligned, max_dims=k, eps_scale=1e-8, k_pca=K_PCA)
    shared = ex["shared_dirs"]
    C_total = ex["C_total"]
    comp = complement_top_k(shared, C_total, k)
    return {
        "shared": shared,
        "complement": comp,
        "C_total": C_total,
        "d": ex["d"],
    }


def project_to_complement(aligned_X: np.ndarray, comp_basis: np.ndarray) -> np.ndarray:
    """Project aligned activations onto the k-dim complement basis (Euclidean).
    Returns [n_samples, k] coordinates in the complement basis."""
    # complement_top_k returns rows C_total-orthonormal (not Euclidean). Take
    # Euclidean coordinates via pinv.
    B = comp_basis.T              # [d, k]
    # Gram = B^T B; coords = X B (B^T B)^{-1}
    G = B.T @ B
    G_inv = np.linalg.pinv(G)
    return aligned_X @ B @ G_inv  # [n, k]


def pairwise_cka_in_complement(aligned_zoo_A: dict, aligned_zoo_B: dict,
                                 comp_A: np.ndarray, comp_B: np.ndarray,
                                 n_samples_cap: int = 1000):
    """For each pair (m_A, m_B), compute CKA between complement-projected
    activations. Uses a sample-matched common index.

    The two zoos may share evaluation support; we truncate to the common
    length to ensure aligned row-wise comparison. The coordinates are in
    different complement bases (A vs B), so CKA in those coordinates
    measures relational structure similarity irrespective of basis.
    """
    names_A = list(aligned_zoo_A.keys())
    names_B = list(aligned_zoo_B.keys())
    # common row count
    n_min = min(min(arr.shape[0] for arr in aligned_zoo_A.values()),
                 min(arr.shape[0] for arr in aligned_zoo_B.values()))
    n_use = min(n_min, n_samples_cap)
    coords_A = {n: project_to_complement(aligned_zoo_A[n][:n_use], comp_A) for n in names_A}
    coords_B = {n: project_to_complement(aligned_zoo_B[n][:n_use], comp_B) for n in names_B}

    M = np.zeros((len(names_A), len(names_B)), dtype=np.float64)
    for i, na in enumerate(names_A):
        for j, nb in enumerate(names_B):
            M[i, j] = linear_cka(coords_A[na], coords_B[nb])
    return M, names_A, names_B


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 271828) -> tuple:
    rng = np.random.RandomState(seed)
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([rng.choice(v, size=len(v), replace=True).mean()
                        for _ in range(n_boot)])
    return float(v.mean()), float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # --- Load 4L main zoo aligned activations at layer3_result_0 ---
    main_aligned_full = load_main_aligned_acts("layer3_result_0")
    print(f"Main 4L aligned activations: {len(main_aligned_full)} models")
    # Restrict to baselines for the cleanest within-task universality estimate.
    main_aligned = {n: a for n, a in main_aligned_full.items() if "baseline" in n}
    if len(main_aligned) < 3:
        # Fall back to all models if too few baselines
        main_aligned = main_aligned_full
    print(f"  baselines used: {list(main_aligned.keys())}")

    # --- Load 4L mod-p zoo aligned activations at layer3_equals (N-1 for 4L) ---
    # The mod-p paper reported layer1/2/3 at '=' as primary sites; layer3_equals
    # is the N-1 site (highest hidden load 0.141), matching main layer3_result_0
    # (hidden load 0.36) for parity.
    modp_aligned_full = load_modp_aligned_acts("layer3_equals")
    print(f"Mod-p 4L aligned at layer3_equals: {len(modp_aligned_full)} models")
    modp_aligned = {n: a for n, a in modp_aligned_full.items() if "baseline" in n}
    if len(modp_aligned) < 3:
        modp_aligned = modp_aligned_full

    if not main_aligned or not modp_aligned:
        print("ERROR: missing aligned activations. Need to re-run step9/step9_p1_modp "
              "to produce aligned_layer3_result_0 (main) and aligned_layer2_equals (modp).")
        return

    # --- Extraction on each zoo independently ---
    ex_main = complement_directions(main_aligned, k=K)
    ex_modp = complement_directions(modp_aligned, k=K)
    print(f"\nExtractions:")
    print(f"  main: d={ex_main['d']}, K={K}, complement shape={ex_main['complement'].shape}")
    print(f"  modp: d={ex_modp['d']}, K={K}, complement shape={ex_modp['complement'].shape}")

    # Zoo-internal CKA: A vs A (main within), B vs B (modp within), A vs B (cross).
    M_main, names_main, _ = pairwise_cka_in_complement(
        main_aligned, main_aligned, ex_main["complement"], ex_main["complement"])
    M_modp, names_modp, _ = pairwise_cka_in_complement(
        modp_aligned, modp_aligned, ex_modp["complement"], ex_modp["complement"])

    # For the cross comparison, activations live in potentially different
    # residual-stream bases and potentially different dimensions. If d_main
    # != d_modp, we cannot project directly with a single complement basis.
    # Check dimensions.
    if ex_main["d"] != ex_modp["d"]:
        print(f"WARN: main and modp d_model differ ({ex_main['d']} vs {ex_modp['d']}). "
              f"Cross-task comparison requires dimension match.")
        M_cross = None
    else:
        M_cross, _, _ = pairwise_cka_in_complement(
            main_aligned, modp_aligned, ex_main["complement"], ex_modp["complement"])

    # Off-diagonal CKA values
    def offdiag(M):
        if M is None: return np.array([])
        n = min(M.shape)
        return np.array([M[i, j] for i in range(M.shape[0]) for j in range(M.shape[1])
                         if i != j])

    within_main = offdiag(M_main)
    within_modp = offdiag(M_modp)
    cross_main_modp = offdiag(M_cross) if M_cross is not None else np.array([])

    # Stats
    def stat(name, vals):
        m, lo, hi = bootstrap_mean_ci(vals)
        return {"n_pairs": int(vals.size), "mean": m, "ci95_lo": lo, "ci95_hi": hi}

    wm = stat("within_main", within_main)
    wp = stat("within_modp", within_modp)
    cx = stat("cross_main_modp", cross_main_modp)

    # Gap: within_main mean - cross mean, with CI via bootstrap of the
    # difference of means.
    rng = np.random.RandomState(161803)
    gap_boot = []
    if within_main.size and cross_main_modp.size:
        for _ in range(2000):
            a = rng.choice(within_main, size=within_main.size, replace=True)
            b = rng.choice(cross_main_modp, size=cross_main_modp.size, replace=True)
            gap_boot.append(float(a.mean() - b.mean()))
    gap_boot = np.asarray(gap_boot)
    if gap_boot.size:
        gap_mean = float(gap_boot.mean())
        gap_lo = float(np.quantile(gap_boot, 0.025))
        gap_hi = float(np.quantile(gap_boot, 0.975))
    else:
        gap_mean = gap_lo = gap_hi = float("nan")

    # Verdict per pre-reg
    if np.isnan(gap_mean):
        verdict = "INCONCLUSIVE_NO_CROSS"
    elif gap_mean >= 0.15 and gap_lo >= 0.05:
        verdict = "CONFIRMED"
    elif gap_mean < 0.05 and gap_hi < 0.10:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "config": {"K": K, "ridge_eps": 1e-8, "K_PCA": K_PCA,
                    "main_site": "layer3_result_0",
                    "modp_site": "layer3_equals"},
        "n_main_models": len(main_aligned),
        "n_modp_models": len(modp_aligned),
        "d_main": ex_main["d"],
        "d_modp": ex_modp["d"],
        "within_main": wm,
        "within_modp": wp,
        "cross_main_modp": cx,
        "gap_within_main_minus_cross": {
            "mean": gap_mean, "ci95_lo": gap_lo, "ci95_hi": gap_hi,
        },
        "verdict_H_A6": verdict,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n=== H-A6 verdict: {verdict} ===")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
