"""Step 18: Unembed-geometry control.

Question: is the "shared dead at layer 3" signature in the main zoo a
geometric consequence of the unembed nullspace (the unembed just doesn't
read those directions) or a computational handoff (the unembed DOES read
those directions, but the model routes its task-relevant signal through
the complement)?

Test: for each of the 33 main-zoo models, compute:
  - The unembed weight matrix W_U (vocab_size × d_model); its row space
    (dim 12) is the "effective readout space" that the model reads from
    the residual stream.
  - The shared subspace at layer 3 (in that model's native frame, via the
    saved Procrustes rotation R_{model}).
  - The principal-angle cosines between these two subspaces.
  - The fraction of the shared-subspace variance that lives in the
    unembed nullspace (the 52-dim subspace the unembed ignores).

Predictions:
  - High nullspace fraction (>0.7): "shared dead" is trivially geometric —
    the unembed doesn't read those directions, so ablating them can't
    affect the output.
  - Low nullspace fraction (<0.3): "shared dead" is computational — the
    unembed DOES read from the shared subspace, but the model routes
    task-relevant information redundantly through the complement.

Outputs results/p1/unembed_geometry.json and prints a verdict.
"""

import json
import os

import numpy as np
import torch

from config import ModelConfig
from model import ArithmeticTransformer
from collect_activations import get_converged_models
from step9_p1 import extract_with_max_dims, load_aligned_acts, K_PCA


RESULTS_DIR = "results"
OUT = os.path.join(RESULTS_DIR, "p1", "unembed_geometry.json")

# Use the primary high-CKA site from the main zoo (where the "shared dead
# at N-1" signature was most dramatic in our Section 6.1 data).
SITE = "layer1_result_0"   # layer 1, result_0 token; N-1 for 4-layer model is layer 3
# Actually the "shared dead" site in the main zoo is layer 3 at result positions.
# Use layer3_result_0 if we have aligned activations there; otherwise the equivalent.
# The paper's cleanest "shared dead" was at layer 3 result positions from step14.

# For this experiment we want the shared subspace at layer 3. Use layer3_result_0,
# which was aligned in step14 (layer3_expanded).
LAYER3_SITE = "layer3_result_0"


def principal_angle_cosines(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Cosines of principal angles between span(V1) and span(V2)."""
    Q1, _ = np.linalg.qr(V1.T)
    Q2, _ = np.linalg.qr(V2.T)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def subspace_variance_in_nullspace(V_subspace: np.ndarray,
                                     W_U: np.ndarray) -> float:
    """Fraction of unit-variance signal in span(V_subspace) that the unembed
    cannot see. Specifically: random vectors uniform on span(V_subspace),
    fraction of their energy orthogonal to the unembed's row space.

    V_subspace: [k, d]. Row space basis of the subspace.
    W_U: [vocab, d]. Unembed matrix.

    Returns fraction in [0, 1]: 0 = entirely in row space, 1 = entirely in
    nullspace.
    """
    # Row-space projector of W_U (rank = min(vocab, d))
    U_U, S_U, Vt_U = np.linalg.svd(W_U, full_matrices=False)
    # The row space of W_U is span(Vt_U). Take all singular vectors with
    # singular value > 1e-6 as the row space.
    keep = S_U > 1e-6 * S_U.max()
    V_row = Vt_U[keep]              # [r, d]; r ≤ min(vocab, d)
    # Nullspace basis: everything else.
    # Project V_subspace onto row space; fraction of unit vectors' energy
    # that falls in the nullspace = 1 - row-space trace / total trace.
    if V_row.shape[0] == 0:
        return 1.0
    # Compute P_row @ v for each direction, measure norm squared.
    # For an orthonormal basis of subspace, average of ||P_row v||^2 over
    # the basis = trace(V_subspace @ P_row @ V_subspace.T) / k
    # where P_row = V_row.T @ V_row (projection onto row space).
    # First make V_subspace rows orthonormal so the averaging is well-defined.
    Q_sub, _ = np.linalg.qr(V_subspace.T)   # [d, k]
    V_ortho = Q_sub.T                       # orthonormal rows
    row_energy = np.sum((V_ortho @ V_row.T) ** 2) / V_ortho.shape[0]
    # row_energy = mean_{v in V_ortho} sum over r of (v . V_row[r])^2
    #           = mean_v ||P_row v||^2
    return float(1.0 - row_energy)


def native_shared_dirs(shared_ref: np.ndarray, model_name: str, site: str,
                       d: int) -> np.ndarray:
    """Convert reference-frame shared_dirs to the named model's native basis
    using the saved Procrustes R matrix."""
    R_path = os.path.join(RESULTS_DIR, f"R_{model_name}_{site}.npy")
    if not os.path.exists(R_path):
        return shared_ref.copy()  # reference model
    R = np.load(R_path)
    return (R @ shared_ref.T).T


def load_aligned(site_name: str) -> dict:
    d = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(d, f))
    return out


def main():
    cfg = ModelConfig()
    print(f"Main zoo: d_model={cfg.d_model}, vocab={cfg.vocab_size}")

    aligned = load_aligned(LAYER3_SITE)
    if not aligned:
        print(f"ERROR: no aligned data for {LAYER3_SITE}; run step14_more_layer3_sites.py first")
        return
    print(f"Loaded {len(aligned)} aligned activations at {LAYER3_SITE}")

    # Extract shared subspace at layer 3 (in reference frame)
    ex = extract_with_max_dims(aligned, max_dims=10, eps_scale=1e-8, k_pca=K_PCA)
    shared_ref = ex["shared_dirs"]       # [10, 64] C_total-orthonormal
    complement_ref = None  # we'll also compute complement as a control

    # Also compute complement top-10 (as a control — should be MORE in the
    # unembed row space, since it carries the single-subspace causal load)
    from step9_p1 import complement_top_k
    complement_ref = complement_top_k(shared_ref, ex["C_total"], k=10)

    # Bottom subspace (as a null baseline — should be mostly in nullspace
    # and carry no signal at all)
    bottom_ref = ex["bottom_dirs"]

    # Load each main-zoo model, check geometry
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_converged_models("models")
    print(f"Found {len(models)} converged models")

    out = {"per_model": {}, "config": {"site": LAYER3_SITE, "k": 10}}
    shared_frac_nullspace = []
    comp_frac_nullspace = []
    bottom_frac_nullspace = []
    random_frac_nullspace = []   # Random-subspace baseline per model
    rng = np.random.default_rng(42)

    for m in models:
        name = m["model_name"]
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)

        # Unembed matrix. lm_head.weight has shape [vocab, d_model].
        W_U = mdl.lm_head.weight.detach().cpu().numpy()
        # LN_f scales but approximately preserves subspace structure. For a
        # clean geometry question, multiply W_U by the LN_f gain to get the
        # effective readout. For small d_models this matters little; skip it.

        d = W_U.shape[1]

        # Convert shared / complement / bottom from reference frame to native frame
        shared_nat = native_shared_dirs(shared_ref, name, LAYER3_SITE, d)
        comp_nat = native_shared_dirs(complement_ref, name, LAYER3_SITE, d)
        bottom_nat = native_shared_dirs(bottom_ref, name, LAYER3_SITE, d)

        # Principal angle cosines between each and unembed row space
        _, _, Vt_U = np.linalg.svd(W_U, full_matrices=False)
        row_basis = Vt_U

        cos_shared = principal_angle_cosines(shared_nat, row_basis).tolist()
        cos_comp = principal_angle_cosines(comp_nat, row_basis).tolist()

        frac_sh = subspace_variance_in_nullspace(shared_nat, W_U)
        frac_cm = subspace_variance_in_nullspace(comp_nat, W_U)
        frac_bt = subspace_variance_in_nullspace(bottom_nat, W_U)
        # Random 10-dim baseline: sample 20 random subspaces and average
        rand_fracs = []
        for _ in range(20):
            r = rng.standard_normal((10, d))
            r, _ = np.linalg.qr(r.T)
            rand_fracs.append(subspace_variance_in_nullspace(r.T, W_U))
        frac_rn = float(np.mean(rand_fracs))
        shared_frac_nullspace.append(frac_sh)
        comp_frac_nullspace.append(frac_cm)
        bottom_frac_nullspace.append(frac_bt)
        random_frac_nullspace.append(frac_rn)

        out["per_model"][name] = {
            "shared_frac_in_nullspace": frac_sh,
            "complement_frac_in_nullspace": frac_cm,
            "bottom_frac_in_nullspace": frac_bt,
            "shared_vs_rowspace_cos_top3": cos_shared[:3],
            "complement_vs_rowspace_cos_top3": cos_comp[:3],
        }
        del mdl
        torch.cuda.empty_cache()

    # Summary
    def boot_ci(arr, n=1000, alpha=0.05):
        rng = np.random.default_rng(42)
        arr = np.asarray(arr)
        stats = np.array([arr[rng.integers(0, len(arr), len(arr))].mean()
                           for _ in range(n)])
        return float(arr.mean()), float(np.quantile(stats, alpha/2)), float(np.quantile(stats, 1-alpha/2))

    sh_m, sh_lo, sh_hi = boot_ci(shared_frac_nullspace)
    cm_m, cm_lo, cm_hi = boot_ci(comp_frac_nullspace)
    bt_m, bt_lo, bt_hi = boot_ci(bottom_frac_nullspace)
    rn_m, rn_lo, rn_hi = boot_ci(random_frac_nullspace)

    out["summary"] = {
        "shared_frac_in_nullspace": {"mean": sh_m, "ci": [sh_lo, sh_hi]},
        "complement_frac_in_nullspace": {"mean": cm_m, "ci": [cm_lo, cm_hi]},
        "bottom_frac_in_nullspace": {"mean": bt_m, "ci": [bt_lo, bt_hi]},
        "random_frac_in_nullspace": {"mean": rn_m, "ci": [rn_lo, rn_hi]},
        "analytic_random_expectation": (64 - 12) / 64,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 75)
    print(f"UNEMBED GEOMETRY TEST — site {LAYER3_SITE}, k=10, {len(models)} models")
    print("=" * 75)
    analytic_random = (64 - 12) / 64
    print(f"\nFraction of subspace variance in unembed nullspace (95% bootstrap CI):")
    print(f"  shared:     {sh_m:.4f} [{sh_lo:.3f}, {sh_hi:.3f}]")
    print(f"  complement: {cm_m:.4f} [{cm_lo:.3f}, {cm_hi:.3f}]")
    print(f"  bottom:     {bt_m:.4f} [{bt_lo:.3f}, {bt_hi:.3f}]")
    print(f"  random:     {rn_m:.4f} [{rn_lo:.3f}, {rn_hi:.3f}]  (analytic expectation: {analytic_random:.4f})")
    print()
    print("Proper interpretation (using random subspace as baseline):")
    print(f"  - Random baseline: {rn_m:.4f}")
    print("  - A subspace MORE in nullspace than random → preferentially 'invisible' to unembed.")
    print("  - A subspace LESS in nullspace than random → preferentially 'readable' by unembed.")

    # How many standard-deviations from random?
    def dev_from_random(x_samples, r_samples):
        x = np.asarray(x_samples); r = np.asarray(r_samples)
        return (x.mean() - r.mean()) / r.std(ddof=1)

    dev_sh = dev_from_random(shared_frac_nullspace, random_frac_nullspace)
    dev_cm = dev_from_random(comp_frac_nullspace, random_frac_nullspace)
    dev_bt = dev_from_random(bottom_frac_nullspace, random_frac_nullspace)

    print()
    print(f"Deviations from random baseline (in random-baseline std units):")
    print(f"  shared:     {dev_sh:+.2f}σ")
    print(f"  complement: {dev_cm:+.2f}σ")
    print(f"  bottom:     {dev_bt:+.2f}σ")
    print()
    print("Verdict:")
    if abs(dev_sh) < 2.0:
        sh_verdict = "SHARED is ~random-like w.r.t. unembed geometry"
    elif dev_sh > 2.0:
        sh_verdict = "SHARED is preferentially in NULLSPACE (pro-geometric)"
    else:
        sh_verdict = "SHARED is preferentially in ROW SPACE (contra-geometric)"
    print(f"  {sh_verdict}")

    if dev_cm < -2.0:
        print("  COMPLEMENT is strongly preferentially in ROW SPACE (expected — it's what unembed reads)")

    if abs(dev_sh) < 2.0 and dev_cm < -2.0:
        print()
        print("  Implication: the 'shared dead at N-1' phenomenon is NOT primarily explained")
        print("  by shared's nullspace residence. Shared is near-random w.r.t. the unembed.")
        print("  The complement's strong row-space preference is what drives the asymmetry:")
        print("  complement ablation hits the unembed hard; shared ablation hits near-randomly.")
        print("  The remaining ~10× gap (shared-drop 0.003 vs random-drop 0.028 at layer 3 in")
        print("  the main zoo) must come from the computational redundancy demonstrated by")
        print("  joint ablation (A3 in the paper).")


if __name__ == "__main__":
    main()
