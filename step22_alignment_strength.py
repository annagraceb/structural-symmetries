"""Step 22: Weaker-alignment controls for cross-model subspace swap.

Motivation (Claude paradox-hunter scale-critique, 2026-04):
    Step 19 showed that aligning two models via Procrustes then swapping their
    shared-subspace components on matched input produces ~0 drop. The critic
    worry: Procrustes could be "laundering" cross-model mismatch into the
    fitted rotation R, so that by the time we do the swap, "shared" has been
    defined *after* a fit that already makes the subspaces coincide. What
    looks like universal values might be downstream of a strong alignment
    fit, not intrinsic cross-model agreement.

This script compares the swap drop under THREE alignment strengths:
    1. mean-only    : align by subtracting per-model mean, no rotation
    2. random-orth  : align by subtracting mean + applying a fixed random
                      orthogonal rotation (not fit to anything)
    3. procrustes   : the full alignment used throughout the paper
                      (from results/R_{model}_{site}.npy, already fit to ref)

If the paradox-hunter critique is correct, mean-only and random-orth should
show large drops — "universal values" would be an artifact of the fit that
Procrustes performs. If the finding is real, the shared swap should
produce near-zero drop under all three alignment schemes (since the subspace
we chose already points in the same direction across models up to alignment).

Note: random-orth is expected to fail on complement, shared, AND random
subspaces because a non-identity random rotation mangles the activation.
The MEANINGFUL test is whether the per-subspace DIFFERENCES are preserved.
i.e., even with random rotation: does shared-swap still hurt less than
complement-swap?

Outputs results/p1/alignment_strength.json.
"""

import json
import os

import numpy as np
import torch

from config import ModelConfig
from model import ArithmeticTransformer
from collect_activations import get_converged_models
from data import load_eval_set
from step9_p1 import extract_with_max_dims, complement_top_k, whitened_random_subspace, K_PCA


RESULTS_DIR = "results"
SITE = "layer3_result_0"
LAYER = 3
POS = 12
K = 10
OUT = os.path.join(RESULTS_DIR, "p1", "alignment_strength.json")


def load_aligned(site_name):
    d = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    return {f[:-4]: np.load(os.path.join(d, f))
            for f in sorted(os.listdir(d)) if f.endswith(".npy")}


def load_R(model_name, site_name, d):
    p = os.path.join(RESULTS_DIR, f"R_{model_name}_{site_name}.npy")
    if os.path.exists(p):
        return np.load(p)
    return np.eye(d)


def projection_matrix(V):
    VVt_inv = np.linalg.pinv(V @ V.T)
    return V.T @ VVt_inv @ V


def compute_native_activations(model, eval_tokens, device, batch_size=256):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], batch_size):
            batch = eval_tokens[i:i+batch_size].to(device)
            _, hiddens = model(batch, return_all_hiddens=True)
            chunks.append(hiddens[LAYER][:, POS, :].cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def eval_with_inject(model, eval_tokens, device, injected, batch_size=256):
    injected_t = torch.tensor(injected, dtype=torch.float32, device=device)
    cur = [0]
    def hook(module, inputs, output):
        b = output.shape[0]
        output[:, POS, :] = injected_t[cur[0]:cur[0]+b]
        return output
    handle = model.layers[LAYER].register_forward_hook(hook)
    correct, total = 0, 0
    margin_sum, margin_count = 0.0, 0
    for i in range(0, eval_tokens.shape[0], batch_size):
        cur[0] = i
        batch = eval_tokens[i:i+batch_size].to(device)
        logits = model(batch)
        pred_logits = logits[:, 11:17, :]
        true = batch[:, 12:18]
        pred = pred_logits.argmax(dim=-1)
        match = (pred == true).all(dim=-1)
        correct += match.sum().item()
        total += batch.shape[0]
        true_logit = pred_logits.gather(-1, true.unsqueeze(-1)).squeeze(-1)
        masked = pred_logits.clone()
        masked.scatter_(-1, true.unsqueeze(-1), float("-inf"))
        runner_up = masked.max(dim=-1).values
        margin = (true_logit - runner_up)
        margin_sum += margin.sum().item()
        margin_count += margin.numel()
    handle.remove()
    return correct / total, margin_sum / margin_count


def swap_under_alignment(h_A, h_B, mean_A, mean_B, R_A, R_B, V_ref, P_proj):
    """Given a pair of aligners (R_A, R_B) and projector P_proj in ref frame,
    compute A's new native activation with A's V-component replaced by B's."""
    h_A_ref = (h_A - mean_A) @ R_A
    h_B_ref = (h_B - mean_B) @ R_B
    h_A_new_ref = h_A_ref + (h_B_ref - h_A_ref) @ P_proj.T
    return h_A_new_ref @ R_A.T + mean_A


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    N = eval_tokens.shape[0]
    print(f"Eval set: {N} problems, device={device}")

    # Extract subspaces in Procrustes-aligned reference frame (this is the
    # canonical subspace for the experiment — we will test whether swapping
    # WITHIN that same subspace works under weaker alignment during application)
    aligned = load_aligned(SITE)
    ex = extract_with_max_dims(aligned, max_dims=K, eps_scale=1e-8, k_pca=K_PCA)
    V_S_ref = ex["shared_dirs"][:K]
    C_total = ex["C_total"]
    V_C_ref = complement_top_k(V_S_ref, C_total, k=K)
    d = ex["d"]
    rng_leg = np.random.RandomState(123)
    V_R_ref = whitened_random_subspace(K, d, ex["C_total_plus_ridge"], rng_leg).astype(np.float32)

    P_S = projection_matrix(V_S_ref)
    P_C = projection_matrix(V_C_ref)
    P_R = projection_matrix(V_R_ref)

    # Two baselines as A, B
    all_models = get_converged_models("models")
    want = ["baseline_seed0", "baseline_seed1"]
    models = [m for m in all_models if m["model_name"] in want]
    if len(models) != 2:
        print("Need baseline_seed0 and baseline_seed1")
        return

    native_cache = {}
    for m in models:
        name = m["model_name"]
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()
        h = compute_native_activations(mdl, eval_tokens, device)
        native_cache[name] = {
            "h": h, "mean": h.mean(axis=0), "R": load_R(name, SITE, d),
            "model_path": m["model_path"],
        }
        del mdl
        torch.cuda.empty_cache()

    A, B = "baseline_seed0", "baseline_seed1"
    h_A, h_B = native_cache[A]["h"], native_cache[B]["h"]
    mean_A, mean_B = native_cache[A]["mean"], native_cache[B]["mean"]
    R_A_proc = native_cache[A]["R"]
    R_B_proc = native_cache[B]["R"]

    # Alignment schemes
    I = np.eye(d)
    rng = np.random.RandomState(777)
    random_rot, _ = np.linalg.qr(rng.randn(d, d).astype(np.float32))

    schemes = {
        "procrustes": (R_A_proc, R_B_proc),
        "mean_only":  (I, I),
        # Same random rotation applied to both models — if universal values
        # claim is right, swap should still work (we're consistent at least).
        "random_orth_same": (random_rot, random_rot),
        # Different random rotations — breaks the frame alignment, should
        # produce larger drops even on shared (control showing that alignment
        # matters, but not that Procrustes specifically "laundered" the result)
        "random_orth_diff": (random_rot, np.linalg.qr(rng.randn(d, d).astype(np.float32))[0]),
    }

    # Load A for forward passes
    mdl = ArithmeticTransformer(cfg).to(device)
    sd = torch.load(native_cache[A]["model_path"], map_location=device, weights_only=True)
    mdl.load_state_dict(sd)
    mdl.eval()

    # Baseline (A's own)
    base_acc, base_mg = eval_with_inject(mdl, eval_tokens, device, h_A)
    print(f"\nBaseline (A self-inject): acc={base_acc:.4f}  margin={base_mg:.4f}")

    results = {"config": {"site": SITE, "layer": LAYER, "pos": POS, "k": K,
                          "A": A, "B": B, "n_eval": int(N),
                          "baseline": {"acc": float(base_acc), "margin": float(base_mg)}},
               "schemes": {}}

    # We CANNOT directly swap a subspace defined in the Procrustes reference
    # frame using a different R. The fair comparison is: re-express the
    # subspace in the alignment-scheme's reference frame. That means
    # V_proj_ref = V_ref  (subspace lives in the canonical aligned ref
    # frame, but the coordinate change R_A/R_B defines a DIFFERENT reference
    # frame for the mean-only and random-orth schemes). We keep P_ref (the
    # Procrustes frame) fixed to define the subspace, but we USE the scheme's
    # R for mapping native→ref→native during the swap.
    #
    # Concretely under a scheme with (R_A, R_B):
    #     h_A_ref' = (h_A - mean_A) @ R_A
    #     h_B_ref' = (h_B - mean_B) @ R_B
    # If R_A is a random rotation this ref' is not aligned with the canonical
    # Procrustes ref frame where we defined V_S_ref. So we measure the
    # swap drop as-is. Note for mean-only (R = I), ref' is each model's native
    # frame without rotation — the V_S_ref directions land at arbitrary
    # orientations relative to each model's natural basis.
    #
    # This exactly tests the critique: is it the Procrustes fit that
    # makes the shared subspace coincide across models, or is it a
    # property of the subspace itself?

    for sname, (R_A, R_B) in schemes.items():
        print(f"\n--- scheme: {sname} ---")
        row = {}
        for lbl, V_ref, P_proj in [("shared", V_S_ref, P_S),
                                     ("complement", V_C_ref, P_C),
                                     ("random", V_R_ref, P_R)]:
            swap = swap_under_alignment(h_A, h_B, mean_A, mean_B, R_A, R_B, V_ref, P_proj)
            acc, mg = eval_with_inject(mdl, eval_tokens, device, swap)
            row[lbl] = {"acc": float(acc), "margin": float(mg),
                        "d_acc": float(base_acc - acc),
                        "d_margin": float(base_mg - mg)}
            print(f"  {lbl:>11} swap:  acc={acc:.4f}  margin={mg:.4f}  "
                  f"Δacc={base_acc-acc:+.4f}  Δmargin={base_mg-mg:+.4f}")

        # Also measure: the WHOLE aligned activation swap (identity subspace)
        full = swap_under_alignment(h_A, h_B, mean_A, mean_B, R_A, R_B,
                                     np.eye(d), np.eye(d))
        acc_f, mg_f = eval_with_inject(mdl, eval_tokens, device, full)
        row["full_swap"] = {"acc": float(acc_f), "margin": float(mg_f),
                            "d_acc": float(base_acc - acc_f),
                            "d_margin": float(base_mg - mg_f)}
        print(f"  {'full':>11} swap:  acc={acc_f:.4f}  margin={mg_f:.4f}  "
              f"Δacc={base_acc-acc_f:+.4f}  Δmargin={base_mg-mg_f:+.4f}")

        results["schemes"][sname] = row

    del mdl
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")

    print("\nInterpretation guide:")
    print("  If shared-swap drop is ~0 across all schemes, 'universal values'")
    print("  is intrinsic — not laundered by Procrustes fit.")
    print("  If shared-swap is ~0 only under Procrustes, the laundering")
    print("  critique wins and we must re-interpret step 19.")


if __name__ == "__main__":
    main()
