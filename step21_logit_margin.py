"""Step 21: Logit-margin reporting for ablation + swap experiments.

Motivation (Claude paradox-hunter scale-critique, 2026-04):
    At 99.6% baseline accuracy, a "≤0.003 accuracy drop" could mean either
    (a) the subspace is causally irrelevant or (b) the output head is saturated
    and we cannot detect dents until the model flips a prediction. A logit
    margin (correct answer logit minus runner-up logit, averaged over problems)
    is a continuous measure that exposes sub-flip perturbations.

This script reruns five key conditions at the primary layer-3 site
(layer3_result_0, layer 3, position 12) for two baseline models and reports
BOTH accuracy and mean logit margin across the 6 result positions:

    1. baseline            - no intervention (sanity)
    2. shared_ablate       - project out k=10 shared dirs (A2-normalized)
    3. complement_ablate   - project out k=10 complement dirs
    4. joint_ablate        - project out both (step15 result)
    5. shared_swap (A←B)   - A's shared component replaced with B's (step19)

We also include 'complement_swap' as a null control — step19 found it also
yielded ~0 drop, but its logit margin should show the difference.

Outputs results/p1/logit_margin.json.
"""

import json
import os

import numpy as np
import torch

from config import ModelConfig
from model import ArithmeticTransformer
from collect_activations import get_converged_models
from data import load_eval_set
from step9_p1 import (
    extract_with_max_dims,
    complement_top_k,
    make_ablation_hook,
    K_PCA,
)


RESULTS_DIR = "results"
SITE = "layer3_result_0"
LAYER = 3
POS = 12
K = 10
OUT = os.path.join(RESULTS_DIR, "p1", "logit_margin.json")


def load_aligned(site_name):
    d = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(d, f))
    return out


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
def eval_acc_and_margin(model, eval_tokens, device, injected=None,
                         ablate_hook=None, batch_size=256):
    """Returns (accuracy, mean_logit_margin_per_position_avg).

    Logit margin per (problem, result position) = logit(correct) - max(logit(wrong)).
    Averaged over all 6 result positions and all problems in eval set.
    """
    model.eval()

    injected_t = None
    if injected is not None:
        injected_t = torch.tensor(injected, dtype=torch.float32, device=device)

    cur = [0]
    def inject_hook(module, inputs, output):
        b = output.shape[0]
        output[:, POS, :] = injected_t[cur[0]:cur[0]+b]
        return output

    handle = None
    if injected is not None:
        handle = model.layers[LAYER].register_forward_hook(inject_hook)
    elif ablate_hook is not None:
        handle = model.layers[LAYER].register_forward_hook(ablate_hook)

    correct = 0
    total = 0
    margin_sum = 0.0
    margin_count = 0

    for i in range(0, eval_tokens.shape[0], batch_size):
        cur[0] = i
        batch = eval_tokens[i:i+batch_size].to(device)
        logits = model(batch)                            # [B, T, V]
        pred_logits = logits[:, 11:17, :]                 # [B, 6, V]
        true = batch[:, 12:18]                            # [B, 6]

        pred = pred_logits.argmax(dim=-1)                 # [B, 6]
        match = (pred == true).all(dim=-1)
        correct += match.sum().item()
        total += batch.shape[0]

        # Logit margin: correct-logit - max(wrong-logit)
        B, P, V = pred_logits.shape
        true_logit = pred_logits.gather(-1, true.unsqueeze(-1)).squeeze(-1)  # [B, 6]
        masked = pred_logits.clone()
        masked.scatter_(-1, true.unsqueeze(-1), float("-inf"))
        runner_up = masked.max(dim=-1).values                                # [B, 6]
        margin = (true_logit - runner_up)                                    # [B, 6]
        margin_sum += margin.sum().item()
        margin_count += margin.numel()

    if handle is not None:
        handle.remove()

    return correct / total, margin_sum / margin_count


def make_ablate_hook_numpy(V_native, acts_cpu, device):
    """V_native: [k, d] in native basis. acts_cpu: [N, d] for computing mean proj."""
    dirs_t = torch.tensor(V_native, dtype=torch.float32, device=device)
    dirs_cpu = dirs_t.cpu()
    projs = acts_cpu @ dirs_cpu.T
    mean_projs = projs.mean(dim=0).to(device)
    return make_ablation_hook(dirs_t, mean_projs, POS)


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    N = eval_tokens.shape[0]
    print(f"Eval set: {N} problems, device={device}")

    # Get shared / complement directions (reference frame) exactly as step9
    aligned = load_aligned(SITE)
    ex = extract_with_max_dims(aligned, max_dims=K, eps_scale=1e-8, k_pca=K_PCA)
    V_S_ref = ex["shared_dirs"][:K]                     # [K, d]
    C_total = ex["C_total"]
    V_C_ref = complement_top_k(V_S_ref, C_total, k=K)   # [K, d]
    d = ex["d"]
    P_S_ref = projection_matrix(V_S_ref)
    P_C_ref = projection_matrix(V_C_ref)
    print(f"Extracted: d={d}, K={K}, trace(V_S C V_S^T)={np.trace(V_S_ref @ C_total @ V_S_ref.T):.3f}, "
          f"trace(V_C C V_C^T)={np.trace(V_C_ref @ C_total @ V_C_ref.T):.3f}")

    # Two baseline models as A and B (as in step20)
    all_models = get_converged_models("models")
    want = ["baseline_seed0", "baseline_seed1"]
    models = [m for m in all_models if m["model_name"] in want]
    if len(models) != 2:
        print("Need baseline_seed0 and baseline_seed1")
        return

    # Cache native activations and means
    native_cache = {}
    for m in models:
        name = m["model_name"]
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()
        h = compute_native_activations(mdl, eval_tokens, device)
        mean_native = h.mean(axis=0)
        R = load_R(name, SITE, d)
        native_cache[name] = {
            "h": h, "mean": mean_native, "R": R, "model_path": m["model_path"],
        }
        del mdl
        torch.cuda.empty_cache()
        print(f"  cached {name}: h.shape={h.shape}")

    results = {
        "config": {"site": SITE, "layer": LAYER, "pos": POS, "k": K, "n_eval": int(N)},
        "models": {},
    }

    A, B = "baseline_seed0", "baseline_seed1"
    h_A, h_B = native_cache[A]["h"], native_cache[B]["h"]
    mean_A, mean_B = native_cache[A]["mean"], native_cache[B]["mean"]
    R_A, R_B = native_cache[A]["R"], native_cache[B]["R"]
    acts_cpu = torch.tensor(h_A)

    # Load A for all runs
    mdl = ArithmeticTransformer(cfg).to(device)
    sd = torch.load(native_cache[A]["model_path"], map_location=device, weights_only=True)
    mdl.load_state_dict(sd)
    mdl.eval()

    # Translate reference-frame directions into A's native basis
    # native = R @ ref^T pattern (analysis.py convention used throughout the codebase)
    V_S_native_A = (R_A @ V_S_ref.T).T
    V_C_native_A = (R_A @ V_C_ref.T).T
    V_joint_native_A = np.concatenate([V_S_native_A, V_C_native_A], axis=0)

    print(f"\n=== Model A = {A} ===")

    # 1. Baseline (no intervention)
    acc_base, m_base = eval_acc_and_margin(mdl, eval_tokens, device)
    print(f"  baseline:          acc={acc_base:.4f}  margin={m_base:.4f}")

    # 2. Shared ablate (k=10)
    h_shared = make_ablate_hook_numpy(V_S_native_A, acts_cpu, device)
    acc_s, m_s = eval_acc_and_margin(mdl, eval_tokens, device, ablate_hook=h_shared)
    print(f"  shared ablate:     acc={acc_s:.4f}  margin={m_s:.4f}   "
          f"(Δacc={acc_base-acc_s:+.4f}  Δmargin={m_base-m_s:+.4f})")

    # 3. Complement ablate (k=10)
    h_comp = make_ablate_hook_numpy(V_C_native_A, acts_cpu, device)
    acc_c, m_c = eval_acc_and_margin(mdl, eval_tokens, device, ablate_hook=h_comp)
    print(f"  complement ablate: acc={acc_c:.4f}  margin={m_c:.4f}   "
          f"(Δacc={acc_base-acc_c:+.4f}  Δmargin={m_base-m_c:+.4f})")

    # 4. Joint ablate (k=10 shared + k=10 complement = 20-dim joint ablation)
    h_joint = make_ablate_hook_numpy(V_joint_native_A, acts_cpu, device)
    acc_j, m_j = eval_acc_and_margin(mdl, eval_tokens, device, ablate_hook=h_joint)
    print(f"  joint ablate (20): acc={acc_j:.4f}  margin={m_j:.4f}   "
          f"(Δacc={acc_base-acc_j:+.4f}  Δmargin={m_base-m_j:+.4f})")

    # 5. Cross-model shared swap (step19 protocol)
    h_A_ref = (h_A - mean_A) @ R_A
    h_B_ref = (h_B - mean_B) @ R_B
    h_A_sswap_ref = h_A_ref + (h_B_ref - h_A_ref) @ P_S_ref.T
    h_A_sswap_native = h_A_sswap_ref @ R_A.T + mean_A
    acc_ss, m_ss = eval_acc_and_margin(mdl, eval_tokens, device, injected=h_A_sswap_native)
    print(f"  shared swap A←B:   acc={acc_ss:.4f}  margin={m_ss:.4f}   "
          f"(Δacc={acc_base-acc_ss:+.4f}  Δmargin={m_base-m_ss:+.4f})")

    # 6. Cross-model complement swap (step19 protocol)
    h_A_cswap_ref = h_A_ref + (h_B_ref - h_A_ref) @ P_C_ref.T
    h_A_cswap_native = h_A_cswap_ref @ R_A.T + mean_A
    acc_cs, m_cs = eval_acc_and_margin(mdl, eval_tokens, device, injected=h_A_cswap_native)
    print(f"  complement swap A←B: acc={acc_cs:.4f}  margin={m_cs:.4f}  "
          f"(Δacc={acc_base-acc_cs:+.4f}  Δmargin={m_base-m_cs:+.4f})")

    # 7. Positive control: shuffle A's own activations across inputs
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    while (perm == np.arange(N)).any():
        perm = rng.permutation(N)
    h_shuf = h_A[perm]
    acc_sh, m_sh = eval_acc_and_margin(mdl, eval_tokens, device, injected=h_shuf)
    print(f"  self-input-shuffle: acc={acc_sh:.4f}  margin={m_sh:.4f}  "
          f"(Δacc={acc_base-acc_sh:+.4f}  Δmargin={m_base-m_sh:+.4f})")

    # 8. Zero-out (vacuous) for absolute upper bound of perturbation
    acc_z, m_z = eval_acc_and_margin(mdl, eval_tokens, device, injected=np.zeros_like(h_A))
    print(f"  zero-out:           acc={acc_z:.4f}  margin={m_z:.4f}  "
          f"(Δacc={acc_base-acc_z:+.4f}  Δmargin={m_base-m_z:+.4f})")

    results["models"][A] = {
        "baseline":            {"acc": float(acc_base), "margin": float(m_base)},
        "shared_ablate":       {"acc": float(acc_s),  "margin": float(m_s),
                                 "d_acc": float(acc_base - acc_s),
                                 "d_margin": float(m_base - m_s)},
        "complement_ablate":   {"acc": float(acc_c),  "margin": float(m_c),
                                 "d_acc": float(acc_base - acc_c),
                                 "d_margin": float(m_base - m_c)},
        "joint_ablate":        {"acc": float(acc_j),  "margin": float(m_j),
                                 "d_acc": float(acc_base - acc_j),
                                 "d_margin": float(m_base - m_j)},
        "shared_swap":         {"acc": float(acc_ss), "margin": float(m_ss),
                                 "d_acc": float(acc_base - acc_ss),
                                 "d_margin": float(m_base - m_ss)},
        "complement_swap":     {"acc": float(acc_cs), "margin": float(m_cs),
                                 "d_acc": float(acc_base - acc_cs),
                                 "d_margin": float(m_base - m_cs)},
        "self_input_shuffle":  {"acc": float(acc_sh), "margin": float(m_sh),
                                 "d_acc": float(acc_base - acc_sh),
                                 "d_margin": float(m_base - m_sh)},
        "zero_out":            {"acc": float(acc_z),  "margin": float(m_z),
                                 "d_acc": float(acc_base - acc_z),
                                 "d_margin": float(m_base - m_z)},
    }

    del mdl
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")

    print("\nInterpretation guide:")
    print("  If shared_ablate / shared_swap show Δmargin << Δmargin(complement_ablate / zero_out),")
    print("  the finding survives under margin metric — shared single-subspace ablation")
    print("  really is near-vacuous even below the accuracy-flipping threshold.")
    print("  If shared_ablate has a SIMILAR margin drop to complement, the accuracy finding")
    print("  was an artifact of the saturated output head — the margin is the honest metric.")


if __name__ == "__main__":
    main()
