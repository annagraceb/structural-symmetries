"""Step 20: Sensitivity controls for step19 cross-model swap.

Claude paradox-hunter review concern: if shared/complement/random swaps ALL
show zero drop, we can't distinguish "universal values" from "layer 3 pos 12
is insensitive to any perturbation at this magnitude." Need positive controls
that SHOULD break accuracy to demonstrate the metric has dynamic range.

Controls run:
  1. self-input-shuffle: for model A, swap A's own activation at pos 12
     input X with A's activation at pos 12 input Y (Y != X, random permutation).
     This destroys the per-input computational state but keeps model A's own
     activation statistics. If layer 3 pos 12 is causal at all, this SHOULD hurt.
  2. native-frame raw swap: swap model A's layer-3 pos 12 activation with
     model B's RAW activation (no Procrustes alignment, no subspace
     projection). If the downstream readout is input-frame-specific, this
     should hurt too.
  3. full-activation swap (aligned): swap the ENTIRE activation in aligned
     frame between A and B on same input (not just a subspace).
     If swap protocol works, this should behave like the subspace swaps —
     zero drop if universal values, otherwise hurts.

Outputs results/p1/swap_sensitivity.json.
"""

import json
import os

import numpy as np
import torch

from config import ModelConfig
from model import ArithmeticTransformer
from collect_activations import get_converged_models
from data import load_eval_set


RESULTS_DIR = "results"
SITE = "layer3_result_0"
LAYER = 3
POS = 12
OUT = os.path.join(RESULTS_DIR, "p1", "swap_sensitivity.json")


def load_R(model_name: str, d: int) -> np.ndarray:
    p = os.path.join(RESULTS_DIR, f"R_{model_name}_{SITE}.npy")
    if os.path.exists(p):
        return np.load(p)
    return np.eye(d)


def compute_activations(model, eval_tokens, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], 256):
            batch = eval_tokens[i:i+256].to(device)
            _, hiddens = model(batch, return_all_hiddens=True)
            chunks.append(hiddens[LAYER][:, POS, :].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_with_injection(model, eval_tokens, device, injected, batch_size=256):
    t = torch.tensor(injected, dtype=torch.float32, device=device)
    cur = [0]
    def hook(module, inputs, output):
        b = output.shape[0]
        output[:, POS, :] = t[cur[0]:cur[0]+b]
        return output
    handle = model.layers[LAYER].register_forward_hook(hook)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], batch_size):
            cur[0] = i
            batch = eval_tokens[i:i+batch_size].to(device)
            logits = model(batch)
            pred = logits[:, 11:17, :].argmax(dim=-1)
            true = batch[:, 12:18]
            match = (pred == true).all(dim=-1)
            correct += match.sum().item()
            total += batch.shape[0]
    handle.remove()
    return correct / total


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    N = eval_tokens.shape[0]
    print(f"Eval set: {N} problems, device={device}")

    all_models = get_converged_models("models")
    want = ["baseline_seed0", "baseline_seed1"]
    models = [m for m in all_models if m["model_name"] in want]
    if len(models) != 2:
        print("Need baseline_seed0 and baseline_seed1")
        return

    # Load both, compute native activations at pos 12 layer 3
    native_cache = {}
    for m in models:
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()
        h = compute_activations(mdl, eval_tokens, device)
        native_cache[m["model_name"]] = {"h": h, "mean": h.mean(axis=0),
                                           "R": load_R(m["model_name"], h.shape[1]),
                                           "model_path": m["model_path"]}
        del mdl
        torch.cuda.empty_cache()

    A = "baseline_seed0"
    B = "baseline_seed1"
    h_A = native_cache[A]["h"]
    h_B = native_cache[B]["h"]
    mean_A, mean_B = native_cache[A]["mean"], native_cache[B]["mean"]
    R_A, R_B = native_cache[A]["R"], native_cache[B]["R"]
    d = h_A.shape[1]

    # Load model A for the forward pass
    mdl = ArithmeticTransformer(cfg).to(device)
    sd = torch.load(native_cache[A]["model_path"], map_location=device, weights_only=True)
    mdl.load_state_dict(sd)
    mdl.eval()

    # Baseline (inject A's own h_A → should be identity)
    base_acc = evaluate_with_injection(mdl, eval_tokens, device, h_A)
    print(f"\nBaseline (self-injection, {A}): {base_acc:.4f}")

    results = {"config": {"A": A, "B": B, "N_eval": int(N),
                          "site": SITE, "layer": LAYER, "pos": POS},
               "conditions": {}}

    # CONTROL 1: self-input-shuffle
    # For each problem i, inject A's activation from a DIFFERENT random problem j
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    while (perm == np.arange(N)).any():  # ensure no fixed points
        perm = rng.permutation(N)
    h_shuffled = h_A[perm]
    acc = evaluate_with_injection(mdl, eval_tokens, device, h_shuffled)
    drop = base_acc - acc
    results["conditions"]["self_input_shuffle"] = {"acc": float(acc), "drop": float(drop)}
    print(f"  self_input_shuffle (A's own acts, shuffled across inputs): "
          f"acc={acc:.4f}  drop={drop:+.4f}  (SHOULD hurt — positive control)")

    # CONTROL 2: full-activation swap in ALIGNED frame
    # h_A_new = h_A - h_A_mean + h_B_mean - (h_A - mean_A)@R_A @ R_A.T + (h_B - mean_B)@R_B @ R_A.T
    # Equivalent: convert h_A to ref, replace ENTIRELY with h_B_ref, convert back to A's native frame.
    h_A_ref = (h_A - mean_A) @ R_A
    h_B_ref = (h_B - mean_B) @ R_B
    h_A_fullswap = h_B_ref @ R_A.T + mean_A    # A's native frame but values come entirely from B
    acc_fs = evaluate_with_injection(mdl, eval_tokens, device, h_A_fullswap)
    drop_fs = base_acc - acc_fs
    results["conditions"]["full_aligned_swap"] = {"acc": float(acc_fs), "drop": float(drop_fs)}
    print(f"  full_aligned_swap (A's native → B's full aligned act, no subspace filter): "
          f"acc={acc_fs:.4f}  drop={drop_fs:+.4f}")

    # CONTROL 3: RAW NATIVE swap (no Procrustes alignment)
    # Literally inject B's RAW native activation into A's forward pass.
    acc_rn = evaluate_with_injection(mdl, eval_tokens, device, h_B)
    drop_rn = base_acc - acc_rn
    results["conditions"]["raw_native_swap"] = {"acc": float(acc_rn), "drop": float(drop_rn)}
    print(f"  raw_native_swap   (A's native → B's raw native, NO alignment): "
          f"acc={acc_rn:.4f}  drop={drop_rn:+.4f}  (SHOULD hurt — alignment control)")

    # CONTROL 4: shuffled ACROSS MODELS — take B's aligned activations,
    # shuffle them, then put them in A's frame. Like self_input_shuffle but
    # with cross-model source. Should also hurt.
    h_B_ref_shuffled = h_B_ref[perm]
    h_crossshuffled = h_B_ref_shuffled @ R_A.T + mean_A
    acc_cs = evaluate_with_injection(mdl, eval_tokens, device, h_crossshuffled)
    drop_cs = base_acc - acc_cs
    results["conditions"]["cross_model_input_shuffle"] = {"acc": float(acc_cs), "drop": float(drop_cs)}
    print(f"  cross_model_shuffle (B's aligned acts, shuffled across inputs, back to A): "
          f"acc={acc_cs:.4f}  drop={drop_cs:+.4f}  (SHOULD hurt)")

    # CONTROL 5: Zero out (vacuous ablation — replace with zeros, no mean preserved)
    acc_z = evaluate_with_injection(mdl, eval_tokens, device, np.zeros_like(h_A))
    drop_z = base_acc - acc_z
    results["conditions"]["zero_out"] = {"acc": float(acc_z), "drop": float(drop_z)}
    print(f"  zero_out            (destroy entire layer-3 pos 12 activation): "
          f"acc={acc_z:.4f}  drop={drop_z:+.4f}  (SHOULD hurt a lot)")

    del mdl
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")

    print("\nInterpretation guide:")
    print("  If self_input_shuffle hurts a lot and cross-model-same-input swaps show 0 drop,")
    print("  the measurement has dynamic range and the 'universal values' claim is real.")
    print("  If self_input_shuffle shows ~0 drop, layer 3 pos 12 is just insensitive.")


if __name__ == "__main__":
    main()
