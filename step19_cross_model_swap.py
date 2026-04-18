"""Step 19: Cross-model shared-subspace swap (Gemini's Q2 experiment).

Test of "universal redundancy": if the shared subspace at layer 3 is truly
universal across independently-trained zoo models, we should be able to swap
ONE model's shared-subspace component at layer 3 for ANOTHER model's component
on the same input, and the first model should still compute the task correctly.

Swap protocol (for a pair of models A, B, at layer 3, position 12):
  1. Align both models' layer-3 activations to reference frame.
  2. In reference frame, compute the shared-subspace projector P_S.
  3. For each input x: h_A_new_ref(x) = h_A_ref(x) + P_S @ (h_B_ref(x) - h_A_ref(x))
     (i.e., A's activation except that the shared-subspace piece comes from B)
  4. Map back to A's native basis, inject into A's forward pass at layer 3
     position 12, measure accuracy.

Controls run in parallel:
  - complement_swap: same swap but for complement subspace (should hurt — complement
    is where model-specific computation lives).
  - random_swap: swap a whitened-random 10-d subspace (should hurt ~0; random is
    dominated by nullspace).
  - self_swap (sanity): same model on both sides; should be 0 drop exactly.

If universal-redundancy is correct:
  - shared_swap accuracy drop ≈ 0 (or small)
  - complement_swap accuracy drop is large (complement is model-specific)
  - random_swap drop tiny
  - self_swap drop = 0

Outputs results/p1/cross_model_swap.json.
"""

import json
import os
import time

import numpy as np
import torch

from config import ModelConfig
from model import ArithmeticTransformer
from collect_activations import get_converged_models
from data import load_eval_set
from step9_p1 import extract_with_max_dims, complement_top_k, K_PCA


RESULTS_DIR = "results"
SITE = "layer3_result_0"
LAYER = 3
POS = 12
K = 10
OUT = os.path.join(RESULTS_DIR, "p1", "cross_model_swap.json")


def load_aligned(site_name: str):
    d = os.path.join(RESULTS_DIR, f"aligned_{site_name}")
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(d, f))
    return out


def load_R(model_name: str, site_name: str, d: int) -> np.ndarray:
    p = os.path.join(RESULTS_DIR, f"R_{model_name}_{site_name}.npy")
    if os.path.exists(p):
        return np.load(p)
    return np.eye(d)


def compute_native_mean(model, eval_tokens, layer: int, pos: int, device: str) -> np.ndarray:
    """Compute the per-dimension mean of layer-`layer` activations at position
    `pos` over the eval set (needed for the Procrustes inverse map)."""
    model.eval()
    sums = None
    n = 0
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], 256):
            batch = eval_tokens[i:i+256].to(device)
            _, hiddens = model(batch, return_all_hiddens=True)
            h = hiddens[layer][:, pos, :].cpu().numpy()
            if sums is None:
                sums = h.sum(axis=0)
            else:
                sums += h.sum(axis=0)
            n += h.shape[0]
    return sums / n


def compute_native_activations(model, eval_tokens, layer: int, pos: int,
                                 device: str) -> np.ndarray:
    """[N_inputs, d_model] at (layer, pos) in native frame."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], 256):
            batch = eval_tokens[i:i+256].to(device)
            _, hiddens = model(batch, return_all_hiddens=True)
            chunks.append(hiddens[layer][:, pos, :].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def projection_matrix(V: np.ndarray) -> np.ndarray:
    """P = V^T (V V^T)^{-1} V, for V of shape [k, d]. Projects onto span of rows."""
    VVt_inv = np.linalg.pinv(V @ V.T)
    return V.T @ VVt_inv @ V


def evaluate_with_injection(model, eval_tokens, cfg: ModelConfig, device: str,
                              injected_h: np.ndarray, layer: int, pos: int,
                              batch_size: int = 256) -> float:
    """Evaluate model accuracy, replacing layer-`layer` activations at position
    `pos` with the corresponding row of `injected_h` (shape [N, d_model]).

    Accuracy = all 6 result digits correct.
    """
    injected_t = torch.tensor(injected_h, dtype=torch.float32, device=device)
    cur_start = [0]

    def hook_fn(module, inputs, output):
        b = output.shape[0]
        start = cur_start[0]
        output[:, pos, :] = injected_t[start:start+b]
        return output

    handle = model.layers[layer].register_forward_hook(hook_fn)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, eval_tokens.shape[0], batch_size):
            cur_start[0] = i
            batch = eval_tokens[i:i+batch_size].to(device)
            logits = model(batch)
            pred = logits[:, 11:17, :].argmax(dim=-1)
            true = batch[:, 12:18]
            match = (pred == true).all(dim=-1)
            correct += match.sum().item()
            total += batch.shape[0]
    handle.remove()
    return correct / total


def swap_subspace(h_A_native: np.ndarray, h_B_native: np.ndarray,
                   mean_A: np.ndarray, mean_B: np.ndarray,
                   R_A: np.ndarray, R_B: np.ndarray,
                   V_S_ref: np.ndarray) -> np.ndarray:
    """Compute patched h_A in A's native frame: h_A with its V_S-component
    replaced by B's V_S-component in the reference frame.

    V_S_ref has shape [k, d]; projects on span(V_S_ref).
    Returns [N, d] in A's native basis.
    """
    # Map to reference frame
    h_A_ref = (h_A_native - mean_A) @ R_A  # [N, d]
    h_B_ref = (h_B_native - mean_B) @ R_B
    P_S = projection_matrix(V_S_ref)       # [d, d]

    # Swap shared-subspace component
    h_A_new_ref = h_A_ref + (h_B_ref - h_A_ref) @ P_S.T

    # Map back to A's native frame (R_A orthogonal so R_A^-1 = R_A^T)
    h_A_new_native = h_A_new_ref @ R_A.T + mean_A
    return h_A_new_native


def main():
    cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Main zoo, d_model={cfg.d_model}, device={device}")

    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    print(f"Eval set: {eval_tokens.shape[0]} problems")

    # Extract shared and complement subspaces (reference frame)
    aligned = load_aligned(SITE)
    ex = extract_with_max_dims(aligned, max_dims=K, eps_scale=1e-8, k_pca=K_PCA)
    V_S_ref = ex["shared_dirs"]       # [K, d]
    C_total = ex["C_total"]
    V_C_ref = complement_top_k(V_S_ref, C_total, k=K)
    d = ex["d"]

    # Whitened random subspace (in ref frame; for random control)
    from step9_p1 import whitened_random_subspace
    rng_legacy = np.random.RandomState(123)   # legacy API required by that helper
    V_R_ref = whitened_random_subspace(K, d, ex["C_total_plus_ridge"], rng_legacy).astype(np.float32)

    # Use all 3 baselines + 3 freeze-variants (maximally different training).
    # Freezing different components should force different internal computations
    # if there's any computational idiosyncrasy — this should make the swap
    # harder if it's going to fail anywhere.
    all_models = get_converged_models("models")
    baselines = [m for m in all_models if m["frozen_component"] is None]
    # Pick freeze variants spanning different components
    want_freezes = ["freeze_embed_seed0", "freeze_layer0.mlp_seed0",
                    "freeze_layer2.attn_seed0"]
    freezes = [m for m in all_models if m["model_name"] in want_freezes]
    selected = baselines + freezes
    print(f"\nModels for swap (baselines + freeze variants):")
    for m in selected:
        print(f"  {m['model_name']}  (frozen={m['frozen_component']})")
    baselines = selected  # reuse the variable name below

    # For each baseline, load once, compute native activations + mean
    native_cache = {}
    for m in baselines:
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(m["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()
        h_native = compute_native_activations(mdl, eval_tokens, LAYER, POS, device)
        mean_native = h_native.mean(axis=0)
        R = load_R(m["model_name"], SITE, d)
        native_cache[m["model_name"]] = {
            "h_native": h_native,
            "mean_native": mean_native,
            "R": R,
            "model_path": m["model_path"],
        }
        del mdl
        torch.cuda.empty_cache()
        print(f"  cached: {m['model_name']}  ({h_native.shape})")

    # For each ordered pair (A, B), run A's forward pass with A's layer-3 position-12
    # activation replaced per the swap.
    results = {"config": {"site": SITE, "layer": LAYER, "pos": POS, "k": K,
                          "n_eval": int(eval_tokens.shape[0])},
               "pairs": []}

    for A in baselines:
        name_A = A["model_name"]
        # Need model A to run forward with the injection
        mdl = ArithmeticTransformer(cfg).to(device)
        sd = torch.load(A["model_path"], map_location=device, weights_only=True)
        mdl.load_state_dict(sd)
        mdl.eval()

        # Baseline accuracy (no injection)
        base_acc = evaluate_with_injection(
            mdl, eval_tokens, cfg, device,
            native_cache[name_A]["h_native"],   # inject the model's own h → no change
            LAYER, POS,
        )
        print(f"\n{name_A} baseline (self-injection sanity): {base_acc:.4f}")

        for B in baselines:
            name_B = B["model_name"]
            if name_B == name_A:
                # Self-swap: should be identity
                self_swap = swap_subspace(
                    native_cache[name_A]["h_native"], native_cache[name_B]["h_native"],
                    native_cache[name_A]["mean_native"], native_cache[name_B]["mean_native"],
                    native_cache[name_A]["R"], native_cache[name_B]["R"],
                    V_S_ref,
                )
                acc_self = evaluate_with_injection(mdl, eval_tokens, cfg, device,
                                                    self_swap, LAYER, POS)
                results["pairs"].append({
                    "A": name_A, "B": name_B, "kind": "self_swap",
                    "accuracy": float(acc_self), "drop": float(base_acc - acc_self),
                })
                print(f"  self_swap (A=A):     acc={acc_self:.4f}")
                continue

            # Shared swap
            swap_shared = swap_subspace(
                native_cache[name_A]["h_native"], native_cache[name_B]["h_native"],
                native_cache[name_A]["mean_native"], native_cache[name_B]["mean_native"],
                native_cache[name_A]["R"], native_cache[name_B]["R"],
                V_S_ref,
            )
            acc_shared = evaluate_with_injection(mdl, eval_tokens, cfg, device,
                                                  swap_shared, LAYER, POS)

            # Complement swap
            swap_comp = swap_subspace(
                native_cache[name_A]["h_native"], native_cache[name_B]["h_native"],
                native_cache[name_A]["mean_native"], native_cache[name_B]["mean_native"],
                native_cache[name_A]["R"], native_cache[name_B]["R"],
                V_C_ref,
            )
            acc_comp = evaluate_with_injection(mdl, eval_tokens, cfg, device,
                                                swap_comp, LAYER, POS)

            # Random swap
            swap_rand = swap_subspace(
                native_cache[name_A]["h_native"], native_cache[name_B]["h_native"],
                native_cache[name_A]["mean_native"], native_cache[name_B]["mean_native"],
                native_cache[name_A]["R"], native_cache[name_B]["R"],
                V_R_ref,
            )
            acc_rand = evaluate_with_injection(mdl, eval_tokens, cfg, device,
                                                swap_rand, LAYER, POS)

            drop_s = base_acc - acc_shared
            drop_c = base_acc - acc_comp
            drop_r = base_acc - acc_rand

            print(f"  {name_A} ← {name_B}:")
            print(f"    shared_swap:     acc={acc_shared:.4f}  drop={drop_s:+.4f}")
            print(f"    complement_swap: acc={acc_comp:.4f}  drop={drop_c:+.4f}")
            print(f"    random_swap:     acc={acc_rand:.4f}  drop={drop_r:+.4f}")

            results["pairs"].append({
                "A": name_A, "B": name_B, "baseline_acc": float(base_acc),
                "shared_swap_acc": float(acc_shared), "shared_swap_drop": float(drop_s),
                "complement_swap_acc": float(acc_comp), "complement_swap_drop": float(drop_c),
                "random_swap_acc": float(acc_rand), "random_swap_drop": float(drop_r),
            })

        del mdl
        torch.cuda.empty_cache()

    # Summary
    drops = {
        "shared_swap": [p["shared_swap_drop"] for p in results["pairs"] if "shared_swap_drop" in p],
        "complement_swap": [p["complement_swap_drop"] for p in results["pairs"] if "complement_swap_drop" in p],
        "random_swap": [p["random_swap_drop"] for p in results["pairs"] if "random_swap_drop" in p],
    }

    def mean_ci(xs):
        if not xs:
            return 0, 0, 0
        rng2 = np.random.default_rng(42)
        arr = np.asarray(xs)
        boots = np.array([arr[rng2.integers(0, len(arr), len(arr))].mean() for _ in range(1000)])
        return float(arr.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    summary = {}
    for k, xs in drops.items():
        m, lo, hi = mean_ci(xs)
        summary[k] = {"mean_drop": m, "ci": [lo, hi], "n": len(xs)}

    results["summary"] = summary

    print("\n" + "=" * 75)
    print(f"CROSS-MODEL SWAP SUMMARY ({len(drops['shared_swap'])} ordered pairs)")
    print("=" * 75)
    for k, s in summary.items():
        print(f"  {k:>18}: mean drop = {s['mean_drop']:.4f}  95% CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]  (n={s['n']})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
