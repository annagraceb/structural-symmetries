"""Step 8: Train fresh models with auxiliary loss conditions A-D'."""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig
from model import ArithmeticTransformer, CarryHead
from data import generate_batch, load_eval_set, generate_eval_set
from train import set_seed, evaluate


def linear_cka_batch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Differentiable linear CKA on a mini-batch.

    X, Y: [batch, d_model]
    Returns scalar CKA (differentiable).
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X

    num = (YtX ** 2).sum()
    denom = torch.sqrt((XtX ** 2).sum() * (YtY ** 2).sum())
    return num / (denom + 1e-8)


def procrustes_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute orthogonal Procrustes rotation R such that X @ R ≈ Y.

    Returns R: [d, d].
    """
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    M = Y_c.T @ X_c
    U, S, Vt = torch.linalg.svd(M)
    R = Vt.T @ U.T
    return R


def train_auxiliary(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    eval_tokens: torch.Tensor,
    save_dir: str,
    condition: str,
    shared_dirs: np.ndarray = None,
    reference_acts: np.ndarray = None,
    cka_eval_tokens: torch.Tensor = None,
    alpha: float = 0.01,
    target_layer: int = 2,
    target_pos: int = 14,
    device: str = "cuda",
) -> dict:
    """Train a model with one of the auxiliary loss conditions.

    Args:
        cka_eval_tokens: Fixed eval inputs for C1 condition. CKA requires matching
            inputs — reference_acts[i] must be the reference model's activation for
            cka_eval_tokens[i]. If None, C1 falls back to C2 behavior.

    Conditions:
        A: baseline (no auxiliary loss)
        B: carry head (ground truth carry labels)
        C1: CKA-based geometric loss (requires matching inputs)
        C2: Procrustes projection loss
        D: random subspace (same dim as shared_dirs)
        D_prime: bottom eigenvectors
    """
    set_seed(train_cfg.seed)
    os.makedirs(save_dir, exist_ok=True)

    model = ArithmeticTransformer(model_cfg).to(device)
    carry_head = None
    if condition == "B":
        carry_head = CarryHead(model_cfg.d_model).to(device)

    params = list(model.parameters())
    if carry_head is not None:
        params += list(carry_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)

    # Prepare reference data for geometric conditions
    ref_acts_t = None
    shared_dirs_t = None
    cached_R = None
    R_recompute_interval = 100

    if condition in ("C1", "C2", "D", "D_prime") and reference_acts is not None:
        ref_acts_t = torch.tensor(reference_acts, dtype=torch.float32, device=device)
    if condition in ("C2", "D", "D_prime") and shared_dirs is not None:
        shared_dirs_t = torch.tensor(shared_dirs, dtype=torch.float32, device=device)

    # Training log
    log = {"steps": [], "losses": [], "aux_losses": [], "accuracies": []}
    best_acc = 0.0
    converged = False
    converge_step = None
    start_time = time.time()

    model.train()
    for step in range(1, train_cfg.max_steps + 1):
        tokens, carries = generate_batch(train_cfg.batch_size, model_cfg, device=device)

        need_hidden = condition in ("B", "C1", "C2", "D", "D_prime")
        if need_hidden:
            logits, hiddens = model(tokens, return_all_hiddens=True)
        else:
            logits = model(tokens)

        # Main loss
        result_logits = logits[:, 11:17, :]
        result_targets = tokens[:, 12:18]
        main_loss = F.cross_entropy(result_logits.reshape(-1, model_cfg.vocab_size),
                                     result_targets.reshape(-1))

        # Auxiliary loss
        aux_loss = torch.tensor(0.0, device=device)

        if condition == "B" and carry_head is not None:
            carry_hidden = hiddens[train_cfg.carry_loss_layer][:, 12:17, :]
            carry_logits = carry_head(carry_hidden)
            aux_loss = F.binary_cross_entropy_with_logits(carry_logits, carries)

        elif condition == "C1" and ref_acts_t is not None and cka_eval_tokens is not None:
            # CKA-based loss: CKA requires the SAME inputs for both models.
            # Sample a subset of the eval set, run the current model on it,
            # and compare with precomputed reference activations.
            cka_batch_size = min(256, ref_acts_t.shape[0])
            idx = torch.randint(0, ref_acts_t.shape[0], (cka_batch_size,), device=device)
            h_ref = ref_acts_t[idx]
            cka_batch = cka_eval_tokens[idx.cpu()].to(device)
            _, cka_hiddens = model(cka_batch, return_all_hiddens=True)
            h_current = cka_hiddens[target_layer][:, target_pos, :]
            aux_loss = -linear_cka_batch(h_current, h_ref)

        elif condition in ("C2", "D", "D_prime") and shared_dirs_t is not None:
            h_current = hiddens[target_layer][:, target_pos, :]  # [B, d]
            B = h_current.shape[0]

            # Online Procrustes alignment (recompute every N steps)
            if cached_R is None or step % R_recompute_interval == 0:
                with torch.no_grad():
                    idx = torch.randint(0, ref_acts_t.shape[0], (B,), device=device)
                    h_ref = ref_acts_t[idx]
                    cached_R = procrustes_rotation(h_current.detach(), h_ref)

            h_aligned = (h_current - h_current.mean(dim=0, keepdim=True)) @ cached_R
            # Projection onto shared subspace
            proj = h_aligned @ shared_dirs_t.T @ shared_dirs_t  # [B, d]
            aux_loss = F.mse_loss(h_aligned, proj)

        total_loss = main_loss + alpha * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
        optimizer.step()

        # Logging
        if step % train_cfg.log_every == 0:
            log["steps"].append(step)
            log["losses"].append(main_loss.item())
            log["aux_losses"].append(aux_loss.item())

        # Evaluation
        if step % train_cfg.eval_every == 0:
            acc = evaluate(model, eval_tokens, model_cfg)
            log["accuracies"].append({"step": step, "accuracy": acc})
            elapsed = time.time() - start_time

            print(f"  Step {step:>6d} | loss {main_loss.item():.4f} "
                  f"| aux {aux_loss.item():.4f} | acc {acc:.4f} | {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

            if acc >= train_cfg.target_accuracy and not converged:
                converged = True
                converge_step = step
                print(f"  >>> Converged at step {step}")

        if converged and step >= converge_step + train_cfg.eval_every * 2:
            break

        # Early termination for hopeless models
        if step == 10000 and best_acc < 0.5:
            print(f"  >>> Early termination: acc={best_acc:.4f} at step 10000")
            break

    total_time = time.time() - start_time
    final_acc = evaluate(model, eval_tokens, model_cfg)

    metadata = {
        "condition": condition,
        "alpha": alpha,
        "seed": train_cfg.seed,
        "converged": converged,
        "converge_step": converge_step,
        "final_accuracy": final_acc,
        "best_accuracy": max(best_acc, final_acc),
        "total_steps": step,
        "total_time_seconds": total_time,
        "target_layer": target_layer,
        "target_pos": target_pos,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(log, f)

    return metadata


def main():
    model_cfg = ModelConfig()
    results_dir = "results"
    aux_dir = "models_auxiliary"
    eval_dir = "eval_sets"

    # Load eval set
    eval_data = load_eval_set(os.path.join(eval_dir, "convergence_eval"))
    eval_tokens = eval_data["tokens"]

    # Load shared directions from Step 5
    shared_dirs_path = os.path.join(results_dir, "shared_dirs.npy")
    if not os.path.exists(shared_dirs_path):
        print("ERROR: Run analysis.py first to extract shared directions")
        return

    shared_dirs = np.load(shared_dirs_path)  # [n_dirs, 128]
    print(f"Loaded shared directions: {shared_dirs.shape}")

    d = shared_dirs.shape[1]
    n_dirs = shared_dirs.shape[0]

    # Random directions for condition D
    rng = np.random.RandomState(12345)
    random_dirs = rng.randn(n_dirs, d).astype(np.float32)
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

    # Bottom eigenvectors for D' condition: load from Step 5 extraction
    bottom_dirs_path = os.path.join(results_dir, "bottom_dirs.npy")
    if os.path.exists(bottom_dirs_path):
        bottom_dirs = np.load(bottom_dirs_path)  # [n_dirs, 128]
        print(f"Loaded bottom eigenvectors: {bottom_dirs.shape}")
    else:
        # Fallback: generate random directions orthogonal to shared dirs
        print("Warning: bottom_dirs.npy not found, using orthogonal complement")
        Q, _ = np.linalg.qr(shared_dirs.T)
        null_space = np.eye(d) - Q @ Q.T
        bottom_dirs_raw = rng.randn(n_dirs, d).astype(np.float32)
        bottom_dirs = (bottom_dirs_raw @ null_space)
        bottom_dirs /= (np.linalg.norm(bottom_dirs, axis=1, keepdims=True) + 1e-8)

    # Load reference activations for geometric conditions
    # Use the reference model's activations at the selected site
    sites_path = os.path.join(results_dir, "selected_sites.json")
    if os.path.exists(sites_path):
        with open(sites_path) as f:
            selected_sites = json.load(f)
        top_site = selected_sites[0]
        target_layer = top_site["layer"]
        target_pos = top_site["position_idx"]
    else:
        target_layer = 2
        target_pos = 14  # result position 2

    # Find reference model activations
    ref_act_path = None
    for entry in sorted(os.listdir("activations")):
        if "baseline" in entry and "seed0" in entry:
            ref_act_path = os.path.join("activations", entry,
                                         f"layer{target_layer}_stratified.npy")
            break
    if ref_act_path is None:
        # Fallback: use first available
        for entry in sorted(os.listdir("activations")):
            path = os.path.join("activations", entry,
                                 f"layer{target_layer}_stratified.npy")
            if os.path.exists(path):
                ref_act_path = path
                break

    reference_acts = np.load(ref_act_path)[:, target_pos, :]  # [2000, 128]
    print(f"Loaded reference activations: {reference_acts.shape}")
    print(f"Target site: layer {target_layer}, position {target_pos}")

    # Load stratified eval tokens for C1 CKA (needs matching inputs)
    strat_data = load_eval_set(os.path.join(eval_dir, "stratified_2000"))
    cka_eval_tokens = strat_data["tokens"]  # [2000, 18]

    # ---- Phase 1: Alpha sweep (2 seeds per alpha) ----
    print("\n" + "=" * 60)
    print("PHASE 1: Alpha Sweep")
    print("=" * 60)

    conditions = {
        "A": {"shared": None, "ref": None},
        "B": {"shared": None, "ref": None},
        "C1": {"shared": shared_dirs, "ref": reference_acts},
        "C2": {"shared": shared_dirs[:5], "ref": reference_acts},
        "D": {"shared": random_dirs[:5], "ref": reference_acts},
        "D_prime": {"shared": bottom_dirs[:5], "ref": reference_acts},
    }

    alphas = [0.1, 0.01, 0.001]
    sweep_results = {}

    for cond_name, cond_data in conditions.items():
        if cond_name in ("A", "B"):
            # These don't use alpha
            alpha_list = [0.1]  # placeholder
        else:
            alpha_list = alphas

        for alpha in alpha_list:
            for seed in range(2):
                run_name = f"{cond_name}_alpha{alpha}_seed{seed}"
                save_path = os.path.join(aux_dir, "sweep", run_name)

                if os.path.exists(os.path.join(save_path, "metadata.json")):
                    with open(os.path.join(save_path, "metadata.json")) as f:
                        meta = json.load(f)
                    print(f"  [SKIP] {run_name}: "
                          f"{'converged' if meta['converged'] else 'failed'} "
                          f"at step {meta.get('converge_step', 'N/A')}")
                    sweep_results[run_name] = meta
                    continue

                print(f"\n  Training {run_name}...")
                tcfg = TrainConfig(
                    seed=seed + 100,
                    use_carry_head=(cond_name == "B"),
                    carry_loss_weight=alpha if cond_name == "B" else 0.1,
                )

                meta = train_auxiliary(
                    model_cfg, tcfg, eval_tokens, save_path,
                    condition=cond_name,
                    shared_dirs=cond_data["shared"],
                    reference_acts=cond_data["ref"],
                    cka_eval_tokens=cka_eval_tokens,
                    alpha=alpha,
                    target_layer=target_layer,
                    target_pos=target_pos,
                )
                sweep_results[run_name] = meta

    # Select best alpha per condition
    best_alphas = {}
    for cond_name in conditions:
        if cond_name in ("A", "B"):
            best_alphas[cond_name] = 0.1
            continue
        best_alpha = None
        best_step = float("inf")
        for alpha in alphas:
            steps = []
            for seed in range(2):
                key = f"{cond_name}_alpha{alpha}_seed{seed}"
                if key in sweep_results and sweep_results[key]["converged"]:
                    steps.append(sweep_results[key]["converge_step"])
            if steps:
                avg_step = np.mean(steps)
                if avg_step < best_step:
                    best_step = avg_step
                    best_alpha = alpha
        best_alphas[cond_name] = best_alpha or 0.01
        print(f"  Best alpha for {cond_name}: {best_alphas[cond_name]}")

    # ---- Phase 2: Full evaluation (5 seeds at best alpha) ----
    print("\n" + "=" * 60)
    print("PHASE 2: Full Evaluation")
    print("=" * 60)

    eval_results = {}
    for cond_name, cond_data in conditions.items():
        alpha = best_alphas[cond_name]
        for seed in range(5):
            run_name = f"{cond_name}_eval_seed{seed}"
            save_path = os.path.join(aux_dir, "eval", run_name)

            if os.path.exists(os.path.join(save_path, "metadata.json")):
                with open(os.path.join(save_path, "metadata.json")) as f:
                    meta = json.load(f)
                print(f"  [SKIP] {run_name}: "
                      f"{'converged' if meta['converged'] else 'failed'}")
                eval_results[run_name] = meta
                continue

            print(f"\n  Training {run_name} (alpha={alpha})...")
            tcfg = TrainConfig(
                seed=seed + 200,
                use_carry_head=(cond_name == "B"),
                carry_loss_weight=alpha if cond_name == "B" else 0.1,
            )

            meta = train_auxiliary(
                model_cfg, tcfg, eval_tokens, save_path,
                condition=cond_name,
                shared_dirs=cond_data["shared"],
                reference_acts=cond_data["ref"],
                alpha=alpha,
                target_layer=target_layer,
                target_pos=target_pos,
            )
            eval_results[run_name] = meta

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("STEP 8 RESULTS")
    print("=" * 60)

    summary = {}
    for cond_name in conditions:
        steps = []
        accs = []
        for seed in range(5):
            key = f"{cond_name}_eval_seed{seed}"
            if key in eval_results:
                meta = eval_results[key]
                if meta["converged"]:
                    steps.append(meta["converge_step"])
                accs.append(meta["best_accuracy"])

        summary[cond_name] = {
            "mean_converge_step": float(np.mean(steps)) if steps else None,
            "std_converge_step": float(np.std(steps)) if len(steps) > 1 else None,
            "n_converged": len(steps),
            "n_total": 5,
            "mean_accuracy": float(np.mean(accs)) if accs else None,
            "best_alpha": best_alphas[cond_name],
        }
        if steps:
            print(f"  {cond_name}: steps_to_99={np.mean(steps):.0f}±{np.std(steps):.0f} "
                  f"({len(steps)}/5 converged)")
        else:
            print(f"  {cond_name}: did not converge")

    with open(os.path.join(results_dir, "step8_auxiliary.json"), "w") as f:
        json.dump({"sweep": sweep_results, "eval": eval_results, "summary": summary},
                  f, indent=2, default=str)

    # Decision
    print("\n  Decision:")
    if summary.get("C2", {}).get("mean_converge_step") and summary.get("D", {}).get("mean_converge_step"):
        c2_steps = summary["C2"]["mean_converge_step"]
        d_steps = summary["D"]["mean_converge_step"]
        dp_steps = summary.get("D_prime", {}).get("mean_converge_step")

        if c2_steps < d_steps * 0.9:
            print(f"  C2 ({c2_steps:.0f}) < D ({d_steps:.0f}): "
                  f"Shared directions provide value beyond regularization!")
        elif c2_steps < d_steps * 1.1:
            print(f"  C2 ({c2_steps:.0f}) ≈ D ({d_steps:.0f}): "
                  f"Any auxiliary loss helps, direction doesn't matter much")
        else:
            print(f"  C2 ({c2_steps:.0f}) > D ({d_steps:.0f}): "
                  f"Shared directions may hurt vs random")

        if dp_steps and c2_steps < dp_steps * 0.9:
            print(f"  C2 ({c2_steps:.0f}) < D' ({dp_steps:.0f}): "
                  f"Shared directions better than bottom eigenvectors!")


if __name__ == "__main__":
    main()
