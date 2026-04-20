"""Phase 4 — train permuted-label 8L null zoo.

Per PHASE4_A4_PREREGISTRATION.md:
- 10 × 8L models on 5-digit addition, with per-model seed-indexed digit
  bijection P_s applied to each result token.
- Each model sees targets (P_s(r_5), P_s(r_4), ..., P_s(r_0)) for result tokens.
- Training budget: 20,000 steps (fixed).
- Inclusion: best-accuracy ≥ 0.20 within budget; drop models below.

Null intent: the per-model bijection destroys cross-model readout alignment
at N-1 (layer 7, result-digit-0). If the A4 rule fires at `primary` but also
at this null, the structured-vs-random gap reflects a pipeline artifact
rather than task-specific distributed redundancy.
"""

import dataclasses
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig, PLUS_TOKEN, EQUALS_TOKEN
from config_deep8 import ModelConfigDeep8, TrainConfigDeep8
from data import int_to_digits, compute_carries, load_eval_set
from model import ArithmeticTransformer, freeze_component


MODELS_DIR = "models_deep8_permuted"
N_MODELS = 10
STEP_BUDGET = 20_000
ACCEPT_ACC = 0.20


def seed_permutation(seed: int) -> list[int]:
    """Return a fixed bijection P: digits 0..9 -> 0..9 for this seed."""
    rng = random.Random(12345 + seed * 7919)
    digits = list(range(10))
    rng.shuffle(digits)
    # Ensure P is not the identity; re-shuffle if so.
    while digits == list(range(10)):
        rng.shuffle(digits)
    return digits


def make_tokens_permuted(a: int, b: int, cfg: ModelConfig, perm: list[int]) -> list[int]:
    """Addition sequence with result digits mapped through `perm`."""
    a_digits = int_to_digits(a, cfg.n_digits)
    b_digits = int_to_digits(b, cfg.n_digits)
    r = int_to_digits(a + b, cfg.n_result_digits)
    r_perm = [perm[d] for d in r]
    tokens = a_digits + [PLUS_TOKEN] + b_digits + [EQUALS_TOKEN] + r_perm
    assert len(tokens) == cfg.max_seq_len
    return tokens


def generate_batch_permuted(batch_size: int, cfg: ModelConfig, perm: list[int],
                             device: str = "cuda"):
    max_val = 10 ** cfg.n_digits
    tokens_list = []
    carries_list = []
    for _ in range(batch_size):
        a = random.randint(0, max_val - 1)
        b = random.randint(0, max_val - 1)
        tokens_list.append(make_tokens_permuted(a, b, cfg, perm))
        carries_list.append(compute_carries(a, b, cfg.n_digits))
    tokens = torch.tensor(tokens_list, dtype=torch.long, device=device)
    carries = torch.tensor(carries_list, dtype=torch.float, device=device)
    return tokens, carries


def permute_eval_tokens(eval_tokens: torch.Tensor, cfg: ModelConfig,
                         perm: list[int]) -> torch.Tensor:
    """Apply per-model permutation to the result-token positions of a fixed eval set."""
    result_start = cfg.result_start_pos
    out = eval_tokens.clone()
    # Only permute the result digit tokens (10 possible digits). Non-digit tokens
    # like PLUS/EQUALS should not occur in the result region, but guard anyway.
    result_region = out[:, result_start:]
    perm_tensor = torch.tensor(perm, dtype=result_region.dtype)
    mask = result_region < 10
    mapped = perm_tensor[result_region.clamp(max=9)]
    out[:, result_start:] = torch.where(mask, mapped, result_region)
    return out


@torch.no_grad()
def evaluate_permuted(model: ArithmeticTransformer, eval_tokens_perm: torch.Tensor,
                       cfg: ModelConfig, batch_size: int = 512) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    for i in range(0, eval_tokens_perm.shape[0], batch_size):
        batch = eval_tokens_perm[i:i + batch_size].to(device)
        logits = model(batch)
        pred_logits = logits[:, 11:17, :]
        pred = pred_logits.argmax(dim=-1)
        true = batch[:, 12:18]
        match = (pred == true).all(dim=-1)
        correct += match.sum().item()
        total += batch.shape[0]
    model.train()
    return correct / total


def train_one_permuted(seed: int, cfg: ModelConfigDeep8, eval_tokens: torch.Tensor,
                        save_dir: str, device: str) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    perm = seed_permutation(seed)
    eval_perm = permute_eval_tokens(eval_tokens, cfg, perm)

    model = ArithmeticTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  seed={seed} perm={perm} params={n_params:,}")

    log = {"steps": [], "losses": [], "accuracies": []}
    best_acc = 0.0
    start = time.time()
    model.train()
    for step in range(1, STEP_BUDGET + 1):
        tokens, _ = generate_batch_permuted(512, cfg, perm, device=device)
        logits = model(tokens)
        result_logits = logits[:, 11:17, :]
        result_targets = tokens[:, 12:18]
        loss = F.cross_entropy(result_logits.reshape(-1, cfg.vocab_size),
                               result_targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0:
            log["steps"].append(step)
            log["losses"].append(float(loss.item()))
        if step % 1000 == 0:
            acc = evaluate_permuted(model, eval_perm, cfg)
            log["accuracies"].append({"step": step, "accuracy": acc})
            elapsed = time.time() - start
            print(f"    step {step:>6d} | loss {loss.item():.4f} | acc {acc:.4f} | {elapsed:.0f}s")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        # Stop rule: if at step 10k we are still below 0.03 accuracy, abort early.
        if step == 10_000 and best_acc < 0.03:
            print(f"    >>> Early abort at step 10k (best_acc={best_acc:.4f})")
            break

    final_acc = evaluate_permuted(model, eval_perm, cfg)
    if final_acc > best_acc:
        best_acc = final_acc
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    total_time = time.time() - start

    meta = {
        "seed": seed,
        "permutation": perm,
        "best_accuracy": best_acc,
        "final_accuracy": final_acc,
        "converged": bool(best_acc >= ACCEPT_ACC),
        "total_steps": step,
        "total_time_seconds": total_time,
        "n_params_total": n_params,
        "n_params_trainable": n_params,
        "frozen_component": None,
        "use_carry_head": False,
        "converge_step": None,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(log, f)
    return meta


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfigDeep8()
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    print(f"n_layers={cfg.n_layers} device={device} eval_size={eval_tokens.shape[0]}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    summary = {"models": []}
    summary_path = os.path.join(MODELS_DIR, "_summary.json")
    for seed in range(N_MODELS):
        mname = f"permuted_seed{seed}"
        save_dir = os.path.join(MODELS_DIR, mname)
        if os.path.exists(os.path.join(save_dir, "metadata.json")):
            print(f"[{seed+1}/{N_MODELS}] {mname}: already trained")
            with open(os.path.join(save_dir, "metadata.json")) as f:
                summary["models"].append(json.load(f) | {"model_name": mname})
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            continue
        print(f"\n[{seed+1}/{N_MODELS}] {mname}")
        meta = train_one_permuted(seed, cfg, eval_tokens, save_dir, device)
        meta["model_name"] = mname
        summary["models"].append(meta)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    n_accept = sum(1 for m in summary["models"] if m.get("converged"))
    print(f"\nDone. Accepted {n_accept}/{N_MODELS} (acc >= {ACCEPT_ACC})")


if __name__ == "__main__":
    main()
