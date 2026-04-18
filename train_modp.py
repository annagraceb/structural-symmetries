"""Training loop for mod-p transformer. Adapted from train.py."""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from config_modp import ModelConfigModP, TrainConfigModP
from data_modp import generate_batch
from model import ArithmeticTransformer, freeze_component


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# In mod-p tokens: positions 0,1,2,3,4 = a, +, b, =, c.
# We predict c at position 4 from position 3 (the '=' token).
# So result_logit_pos = 3, target_token_pos = 4.
RESULT_PRED_POS = 3
RESULT_TGT_POS = 4


@torch.no_grad()
def evaluate(model, eval_tokens: torch.Tensor, cfg: ModelConfigModP,
             batch_size: int = 1024) -> float:
    model.eval()
    device = next(model.parameters()).device
    n = eval_tokens.shape[0]
    correct = 0
    for i in range(0, n, batch_size):
        batch = eval_tokens[i:i + batch_size].to(device)
        logits = model(batch)
        pred = logits[:, RESULT_PRED_POS, :].argmax(dim=-1)
        true = batch[:, RESULT_TGT_POS]
        correct += (pred == true).sum().item()
    model.train()
    return correct / n


def train_model(
    model_cfg: ModelConfigModP,
    train_cfg: TrainConfigModP,
    eval_tokens: torch.Tensor,
    save_dir: str,
    device: str = "cuda",
) -> dict:
    set_seed(train_cfg.seed)
    os.makedirs(save_dir, exist_ok=True)

    model = ArithmeticTransformer(model_cfg).to(device)
    if train_cfg.frozen_component:
        n_frozen = freeze_component(model, train_cfg.frozen_component)
        print(f"  Froze {train_cfg.frozen_component} ({n_frozen} tensors)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg.lr,
                                    weight_decay=train_cfg.weight_decay)

    log = {"steps": [], "losses": [], "accuracies": []}
    best_acc = 0.0
    converged = False
    converge_step = None
    start = time.time()
    step = 0

    model.train()
    for step in range(1, train_cfg.max_steps + 1):
        tokens = generate_batch(train_cfg.batch_size, model_cfg, device=device)
        logits = model(tokens)
        pred_logits = logits[:, RESULT_PRED_POS, :]
        targets = tokens[:, RESULT_TGT_POS]
        loss = F.cross_entropy(pred_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.log_every == 0:
            log["steps"].append(step)
            log["losses"].append(loss.item())

        if step % train_cfg.eval_every == 0:
            acc = evaluate(model, eval_tokens, model_cfg)
            log["accuracies"].append({"step": step, "accuracy": acc})
            elapsed = time.time() - start
            print(f"  Step {step:>6d} | loss {loss.item():.4f} | acc {acc:.4f} | {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            if acc >= train_cfg.target_accuracy and not converged:
                converged = True
                converge_step = step
                print(f"  >>> Converged at step {step} (acc={acc:.4f})")

        if converged and step >= converge_step + train_cfg.eval_every * 2:
            break
        # Early termination for hopeless models
        if step == 5000 and best_acc < 0.4:
            print(f"  >>> Early termination: acc={best_acc:.4f} at step 5000")
            break

    total_time = time.time() - start
    final_acc = evaluate(model, eval_tokens, model_cfg)
    if final_acc >= best_acc:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    meta = {
        "seed": train_cfg.seed,
        "frozen_component": train_cfg.frozen_component,
        "converged": converged,
        "converge_step": converge_step,
        "final_accuracy": final_acc,
        "best_accuracy": max(best_acc, final_acc),
        "total_steps": step,
        "total_time_seconds": total_time,
        "p": model_cfg.p,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(log, f)
    return meta
