"""Training loop for the arithmetic transformer."""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig
from model import ArithmeticTransformer, CarryHead, freeze_component
from data import generate_batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: ArithmeticTransformer, eval_tokens: torch.Tensor,
             cfg: ModelConfig, batch_size: int = 512) -> float:
    """Compute exact-match accuracy on a fixed eval set."""
    model.eval()
    device = next(model.parameters()).device
    n = eval_tokens.shape[0]
    correct = 0
    total = 0

    for i in range(0, n, batch_size):
        batch = eval_tokens[i:i + batch_size].to(device)
        logits = model(batch)  # [B, seq_len, vocab]

        # Predictions at positions 11..16 predict result tokens at 12..17
        pred_logits = logits[:, 11:17, :]  # [B, 6, vocab]
        pred_tokens = pred_logits.argmax(dim=-1)  # [B, 6]
        true_tokens = batch[:, 12:18]  # [B, 6]

        # Exact match: all 6 result digits correct
        match = (pred_tokens == true_tokens).all(dim=-1)
        correct += match.sum().item()
        total += batch.shape[0]

    model.train()
    return correct / total


def train_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    eval_tokens: torch.Tensor,
    save_dir: str,
    device: str = "cuda",
) -> dict:
    """Train a single model. Returns metadata dict."""
    set_seed(train_cfg.seed)
    os.makedirs(save_dir, exist_ok=True)

    # Build model
    model = ArithmeticTransformer(model_cfg).to(device)

    # Freeze component if specified
    if train_cfg.frozen_component:
        n_frozen = freeze_component(model, train_cfg.frozen_component)
        print(f"  Froze {train_cfg.frozen_component} ({n_frozen} param tensors)")

    # Carry head (optional)
    carry_head = None
    if train_cfg.use_carry_head:
        carry_head = CarryHead(model_cfg.d_model).to(device)

    # Optimizer: only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if carry_head is not None:
        trainable_params += list(carry_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_total:,} total, {n_trainable:,} trainable")

    # Training log
    log = {"steps": [], "losses": [], "accuracies": [], "carry_losses": []}
    best_acc = 0.0
    converged = False
    converge_step = None
    start_time = time.time()

    model.train()
    for step in range(1, train_cfg.max_steps + 1):
        tokens, carries = generate_batch(train_cfg.batch_size, model_cfg, device=device)

        # Forward
        need_hidden = train_cfg.use_carry_head
        if need_hidden:
            logits, hiddens = model(tokens, return_all_hiddens=True)
        else:
            logits = model(tokens)

        # Main loss: CE on result positions
        result_logits = logits[:, 11:17, :]  # [B, 6, vocab]
        result_targets = tokens[:, 12:18]     # [B, 6]
        main_loss = F.cross_entropy(result_logits.reshape(-1, model_cfg.vocab_size),
                                     result_targets.reshape(-1))

        # Carry loss
        carry_loss_val = 0.0
        if carry_head is not None:
            carry_hidden = hiddens[train_cfg.carry_loss_layer][:, 12:17, :]  # [B, 5, d]
            carry_logits = carry_head(carry_hidden)  # [B, 5]
            carry_loss = F.binary_cross_entropy_with_logits(carry_logits, carries)
            total_loss = main_loss + train_cfg.carry_loss_weight * carry_loss
            carry_loss_val = carry_loss.item()
        else:
            total_loss = main_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.grad_clip)
        optimizer.step()

        # Logging
        if step % train_cfg.log_every == 0:
            log["steps"].append(step)
            log["losses"].append(main_loss.item())
            log["carry_losses"].append(carry_loss_val)

        # Evaluation
        if step % train_cfg.eval_every == 0:
            acc = evaluate(model, eval_tokens, model_cfg)
            log["accuracies"].append({"step": step, "accuracy": acc})
            elapsed = time.time() - start_time

            status = f"  Step {step:>6d} | loss {main_loss.item():.4f}"
            if carry_head:
                status += f" | carry_loss {carry_loss_val:.4f}"
            status += f" | acc {acc:.4f} | {elapsed:.0f}s"
            print(status)

            if acc > best_acc:
                best_acc = acc
            if acc >= train_cfg.target_accuracy and not converged:
                converged = True
                converge_step = step
                print(f"  >>> Converged at step {step} ({acc:.4f})")

            # Save best model whenever we improve
            if acc >= best_acc:
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                if carry_head is not None:
                    torch.save(carry_head.state_dict(),
                               os.path.join(save_dir, "carry_head.pt"))

        # Early stop after convergence (give some extra steps for stability)
        if converged and step >= converge_step + train_cfg.eval_every * 2:
            break

        # Early termination for hopeless models (< 50% acc at step 10K)
        if step == 10000 and best_acc < 0.5:
            print(f"  >>> Early termination: acc={best_acc:.4f} at step 10000")
            break

    total_time = time.time() - start_time

    # Final eval
    final_acc = evaluate(model, eval_tokens, model_cfg)

    # Save final model if better
    if final_acc >= best_acc:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save training log
    metadata = {
        "seed": train_cfg.seed,
        "frozen_component": train_cfg.frozen_component,
        "converged": converged,
        "converge_step": converge_step,
        "final_accuracy": final_acc,
        "best_accuracy": max(best_acc, final_acc),
        "total_steps": step,
        "total_time_seconds": total_time,
        "n_params_total": n_total,
        "n_params_trainable": n_trainable,
        "use_carry_head": train_cfg.use_carry_head,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(log, f)

    return metadata
