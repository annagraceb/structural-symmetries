"""Step 1: Train the model zoo (33 models)."""

import os
import sys
import json
import time

from config import ModelConfig, TrainConfig, FREEZABLE_COMPONENTS
from data import create_all_eval_sets, load_eval_set
from train import train_model


def main():
    model_cfg = ModelConfig()
    base_dir = "models"
    eval_dir = "eval_sets"

    # Create eval sets if they don't exist
    if not os.path.exists(os.path.join(eval_dir, "convergence_eval.pt")):
        print("=" * 60)
        print("Creating evaluation sets...")
        print("=" * 60)
        create_all_eval_sets(model_cfg, eval_dir)

    # Load convergence eval set
    eval_data = load_eval_set(os.path.join(eval_dir, "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    print(f"\nLoaded convergence eval set: {eval_tokens.shape[0]} problems")

    # Define all configs: 3 baselines + 10 freeze × 3 seeds = 33
    configs = []

    # Baselines
    for seed in range(3):
        configs.append(("baseline", None, seed))

    # Single-component freeze
    for component in FREEZABLE_COMPONENTS:
        for seed in range(3):
            configs.append((f"freeze_{component}", component, seed))

    print(f"\nTotal configs to train: {len(configs)}")
    print("=" * 60)

    # Check which models already exist (resume support)
    remaining = []
    for name, frozen, seed in configs:
        save_dir = os.path.join(base_dir, f"{name}_seed{seed}")
        meta_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            status = "CONVERGED" if meta["converged"] else "FAILED"
            print(f"  [SKIP] {name} seed={seed}: {status} "
                  f"(acc={meta['best_accuracy']:.4f})")
        else:
            remaining.append((name, frozen, seed))

    if not remaining:
        print("\nAll models already trained!")
        print_summary(base_dir, configs)
        return

    print(f"\n{len(remaining)} models remaining to train")
    print("=" * 60)

    zoo_start = time.time()
    results = []

    for i, (name, frozen, seed) in enumerate(remaining):
        save_dir = os.path.join(base_dir, f"{name}_seed{seed}")
        print(f"\n[{i+1}/{len(remaining)}] Training {name} seed={seed}")
        print("-" * 40)

        train_cfg = TrainConfig(
            seed=seed,
            frozen_component=frozen,
            use_carry_head=False,
        )

        try:
            meta = train_model(model_cfg, train_cfg, eval_tokens, save_dir)
            status = "CONVERGED" if meta["converged"] else "FAILED"
            print(f"  Result: {status} | acc={meta['best_accuracy']:.4f} "
                  f"| {meta['total_time_seconds']:.1f}s")
            results.append((name, seed, meta))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, seed, {"error": str(e)}))

    total_time = time.time() - zoo_start
    print(f"\n{'=' * 60}")
    print(f"Zoo training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print_summary(base_dir, configs)


def print_summary(base_dir, configs):
    """Print convergence summary for all models."""
    print(f"\n{'=' * 60}")
    print("MODEL ZOO SUMMARY")
    print(f"{'=' * 60}")

    converged = []
    failed = []
    missing = []

    for name, frozen, seed in configs:
        save_dir = os.path.join(base_dir, f"{name}_seed{seed}")
        meta_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            entry = (name, seed, meta["best_accuracy"],
                     meta.get("converge_step"), meta.get("total_time_seconds", 0))
            if meta["converged"]:
                converged.append(entry)
            else:
                failed.append(entry)
        else:
            missing.append((name, seed))

    print(f"\nConverged ({len(converged)}):")
    for name, seed, acc, step, t in converged:
        print(f"  {name} seed={seed}: acc={acc:.4f} at step {step} ({t:.0f}s)")

    if failed:
        print(f"\nFailed to converge ({len(failed)}):")
        for name, seed, acc, step, t in failed:
            print(f"  {name} seed={seed}: best_acc={acc:.4f} ({t:.0f}s)")

    if missing:
        print(f"\nMissing ({len(missing)}):")
        for name, seed in missing:
            print(f"  {name} seed={seed}")

    # Decision gate check
    n_freeze_configs_converged = len(set(
        name for name, seed, acc, step, t in converged
        if name != "baseline"
    ))
    print(f"\nDecision gate: {n_freeze_configs_converged} distinct freeze configs converged")
    if n_freeze_configs_converged >= 6:
        print("  >>> PASS: Sufficient diversity to proceed to Step 2")
    elif n_freeze_configs_converged >= 4:
        print("  >>> MARGINAL: Minimum viable, proceed with caution")
    else:
        print("  >>> FAIL: Insufficient diversity. Stop here.")


if __name__ == "__main__":
    main()
