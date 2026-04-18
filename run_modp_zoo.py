"""Train the mod-p zoo (3 baselines + 10 freezes × 3 seeds = 33 models)."""

import os
import json
import time
import torch

from config_modp import ModelConfigModP, TrainConfigModP, FREEZABLE_COMPONENTS
from data_modp import load_eval_set, create_all_eval_sets
from train_modp import train_model


MODELS_DIR = "models_modp"
EVAL_DIR = "eval_sets_modp"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelConfigModP()
    print(f"p={model_cfg.p}  vocab_size={model_cfg.vocab_size}  device={device}")

    # Eval sets
    if not os.path.exists(os.path.join(EVAL_DIR, "full_grid.pt")):
        print("Creating eval sets...")
        create_all_eval_sets(model_cfg, EVAL_DIR)
    eval_data = load_eval_set(os.path.join(EVAL_DIR, "full_grid"))
    eval_tokens = eval_data["tokens"]
    print(f"Full-grid eval set: {eval_tokens.shape[0]} problems")

    # Baselines
    configs = []
    for seed in range(3):
        configs.append(("baseline", seed, None))
    for comp in FREEZABLE_COMPONENTS:
        for seed in range(3):
            configs.append((comp.replace(".", "_") + "_freeze", seed, comp))

    print(f"\nTraining {len(configs)} models on mod-{model_cfg.p} arithmetic...")

    summary = {"models": []}
    start_total = time.time()
    for i, (name, seed, component) in enumerate(configs):
        if component:
            model_name = f"freeze_{component.replace('.', '_')}_seed{seed}"
        else:
            model_name = f"baseline_seed{seed}"
        save_dir = os.path.join(MODELS_DIR, model_name)

        # Skip if already trained
        meta_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(meta_path):
            print(f"\n[{i+1}/{len(configs)}] {model_name}: already trained, skipping")
            with open(meta_path) as f:
                summary["models"].append(json.load(f) | {"model_name": model_name})
            continue

        print(f"\n[{i+1}/{len(configs)}] {model_name}:")
        train_cfg = TrainConfigModP(seed=seed, frozen_component=component)
        meta = train_model(model_cfg, train_cfg, eval_tokens, save_dir, device=device)
        meta["model_name"] = model_name
        summary["models"].append(meta)
        # Incremental save
        with open(os.path.join(MODELS_DIR, "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    total_time = time.time() - start_total
    print(f"\nAll {len(configs)} models done in {total_time/60:.1f} min")
    n_conv = sum(1 for m in summary["models"] if m.get("converged"))
    print(f"Converged (acc >= 99%): {n_conv}/{len(configs)}")


if __name__ == "__main__":
    main()
