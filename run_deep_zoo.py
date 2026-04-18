"""Train a 6-layer zoo (Phase 2). Same task (5-digit addition) as main zoo."""

import os
import json
import time
import torch

from config_deep import ModelConfigDeep, TrainConfigDeep, FREEZABLE_COMPONENTS_DEEP
from data import load_eval_set, create_all_eval_sets
from train import train_model


MODELS_DIR = "models_deep"


def get_model_cfg():
    return ModelConfigDeep()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelConfigDeep()
    print(f"n_layers={model_cfg.n_layers}  d_model={model_cfg.d_model}  device={device}")

    # Reuse existing eval sets from 4-layer zoo (same task)
    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    print(f"Convergence eval set: {eval_tokens.shape[0]} problems")

    configs = []
    for seed in range(3):
        configs.append(("baseline", seed, None))
    for comp in FREEZABLE_COMPONENTS_DEEP:
        for seed in range(3):
            configs.append((comp.replace(".", "_") + "_freeze", seed, comp))

    print(f"\nTraining {len(configs)} models (6 layers, 5-digit addition)...")
    summary = {"models": []}
    start_total = time.time()
    for i, (name, seed, component) in enumerate(configs):
        if component:
            model_name = f"freeze_{component.replace('.', '_')}_seed{seed}"
        else:
            model_name = f"baseline_seed{seed}"
        save_dir = os.path.join(MODELS_DIR, model_name)

        meta_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(meta_path):
            print(f"[{i+1}/{len(configs)}] {model_name}: already trained, skipping")
            with open(meta_path) as f:
                summary["models"].append(json.load(f) | {"model_name": model_name})
            continue

        print(f"\n[{i+1}/{len(configs)}] {model_name}:")
        train_cfg = TrainConfigDeep(seed=seed, frozen_component=component)
        meta = train_model(model_cfg, train_cfg, eval_tokens, save_dir, device=device)
        meta["model_name"] = model_name
        summary["models"].append(meta)
        with open(os.path.join(MODELS_DIR, "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    total_time = time.time() - start_total
    print(f"\nAll {len(configs)} models done in {total_time/60:.1f} min")
    n_conv = sum(1 for m in summary["models"] if m.get("converged"))
    print(f"Converged (acc >= 99%): {n_conv}/{len(configs)}")


if __name__ == "__main__":
    main()
