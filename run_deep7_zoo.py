"""Train the 7-layer zoo. 2-3 seeds per freeze to fit time budget."""

import os
import json
import time
import torch

from config_deep7 import (
    ModelConfigDeep7, TrainConfigDeep7,
    FREEZABLE_COMPONENTS_DEEP7, N_SEEDS_PER_FREEZE, N_BASELINE_SEEDS,
)
from data import load_eval_set
from train import train_model


MODELS_DIR = "models_deep7"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfigDeep7()
    print(f"n_layers={cfg.n_layers}  d_model={cfg.d_model}  device={device}")

    eval_data = load_eval_set(os.path.join("eval_sets", "convergence_eval"))
    eval_tokens = eval_data["tokens"]
    print(f"Convergence eval set: {eval_tokens.shape[0]} problems")

    configs = []
    for seed in range(N_BASELINE_SEEDS):
        configs.append(("baseline", seed, None))
    for comp in FREEZABLE_COMPONENTS_DEEP7:
        for seed in range(N_SEEDS_PER_FREEZE):
            configs.append((comp, seed, comp))

    print(f"\nTraining {len(configs)} models (7 layers)...")
    summary = {"models": []}
    start = time.time()
    for i, (name, seed, component) in enumerate(configs):
        if component:
            mname = f"freeze_{component.replace('.', '_')}_seed{seed}"
        else:
            mname = f"baseline_seed{seed}"
        save_dir = os.path.join(MODELS_DIR, mname)

        if os.path.exists(os.path.join(save_dir, "metadata.json")):
            print(f"[{i+1}/{len(configs)}] {mname}: already trained")
            with open(os.path.join(save_dir, "metadata.json")) as f:
                summary["models"].append(json.load(f) | {"model_name": mname})
            continue

        print(f"\n[{i+1}/{len(configs)}] {mname}:")
        train_cfg = TrainConfigDeep7(seed=seed, frozen_component=component)
        meta = train_model(cfg, train_cfg, eval_tokens, save_dir, device=device)
        meta["model_name"] = mname
        summary["models"].append(meta)
        with open(os.path.join(MODELS_DIR, "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    total = time.time() - start
    print(f"\nDone. {total/60:.1f} min.")
    n_conv = sum(1 for m in summary["models"] if m.get("converged"))
    print(f"Converged: {n_conv}/{len(configs)}")


if __name__ == "__main__":
    main()
