"""Step 2: Collect activations from all converged models on the eval sets."""

import os
import json
import numpy as np
import torch
from config import ModelConfig, FREEZABLE_COMPONENTS
from model import ArithmeticTransformer
from data import load_eval_set


def collect_activations_for_model(
    model: ArithmeticTransformer,
    eval_tokens: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[int, np.ndarray]:
    """Run model on eval set, return {layer_idx: activations_array}.

    Each array has shape [n_problems, seq_len, d_model].
    """
    model.eval()
    n = eval_tokens.shape[0]
    all_hiddens_by_layer = {i: [] for i in range(model.cfg.n_layers)}

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = eval_tokens[start:start + batch_size].to(device)
            _, hiddens = model(batch, return_all_hiddens=True)
            for layer_idx, h in hiddens.items():
                all_hiddens_by_layer[layer_idx].append(h.cpu().numpy())

    # Concatenate batches
    result = {}
    for layer_idx, chunks in all_hiddens_by_layer.items():
        result[layer_idx] = np.concatenate(chunks, axis=0).astype(np.float32)

    return result


def get_converged_models(models_dir: str = "models") -> list[dict]:
    """Find all converged models and return their metadata."""
    models = []
    for entry in sorted(os.listdir(models_dir)):
        meta_path = os.path.join(models_dir, entry, "metadata.json")
        model_path = os.path.join(models_dir, entry, "model.pt")
        if os.path.exists(meta_path) and os.path.exists(model_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta["converged"]:
                meta["model_name"] = entry
                meta["model_path"] = model_path
                models.append(meta)
    return models


def main():
    model_cfg = ModelConfig()
    device = "cuda"
    models_dir = "models"
    act_dir = "activations"

    # Load eval sets
    print("Loading eval sets...")
    stratified = load_eval_set(os.path.join("eval_sets", "stratified_2000"))
    natural = load_eval_set(os.path.join("eval_sets", "natural_2000"))

    # Find converged models
    converged = get_converged_models(models_dir)
    print(f"Found {len(converged)} converged models")

    for i, meta in enumerate(converged):
        name = meta["model_name"]
        out_dir = os.path.join(act_dir, name)

        # Check if already collected
        if os.path.exists(os.path.join(out_dir, "layer0_stratified.npy")):
            print(f"  [{i+1}/{len(converged)}] {name}: already collected, skipping")
            continue

        print(f"  [{i+1}/{len(converged)}] Collecting activations for {name}...")
        os.makedirs(out_dir, exist_ok=True)

        # Load model
        model = ArithmeticTransformer(model_cfg).to(device)
        state_dict = torch.load(meta["model_path"], map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        # Collect on stratified set
        hiddens_strat = collect_activations_for_model(model, stratified["tokens"], device)
        for layer_idx, arr in hiddens_strat.items():
            np.save(os.path.join(out_dir, f"layer{layer_idx}_stratified.npy"), arr)

        # Collect on natural set
        hiddens_nat = collect_activations_for_model(model, natural["tokens"], device)
        for layer_idx, arr in hiddens_nat.items():
            np.save(os.path.join(out_dir, f"layer{layer_idx}_natural.npy"), arr)

        # Save metadata
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump({
                "model_name": name,
                "frozen_component": meta["frozen_component"],
                "seed": meta["seed"],
                "final_accuracy": meta["final_accuracy"],
                "token_positions": {
                    "operand_a": list(range(0, model_cfg.n_digits)),
                    "plus": model_cfg.n_digits,
                    "operand_b": list(range(model_cfg.n_digits + 1,
                                            2 * model_cfg.n_digits + 1)),
                    "equals": model_cfg.result_start_pos - 1,
                    "result": list(range(model_cfg.result_start_pos,
                                         model_cfg.max_seq_len)),
                },
            }, f, indent=2)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    print(f"\nActivation collection complete. Saved to {act_dir}/")


if __name__ == "__main__":
    main()
