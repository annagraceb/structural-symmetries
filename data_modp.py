"""Data generation for mod-p addition: `a + b = (a+b) mod p`."""

import os
import json
import random
import numpy as np
import torch

from config_modp import ModelConfigModP


def make_tokens(a: int, b: int, cfg: ModelConfigModP) -> list[int]:
    """5-token sequence: a + b = c."""
    c = (a + b) % cfg.p
    return [a, cfg.plus_token, b, cfg.equals_token, c]


def generate_batch(batch_size: int, cfg: ModelConfigModP, device="cuda"):
    tokens_list = []
    for _ in range(batch_size):
        a = random.randint(0, cfg.p - 1)
        b = random.randint(0, cfg.p - 1)
        tokens_list.append(make_tokens(a, b, cfg))
    return torch.tensor(tokens_list, dtype=torch.long, device=device)


def generate_eval_set(n_problems: int, cfg: ModelConfigModP,
                      full_grid: bool = False, seed: int = 42) -> dict:
    """Generate a fixed evaluation set.

    If full_grid=True, include every (a, b) pair exactly once (cfg.p^2 samples).
    Otherwise sample uniformly with replacement.
    """
    rng = random.Random(seed)
    if full_grid:
        problems = [(a, b) for a in range(cfg.p) for b in range(cfg.p)]
        rng.shuffle(problems)
    else:
        problems = [(rng.randint(0, cfg.p - 1), rng.randint(0, cfg.p - 1))
                     for _ in range(n_problems)]

    tokens_list = []
    metadata = []
    for a, b in problems:
        tokens_list.append(make_tokens(a, b, cfg))
        metadata.append({"a": a, "b": b, "result": (a + b) % cfg.p, "p": cfg.p})

    return {
        "tokens": torch.tensor(tokens_list, dtype=torch.long),
        "metadata": metadata,
    }


def save_eval_set(eval_set: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"tokens": eval_set["tokens"]}, path + ".pt")
    with open(path + ".json", "w") as f:
        json.dump(eval_set["metadata"], f)


def load_eval_set(path: str) -> dict:
    data = torch.load(path + ".pt", weights_only=True)
    with open(path + ".json") as f:
        metadata = json.load(f)
    return {"tokens": data["tokens"], "metadata": metadata}


def create_all_eval_sets(cfg: ModelConfigModP, base_dir: str = "eval_sets_modp"):
    os.makedirs(base_dir, exist_ok=True)
    # Full grid (used for rigorous eval — all p^2 pairs)
    full = generate_eval_set(0, cfg, full_grid=True, seed=42)
    save_eval_set(full, os.path.join(base_dir, "full_grid"))
    # Natural distribution (used for training eval / activations)
    natural = generate_eval_set(2000, cfg, full_grid=False, seed=100)
    save_eval_set(natural, os.path.join(base_dir, "natural_2000"))

    for name in ["full_grid", "natural_2000"]:
        data = torch.load(os.path.join(base_dir, name + ".pt"), weights_only=True)
        print(f"  {name}: {data['tokens'].shape[0]} problems")


if __name__ == "__main__":
    cfg = ModelConfigModP()
    create_all_eval_sets(cfg)
