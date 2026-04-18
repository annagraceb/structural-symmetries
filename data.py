"""Data generation for N-digit addition problems."""

import os
import json
import random
import numpy as np
import torch
from config import ModelConfig, PLUS_TOKEN, EQUALS_TOKEN


def int_to_digits(n: int, n_digits: int) -> list[int]:
    """Convert integer to list of digits, MSB first, zero-padded."""
    digits = []
    for _ in range(n_digits):
        digits.append(n % 10)
        n //= 10
    return list(reversed(digits))


def compute_carries(a: int, b: int, n_digits: int) -> list[int]:
    """Compute carry-out for each column (LSB first). Returns n_digits carry values."""
    carries = []
    carry = 0
    for i in range(n_digits):
        a_digit = (a // (10 ** i)) % 10
        b_digit = (b // (10 ** i)) % 10
        total = a_digit + b_digit + carry
        carry = 1 if total >= 10 else 0
        carries.append(carry)
    return carries  # carries[i] = carry out of column i


def count_carries(a: int, b: int, n_digits: int) -> int:
    return sum(compute_carries(a, b, n_digits))


def make_tokens(a: int, b: int, cfg: ModelConfig) -> list[int]:
    """Build the full 18-token sequence for a + b.

    Format: A₄A₃A₂A₁A₀ + B₄B₃B₂B₁B₀ = R₅R₄R₃R₂R₁R₀
    Result is FORWARD order (most significant digit first).
    This is much harder — the model must solve the full carry chain
    before emitting the first output token.
    """
    a_digits = int_to_digits(a, cfg.n_digits)
    b_digits = int_to_digits(b, cfg.n_digits)
    result = a + b
    # Result padded to n_digits+1, forward order (MSB first)
    result_digits = int_to_digits(result, cfg.n_result_digits)
    result_reversed = result_digits  # forward order — no reversal

    tokens = a_digits + [PLUS_TOKEN] + b_digits + [EQUALS_TOKEN] + result_reversed
    assert len(tokens) == cfg.max_seq_len
    return tokens


def generate_batch(batch_size: int, cfg: ModelConfig, device="cuda"):
    """Generate a random batch of addition problems."""
    max_val = 10 ** cfg.n_digits
    tokens_list = []
    carries_list = []

    for _ in range(batch_size):
        a = random.randint(0, max_val - 1)
        b = random.randint(0, max_val - 1)
        tokens_list.append(make_tokens(a, b, cfg))
        carries_list.append(compute_carries(a, b, cfg.n_digits))

    tokens = torch.tensor(tokens_list, dtype=torch.long, device=device)
    carries = torch.tensor(carries_list, dtype=torch.float, device=device)
    return tokens, carries


def generate_eval_set(n_problems: int, cfg: ModelConfig, stratified: bool = False,
                      seed: int = 42) -> dict:
    """Generate a fixed evaluation set.

    If stratified=True, generates equal numbers per carry-count bucket:
      500 × 0 carries, 500 × 1 carry, 500 × 2 carries, 500 × 3+ carries
    Otherwise, samples from the natural distribution.
    """
    rng = random.Random(seed)
    max_val = 10 ** cfg.n_digits

    if stratified:
        assert n_problems == 2000, "Stratified set is fixed at 2000"
        buckets = {0: [], 1: [], 2: [], "3+": []}
        target = 500

        while any(len(v) < target for v in buckets.values()):
            a = rng.randint(0, max_val - 1)
            b = rng.randint(0, max_val - 1)
            nc = count_carries(a, b, cfg.n_digits)
            key = nc if nc <= 2 else "3+"
            if len(buckets[key]) < target:
                buckets[key].append((a, b))

        problems = buckets[0] + buckets[1] + buckets[2] + buckets["3+"]
    else:
        problems = [(rng.randint(0, max_val - 1), rng.randint(0, max_val - 1))
                     for _ in range(n_problems)]

    tokens_list = []
    carries_list = []
    metadata = []
    for a, b in problems:
        tokens_list.append(make_tokens(a, b, cfg))
        c = compute_carries(a, b, cfg.n_digits)
        carries_list.append(c)
        metadata.append({
            "a": a, "b": b, "result": a + b,
            "n_carries": sum(c), "carries": c,
        })

    return {
        "tokens": torch.tensor(tokens_list, dtype=torch.long),
        "carries": torch.tensor(carries_list, dtype=torch.float),
        "metadata": metadata,
    }


def save_eval_set(eval_set: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "tokens": eval_set["tokens"],
        "carries": eval_set["carries"],
    }, path + ".pt")
    with open(path + ".json", "w") as f:
        json.dump(eval_set["metadata"], f)


def load_eval_set(path: str) -> dict:
    data = torch.load(path + ".pt", weights_only=True)
    with open(path + ".json") as f:
        metadata = json.load(f)
    return {"tokens": data["tokens"], "carries": data["carries"], "metadata": metadata}


def create_all_eval_sets(cfg: ModelConfig, base_dir: str = "eval_sets"):
    """Create and save all evaluation sets needed for the experiment."""
    os.makedirs(base_dir, exist_ok=True)

    # 1. Convergence eval (5000 problems, natural distribution)
    convergence = generate_eval_set(5000, cfg, stratified=False, seed=100)
    save_eval_set(convergence, os.path.join(base_dir, "convergence_eval"))

    # 2. Stratified set (2000 problems, carry-stratified)
    stratified = generate_eval_set(2000, cfg, stratified=True, seed=200)
    save_eval_set(stratified, os.path.join(base_dir, "stratified_2000"))

    # 3. Natural validation set (2000 problems, natural distribution)
    natural = generate_eval_set(2000, cfg, stratified=False, seed=300)
    save_eval_set(natural, os.path.join(base_dir, "natural_2000"))

    print(f"Created eval sets in {base_dir}/")
    for name in ["convergence_eval", "stratified_2000", "natural_2000"]:
        path = os.path.join(base_dir, name)
        data = torch.load(path + ".pt", weights_only=True)
        n = data["tokens"].shape[0]
        print(f"  {name}: {n} problems")

    return convergence, stratified, natural
