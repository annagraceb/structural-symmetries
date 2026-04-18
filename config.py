"""Central configuration for the structural symmetries experiment."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfig:
    n_digits: int = 5
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    vocab_size: int = 12  # 0-9 digits, 10='+', 11='='

    @property
    def max_seq_len(self) -> int:
        return self.n_digits + 1 + self.n_digits + 1 + self.n_result_digits

    # Derived
    @property
    def n_result_digits(self) -> int:
        return self.n_digits + 1  # carry can extend by 1

    @property
    def result_start_pos(self) -> int:
        return self.n_digits + 1 + self.n_digits + 1  # after A + B =

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 50_000
    eval_every: int = 1000
    log_every: int = 500
    target_accuracy: float = 0.99
    grad_clip: float = 1.0
    seed: int = 0
    frozen_component: Optional[str] = None

    # Carry head (used in Step 8 Condition B, not in zoo training)
    use_carry_head: bool = False
    carry_loss_weight: float = 0.1
    carry_loss_layer: int = 2


# Token constants
PLUS_TOKEN = 10
EQUALS_TOKEN = 11

FREEZABLE_COMPONENTS = [
    "embed",
    "layer0.attn", "layer0.mlp",
    "layer1.attn", "layer1.mlp",
    "layer2.attn", "layer2.mlp",
    "layer3.attn", "layer3.mlp",
    "unembed",
]
