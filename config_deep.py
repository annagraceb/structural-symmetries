"""Config for the 6-layer 'deep' zoo (Phase 2 architecture replication).

Same task as the main zoo (5-digit addition) but with 6 layers instead of 4.
Distinguishes "layer-3 redundancy is a last-residual-before-unembed artifact"
from "layer-k redundancy is a genuine late-layer property."

Architecture: 6 layers, d_model = 64 (same as main zoo), 4 heads,
d_ff = 256. Vocab + seq_len identical to main zoo (N=5 digit addition).
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfigDeep:
    n_digits: int = 5
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 6                  # vs 4 in main zoo
    d_ff: int = 256
    vocab_size: int = 12

    @property
    def max_seq_len(self) -> int:
        return self.n_digits + 1 + self.n_digits + 1 + self.n_result_digits

    @property
    def n_result_digits(self) -> int:
        return self.n_digits + 1

    @property
    def result_start_pos(self) -> int:
        return self.n_digits + 1 + self.n_digits + 1

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclasses.dataclass
class TrainConfigDeep:
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
    # Carry head fields required by train.train_model; kept off for zoo training
    use_carry_head: bool = False
    carry_loss_weight: float = 0.1
    carry_loss_layer: int = 2


# For a 6-layer model, the freezable components include all 6 layers' attn/mlp.
FREEZABLE_COMPONENTS_DEEP = [
    "embed",
    "layer0.attn", "layer0.mlp",
    "layer1.attn", "layer1.mlp",
    "layer2.attn", "layer2.mlp",
    "layer3.attn", "layer3.mlp",
    "layer4.attn", "layer4.mlp",
    "layer5.attn", "layer5.mlp",
    "unembed",
]

PLUS_TOKEN = 10
EQUALS_TOKEN = 11
