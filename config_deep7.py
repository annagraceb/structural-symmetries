"""Config for 7-layer zoo (Phase 4.5 — 7-layer cliff test).

H4 from round-1 critic: is the monotonic hidden-load shrinkage from 4L
(0.36) to 6L (0.17) to 8L (0.00) at N-1 a smooth log/sigmoid curve, or
is there a discrete phase transition between 6L and 8L? A 7L data
point resolves this.

Architecture: 7 layers, identical otherwise to the 4L/6L/8L zoos.
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfigDeep7:
    n_digits: int = 5
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 7
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
class TrainConfigDeep7:
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
    use_carry_head: bool = False
    carry_loss_weight: float = 0.1
    carry_loss_layer: int = 2


FREEZABLE_COMPONENTS_DEEP7 = [
    "embed",
    "layer0.attn", "layer0.mlp",
    "layer1.attn", "layer1.mlp",
    "layer2.attn", "layer2.mlp",
    "layer3.attn", "layer3.mlp",
    "layer4.attn", "layer4.mlp",
    "layer5.attn", "layer5.mlp",
    "layer6.attn", "layer6.mlp",
    "unembed",
]
# 1 seed per freeze, 3 baselines = 3 + 15 = 18 models. Slightly fewer than 8L.
N_SEEDS_PER_FREEZE = 1
N_BASELINE_SEEDS = 3


PLUS_TOKEN = 10
EQUALS_TOKEN = 11
