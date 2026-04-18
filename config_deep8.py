"""Config for the 8-layer zoo (Phase 3)."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfigDeep8:
    n_digits: int = 5
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 8
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
class TrainConfigDeep8:
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


# Deep-8 training is ~2x slower than 6-layer. To fit time budget: use 1 seed
# per freeze (not 3) and 3 baselines. Total: 3 + 18 × 1 = 21 models.
# This is below the main-zoo statistical power but enough for a tight-CI
# check of which layer has shared-drop near zero (single-model-level
# effect, not a cross-seed variance estimate).
FREEZABLE_COMPONENTS_DEEP8 = [
    "embed",
    "layer0.attn", "layer0.mlp",
    "layer1.attn", "layer1.mlp",
    "layer2.attn", "layer2.mlp",
    "layer3.attn", "layer3.mlp",
    "layer4.attn", "layer4.mlp",
    "layer5.attn", "layer5.mlp",
    "layer6.attn", "layer6.mlp",
    "layer7.attn", "layer7.mlp",
    "unembed",
]
N_SEEDS_PER_FREEZE = 1
N_BASELINE_SEEDS = 3
