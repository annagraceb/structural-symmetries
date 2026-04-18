"""Config for the mod-p arithmetic transformer (Phase 1 replication)."""

import dataclasses
from typing import Optional


# Pre-registered in MOD_P_PREREGISTRATION.md
P_MODULUS = 23


@dataclasses.dataclass
class ModelConfigModP:
    p: int = P_MODULUS
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256

    @property
    def vocab_size(self) -> int:
        # p tokens for 0..p-1, + '+', '='
        return self.p + 2

    @property
    def plus_token(self) -> int:
        return self.p        # id = p

    @property
    def equals_token(self) -> int:
        return self.p + 1    # id = p+1

    @property
    def max_seq_len(self) -> int:
        return 5             # "a + b = c" — 5 tokens

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclasses.dataclass
class TrainConfigModP:
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 20_000
    eval_every: int = 500
    log_every: int = 200
    target_accuracy: float = 0.99
    grad_clip: float = 1.0
    seed: int = 0
    frozen_component: Optional[str] = None


FREEZABLE_COMPONENTS = [
    "embed",
    "layer0.attn", "layer0.mlp",
    "layer1.attn", "layer1.mlp",
    "layer2.attn", "layer2.mlp",
    "layer3.attn", "layer3.mlp",
    "unembed",
]
