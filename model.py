"""Arithmetic transformer: 4-layer, 4-head, d_model=128 decoder-only."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, mask):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class ArithmeticTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Causal mask (registered as buffer so it moves with the model)
        mask = torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
            # LayerNorm weight defaults to 1, bias to 0 — keep those

    def forward(self, input_ids, return_all_hiddens=False):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.tok_embed(input_ids) + self.pos_embed(positions)

        all_hiddens = {} if return_all_hiddens else None
        for i, layer in enumerate(self.layers):
            x = layer(x, self.causal_mask)
            if return_all_hiddens:
                all_hiddens[i] = x  # residual stream after layer i

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_all_hiddens:
            return logits, all_hiddens
        return logits


class CarryHead(nn.Module):
    """Linear probe predicting carry-out at each result position."""

    def __init__(self, d_model: int):
        super().__init__()
        self.probe = nn.Linear(d_model, 1)

    def forward(self, hidden_states):
        """hidden_states: [batch, n_carry_positions, d_model] -> [batch, n_carry_positions]"""
        return self.probe(hidden_states).squeeze(-1)


def get_component_params(model: ArithmeticTransformer, component: str):
    """Return the list of parameters belonging to a freezable component."""
    if component == "embed":
        return list(model.tok_embed.parameters()) + list(model.pos_embed.parameters())
    if component == "unembed":
        return list(model.ln_f.parameters()) + list(model.lm_head.parameters())

    # "layer{i}.attn" or "layer{i}.mlp"
    parts = component.split(".")
    layer_idx = int(parts[0].replace("layer", ""))
    block = parts[1]
    layer = model.layers[layer_idx]

    if block == "attn":
        return list(layer.ln1.parameters()) + list(layer.attn.parameters())
    if block == "mlp":
        return list(layer.ln2.parameters()) + list(layer.mlp.parameters())

    raise ValueError(f"Unknown component: {component}")


def freeze_component(model: ArithmeticTransformer, component: str):
    """Freeze all parameters of the given component (set requires_grad=False)."""
    params = get_component_params(model, component)
    for p in params:
        p.requires_grad = False
    return len(params)
