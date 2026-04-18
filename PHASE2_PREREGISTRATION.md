# Phase-2 (6-layer zoo) — Pre-Registration

**Status:** LOCKED before the 6-layer zoo begins training.
**Purpose:** Distinguish three competing hypotheses for the main-zoo layer-3
"shared-dead, complement-primary, joint >> both" signature:

1. **Absolute depth (layer-3):** the pattern is anchored to layer-3
   regardless of total depth; would appear at layer-3 in the 6-layer zoo.
2. **Last-residual-before-unembed:** the pattern is anchored to the last
   layer; would appear at layer-5 in the 6-layer zoo.
3. **Normalized depth:** the pattern tracks layer/total_layers; would
   appear at layer ~5.25 in 6-layer zoo (i.e., layer 5 in a 6-layer model).

Hypotheses 2 and 3 make similar predictions here; H1 makes a different
prediction.

## Architecture

- 6 layers, d_model = 64, 4 heads, d_ff = 256. Same 5-digit addition task
  as main zoo. Vocabulary/tokenization identical to main zoo.
- 33 models: 3 baselines + 13 single-component freezes × 3 seeds (embed,
  unembed, 6 layers' attn + 6 layers' mlp = 14 freeze configs...
  actually 13 in original + layer4.attn + layer4.mlp + layer5.attn +
  layer5.mlp = 14 new configs, so total = 3 + 14×3 = 45 models.

  To keep the zoo at 33, we use the same 10 freeze configs from main zoo
  (for comparability) minus any 6-layer-specific ones. Actually the
  main zoo had 10: embed, unembed, 4×(attn+mlp) = 10. For 6-layer: embed,
  unembed, 6×(attn+mlp) = 14. This gives 3 + 14×3 = 45 total.

  Accept 45 rather than 33. Either way, the statistical claim is
  "pattern at layer X in Y-layer zoo across Z+ converged models."

## Sites — PRE-SPECIFIED

**Primary (6 sites, one per layer, at result_0 token position 12):**
- `layer0_result_0`, `layer1_result_0`, `layer2_result_0`,
  `layer3_result_0`, `layer4_result_0`, `layer5_result_0`.

**Control (1 site):**
- `layer5_plus` (low-CKA analog to the main-zoo control site).

## Pre-registered predictions (Claude paradox hunter, locked)

For each layer L ∈ {0,...,5}, we compute at k=8:
- `d_shared(L)` — shared-alone single-subspace drop
- `d_comp(L)` — complement-alone single-subspace drop
- `d_joint(L)` — joint ablation drop
- `hidden_load(L)` = d_joint(L) − d_comp(L)

**Prediction under "last-residual-before-unembed" (H2/H3):**
- d_shared is near-zero (≤ 0.005) at layer 5.
- d_comp is 0.3-0.5 at layer 5.
- Hidden load > 0 at layer 5.
- At layer 1, d_shared is comparable to main zoo's layer-1
  shared-primary (~0.3-0.4). The shared→complement handoff occurs in the
  middle layers (2-4).

**Prediction under "absolute depth" (H1):**
- d_shared near-zero and complement large specifically at layer 3.
- Layer 5 looks qualitatively like main-zoo's layer-1 (shared-primary).

**Falsifier for our interpretation:**
If layer 3 shows the shared-dead/complement-primary/hidden-load-~0.4
signature in the 6-layer zoo while layer 5 looks like main-zoo's layer-1
(shared-primary), this refutes both "last-residual" and "normalized
depth" and requires the paper to acknowledge "shared becomes dead at
absolute layer index 3" — which would be an unusual depth-anchored
claim requiring further investigation.

## Outcome classification

| Outcome | Criterion |
|---|---|
| H2/H3 wins (last-residual) | d_shared ≤ 0.005 at layer 5; d_shared > 0.1 at layer 1-2; hidden_load at layer 5 is within CI overlap of main-zoo layer-3 hidden_load |
| H1 wins (absolute depth) | d_shared ≤ 0.005 at layer 3; d_shared > 0.1 at layer 5; hidden_load at layer 3 matches main-zoo layer-3 |
| Monotone depth gradient | d_shared decreases monotonically with layer index; no sharp transition; interpretation: shared-dominance decays gradually |
| Other | Any qualitatively different pattern; report as exploratory |

## What this does NOT license

- Post-hoc swapping of sites for better-looking ones.
- Reporting "inconclusive" because results don't match a hypothesis.
- Training longer / changing hyperparameters if convergence is slow.
- Amending the prediction after seeing any data.

This pre-registration is committed before the deep zoo training begins.
