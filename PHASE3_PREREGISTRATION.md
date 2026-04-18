# Phase-3 (8-layer zoo) — Pre-Registration

**Status:** LOCKED before the 8-layer zoo begins training.
**Purpose:** Test N-1 invariance. Phase 2 (6-layer zoo) showed the
"shared-dead, complement-primary" signature at layer 5 (the 6-layer
last-residual-before-unembed). If the effect is truly "one layer
before the unembed regardless of total depth," then in an 8-layer
zoo we predict the signature at layer 7. If it appears at layer 5
(matching the 6-layer zoo's absolute layer), the effect has a
surprising depth-anchored component.

## Architecture

- 8 layers, d_model = 64, 4 heads, d_ff = 256 (same as 4-layer main
  zoo except n_layers = 8). Identical 5-digit addition task.
- 54 models: 3 baselines + 17 single-component freezes × 3 seeds.
  Freezable components: embed, unembed, 8×(attn+mlp) = 18. So
  3 + 18×3 = 57 models.

  If training is slow we will reduce seeds from 3 to 2 (baseline + 18×2
  = 39 models).

## Sites — PRE-SPECIFIED

**Key sites (locked, 3 sites at `result_0` token position):**
- `layer3_result_0` — matches main zoo's layer 3 in absolute index
- `layer5_result_0` — matches 6-layer zoo's layer 5
- `layer7_result_0` — 8-layer zoo's own last-residual-before-unembed

**Control:**
- `layer7_plus` — low-relevance token at last layer.

If time permits, also run layers 0, 1, 2, 4, 6 for a full depth sweep,
but the 3 primary sites above are the pre-registered test.

## Pre-registered predictions

At k=8 in the 8-layer zoo:

| Hypothesis | Prediction |
|---|---|
| H2a — "N-1 invariance" (last-residual-before-unembed) | shared ≈ 0 at layer 7; shared > 0.3 at layer 5; shared > 0.3 at layer 3 |
| H2b — "layer 5 absolute" (6-layer zoo's result was at layer 5, that index is special) | shared ≈ 0 at layer 5; shared > 0.3 at layer 3 and layer 7 |
| H4 — "normalized depth" (constant fraction of total depth) | shared ≈ 0 at layer 5 (5/8 = 0.625 ≈ 3/4 = 0.75 close enough given zoo variance); shared substantial at layer 7 |
| H5 — "no preferred layer" | shared roughly constant across 3-5-7; no collapse anywhere |

**Decisive outcome:** whichever layer has shared drop ≤ 0.005 at k=8
with 95% CI fully below 0.01.

## Outcome classification

| Outcome | Criterion |
|---|---|
| H2a wins (N-1 invariance) | shared near-zero at layer 7; shared substantial at layers 3 and 5 |
| H2b or H4 wins | shared near-zero at layer 5 in 8-layer zoo |
| H5 wins | shared drop > 0.1 at all of layers 3, 5, 7 |
| Exotic | Pattern doesn't match any above |

## What this does NOT license

- Post-hoc swapping which layer counts as "key."
- Adjusting architecture or training after seeing convergence.
- Amending the prediction after seeing data.

Committed before the 8-layer zoo training begins.
