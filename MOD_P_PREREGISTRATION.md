# Mod-p Replication — Pre-Registration

**Status:** LOCKED before any training runs or code executes the protocol.
**Purpose:** Both multi-AI critics flagged "confirmation-seeking on mod-p"
as the #1 rabbit hole of the 10-hour plan. This file predefines the
interpretation rules so we cannot rationalize whatever comes out.

## Task

Addition modulo a prime: output = (a + b) mod p, where a, b ∈ {0, ..., p−1}.

**p = 23** (locked). Chosen by Codex's recommendation: "p=7 too
close to lookup table; p=113 drifts into grokking regime; p=23 gives
529 input pairs with enough structure to be nontrivial."

## Architecture

Identical to the main zoo: 4 layers, d_model = 64, 4 heads, d_ff = 256.
Only changes forced by the different task:
- Vocabulary: {0, 1, ..., 22, +, =} → 25 tokens.
- Sequence: `a + b = c` → 5 tokens (1 digit each).
- Max seq len: 5.

Training hyperparameters: identical to main zoo (same optimizer, LR,
batch size, seed handling). Training data: all 529 pairs with any
uniform weighting; convergence target = 99% exact-match on held-out
eval set of size 2000 (sampled with replacement from the full grid).

## Zoo

Identical freeze protocol: 3 baselines + 10 single-component freezes × 3
seeds = 33 models. A model "converges" if it reaches ≥ 99% accuracy
within the training budget (conservative: 20,000 steps, which is 2× the
main-zoo typical convergence).

Inclusion criterion: ≥ 99% accuracy. Models below the threshold are
documented but not included in analysis.

## Sites — PRE-SPECIFIED

Because there are no "result digit" positions in the mod-p format, the
six-token schema of the main zoo does not apply. Sites to analyze:

**Primary (locked, 3 sites):**

- `layer1_equals`: layer 1 hidden at the `=` token (position 3).
- `layer2_equals`: layer 2 hidden at the `=` token.
- `layer3_equals`: layer 3 hidden at the `=` token.

The `=` token is the natural analog to the addition-zoo result positions
in the sense that it is the position immediately after the two operands
and immediately before output prediction.

**Control (locked, 1 site):**

- `layer3_plus`: layer 3 hidden at the `+` token (position 1). Analog to
  the main-zoo control site. Expected low CKA, should show zero drop
  for any condition under the corrected protocol.

**Tertiary (if time permits):** layer 0 and layer 1 at `+`; layer 3 at
operand positions. These are NOT in the primary decision rule.

## Decision rule — LOCKED

For each of the 3 primary sites, we compute at k = 8 (the main-zoo
primary k):
- `d_shared` = mean across 33 models of single-subspace ablation drop for
  the shared (A1-restricted C_total-orthonormal) subspace.
- `d_comp` = mean drop for the complement-top-k subspace.
- `d_joint` = mean drop for joint (shared ∪ complement) ablation.
- `d_rand` = mean drop for whitened random directions (100 trials/model).
- `hidden_load` = d_joint − d_comp, reported with 95% bootstrap CI.

**Replication outcomes (outcome-rule-before-seeing-data):**

| Outcome | Criterion |
|---|---|
| **REPLICATES** | Hidden-load CI lower bound > 0 at ≥ 2 of 3 primary sites; d_shared and d_comp both exceed 5× d_rand at those sites; control site shows drop ≤ 0.02 for all conditions. |
| **PARTIAL** | Hidden-load positive at 1 of 3 sites; OR d_shared ≈ 0 and d_comp >> d_rand (complement-dominant without redundancy); OR one or two sites pattern-match the main zoo but the third doesn't. |
| **FALSIFIES** | No hidden-load at any of 3 primary sites (i.e. d_joint ≈ d_comp); OR shared and complement both produce drops ≈ random (A1+A2 extraction found nothing causal on this task). |
| **INCONCLUSIVE** | Training fails for too many models to reach 30+ of 33; OR control site shows substantial drop, invalidating the comparison. |

**What we will NOT do post-hoc:**
- Change p after seeing training time or convergence behavior.
- Swap sites for "more task-relevant" alternatives after ablation results.
- Train longer if initial training fails to converge (stop at 20k steps).
- Re-run with different hyperparameters to "fix" a partial or falsifying outcome.
- Reinterpret the decision rule. If outcome says PARTIAL, we report PARTIAL
  in the paper.

## Commitment

This pre-registration is committed before the first mod-p training step
runs. Any deviation after training begins is a failed run and the entire
Phase 1 is reported as INCONCLUSIVE.

## Expected artifacts

- `data_modp.py`, `train_modp.py`, `run_modp_zoo.py` (task-specific)
- `models_modp/` — 33 saved model weights
- `results/modp/` — aligned activations, extraction, ablation results,
  joint-ablation data
- `MOD_P_REPORT.md` — report with verdict against this rule
