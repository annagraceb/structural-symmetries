# Phase 4 Addendum — Critic Round 11 and Cross-Digit Resample Test

Added 2026-04-18 in response to a late-round adversarial critic flagging
two potential confounds not covered by earlier rounds.

## Attack 1: Mean-ablation confound (Claude R11)

**Attack claim**: mean-ablation on an answer-consistent subspace at the
penultimate residual is inert by construction because the batch-mean
digit direction is near-null in unembed-relevant space. The Phase 4
"shared is causally silent" finding could be a mean-ablation artifact
rather than a real decoupling.

**Killer test**: resample ablation with same-digit vs cross-digit
counterfactuals (instead of mean-ablation). Replace the per-sample
projection onto the subspace with the projection from a different
sample with (a) the same correct digit at position 12 or (b) a
different correct digit.

If the subspace really is silent, both resamples should also give near-
zero drop. If it's mean-ablation confound, cross-digit should drop
accuracy catastrophically while same-digit preserves it.

**Result** (3 8L baseline models, full 5000-sample eval set, using
21-model-extracted shared/complement from the canonical Phase 4
pipeline):

| Subspace | Mean-ablation drop | Same-digit resample drop | Cross-digit resample drop |
|---|---|---|---|
| SHARED (k=8) | 0.000-0.004 | 0.000-0.004 | 0.000-0.004 |
| COMPLEMENT (k=8) | 0.73-0.80 | 0.85 | 0.91-0.92 |

**Shared subspace is silent under ALL per-sample manipulations** —
even swapping in a wrong-digit sample's projection doesn't hurt
accuracy. This is *stronger* evidence for "shared is causally silent"
than mean-ablation alone.

**Complement subspace encodes digit-specific information**: the
0.07-0.08 gap between cross-digit (0.92) and same-digit (0.85)
resample drops quantifies the digit-specific causal load. The
complement is both (a) globally causal (mean-ablation removes 0.79 of
accuracy) and (b) specifically digit-encoding (cross-digit > same-
digit resample).

**Attack refuted.** Mean-ablation was NOT the confound. If anything the
resample controls strengthen the Phase 4 claim.

## Attack 2: Architectural prior (Codex R11 / Claude R11)

**Attack claim**: CKA 0.85 across trained 8L models at L7 N-1 may be
largely an architectural/initialization property. Untrained
same-architecture models may already share a ≥0.75 CKA at this site,
in which case the "cross-model shared subspace" is background geometry
rather than learned arithmetic mechanism.

**Killer test**: compute shared-subspace CKA across 8 untrained 8L
models (random init, same config) at layer 7 position 12, using the
identical `extract_with_max_dims(..., k=8)` extraction.

**Result** (8 untrained seeds, 28 pairs):

| | Untrained 8L L7 raw CKA | Untrained shared CKA (k=8) | Untrained complement CKA (k=8) |
|---|---|---|---|
| mean | 0.88 ± 0.06 | **0.33 ± 0.08** | 0.93 |
| median | 0.91 | 0.32 | — |

**Trained 8L L7 (3 baseline models)**: raw=0.78, shared=0.85, comp=0.78.

**Delta shared (trained − untrained)**: +0.52 (≈ 6.5σ of untrained
baseline std).

**Attack refuted.** Training substantially *increases* shared-CKA
above the untrained architectural baseline. The shared-subspace
universality reflects learned computation, not architectural prior.

**Interesting note**: raw CKA *decreases* with training (0.88 → 0.78),
while shared-CKA increases (0.33 → 0.85). Training shifts cross-model
similarity from the residual stream overall into a low-dimensional
shared subspace. Supports the "consensus interface" reading from the
earlier critic rounds: deep models *concentrate* their cross-model
similarity into a readout-aligned format at N-1.

## Updated survival probability

Both R11 attacks refuted by direct measurement. Final workshop-paper
survival estimate: 70-80% (up from 60-75% before R11) given the
additional defensive evidence.
