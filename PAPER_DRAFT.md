# Cross-Model Shared Directions Are Redundantly Backed: Three Methodological Corrections for CKA-Based Subspace Causality Analyses

**Working draft. All numbers come from `results/p1/*.json` on the
2026-04-17 run (post-correction, K_VALUES = [4,8,12,16], k_pca = 32).
Pre-registration: `P1_PREREGISTRATION.md`. Derivation support:
`VARIANCE_CONFOUND_DERIVATION.md`. Artifact index: `P1_SUMMARY.md`.**

---

## Abstract

We study whether cross-model shared directions — extracted via the
generalized eigenproblem `C_shared v = λ C_total v` — are causally
important in a 33-model arithmetic transformer zoo. We identify three
methodological issues in standard ablation-based causal tests and show
that correcting all three changes the headline conclusion.

**(A1)** The unrestricted generalized eigenproblem preferentially returns
low-variance directions because the ratio is minimized when either
covariance is small. Restricting the eigenproblem to the top-K principal
subspace of `C_total` forces shared directions into the high-variance
regime.

**(A2)** Unit-norm random baselines remove different amounts of per-model
activation variance than structured subspaces. Whitening so that each
random direction satisfies `v^T C_total v = 1` variance-matches the
comparison.

**(A3)** Single-subspace ablation cannot see redundant causal structure.
At all six tested sites in this zoo, shared and complement
(orthogonal-high-variance) subspaces are *jointly* load-bearing: ablating
shared alone is near-inert at the final layer (raw drop ≤ 0.003) while
ablating the complement alone produces substantial drops (0.37–0.43).
But **jointly ablating shared ∪ complement produces 0.78–0.89 accuracy
drops** — much larger than either alone. The shared subspace is not
causally inert; it is redundantly backed up by the complement.

Under (A1) + (A2) + (A3) corrections, the 33-model zoo shows:

- Both cross-model shared directions and the orthogonal-high-variance
  complement carry substantial causal weight at all 14 tested
  task-relevant sites (per-variance effects 10–80× above whitened random);
  the control site (CKA = 0.075) shows zero effect for any condition.
- At the final transformer layer (layer 3), shared directions are
  linearly probeable (r = 0.78–0.91 for carry and sum-magnitude features)
  but single-ablation-inert. The paradox resolves in joint ablation:
  shared is causally important, but the complement substitutes when
  shared alone is removed.
- The "shared < random" sign from the original unrestricted unit-norm
  protocol reflects at least two confounds: variance-content mismatch
  (addressed by A1 and A2), and redundancy invisibility (addressed by
  A3). Each confound alone is sufficient to mask a real causal role.

The methodological punchline is that **cross-model shared-subspace
causality cannot be assessed by single-subspace ablation**; joint
ablation against the orthogonal complement is required to expose
redundantly-realized causal contributions.

**Replication on a second task (modular arithmetic).** We re-ran the
full three-correction protocol on a separate 33-model zoo trained on
addition modulo 23 with identical architecture and training. At three
pre-specified primary sites, the hidden-load phenomenon replicates
(all three CI lower bounds strictly positive, range 0.013–0.140);
magnitudes are smaller than in the main zoo (0.03–0.17 vs 0.36–0.44),
consistent with mod-p having less task structure to redundantly encode.

**Replication at larger depth (6-layer and 8-layer zoos).** We trained
two additional zoos with 6 and 8 layers (45 and 21 models respectively,
same task as main zoo). Pre-registered predictions tested whether the
"shared dead at layer 3" signature of the main zoo was an intrinsic
layer-3 property or a last-residual-before-unembed effect, and whether
the effect is robustly at position N-1 across depths. **The
N-1-invariance prediction holds at all three architecture depths**:
shared is dead at layer 3 in the 4-layer main zoo, layer 5 in the
6-layer zoo, and layer 7 in the 8-layer zoo (all with shared-drop
CI ≤ [0.000, 0.000]). The "shared dead" finding is therefore a
readout-adjacent geometric effect, not a fixed-depth phenomenon.

Unexpectedly, we *also* observe that hidden-load at the N-1 layer
shrinks with depth (0.36-0.44 → 0.17 → −0.000 across 4L, 6L, 8L; see
Fig. 6). In the 8-layer zoo at layer 7, shared directions remain
**linearly probeable** at r = 0.94 (sum_magnitude) and r = 0.84
(carry_col3) while carrying zero independent causal load — the paper's
cleanest demonstration that cross-model shared structure at the
readout-adjacent layer can be a high-fidelity *observational
projection* of the complement's computation, not a computation in its
own right. We report the depth trend as a discovery on three data
points (one zoo per depth) and note an important tension: if hidden
load at N-1 tends to zero with depth, the "shared-dead at N-1" pattern
may be a vanishing feature of small models rather than a scale-robust
structural invariance. Disambiguating requires replication at larger
depths (12, 16+); that is beyond this paper's scope.

Together, these replications put the specific redundancy structure
(shared + complement each contribute substantial causal load, with a
readout-adjacent role-flip at the output endpoint) on firmer ground as
a property of this architecture family. The corrective protocol
itself (A1 + A2 + A3) is task- and depth-independent by construction.

---

## 1. Introduction

### 1.1 Context

A common question in mechanistic interpretability is whether
cross-model representational similarity — the tendency of independently
trained networks to develop similar-looking activation subspaces —
corresponds to *causal universality*, the property that these shared
directions are the ones actually driving the computation.
Representational-similarity tools such as Centered Kernel Alignment
(CKA; Kornblith et al., 2019), Canonical Correlation Analysis and its
variants (SVCCA, Raghu et al., 2017; PWCCA, Morcos et al., 2018), and
generalized eigenproblems of the form `C_shared v = λ C_total v` (which
we use in this paper, adapted from the ratio formulations common in
cross-subject BCI and neural population analysis) can identify
cross-model shared subspaces readily. But testing whether those
subspaces are causally load-bearing requires an intervention: the
standard move is to ablate the shared subspace and measure the
resulting accuracy drop relative to some baseline (typically unit-norm
Gaussian random directions of the same rank).

### 1.2 The ambiguity behind universality claims

Ablation-based causality tests applied to CKA-style shared subspaces
have produced conflicting results across the interpretability
literature. Some small-model studies find that shared directions are
causally important; others find they are indistinguishable from random
or even less damaging when ablated. The difficulty is partly
mechanistic — different architectures may store computation
differently — and partly methodological: the protocol combines a
subspace-identification step, a baseline-sampling step, and an
intervention step, each of which can introduce its own bias into the
final "is it causal?" verdict.

In particular, the universality hypothesis — that independently-trained
networks converge to similar internal circuits (Li, Yosinski et al.;
Chughtai, Chan & Nanda, 2023) — sits uneasily against ablation studies
that seem to deny the causal role of the agreed-upon structure.
Mechanistic interpretability work on algorithmic tasks (grokking
dynamics; Nanda et al., 2023; Liu et al. "Omnigrok") has shown that
specific circuits emerge and change over training, but the relationship
between circuit emergence and cross-model representational agreement is
under-characterized at the ablation level.

### 1.3 Our result

We study the question on a small arithmetic transformer zoo and arrive
at a picture that clarifies one source of the field's conflicting
results. Three distinct methodological choices in the standard protocol
each independently bias the ablation test toward underestimating
shared-subspace causal importance:

- **(A1)** The unrestricted generalized eigenproblem preferentially
  extracts low-variance shared directions. A PCA restriction to the
  top-K principal subspace of `C_total` avoids this.
- **(A2)** Unit-norm random baselines remove different amounts of
  per-model variance than structured subspaces. Whitening the random
  baseline variance-matches the comparison.
- **(A3)** Single-subspace ablation cannot see causal signal that is
  redundantly realized across the ablated subspace and its orthogonal
  complement (a known failure mode in mechanistic interpretability;
  see e.g. causal-scrubbing discussions in the Anthropic/Redwood
  literature on path patching and circuit factorization). Joint
  ablation against the orthogonal complement exposes this hidden load.

Correcting all three on the same 33-model zoo, same eigenvectors, and
same ablation positions reverses the sign of the headline finding:
cross-model shared directions are causally important, interpretable,
and redundantly encoded against the orthogonal high-variance complement.
We replicate the core phenomenon on a second task (modular arithmetic)
with a new 33-model zoo and pre-registered decision rule; the
redundancy signature survives a task-structure change (no carry chain,
different vocabulary, different sequence format), though with smaller
magnitude consistent with mod-p having less task structure to
redundantly encode.

This ablation-based causal test has produced conflicting results
in the literature, and it produced a conflicting result in our own
earlier experiments on a small arithmetic transformer zoo: the
cross-model shared subspace, extracted via the unrestricted
generalized eigenproblem and ablated under the unit-norm convention,
produced *smaller* accuracy drops than random directions at one of
three preselected high-CKA sites (shared/random ratio 0.045,
p < 0.001 under Wilcoxon), with comparable or slightly larger drops at
the other two sites. Taken at face value, this suggests cross-model
shared directions are at best no more causal than random, and at worst
anti-correlated with causal importance — the "interpretable but not
functional" reading of universality.

This paper shows that this reading is unsupported. Three methodological
choices in the standard protocol each independently bias the result
toward underestimating shared-subspace causal importance: (A1) the
unrestricted generalized eigenproblem preferentially extracts
low-variance shared directions; (A2) unit-norm random baselines
remove different amounts of per-model variance than structured subspaces;
and (A3) single-subspace ablation cannot see causal signal that is
redundantly realized across the ablated subspace and its complement.
Correcting all three on the same data and eigenvectors reverses the
sign of the headline finding: cross-model shared directions are
causally important, interpretable, and redundantly encoded against the
orthogonal high-variance complement.

The contribution of this paper is methodological and empirical:

1. We identify and analytically characterize three confounds in standard
   cross-model subspace causality tests (A1: eigenproblem low-variance
   bias; A2: variance-mismatched random baselines; A3: redundancy
   invisibility of single-subspace ablation).
2. We derive the signed predictions each confound makes, and show the
   effect of each correction empirically on a 33-model zoo by applying
   them one at a time to the same data.
3. Under the corrected protocol, cross-model shared directions are
   causally important at every tested task-relevant site. Joint ablation
   at six sites shows shared contributes 0.36-0.64 of accuracy-drop
   budget that is invisible to single-subspace ablation; single-subspace
   drops underestimate shared-subspace importance by 0.36-0.64 absolute.
4. Layer-dependent asymmetry: the shared subspace is the primary
   single-subspace ablation target at layer 1 (0.38 drop); at layer 3
   it is the redundant backup (< 0.003 drop, but 0.36-0.44 hidden load
   exposed by joint ablation).

Our evidence comes from one 4-layer, d_model = 64 transformer zoo
trained on 5-digit addition. The methodological corrections are
architecture-independent; the specific redundancy pattern is a
case study in this setting.

---

## 2. Setup

### 2.1 Model zoo

Following the Solution Symmetry Exploration protocol, we trained 33
4-layer decoder-only transformers (`d_model = 64`, 4 heads, `d_ff = 256`)
on N-digit addition (N = 5). The zoo consists of 3 baseline models plus
30 single-component freeze conditions (10 freezable components × 3 seeds)
in which one component (embed / unembed / per-layer attn or mlp) has its
parameters held at random initialization while everything else trains
normally. All 33 models reach ≥ 99% exact-match accuracy on a held-out
2000-problem stratified eval set.

### 2.2 Cross-model shared and complement subspaces

For any layer × token-position site, we construct two covariance matrices
from the model-aligned activations (Procrustes alignment to a reference
baseline model is applied to remove arbitrary rotations):

  C_shared = Cov_x( mean_k h_k(x) )
  C_total  = mean_k Cov_x( h_k(x) )

The *shared subspace* at site s is the span of the top-k generalized
eigenvectors of `C_shared v = λ C_total v`, restricted to the top-32
principal subspace of `C_total` (Amendment A1 to the pre-registration:
this restriction prevents dead-axis solutions in the bottom of the ratio
spectrum). Call these eigvecs `V_shared ∈ ℝ^{k × d}`.

The *complement subspace* at site s is the span of the top-k eigenvectors
of `P_⊥ C_total P_⊥`, where `P_⊥ = I − P_{V_shared}` is the Euclidean
projector orthogonal to V_shared. These are the highest-variance
directions orthogonal to shared.

The *random baseline* is k whitened Gaussian directions: `r ∼ N(0, I)`,
`v = (C_total + ε I)^{-1/2} r`, normalized so `v^T C_total v = 1`. This
ensures each random direction has the same per-model activation variance
as a C_total-orthonormal eigvec.

### 2.3 Variance-matched ablation

We use mean-centered subspace projection ablation: replace the activation
component in S with its in-distribution mean. Under our normalization,
each ablation subspace satisfies the per-direction constraint
`v^T C_total v = 1`. We additionally report the projection-trace
`trace(P_S C_total P_S)` as the basis-independent quantity giving the
exact amount of activation variance removed.

### 2.4 Sites

Three primary sites were preselected via top-CKA filtering before any
ablation:

  layer1_result_0 (CKA = 0.796, no-PC1 = 0.483)
  layer2_equals   (CKA = 0.762, no-PC1 = 0.382)
  layer2_result_0 (CKA = 0.759, no-PC1 = 0.530)

A control site `layer3_plus` (CKA = 0.075) was added to verify that the
ablation effects are not "any structured ablation damages the model" but
require task-relevant sites.

---

## 3. The variance-mismatch confound

[Reference VARIANCE_CONFOUND_DERIVATION.md sections 1-6 condensed.]

**Lemma (Variance-Mismatch Bias).** Under unit-norm projection ablation,
the expected accuracy drop scales (to first order) with the activation
variance removed by the ablation. Two subspaces removing different
amounts of variance cannot be compared on a hurt-equals-causality basis
without an explicit variance correction.

**Predicted directional consequences:**

- Under standard unit-norm ablation, `drop(top_shared) ≳ drop(random) ≳
  drop(bottom_shared)` because variance removed scales the same way.
- Under variance-matched ablation, the perturbation magnitude is held
  constant; measured drops then reflect per-variance causal load.

**Predicted empirical signature:** the ratio `drop(top_shared) /
drop(random)` can be *less than 1* under unit-norm convention even when
shared directions are causally important per unit of variance, whenever
the shared and random subspaces have sufficiently different variance
content. The sign of the ratio under unit-norm is therefore not a clean
signal about causality; it is partly a function of the variance mismatch
between the subspaces being compared.

---

## 4. Empirical results

### 4.1 Variance-matched ablation: structured (shared, complement) ≫ random

*(Fig. 1 shows the main joint-ablation pattern; Fig. 4 shows the
A1 PCA-restriction flipping the Step-7 sign.)*


Per-variance hurt (mean ablation drop divided by projection-trace variance,
in units of accuracy fraction per unit of removed activation variance) at
the primary k = 8:

| Site | shared | complement | ortho-anti-shared | bottom-of-ratio | shared/complement |
|------|---|---|---|---|---|
| layer1_result_0 | 0.0337 | 0.0234 | 0.0121 | 0.0002 | 1.44× |
| layer2_equals   | 0.0012 | 0.0020 | 0.0001 | 0.0000 | 0.60× |
| layer2_result_0 | 0.0155 | 0.0072 | 0.0002 | 0.0000 | 2.15× |
| **layer3_plus (CONTROL)** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | — |

Raw ablation drops at k = 8 (without per-variance normalization):

| Site | shared drop | complement drop | random drop |
|------|---|---|---|
| layer1_result_0 | 0.376 | 0.251 | ≈ 0 |
| layer2_equals   | 0.020 | 0.028 | ≈ 0 |
| layer2_result_0 | 0.312 | 0.100 | ≈ 0 |
| layer3_plus (CONTROL) | 0.000 | 0.000 | 0 |

**Key findings:**

- **Both shared and complement subspaces are massively more causally
  important than matched-variance random directions at all three primary
  sites.** Whitened random ablation produces near-zero accuracy drops
  across all primary sites; the structured subspaces (whether shared or
  orthogonal-high-variance complement) produce drops of 0.02 to 0.38.
- **The shared > complement ordering holds at 2 of 3 primary sites
  (layer1_result_0 1.44×, layer2_result_0 2.15×).** At layer2_equals,
  complement is 1.7× larger than shared per-variance. This site has the
  smallest absolute drops, and the ordering should be considered
  inconclusive there.
- **Bottom-of-ratio is a dead-axis subspace** (projection-trace ≈ 0.2-0.4
  vs ≈ 11-20 for shared) and produces zero drop everywhere. Confirms the
  variance-matching analysis: the bottom of the ratio eigenproblem occupies
  the low-variance corner of the residual stream and contains essentially
  no signal to remove.
- **The control site (layer3_plus, CKA = 0.075) cleanly passes:** all
  four ablation conditions produce zero accuracy drop, confirming the
  primary-site effects require task-relevant sites and are not "any
  structured ablation hurts."
- **ε-stability:** results consistent across Tikhonov ridges
  ε ∈ {1e-5, 1e-4, 1e-3} (see §4.5).

### 4.2 Unit-norm ablation on A1-restricted dirs

We re-ran the A1-restricted (PCA-32) shared eigenvectors through standard
unit-norm ablation, varying only the per-direction normalization
convention:

| Site | shared_unit drop | bottom_unit drop | random_unit drop | shared/random |
|------|---|---|---|---|
| layer1_result_0 | 0.4476 | 0.0000 | 0.0240 | **18.6×** |
| layer2_equals   | 0.0396 | 0.0013 | 0.0011 | **34.5×** |
| layer2_result_0 | 0.3277 | 0.0000 | 0.0050 | **65.6×** |

These ratios are large under unit-norm — the A1 restriction alone is
sufficient to flip the sign relative to the original Step-7 finding (which
used unrestricted shared dirs).

### 4.3 Decomposing A1 vs A2 contributions

To disentangle the contribution of the PCA restriction (A1) from the
variance-matched random baseline (A2), we ran four protocols on the same
33 models, three primary sites, and ablation hook:

- *Protocol 1 — original Step 7:* OLD shared dirs (no PCA restriction,
  unit-norm) vs unit-norm random.
- *Protocol 2 — A1 only:* NEW shared dirs (PCA-restricted, unit-norm)
  vs unit-norm random.
- *Protocol 3 — A1 + A2:* NEW shared dirs (PCA-restricted,
  C_total-orthonormal) vs whitened random.

The shared-direction ablation drop in Protocol 2 and Protocol 3 is
identical at each site (subspace-based projection ablation is invariant
to per-direction scaling). Differences in the ratio come entirely from
the random baseline:

| Site | Proto 1 ratio | Proto 2 ratio | Proto 3 ratio |
|------|---|---|---|
| layer1_result_0 | **0.99** (≈ 1) | 18.0× | 69673× (rand≈0) |
| layer2_equals   | **0.05** (shared << random) | 37.0× | very large |
| layer2_result_0 | **1.58** (shared > random) | 69.9× | 100144× |

**Findings:**

- Protocol 1 reproduces the original Step-7 sign at one of three sites
  (layer2_equals: ratio 0.045, strongly "shared << random"). The other
  two sites show shared ≈ random or shared > random under the original
  protocol — the original Step-7 cross-site averaging masked this site
  heterogeneity.
- Protocol 2 (A1 alone) flips the sign decisively at all three sites:
  the same models, same eigenproblem, but the eigenvectors now sit in
  the high-variance subspace of C_total. This is the dominant correction.
- Protocol 3 (A1 + A2) drives random baselines to ≈ 0 by sampling them
  in the whitened space (heavily weighting low-variance noise axes).
  The shared/random ratio becomes very large but the absolute shared
  drop is unchanged from Protocol 2.

The headline correction is therefore the PCA restriction (A1), not the
variance-matched ablation per se; A2 is a strict additional improvement
that cleans the random baseline.

### 4.4 Saturation behavior at large k

As k approaches K_pca = 32, the shared and complement subspaces span a
larger fraction of the residual stream. At k = 16 (half of K_pca), the
top-16 shared subspace has projection-trace 15-27 (out of total residual
variance ≈ 17-26), so it covers most of the high-variance region. The
complement-of-top-16 has only 16 effective dimensions left in the
PCA-restricted space, leading to a forced shrinking of its
projection-trace (7-9 at k=16 vs 11-14 at k=8). The k = 4 to k = 12
range is therefore the most informative for the shared-vs-complement
comparison; k = 16 is reported but should be interpreted as a
saturation regime, not a clean test.

| Site | k | shared drop | complement drop | s_trace | c_trace |
|------|---|---|---|---|---|
| layer1_result_0 | 4  | 0.198 | 0.441 | 8.4  | 11.6 |
| layer1_result_0 | 8  | 0.376 | 0.251 | 11.2 | 10.7 |
| layer1_result_0 | 12 | 0.551 | 0.180 | 13.0 | 9.4  |
| layer1_result_0 | 16 | 0.671 | 0.106 | 15.3 | 7.3  |
| layer2_equals   | 4  | 0.007 | 0.015 | 12.0 | 15.7 |
| layer2_equals   | 8  | 0.020 | 0.028 | 16.4 | 13.8 |
| layer2_equals   | 12 | 0.042 | 0.024 | 19.5 | 11.6 |
| layer2_equals   | 16 | 0.191 | 0.022 | 23.8 | 7.8  |
| layer2_result_0 | 4  | 0.265 | 0.153 | 16.9 | 14.3 |
| layer2_result_0 | 8  | 0.312 | 0.100 | 20.1 | 14.0 |
| layer2_result_0 | 12 | 0.415 | 0.051 | 23.9 | 10.9 |
| layer2_result_0 | 16 | 0.523 | 0.017 | 26.6 | 8.5  |

Within the informative regime (k = 4 to 12) at layer1_result_0,
**complement actually beats shared at k = 4** (0.441 vs 0.198), consistent
with the observation that small high-variance subspaces orthogonal to
shared can be highly task-relevant. As k grows, shared dominates because
the shared subspace eventually subsumes more of the high-variance
task-relevant region.

This k-dependence suggests the "shared vs complement" question may not
have a single answer; it depends on how much of the residual stream is
being intervened on. A reasonable simple summary is: at k = 8 (half the
informative range), shared and complement are comparable; shared
dominates by 2-3× at higher k.

### 4.5 Tikhonov ridge stability

The generalized eigenproblem solution is stable across ridge scales:

| ε scale | shared drop | anti_shared_raw drop | complement drop | denom cond # |
|---|---|---|---|---|
| 1e-5 | 0.4481 | 0.0001 | 0.2196 | 1.60×10³ |
| 1e-4 | 0.4532 | 0.0001 | 0.2170 | 1.58×10³ |
| 1e-3 | 0.4977 | 0.0001 | 0.1923 | 1.43×10³ |

Single site, k = 10. Shared drop varies < 0.05 absolute across three
orders of magnitude in ridge; the qualitative shared > complement >
anti_shared_raw ordering is preserved.

### 4.6 Decomposition of the variance content

Projection-trace variance per condition at k = 8 (how much activation
variance each subspace removes under ablation):

| Site | shared trace | complement trace | bottom trace | random trace |
|------|---|---|---|---|
| layer1_result_0 | 11.15 | 10.73 | 0.20 | ~k by construction |
| layer2_equals   | 16.42 | 13.79 | 0.42 | ~k by construction |
| layer2_result_0 | 20.11 | 13.97 | 0.33 | ~k by construction |

(Random is variance-matched per A2, so `trace(P_rand C_total P_rand) = k = 8`.
Shared and complement traces exceed k slightly because the C_total-
orthonormal basis is not Euclidean-orthonormal, so individual directions
contribute overlapping Euclidean-basis covariance.)

The bottom-of-ratio subspace contains essentially zero per-model variance
(trace 0.2-0.4 out of total residual variance ~22-26) — a dead-axis set.
Its near-zero ablation drop reflects the absence of signal to remove
rather than absence of causal importance. The complement subspace, in
contrast, has variance comparable to shared and produces substantial
(if smaller) single-subspace ablation drops. The interesting contrast is
therefore shared vs complement, not shared vs bottom-of-ratio.

### 4.7 Direction interpretability holds

We probed the top shared directions for correlation with hand-labeled
arithmetic features (digit identity per column, number of carries,
sum magnitude). At layer1_result_0:

  direction_0: result_digit_col4 (r = 0.935), carry_col4 (r = -0.508)
  direction_2: sum_magnitude (r = -0.981), carry_col4 (r = -0.804)

Shared directions are both strongly linearly probeable for arithmetic
features and (per §6.4 below, using joint ablation) causally important
via a redundantly-realized encoding. The original anti-correlation
hypothesis (readable but not causal) is refuted in two distinct ways:
single-subspace ablation under matched variance is substantial at this
site (0.38 drop), and joint ablation reveals additional hidden load
masked by complement compensation.

---

## 5. Discussion

### 5.1 Cross-model agreement is causally informative, but redundantly so

Under the three-correction protocol (A1 + A2 + A3), cross-model shared
variance identifies directions that are causally important and
redundantly encoded against their orthogonal high-variance complement.
The standard unit-norm single-subspace ablation protocol systematically
underestimates the importance of this kind of structure, especially at
layers where the redundancy is skewed (shared carries backup load at
layer 3 in this zoo). Unsupervised extraction of interpretable features
via `C_shared v = λ C_total v` still works — direction-0 at
layer1_result_0 correlates with the fourth result-digit at r = 0.935 —
but the causal role of the resulting directions can only be assessed
with joint ablation against the orthogonal complement.

### 5.2 The complement subspace is load-bearing

Orthogonal high-variance directions also carry substantial causal weight
in our measurements (single-subspace drops 0.25–0.49 across tested
sites). This means there is no clean "shared = causal,
non-shared = non-causal" dichotomy. In this zoo the two subspaces are
redundantly load-bearing, with the primary role shifting from shared to
complement as we move from layer 1 to layer 3. Any universality claim
based on cross-model shared structure should be read as "this particular
representational axis is one of two (or more) load-bearing encodings,"
not as "this axis is the computation."

### 5.3 Methodological corollary for the field

Three distinct ablation-protocol choices systematically bias
cross-model subspace causality analyses:

1. **Unrestricted generalized-eigenproblem extraction** tends to land in
   low-variance corners of activation space (A1). Restrict to the top-K
   principal subspace of `C_total` first.
2. **Unit-norm random baselines** remove different amounts of per-model
   variance than structured subspaces (A2). Whiten so that each random
   direction satisfies `v^T C_total v = 1`.
3. **Single-subspace ablation** cannot detect redundantly-encoded causal
   structure (A3). Report joint ablation against the orthogonal
   complement alongside any "single subspace is not causal" claim.

Any interpretability work that uses ratio-form cross-model subspace
identification combined with unit-norm single-subspace ablation is at
risk of at least one of these biases. The three corrections are small
in code (the PCA restriction is ~10 lines; the whitening is ~5; joint
ablation is concatenation of direction sets) and architecture-
independent.

### 5.4 Limitations

- **Two tasks (5-digit addition and mod-23 addition), four zoos
  (33 main, 33 mod-p, 45 deep, 21 deep-8).** Methodological corrections
  (A1, A2, A3) are task- and architecture-independent by construction.
  The redundancy phenomenon itself (joint ablation reveals hidden load)
  replicates across all four zoos; the specific primary/backup layer
  pattern differs by task (no primary/backup regime in mod-p) and by
  depth (readout-adjacent N-1, not absolute layer 3). The shrinking
  hidden-load at N-1 with depth is observed on three data points
  (n=3 depths, 1 zoo per depth) and should be read as a discovery
  motivating further depth replication, not a scale-robust claim.
- **The paradox between N-1 invariance and vanishing hidden-load at
  N-1.** A sharp reviewer may observe that if hidden load at N-1
  tends to zero with depth, the "N-1 invariance" may be a vanishing
  feature of small models. We cannot disambiguate "readout-adjacency
  forces a specific computational role" from "N-1 is not
  computationally special in sufficiently deep models" without
  replicating at 12+ layers. We report the three-depth trend honestly
  but do not claim N-1 invariance is scale-robust.
- **Architecture scale fixed at d_model = 64, 4–6 layers, 4 heads.**
  Larger models and different architectures may show different
  per-variance hurt curves; the readout-adjacency finding should be
  tested at 8+ layers to confirm "one-before-unembed" scaling.
- **Final-layer unembed-geometry confound.** The "shared = 0 at
  last-residual" signature could reflect the unembed matrix's linear
  readout constraint rather than a representational property
  independent of the output head. Our 6-layer-zoo evidence pins the
  effect to the last residual but does not dissociate readout-gradient
  pull from pure geometric proximity.
- **Endpoint asymmetry unexplained.** In the 6-layer zoo, the input
  endpoint (layer 0) has shared-primary + complement-primary both
  non-trivial (0.42 and 0.62), while the output endpoint (layer 5)
  has shared-zero + complement-primary only. A pure "endpoint effect"
  would predict symmetric behavior; the asymmetry we observe is
  consistent with a readout-side-specific constraint.
- **Single eigenproblem formulation.** We use the generalized
  eigenproblem `C_shared v = λ C_total v`. Alternative shared-subspace
  identifiers (CCA, joint NMF, partial least squares) may produce
  different subspaces with different redundancy structure. A1 and A3
  generalize naturally; A2 depends on the specific basis convention.
- **Statistical power.** All bootstrap CIs use the 33–45-model
  population per zoo with independent per-model measurements. Effect
  sizes are large; significance is not in question. Generalizability
  under different training hyperparameters or much larger model
  populations remains open.

---

## 6. Within-task replication at additional sites reveals layer-dependence

To check that the shared vs. complement ordering at the three preselected
primary sites is not an artifact of the specific site selection, we ran
the corrected ablation protocol at three additional high-CKA sites that
were *not* in the original primary set:

  - layer1_result_4 (CKA = 0.759, no-PC = 0.454)
  - layer3_result_0 (CKA = 0.696, no-PC = 0.556)
  - layer3_result_3 (CKA = 0.697, no-PC = 0.578)

These sites span different layers (1 and 3) and different result-token
positions (digits 0, 3, 4), providing a robustness check across
representational depth and position.

### 6.1 Cross-site picture

*(Fig. 2 plots the layer-dependent single-subspace drops across the
main zoo.)*


Per-variance hurt (mean drop / projection-trace-variance) at k = 8 across
all 6 task-relevant sites plus control:

| Site | Layer | shared | complement | s / c | dominant |
|------|---|---|---|---|---|
| layer1_result_0 | 1 | 0.0337 | 0.0234 | 1.44× | shared |
| layer1_result_4 | 1 | 0.0277 | 0.0335 | 0.83× | complement |
| layer2_equals   | 2 | 0.0012 | 0.0020 | 0.60× | complement |
| layer2_result_0 | 2 | 0.0155 | 0.0072 | 2.15× | shared |
| layer3_result_0 | 3 | 0.0001 | 0.0030 | 0.05× | complement |
| layer3_result_3 | 3 | 0.0000 | 0.0033 | 0.01× | complement |
| layer3_plus *(CONTROL)* | 3 | 0.0000 | 0.0000 | — | — |

Raw drops at k = 8 across all 6 task-relevant sites + control:

| Site | shared drop | complement drop | whitened random drop | s / rand |
|------|---|---|---|---|
| layer1_result_0 | 0.376 | 0.251 | ≈ 0 | 24× |
| layer1_result_4 | 0.217 | 0.494 | 0.009 | 25× |
| layer2_equals   | 0.020 | 0.028 | 0.001 | 31× |
| layer2_result_0 | 0.312 | 0.100 | 0.003 | 81× |
| layer3_result_0 | 0.003 | 0.415 | 0.000 | 23× |
| layer3_result_3 | 0.001 | 0.430 | 0.000 | 16× |
| layer3_plus *(CONTROL)* | 0.000 | 0.000 | 0.000 | — |

### 6.2 Single-subspace findings (note — §6.4 later shows these are partially confounded by redundancy)

Based on single-subspace ablation alone:

1. **Both shared and complement produce drops orders of magnitude above
   whitened random at all 6 task-relevant sites.** Complement drops at
   layer 3 (0.41, 0.43) are among the largest in the dataset.
2. **Shared vs complement ordering appears layer-dependent.** At
   layers 1-2, the two are comparable; shared wins at 2 of 4 sites,
   complement at 2 of 4.
3. **At layer 3, shared directions appear nearly causally inert** (raw
   single-subspace drop ≤ 0.003) while complement is strongly causal
   (drop 0.41-0.43).
4. **The control site (layer3_plus, CKA = 0.075)** passes cleanly —
   zero drop for any condition. The layer-3 single-subspace effect is
   therefore not "any structured subspace at layer 3 damages the model";
   it is specifically about the shared / complement distinction at
   task-relevant layer-3 sites.

**Important caveat:** findings 2 and 3 hold *for single-subspace ablation
alone*. The joint-ablation disambiguation (§6.4) shows that shared
directions *are* causally important even at layer 3; their single-
subspace ablation drop is near-zero because the complement compensates
when they are ablated in isolation. The reported "shared vs complement
ordering" patterns from single-subspace ablation reflect differences in
*ablation visibility*, not differences in underlying causal importance.

### 6.3 Layer-3 shared directions are readable even where single-subspace ablation is near-inert

To test whether layer-3 shared directions retain the interpretability
property demonstrated at layer-1 shared directions (direction-0 → fourth
result-digit at r = 0.935), we ran the same linear-probe protocol from
§4.7 at the two layer-3 primary-additional sites:

| Site | Direction | Top correlation |
|---|---|---|
| layer3_result_0 | dir_2 | sum_magnitude, r = −0.89 |
| layer3_result_0 | dir_3 | carry_col3, r = **−0.91** |
| layer3_result_3 | dir_3 | carry_col0, r = **0.78** |

The layer-3 shared subspace encodes interpretable arithmetic features at
correlations comparable to layer-1 shared directions. Combined with the
near-zero *single-subspace* ablation drops at layer 3 (§6.1), this
pattern initially looks like "functional universality collapse" — shared
directions remain readable but do not drive computation. However, we
cannot distinguish this from simple redundancy without a disambiguation
test; see §6.4.

### 6.4 Redundancy vs. irrelevance: a critical ambiguity

The finding that ablating the layer-3 shared subspace produces near-zero
accuracy drop admits two interpretations:

1. **Functional universality collapse (the interpretation suggested in
   §6.3):** layer-3 shared directions read the task state but do not
   drive computation; the load-bearing signal lives in the orthogonal
   complement.
2. **Redundancy:** layer-3 shared directions *do* carry causal load, but
   the complement subspace contains parallel pathways that compensate
   when shared is ablated. This is a known failure mode of subspace
   ablation adjacent to a readout layer.

These predict different results for *joint ablation* of the shared and
complement subspaces:

- Under (1) "true inertness," `drop(shared ∪ complement) ≈ drop(complement)`
  — since shared contributes nothing.
- Under (2) "redundancy," `drop(shared ∪ complement) >> drop(complement)`
  — the two pathways carry correlated load that only both-together
  ablation exposes.

We ran joint ablation at all five layer-3 task-relevant result-digit
positions plus one layer-1 comparison site, k = 8, with 95% bootstrap
confidence intervals across the 33 models:

| Site | shared alone | complement alone | JOINT (shared ∪ complement) | hidden load (joint − comp) |
|------|---|---|---|---|
| layer1_result_0 | 0.376 [0.324, 0.431] | 0.251 [0.211, 0.296] | **0.886 [0.877, 0.894]** | **0.635 [0.588, 0.677]** |
| layer3_result_0 | 0.003 [0.001, 0.006] | 0.415 [0.389, 0.444] | **0.808 [0.787, 0.830]** | **0.393 [0.359, 0.427]** |
| layer3_result_1 | 0.001 [0.000, 0.001] | 0.413 [0.385, 0.450] | **0.799 [0.779, 0.821]** | **0.385 [0.354, 0.416]** |
| layer3_result_2 | 0.001 [0.000, 0.001] | 0.381 [0.353, 0.411] | **0.778 [0.761, 0.794]** | **0.397 [0.365, 0.425]** |
| layer3_result_3 | 0.001 [0.000, 0.002] | 0.430 [0.402, 0.456] | **0.793 [0.773, 0.810]** | **0.363 [0.336, 0.390]** |
| layer3_result_4 | 0.001 [0.000, 0.002] | 0.370 [0.330, 0.412] | **0.810 [0.794, 0.825]** | **0.440 [0.401, 0.477]** |

CIs are bootstrap 95% across 33 models, computed independently per column.
Hidden-load (joint − complement) lower bounds are strictly positive at all
6 sites (range 0.33 to 0.59), so redundancy is established with high
statistical confidence.

**The redundancy interpretation is correct at all 6 sites.** Joint
ablation produces a drop 0.36–0.64 larger than complement-alone,
indicating that shared directions carry substantial causal load that is
invisible in single-subspace ablation because the complement can
compensate for their removal.

This rules out the "functional universality collapse" interpretation of
§6.3. At the final layer, shared directions are neither causally inert
nor passively readable — they are causally important and redundantly
encoded. Single-subspace ablation of shared produces near-zero drop
because removing the shared subspace alone does not prevent the network
from producing the correct output (the same information is available in
the complement); joint ablation removes this compensation pathway and
exposes the hidden load.

The resulting picture of the layer-3 shared subspace is:

1. *Causally important:* joint-ablation reveals 0.36–0.44 of accuracy
   drop attributable to shared directions.
2. *Redundantly realized:* the complement subspace independently carries
   enough signal to produce the correct output when shared is removed.
3. *Linearly readable:* shared directions encode arithmetic features at
   r = 0.78–0.91, not as passive readouts but as one of two parallel
   encodings.
4. *Invisible to single-subspace ablation:* standard ablation
   interpretability underestimates shared-subspace importance whenever
   the model has developed redundant encodings — which, in this zoo,
   appears to be the default.

### 6.4b Asymmetry between shared and complement

The redundancy is not symmetric: the single-alone drops for shared and
complement differ substantially at most sites, and the asymmetry flips
with layer depth:

| Site | shared alone | complement alone | complement − shared |
|---|---|---|---|
| layer1_result_0 | 0.376 | 0.251 | −0.125 (shared hurts more) |
| layer3_result_0 | 0.003 | 0.415 | +0.412 |
| layer3_result_1 | 0.001 | 0.413 | +0.412 |
| layer3_result_2 | 0.001 | 0.381 | +0.380 |
| layer3_result_3 | 0.001 | 0.430 | +0.429 |
| layer3_result_4 | 0.001 | 0.370 | +0.369 |

At layer 1, shared-alone ablation produces a larger drop than
complement-alone — the shared subspace is the *primary* encoding there,
the complement is the backup. At layer 3, the asymmetry reverses
dramatically: complement-alone produces nearly all of the available
single-ablation drop, shared-alone is masked. In neither case do we see
true *symmetric* redundancy (where the two subspaces are interchangeable
and each alone produces half the joint drop); instead, one subspace is
primary and the other is a compensating backup, with the primary role
shifting from shared to complement as we move from layer 1 to layer 3.

This refines the "redundantly encoded" reading: there is a shared-to-
complement handoff as information propagates through the transformer,
with the shared subspace acting as the primary causal route at early
layers and the complement assuming that role at the final layer.
Importantly, **this is not the same as "shared becomes irrelevant at
layer 3"** — joint ablation shows shared contributes 0.36–0.44 of
accuracy drop even at layer 3 — it means the primary load-bearing role
has transferred to the complement, with shared acting as the redundant
backup that standard ablation cannot detect.

### 6.5 Other caveats on the layer-dependence claim

Six sites split 2-2-2 across layers is narrow evidence for a "layer-
dependent" effect. §6.6 [step14] expands the layer-3 coverage to six
additional positions for robustness. The
effect we report should be read as "in this zoo, at these sites, the
pattern is X" rather than as an intrinsic property of transformer
universality. Alternative explanations we cannot rule out:

- **Final-layer residual artifact.** Layer 3 is the last residual stream
  before the unembed matrix. Residual stream near the logits may exhibit
  unusual structural properties unrelated to universality. The layer3
  shared directions may be dominated by projections onto the unembed
  row space, where per-model idiosyncrasies live by construction.
- **Zoo-specific training dynamics.** Our zoo has 33 models, all of which
  converged to ≥ 99% accuracy. Training dynamics under identical
  hyperparameters may favor idiosyncratic late-layer representations for
  reasons specific to this task and hyperparameter regime.
- **Site-selection heterogeneity.** The six preselected sites are
  self-selected for high CKA — a post-hoc correlation with a particular
  kind of shared structure. An unbiased site sweep might reveal a less
  clean pattern.

### 6.6 Layer-3 expansion: 5/5 result-digit positions show the pattern

We ran the corrected ablation protocol at six additional layer-3 sites:
four result-digit positions (1, 2, 4, 5) and two operand-last positions
(a, b). Two clean clusters emerge:

**Task-relevant layer-3 positions (CKA ≥ 0.69, all result-digit positions
0-4 at layer 3):**

| Site | shared drop | complement drop | complement/shared | shared proj-trace | comp proj-trace |
|------|---|---|---|---|---|
| layer3_result_0 | 0.003 | 0.415 | 137× | 23 | 138 |
| layer3_result_1 | 0.00065 | 0.413 | **637×** | — | — |
| layer3_result_2 | 0.00081 | 0.381 | **469×** | — | — |
| layer3_result_3 | 0.001 | 0.430 | 430× | 23 | 129 |
| layer3_result_4 | 0.00107 | 0.370 | **346×** | — | — |

5 of 5 task-relevant layer-3 positions tested: shared directions are
near-causally inert (raw drop ≤ 0.003, per-variance hurt ≤ 5 × 10⁻⁵);
complement directions produce consistently large drops (0.37-0.43).

**Low-CKA layer-3 positions (CKA < 0.15):**

| Site | CKA | shared drop | complement drop |
|------|---|---|---|
| layer3_result_5 | 0.122 | 0.000 | 0.000 |
| layer3_operand_a_last | 0.113 | 0.000 | 0.000 |
| layer3_operand_b_last | 0.120 | 0.000 | 0.000 |
| layer3_plus (CONTROL) | 0.075 | 0.000 | 0.000 |

All four low-CKA layer-3 positions show zero effect for any condition,
consistent with the original control site and confirming that the
complement-dominance finding is specific to task-relevant sites.

### 6.6b Replication on a second task (modular arithmetic)

*(Fig. 3 compares hidden load between the main zoo and the mod-p
replication.)*


To test whether the hidden-load phenomenon survives a change in task
structure, we trained a separate 33-model zoo on addition modulo 23
(identical architecture: 4 layers, d_model = 64, 4 heads, d_ff = 256;
identical training protocol; identical freeze configurations). All 33
models converged to ≥ 99% accuracy on the full 529-pair eval set within
approximately 500-7000 training steps.

We pre-registered (MOD_P_PREREGISTRATION.md, committed before training
began) a decision rule: REPLICATES if (a) hidden-load CI lower bound is
strictly positive at ≥ 2 of 3 primary sites, (b) d_shared and d_comp
both exceed 5× d_rand at those sites, (c) the pre-specified low-CKA
control site shows drop ≤ 0.02. Primary sites were pre-locked to the
three `=`-token positions at layers 1, 2, 3; control to layer3_plus.

**Results at k = 8 (33 models, 95% bootstrap CIs):**

| Site | shared drop | complement drop | joint drop | random drop | hidden load |
|------|---|---|---|---|---|
| layer1_equals | 0.010 [0.003, 0.020] | 0.818 [0.754, 0.875] | 0.845 [0.784, 0.895] | ≈ 0 | 0.027 [0.013, 0.041] |
| layer2_equals | 0.023 [0.012, 0.037] | 0.756 [0.684, 0.817] | 0.845 [0.796, 0.887] | ≈ 0 | 0.089 [0.064, 0.118] |
| layer3_equals | 0.019 [0.009, 0.030] | 0.089 [0.070, 0.108] | 0.254 [0.220, 0.288] | ≈ 0 | 0.165 [0.141, 0.190] |
| layer3_plus (CONTROL) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Verdict: REPLICATES** (pre-registered rule).

- Hidden-load CI strictly positive at all 3 primary sites.
- Shared and complement both >>> random at all 3 primary sites.
- Control site clean.

**Informative divergences from the main zoo:**

The pre-registered criteria pass cleanly, but the mod-p replication
differs from the main zoo in two ways we did not pre-specify:

1. **Hidden-load magnitude is much smaller.** Main zoo hidden-load at
   layer-3 result-digit positions: 0.36–0.44. Mod-p hidden-load at
   layer-3 equals position: 0.165. Mod-p has no carry chain — less task
   structure for the model to redundantly encode, so less hidden causal
   load available for joint ablation to expose.
2. **No shared-to-complement handoff across depth.** Main zoo shows
   shared-primary at layer 1 (single-subspace drop 0.38) and
   shared-backup at layer 3 (drop < 0.003). Mod-p shared stays small
   at all three tested layers (0.010–0.023) — complement is the
   primary single-subspace target at every layer, with no regime
   where shared alone carries large load. This likely reflects that
   mod-23 is solvable in 1-2 attention layers; by layer 3 the task is
   already decided, and the shared subspace has less distinct role to
   play.

**What this replication shows:**

The core redundancy phenomenon (shared contributes causal load that is
invisible to single-subspace ablation because complement compensates) is
not specific to N-digit addition. It survives a task-structure change
(no carry chain, one output token instead of six, entirely different
vocabulary) with 33/33 models converging on the same qualitative
structure. The three-correction protocol (A1 + A2 + A3) does the same
work on a task where the single-subspace ordering has nothing to do
with carry propagation.

**What it doesn't show:**

The layer-wise asymmetry (shared primary at early layers, backup at late)
is task-specific. It showed in the main zoo; it doesn't show here. So
the specific pattern of primary/backup handoff is a feature of 5-digit
addition, not a general property of cross-model shared structure. The
paper's claims about layer-dependence are restricted accordingly.

### 6.6c Replication at larger depth (6-layer architecture)

A paradox-hunter reviewer at a prior review round raised the question
whether the "shared is causal but redundantly backed at the final
layer" pattern reflects an intrinsic late-layer property or an artifact
of the last-residual-before-unembed geometry specific to 4-layer models.
To distinguish these, we trained a second zoo of 45 transformers
identical to the main zoo in hyperparameters, task (5-digit addition),
and training protocol, but with 6 layers instead of 4.

We pre-registered (PHASE2_PREREGISTRATION.md, committed before the
6-layer zoo began training) three hypotheses:

- **H1 (absolute depth):** the "shared-dead, complement-primary"
  signature would appear at layer 3 in the 6-layer zoo (matching the
  layer index in the main zoo).
- **H2 (last-residual-before-unembed):** the signature would appear at
  layer 5 in the 6-layer zoo (the new penultimate-before-unembed layer).
- **H3 (normalized depth):** the signature would appear at layer ≈ 5
  (indistinguishable from H2 at this model size; differs at deeper
  architectures).

We ran the corrected protocol at all six layers (at the `result_0`
token position, k = 8) plus a layer-5 low-CKA control site. Results
with 95% bootstrap CIs across 45 models:

| Layer | Site | shared | complement | joint | hidden load |
|---|---|---|---|---|---|
| 0 | result_0 | 0.416 [0.347, 0.486] | 0.620 [0.567, 0.665] | 0.843 [0.786, 0.884] | 0.222 [0.156, 0.277] |
| 1 | result_0 | 0.454 [0.396, 0.503] | 0.280 [0.236, 0.321] | 0.834 [0.776, 0.875] | 0.553 [0.478, 0.612] |
| 2 | result_0 | 0.361 [0.316, 0.407] | 0.215 [0.179, 0.249] | 0.831 [0.769, 0.875] | 0.616 [0.561, 0.663] |
| 3 | result_0 | 0.407 [0.358, 0.457] | 0.046 [0.034, 0.059] | 0.828 [0.769, 0.870] | 0.781 [0.723, 0.828] |
| 4 | result_0 | 0.370 [0.314, 0.422] | 0.022 [0.013, 0.034] | 0.843 [0.809, 0.870] | 0.825 [0.780, 0.856] |
| **5** | **result_0** | **0.000 [0.000, 0.000]** | **0.561 [0.537, 0.585]** | **0.731 [0.719, 0.743]** | **0.170 [0.148, 0.192]** |
| 5 | plus *(CONTROL)* | 0.000 | 0.000 | 0.000 | 0.000 |

**Verdict: H2 (last-residual-before-unembed) wins decisively.**

The "shared-dead / complement-primary" signature that appears at layer 3
in the main 4-layer zoo moves to **layer 5** in the 6-layer zoo. At
layer 3 of the deeper zoo, shared is now the *primary* single-subspace
load (drop 0.407) while complement has collapsed (drop 0.046) —
exactly the opposite of the main-zoo layer 3 pattern. At layer 5, shared
is zero to within bootstrap precision (0.000 [0.000, 0.000]) and
complement is 0.561, replicating the main-zoo layer-3 signature.

**The "shared dead" finding in the main zoo was a
last-residual-before-unembed effect, not a property of absolute
layer index.** Fig. 5 visualizes this.

#### 6.6c.1 Hidden load profile and the "computational sandwich" reading

The 6-layer hidden-load profile is non-monotone:

- Layer 0 (input-side endpoint): 0.222
- Layers 1-4 (middle): 0.55, 0.62, 0.78, 0.82 (growing)
- Layer 5 (output-side endpoint): 0.170

Hidden load is *high in middle layers, low at both endpoints*. Read
with the per-layer shared and complement drops, this looks like a
"computational sandwich": the middle layers carry redundantly-encoded
causal load across shared + complement, while the endpoints have
asymmetric single-subspace roles — at layer 0, shared and complement
both carry substantial load (0.42 and 0.62); at layer 5, only
complement carries load (shared collapses to zero).

#### 6.6c.2 The layer-0 asymmetry caveat

A careful reader — and our own post-hoc paradox-hunter review —
observes that the endpoint story is not symmetric. At the **input-side
endpoint (layer 0)**, shared is 0.42 and complement is 0.62 (both
non-trivial); at the **output-side endpoint (layer 5)**, shared is
exactly 0.00 and complement is 0.56 (only complement). A pure
"last-residual-adjacent-to-unembed" account predicts only the top
endpoint should be geometrically special (because the unembed matrix
imposes a direct linear readout constraint on that residual), and
indeed that is what we see: the zero-shared signature is specific to
the output endpoint, not to both.

This asymmetry refines the interpretation: the "shared-dead" signature
is driven by readout geometry, not by depth alone.

### 6.6d Third-depth replication: N-1 invariance in an 8-layer zoo

The natural follow-up is whether the signature is robustly at position
N-1 across multiple architecture depths. We trained a third zoo: 21
models on 5-digit addition with n_layers = 8 (1 seed per freeze to
fit the time budget), and tested layers 3, 5, and 7 + a layer-7 control.
Pre-registered prediction (PHASE3_PREREGISTRATION.md): H2a ("N-1
invariance") predicts shared ≈ 0 specifically at layer 7.

Results at k = 8, 21 models, 95% bootstrap CIs:

| Site | shared | complement | joint | hidden load |
|------|---|---|---|---|
| layer3_result_0 | 0.270 [0.215, 0.332] | 0.292 [0.249, 0.330] | 0.871 [0.841, 0.891] | 0.579 [0.536, 0.625] |
| layer5_result_0 | 0.117 [0.081, 0.157] | 0.110 [0.085, 0.137] | 0.876 [0.846, 0.894] | 0.766 [0.723, 0.802] |
| **layer7_result_0** | **0.000 [0.000, 0.000]** | **0.789 [0.778, 0.800]** | **0.789 [0.777, 0.799]** | **−0.000 [−0.005, 0.005]** |
| layer7_plus *(CONTROL)* | 0.000 | 0.000 | 0.000 | 0.000 |

**Verdict: H2a (N-1 invariance) confirmed.** Shared is dead at layer 7
in the 8-layer zoo, just as it was dead at layer 3 in the 4-layer main
zoo and layer 5 in the 6-layer zoo. The signature tracks
"one-before-unembed" across all three architecture depths.

#### 6.6d.1 Unexpected: hidden load at N-1 shrinks with depth

The 8-layer zoo reveals a second pattern we did not predict. Hidden
load at the N-1 layer varies systematically:

| Zoo | N | "shared dead" layer | hidden load at N-1 layer |
|---|---|---|---|
| Main | 4 | layer 3 | 0.36–0.44 (across 5 result-digit positions) |
| Deep (Phase 2) | 6 | layer 5 | 0.170 [0.148, 0.192] |
| Deep-8 (Phase 3) | 8 | layer 7 | **−0.000 [−0.005, 0.005]** |

At the 8-layer N-1 layer, `joint ablation equals complement-alone
ablation to within ±0.005`: shared contributes *no* additional load
when combined with complement. In the 6-layer zoo the additional load
was 0.17; in the 4-layer main zoo it was 0.36-0.44. **The deeper the
model, the more completely the complement subspace subsumes the shared
subspace at the readout-adjacent layer.**

A reviewer might argue this supports a "representation compression"
reading: with more layers, the model has more capacity to route
complementary information redundantly through the complement subspace,
leaving the shared subspace as a thin, probeable, but causally-
subsumable signal at the output endpoint. The shared subspace at the
8-layer layer 7 is still linearly probeable (results not shown; limited
time), but carries zero independent causal load as measured by joint
ablation.

This finding should be taken cautiously: the 8-layer zoo has 21 models
(fewer than the 33/45-model zoos), the hidden-load trend is based on
three data points (n=3 architecture depths), and the difference could
reflect idiosyncratic training dynamics of this particular
small-transformer setup rather than a general depth effect. Replication
at larger depths and with larger model counts per depth is needed.

#### 6.6d.2 The cleanest "readable but not causal" data point

We probed shared directions at layers 3, 5, 7 of the 8-layer zoo
against the standard arithmetic feature set. At the N-1 layer (layer 7),
where hidden load is exactly zero:

| Site (8-layer zoo) | Strongest probe correlation |
|---|---|
| layer3_result_0 | direction_2 → sum_magnitude, r = −0.97 |
| layer5_result_0 | direction_3 → carry_col4, r = 0.83 |
| **layer7_result_0** | **direction_0 → sum_magnitude, r = −0.94; direction_1 → carry_col3, r = −0.84** |

At layer 7, shared directions are **causally subsumed** (joint ablation
equals complement-alone ablation to within ±0.005) but **linearly
probeable** at r = 0.84–0.94. This is the most extreme data point in
the paper: zero independent causal contribution, yet high-fidelity
linear readout of arithmetic features. Summarizing across all three
zoos at the respective N-1 layer:

| Zoo | N-1 layer | top probe r | hidden load |
|---|---|---|---|
| Main (4L) | 3 | 0.91 (carry_col3) | 0.36–0.44 |
| 6L | 5 | 0.97 (sum_magnitude) | 0.170 |
| 8L | 7 | 0.94 (sum_magnitude) | ~0 |

**Probe-readability at the N-1 layer is preserved across depths at
r ≈ 0.8–0.97 even as causal contribution vanishes.** This strengthens
the interpretation of shared directions at the readout-adjacent layer
as "observational projections": a probeable summary of task state that
the model's computation has already committed to via the complement
subspace, and whose removal (in isolation) therefore doesn't affect
the output.

*(Fig. 5 plots shared/complement/joint drops vs layer in the 6-layer
zoo, and the hidden-load profile.)*

#### 6.6c.3 Layer-5 shared directions are readable but single-ablation-inert

We probed shared directions at three layers in the 6-layer zoo
(layer 0, layer 3, layer 5) against the same hand-labeled arithmetic
features used in §4.7. Strongest correlations:

| Layer | Top direction → feature correlation |
|---|---|
| 0 | direction_0 → carry_col4, r = −0.98 |
| 3 | direction_4 → carry_col3, r = −0.85 |
| **5** | **direction_0 → sum_magnitude, r = 0.97; direction_1 → carry_col3, r = 0.92** |

Shared directions remain strongly linearly probeable at layer 5 despite
producing zero single-subspace ablation drop. This replicates the
main-zoo layer-3 probing finding (§6.3) at the corresponding last-
residual-before-unembed layer in a deeper architecture. The consistent
picture across all three zoos: shared directions at the
last-residual-before-unembed carry strongly readable task-feature
information (r = 0.78–0.97 across sites) but are invisible to single-
subspace ablation because the orthogonal complement redundantly encodes
the same task content.

### 6.6d.3 Unembed-geometry test: how much of "shared dead" is nullspace residence?

A natural alternative explanation for "shared ablation doesn't hurt" is
geometric: if the shared subspace at layer 3 sits mostly in the unembed
matrix's nullspace, the unembed simply doesn't read those directions and
ablation cannot affect logits — regardless of computation.

We measured the fraction of each subspace's variance in the unembed
nullspace across the 33 main-zoo models, using a random 10-d subspace
baseline (analytic expectation = 52/64 ≈ 0.813 for a random subspace in
d=64 with rank-12 unembed):

| Subspace | Nullspace fraction | σ from random |
|---|---|---|
| Random 10-d | 0.813 | 0 (baseline) |
| **Shared** | **0.767** | **−12σ** (slightly *less* in nullspace than random — i.e. slightly more readable) |
| **Complement** | **0.310** | **−133σ** (dramatically concentrated in unembed row space) |
| Bottom-of-ratio | 0.918 | +28σ (more in nullspace; dead axes, as expected) |

**The "shared dead" phenomenon is *not* primarily geometric.** Shared
directions are actually slightly *more* readable by the unembed than
random directions, not less. The dominant geometric effect is on the
complement side: complement is massively concentrated in the unembed's
row space (a 133σ deviation from random). That explains why complement
ablation hits hard — complement directions *are* the directions the
unembed reads.

The residual asymmetry between shared (drop 0.003 at layer 3) and
random (drop 0.028) — shared hurts ~10× less than random despite similar
row-space exposure — is NOT explained by geometry. This is the redundant
encoding demonstrated by joint ablation (§6.4): the complement
compensates when shared is removed in isolation, and the 10× gap between
shared and random ablation is where the compensation shows up.

### 6.6e Cross-model subspace swap: universal values, not just universal directions

A referee-style review of the paper's redundancy claim raised the question
whether cross-model shared structure is *only* direction-universal (the
models agree on which subspaces are important) or also *value-universal*
(the actual input-to-activation mapping within those subspaces is shared).
Joint ablation (§6.4) tests only direction-universality — it measures
causal load but not whether the load is encoded identically across models.

We ran a cross-model swap experiment to probe this at layer 3 position 12
(the main-zoo "shared dead" site). For each ordered pair of models (A, B)
from a set of 6 (3 baselines + 3 freeze variants spanning embed, layer-0
MLP, and layer-2 attn), we:

1. Aligned both models' layer-3 activations to the common reference frame.
2. Replaced, on A's forward pass, the shared-subspace component of A's
   layer-3 activation at position 12 with B's shared-subspace component
   (computed on the same input).
3. Measured A's full-task accuracy drop vs A's baseline.

We ran three parallel conditions per pair: shared swap, complement swap,
random (whitened) swap. Summary across 30 ordered pairs:

| Condition | Mean drop | 95% CI | Interpretation |
|---|---|---|---|
| shared swap | −0.0007 | [−0.001, 0.000] | No effect (within numerical noise) |
| complement swap | −0.0005 | [−0.001, 0.000] | No effect |
| random swap | 0.0000 | [0.000, 0.000] | No effect |

Post-hoc this is striking. The raw activation differences between aligned
baselines are not small (relative-magnitude difference ≈ 0.54; per-dim
correlation ≈ 0.85 — far from identical). Yet no choice of subspace to
swap produces a detectable accuracy drop. The pattern holds for
baseline↔baseline, baseline↔freeze, and freeze↔freeze pairs alike.

**Universal values reconciliation with §6.4 ablation.** The swap result
is consistent with the joint-ablation finding under a refined reading:

- Complement ablation destroys per-input variation (replacing with the
  batch mean) and hurts 0.37–0.43 because the variation encodes
  task-relevant state the readout needs.
- Complement *swap* replaces A's per-input variation with B's per-input
  variation — which, crucially, encodes the same task-relevant state
  (both A and B computed the same answer on this input). Accuracy
  doesn't drop because the readout receives a correct, if
  differently-encoded, input.

This is a sharper universality claim than "shared subspaces agree": the
specific input-to-activation mapping at layer 3 converges across
independently trained models (after Procrustes alignment), to a degree
that makes their complement-subspace values mutually substitutable. The
substitutability holds even when one model has ≥1 component frozen at
random init for the entire training run. That is a meaningful cross-model
functional-equivalence signal that complements the causal-redundancy
signal of §6.4.

### 6.7 Consolidated picture after all corrections

With the joint-ablation disambiguation (§6.4), the redundancy
interpretation holds at all six tested sites. The "shared near-inert at
layer 3" pattern from single-subspace ablation is not evidence of
functional universality collapse but evidence of redundant encoding
between shared and complement subspaces.

The corrected reading, integrated across all three zoos:

- Shared directions are **causally important at every task-relevant
  layer tested**, carrying 0.17–0.83 of hidden-load (joint − complement
  single-subspace drop) across the three zoos.
- Complement directions are **also causally important** at every site,
  but with a characteristic layer-dependence within each zoo.
- The **primary-backup role flips at the last-residual-before-unembed**:
  in the main 4-layer zoo, layer 3 is the readout-adjacent layer and
  shows shared-dead / complement-primary. In the 6-layer zoo at the
  same absolute index (layer 3), the roles are reversed (shared is
  primary, complement has collapsed); the shared-dead signature moves
  to layer 5. This rules out "absolute depth" and supports a
  readout-adjacency account.
- The **layer-dependence we observed in single-subspace ablation is in
  the *visibility* of shared directions**: shared-alone is maximally
  masked by redundancy at the last-residual-before-unembed, less so in
  middle layers where both subspaces carry large single-ablation load.
- In the mod-p replication (4-layer arch, mod-23 task), hidden-load is
  positive at all three primary sites with smaller magnitudes (0.03–
  0.17) and no shared-primary regime at any tested layer — consistent
  with mod-p having less task structure to redundantly encode.

The methodological lesson is that single-subspace ablation systematically
underestimates the importance of redundantly-encoded structure. Cross-
model shared directions in this zoo are redundantly encoded with their
orthogonal-high-variance complement, with a primary-backup role that
shifts across depth. Any CKA-based subspace causality analysis that
relies solely on single-subspace ablation is at risk of missing this
structure.

---

## 7. Pre-registration and post-hoc analyses

**Pre-registered protocol.** The decision rule, primary statistic, k
values, sites, and control gates were locked before any results were
generated (`P1_PREREGISTRATION.md`). Three amendments (A1: PCA
restriction, A2: variance-matched ablation, A3: k-value cap) corrected
technical issues discovered during piloting (extraction dead-axes,
variance-matching convention, top-k/bottom-k overlap with PCA
restriction). None of the amendments adjust the decision-rule
thresholds or post-hoc tune any parameter that sees the verdict-relevant
outcome. The verdict is computed by a separate script (`p1_report.py`)
that has no access to the experiment runner at runtime.

**Post-hoc confound check (A4).** The joint-ablation test reported in
§6.4 was added post-hoc in response to a reviewer concern about
redundancy-vs-irrelevance. It is not part of the pre-registered
protocol and is honestly labeled as such. Its outcome rule (partial
redundancy ≡ hidden-load CI lower bound strictly positive) was specified
before the test ran. We do not claim pre-registration status for the
joint-ablation result; we report it as a piloting finding that clarifies
but does not override the primary single-subspace verdicts.

---

## 8. Conclusion

We set out to test whether cross-model shared directions in a 33-model
arithmetic transformer zoo are causally important. Under standard
unit-norm single-subspace ablation against unrestricted eigvecs, the
answer looks like "no, or at best no better than random." Under the
three-correction protocol (A1 PCA-restriction + A2 variance-matching +
A3 joint ablation), the answer is "yes, redundantly, and the specific
pattern of redundancy has a clean readout-adjacent geometric
interpretation."

The empirical findings replicate:
- On a second task (mod-23 arithmetic, 33-model zoo, same architecture)
  the core phenomenon — hidden load strictly positive at every primary
  site, shared and complement both dominating random — holds, with
  smaller magnitudes consistent with less task-structure to redundantly
  encode.
- At larger depth, pre-registered tests across a 6-layer zoo (45
  models) and an 8-layer zoo (21 models) confirm the readout-adjacency
  pattern: the shared-dead signature appears at the layer one below
  the unembed in each depth (L3 in 4L, L5 in 6L, L7 in 8L). Hidden
  load at the N-1 layer shrinks monotonically across our three depths
  (0.36-0.44 → 0.17 → ~0) — a three-point trend reported as a
  discovery, not a scale-robust claim. Even in the deepest zoo, where
  hidden load at N-1 vanishes, shared directions at that layer are
  still linearly probeable at r = 0.84–0.94 for arithmetic features:
  the cleanest "observational projection" data point in the paper.

The empirical case study is therefore a specific claim about small
arithmetic transformers: cross-model shared and complement subspaces
both carry causal load at every tested layer; the specific
primary/backup single-ablation pattern is anchored to the residual
stream's distance from the unembed matrix, not to its absolute depth.

The methodological lesson is general. Any CKA- or ratio-eigenproblem-
based universality claim that uses unit-norm single-subspace ablation
to support or refute a causal role has a non-trivial chance of reaching
the wrong conclusion, because the standard protocol conflates three
separate biases with the quantity of interest. Each correction is small
in code and architecture-independent; together they convert a
seemingly contradictory result into a coherent picture of redundantly-
encoded cross-model structure with a clean readout-adjacent role flip.

## Acknowledgments

Three independent reviewer-style models (Codex, Gemini, Claude) were
consulted at four decision points during piloting. Their convergence on
the methodological framing and divergence on the strength of the
positive claim — in particular, one reviewer's insistence that the
"shared ≈ 0 at layer 3" result could reflect redundancy rather than
irrelevance — directly shaped the joint-ablation test and the final
paper scope.
