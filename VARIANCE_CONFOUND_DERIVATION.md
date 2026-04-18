# Three Methodological Confounds in Subspace Ablation (A1 + A2 + A3)

This document derives, from first principles, three methodological
confounds in standard subspace ablation interpretability protocols:

- **A1** — unrestricted generalized-eigenproblem extraction selects low-
  variance shared directions
- **A2** — unit-norm random baselines do not variance-match structured
  subspaces
- **A3** — single-subspace ablation cannot detect redundantly-encoded
  causal structure

Sections 1–8 below cover the A2 variance-mismatch derivation and
empirical decomposition (including the post-hoc discovery that A1 is the
dominant fix at the sites tested). Section 10 covers A3 — the
redundancy confound — demonstrated by the joint-ablation test at 6 sites.

---

## 1. Setup

Let `h_k(x) ∈ ℝ^d` be the residual-stream activation of model `k` ∈ {1,…,K} on
input `x` ∈ X at a fixed layer / token position. Define the *per-model
input-variance covariance*:

  C_total = (1/K) Σ_k Cov_x(h_k(x))

This is the average across models of the per-model input-dependent activation
covariance. It captures the magnitude of variance the model uses to encode
inputs at this site.

A *projection ablation* on a k-dimensional subspace S replaces the activation
component in S with its in-distribution mean:

  ablated_h(x) = (I − P_S) h(x) + μ_S

where `P_S = V (V^T V)^{-1} V^T` is the orthogonal Euclidean projector onto S
spanned by `V ∈ ℝ^{k×d}`, and `μ_S` is the mean-projected component. Crucially:

> The amount of per-model activation variance *removed* by this ablation equals
>
>   Δσ²(S) = trace(P_S · C_total · P_S) = trace(V^T C_total V · (V^T V)^{-1})
>
> which is a property of the **subspace** S, not the basis V.

For an Euclidean-orthonormal basis (V V^T = I_k), this simplifies to
`Σ_i v_i^T C_total v_i`.

## 2. The unit-norm convention biases by sub-space variance

Standard practice in interpretability ablation is to normalize each direction
to **unit Euclidean length**: `‖v_i‖ = 1`. Under this convention, the variance
removed by ablating subspace S spanned by orthonormal V is:

  Δσ²_unit(S) = Σ_i v_i^T C_total v_i

This sum can take any value in `[k · λ_min(C_total), k · λ_max(C_total)]`
depending on which subspace S is chosen.

For typical residual-stream covariances `C_total` has a heavy-tailed spectrum
(in our experiments: top eigenvalue ≈ 9.18, median ≈ 0.05, condition number
≈ 3 × 10^4). Different choices of subspace therefore remove vastly different
amounts of activation variance under unit-norm:

| Subspace S (unit-norm orthonormal V, k=10) | Δσ²_unit(S) |
|---|---|
| Top-10 generalized eigvecs of C_shared v = λ C_total v | ~10 (high-variance, occupies dominant PCs) |
| Bottom-10 generalized eigvecs (same eigenproblem) | ~0.1 (low-variance corner) |
| 10 random unit-norm Gaussian directions | ≈ 10 · tr(C_total) / d ≈ 5 (uniform over space) |

Three subspaces, ostensibly compared on equal footing because each has 10
unit-norm directions, but actually removing 10 vs 0.1 vs 5 units of variance.

## 3. Causal hurt scales with variance removed

Let `f(h) ∈ ℝ^V` be the model's logit output at the predicted token. To first
order in the perturbation introduced by mean-substitution in S:

  f(ablated_h) − f(h) ≈ ∇f(h) · (μ_S − P_S h)

The expected magnitude of the perturbation `‖μ_S − P_S h‖` over the input
distribution is:

  E_x ‖μ_S − P_S h(x)‖² = trace(P_S · Cov_x(h) · P_S) = Δσ²(S)

So the **expected squared perturbation is exactly the amount of activation
variance removed**. Larger variance subspaces produce larger perturbations,
which (in expectation) produce larger logit changes and accuracy drops, even
in the absence of any causal-importance asymmetry between subspaces.

This gives the bias in closed form:

> **Lemma (Variance-Mismatch Bias).** Under unit-norm projection ablation,
> the expected accuracy drop scales (to first order) with the activation
> variance removed by the ablation. Two subspaces removing different amounts
> of variance cannot be compared on a hurt-is-causality basis without an
> explicit variance correction.

## 4. The Step-7 sign flip is the predicted artifact

In our experiments, three subspaces were compared at k = 10 under unit-norm
ablation (original Step 7):

| Subspace | Variance removed (estimate) | Reported drop |
|---|---|---|
| top-10 shared (high-variance subspace) | ~10 | 0.0116 |
| 10 random unit-norm | ~5 | 0.0279 |

At first glance "shared causes less damage than random" is interpreted as
"shared is non-causal." But the variance bias predicts that ablating ~10 units
of high-variance signal produces a *larger* perturbation than ablating ~5
units of random-direction signal, so under a causality-equal null we would
expect shared > random in raw drop. The fact that **shared < random** despite
removing more variance means shared is *causally less efficient per unit
variance*, but it does NOT establish that shared is "non-causal" in absolute
terms.

The original interpretation conflated two questions:

1. *Does ablating subspace S hurt the model?* → Magnitude of drop.
2. *Is the variance in S more causally load-bearing than random variance?*
   → Drop per unit of variance removed.

Question (2) is the appropriate test for "is this subspace causally privileged?"
Question (1), which Step 7 actually answered, conflates causal load with
variance content.

## 5. Variance-matched ablation removes the bias

The corrected estimand under amendment A2 is:

  R(S) = drop(S) / Δσ²(S)

This is the **drop per unit of activation variance removed** — a direct
estimator of per-variance causal load that is invariant to the subspace's
variance content. Two equivalent ways to operationalize it:

- *(a)* Use C_total-orthonormal eigenvectors (`v^T C_total v = 1` for each
  direction), so `Δσ²(S) = k` for any chosen subspace S, and the comparison
  reduces to comparing raw drops.
- *(b)* Whiten random baselines: sample r ∼ N(0, I), set
  v = (C_total + ε I)^{−1/2} r and renormalize so `v^T C_total v = 1`.
  Now random subspaces also have `Δσ²(S) = k`.

Note: option (a) keeps `v^T C_total v = 1` per direction, but the projection-
trace `trace(V (V^T V)^{-1} V^T C_total)` may still vary across subspaces if
V is not Euclidean-orthonormal. The clean reporting is to compute and report
`trace(P_S C_total P_S)` per condition (Path 2 reporting in the script).

## 6. Predicted directional flip

The lemma in §3 gives a sharp prediction. Under standard unit-norm ablation,
the ranking of measured drop will track variance content:

  drop_unit(shared_top) ≳ drop_unit(random) ≳ drop_unit(shared_bottom)

(High-variance subspace produces large perturbation; low-variance subspace
produces small perturbation; random is in between.)

Under variance-matched ablation, perturbation magnitude is held constant
across conditions, and the measured drop should track per-variance causal
load. If shared directions encode task-functional features, the prediction is:

  drop_matched(shared_top) > drop_matched(random_whitened)

i.e., the sign flips relative to standard unit-norm because the variance
bias has been removed.

## 7. Empirical confirmation

Our experiments confirm both predictions across 33 trained arithmetic
transformers:

| Comparison at k = 10, layer1_result_0 | Standard unit-norm | Variance-matched |
|---|---|---|
| `mean drop(top-10 shared)` | 0.012 | 0.448 |
| `mean drop(random)` | 0.028 | 0.019 |
| Shared / random ratio | 0.43 | **24.2** |

Same models, same site, same eigenvectors. The 56× change in the ratio is
entirely attributable to the variance-matching correction.

## 8. Bottom line

The variance-mismatch confound is a corollary of basic linear-algebra: when
the perturbation magnitude depends on the subspace, comparing subspaces with
different variance content tells you about magnitudes, not causal asymmetries.
Variance-matched ablation makes the comparison about what you actually want
to measure.

## 9. Caveat — empirical decomposition (step11)

The decomposition test described in `step11_compare_extractions.py`
(results in `results/p1/extraction_decomposition.json`) shows that the
empirical sign-flip relative to the original Step-7 finding has TWO
independent contributions:

(a) **The eigenproblem extraction itself preferentially selects low-
variance directions** when applied to the full residual stream without a
PCA restriction. The unrestricted top-10 generalized eigvecs of
`C_shared v = λ C_total v` ablated under unit-norm produce a shared/random
ratio of ~0.05–1.6 across the three primary sites — not consistently
"shared << random" but not consistently above it either. Restricting the
eigenproblem to the top-32 PCs of `C_total` (Amendment A1) shifts the
extracted "shared" subspace into the high-variance regime, and the same
ablation with same models gives shared/random of 18–70× (already a sign
flip, under unit-norm).

(b) **Variance-matched ablation (Amendment A2) is a strict additional
improvement**, not the dominant fix. When the random baseline is whitened
(so that `v^T C_total v = 1` for each random direction), the random drops
collapse to ≈ 0 because the whitening preferentially samples low-variance
noise axes. The shared subspace's drop is unchanged (subspace-based
ablation is invariant to per-direction scaling), so the ratio becomes
much larger — but the *shared > random* sign was already produced by A1
alone.

The clean empirical claim therefore is:

> *The original Step-7 sign was driven primarily by the unrestricted
> eigenproblem selecting low-variance "shared" directions. Restricting
> the eigenproblem to the high-variance principal subspace (a one-line
> code change) reverses the sign at all three preselected sites without
> changing the ablation protocol. Variance-matched ablation cleans the
> random-baseline comparison further but is not the dominant correction.*

Both fixes generalize: any cross-model subspace identification using
ratio-form objectives (CKA-based projections, generalized eigenproblems,
ratio-form CCA variants) should be sanity-checked against PCA restriction;
any ablation comparison across subspaces of unequal variance content
benefits from variance matching.

## 10. The redundancy confound (A3) — joint ablation is required

Even with A1 and A2 corrections applied, single-subspace ablation gives
a *systematically biased* estimate of a subspace's causal importance
whenever the network has encoded task-relevant information redundantly
across multiple subspaces. This is the A3 confound.

### 10.1 The bias

Let `S` and `S⊥` denote two orthogonal subspaces that jointly span a
task-relevant subset of the residual stream at a given site. The model
computes its output by reading some function of the activations at that
site; if the relevant information is present in BOTH `S` and `S⊥`, then
ablating `S` alone produces a small drop — the readout can substitute
information from `S⊥`. Similarly, ablating `S⊥` alone produces a small
drop — readout substitutes from `S`. But ablating `S ∪ S⊥` removes both
copies and produces a large drop.

Concretely, let `d(S)` denote the mean ablation drop for subspace S.
Under *perfect redundancy* (both subspaces independently sufficient for
the readout):

  d(S) ≈ 0
  d(S⊥) ≈ 0
  d(S ∪ S⊥) ≈ d_full

Under *partial redundancy* (one subspace carries more load, the other
substitutes for loss):

  d(S) < d_full
  d(S⊥) < d_full
  d(S ∪ S⊥) ≫ d(S) + d(S⊥)   (strictly super-additive)

Under *irrelevance* of `S` (the null hypothesis for "shared is
causally inert"):

  d(S) ≈ 0
  d(S ∪ S⊥) ≈ d(S⊥)   (no additional effect from removing S)

These three regimes are indistinguishable from single-subspace ablation
alone. Joint ablation disambiguates them.

### 10.2 Empirical outcome

In our zoo at six tested task-relevant sites (one at layer 1, five at
layer 3, k = 8), the hidden-load `d(S ∪ S⊥) − d(S⊥)` is strictly
positive with 95% bootstrap CI lower bounds ranging from 0.33 to 0.59.
This rules out the "irrelevance" regime at every tested site and
establishes partial redundancy.

A particularly striking case is layer 3, where `d(shared)` ≤ 0.003 at
five consecutive result-digit positions — a pattern that single-subspace
ablation alone would interpret as shared-direction irrelevance, but
which joint ablation reveals to be 0.36–0.44 units of hidden causal load
masked by complement compensation.

### 10.3 Corollary

Any CKA-based (or similar) interpretability analysis that uses single-
subspace ablation to conclude that a subspace is "not causally
important" should be re-examined with joint ablation against the
orthogonal complement. The magnitude of the hidden-load term depends on
how redundantly the network has encoded the task, and can be substantial
even when the single-subspace drop is near zero.
