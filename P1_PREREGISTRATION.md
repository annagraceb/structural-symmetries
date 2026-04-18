# P1 Pre-Registration — Inverted Eigenproblem / Anti-Shared Subspace

**Status:** LOCKED before any P1 code runs. Commit this file to git before executing
`step9_p1.py`. Reference the commit SHA in any paper draft.

**Purpose:** Protect against post-hoc rule-moving. If the results come back and the
decision rule says LOSS, the decision rule does not change.

---

## Hypothesis (H1)

Representations decompose into a readable subspace (what models agree on, low causal
importance) and a causal subspace (model-specific variance, high causal importance),
and these subspaces are distinguishable in the residual stream.

**Operational claim under H1:** The bottom-k eigenvectors of the generalized
eigenproblem `C_shared v = λ C_total v` (defined in `analysis.py::extract_shared_subspace`)
are anti-shared directions. When ablated via Step-7's mean-centered subspace
projection, they cause a larger accuracy drop than norm-matched random directions of
equal rank. This gap is larger than the existing shared-subspace gap at the same k.

**Null (H0):** The bottom-k eigenvectors behave no differently from random directions
under ablation, OR behave similarly to the top-k shared directions (both less harmful
than random). Either outcome falsifies H1.

---

## Decision Rule (LOCKED)

### Primary statistic

For each site `s` and direction count `k`, define the ablation-drop ratio:

```
R(s, k) = mean_drop(anti_shared_k, s) / mean_drop(random_k, s)
```

where `mean_drop` is the mean (across the 33 converged models) of
`baseline_acc − ablated_acc`, computed with identical ablation code to Step 7
(`analysis.py::run_ablation`, line 896 `make_ablation_hook`).

### Outcome classification (3-tier)

The primary preselected sites are the three chosen in `selected_sites.json`:
`layer1_result_0`, `layer2_equals`, `layer2_result_0`.

| Outcome | Criterion (ALL conditions must hold) |
|---------|--------------------------------------|
| **STRONG WIN** | R ≥ 3.0 at ≥2 of 3 sites; 95% CI lower bound > 1.5 at those sites; ≥75% of models show R > 1 at those sites; `drop(anti_shared) ≥ 2 × drop(shared)` by paired bootstrap; result stable across `ε ∈ {1e-5, 1e-4, 1e-3}` Tikhonov regularization |
| **WIN** | R ≥ 2.0 at ≥2 of 3 sites; 95% CI lower bound > 1.25; ≥75% of models show R > 1; `drop(anti_shared) > drop(shared)` paired (no 2× requirement); result stable across ε |
| **AMBIGUOUS** | 1.1 < R < 2.0 at primary sites; OR single-site-only effect; OR within-model inconsistency (<75% of models agree); OR unstable across ε |
| **LOSS** | R ≤ 1.1 at all 3 sites; OR 95% CI includes 1.0 at all sites; OR ≥50% of models show R ≤ 1 |

### Secondary gate: control site

At the low-CKA control site `layer3_plus` (CKA=0.075), anti-shared ablation
must produce a drop **smaller than 2% absolute** and **R < 1.5** relative to
random at that site. Failure on the control gate downgrades the verdict by
one tier (STRONG WIN → WIN, WIN → AMBIGUOUS, AMBIGUOUS → LOSS).

Rationale: if anti-shared hurts even at an irrelevant site, we are picking up
structural-ablation damage rather than causal signal.

### Tertiary gate: orthogonality variant

Run the entire P1 protocol twice at each site:

1. **Raw anti-shared**: bottom-k eigenvectors as-is.
2. **Shared-orthogonal anti-shared**: bottom-k eigenvectors after Gram-Schmidt
   projection out of the top-k shared subspace.

Both variants must independently achieve at least WIN tier for the verdict to
stand. If only the raw variant wins, the result is contaminated by residual
shared structure and downgrades to AMBIGUOUS.

---

## Confound Controls (LOCKED)

All are required. Failure to implement any control invalidates the run.

1. **Tikhonov regularization** — Use `C_total + ε · tr(C_total)/d · I` with
   ε ∈ {1e-5, 1e-4, 1e-3}. Report eigenvalue spectrum, condition number, and
   R(s, k) separately for each ε. Conclusion must hold across all three.

2. **Dead-direction handling** — Step 7's ablation is already mean-centered
   (analysis.py:905), so the constant-offset damage mode is already neutralized.
   No additional filter on `v^T C_between v` is needed. Document this so reviewers
   can verify.

3. **Norm matching** — All ablation directions (anti-shared, shared, random) are
   unit-norm. Random baselines are sampled as Gaussian and normalized, identical
   to the existing random protocol in run_ablation.

4. **Random baseline count** — 100 random subspaces per (site, k), not 10.
   Report the distribution of random drops and anti-shared's percentile within
   that distribution, in addition to the mean.

5. **k sweep** — k ∈ {5, 10, 20, 30, 50}. Verdict must be consistent at k=10 and
   at least one other k. If verdict flips at different k, downgrade to AMBIGUOUS
   and report the k-dependence.

6. **Per-model reporting** — Report R per model. Decision rule requires
   ≥75% of models showing R > 1 at each winning site. Do not average R directly
   without showing the distribution.

7. **Identical ablation code** — P1 reuses `run_ablation` from analysis.py
   verbatim. Any semantic change to the intervention invalidates the comparison.

8. **Site selection constraint** — The three primary sites are locked to the
   existing `selected_sites.json`. No new sites may be added. The control site
   is locked to `layer3_plus`. Exploratory low-CKA sites may be reported as
   supplementary but do not enter the decision rule.

---

## What Passing P1 Does and Does Not Show

### Does show
- Anti-shared directions have greater causal importance than random directions
  of equal rank at the preselected sites.
- Causal importance and cross-model shared variance dissociate in this setting.

### Does NOT show
- Any mechanistic interpretation of what anti-shared directions encode.
- That the readable/causal subspaces are the optimal decomposition.
- That this generalizes beyond N-digit addition.
- That anti-shared directions are unreadable (that requires the readability
  side-check, below).

### Readability side-check (run after P1, reported separately)

Probe the anti-shared top-10 directions against the same feature set used for
shared (result_digit_col*, carries, sum_magnitude). Use both linear probe and
MLP probe (as readability ceiling). For the two-subspace story to hold:

- Anti-shared linear-probe best correlation should be substantially smaller
  than the existing shared result (r=0.935 for direction_0).
- Anti-shared MLP-probe best correlation should also be smaller — if MLP
  probes recover near-identical correlations on anti-shared as on shared, the
  "readable vs causal" story collapses (anti-shared is just nonlinearly
  readable).

---

## Scope Restrictions

- P1 runs the above and nothing else. No feature interpretation, no auxiliary
  losses, no new training.
- A follow-up experiment ("P1b") characterizing anti-shared directions
  mechanistically (attention-head/MLP attribution, activation patching) is
  **required** for a publishable paper but is explicitly out of scope for P1.
- Do not adjust the decision rule after seeing the results. If the rule was
  wrong, note that honestly in the paper and re-register before any further runs.

---

## Files Affected

| File | Purpose |
|------|---------|
| `P1_PREREGISTRATION.md` (this file) | Locked decision rule |
| `step9_p1.py` | Experiment runner |
| `p1_report.py` | Applies the decision rule to `results/p1_results.json` |
| `results/p1_results.json` | Raw results |
| `results/p1_verdict.json` | Computed verdict (STRONG WIN / WIN / AMBIGUOUS / LOSS) |

The verdict file must be generated in a separate pass (`p1_report.py`), not
during `step9_p1.py` execution. This prevents the experiment from "knowing"
the decision rule at runtime.

---

## Amendment A1 — PCA subspace restriction

**Date:** 2026-04-16, applied BEFORE any full-run P1 results were generated.
**Trigger:** a single-model single-site smoke test revealed that the bottom
eigenvectors of the generalized eigenproblem `C_shared v = λ C_total v` have
near-zero per-model variance (`v^T C_total v` ≈ 0) after unit-norm
renormalization. The hypothesized "privileged/causal" directions were supposed
to have HIGH per-model variance and LOW cross-model agreement; the bottom
eigenvectors as originally defined instead pick up dead axes (both covariances
near zero). See the smoke log for numerics.

**Amendment:** all ratio eigenproblems (both original extraction and P1's
bottom-k) are now solved *within* the top-`K_pca` principal subspace of
`C_total`, where `K_pca = 32` (half of d=64). Concretely:

  1. Diagonalize `C_total = U Λ U^T`, ordered descending by eigenvalue.
  2. Let `U_k = U[:, :K_pca]` — the top-K_pca PCA directions of the pooled
     per-model variance.
  3. Project `C_shared_reduced = U_k^T C_shared U_k` and
     `C_total_reduced = U_k^T C_total U_k`.
  4. Solve the generalized eigenproblem in the reduced basis.
  5. Pull eigenvectors back: `v_full = U_k @ v_reduced`, renormalize to unit norm.

The decision rule (3-tier criteria, control site, orthogonality tier) is
UNCHANGED. Tikhonov regularization ε-sweep is UNCHANGED (applied to
`C_total_reduced`, not `C_total`).

**Rationale for this amendment being pre-registration-compatible:**
- It was decided via multi-AI consensus (Codex + Gemini + Claude all
  independently chose Option 2 from three presented options) before any full
  results were generated.
- It is a technical fix to the extraction, not a change to the decision rule
  or the outcome criteria. No threshold, effect-size cutoff, or gate was
  altered after seeing data.
- The original definition ("bottom-k eigenvectors") is preserved in intent
  ("directions of low cross-model agreement") but the geometric implementation
  is corrected so the solution cannot land on dead axes.

**What the amendment does NOT license:**
- Further post-hoc tuning of `K_pca`. It is locked at `K_pca = 32`.
- Changing which site is the control.
- Changing the R ≥ 2.0 / R ≥ 3.0 / ≥75% model-agreement criteria.
- Amending again after seeing full results.

---

## Amendment A2 — Variance-matched ablation

**Date:** 2026-04-16, applied AFTER A1 but BEFORE any full-run P1 results.
**Trigger:** re-smoke after A1 revealed that PCA restriction reduced but did not
eliminate the dead-direction problem. Unit-norm shared, anti-shared, and random
directions remove very different amounts of per-model activation variance:

| Subspace | trace(V^T C_total V) (unit-norm) |
|---|---|
| top-10 shared | ~10 (high-variance region) |
| bottom-10 anti-shared (post-A1) | ~0.1 (low-variance corner of PCA subspace) |
| random-10 (unit-norm Gaussian) | ~5 (uniform sampling across full space) |

Because the three subspaces remove different amounts of variance, the
decision statistic R = drop(anti_shared) / drop(random) is confounded: random
wins on raw drop because it removes ~50× more variance, not because it is
causally more important per unit of variance removed. This makes any null
result on anti-shared uninterpretable ("did it fail because it's causally
inert, or because it barely perturbed the model?").

**Amendment:** make every ablation subspace remove the SAME amount of
per-model activation variance, so the decision statistic measures causal
damage *per matched unit of variance removed*.

Concretely:

  1. **Shared and anti-shared directions**: keep the eigenvectors in their
     raw `C_total_reduced`-orthonormal form (not renormalized to unit
     Euclidean norm). By construction, each direction `v` satisfies
     `v^T C_total v = 1`. The subspace projector `P = V(V^T V)^{-1} V^T` is
     scale-invariant, so the mean-centered ablation hook in
     `analysis.py::make_ablation_hook` works unchanged. But `trace(V^T C_total V) = k`
     always, giving each ablation subspace exactly k units of variance.

  2. **Random baselines**: whiten. Sample `r ~ N(0, I_d)`, compute
     `r_white = C_total^{-1/2} r`, then renormalize so `r_white^T C_total r_white = 1`.
     Build random k-subspaces from k such whitened vectors. Now the random
     baseline also satisfies `trace(V^T C_total V) = k`. Uses the regularized
     `C_total + ridge · I` inverse square root (same ridge as the
     eigenproblem).

  3. **Report** `trace(V^T C_total V)` for every ablation condition in the
     results JSON so reviewers can verify the variance-match held.

**Decision rule adjustment (minimal, derived):** the decision rule
(R ≥ 2.0 / ≥ 3.0, ≥75% per-model, drop(anti_shared) ≥ 2 × drop(shared)) is
retained. The interpretation of R is now "drop per matched unit of variance
removed," which is strictly sharper than before — previously R conflated
causal importance with variance-removal magnitude. Because the
variance-removal is now matched across conditions, a genuine signal is *more*
likely to survive the criteria, not less.

**Rationale for this amendment being pre-registration-compatible:**
- Decided via multi-AI consensus (Codex, Gemini, Claude all picked "Option A"
  from three presented options) before any full-run results existed.
- The motivation is to sharpen what the existing decision rule *measures*,
  not to change the decision rule itself.
- Shared-direction ablation numbers from Step 7 are NOT carried over; they
  will be recomputed in P1 under the new matched-variance protocol so the
  comparison "drop(anti_shared) vs drop(shared)" is apples-to-apples.

**What the amendment does NOT license:**
- Changing the decision rule thresholds.
- Changing the primary/control sites.
- Further amendments after seeing full results.

---

## Amendment A3 — k-value cap

**Date:** 2026-04-16, applied AFTER first full run revealed an extraction bug
discovered by inspecting per-condition outputs.
**Trigger:** With A1 restricting to the top-`K_pca = 32` PCA subspace of
`C_total`, only 32 generalized-eigenvectors exist in the reduced basis. The
original `K_VALUES = [5, 10, 20, 30, 50]` requested top-k and bottom-k from
that 32-eigenvector spectrum. For k > 16, the top-k and bottom-k sets overlap;
for k = 32 they are identical. The first run (config locked) had this bug and
produced `anti_shared_raw` data that aliased to `shared` data — invalidating
the bottom-of-ratio comparison.

**Amendment:** `K_VALUES` is capped to `[4, 8, 12, 16]`. Maximum k is
`K_pca // 2 = 16`, ensuring disjoint top-k / bottom-k sets. The script now
truncates `n = min(max_dims, K_pca_eff)` to `K_pca_eff // 2` if `2n > K_pca_eff`
as a defensive guard.

**Decision rule unchanged.** Primary k for the verdict shifts from 10 to 8
(the closest remaining value below 10 in the new K_VALUES set). Multi-k
consistency now uses {4, 8, 12, 16}. The "≥2 of 3 sites" criterion still
applies.

**What this amendment does NOT license:**
- Adjusting the verdict thresholds because k is smaller.
- Re-doing the eigenproblem extraction differently than A1+A2 specified.
- Further amendments after the corrected run produces results.

---

## Protocol addition A4 — Joint ablation (added post-hoc for a follow-up analysis)

**Date:** 2026-04-17, added AFTER the corrected P1 run completed and
additional-sites (§6.1) and layer-3 expansion (§6.5) results were in
hand.
**Trigger:** a paradox-hunter review (third round) observed that
"shared near-inert at layer 3" from single-subspace ablation is
ambiguous between two interpretations: (a) shared directions are
causally irrelevant, or (b) shared directions carry causal load that is
redundantly backed up by the complement and therefore invisible to
single-subspace ablation. The pre-registered decision rule covers only
the single-subspace comparison and cannot disambiguate these
interpretations.

**Protocol addition:** a joint-ablation test running shared ∪ complement
at six sites (5 layer-3 result-digit positions + 1 layer-1 comparison).
Predicted signatures under the two interpretations are listed in
`VARIANCE_CONFOUND_DERIVATION.md` §10.1. The pre-registered decision
rule for the main P1 comparison is unchanged; the joint-ablation test is
a follow-up confound check that informs interpretation.

**Explicit note on pre-registration status:** A4 is not a pre-registered
correction to the P1 decision rule. It is a post-hoc confound check
motivated by a reviewer concern, with a pre-specified outcome rule
("partial redundancy ≡ joint drop strictly greater than either single
drop with positive-CI hidden-load"). The outcome is reported
transparently as a piloting result, not as a primary P1 finding.

**What A4 does NOT license:**
- Reinterpreting the primary P1 decision-rule verdicts using joint-
  ablation numbers.
- Claiming pre-registration status for the joint-ablation result.
- Further confound checks motivated by the joint-ablation outcome.
