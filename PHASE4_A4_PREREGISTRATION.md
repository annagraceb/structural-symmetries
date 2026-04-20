# Phase 4 — A4 Confound (Distributed-Redundancy vs. Generic-Damage) — Pre-Registration

**Status:** LOCKED before any new code runs that touches Phase-4 artifacts.
**Author-time:** drafted 2026-04-17 evening PT, to be committed before first new run.
**Deadline on target experiments:** 06:00 PT 2026-04-18.

## Motivation

The Phase-3 bonus finding — hidden load at the N-1 layer shrinking monotonically with depth (0.36 at 4L, 0.17 at 6L, 0.00 at 8L) — admits two incompatible readings:

1. **Compression**: Deeper models route the task-relevant representation *out* of the N-1 residual stream (e.g., into the unembed weights or earlier layers). Shared and complement both approach inertness at N-1.
2. **Distributed redundancy**: Deeper models *spread* the causal signal across more complement directions at N-1. Single-direction ablation looks inert, but joint ablation of k ≥ 3 structured complement directions recovers 4L-scale damage.

The existing A1/A2/A3 protocol cannot separate these readings. Three adversarial critic rounds (2026-04-17) converged that resolving the ambiguity requires a fourth confound control — **A4: structured-vs-random-matched k-ablation baselines** — without which any positive or negative result at a single k is under-determined.

Step-18 (`unembed_geometry.py`) rules out the trivial-geometric reading at 4L layer 3 (shared_frac_in_nullspace = 0.767 < analytic-random 0.813; complement_frac = 0.310, strongly readable). Phase 4 therefore targets the 8L zoo where hidden load is observationally zero.

## Hypothesis

**H-A4 (primary).** At the N-1 layer of the 8L zoo, structured k-dimensional complement ablation produces greater behavioral damage than task-subspace-projection-matched random k-dimensional complement ablation, with the area-under-curve gap (AUC-gap) across k = 1..8 anchored by an effect-size reference class, failing at three pre-specified null sites, and surviving a held-out direction-identification split.

**H-A6 (secondary, cheap).** At 4L layer-3 N-1, complement-CKA between independently-trained addition models exceeds cross-task complement-CKA between addition and mod-23 models at matched depth, after matching on subspace dimension and eigen-spectrum.

## Architectures & Zoos — INHERITED

Reuses existing artifacts verbatim:

- `models_deep8/` (21 models, 8L, 5-digit addition). Primary zoo for H-A4.
- `models/` (33 models, 4L, 5-digit addition). Reference zoo for effect-size anchor + H-A6.
- `models_modp/` (33 models, 4L, mod-23 addition). H-A6 cross-task comparator.
- `models_deep/` (45 models, 6L, 5-digit addition). Depth-interpolation only if budget permits.

**New zoo required**: `models_deep8_permuted/` — 10 × 8L models trained on 5-digit addition **with randomly permuted target labels** (each model's permutation is its seed-indexed fixed permutation). This serves as the Phase-4 null-task control.

Training protocol for `models_deep8_permuted/`: identical hyperparameters to `models_deep8/` except for the target-permutation wrapper in `data.py`. Convergence target: within the same step budget, the permuted-label models will **not** reach >99% exact-match (by construction); we train to the fixed step budget of 20,000 steps and select the ten that reached the highest per-model exact-match accuracy (≥ 20%) as the permuted-zoo inclusion set. If fewer than 10 reach ≥ 20% within budget, Phase 4 is marked INCONCLUSIVE on the permuted-null axis.

## Sites — PRE-SPECIFIED

**Primary (locked):**
- `deep8_layer7_result_0`: 8L zoo, layer 7, result-digit-0 position. This is the N-1 site where step9_p1_deep8 reported shared = 0.000 [0.000, 0.000].

**Null sites (locked, three, each must FAIL the rule):**
- `deep8_layer1_result_0`: 8L zoo, **early layer**, same result-position. Pre-computation; distributed redundancy should not be observable here.
- `deep8_layer7_position_N-3`: 8L zoo, same layer as primary, **wrong digit position** (3 positions left of result-digit-0, i.e., one of the operand digit positions). Same layer and activation scale; hypothesis predicts the complement directions matter at N-1 *for result positions*, not arbitrary digit sites.
- `permuted_deep8_layer7_result_0`: permuted-label zoo, same layer/position as primary. Label structure destroyed; no task-relevant direction identification should be possible.

**Reference anchor (locked):**
- `main_layer3_result_0_4L`: 4L main zoo, layer 3 (= N-1 for 4L), result-digit-0. Used only to compute the 4L single-direction reference drop (`d_4L_k1`) for threshold anchoring. This site was reported in the published paper; no new analysis is performed on it beyond reading the existing `results/p1/p1_results.json`.

## Direction-identification procedure — LOCKED

The complement directions at each primary and null site are extracted via the existing `step9_p1.extract_with_max_dims(aligned_acts, max_dims=8, eps_scale=1e-8, k_pca=K_PCA)` followed by `step9_p1.complement_top_k(shared, C_total, k=8)`. No substitution, re-weighting, or variant identification algorithm is permitted.

**Held-out split (new, mandatory):** The aligned activations at each site are deterministically split 50/50 by the evaluation-set sequence index (even-indexed eval sequences → `split_A`, odd-indexed → `split_B`). Direction identification (PCA + ratio eigenproblem + ordering) is computed on `split_A`. Ablation damage is measured on `split_B`. No peeking at `split_B` during identification.

## Random-null matching — LOCKED (A2-prime)

For each model × k, the 100 random k-dimensional subspaces in the complement are sampled such that:

1. They are orthonormal and lie entirely within the PCA-restricted complement subspace `C_total \ shared_subspace` (matches existing A2 geometry).
2. **Task-subspace projection-energy match (new, A2-prime)**: each sampled random subspace is rejection-sampled or re-projected so that its total energy projection onto the identified top-k structured directions is within ±20% of the median same-metric value across 1000 pre-samples. This prevents the null from being dominated by subspaces that live orthogonally to the structured bundle (which would understate random drop and inflate the gap).
3. Their total per-model ablation variance mass (sum of projected variance of eval activations) is within ±10% of the structured-direction total variance mass (the standard A2 variance match).

If either match fails for more than 10% of draws at a given k, the null is flagged and that k is reported with a caveat; the overall AUC is not computed across flagged k unless ≥ 6 of 8 values remain clean.

## Decision rule — LOCKED

**H-A4 is CONFIRMED iff ALL of the following hold:**

1. **Primary effect**: at `deep8_layer7_result_0`,
   - `AUC_k=1..8[structured_drop(k) − median_random_drop(k)] ≥ 0.12`
   - with 95% bootstrap CI lower bound `≥ 0.04`
   - bootstrapped hierarchically over 21 models × 100 random-null draws per k, 2000 resamples.

2. **Ranking sanity**: structured_drop(k) > random_drop(k) at `≥ 6 of 8` k values, using the *median* random draw per k.

3. **Null-site FAILURE (all three)**: At each of `deep8_layer1_result_0`, `deep8_layer7_position_N-3`, and `permuted_deep8_layer7_result_0`, the primary-effect AUC is `< 0.03` with 95% bootstrap CI upper bound `< 0.06`.

4. **Relative-to-4L anchor reporting (not a pass/fail, but locked presentation)**: the ratio `AUC(primary) / d_4L_k1` is reported in the abstract. If `< 0.20` the finding is presented as "detectable but subcritical"; if `≥ 0.50` presented as "comparable magnitude"; anything in between gets plain numerical reporting with no adjective.

**H-A4 is FALSIFIED iff the primary-effect AUC < 0.03 with CI upper bound < 0.06** — the compression reading stands, and the paper is rewritten as a methodological contribution (A4 as a negative-result confound control).

**H-A4 is INCONCLUSIVE iff either**:
- fewer than 6 k-values have clean matched nulls, OR
- bootstrap CI on primary AUC spans the [0.03, 0.12] range,
- OR any null site fails the rule (< 0.03, CI upper < 0.06) while the primary succeeds (this would indicate the structured-vs-random gap is layer/position/task-agnostic, which *invalidates* the "distributed redundancy at N-1" reading and turns the finding into a claim about a generic artifact of the identification pipeline).

## H-A6 — LOCKED secondary rule

At `main_layer3_result_0_4L` (4L main zoo) and `modp_layer3_equals_4L` (4L mod-23 zoo), compute complement-CKA across all pairs of baseline-seed models within each zoo and across zoos.

**H-A6 CONFIRMED iff**:
- `complement-CKA(within main 4L) − complement-CKA(main 4L × mod-p 4L) ≥ 0.15`
- after eigen-spectrum matching (top-k singular-value-normalized complements, k = 8)
- with 95% bootstrap CI lower bound `≥ 0.05`.

If the cross-task complement-CKA is within 0.05 of the within-task value, **H-A6 is FALSIFIED** and the "complement-universality" framing is retired immediately. The original paper's A1/A2/A3 story is unaffected.

## What we will NOT do post-hoc

- Swap primary site from `deep8_layer7_result_0` for any reason after observing results.
- Change k-sweep range from 1..8.
- Pick a different "better" null site after seeing the three locked ones fail ambiguously.
- Re-run with different random seeds in hopes of tighter CIs.
- Add a post-hoc direction-identification procedure if the held-out split's AUC is null.
- Train `models_deep8_permuted/` for longer than the 20k-step budget.
- Change the threshold constants (0.12, 0.04, 0.03, 0.06, 0.15, 0.05) after seeing any observational data.

## Artifacts

- `PHASE4_A4_PREREGISTRATION.md` (this file, committed unchanged).
- `step19_a4_kssweep.py` (new): implements split, A2-prime matching, k-sweep, null-site loop, and bootstrap.
- `step20_complement_cka.py` (new): implements H-A6 complement-CKA cross-task test.
- `run_deep8_permuted_zoo.py` (new): trains the permuted-label null zoo.
- `models_deep8_permuted/` (new, 10 models).
- `results/p1/a4_ksweep.json` (new).
- `results/p1/complement_cka.json` (new).
- `figures/fig7_a4_ksweep.png`, `figures/fig8_complement_cka.png` (new).
- `PHASE4_REPORT.md` (to be written against this pre-reg, recording verdict verbatim).

## Stop rules

- If wall-clock reaches 04:30 PT 2026-04-18 without a clean primary-site AUC (either confirm or falsify), **report INCONCLUSIVE** and leave `models_deep8_permuted/` for future work.
- If the permuted-label null zoo fails to produce 10 models at ≥ 20% accuracy within 20k steps, record the permuted-null as INCONCLUSIVE and evaluate the primary rule on the two remaining null sites only. In that case, the "strict 3/3 null failure" becomes "strict 2/2 null failure", and we flag in the verdict that the label-permuted null was not reachable in budget.
- If held-out direction identification produces directions that have mean structured_drop(k=1) `≤ 1.5× median_random_drop(k=1)` across models, the pipeline is declared untrustworthy on this data and Phase 4 is INCONCLUSIVE regardless of AUC.

## Commitment

This pre-registration is committed before any Phase-4 code runs. Any deviation after first experiment is a failed run and Phase-4 verdict is INCONCLUSIVE.
