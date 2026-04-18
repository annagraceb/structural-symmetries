# Solution Symmetry Exploration — Design Spec v0.3

## Contribution Statement

Individual methods used in this pipeline (CKA, Procrustes alignment, PCA, probing, ablation, auxiliary losses) are each well-established. The contribution is the end-to-end pipeline and the question it answers: **can you extract a useful training signal from the geometric agreement between independently trained models, without knowing what the agreement represents, and use it to accelerate training?**

The cross-model shared variance extraction (Step 5c) adapts the ratio eigenproblem from BCI/EEG literature to a new setting. In BCI, the numerator and denominator covariances represent two experimental conditions (e.g., left vs. right hand). Here, they represent variance across inputs (numerator: input-dependent, informative) and variance across models (denominator: model-specific, noise). The extracted directions maximize "fraction of variance that is shared across independently trained models" — a quantity specific to this cross-model comparison setting that standard single-model PCA does not isolate.

## Goal

Determine whether models trained under different component-freezing constraints develop shared structure in activation space, whether that structure is causally necessary, and whether it can accelerate training when used as an auxiliary signal — all without assuming what the shared structure represents.

## Prerequisites

Existing arithmetic transformer pipeline: 4 layers, 4 heads, d_model=128, d_ff=512, decoder-only, trained on N-digit addition to ≥99% exact-match accuracy. Carry head auxiliary loss already implemented. Training runs on RTX 3060 (12GB VRAM).

---

## Step 1: Generate the Model Zoo

### Components

A 4-layer, 4-head model has 10 freezable components:

- `embed` (token + positional embeddings)
- `layer{0-3}.attn` (full attention block: QKV + output projection)
- `layer{0-3}.mlp` (full MLP block: up + down projection)
- `unembed` (final output projection / lm_head)

### Freeze protocol

For each config, set `requires_grad=False` on all parameters belonging to the frozen component before optimizer creation. Frozen weights stay at their random initialization values for the entire training run. Everything else trains normally with identical hyperparameters.

### Configs to train

| Config type | Count | Purpose |
|---|---|---|
| Baseline (nothing frozen) | 3 seeds | Reference point. Cross-seed variance tells you how much variation is "normal." |
| Single-component freeze | 10 components × 3 seeds = 30 | Core experimental conditions. Each freezes exactly one component. |
| **Total** | **33 models** | |

Three seeds per config provides a within-config variance estimate (not just a point estimate), enabling meaningful within-config vs. across-config comparison in Step 3c.

### Inclusion criterion

A model enters the analysis pool only if it reaches ≥99% exact-match accuracy on the held-out eval set. Log all models that fail to converge — the set of components whose freezing prevents convergence is itself informative (it tells you which components are non-redundant for the task).

### What to save per model

- Final model weights (full state_dict)
- Training config (seed, frozen component, hyperparameters)
- Training curve (loss + accuracy at every epoch)
- Final eval accuracy (in-distribution and OOD if available)
- Whether the model converged (≥99%) or not, and at what step it crossed 99%

---

## Step 2: Collect Activations

### Shared evaluation sets

Create two fixed evaluation sets and save both to disk. Never regenerate them.

**Primary set (stratified, 2000 problems)** — used for Steps 3–6 extraction and analysis:

- 500 problems with 0 carries
- 500 problems with 1 carry
- 500 problems with 2 carries
- 500 problems with 3+ carries

The heavy stratification toward high-carry problems is intentional: it ensures statistical power for carry-conditioned analysis in Step 6.

**Validation set (natural distribution, 2000 problems)** — sampled from the natural distribution of N-digit addition without carry stratification. Used after Step 5 to verify that shared directions extracted on the stratified set generalize to the natural input distribution (see Step 5e).

Use the same primary set for every model during extraction. Use the validation set only for the checks specified in Step 5e.

### Extraction sites

For each model and each input problem, extract the residual stream vector (the hidden state after the full residual connection, i.e., after attention + MLP for that layer) at every combination of:

- **Layer**: after layer 0, 1, 2, 3 (4 layers)
- **Token position**: every token position in the sequence

This gives a 3D tensor per model: `[n_problems, n_layers × n_positions, d_model]`. In practice, store as separate files per layer to keep memory manageable.

### Storage format

For each model `m` and layer `l`, save:

- `activations/{m}/layer{l}.npy` — shape `[2000, n_positions, 128]`, float32
- `activations/{m}/metadata.json` — frozen component, seed, final accuracy, token-position-to-meaning mapping (which positions are operand digits, operator, equals sign, result digits)

### Token position mapping

The token position meaning depends on the input. For each problem, record which token positions correspond to: each digit of operand A, the operator, each digit of operand B, the equals sign, each digit of the result. This mapping is needed later to analyze position-specific behavior.

---

## Step 3: Cross-Model Similarity Analysis (Rotation-Invariant)

### 3a: CKA heatmap (where do models agree?)

For every pair of models (m_i, m_j), at every extraction site (layer × token position), compute linear CKA between their activation matrices.

**Inputs:** Two matrices of shape `[2000, 128]` — the activations from both models at the same layer and token position, over the same 2000 problems.

**Output:** A single scalar (CKA similarity) per model-pair × layer × position.

**Aggregation:** For each layer × position cell, report the mean CKA across all model pairs plus a 95% bootstrap confidence interval (resample model pairs with replacement, 1000 iterations). This is critical because you have O(K²) pairwise comparisons and need to know whether high mean CKA is driven by a few similar pairs or is consistent.

**Top-PC dominance check:** For each layer × position cell, also compute CKA after projecting out the top-1 principal component from both activation matrices. Report both values. If full CKA is high but top-1-removed CKA drops substantially, the similarity is driven by a single dominant direction (likely trivial, e.g., digit magnitude) rather than rich shared structure. Flag any sites where the drop exceeds 0.3.

**What to look for:**

- Are there specific layer × position sites where CKA is consistently high (>0.7) across all model pairs? These are the candidate sites for shared structure.
- Does agreement increase or decrease with layer depth?
- Is agreement higher at result token positions than operand positions?
- Is the agreement robust to removing the top PC, or is it superficial?

### 3b: Three-group comparison

Compute CKA separately for three groups:

1. **Baseline vs baseline** (different seeds, nothing frozen) — how much similarity does the task alone induce?
2. **Freeze config vs freeze config** (different frozen components) — does freezing different components create more divergence than seed variation?
3. **Freeze config vs baseline** — are frozen models as similar to baselines as baselines are to each other?

If group 2 shows substantially lower CKA than group 1 at certain sites, freezing is creating real divergence. If groups 1 and 2 are similar, the divergence is seed-level noise and freezing adds little.

### 3c: Within-config vs across-config

For each frozen component (e.g., "layer2.mlp frozen"), you have 3 seeds. Compute within-config CKA (same freeze, different seed — 3 pairs per config) and across-config CKA (different freeze). With 3 seeds, each config yields 3 within-config CKA values, providing a distributional estimate rather than a point estimate. If within-config >> across-config, each freeze configuration induces a meaningfully distinct solution.

### Decision gate

Select the top 2-3 extraction sites (layer × position combinations) where:

- Mean CKA > 0.5 with bootstrap lower bound > 0.35
- Top-1-removed CKA remains above 0.3 (similarity is not driven by a single trivial direction)
- Freeze-vs-freeze CKA is meaningfully above chance but below ceiling (the interesting regime where models partially agree)

If no sites meet these criteria, stop. Either the models are too similar (CKA uniformly >0.9 — nothing to extract) or too different (CKA uniformly <0.3 — nothing is shared).

---

## Step 4: Model Alignment

**This step brings all models into a common coordinate frame. Steps 5-7 depend entirely on this alignment being sound. Without it, direction-level analysis is meaningless because separately trained models use arbitrarily rotated bases.**

### 4a: Choose a reference model

Pick one baseline model (seed 0, nothing frozen) as the reference. All other models will be aligned to this reference's activation space. The choice of reference is arbitrary — the extracted shared subspace should be invariant to it. Verify this later by repeating with a different reference and checking that the results are stable (Step 5f).

### 4b: Procrustes alignment

For each non-reference model m_k, at each selected extraction site (from Step 3):

1. Let X_ref be the reference model's activation matrix, shape [2000, 128].
2. Let X_k be model m_k's activation matrix, same shape.
3. Center both matrices (subtract their respective column means).
4. Compute the orthogonal Procrustes solution: find the orthogonal matrix R_k that minimizes ||X_k R_k - X_ref||²_F. This is solved via SVD of X_ref^T X_k = U S V^T, giving R_k = V U^T.
5. Store R_k and the Procrustes distance (residual after alignment) for each model.

After this step, every model's activations are expressed in the reference model's coordinate frame.

### 4c: Alignment quality check

The Procrustes distance tells you how well each model aligns to the reference. Report:

- Distribution of Procrustes distances across models. If some models have much higher residuals, their representations may be genuinely different (not just rotated).
- Compare Procrustes distance to a null baseline: align each model to a random orthogonal rotation of the reference. If real alignment isn't substantially better than random alignment, the models don't share enough structure for direction-level analysis to be meaningful.
- Separate Procrustes residuals for baseline models vs. freeze-config models. If freeze-config residuals are >2× the baseline-vs-baseline residuals, trigger the SVCCA fallback (Step 4e).

### 4d: SVCCA cross-check

As a robustness check, also compute SVCCA similarity between each aligned model pair. SVCCA first reduces dimensionality via SVD (keeping components explaining 95% of variance), then computes canonical correlations. If SVCCA tells a substantially different story than CKA + Procrustes, investigate why — it likely means the shared structure is low-rank and concentrated in a subspace, which is actually useful information.

### 4e: SVCCA fallback for high-residual models

**Trigger:** Procrustes residuals for a model (or group of models) exceed 2× the mean baseline-vs-baseline Procrustes residual.

**Interpretation:** These models share structure with the reference, but the relationship is not well-described by a global rotation. They may use overlapping but non-orthogonally-related subspaces.

**Procedure:**

1. For each high-residual model m_k, compute SVCCA between X_k and X_ref: project both to their respective top-k subspaces (retaining 95% of variance), then compute canonical correlations.
2. The canonical correlation vectors define a paired set of directions — one in each model's native space — that are maximally correlated.
3. For Steps 5-7, use the canonical correlation vectors in the reference model's space as the "aligned directions" for these models, instead of Procrustes-rotated activations.
4. Flag these models in all downstream results so it is clear which alignment method was used.

If >50% of converged models trigger the SVCCA fallback, Procrustes alignment is unsuitable for this setting. In that case, switch entirely to SVCCA-based analysis for all models and note the implication: the solutions share subspace structure but not a global coordinate relationship.

### What to store

For each non-reference model m_k, at each selected site:

- The rotation matrix R_k (shape [128, 128]) — or SVCCA canonical vectors if fallback was triggered
- The Procrustes residual (scalar)
- The aligned activation matrix X_k R_k (shape [2000, 128])
- Which alignment method was used (Procrustes or SVCCA)

---

## Step 5: Shared Subspace Extraction (in Aligned Space)

All analysis in this step operates on Procrustes-aligned activations from Step 4. Every model's hidden states are now in the reference model's coordinate frame. (For SVCCA-fallback models, use the canonical-correlation-projected activations in the reference frame.)

### 5a: Cross-model variance decomposition

For each input problem x (of the 2000), collect aligned hidden states from all K converged models at the chosen site. Each is a vector in R^128 in the common frame.

Compute two quantities per dimension d:

- **W_bar[d]**: Mean within-model, across-input variance. Average over models of Var_x[h_k(x)[d]]. Measures how much dimension d varies with the input.
- **V_bar[d]**: Mean across-model, per-input variance. Average over inputs of Var_k[h_k(x)[d]]. Measures how much dimension d varies across models for the same input.

Compute a "shared and informative" score:

    score(d) = W_bar[d] / (V_bar[d] + epsilon)

High score = varies with input (informative) but stable across models (shared). Rank dimensions by this score.

### 5b: Subspace via aligned PCA

Stack all K aligned activation matrices vertically: shape [2000 × K, 128]. Run PCA. The top components capture directions of high total variance, which after alignment should correspond to shared high-variance directions.

### 5c: Subspace via cross-model agreement (the core extraction)

1. For each input x, compute the mean aligned hidden state across models: h_bar(x) = (1/K) Σ_k h_k(x).
2. Compute the covariance matrix of h_bar(x) across inputs: C_shared = Cov_x[h_bar(x)], shape [128, 128]. This captures directions that are both input-dependent and consistent across models (because averaging over models suppresses model-specific noise).
3. Also compute the total covariance of individual model activations: C_total = (1/K) Σ_k Cov_x[h_k(x)].
4. The ratio eigenproblem C_shared v = λ C_total v gives directions ranked by "fraction of variance that is shared." Top eigenvectors have the highest shared-to-total variance ratio.

Extract top 1, 3, 5, and 10 directions from each method (5a, 5b, 5c). Report how much cross-model variance each explains.

### 5d: PCA baseline comparison

Compare the directions from 5c (shared-variance ratio) against vanilla PCA on a single model's activations (5b). Compute cosine similarity between corresponding directions. If they're nearly identical, the cross-model method isn't finding anything PCA wouldn't. If they diverge, the cross-model structure is providing new information.

### 5e: Natural-distribution validation

Run each converged model on the 2000-problem natural-distribution validation set (from Step 2). Collect activations at the selected extraction sites and align using the rotation matrices from Step 4.

Compute the shared-variance ratio (from 5c) on the validation set using the same shared directions extracted from the stratified set. Report:

- Fraction of shared variance explained by the top-k stratified directions on the validation set
- Whether the ranking of directions by shared-to-total ratio is preserved

If the top shared directions explain substantially less variance on the natural set (e.g., <50% of what they explain on the stratified set), the extraction is carry-specific. This is still a valid finding but must be reported honestly, and downstream results (Steps 6-8) should be interpreted accordingly.

### 5f: Reference invariance check

Repeat Steps 4-5 using a different reference model (e.g., baseline seed 1 instead of seed 0). Compute cosine similarity between the shared directions extracted under each reference. If the directions are stable (cosine > 0.8), the extraction is robust. If they differ, the result is reference-dependent and suspect.

---

## Step 6: Interpretation (What Does It Encode?)

Project all models' aligned activations onto the top shared directions from Step 5c. For each direction, compute correlation with every available task variable:

- Carry bit at each column position (binary)
- Digit value of operand A at each position (0-9)
- Digit value of operand B at each position (0-9)
- Result digit at each position (0-9)
- Number of total carries in the problem (integer)
- Sum magnitude (operand A + operand B as integers)
- Whether this column has a carry-in from the previous column (binary)

For each direction × task variable pair, compute:

- Mutual information (discretize the projection into 10 bins)
- Linear probe accuracy (logistic or linear regression using only the 1D projection as input)

### Baseline comparison requirement

For every shared direction that correlates with a task variable, also measure the same correlation using baseline-only models (no freezing). If the correlation strength is equivalent in baseline-only models and freeze-config models, then the task alone induces this structure — freezing is not contributing. If the correlation is *tighter* across freeze-config models than across baselines, freezing is constraining models toward a more consistent encoding, which is a distinct (and interesting) finding.

### What to look for

- Does the top shared direction correlate cleanly with carry? If yes, you've recovered the known result blindly. Expected but validating.
- Does it correlate with something else? More interesting.
- Are there directions that are highly shared (low V_bar) and informative (high W_bar) but don't correlate with any tested task variable? These might encode intermediate computations with no obvious label. Flag them for further investigation.
- Is the shared structure equally present in baseline-only models, or is it amplified by freezing?

---

## Step 7: Ablation (Causal Necessity)

For each converged model independently:

1. Run the model on the 2000 eval problems and record baseline accuracy.
2. Rotate the shared directions from Step 5 into the model's native basis using the inverse of its Procrustes rotation: v_native = R_k^T v_shared. This is critical — you must ablate in the model's own coordinate frame, not the reference frame.
3. Hook into the chosen extraction site. During the forward pass, project the hidden state onto v_native and replace the projection with its mean value across the eval set (mean-ablation). This preserves the mean activation statistics at that site while removing the input-dependent information carried by that direction.
4. Measure accuracy after ablation.
5. Repeat with a control direction matched for activation variance (see below).

### Control direction matching

The shared directions likely capture high-variance components of the activations. Ablating a high-variance direction removes more information than ablating a low-variance one, regardless of content. To isolate the effect of *which* direction is ablated from *how much* activation is removed:

1. Compute the activation variance along the shared direction on the eval set: σ²_shared = Var_x[(h(x) · v_native)²].
2. Sample 100 random unit directions. For each, compute its activation variance.
3. Select the 10 random directions whose activation variance is closest to σ²_shared.
4. Mean-ablate each of the 10 matched random directions. Report the mean and spread of accuracy drop across these 10 controls.

### What to measure

- Accuracy drop when mean-ablating top-1 shared direction vs. variance-matched random direction
- Accuracy drop when mean-ablating top-3, top-5, top-10
- Accuracy drop specifically on high-carry problems vs. low-carry problems
- Accuracy drop per model (does the effect size vary across freeze configs?)

### Pass criterion

Mean-ablating the shared direction causes a significantly larger accuracy drop than mean-ablating variance-matched random directions. Test with a paired t-test or Wilcoxon signed-rank test across models, p < 0.01. If this fails, the shared structure is present but not causally used.

---

## Step 8: Self-Supervised Auxiliary Loss

Train fresh models (from scratch, nothing frozen) with five conditions:

### Condition A: Baseline

No auxiliary loss. Standard training.

### Condition B: Carry head (known-good benchmark)

Existing carry head auxiliary loss using ground-truth carry labels. This is the benchmark that uses domain knowledge.

### Condition C: Geometric head (blind extraction)

Auxiliary loss that encourages the fresh model's representations to align with the shared subspace from Step 5, defined in a rotation-invariant way.

**Implementation — two options, try both:**

**C1 — CKA-based loss (fully rotation-invariant):** Compute mini-batch CKA between the fresh model's activation matrix at the target site and a bank of reference activations (precomputed from Step 2, fixed). Maximize CKA as the auxiliary objective. This never assumes a shared basis — CKA handles rotations internally. Downside: CKA on mini-batches is noisy; use batch size ≥256.

    geometric_loss = -CKA(h_current_batch, h_reference_batch)
    total_loss = main_loss + alpha * geometric_loss

**C2 — Projection loss with online alignment:** At each training step, Procrustes-align the fresh model's current activations (on the mini-batch) to the reference, then penalize deviation from the shared subspace in the aligned frame. This is more direct but adds alignment cost per step. Use a cached/periodically-updated alignment matrix (recompute every N steps, not every step) to keep overhead manageable.

    R = procrustes(h_current_batch, h_reference_batch)  # recompute every 100 steps
    h_aligned = h_current_batch @ R
    projection = h_aligned @ shared_dirs @ shared_dirs.T
    geometric_loss = MSE(h_aligned, projection)
    total_loss = main_loss + alpha * geometric_loss

### Condition D: Random subspace head (control)

Same as Condition C2, but using random orthogonal directions instead of the extracted shared directions. Same dimensionality, same alpha, same online alignment. This controls for the possibility that *any* auxiliary loss on hidden states helps via regularization, regardless of direction.

### Condition D': Bottom-eigenvector head (hard control)

Same as Condition C2, but using the *bottom* eigenvectors from the Step 5c ratio eigenproblem — directions with the *lowest* shared-to-total variance ratio. These are real directions from the teacher activations (so they carry genuine activation signal), but they are the least shared across models. This is a harder control than random directions: if C outperforms D' , the *specific shared directions* matter, not merely having a teacher-derived auxiliary target.

Same dimensionality and alpha as Condition C2.

### Alpha selection protocol

Alpha selection is two-phase to avoid spending full compute on suboptimal values:

1. **Sweep phase:** For each of C1, C2, D, and D' , train 2 seeds at each alpha ∈ {0.1, 0.01, 0.001}. Evaluate on a held-out validation split (separate from the 2000-problem eval sets used in Steps 3-7). Select the alpha that yields the fastest convergence to 99% accuracy (averaged over 2 seeds).
2. **Evaluation phase:** At the selected alpha, train 5+ seeds per condition with identical hyperparameters. All metrics below are reported from this phase.

### Protocol

5+ seeds per condition in the evaluation phase. Same hyperparameters, same data, same compute budget. Measure:

- Steps to 99% accuracy (primary metric)
- Training loss curve
- Final accuracy
- Carry probe accuracy at convergence (post-hoc, not used during training)

### What would be interesting

- **Condition C ≈ Condition B:** The blind extraction recovers the carry head's benefit without domain knowledge. Clean result, validates the pipeline.
- **Condition C > Condition B:** The shared subspace contains more useful structure than carry alone. Surprising, high-value finding.
- **Condition C ≈ Condition D:** The direction doesn't matter, any auxiliary hidden-state loss helps. The shared subspace extraction is not adding value beyond regularization.
- **Condition C ≈ Condition D':** The specific shared directions don't matter — any teacher-derived direction helps equally. The extraction pipeline identifies real directions but not meaningfully better ones.
- **Condition C > Condition D' > Condition D:** The shared directions are best, but even non-shared teacher directions help more than random. A gradient of usefulness that supports the pipeline's value.
- **Condition C < Condition A:** The geometric head hurts. The extracted directions are misleading the optimization.
- **C1 vs C2:** If the CKA loss (fully rotation-invariant) works as well as the Procrustes-projection loss, that's cleaner and suggests the alignment step isn't adding much at training time.

---

## Decision Points

| After Step | Continue if | Stop if |
|---|---|---|
| Step 1 | ≥6 distinct freeze configs converge to 99% (≥2 seeds each) | Fewer than 4 configs converge. Not enough diversity. |
| Step 3 | CKA heatmap shows ≥1 site with mean CKA > 0.5 (bootstrap lower bound > 0.35) and top-1-removed CKA > 0.3; freeze-vs-freeze CKA is meaningfully above random | CKA is uniformly low (<0.3) or uniformly high (>0.9); or all similarity is driven by a single PC. |
| Step 4 | Procrustes alignment quality is significantly better than random-rotation baseline for most models (or SVCCA fallback succeeds for high-residual models) | Procrustes residuals are no better than random alignment AND SVCCA canonical correlations are low. Models don't share enough directional structure. |
| Step 5 | Top shared direction from 5c is distinguishable from vanilla PCA (cosine < 0.8 with PCA top component), OR if they agree, the shared direction is stable across reference models (5f) and generalizes to the natural distribution (5e) | Shared directions are unstable across reference choices (cosine < 0.5). Extraction is not robust. |
| Step 6 | ≥1 shared direction correlates with a task variable (MI > 0.1 bits) | No correlations found. Proceed to Step 7 anyway — the direction might still be causally necessary without being interpretable. |
| Step 7 | Mean-ablation of shared directions drops accuracy significantly more than variance-matched random ablation (p < 0.01) | No significant difference. Shared structure is present but not used. |
| Step 8 | Condition C outperforms both Condition D and Condition D' | Condition C ≈ Condition D' ≈ Condition D. Direction doesn't matter. |

---

## Compute Budget

| Step | Training runs | Inference/analysis | Estimated wall time (3060) |
|---|---|---|---|
| Step 1 | 33 models × ~5 min each | — | ~2.75 hours |
| Step 2 | — | 33 models × 2000 forward passes (×2 eval sets) | ~45 min |
| Step 3 | — | CKA pairwise + bootstrap + top-PC-removed CKA | ~45 min |
| Step 4 | — | Procrustes SVDs + SVCCA cross-check (CPU) | ~20 min |
| Steps 5-6 | — | Covariance eigenproblems + probing + validation set check | ~45 min |
| Step 7 | — | Mean-ablated forward passes + variance-matched controls | ~45 min |
| Step 8 (sweep) | 24 models (4 conditions × 3 alphas × 2 seeds) | — | ~2 hours |
| Step 8 (eval) | 25+ models (5 conditions × 5 seeds) | — | ~2 hours |
| **Total** | **~82 training runs** | | **~10 hours** |

Steps 1-7 (full diagnostic pipeline) fit in ~6 hours. Step 8 (payoff) adds ~4 hours. Fits in a weekend on the 3060.

---

## Appendix A: Alignment Pitfalls to Watch For

**Procrustes assumes the shared structure is related by a global rotation.** If two models encode the same information but in overlapping, non-orthogonally-related subspaces (e.g., one uses dimensions 1-10, the other uses dimensions 5-15), Procrustes will find a compromise rotation that partially aligns both but perfectly aligns neither. SVCCA handles this better because it projects to the shared variance subspace first. Step 4e provides a concrete fallback for this case.

**Procrustes alignment is input-dependent.** The rotation R_k is computed on the 2000 eval problems. If you later apply R_k to activations on different inputs, the alignment may not hold (especially if the activation manifold has different geometry on different input distributions). Always validate alignment on held-out inputs.

**The reference model choice should not matter, but might.** If the extraction site happens to be a layer where the reference model has an atypical representation (e.g., it found a rare solution), all other models will be aligned to that atypical frame. The reference invariance check in Step 5f catches this — if shared directions shift substantially with a different reference, the result is fragile.

**Online Procrustes alignment in Step 8 (Condition C2) is expensive if done per-step.** Cache the rotation matrix and recompute periodically (every 100-500 steps). The fresh model's representations change gradually during training, so the alignment doesn't need to be updated every step. Monitor alignment quality at each recomputation to make sure it isn't degrading.

## Appendix B: Interpreting the Freezing Intervention

Freezing a component at random initialization is not equivalent to removing it. The frozen component actively transforms inputs through a random projection at every forward pass. The rest of the network must compensate for (or route around) this persistent random transformation.

This means "shared structure across freeze configs" could reflect either:

1. **Task-determined structure:** The task forces a specific representation regardless of constraints. Evidence: shared structure is equally present in baseline-vs-baseline comparisons (no freezing).
2. **Compensation-determined structure:** The only viable way to work around a random bottleneck. Evidence: shared structure is stronger or qualitatively different in freeze-vs-freeze comparisons than in baseline-vs-baseline.
3. **Both:** Some directions are task-determined (shared across all models) and others are compensation-determined (shared only across freeze configs).

Step 3b (three-group comparison) and Step 6 (baseline comparison requirement) are designed to distinguish these cases. All interpretive claims must reference this comparison.
