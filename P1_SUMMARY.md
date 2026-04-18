# P1 Protocol: Summary of Artifacts

This directory contains the complete P1 experiment from pre-registration
through multi-site ablation, probing, and redundancy testing.

## Entry points

- `PAPER_DRAFT.md` — full paper draft, multi-AI reviewed, incorporates
  all findings below.
- `VARIANCE_CONFOUND_DERIVATION.md` — first-principles mathematical
  derivation of the A1 and A2 corrections.
- `P1_PREREGISTRATION.md` — locked decision rule + amendments (A1, A2, A3).

## Code

Written and executed in order:

| File | Purpose | Status |
|------|---------|--------|
| `step9_p1.py` | Main P1 runner: extraction + ablation protocol | Final |
| `p1_report.py` | Applies pre-reg decision rule → `p1_verdict.json` | Final |
| `p1_full_table.py` | Multi-k, multi-site results markdown | Final |
| `step10_unit_norm_failure_mode.py` | Re-runs A1-restricted shared under unit-norm ablation (predicted failure mode) | Done |
| `step11_compare_extractions.py` | Decomposes A1 vs A2 contributions to sign-flip | Done |
| `step12_additional_sites.py` | Replication at 3 high-CKA sites beyond primary | Done |
| `step13_layer3_probe.py` | Linear probe of layer-3 shared directions | Done |
| `step14_more_layer3_sites.py` | Expand layer-3 coverage (6 more sites) | Done |
| `step15_joint_ablation.py` | Disambiguates "inertness" vs "redundancy" at layer 3 | Done |
| `data_modp.py` / `config_modp.py` / `train_modp.py` / `run_modp_zoo.py` | Mod-p (p=23) task variant | Done |
| `step9_p1_modp.py` | Full P1 protocol on mod-p zoo | Done |
| `config_deep.py` / `run_deep_zoo.py` | 6-layer deep architecture variant (45 models) | Done |
| `step9_p1_deep.py` / `deep_p1_report.py` | Full P1 protocol on 6-layer zoo (7 sites) | Done |
| `step16_deep_probing.py` | Probe 6-layer zoo shared directions at layers 0/3/5 | Done |
| `config_deep8.py` / `run_deep8_zoo.py` | 8-layer zoo variant (21 models) | Done |
| `step9_p1_deep8.py` | P1 protocol on 8-layer zoo at L3/L5/L7 + control | Done |
| `step17_deep8_probing.py` | Probe 8-layer zoo shared at L3/L5/L7 | Done |
| `make_figures.py` / `make_fig5.py` / `make_fig6.py` | Paper figures fig1-6 | Done |

## Result files

All under `results/p1/`:

| File | What it holds |
|------|---------------|
| `p1_results.json` | Full P1 per-model per-k data for 3 primary + 1 control site (corrected for A3 bug) |
| `p1_verdict.json` | Decision-rule verdict |
| `p1_full_table.md` | Human-readable per-site per-k table |
| `unit_norm_failure_mode.json` | Step 10 unit-norm results |
| `extraction_decomposition.json` | Step 11 A1-vs-A2 split |
| `additional_sites.json` | 3 additional high-CKA sites (layers 1, 3, 3) |
| `layer3_probing.json` | Probing layer-3 shared against arithmetic features |
| `layer3_expanded.json` | 6 additional layer-3 positions |
| `joint_ablation.json` | Shared ∪ complement joint ablation (step 15) |

## Pre-registration commits

Three amendments layered during piloting (each before any clean run):

- **A1**: restrict generalized eigenproblem to top-32 PCA subspace of C_total.
- **A2**: use C_total-orthonormal eigenvectors + whitened random baselines.
- **A3**: cap K_VALUES at K_pca/2 = 16 so top-k and bottom-k are disjoint.

Full text and timestamps in `P1_PREREGISTRATION.md`.

## Multi-AI review history

Four multi-AI review rounds during the project:

1. Initial review of existing experiment (before any P1 work): all 3
   providers said "rewrite as positive reframe via the inverted
   eigenproblem." Motivated P1 design.
2. Design review of P1: unanimous ENDORSE-WITH-CHANGES, led to tightened
   decision rule + confound controls.
3. Post-smoke design decision: unanimous endorsement of variance-matched
   ablation (Option A, leading to Amendment A2).
4. Final paper review: unanimous ENDORSE-WITH-CHANGES with consensus:
   lead with methodology (A1/A2), present layer-dependence as hypothesis,
   add joint-ablation disambiguation test (step 15).

Response records live in `/tmp/octo-*/` (ephemeral, not committed).

## Headline findings

1. **Methodological corrections (A1, A2) are publishable alone**, per
   consensus of three reviewers. Standard unit-norm ablation on
   unrestricted generalized-eigenproblem shared subspaces is confounded
   in two ways; restricting the eigenproblem and variance-matching the
   random baseline produces qualitatively different conclusions.
2. **Both shared and orthogonal-high-variance complement directions are
   causally important** at all 14 tested task-relevant sites across this
   zoo, at per-variance rates 10-80× above variance-matched random.
3. **Layer-dependence pattern** at 5/5 task-relevant layer-3 result-digit
   positions: single-subspace ablation of shared produces near-zero drop
   while single-subspace ablation of complement produces 0.37-0.43 drop.
   **Joint-ablation (step 15) reveals this is redundancy, not inertness** —
   at every tested site (6/6 including layer 1), jointly ablating shared
   ∪ complement produces 0.78-0.89 drops, much larger than either alone.
   Shared directions are causally important; their load is masked by the
   complement in single-subspace ablation.
4. **Control site passes cleanly** at all tested low-CKA layer-3
   positions (CKA ≤ 0.15): zero drop for any condition.
5. **The corrected protocol requires joint ablation** for any subspace
   causality claim; single-subspace ablation systematically
   underestimates redundantly-encoded structure.
6. **Mod-p replication (33-model zoo, mod 23 task)**: pre-registered
   verdict REPLICATES. Hidden load positive at all 3 primary sites;
   smaller magnitudes than main zoo (0.03-0.17 vs 0.36-0.64) consistent
   with less task structure to redundantly encode.
7. **6-layer zoo (45 models, same task as main zoo)**: pre-registered
   test of absolute-depth vs last-residual hypothesis. H2
   (last-residual-before-unembed) wins decisively. Shared is primary
   at layer 3 (0.41), dead at layer 5 (0.000 exactly). Hidden-load
   profile shows "computational sandwich": high in middle, low at
   endpoints.
8. **8-layer zoo (21 models, same task)**: pre-registered test of
   N-1 invariance. Shared is dead at layer 7, confirming N-1 across
   three depths (4L→L3, 6L→L5, 8L→L7). Unexpected: hidden load at
   N-1 shrinks with depth (0.36-0.44 → 0.17 → ~0). At 8L layer 7,
   shared directions are still linearly probeable (r=0.94 sum_magnitude,
   r=0.84 carry_col3) despite zero independent causal load — cleanest
   "readable but not causal" data point in the paper.
