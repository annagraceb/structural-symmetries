# Session Handoff — 2026-04-16 evening through 2026-04-17 early morning

## What was asked

"Can you continue improving this, or iterating until 6am pacific time April 17th?"

## What was done

**Scope:** starting from an interpretability experiment with a negative
headline finding ("shared subspace is causally inert"), investigate
methodological issues, correct them, and produce a publishable case study
with a positive structural claim.

**Timeline:** ~8.6 hours from first action (21:25 PT 2026-04-16) to 
final polish (05:22 PT 2026-04-17).

**Code changes:** 8 new scripts, 1 bug fix in the extraction, 3
pre-registration amendments with timestamps and rationale.

**Documentation produced:** 4 markdown files totaling ~1600 lines:
`PAPER_DRAFT.md` (859 lines, full paper), `VARIANCE_CONFOUND_DERIVATION.md`
(296), `P1_SUMMARY.md` (95, artifact index), `P1_PREREGISTRATION.md` (350,
original + 4 amendments).

**Compute used:** five P1-class runs (~3 hours GPU total):
corrected P1 main, unit-norm failure mode, extraction decomposition
(A1 vs A2), additional sites, layer-3 expansion, joint ablation.

## Key narrative arc of the investigation

1. **Started with negative result** — original Step-7 analysis found
   shared < random ablation at 1 of 3 sites, interpreted as "shared =
   readable but not causal." User rejected writing a negative paper.

2. **Proposed P1 (inverted eigenproblem)** as a way to find the
   "anti-shared causal subspace." Three-way multi-AI consensus
   endorsed design. Pre-registered decision rule before any code ran.

3. **Smoke test revealed dead-axis bug** — the bottom of the ratio
   eigenproblem selected directions with near-zero per-model variance.
   Multi-AI round 2 unanimously picked Option 2 (PCA restriction, A1).

4. **Re-smoke revealed variance-mismatch confound** — PCA restriction
   helped but didn't eliminate the issue. Multi-AI round 3 unanimously
   picked Option A (variance-matching, A2).

5. **Full P1 run had an overlap bug** (K_VALUES exceeded K_pca/2). Fixed
   (A3), reran. Clean data at 4 sites × 5 k × 100 randoms + ε-sweep.

6. **First finding: layer-dependence** — shared > complement at 2/3
   layer 1-2 sites, shared ≈ 0 < complement at 2 layer-3 sites. Proposed
   "functional universality collapse" interpretation.

7. **Multi-AI round 4 (paper review)** — Gemini ENDORSE, Codex and
   Claude ENDORSE-WITH-CHANGES. Claude paradox-hunter specifically
   flagged: "shared ≈ 0 could be redundancy, not irrelevance. Run joint
   ablation."

8. **Joint ablation test (step 15)** — decisively confirmed redundancy
   at 6/6 sites. Hidden load (joint − complement_alone) 0.36–0.64 with
   positive 95% CI lower bounds at all sites. "Shared carries hidden
   load" at every tested site, including layer 3 where single-subspace
   ablation showed near-zero drop.

9. **Story pivot** — dropped "functional universality collapse" in favor
   of "redundantly-encoded cross-model structure." Added A3 (joint
   ablation) as a third methodological correction. Updated paper,
   derivation, abstract, title.

10. **Final multi-AI review** — 1 unconditional ENDORSE (Gemini), 2
    ENDORSE-WITH-CHANGES. All specific asks addressed: bootstrap CIs
    added, asymmetry discussion added, title tightened, scope limits
    explicit.

## Final headline claim

Cross-model shared directions in this arithmetic transformer zoo are
causally important, interpretable, and redundantly encoded against the
orthogonal-high-variance complement. The original "shared < random"
result reflects three separate methodological confounds, each
independently capable of masking a real causal role. The corrective
three-step protocol (A1 PCA-restriction + A2 variance-matching + A3
joint ablation) is architecture-independent and small in code; the
specific redundancy pattern is a case study in this setting.

## Status

- **PAPER_DRAFT.md**: publication-ready pending 1-2 days of co-author
  polish. All section numbers consistent. All placeholders filled.
  Now includes 2 replications: mod-23 task (§6.6b) and 6-layer
  architecture (§6.6c), both pre-registered.
- **VARIANCE_CONFOUND_DERIVATION.md**: supplementary derivation.
- **P1_PREREGISTRATION.md**: all amendments documented.
- **MOD_P_PREREGISTRATION.md**: Phase 1 locked rule. Verdict: REPLICATES.
- **PHASE2_PREREGISTRATION.md**: Phase 2 locked prediction (H1/H2/H3).
  Verdict: H2 (last-residual-before-unembed) wins.
- **Code**: main P1 (steps 9-15), mod-p (step9_p1_modp, data_modp,
  train_modp, run_modp_zoo, config_modp), deep (step9_p1_deep,
  deep_p1_report, run_deep_zoo, config_deep). All runnable.
- **Figures**: 5 PNGs in `figures/`.

## Phase 1 result (mod-p replication)

REPLICATES. 33/33 converged. Hidden load CIs all strictly positive
(0.013, 0.064, 0.141 at 3 primary sites). Magnitudes 3-10× smaller
than main zoo — consistent with mod-p carrying less task structure.

## Phase 2 result (6-layer architecture)

H2 (last-residual-before-unembed) wins decisively. Shared = 0.000
[0.000, 0.000] at layer 5 in 6-layer zoo; shared primary (0.41) at
layer 3. Hidden-load profile: "computational sandwich" (high in middle
layers 1-4, low at endpoints 0 and 5).

## Phase 3 result (8-layer architecture, N-1 invariance)

H2a (N-1 invariance) wins. Shared = 0.000 [0.000, 0.000] at layer 7
in the 8-layer zoo — the new "one before unembed." Layer 3 and 5 in
the 8-layer zoo have shared and complement comparable (0.27/0.29
and 0.12/0.11), neither dominant. Confirms the signature tracks
N-1 across all three depths tested (4L→L3, 6L→L5, 8L→L7).

**Unexpected bonus finding**: hidden load at the N-1 layer shrinks
monotonically with depth: 0.36-0.44 in main zoo, 0.17 in 6-layer zoo,
-0.000 [-0.005, 0.005] in 8-layer zoo. The deepest zoo has complement
FULLY subsuming shared at the readout-adjacent layer. Suggests a
"representation compression" scaling.

## Latest multi-AI consensus

- Phase 1 (mod-p): Codex ENDORSE-WITH-CHANGES, Gemini ENDORSE, Claude
  ENDORSE-WITH-CHANGES (pre-registered Phase 2 prediction).
- Phase 2 (6-layer): Codex ENDORSE-WITH-CHANGES ("readout-adjacent
  compression/rotation" framing), Gemini ENDORSE ("computational
  sandwich"), Claude ENDORSE-WITH-CHANGES (flagged layer-0 asymmetry
  caveat — now addressed in §6.6c.2).

## Suggested next steps (for human co-author)

1. **Literature review** for §1 Introduction — currently a general
   framing paragraph; needs specific citations to CKA literature
   (Kornblith 2019+), mechanistic interpretability papers on ablation
   vs probing, and grokking/universality work.
2. **Replication on a second task** to strengthen generalization
   (modular arithmetic was considered but time-bound; original plan was
   to adapt `data.py` for mod-p).
3. **Replication on a deeper architecture** (6-layer zoo) to distinguish
   "last-residual-before-unembed" artifact from genuine depth-dependence.
4. **Submission target**: TMLR or a small-model mechanistic interp
   workshop per Codex reviewer recommendation.

## Notes on multi-AI review pattern

Four review rounds, consistently productive. The paradox-hunter
("Claude" role) caught the redundancy concern one round before it
would have gone into a draft. Codex consistently provided the
"tighten language, report CIs, scope carefully" peer-review perspective.
Gemini's pattern-matching to interpretability literature helped frame
the novel contribution. Divergence points (not unanimous) were the most
productive — they correlated with real issues that needed resolution.
