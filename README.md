# Cross-Model Shared Directions Are Redundantly Backed

**Three methodological corrections for CKA-based subspace causality analyses, evidenced across four small arithmetic transformer zoos.**

This repository contains the full code, pre-registrations, trained model weights, result JSONs, figures, and paper draft for the study.

## The finding in one paragraph

Standard cross-model universality analyses use CKA or generalized eigenproblems to identify "shared" subspaces across independently-trained networks, then ablate those subspaces to test causality. On a 33-model zoo trained to 5-digit addition, this standard protocol reported "shared subspace ablation hurts *less* than random" (p<0.001) — which would suggest shared directions are not causally load-bearing. We show this finding is an artifact of three independent methodological confounds: the unrestricted generalized eigenproblem selects low-variance directions (A1); unit-norm random baselines are not variance-matched to the structured subspace (A2); and single-subspace ablation is blind to causal signal that the network encodes redundantly across the subspace and its orthogonal complement (A3). Joint ablation of shared ∪ complement reveals **hidden load of 0.36–0.64 of accuracy drop** that single-subspace ablation cannot see. The phenomenon replicates on a second task (mod-23 arithmetic) and at two additional depths (6-layer and 8-layer architectures), with the "shared dead at last-residual-before-unembed" signature appearing robustly at layer N-1 across all three tested depths.

## Read the paper

- **`PAPER.pdf`** — 20-page paper with all figures inlined
- `PAPER_DRAFT.md` — same content in Markdown
- `VARIANCE_CONFOUND_DERIVATION.md` — first-principles derivation of the A1/A2/A3 corrections

## Pre-registrations (all committed before their respective experiments)

| File | Experiment |
|------|------------|
| `P1_PREREGISTRATION.md` | Main zoo (5-digit addition, 4 layers). Includes amendments A1, A2, A3, A4 documented with timestamps. |
| `MOD_P_PREREGISTRATION.md` | Phase 1: Mod-23 replication. |
| `PHASE2_PREREGISTRATION.md` | Phase 2: 6-layer zoo. H1/H2/H3 hypotheses for layer-dependence. |
| `PHASE3_PREREGISTRATION.md` | Phase 3: 8-layer zoo. H2a (N-1 invariance). |

## Four zoos

- `models/` — main zoo, 33 models, 4 layers, 5-digit addition
- `models_modp/` — Phase 1, 33 models, 4 layers, mod-23 addition
- `models_deep/` — Phase 2, 45 models, 6 layers, 5-digit addition
- `models_deep8/` — Phase 3, 21 models, 8 layers, 5-digit addition

All `metadata.json` files record training hyperparameters and convergence step; all `model.pt` files are the final weights used in the reported analysis.

## Results

All under `results/`:

- `results/p1/` — main-zoo P1 protocol (all corrections applied) + joint ablation + extraction decomposition + additional sites + layer-3 expansion
- `results/modp/` — Phase 1 mod-p results
- `results/deep/` — Phase 2 6-layer results (including layer-sweep and probing)
- `results/deep8/` — Phase 3 8-layer results (including probing at N-1)

Large `aligned_*/` directories are gitignored (they are regenerable from the saved model weights and the alignment step of any `step9_p1*.py`).

## Figures

All in `figures/`:

1. `fig1_joint_ablation.png` — Joint ablation at six main-zoo sites.
2. `fig2_layer_dependence.png` — Single-subspace drops by layer (main zoo).
3. `fig3_modp_vs_main.png` — Hidden load main zoo vs mod-p.
4. `fig4_unit_norm_vs_matched.png` — A1 PCA-restriction flips the Step-7 sign.
5. `fig5_deep_layer_sweep.png` — 6-layer zoo depth sweep.
6. `fig6_hidden_load_vs_depth.png` — Hidden load at N-1 across 4L/6L/8L.

## Code

### Main zoo (pipeline from design spec, with P1 corrections on top)

- `config.py`, `data.py`, `model.py`, `train.py`, `collect_activations.py`, `analysis.py`, `report.py`
- `run_zoo.py` — trains the 33-model zoo
- `run_pipeline.py` — runs Steps 3-7 (CKA, Procrustes, shared subspace, probing, Step-7 ablation)

### P1 protocol (the three corrections and follow-ups)

- `step9_p1.py` — main P1 runner (applies A1 + A2)
- `step10_unit_norm_failure_mode.py` — reproduces Step-7's "shared < random" sign under unit-norm on corrected extraction
- `step11_compare_extractions.py` — decomposes A1 vs A2 contributions to the sign flip
- `step12_additional_sites.py` — replication at 3 new high-CKA sites
- `step13_layer3_probe.py` — probes layer-3 shared directions against arithmetic features
- `step14_more_layer3_sites.py` — expands layer-3 coverage to 6 additional positions
- `step15_joint_ablation.py` — the A3 disambiguation test (joint ablation)
- `p1_report.py`, `p1_full_table.py` — apply pre-registered decision rule / produce summary tables

### Phase 1 — mod-p zoo

- `config_modp.py`, `data_modp.py`, `train_modp.py`, `run_modp_zoo.py`
- `step9_p1_modp.py` — full protocol on the mod-p zoo

### Phase 2 — 6-layer zoo

- `config_deep.py`, `run_deep_zoo.py`
- `step9_p1_deep.py` — protocol at all 6 layers at `result_0`
- `deep_p1_report.py` — applies PHASE2_PREREGISTRATION decision rule
- `step16_deep_probing.py` — probes layers 0/3/5 of the 6-layer zoo

### Phase 3 — 8-layer zoo

- `config_deep8.py`, `run_deep8_zoo.py`
- `step9_p1_deep8.py` — protocol at layers 3/5/7 + control
- `step17_deep8_probing.py` — probes shared directions at L3/L5/L7 of the 8-layer zoo

### Figure + report scripts

- `make_figures.py` — figures 1–4
- `make_fig5.py` — 6-layer layer sweep
- `make_fig6.py` — hidden-load vs depth
- `build_pdf.py` — markdown → PDF build script

## Reproducing the paper

Rough procedure (each zoo takes minutes to hours on an RTX 3060):

```bash
# 1. Train the main zoo and the three replication zoos
python run_zoo.py          # main zoo (5-digit addition, 4 layers, 33 models)
python run_modp_zoo.py     # mod-23 zoo (4 layers, 33 models)
python run_deep_zoo.py     # 6-layer zoo (45 models)
python run_deep8_zoo.py    # 8-layer zoo (21 models)

# 2. Collect activations and run the original pipeline (Steps 3-7) on the main zoo
python collect_activations.py
python run_pipeline.py

# 3. Run the three P1 corrections (A1 + A2 + A3) on the main zoo
python step9_p1.py
python p1_report.py
python p1_full_table.py

# 4. Supporting analyses on the main zoo
python step10_unit_norm_failure_mode.py
python step11_compare_extractions.py
python step12_additional_sites.py
python step13_layer3_probe.py
python step14_more_layer3_sites.py
python step15_joint_ablation.py

# 5. Phase 1 mod-p replication
python step9_p1_modp.py

# 6. Phase 2 6-layer replication
python step9_p1_deep.py
python deep_p1_report.py
python step16_deep_probing.py

# 7. Phase 3 8-layer N-1 invariance test
python step9_p1_deep8.py
python step17_deep8_probing.py

# 8. Generate figures and PDF
python make_figures.py
python make_fig5.py
python make_fig6.py
python build_pdf.py
```

## How this work was produced

This project was built in a tight iterative loop with multi-provider LLM critic rounds at every major decision point. The critic rounds are not merely a stylistic choice — the Claude paradox-hunter review that caught redundancy-vs-inertness ambiguity directly led to the A3 joint-ablation correction, without which the paper's conclusion would have been inverted. Each critic round's verdicts and the specific changes they triggered are listed in `SESSION_HANDOFF.md`.

## License

MIT.
