"""Phase 4 — Dissociation figure.

Shows at the N-1 layer of each zoo (4L, 6L, 8L):
  - Shared-CKA (how cross-model-similar the shared subspace is)
  - Complement-CKA
  - Shared ablation drop (causal importance)
  - Complement ablation drop

The finding: shared-CKA RISES or stays high with depth at N-1, while shared-
drop GOES TO ZERO. Universality and causality dissociate.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGS_DIR = "figures"
OUT = os.path.join(FIGS_DIR, "fig9_universality_vs_causality.png")


# Hard-coded from the Phase 4 analyses and the existing paper summaries.
# (Shared-CKA and complement-CKA were computed live in the Phase 4 analysis
# scripts; drops come from results/p1/p1_results.json, results/deep/p1_results.json,
# results/deep8/p1_results.json.)
data = [
    # depth, layer, label, shared_cka, complement_cka, shared_drop, complement_drop, hidden_load, shared_logit_var, comp_logit_var
    (4, 3, "4L  L3 (N-1)", 0.8055, 0.7806, 0.27,  0.29,  0.58, 0.032, 0.848),
    (6, 5, "6L  L5 (N-1)", 0.8092, 0.8001, 0.00,  0.56,  0.17, None,  None),
    (7, 6, "7L  L6 (N-1)", 0.8158, 0.8519, None,  None,  None, None,  None),  # 3 baselines only, drops inconclusive
    (8, 7, "8L  L7 (N-1)", 0.8505, 0.7844, 0.00,  0.79, -0.00, 0.144, 0.594),
]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.4))

depths = [d[0] for d in data]
shared_cka = [d[3] for d in data]
comp_cka = [d[4] for d in data]
shared_drop = [d[5] for d in data]
comp_drop = [d[6] for d in data]

# Left panel: CKA
ax1.plot(depths, shared_cka, "-o", color="#2980b9", lw=2.2, ms=9, label="Shared-CKA (trained)")
ax1.plot(depths, comp_cka, "--s", color="#8e44ad", lw=2.0, ms=8, label="Complement-CKA (trained)")
ax1.axhline(0.33, color="#e67e22", linestyle=":", lw=1.5, label="Untrained 8L shared-CKA (8 seeds mean)")
ax1.set_xlabel("Transformer depth (n_layers)", fontsize=11)
ax1.set_ylabel("Cross-model CKA at N-1 layer", fontsize=11)
ax1.set_ylim(0.5, 0.95)
ax1.set_xticks([4, 6, 7, 8])
ax1.grid(alpha=0.3)
ax1.legend(fontsize=10, loc="lower right")
ax1.set_title("(a) Cross-model universality at N-1\n(Shared rises; complement slightly declines)",
              fontsize=10.5)

# Right panel: drops (causal importance)
# Only plot depths where drop is measured
drop_depths = [d[0] for d in data if d[5] is not None]
shared_drop_vals = [d[5] for d in data if d[5] is not None]
comp_drop_vals = [d[6] for d in data if d[6] is not None]
ax2.plot(drop_depths, shared_drop_vals, "-o", color="#2980b9", lw=2.2, ms=9, label="Shared ablation drop")
ax2.plot(drop_depths, comp_drop_vals, "--s", color="#8e44ad", lw=2.0, ms=8, label="Complement ablation drop")
ax2.set_xlabel("Transformer depth (n_layers)", fontsize=11)
ax2.set_ylabel("Accuracy drop under ablation at N-1", fontsize=11)
ax2.set_ylim(-0.02, 1.0)
ax2.set_xticks([4, 6, 7, 8])
ax2.grid(alpha=0.3)
ax2.legend(fontsize=10, loc="center right")
ax2.set_title("(b) Causal importance at N-1\n(Shared decays to 0; complement grows)",
              fontsize=10.5)

# Right panel: logit-variance contribution (mechanism)
depths_lv = [d[0] for d in data if d[8] is not None]
shared_lv = [d[8] for d in data if d[8] is not None]
comp_lv = [d[9] for d in data if d[9] is not None]
ax3.plot(depths_lv, shared_lv, "-o", color="#2980b9", lw=2.2, ms=9, label="Shared logit-variance contribution")
ax3.plot(depths_lv, comp_lv, "--s", color="#8e44ad", lw=2.0, ms=8, label="Complement logit-variance contribution")
ax3.axhline(0.049, color="#7f8c8d", linestyle=":", label="Random k=8 baseline (~5%)")
ax3.set_xlabel("Transformer depth (n_layers)", fontsize=11)
ax3.set_ylabel("Fraction of correct-digit logit variance", fontsize=11)
ax3.set_ylim(-0.02, 1.0)
ax3.set_xticks([4, 8])
ax3.grid(alpha=0.3)
ax3.legend(fontsize=9, loc="center right")
ax3.set_title("(c) Logit-variance contribution\n(Shared variance rises with depth; causal drop falls)",
              fontsize=10.5)

fig.suptitle("Stranded universality at N-1: shared-CKA rises with depth while causal role collapses.\n"
             "Shared logit-variance contribution ALSO rises (0.03 → 0.14), but this variance is non-discriminative\n"
             "(mean-ablation preserves 100% accuracy); the complement carries the full decision.",
             fontsize=10.5, y=1.04)

plt.tight_layout()
os.makedirs(FIGS_DIR, exist_ok=True)
plt.savefig(OUT, dpi=160, bbox_inches="tight")
plt.close()
print(f"Wrote {OUT}")
