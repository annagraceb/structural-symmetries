"""R11 defensive-evidence figure: cross-digit resample + untrained CKA baseline."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS_DIR = "figures"
OUT = os.path.join(FIGS_DIR, "fig_r11_defensive.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))

# Left: cross-digit resample ablation
# Values averaged across 3 baseline 8L models
conditions = ["mean\nablation", "same-digit\nresample", "cross-digit\nresample"]
shared_drops = [0.002, 0.002, 0.002]  # mean of 3 models, all ~0.002
comp_drops = [0.77, 0.85, 0.92]  # mean of 3 models

x = np.arange(3)
w = 0.35
ax1.bar(x - w/2, shared_drops, w, color="#2980b9", edgecolor="black", lw=0.6, label="Shared subspace (k=8)")
ax1.bar(x + w/2, comp_drops, w, color="#c0392b", edgecolor="black", lw=0.6, label="Complement subspace (k=8)")
ax1.set_xticks(x)
ax1.set_xticklabels(conditions)
ax1.set_ylabel("Accuracy drop")
ax1.set_ylim(-0.03, 1.05)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, axis="y")
ax1.set_title("(a) R11 attack A refuted: cross-digit resample\n"
              "Shared silent under all manipulations; complement shows\n"
              "+0.07 digit-specificity gap (cross-digit vs same-digit)",
              fontsize=10)

# Right: untrained vs trained CKA baseline
categories = ["Untrained 8L\n(8 seeds, 28 pairs)", "Trained 8L\n(3 seeds, 3 pairs)"]
shared_means = [0.33, 0.85]
shared_std = [0.08, 0.04]
comp_means = [0.93, 0.78]
comp_std = [0.05, 0.04]
raw_means = [0.88, 0.78]
raw_std = [0.06, 0.04]

x2 = np.arange(2)
ww = 0.25
ax2.bar(x2 - ww, raw_means, ww, yerr=raw_std, capsize=3,
         color="#16a085", edgecolor="black", lw=0.5, label="Raw CKA")
ax2.bar(x2,     shared_means, ww, yerr=shared_std, capsize=3,
         color="#2980b9", edgecolor="black", lw=0.5, label="Shared CKA (k=8)")
ax2.bar(x2 + ww, comp_means, ww, yerr=comp_std, capsize=3,
         color="#8e44ad", edgecolor="black", lw=0.5, label="Complement CKA (k=8)")
ax2.set_xticks(x2)
ax2.set_xticklabels(categories)
ax2.set_ylabel("Cross-model CKA at 8L L7 (N-1)")
ax2.set_ylim(-0.03, 1.05)
ax2.legend(fontsize=10, loc="lower right")
ax2.grid(alpha=0.3, axis="y")
ax2.set_title("(b) R11 attack B refuted: untrained baseline\n"
              "Training adds +0.52 to shared-CKA (≈6.5σ);\n"
              "Training decorrelates complement (−0.15)",
              fontsize=10)

fig.suptitle("R11 defensive-evidence tests — both attacks refuted",
              fontsize=12, y=1.01)
plt.tight_layout()
os.makedirs(FIGS_DIR, exist_ok=True)
plt.savefig(OUT, dpi=160, bbox_inches="tight")
plt.close()
print(f"Wrote {OUT}")
