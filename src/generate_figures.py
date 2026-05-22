"""Generate README figures from analysis results."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── Color scheme ──────────────────────────────────────────────
BLUE = "#2563eb"
ORANGE = "#f97316"
GRAY = "#94a3b8"
BG = "#0d1117"
TEXT = "#e6edf3"
GRID = "#21262d"


def style_ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=11)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)


# ── Figure 1: Per-Language Accuracy ───────────────────────────
langs = ["EN", "BG", "ZH", "HR", "IT", "SR"]
visual = [0.572, 0.436, 0.380, 0.561, 0.556, 0.519]
textual = [0.838, 0.676, 0.648, 0.803, 0.850, 0.807]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(langs))
w = 0.35

bars1 = ax.bar(x - w/2, visual, w, label="Visual MCQ", color=ORANGE, edgecolor="none", alpha=0.9)
bars2 = ax.bar(x + w/2, textual, w, label="Textual MCQ (1st Place)", color=BLUE, edgecolor="none", alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(langs, fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
style_ax(ax, "Per-Language MCQ Accuracy")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{bar.get_height():.2f}", ha="center", va="bottom", color=TEXT, fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{bar.get_height():.2f}", ha="center", va="bottom", color=TEXT, fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "per_language_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved per_language_accuracy.png")


# ── Figure 2: Track Results Overview ──────────────────────────
tracks = ["Visual\nMCQ", "Textual\nMCQ", "Visual\nOpenQA", "Textual\nOpenQA"]
scores = [0.5076, 0.7538, 0.4286, 0.5285]
ranks = ["8th/9", "1st/3", "8th/8", "2nd/2"]
colors = [ORANGE, BLUE, ORANGE, BLUE]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(tracks, scores, color=colors, edgecolor="none", width=0.6)
bars[0].set_alpha(0.5)
bars[2].set_alpha(0.5)

for bar, rank, score in zip(bars, ranks, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{score:.4f}\n({rank})", ha="center", va="bottom", color=TEXT,
            fontsize=11, fontweight="bold")

ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.0)
style_ax(ax, "Official Competition Results")

plt.tight_layout()
plt.savefig(OUT / "results_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results_overview.png")


# ── Figure 3: Per-Subject Accuracy (top subjects) ────────────
subjects = [
    "Biology", "Sci (Biology)", "Sci (Chemistry)", "Geografija",
    "Sci (Physics)", "Fizika", "Fine Arts", "Chemistry",
    "Physics", "Mathematics"
]
vis_acc = [0.563, 0.638, 0.631, 0.667, 0.578, 0.524, 0.469, 0.386, 0.330, 0.298]
txt_acc = [0.860, 0.846, 0.818, 0.667, 0.889, 0.902, 0.893, 0.716, 0.634, 0.651]

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(subjects))
w = 0.35

bars1 = ax.barh(x + w/2, vis_acc, w, label="Visual MCQ", color=ORANGE, edgecolor="none", alpha=0.85)
bars2 = ax.barh(x - w/2, txt_acc, w, label="Textual MCQ", color=BLUE, edgecolor="none", alpha=0.85)

ax.set_yticks(x)
ax.set_yticklabels(subjects, fontsize=11)
ax.set_xlabel("Accuracy", fontsize=12)
ax.set_xlim(0, 1.05)
ax.legend(fontsize=11, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="lower right")
ax.invert_yaxis()
style_ax(ax, "Per-Subject Accuracy Comparison")
ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.7)
ax.grid(axis="y", visible=False)

plt.tight_layout()
plt.savefig(OUT / "per_subject_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved per_subject_accuracy.png")

print(f"\nAll figures saved to {OUT}/")
