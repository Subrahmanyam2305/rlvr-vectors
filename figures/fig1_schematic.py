"""Figure 1: Proposition 1 schematic — rank-1 weight update as conditional steering."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

def draw():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 4.0)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.4", facecolor="#e8eaf6", edgecolor="#3949ab", linewidth=1.5)
    box_kw2 = dict(boxstyle="round,pad=0.4", facecolor="#fff3e0", edgecolor="#e65100", linewidth=1.5)
    box_kw3 = dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=1.5)
    result_kw = dict(boxstyle="round,pad=0.5", facecolor="#fce4ec", edgecolor="#c62828", linewidth=2)

    # --- Top row: standard forward pass ---
    ax.text(0.5, 3.0, r"$x$", fontsize=16, ha="center", va="center", bbox=box_kw)
    ax.annotate("", xy=(2.0, 3.0), xytext=(1.1, 3.0),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#3949ab", lw=2))
    ax.text(3.2, 3.0, r"$W_l$", fontsize=16, ha="center", va="center", bbox=box_kw)
    ax.annotate("", xy=(5.0, 3.0), xytext=(4.3, 3.0),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#3949ab", lw=2))
    ax.text(6.2, 3.0, r"$y^{\mathrm{old}} = W_l x$", fontsize=14, ha="center", va="center", bbox=box_kw)

    ax.text(-0.3, 3.0, "Original:", fontsize=11, ha="right", va="center", color="#555", style="italic")

    # --- Bottom row: modified forward pass ---
    ax.text(0.5, 0.5, r"$x$", fontsize=16, ha="center", va="center", bbox=box_kw)

    # Branch 1: through W_l
    ax.annotate("", xy=(2.0, 1.3), xytext=(1.1, 0.7),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#3949ab", lw=1.5))
    ax.text(3.0, 1.5, r"$W_l$", fontsize=14, ha="center", va="center", bbox=box_kw)

    # Branch 2: through v (input detector)
    ax.annotate("", xy=(2.0, -0.3), xytext=(1.1, 0.3),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#e65100", lw=1.5))
    ax.text(3.0, -0.5, r"$v$", fontsize=16, ha="center", va="center", bbox=box_kw2,
            fontweight="bold")
    ax.text(3.0, -1.2, "input detector", fontsize=9, ha="center", color="#e65100", style="italic")

    # v^T x scalar gate
    ax.annotate("", xy=(4.7, -0.5), xytext=(3.5, -0.5),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#e65100", lw=1.5))
    ax.text(5.6, -0.5, r"$g = v^\top x$", fontsize=13, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e1", edgecolor="#f57f17", linewidth=1.5))
    ax.text(5.6, -1.2, "scalar gate", fontsize=9, ha="center", color="#f57f17", style="italic")

    # Multiply by sigma and u
    ax.annotate("", xy=(7.3, -0.1), xytext=(6.5, -0.4),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#2e7d32", lw=1.5))
    ax.text(8.1, 0.2, r"$\sigma_1 \cdot g \cdot u$", fontsize=13, ha="center", va="center",
            bbox=box_kw3)
    ax.text(8.1, -0.5, "steering term", fontsize=9, ha="center", color="#2e7d32", style="italic")

    # u label
    ax.text(7.6, -0.9, r"$u$: output direction", fontsize=9, ha="center", color="#2e7d32", style="italic")

    # W_l output merges with steering
    ax.annotate("", xy=(7.3, 1.5), xytext=(3.8, 1.5),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#3949ab", lw=1.5))

    # Sum node
    ax.text(8.1, 1.5, r"$+$", fontsize=20, ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="#333", linewidth=2))
    ax.annotate("", xy=(8.1, 1.1), xytext=(8.1, 0.6),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#2e7d32", lw=1.5))

    # Result
    ax.annotate("", xy=(9.5, 1.5), xytext=(8.6, 1.5),
                arrowprops=dict(arrowstyle="->,head_width=0.3", color="#333", lw=2))
    ax.text(10.2, 1.5, r"$y^{\mathrm{new}}$", fontsize=14, ha="center", va="center", bbox=result_kw)

    ax.text(-0.3, 0.5, "Modified:", fontsize=11, ha="right", va="center", color="#555", style="italic")

    # Equation at very top
    ax.text(5.0, 3.8, r"$(W_l + \sigma_1 u v^\top)\, x \;=\; W_l x \;+\; \sigma_1 (v^\top x)\, u$",
            fontsize=14, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#999", linewidth=1))

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig1_proposition_schematic.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

if __name__ == "__main__":
    draw()
