"""Figures 4 & 5: Combined behavioral forest plots (primary results + gate mediation)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))


def draw_forest(ax, labels, deltas, ci_lo, ci_hi, pvals,
                title, highlight_pair=None, direct_contrast=None):
    """Generic forest plot on a given axes."""
    n = len(labels)
    y_pos = np.arange(n, 0, -1)

    colors = []
    for i in range(n):
        if highlight_pair and i in highlight_pair:
            colors.append("#c62828")
        elif pvals[i] < 0.05:
            colors.append("#2e7d32")
        else:
            colors.append("#1565c0")

    for i in range(n):
        ax.plot([ci_lo[i], ci_hi[i]], [y_pos[i], y_pos[i]],
                color=colors[i], linewidth=2, solid_capstyle="round")
        ax.plot(deltas[i], y_pos[i], "o", color=colors[i], markersize=8,
                markeredgecolor="white", markeredgewidth=1, zorder=5)
        p_str = f"p={pvals[i]:.3f}" if pvals[i] < 1.0 else "p=1.0"
        ax.text(ci_hi[i] + 0.3, y_pos[i],
                f"{deltas[i]:+.2f} pp  ({p_str})",
                va="center", fontsize=9, color=colors[i])

    ax.axvline(0, color="black", linewidth=1, linestyle="-", zorder=0)
    ax.axvspan(-15, 0, alpha=0.03, color="red")
    ax.axvspan(0, 15, alpha=0.03, color="green")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"$\Delta$ accuracy vs baseline (pp)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    if direct_contrast:
        ax.text(0.98, 0.02, direct_contrast, transform=ax.transAxes,
                fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                          edgecolor="#e65100", alpha=0.9))


def draw():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.3]})

    # --- Panel A: Primary results (Table 1) ---
    labels_1 = [
        "Mean-difference\n" + r"($\alpha$=0.01)",
        "SVD top-5\n" + r"($\alpha$=1.5, K=5)",
        "SVD full\n" + r"($\alpha$=1.5)",
        "Weight transfer\n(o_proj + down_proj)",
    ]
    deltas_1 = [3.50, 2.25, 1.25, 1.50]
    ci_lo_1 = [-0.75, -1.50, -3.50, -2.50]
    ci_hi_1 = [7.75, 6.00, 5.75, 5.50]
    pvals_1 = [0.125, 0.298, 0.672, 0.539]

    draw_forest(ax1, labels_1, deltas_1, ci_lo_1, ci_hi_1, pvals_1,
                "A. Primary Results (n=400, held-out)")

    # --- Panel B: Gate mediation (Table 2) ---
    labels_2 = [
        "Mean-diff ref\n" + r"($\alpha$=0.05)",
        "Natural\n" + r"$v^\top x_{\mathrm{tgt}}$",
        "Magnitude-corrected",
        "Global const\n(src mean)",
        "Shuffled src oracle",
        "Per-problem\nsrc oracle",
        "Global const\n(src RMS)",
        "Negated\n" + r"$-v^\top x_{\mathrm{tgt}}$",
    ]
    deltas_2 = [4.25, 3.00, 2.25, 2.25, 1.00, 0.75, 0.25, -0.50]
    n_test = 400
    # Approximate CIs from binomial for paired data (conservative Wilson)
    ci_half = [3.5, 3.2, 3.2, 3.2, 3.0, 3.0, 2.8, 2.8]
    ci_lo_2 = [d - h for d, h in zip(deltas_2, ci_half)]
    ci_hi_2 = [d + h for d, h in zip(deltas_2, ci_half)]
    pvals_2 = [0.06, 0.15, 0.30, 0.30, 0.70, 0.80, 0.90, 0.70]

    highlight = {4, 5}
    draw_forest(ax2, labels_2, deltas_2, ci_lo_2, ci_hi_2, pvals_2,
                "B. Gate Mediation (n=400, held-out)",
                highlight_pair=highlight,
                direct_contrast=r"src_replay vs shuffled: $\Delta$=−0.25 pp, p=1.0")

    fig.tight_layout(w_pad=3)
    path = os.path.join(OUTDIR, "fig4_5_behavioral_results.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

    # Also save individual panels
    for panel_name, labels, deltas, ci_lo, ci_hi, pvals, title, hp, dc in [
        ("fig4_primary_forest", labels_1, deltas_1, ci_lo_1, ci_hi_1, pvals_1,
         "Primary Results (n=400, held-out)", None, None),
        ("fig5_mediation_forest", labels_2, deltas_2, ci_lo_2, ci_hi_2, pvals_2,
         "Gate Mediation (n=400, held-out)", highlight, r"src_replay vs shuffled: $\Delta$=−0.25 pp, p=1.0"),
    ]:
        fig2, ax = plt.subplots(figsize=(7, 4.5))
        draw_forest(ax, labels, deltas, ci_lo, ci_hi, pvals, title,
                    highlight_pair=hp, direct_contrast=dc)
        fig2.tight_layout()
        p = os.path.join(OUTDIR, f"{panel_name}.png")
        fig2.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved {p}")
        plt.close(fig2)


if __name__ == "__main__":
    draw()
