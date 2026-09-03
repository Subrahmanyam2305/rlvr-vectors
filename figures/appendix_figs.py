"""Appendix figures: heatmap, gate-by-block, validation curves, protocol diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
SPEC_DATA = os.path.join(os.path.dirname(OUTDIR), "outputs", "spectral_data.json")
AC_DATA = os.path.join(os.path.dirname(OUTDIR), "outputs", "analytical_connection.json")
OVERNIGHT = os.path.join(os.path.dirname(OUTDIR), "outputs", "overnight_results.json")


def load_spectral():
    with open(SPEC_DATA) as f:
        return json.load(f)

def load_analytical():
    with open(AC_DATA) as f:
        return json.load(f)


def fig_a1_heatmap():
    """Block x projection-type heatmap of rank-1 fraction."""
    spec = load_spectral()
    proj_types = ["Q", "K", "V", "O", "FFN_gate", "FFN_up", "FFN_down"]
    proj_labels = ["Q", "K", "V", "O", "Gate", "Up", "Down"]
    n_blocks = 28

    grid = np.full((n_blocks, len(proj_types)), np.nan)
    for entry in spec:
        if entry["layer_idx"] < 0:
            continue
        lt = entry["layer_type"]
        if lt in proj_types:
            row = entry["layer_idx"]
            col = proj_types.index(lt)
            grid[row, col] = entry["rank1_frac"]

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.9, interpolation="nearest")
    ax.set_xticks(range(len(proj_labels)))
    ax.set_xticklabels(proj_labels, fontsize=10)
    ax.set_yticks(range(0, n_blocks, 2))
    ax.set_yticklabels(range(0, n_blocks, 2), fontsize=9)
    ax.set_ylabel("Transformer Block", fontsize=12)
    ax.set_xlabel("Projection Type", fontsize=12)
    ax.set_title(r"Rank-1 Fraction $\rho$ by Block and Projection", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label(r"$\rho$", fontsize=12)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figA1_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def fig_a2_gate_by_block():
    """Gate attenuation ratio and sign agreement by transformer block."""
    ac = load_analytical()
    vtx_src = np.array(ac["per_layer_vtx_source"])
    vtx_tgt = np.array(ac["per_layer_vtx_target"])
    n_layers = len(vtx_src)

    ratios = np.abs(vtx_tgt) / (np.abs(vtx_src) + 1e-8)
    sign_agree = (np.sign(vtx_src) == np.sign(vtx_tgt)).astype(float)
    block_indices = np.arange(n_layers) // 2
    n_blocks = int(block_indices.max()) + 1

    block_ratios = np.array([ratios[block_indices == b].mean() for b in range(n_blocks)])
    block_sign = np.array([sign_agree[block_indices == b].mean() for b in range(n_blocks)])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.bar(range(n_blocks), block_ratios, color="#1565c0", alpha=0.7, edgecolor="white")
    ax1.axhline(0.456, color="#c62828", linestyle="--", linewidth=1.5, label="Mean (0.456)")
    ax1.set_ylabel("Magnitude ratio\n" + r"$|g^{\mathrm{tgt}}| / |g^{\mathrm{src}}|$", fontsize=11)
    ax1.set_title("Gate Attenuation by Transformer Block", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.2)

    colors_sign = ["#c62828" if s < 0.8 else "#1565c0" for s in block_sign]
    ax2.bar(range(n_blocks), block_sign * 100, color=colors_sign, alpha=0.7, edgecolor="white")
    ax2.axhline(88.0, color="#c62828", linestyle="--", linewidth=1.5, label="Mean (88%)")
    ax2.set_ylabel("Sign agreement (%)", fontsize=11)
    ax2.set_xlabel("Transformer Block", fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figA2_gate_by_block.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def fig_a3_val_curves():
    """Validation curves for alpha (exploratory n=50 data, clearly labeled)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # SVD alpha sweep (from overnight_results)
    alphas_svd = [0.5, 1.0, 2.0, 3.0, 5.0]
    accs_svd = [54.0, 50.0, 48.0, 52.0, 24.0]
    ax1.plot(alphas_svd, accs_svd, "o-", color="#1565c0", linewidth=2, markersize=8, label="SVD full")

    alphas_topk = [2.0, 2.0, 2.0]
    ks = [5, 10, 15]
    accs_topk = [50.0, 54.0, 56.0]

    alphas_md = [0.02, 0.05, 0.1, 0.2]
    accs_md = [52.0, 60.0, 34.0, 0.0]
    ax1.plot(alphas_md, accs_md, "s-", color="#e65100", linewidth=2, markersize=8, label="Mean-diff")

    ax1.axhline(46.0, color="gray", linestyle="--", alpha=0.6, label="Baseline (46%)")
    ax1.set_xlabel(r"$\alpha$", fontsize=13)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title(r"$\alpha$ Sweep (n=50, exploratory)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)
    ax1.set_ylim(0, 70)

    ax1.text(0.5, 0.02, "CAUTION: same data used for tuning and evaluation",
             transform=ax1.transAxes, fontsize=8, ha="center", color="#c62828",
             style="italic", bbox=dict(facecolor="#fff3e0", edgecolor="#e65100",
                                       alpha=0.8, boxstyle="round,pad=0.3"))

    # Top-K sweep
    ax2.plot(ks, accs_topk, "D-", color="#2e7d32", linewidth=2, markersize=8)
    ax2.axhline(46.0, color="gray", linestyle="--", alpha=0.6, label="Baseline")
    ax2.axhline(48.0, color="#1565c0", linestyle=":", alpha=0.6, label=r"SVD full ($\alpha$=2.0)")
    ax2.set_xlabel("K (top layers)", fontsize=13)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title(r"Top-K Selection at $\alpha$=2.0 (n=50, exploratory)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.2)
    ax2.set_ylim(30, 65)
    ax2.set_xticks(ks)

    ax2.text(0.5, 0.02, "CAUTION: same data used for tuning and evaluation",
             transform=ax2.transAxes, fontsize=8, ha="center", color="#c62828",
             style="italic", bbox=dict(facecolor="#fff3e0", edgecolor="#e65100",
                                       alpha=0.8, boxstyle="round,pad=0.3"))

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figA3_val_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def fig_a4_protocol():
    """Calibration/validation/test protocol diagram."""
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")

    total = 500
    splits = [
        ("Test (n=400)", 0, 400, "#1565c0"),
        ("Val (n=50)", 400, 450, "#e65100"),
        ("Cal (n=50)", 450, 500, "#2e7d32"),
    ]
    bar_y, bar_h = 0.3, 0.25

    for label, start, end, color in splits:
        width = (end - start) / total
        x = start / total
        rect = plt.Rectangle((x, bar_y), width, bar_h, facecolor=color, alpha=0.5,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width / 2, bar_y + bar_h / 2, label,
                ha="center", va="center", fontsize=11, fontweight="bold", color=color)
        ax.text(x + width / 2, bar_y - 0.06, f"[{start}\u2013{end-1}]",
                ha="center", va="top", fontsize=9, color="#555")

    arrows = [
        (0.95, 0.95, "Sign orient +\nmean-diff vectors", "#2e7d32"),
        (0.85, 0.75, r"Select $\alpha$, K", "#e65100"),
        (0.4, 0.75, "Final evaluation\n(reported in tables)", "#1565c0"),
    ]
    for x, y, txt, c in arrows:
        ax.annotate(txt, xy=(x, bar_y + bar_h), xytext=(x, y),
                    fontsize=9, ha="center", color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.12, 1.15)
    ax.set_title("MATH500 Data Split Protocol", fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figA4_protocol.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    fig_a1_heatmap()
    fig_a2_gate_by_block()
    fig_a3_val_curves()
    fig_a4_protocol()
