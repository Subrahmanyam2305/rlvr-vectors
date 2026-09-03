"""Figure 2: Spectral concentration violin/box by matrix type with shape-matched null."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(OUTDIR), "outputs", "spectral_data.json")

TYPE_ORDER = ["Q", "K", "V", "O", "FFN_gate", "FFN_up", "FFN_down"]
TYPE_LABELS = ["Q", "K", "V", "O", "Gate", "Up", "Down"]
COLORS = {
    "Q": "#1565c0", "K": "#1976d2", "V": "#1e88e5", "O": "#42a5f5",
    "FFN_gate": "#e65100", "FFN_up": "#ef6c00", "FFN_down": "#f57c00",
}

def marchenko_pastur_top_fraction(m, n, num_samples=5000, rng_seed=42):
    """Expected rank-1 fraction for i.i.d. Gaussian matrices of shape (m, n)."""
    rng = np.random.RandomState(rng_seed)
    fracs = []
    for _ in range(num_samples):
        G = rng.randn(m, n)
        s = np.linalg.svd(G, compute_uv=False)
        fracs.append((s[0]**2) / np.sum(s**2))
    return np.array(fracs)

def draw():
    with open(DATA) as f:
        spec = json.load(f)

    groups = {t: [] for t in TYPE_ORDER}
    shapes_per_type = {}
    for entry in spec:
        lt = entry["layer_type"]
        if lt in groups:
            groups[lt].append(entry["rank1_frac"])
            shapes_per_type[lt] = tuple(entry["shape"])

    null_medians = {}
    for lt in TYPE_ORDER:
        if lt in shapes_per_type:
            m, n = shapes_per_type[lt]
            null_fracs = marchenko_pastur_top_fraction(min(m, 256), min(n, 256), num_samples=2000)
            null_medians[lt] = np.median(null_fracs)

    fig, ax = plt.subplots(figsize=(9, 5))

    data_list = [groups[t] for t in TYPE_ORDER]
    positions = np.arange(1, len(TYPE_ORDER) + 1)

    parts = ax.violinplot(data_list, positions=positions, showmedians=False, showextrema=False)
    for i, (pc, t) in enumerate(zip(parts["bodies"], TYPE_ORDER)):
        pc.set_facecolor(COLORS[t])
        pc.set_alpha(0.4)

    bp = ax.boxplot(data_list, positions=positions, widths=0.25, patch_artist=True,
                    showfliers=True, flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for i, (patch, t) in enumerate(zip(bp["boxes"], TYPE_ORDER)):
        patch.set_facecolor(COLORS[t])
        patch.set_alpha(0.6)

    for i, t in enumerate(TYPE_ORDER):
        if t in null_medians:
            ax.plot(positions[i], null_medians[t], marker="D", color="red",
                    markersize=7, zorder=10, markeredgecolor="darkred", markeredgewidth=1)

    ax.plot([], [], marker="D", color="red", linestyle="none", markeredgecolor="darkred",
            label="i.i.d. Gaussian null (median)")
    ax.legend(loc="upper right", fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(TYPE_LABELS, fontsize=12)
    ax.set_ylabel(r"Rank-1 fraction $\rho$", fontsize=13)
    ax.set_title("Spectral Concentration by Matrix Type", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.258, color="gray", linestyle="--", alpha=0.5, label="Overall mean (0.258)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    attn_x = np.mean(positions[:4])
    mlp_x = np.mean(positions[4:])
    y_bot = -0.08
    ax.annotate("Attention", xy=(attn_x, y_bot), fontsize=11, ha="center", va="top",
                color="#1565c0", fontweight="bold", annotation_clip=False)
    ax.annotate("MLP", xy=(mlp_x, y_bot), fontsize=11, ha="center", va="top",
                color="#e65100", fontweight="bold", annotation_clip=False)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig2_spectral_concentration.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

if __name__ == "__main__":
    draw()
