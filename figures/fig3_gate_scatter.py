"""Figure 3: Source vs target gate scatter (log-scale)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(OUTDIR), "outputs", "analytical_connection.json")

def draw():
    with open(DATA) as f:
        ac = json.load(f)

    vtx_src = np.abs(np.array(ac["per_layer_vtx_source"]))
    vtx_tgt = np.abs(np.array(ac["per_layer_vtx_target"]))
    rank1_fracs = np.array(ac["per_layer_rank1_fracs"])

    layer_types = []
    for i in range(len(vtx_src)):
        if i % 2 == 0:
            layer_types.append("down_proj")
        else:
            layer_types.append("o_proj")

    fig, ax = plt.subplots(figsize=(7, 7))

    colors = {"o_proj": "#1565c0", "down_proj": "#e65100"}
    markers = {"o_proj": "o", "down_proj": "s"}

    for lt in ["o_proj", "down_proj"]:
        mask = np.array([t == lt for t in layer_types])
        ax.scatter(vtx_src[mask], vtx_tgt[mask],
                   c=colors[lt], marker=markers[lt], s=50, alpha=0.7,
                   label=lt, edgecolors="white", linewidths=0.5, zorder=5)

    lims = [0.01, max(vtx_src.max(), vtx_tgt.max()) * 1.5]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xscale("log")
    ax.set_yscale("log")

    xs = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 100)
    ax.plot(xs, xs, "k--", alpha=0.4, label=r"$y = x$")
    ax.plot(xs, 0.456 * xs, color="#c62828", linestyle="-", alpha=0.7, linewidth=2,
            label=r"$y = 0.456\,x$ (mean attenuation)")

    ax.fill_between(xs, 0.456 * xs, xs, alpha=0.06, color="red")

    outlier_thresh = 3.0
    for i in range(len(vtx_src)):
        ratio = vtx_tgt[i] / vtx_src[i] if vtx_src[i] > 0.05 else 1.0
        if vtx_src[i] > 1.5 or ratio > 1.2 or ratio < 0.15:
            block_idx = i // 2
            ax.annotate(f"L{block_idx}", (vtx_src[i], vtx_tgt[i]),
                        fontsize=7, alpha=0.7, xytext=(4, 4),
                        textcoords="offset points")

    ax.set_xlabel(r"Source $\mathbb{E}_p |v^\top x^{\mathrm{src}}|$", fontsize=13)
    ax.set_ylabel(r"Target $\mathbb{E}_p |v^\top x^{\mathrm{tgt}}|$", fontsize=13)
    ax.set_title("Gate Magnitude: Source vs Target", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, which="both")

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig3_gate_scatter.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

if __name__ == "__main__":
    draw()
