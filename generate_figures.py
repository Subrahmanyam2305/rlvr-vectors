"""
Generate all figures for the paper.

Figures produced:
  fig1_spectral_distribution.png  — rank-1 fraction per layer, by layer type
  fig2_alpha_sweep.png            — accuracy vs alpha for SVD and mean-diff
  fig3_main_results.png           — bar chart: all methods, TEST set, with CIs
  fig4_per_layer_sigma.png        — top-28 layers by sigma value

Reads from:
  outputs/spectral_data.json       (existing)
  outputs/paper_eval_stats.json    (produced by paper_eval_suite.py)
  outputs/paper_eval_results.json  (produced by paper_eval_suite.py)
  outputs/overnight_results.json   (existing, n=50 legacy for alpha curves)
"""

import json
import math
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Resolve paths relative to this script's location
SCRIPT_DIR  = Path(__file__).parent
OUTPUT_DIR  = SCRIPT_DIR / "outputs"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def wilson_ci(correct, total):
    if total == 0:
        return (0, 0)
    z = 1.959964
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return ((center - half) * 100, (center + half) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Spectral concentration by layer type
# ─────────────────────────────────────────────────────────────────────────────

def fig1_spectral_distribution():
    data = load_json(OUTPUT_DIR / "spectral_data.json")
    layers = [d for d in data if d["layer_idx"] >= 0]

    TYPE_MAP = {
        "Q": "Q-proj", "K": "K-proj", "V": "V-proj", "O": "O-proj",
        "FFN_gate": "FFN gate", "FFN_up": "FFN up", "FFN_down": "FFN down",
    }
    COLORS = {
        "Q-proj": "#4E79A7", "K-proj": "#F28E2B", "V-proj": "#E15759",
        "O-proj": "#76B7B2", "FFN gate": "#59A14F", "FFN up": "#EDC948",
        "FFN down": "#B07AA1",
    }

    by_type: dict[str, list] = {}
    for d in layers:
        t = TYPE_MAP.get(d["layer_type"], d["layer_type"])
        by_type.setdefault(t, []).append(d["rank1_frac"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: boxplot by type
    ax = axes[0]
    order = [t for t in TYPE_MAP.values() if t in by_type]
    bplot = ax.boxplot(
        [by_type[t] for t in order],
        patch_artist=True, medianprops=dict(color="black", lw=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5)
    )
    for patch, t in zip(bplot["boxes"], order):
        patch.set_facecolor(COLORS.get(t, "#aaa"))
        patch.set_alpha(0.8)
    ax.axhline(0.001, color="red", lw=1.2, ls="--", label="Random matrix baseline (~0.001)")
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rank-1 fraction  ρ = σ₁² / ‖ΔW‖²_F")
    ax.set_title("Rank-1 Spectral Concentration by Layer Type")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Right: histogram of all ρ values
    ax2 = axes[1]
    all_rho = [d["rank1_frac"] for d in layers]
    ax2.hist(all_rho, bins=30, color="#4E79A7", alpha=0.8, edgecolor="white")
    ax2.axvline(np.mean(all_rho), color="red", lw=1.5, ls="--",
                label=f"Mean ρ = {np.mean(all_rho):.3f}")
    ax2.axvline(0.001, color="orange", lw=1.5, ls=":",
                label="Random baseline ≈ 0.001")
    ax2.set_xlabel("Rank-1 fraction ρ")
    ax2.set_ylabel("Number of layers")
    ax2.set_title(f"Distribution of ρ across {len(layers)} parameter matrices")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig1_spectral_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Alpha sweep curves (using overnight n=50 results as proxy)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_alpha_sweep():
    overnight = load_json(OUTPUT_DIR / "overnight_results.json")
    baseline  = overnight["baseline"]["accuracy"]

    svd_alphas   = [0.5, 1.0, 2.0, 3.0, 5.0]
    svd_accs     = [overnight.get(f"svd_residual_a{a}", {}).get("accuracy", None)
                    for a in svd_alphas]

    md_alphas    = [0.02, 0.05, 0.1, 0.2]
    md_accs      = [overnight.get(f"meandiff_residual_a{a}", {}).get("accuracy", None)
                    for a in md_alphas]

    # Also add top-K points at alpha=2.0
    topk_ks   = [5, 10, 15]
    topk_accs = [overnight.get(f"svd_top{k}_residual_a2.0", {}).get("accuracy", None)
                 for k in topk_ks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: accuracy vs alpha
    ax = axes[0]
    svd_valid = [(a, acc) for a, acc in zip(svd_alphas, svd_accs) if acc is not None]
    md_valid  = [(a, acc) for a, acc in zip(md_alphas, md_accs) if acc is not None]
    if svd_valid:
        ax.plot(*zip(*svd_valid), "o-", color="#4E79A7", lw=2, ms=7, label="SVD (residual, full)")
    if md_valid:
        ax.plot(*zip(*md_valid), "s--", color="#E15759", lw=2, ms=7, label="Mean-diff (residual)")
    ax.axhline(baseline, color="gray", ls=":", lw=1.5, label=f"Baseline ({baseline:.0f}%)")
    ax.set_xlabel("Steering strength  α")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Steering Strength  (n=50, preliminary — calib/eval overlap)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 75)
    # Note: SVD x-axis and MD x-axis are very different scales
    ax.set_xscale("log")

    # Right: top-K ablation
    ax2 = axes[1]
    full_acc = overnight.get("svd_residual_a0.5", {}).get("accuracy", None)
    if topk_accs and any(a is not None for a in topk_accs):
        valid_topk = [(k, a) for k, a in zip(topk_ks, topk_accs) if a is not None]
        ax2.bar([f"Top-{k}" for k, _ in valid_topk],
                [a for _, a in valid_topk],
                color="#76B7B2", alpha=0.85, edgecolor="white", label="SVD top-K  (α=2.0)")
    if full_acc is not None:
        ax2.axhline(full_acc, color="#4E79A7", ls="--", lw=1.5,
                    label=f"SVD full α=0.5 ({full_acc:.0f}%)")
    ax2.axhline(baseline, color="gray", ls=":", lw=1.5, label=f"Baseline ({baseline:.0f}%)")
    best_md = max(a for a in md_accs if a is not None)
    ax2.axhline(best_md, color="#E15759", ls="-.", lw=1.5,
                label=f"Mean-diff best ({best_md:.0f}%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Top-K Layer Sparsity Ablation  (n=50, preliminary — calib/eval overlap)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 75)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig2_alpha_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Main results bar chart (TEST set with CIs)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_main_results():
    # Try paper_eval_stats (n=400 clean results), fall back to overnight (n=50)
    stats_path = OUTPUT_DIR / "paper_eval_stats.json"
    if stats_path.exists():
        stats = load_json(stats_path)
        results_path = OUTPUT_DIR / "paper_eval_results.json"
        results = load_json(results_path)
        baseline_acc = stats["baseline_test_acc"]
        best_svd_a   = stats["best_svd_alpha"]
        best_k       = stats["best_k"]
        best_md_a    = stats["best_md_alpha"]

        def get_r(key, fallback_acc=None):
            r = stats["summary"].get(key)
            if r:
                return r["accuracy"], r["correct"], r["total"]
            return fallback_acc, None, None

        methods = [
            ("Baseline",                   baseline_acc, None, None, "#aaaaaa"),
            ("SVD full",                   *get_r("SVD_full"),            "#4E79A7"),
            (f"SVD top-{best_k} (sparse)", *get_r("SVD_topK"),           "#76B7B2"),
            ("Mean-diff",                  *get_r("MeanDiff"),            "#E15759"),
            ("Random control",             *get_r("Random_control"),      "#BCBD22"),
            ("Sign-flip control",          *get_r("SignFlip_control"),    "#17BECF"),
            ("Wrong-layer control",        *get_r("WrongLayer_control"),  "#9467BD"),
        ]
        title = "Main Results  (TEST set, n=400, problems 0–399)"
        note = "Alpha and K selected on disjoint VAL set (400–449). Error bars = Wilson 95% CI."
    else:
        # Fall back to overnight n=50 results
        overnight = load_json(OUTPUT_DIR / "overnight_results.json")
        baseline_acc = overnight["baseline"]["accuracy"]

        def acc_ci(key):
            r = overnight.get(key)
            if r:
                return r["accuracy"], r["correct"], r["total"]
            return None, None, None

        methods = [
            ("Baseline",               baseline_acc, 23, 50, "#aaaaaa"),
            ("SVD full  α=0.5",        *acc_ci("svd_residual_a0.5"),        "#4E79A7"),
            ("SVD top-15  α=2.0",      *acc_ci("svd_top15_residual_a2.0"),  "#76B7B2"),
            ("Mean-diff  α=0.05",      *acc_ci("meandiff_residual_a0.05"),  "#E15759"),
        ]
        title = "Main Results  (n=50 VAL — preliminary)"
        note  = "⚠ Preliminary: alpha selected on same 50 examples. See paper_eval_suite.py for clean results."

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [m[0] for m in methods]
    accs  = [m[1] for m in methods]
    colors = [m[4] for m in methods]

    x = np.arange(len(names))
    bars = ax.bar(x, accs, color=colors, alpha=0.85, edgecolor="white", width=0.6)

    # Error bars (Wilson CI)
    yerr_low  = []
    yerr_high = []
    for _, acc, corr, tot, _ in methods:
        if corr is not None and tot is not None and acc is not None:
            lo, hi = wilson_ci(corr, tot)
            yerr_low.append(acc - lo)
            yerr_high.append(hi - acc)
        else:
            yerr_low.append(0)
            yerr_high.append(0)
    ax.errorbar(x, accs, yerr=[yerr_low, yerr_high],
                fmt="none", color="black", capsize=5, lw=1.5)

    # Delta labels on bars
    for i, (name, acc, *_) in enumerate(methods):
        if acc is not None:
            delta = acc - baseline_acc
            delta_str = f"+{delta:.1f}pp" if delta >= 0 else f"{delta:.1f}pp"
            label = f"{acc:.1f}%" if i == 0 else f"{acc:.1f}%\n({delta_str})"
            ax.text(x[i], acc + max(yerr_high[i], 0) + 1.5, label,
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(baseline_acc, color="gray", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)  on MATH500")
    ax.set_title(title)
    ax.set_ylim(0, min(100, max(accs) + 18))
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, style="italic", color="#555")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig3_main_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Per-layer sigma (importance weights)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_per_layer_sigma():
    data = load_json(OUTPUT_DIR / "spectral_data.json")

    # Collect o_proj and down_proj sigma per layer (same as SVD steering combines)
    layer_sigmas: dict[int, float] = {}
    for d in data:
        if d["layer_idx"] < 0:
            continue
        name = d["name"]
        if "o_proj" in name or "down_proj" in name:
            li = d["layer_idx"]
            layer_sigmas[li] = layer_sigmas.get(li, 0.0) + d["top_singular_value"]

    if not layer_sigmas:
        print("Skipping fig4: no o_proj/down_proj entries found")
        return

    layers = sorted(layer_sigmas)
    sigmas = [layer_sigmas[l] for l in layers]
    sigma_max = max(sigmas)
    weights = [s / sigma_max for s in sigmas]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#E15759" if w >= sorted(weights)[-15] else "#4E79A7"
              for w in weights]
    ax.bar(layers, weights, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("Normalised importance weight  σ_l / σ_max")
    ax.set_title("Per-Layer SVD Importance Weights  (σ from o_proj + down_proj ΔW)")

    top15_patch = mpatches.Patch(color="#E15759", alpha=0.85, label="Top-15 layers (used in SVD top-K)")
    rest_patch  = mpatches.Patch(color="#4E79A7", alpha=0.85, label="Remaining layers")
    ax.legend(handles=[top15_patch, rest_patch], fontsize=9)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig4_per_layer_sigma.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...", flush=True)
    fig1_spectral_distribution()
    fig2_alpha_sweep()
    fig3_main_results()
    fig4_per_layer_sigma()
    print("All figures saved to", OUTPUT_DIR)
