"""
Gate Mediation Experiment — causal test of the gate-mismatch hypothesis.

Holds u, sigma, location, active layers, and perturbation norm fixed.
Varies only the gate g(x) = v^T x that scales the steering:

  1. Natural target gate      — g(x) = v^T x_tgt        (what weight transfer does)
  2. Source gate replayed     — g(x) = E[v^T x_src]     (constant from source stats)
  3. Magnitude-corrected gate — g(x) = (v^T x_tgt) * c   where c = |E[src]|/|E[tgt]|
  4. Constant RMS gate        — g(x) = rms(v^T x_src)   (constant, source-matched norm)
  5. Shuffled gate            — g(x) drawn from permuted examples
  6. Negated gate             — g(x) = -(v^T x_tgt)

If source-gate-replayed or magnitude-corrected recovers performance near mean-diff
steering, and shuffled/negated do not, the causal evidence for gate mismatch is
substantially strengthened.

Companion to paper_eval_suite.py — imports shared functions from that module.
"""

import torch, json, gc, os, random
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from shared_eval import (
    load_math500_split, evaluate, mcnemar_test, bootstrap_ci,
    wilson_ci, make_prompt, OUTPUT_DIR
)
from paper_eval_suite import (
    collect_proj_inputs, MODELS, TARGET_SUFFIXES,
    get_svd_vectors, get_mean_diff_vectors, apply_meandiff_steering,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Load projection-level SVD (u, sigma, v per module) ────────────────────────
def load_raw_proj_svd():
    base_path = snapshot_download(MODELS["math_base"])
    rlvr_path = snapshot_download(MODELS["rlvr"])
    base_idx, rlvr_idx = {}, {}
    for f in sorted(Path(base_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys(): base_idx[k] = str(f)
    for f in sorted(Path(rlvr_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys(): rlvr_idx[k] = str(f)

    proj_svd = {}
    for pn in sorted(base_idx):
        if not any(pn.endswith(s) for s in TARGET_SUFFIXES): continue
        if pn not in rlvr_idx: continue
        with safe_open(base_idx[pn], framework="pt", device="cpu") as sf:
            wb = sf.get_tensor(pn).float()
        with safe_open(rlvr_idx[pn], framework="pt", device="cpu") as sf:
            wr = sf.get_tensor(pn).float()
        if wb.dim() != 2: continue
        dW = wr - wb
        if dW.norm().item() < 1e-8: continue
        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
        proj_svd[pn] = {
            "u": U[:, 0].clone(), "sigma": S[0].item(),
            "v": Vt[0].clone(), "layer_idx": int(pn.split(".")[2]),
        }
        del wb, wr, dW, U, S, Vt
    gc.collect()
    return proj_svd


# ── Collect per-projection mean gate from source model ────────────────────────
def collect_source_gate_stats(proj_svd, calib_problems):
    """Returns {param_name: {"mean_gate": float, "rms_gate": float}}"""
    print("[GATE_MED] Collecting source gate statistics...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["math_base"], torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODELS["math_base"])
    tok.pad_token = tok.eos_token
    model.eval()

    target_mods = {pn[:-len(".weight")]: pn for pn in proj_svd}
    stores = {pn: [] for pn in proj_svd}
    hooks = []

    def make_hook(pn):
        v = proj_svd[pn]["v"].float()
        def fn(mod, inp, out):
            x = inp[0].detach().float()
            gates = (x[0] @ v).tolist()
            stores[pn].extend(gates)
        return fn

    for mod_name, pn in target_mods.items():
        mod = dict(model.named_modules()).get(mod_name)
        if mod: hooks.append(mod.register_forward_hook(make_hook(pn)))

    for prob in calib_problems:
        text = make_prompt(tok, prob["problem"])
        inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad(): model(**inp)

    for h in hooks: h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()

    stats = {}
    for pn, gates in stores.items():
        g = np.array(gates)
        stats[pn] = {"mean_gate": float(g.mean()), "rms_gate": float(np.sqrt((g**2).mean()))}
    print(f"[GATE_MED] Gate stats for {len(stats)} projections.", flush=True)
    return stats


# ── Gated steering hook factory ───────────────────────────────────────────────
def apply_gated_projection_steering(model, tokenizer, problems, proj_svd,
                                    gate_mode, src_gate_stats,
                                    top_k_layers, alpha, label):
    """
    Apply sigma * gate(x) * u at each target projection.
    gate_mode: 'natural'   — v^T x (what weight transfer does)
               'src_const' — constant = mean(v^T x_src)
               'src_rms'   — constant = rms(v^T x_src)
               'corrected' — (v^T x_tgt) scaled to match src rms
               'negated'   — -(v^T x_tgt)
               'shuffled'  — shuffle gate values across token positions
    """
    hooks = []
    active_layers = set(
        sorted(proj_svd, key=lambda pn: proj_svd[pn]["sigma"], reverse=True)[:top_k_layers * 2]
    )
    # Limit to top_k_layers (each layer has 2 projections → top_k_layers*2 proj)

    for pn, data in proj_svd.items():
        if pn not in active_layers and top_k_layers is not None:
            continue
        mod_name = pn[:-len(".weight")]
        mod = dict(model.named_modules()).get(mod_name)
        if mod is None: continue

        u = data["u"].to(model.device, dtype=model.dtype)
        v = data["v"].to(model.device, dtype=model.dtype)
        sigma = data["sigma"]
        sg = src_gate_stats.get(pn, {"mean_gate": 1.0, "rms_gate": 1.0})
        src_mean = sg["mean_gate"]
        src_rms  = sg["rms_gate"]

        def make_hook(u_, v_, sig_, s_mean, s_rms, mode):
            def fn(mod, inp, out):
                x = inp[0]                            # (B, seq, d_in)
                nat_gate = (x @ v_.unsqueeze(-1)).squeeze(-1)  # (B, seq)
                if mode == "natural":
                    g = nat_gate
                elif mode == "src_const":
                    g = torch.full_like(nat_gate, s_mean)
                elif mode == "src_rms":
                    g = torch.full_like(nat_gate, s_rms)
                elif mode == "corrected":
                    tgt_rms = nat_gate.abs().mean().item() + 1e-8
                    g = nat_gate * (s_rms / tgt_rms)
                elif mode == "negated":
                    g = -nat_gate
                elif mode == "shuffled":
                    flat = nat_gate.reshape(-1)
                    idx = torch.randperm(flat.shape[0], device=flat.device)
                    g = flat[idx].reshape(nat_gate.shape)
                else:
                    g = nat_gate
                steer = (alpha * sig_ * g.unsqueeze(-1) * u_.unsqueeze(0).unsqueeze(0))
                return out + steer.to(out.dtype)
            return fn

        hooks.append(mod.register_forward_hook(
            make_hook(u, v, sigma, src_mean, src_rms, gate_mode)))

    result = evaluate(model, tokenizer, problems, label)
    for h in hooks: h.remove()
    return result


def main():
    print("="*60)
    print("GATE MEDIATION EXPERIMENT")
    print("="*60, flush=True)

    calib = load_math500_split("calib")
    val   = load_math500_split("val")
    test  = load_math500_split("test")

    proj_svd    = load_raw_proj_svd()
    src_stats   = collect_source_gate_stats(proj_svd, calib)
    mean_diff   = get_mean_diff_vectors(calib)
    svd_vecs    = get_svd_vectors(calib, orientation="source_gate")

    print(f"\n[GATE_MED] Loading target model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODELS["instruct"])
    tok.pad_token = tok.eos_token

    # Select alpha on VAL using natural-gate (weight-transfer-equivalent)
    best_alpha = 0.5
    best_val = 0.0
    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0]:
        r = apply_gated_projection_steering(
            model, tok, val, proj_svd, "natural", src_stats, 15, alpha,
            f"val_gate_natural_a{alpha}")
        if r["accuracy"] > best_val:
            best_val = r["accuracy"]; best_alpha = alpha
    print(f"[GATE_MED] Best alpha on VAL: {best_alpha}", flush=True)

    results = {}
    bl = evaluate(model, tok, test, "gate_med_baseline")
    results["baseline"] = bl
    base_items = [it["correct"] for it in bl["items"]]
    print(f"  Baseline: {bl['accuracy']:.1f}%")

    # Run all 6 gate variants
    gate_modes = [
        ("natural",   "Weight-transfer-equiv (natural gate v^T x_tgt)"),
        ("src_const", "Source mean gate replayed (constant E[v^T x_src])"),
        ("src_rms",   "Source RMS gate replayed (constant RMS)"),
        ("corrected", "Magnitude-corrected gate"),
        ("negated",   "Negated gate -(v^T x_tgt)"),
        ("shuffled",  "Shuffled gate (random token order)"),
    ]
    for mode, desc in gate_modes:
        r = apply_gated_projection_steering(
            model, tok, test, proj_svd, mode, src_stats, 15, best_alpha,
            f"gate_med_{mode}")
        results[mode] = r
        mi = [it["correct"] for it in r["items"]]
        mn = mcnemar_test(base_items, mi)
        bc = bootstrap_ci(base_items, mi)
        lo, hi = wilson_ci(r["correct"], r["total"])
        print(f"  {desc[:50]:<50} {r['accuracy']:.1f}% [{lo},{hi}]  "
              f"Δ={bc['mean_diff_pp']:+.1f}pp  McNemar p={mn['p_value']:.3f}")

    # Also run mean-diff for reference
    best_md = 0.05
    r = apply_meandiff_steering(model, tok, test, mean_diff, best_md,
                                "gate_med_meandiff_ref")
    results["meandiff_ref"] = r
    print(f"  Mean-diff reference: {r['accuracy']:.1f}%")

    slim = {k: {kk: vv for kk, vv in v.items() if kk != "items"}
            for k, v in results.items()}
    with open(OUTPUT_DIR / "gate_mediation_results.json", "w") as f:
        json.dump(slim, f, indent=2)
    print("[GATE_MED] Results saved.", flush=True)
    print("DONE.")


if __name__ == "__main__":
    main()
