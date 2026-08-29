"""
Paper Evaluation Suite — addressing all reviewer concerns:

  1. Disjoint splits: CALIB=450-499, VAL=400-449, TEST=0-399
  2. Two-phase protocol: select alpha on VAL, report on TEST
  3. SVD sign disambiguation via cosine alignment with mean-diff direction
  4. Item-level predictions saved for paired bootstrap tests
  5. Random-steering control with matched per-layer norms
  6. Sign-flip control for SVD vectors
  7. Unified evaluator (shared_eval.py)

Runtime estimate: ~14 hours total on a single GPU (1.5B model, n=400 test).
Run inside tmux so it survives disconnections.
"""

import torch
import numpy as np
import json
import gc
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open

from shared_eval import (
    load_math500_split, evaluate, bootstrap_ci, exact_ci_wilson,
    print_result_row, OUTPUT_DIR
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODELS = {
    "math_base": "Qwen/Qwen2.5-Math-1.5B",
    "rlvr":      "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    "instruct":  "Qwen/Qwen2.5-1.5B-Instruct",
}


# ─────────────────────────────────────────────────────────────────────────────
# SVD vector extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_svd_vectors(orient_with_mean_diff: dict | None = None) -> dict:
    """
    Extract per-layer combined SVD steering vectors from RLVR weight deltas.

    orient_with_mean_diff: if provided (layer_idx -> tensor), each u vector is
      flipped if its cosine with the mean-diff vector is negative.  This
      resolves the SVD sign ambiguity in a principled, data-driven way.
    """
    print("[SVD] Computing per-layer SVD steering vectors...", flush=True)
    base_path  = snapshot_download(MODELS["math_base"])
    rlvr_path  = snapshot_download(MODELS["rlvr"])

    base_index, rlvr_index = {}, {}
    for f in sorted(Path(base_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = str(f)
    for f in sorted(Path(rlvr_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                rlvr_index[key] = str(f)

    target_suffixes = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]
    layer_data: dict[int, dict] = {}

    for param_name in sorted(base_index):
        if not any(param_name.endswith(s) for s in target_suffixes):
            continue
        if param_name not in rlvr_index:
            continue

        with safe_open(base_index[param_name], framework="pt", device="cpu") as sf:
            w_base = sf.get_tensor(param_name).float()
        with safe_open(rlvr_index[param_name], framework="pt", device="cpu") as sf:
            w_rlvr = sf.get_tensor(param_name).float()

        if w_base.dim() != 2:
            continue
        dW = w_rlvr - w_base
        if dW.norm().item() < 1e-8:
            continue

        U, S, _ = torch.linalg.svd(dW, full_matrices=False)
        u = U[:, 0].clone()
        sigma = S[0].item()

        layer_idx = int(param_name.split(".")[2])

        if layer_idx not in layer_data:
            layer_data[layer_idx] = {"vecs": [], "sigs": []}
        layer_data[layer_idx]["vecs"].append(u)
        layer_data[layer_idx]["sigs"].append(sigma)

        del w_base, w_rlvr, dW, U, S

    gc.collect()

    combined = {}
    flipped = 0
    total = 0
    for layer_idx, data in layer_data.items():
        vecs, sigs = data["vecs"], data["sigs"]
        total_sigma = sum(sigs)
        u_comb = sum(s * v for s, v in zip(sigs, vecs)) / total_sigma
        u_comb = u_comb / (u_comb.norm() + 1e-8)

        # Sign disambiguation: align with mean-diff direction if available
        if orient_with_mean_diff and layer_idx in orient_with_mean_diff:
            ref = orient_with_mean_diff[layer_idx]
            ref = ref / (ref.norm() + 1e-8)
            if torch.dot(u_comb, ref.to(u_comb)) < 0:
                u_comb = -u_comb
                flipped += 1
        total += 1

        combined[layer_idx] = {"u": u_comb, "sigma": total_sigma}

    print(f"[SVD] {len(combined)} layers. Sign flips applied: {flipped}/{total}", flush=True)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Mean-difference vectors (requires source model inference)
# ─────────────────────────────────────────────────────────────────────────────

def collect_hidden_states(model, tokenizer, problems) -> dict:
    """Collect mean hidden state (over tokens and problems) after each transformer layer."""
    from shared_eval import make_prompt
    model.eval()
    stores: dict[int, list] = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            stores.setdefault(layer_idx, []).append(
                h.detach().float().mean(dim=(0, 1)).cpu()
            )
        return hook_fn

    for name, module in model.named_modules():
        if hasattr(module, "self_attn") and hasattr(module, "mlp"):
            for part in name.split("."):
                try:
                    layer_idx = int(part)
                    hooks.append(module.register_forward_hook(make_hook(layer_idx)))
                    break
                except ValueError:
                    continue

    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    return {idx: torch.stack(vecs).mean(0) for idx, vecs in stores.items()}


def get_mean_diff_vectors(calib_problems) -> dict:
    print("[MEANDIFF] Collecting source base hidden states...", flush=True)
    src = AutoModelForCausalLM.from_pretrained(
        MODELS["math_base"], torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODELS["math_base"])
    tok.pad_token = tok.eos_token
    src_states = collect_hidden_states(src, tok, calib_problems)
    del src; gc.collect(); torch.cuda.empty_cache()

    print("[MEANDIFF] Collecting RLVR hidden states...", flush=True)
    rlvr = AutoModelForCausalLM.from_pretrained(
        MODELS["rlvr"], torch_dtype=torch.float16, device_map="auto")
    tok2 = AutoTokenizer.from_pretrained(MODELS["rlvr"])
    tok2.pad_token = tok2.eos_token
    rlvr_states = collect_hidden_states(rlvr, tok2, calib_problems)
    del rlvr; gc.collect(); torch.cuda.empty_cache()

    diff = {idx: rlvr_states[idx] - src_states[idx]
            for idx in src_states if idx in rlvr_states}
    print(f"[MEANDIFF] Got vectors for {len(diff)} layers", flush=True)
    return diff


# ─────────────────────────────────────────────────────────────────────────────
# Steering application
# ─────────────────────────────────────────────────────────────────────────────

def apply_residual_steering(model, tokenizer, problems, layer_vectors,
                             alpha, label, top_k=None,
                             sign_flip=False, randomize=False) -> dict:
    """
    Apply steering vectors at the residual stream (after each transformer block).

    sign_flip  : if True, negate all u vectors (sign-flip control)
    randomize  : if True, replace each u with a random unit vector of the same
                 norm (matched-norm random control)
    """
    hooks = []
    sigma_max = max(d["sigma"] for d in layer_vectors.values())

    if top_k:
        ranked = sorted(layer_vectors, key=lambda i: layer_vectors[i]["sigma"], reverse=True)
        active = set(ranked[:top_k])
    else:
        active = set(layer_vectors)

    for name, module in model.named_modules():
        if not (hasattr(module, "self_attn") and hasattr(module, "mlp")):
            continue
        layer_idx = None
        for part in name.split("."):
            try:
                layer_idx = int(part)
                break
            except ValueError:
                continue
        if layer_idx is None or layer_idx not in active:
            continue

        data = layer_vectors[layer_idx]
        u = data["u"].clone()

        if randomize:
            u = torch.randn_like(u)
            u = u / (u.norm() + 1e-8)
        elif sign_flip:
            u = -u

        u = u.to(model.device, dtype=model.dtype)
        weight = data["sigma"] / sigma_max

        def make_hook(u_vec, w):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                steer = (alpha * w * u_vec).to(h.dtype)
                h = h + steer.unsqueeze(0).unsqueeze(0)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(u, weight)))

    result = evaluate(model, tokenizer, problems, label)

    for h in hooks:
        h.remove()
    return result


def apply_meandiff_steering(model, tokenizer, problems, diff_vectors,
                             alpha, label) -> dict:
    hooks = []
    for name, module in model.named_modules():
        if not (hasattr(module, "self_attn") and hasattr(module, "mlp")):
            continue
        layer_idx = None
        for part in name.split("."):
            try:
                layer_idx = int(part)
                break
            except ValueError:
                continue
        if layer_idx is None or layer_idx not in diff_vectors:
            continue

        d = diff_vectors[layer_idx].to(model.device, dtype=model.dtype)

        def make_hook(dv):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h + (alpha * dv.to(h.dtype)).unsqueeze(0).unsqueeze(0)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(d)))

    result = evaluate(model, tokenizer, problems, label)
    for h in hooks:
        h.remove()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main: two-phase protocol
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("PAPER EVALUATION SUITE — clean splits, sign-fix, controls")
    print("=" * 72, flush=True)
    print()
    print("Splits:")
    print("  CALIB : problems 450–499  (steering vector extraction)")
    print("  VAL   : problems 400–449  (alpha / K selection)")
    print("  TEST  : problems   0–399  (final reported numbers)")
    print(flush=True)

    calib_problems = load_math500_split("calib")   # 50 problems
    val_problems   = load_math500_split("val")     # 50 problems
    test_problems  = load_math500_split("test")    # 400 problems

    # ── Step 1: Compute steering vectors on CALIB ────────────────────────────
    print("\n[STEP 1] Computing steering vectors on CALIB split...", flush=True)
    mean_diff = get_mean_diff_vectors(calib_problems)
    # SVD with sign oriented to align with mean-diff (resolves sign ambiguity)
    svd_vecs  = get_svd_vectors(orient_with_mean_diff=mean_diff)
    print("[STEP 1] Done.", flush=True)

    # ── Step 2: Load target model ────────────────────────────────────────────
    print("\n[STEP 2] Loading target model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["instruct"])
    tokenizer.pad_token = tokenizer.eos_token
    print("[STEP 2] Done.", flush=True)

    # ── Step 3: Baseline on VAL and TEST ─────────────────────────────────────
    print("\n[STEP 3] Baselines...", flush=True)
    val_baseline  = evaluate(model, tokenizer, val_problems,  "baseline_val")
    test_baseline = evaluate(model, tokenizer, test_problems, "baseline_test")
    bl_val  = val_baseline["accuracy"]
    bl_test = test_baseline["accuracy"]
    print(f"  VAL  baseline: {bl_val:.1f}%")
    print(f"  TEST baseline: {bl_test:.1f}%", flush=True)

    results = {"baseline_val": val_baseline, "baseline_test": test_baseline}
    summary = {}

    # ── Step 4: VAL alpha sweeps → select best config per method ────────────
    print("\n[STEP 4] Hyperparameter selection on VAL...", flush=True)

    # SVD full
    svd_val_scores = {}
    for alpha in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        lbl = f"val_svd_full_a{alpha}"
        r = apply_residual_steering(model, tokenizer, val_problems,
                                    svd_vecs, alpha, lbl)
        svd_val_scores[alpha] = r["accuracy"]
        results[lbl] = r
        print(f"  SVD full α={alpha}: {r['accuracy']:.1f}%", flush=True)
    best_svd_alpha = max(svd_val_scores, key=svd_val_scores.get)
    print(f"  → Best SVD full α = {best_svd_alpha}", flush=True)

    # SVD top-K (fixed alpha = best_svd_alpha)
    svd_topk_val = {}
    for k in [5, 10, 15, 20]:
        lbl = f"val_svd_top{k}_a{best_svd_alpha}"
        r = apply_residual_steering(model, tokenizer, val_problems,
                                    svd_vecs, best_svd_alpha, lbl, top_k=k)
        svd_topk_val[k] = r["accuracy"]
        results[lbl] = r
        print(f"  SVD top-{k}: {r['accuracy']:.1f}%", flush=True)
    best_k = max(svd_topk_val, key=svd_topk_val.get)
    print(f"  → Best K = {best_k}", flush=True)

    # Mean-diff
    md_val_scores = {}
    for alpha in [0.01, 0.02, 0.05, 0.07, 0.1]:
        lbl = f"val_meandiff_a{alpha}"
        r = apply_meandiff_steering(model, tokenizer, val_problems,
                                    mean_diff, alpha, lbl)
        md_val_scores[alpha] = r["accuracy"]
        results[lbl] = r
        print(f"  Mean-diff α={alpha}: {r['accuracy']:.1f}%", flush=True)
    best_md_alpha = max(md_val_scores, key=md_val_scores.get)
    print(f"  → Best mean-diff α = {best_md_alpha}", flush=True)

    # ── Step 5: Final TEST evaluation with selected configs ───────────────────
    print("\n[STEP 5] Final evaluation on TEST (n=400)...", flush=True)

    # SVD full (best alpha)
    lbl = f"test_svd_full_a{best_svd_alpha}"
    r = apply_residual_steering(model, tokenizer, test_problems,
                                svd_vecs, best_svd_alpha, lbl)
    results[lbl] = r
    summary["SVD_full"] = r
    print_result_row("SVD full (best α)", r, bl_test)

    # SVD top-K (best k + best alpha)
    lbl = f"test_svd_top{best_k}_a{best_svd_alpha}"
    r = apply_residual_steering(model, tokenizer, test_problems,
                                svd_vecs, best_svd_alpha, lbl, top_k=best_k)
    results[lbl] = r
    summary["SVD_topK"] = r
    print_result_row(f"SVD top-{best_k} (best α)", r, bl_test)

    # Mean-diff (best alpha)
    lbl = f"test_meandiff_a{best_md_alpha}"
    r = apply_meandiff_steering(model, tokenizer, test_problems,
                                mean_diff, best_md_alpha, lbl)
    results[lbl] = r
    summary["MeanDiff"] = r
    print_result_row("Mean-diff (best α)", r, bl_test)

    # ── Step 6: Controls ──────────────────────────────────────────────────────
    print("\n[STEP 6] Control experiments on TEST...", flush=True)

    # Random steering with matched per-layer norms
    lbl = f"test_svd_random_a{best_svd_alpha}"
    r = apply_residual_steering(model, tokenizer, test_problems,
                                svd_vecs, best_svd_alpha, lbl, randomize=True)
    results[lbl] = r
    summary["Random_control"] = r
    print_result_row("Random steering (matched norm)", r, bl_test)

    # Sign-flip control
    lbl = f"test_svd_signflip_a{best_svd_alpha}"
    r = apply_residual_steering(model, tokenizer, test_problems,
                                svd_vecs, best_svd_alpha, lbl, sign_flip=True)
    results[lbl] = r
    summary["SignFlip_control"] = r
    print_result_row("Sign-flipped SVD", r, bl_test)

    # Top-K with scrambled layer assignment (wrong-layer control)
    ranked_layers = sorted(svd_vecs, key=lambda i: svd_vecs[i]["sigma"], reverse=True)
    shuffled_vecs = dict(svd_vecs)
    top_k_layers = ranked_layers[:best_k]
    shuffled_layers = top_k_layers.copy()
    import random as _rand
    _rand.seed(42)
    _rand.shuffle(shuffled_layers)
    shuffled_map = {orig: shuffled_vecs[shuf]
                    for orig, shuf in zip(top_k_layers, shuffled_layers)}
    shuffled_vecs_reindexed = dict(svd_vecs)
    shuffled_vecs_reindexed.update(shuffled_map)
    lbl = f"test_svd_wronglayer_a{best_svd_alpha}"
    r = apply_residual_steering(model, tokenizer, test_problems,
                                shuffled_vecs_reindexed, best_svd_alpha, lbl,
                                top_k=best_k)
    results[lbl] = r
    summary["WrongLayer_control"] = r
    print_result_row("SVD wrong-layer (shuffled)", r, bl_test)

    # ── Step 7: Paired bootstrap CIs ─────────────────────────────────────────
    print("\n[STEP 7] Paired bootstrap CIs vs baseline...", flush=True)
    base_items = [it["correct"] for it in test_baseline["items"]]

    stats = {}
    for key, res in summary.items():
        method_items = [it["correct"] for it in res["items"]]
        ci = bootstrap_ci(base_items, method_items)
        stats[key] = ci
        print(f"  {key:<30} Δ={ci['mean_diff_pp']:+.2f}pp "
              f"95%CI [{ci['ci_low_pp']:.1f}, {ci['ci_high_pp']:.1f}]  "
              f"p={ci['p_value']:.3f}", flush=True)

    # ── Save everything ───────────────────────────────────────────────────────
    print("\n[SAVE] Writing results...", flush=True)
    with open(OUTPUT_DIR / "paper_eval_results.json", "w") as f:
        # strip item-level for top-level summary
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "items"}
                for k, v in results.items()}
        json.dump(slim, f, indent=2)

    with open(OUTPUT_DIR / "paper_eval_stats.json", "w") as f:
        json.dump({
            "baseline_val_acc": bl_val,
            "baseline_test_acc": bl_test,
            "best_svd_alpha": best_svd_alpha,
            "best_k": best_k,
            "best_md_alpha": best_md_alpha,
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "items"}
                        for k, v in summary.items()},
            "bootstrap_stats": stats,
        }, f, indent=2)

    # ── Print final table ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL RESULTS TABLE  (TEST set, n=400, problems 0–399)")
    print("=" * 72)
    print(f"{'Method':<40} {'Acc':>6}  {'95% CI':>18}  {'Δ (pp)':>8}  {'p':>7}")
    print("-" * 72)
    bl_lo, bl_hi = exact_ci_wilson(test_baseline["correct"], test_baseline["total"])
    print(f"{'Baseline (Qwen2.5-1.5B-Instruct)':<40} "
          f"{bl_test:>5.1f}%  [{bl_lo:.1f}, {bl_hi:.1f}]  {'—':>8}")
    for key, res in summary.items():
        ci = stats[key]
        lo, hi = exact_ci_wilson(res["correct"], res["total"])
        print(f"{key:<40} {res['accuracy']:>5.1f}%  [{lo:.1f}, {hi:.1f}]  "
              f"{ci['mean_diff_pp']:>+7.2f}pp  {ci['p_value']:>6.3f}")
    print("=" * 72)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
