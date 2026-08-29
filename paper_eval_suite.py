"""
Paper Evaluation Suite — all reviewer fixes applied:
  1. SVD sign orientation: orient each u_{l,m} BEFORE combining via sign(E[v^T x])
  2. Wrong-layer control: keep active layers fixed, permute only vectors
  3. 20-seed null controls (random dirs, random layer rankings, random signs)
  4. McNemar primary test + bootstrap CIs
  5. math-verify evaluator (via shared_eval)
  6. Disjoint splits: CALIB=450-499, VAL=400-449, TEST=0-399
"""
import torch, numpy as np, json, gc, os, random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from shared_eval import (
    load_math500_split, evaluate, mcnemar_test, bootstrap_ci,
    wilson_ci, print_result_row, make_prompt, OUTPUT_DIR
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
N_NULL_SEEDS = 20

MODELS = {
    "math_base": "Qwen/Qwen2.5-Math-1.5B",
    "rlvr":      "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    "instruct":  "Qwen/Qwen2.5-1.5B-Instruct",
}


# ── Layer-input hook collector ─────────────────────────────────────────────────
def collect_proj_inputs(model, tokenizer, problems, target_modules):
    """
    Collect mean input activation (averaged over tokens and problems)
    for each named linear projection in target_modules.
    Returns dict: full_module_name -> mean_input_tensor (cpu, float32)
    """
    model.eval()
    stores = {}
    hooks = []

    def make_hook(name):
        def fn(module, inp, out):
            x = inp[0].detach().float()          # (batch, seq, d_in)
            mean_x = x.mean(dim=(0, 1)).cpu()    # (d_in,)
            stores.setdefault(name, []).append(mean_x)
        return fn

    for name, module in model.named_modules():
        if name in target_modules:
            hooks.append(module.register_forward_hook(make_hook(name)))

    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inp = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model(**inp)

    for h in hooks:
        h.remove()

    return {name: torch.stack(vecs).mean(0) for name, vecs in stores.items()}


# ── SVD extraction with per-projection sign orientation ───────────────────────
def get_svd_vectors(calib_problems, orientation="source_gate"):
    """
    Extract per-layer combined SVD steering vectors.

    orientation choices:
      'source_gate'  — sign(E[v^T x]) using source base model inputs (default)
      'weight_only'  — flip u so its max-abs coordinate is positive (no data)
      'none'         — raw PyTorch SVD sign (baseline for sign ablation)
    """
    print(f"[SVD] Computing vectors (orientation={orientation})...", flush=True)
    base_path = snapshot_download(MODELS["math_base"])
    rlvr_path = snapshot_download(MODELS["rlvr"])

    base_index, rlvr_index = {}, {}
    for f in sorted(Path(base_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys(): base_index[k] = str(f)
    for f in sorted(Path(rlvr_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys(): rlvr_index[k] = str(f)

    TARGET_SUFFIXES = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]
    proj_svd = {}  # param_name -> {u, sigma, v, layer_idx}

    for param_name in sorted(base_index):
        if not any(param_name.endswith(s) for s in TARGET_SUFFIXES):
            continue
        if param_name not in rlvr_index:
            continue
        with safe_open(base_index[param_name], framework="pt", device="cpu") as sf:
            w_base = sf.get_tensor(param_name).float()
        with safe_open(rlvr_index[param_name], framework="pt", device="cpu") as sf:
            w_rlvr = sf.get_tensor(param_name).float()
        if w_base.dim() != 2: continue
        dW = w_rlvr - w_base
        if dW.norm().item() < 1e-8: continue

        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
        layer_idx = int(param_name.split(".")[2])
        proj_svd[param_name] = {
            "u": U[:, 0].clone(),
            "sigma": S[0].item(),
            "v": Vt[0].clone(),          # v (right singular vector, shape d_in)
            "layer_idx": layer_idx,
        }
        del w_base, w_rlvr, dW, U, S, Vt
    gc.collect()
    print(f"[SVD] Got SVD for {len(proj_svd)} projections.", flush=True)

    # ── Orientation ──────────────────────────────────────────────────────────
    if orientation == "source_gate":
        # Collect source-model input activations for each target projection
        print("[SVD] Loading source model for gate orientation...", flush=True)
        src_model = AutoModelForCausalLM.from_pretrained(
            MODELS["math_base"], torch_dtype=torch.float16, device_map="auto")
        src_tok = AutoTokenizer.from_pretrained(MODELS["math_base"])
        src_tok.pad_token = src_tok.eos_token

        # Map param_name to module name (strip ".weight")
        target_mod_names = {n[: -len(".weight")] for n in proj_svd}
        mean_inputs = collect_proj_inputs(src_model, src_tok,
                                          calib_problems, target_mod_names)
        del src_model; gc.collect(); torch.cuda.empty_cache()

        flipped = 0
        for param_name, data in proj_svd.items():
            mod_name = param_name[: -len(".weight")]
            if mod_name not in mean_inputs:
                continue
            x_mean = mean_inputs[mod_name]            # (d_in,)
            v = data["v"]                              # (d_in,)
            gate_sign = torch.dot(v.float(), x_mean.float()).sign().item()
            if gate_sign < 0:
                data["u"] = -data["u"]
                data["v"] = -data["v"]
                flipped += 1
        print(f"[SVD] Source-gate orientation: flipped {flipped}/{len(proj_svd)}", flush=True)

    elif orientation == "weight_only":
        flipped = 0
        for data in proj_svd.values():
            u = data["u"]
            if u[u.abs().argmax()].item() < 0:
                data["u"] = -data["u"]
                flipped += 1
        print(f"[SVD] Weight-only orientation: flipped {flipped}/{len(proj_svd)}", flush=True)

    # ── Combine per-layer ────────────────────────────────────────────────────
    layer_data = {}
    for param_name, data in proj_svd.items():
        li = data["layer_idx"]
        layer_data.setdefault(li, {"vecs": [], "sigs": []})
        layer_data[li]["vecs"].append(data["u"])
        layer_data[li]["sigs"].append(data["sigma"])

    combined = {}
    for li, d in layer_data.items():
        total_sig = sum(d["sigs"])
        u_comb = sum(s * v for s, v in zip(d["sigs"], d["vecs"])) / total_sig
        u_comb = u_comb / (u_comb.norm() + 1e-8)
        combined[li] = {"u": u_comb, "sigma": total_sig}

    print(f"[SVD] Combined into {len(combined)} per-layer vectors.", flush=True)
    return combined


# ── Mean-difference vectors ────────────────────────────────────────────────────
def collect_residual_states(model, tokenizer, problems):
    model.eval()
    stores = {}
    hooks = []
    def make_hook(li):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            stores.setdefault(li, []).append(h.detach().float().mean(dim=(0,1)).cpu())
        return fn
    for name, mod in model.named_modules():
        if hasattr(mod, "self_attn") and hasattr(mod, "mlp"):
            for p in name.split("."):
                try:
                    li = int(p)
                    hooks.append(mod.register_forward_hook(make_hook(li)))
                    break
                except ValueError: continue
    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad(): model(**inp)
    for h in hooks: h.remove()
    return {li: torch.stack(vs).mean(0) for li, vs in stores.items()}


def get_mean_diff_vectors(calib_problems):
    print("[MEANDIFF] Source base...", flush=True)
    src = AutoModelForCausalLM.from_pretrained(MODELS["math_base"], torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODELS["math_base"]); tok.pad_token = tok.eos_token
    src_st = collect_residual_states(src, tok, calib_problems)
    del src; gc.collect(); torch.cuda.empty_cache()

    print("[MEANDIFF] RLVR model...", flush=True)
    rlvr = AutoModelForCausalLM.from_pretrained(MODELS["rlvr"], torch_dtype=torch.float16, device_map="auto")
    tok2 = AutoTokenizer.from_pretrained(MODELS["rlvr"]); tok2.pad_token = tok2.eos_token
    rlvr_st = collect_residual_states(rlvr, tok2, calib_problems)
    del rlvr; gc.collect(); torch.cuda.empty_cache()

    diff = {li: rlvr_st[li] - src_st[li] for li in src_st if li in rlvr_st}
    print(f"[MEANDIFF] {len(diff)} layers.", flush=True)
    return diff


# ── Steering application ───────────────────────────────────────────────────────
def apply_svd_steering(model, tokenizer, problems, layer_vecs, alpha, label,
                       top_k=None, sign_flip=False, random_seed=None,
                       wrong_layer_seed=None) -> dict:
    """
    Apply SVD-derived residual steering.
    sign_flip        : negate all u (sign-flip control)
    random_seed      : if set, replace u with random unit vector (null control)
    wrong_layer_seed : if set, permute vectors among active K layers using this seed
    """
    sigma_max = max(d["sigma"] for d in layer_vecs.values())

    # Determine active layers
    if top_k:
        ranked = sorted(layer_vecs, key=lambda i: layer_vecs[i]["sigma"], reverse=True)
        active = ranked[:top_k]
    else:
        active = list(layer_vecs.keys())

    # Build {layer_idx: (u_vector, weight)} map
    vec_map = {}
    for li in active:
        u = layer_vecs[li]["u"].clone()
        w = layer_vecs[li]["sigma"] / sigma_max
        if sign_flip:
            u = -u
        if random_seed is not None:
            rng = torch.Generator(); rng.manual_seed(random_seed * 1000 + li)
            u = torch.randn(u.shape, generator=rng)
            u = u / (u.norm() + 1e-8)
        vec_map[li] = (u, w)

    # Wrong-layer: permute vectors among active positions using given seed
    if wrong_layer_seed is not None and len(active) > 1:
        rng_py = random.Random(wrong_layer_seed)
        shuffled = active.copy()
        rng_py.shuffle(shuffled)
        new_map = {}
        for orig, shuf in zip(active, shuffled):
            u_shuf, _ = vec_map[shuf]
            _, w_orig = vec_map[orig]
            new_map[orig] = (u_shuf, w_orig)
        vec_map = new_map

    hooks = []
    for name, mod in model.named_modules():
        if not (hasattr(mod, "self_attn") and hasattr(mod, "mlp")): continue
        li = None
        for p in name.split("."):
            try: li = int(p); break
            except ValueError: continue
        if li is None or li not in vec_map: continue
        u, w = vec_map[li]
        u_dev = u.to(model.device, dtype=model.dtype)

        def make_hook(uv, wt):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                steer = (alpha * wt * uv).to(h.dtype)
                h = h + steer.unsqueeze(0).unsqueeze(0)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return fn
        hooks.append(mod.register_forward_hook(make_hook(u_dev, w)))

    result = evaluate(model, tokenizer, problems, label)
    for h in hooks: h.remove()
    return result


def apply_meandiff_steering(model, tokenizer, problems, diff_vecs, alpha, label) -> dict:
    hooks = []
    for name, mod in model.named_modules():
        if not (hasattr(mod, "self_attn") and hasattr(mod, "mlp")): continue
        li = None
        for p in name.split("."):
            try: li = int(p); break
            except ValueError: continue
        if li is None or li not in diff_vecs: continue
        dv = diff_vecs[li].to(model.device, dtype=model.dtype)

        def make_hook(d):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h + (alpha * d.to(h.dtype)).unsqueeze(0).unsqueeze(0)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return fn
        hooks.append(mod.register_forward_hook(make_hook(dv)))

    result = evaluate(model, tokenizer, problems, label)
    for h in hooks: h.remove()
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("="*72)
    print("PAPER EVAL SUITE v2 — all reviewer fixes")
    print("  CALIB=450-499  VAL=400-449  TEST=0-399")
    print("="*72, flush=True)

    calib = load_math500_split("calib")
    val   = load_math500_split("val")
    test  = load_math500_split("test")

    # Step 1: vectors
    print("\n[STEP 1] Steering vectors on CALIB...", flush=True)
    mean_diff   = get_mean_diff_vectors(calib)
    svd_vecs    = get_svd_vectors(calib, orientation="source_gate")

    # Step 2: target model
    print("\n[STEP 2] Loading target model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODELS["instruct"])
    tok.pad_token = tok.eos_token

    # Step 3: baselines
    print("\n[STEP 3] Baselines...", flush=True)
    val_bl  = evaluate(model, tok, val,  "baseline_val")
    test_bl = evaluate(model, tok, test, "baseline_test")
    bl_val, bl_test = val_bl["accuracy"], test_bl["accuracy"]
    print(f"  VAL  baseline: {bl_val:.1f}%")
    print(f"  TEST baseline: {bl_test:.1f}%", flush=True)

    results = {"baseline_val": val_bl, "baseline_test": test_bl}

    # Step 4: VAL alpha sweeps
    print("\n[STEP 4] VAL alpha sweeps...", flush=True)
    svd_val = {}
    for alpha in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        r = apply_svd_steering(model, tok, val, svd_vecs, alpha, f"val_svd_a{alpha}")
        svd_val[alpha] = r["accuracy"]; results[f"val_svd_a{alpha}"] = r
        print(f"  SVD full α={alpha}: {r['accuracy']:.1f}%", flush=True)
    best_svd_a = max(svd_val, key=svd_val.get)

    topk_val = {}
    for k in [5, 10, 15, 20]:
        r = apply_svd_steering(model, tok, val, svd_vecs, best_svd_a,
                               f"val_svd_top{k}", top_k=k)
        topk_val[k] = r["accuracy"]; results[f"val_svd_top{k}"] = r
        print(f"  SVD top-{k} α={best_svd_a}: {r['accuracy']:.1f}%", flush=True)
    best_k = max(topk_val, key=topk_val.get)

    md_val = {}
    for alpha in [0.01, 0.02, 0.05, 0.07, 0.1]:
        r = apply_meandiff_steering(model, tok, val, mean_diff, alpha, f"val_md_a{alpha}")
        md_val[alpha] = r["accuracy"]; results[f"val_md_a{alpha}"] = r
        print(f"  MeanDiff α={alpha}: {r['accuracy']:.1f}%", flush=True)
    best_md_a = max(md_val, key=md_val.get)

    print(f"\n  → Best: SVD α={best_svd_a} K={best_k}  MeanDiff α={best_md_a}", flush=True)

    # Step 5: TEST with selected configs
    print("\n[STEP 5] TEST evaluation (n=400)...", flush=True)
    summary = {}

    r = apply_svd_steering(model, tok, test, svd_vecs, best_svd_a,
                           f"test_svd_full_a{best_svd_a}")
    results[r["label"]] = r; summary["SVD_full"] = r
    print_result_row("SVD full", r, [it["correct"] for it in test_bl["items"]])

    r = apply_svd_steering(model, tok, test, svd_vecs, best_svd_a,
                           f"test_svd_top{best_k}_a{best_svd_a}", top_k=best_k)
    results[r["label"]] = r; summary[f"SVD_top{best_k}"] = r
    print_result_row(f"SVD top-{best_k}", r, [it["correct"] for it in test_bl["items"]])

    r = apply_meandiff_steering(model, tok, test, mean_diff, best_md_a,
                                f"test_meandiff_a{best_md_a}")
    results[r["label"]] = r; summary["MeanDiff"] = r
    print_result_row("MeanDiff", r, [it["correct"] for it in test_bl["items"]])

    # Step 6: Controls — 20 seeds each
    print(f"\n[STEP 6] {N_NULL_SEEDS}-seed null controls on TEST...", flush=True)
    base_items = [it["correct"] for it in test_bl["items"]]

    # 6a: Random unit vectors (matched per-layer norms)
    rand_accs = []
    for seed in range(N_NULL_SEEDS):
        r = apply_svd_steering(model, tok, test, svd_vecs, best_svd_a,
                               f"test_rand_s{seed}", random_seed=seed, top_k=best_k)
        rand_accs.append(r["accuracy"])
        results[r["label"]] = r
    print(f"  Random dirs: mean={np.mean(rand_accs):.1f}%  "
          f"std={np.std(rand_accs):.1f}%  "
          f"range=[{min(rand_accs):.1f}, {max(rand_accs):.1f}]", flush=True)

    # 6b: Random sign patterns (independent per-layer sign flips)
    sign_accs = []
    for seed in range(N_NULL_SEEDS):
        rng = random.Random(seed)
        flipped = {li: {**d, "u": d["u"] * (1 if rng.random() > 0.5 else -1)}
                   for li, d in svd_vecs.items()}
        r = apply_svd_steering(model, tok, test, flipped, best_svd_a,
                               f"test_randsign_s{seed}", top_k=best_k)
        sign_accs.append(r["accuracy"])
        results[r["label"]] = r
    print(f"  Random signs: mean={np.mean(sign_accs):.1f}%  "
          f"std={np.std(sign_accs):.1f}%", flush=True)

    # 6c: Wrong-layer (permuted vectors among top-K positions), distinct seed each draw
    wrong_accs = []
    for seed in range(N_NULL_SEEDS):
        r = apply_svd_steering(model, tok, test, svd_vecs, best_svd_a,
                               f"test_wronglayer_s{seed}", top_k=best_k,
                               wrong_layer_seed=seed)
        wrong_accs.append(r["accuracy"])
        results[r["label"]] = r
    print(f"  Wrong-layer:  mean={np.mean(wrong_accs):.1f}%  "
          f"std={np.std(wrong_accs):.1f}%", flush=True)

    # 6d: Random layer rankings (uniformly random K layers, not top-K)
    randlayer_accs = []
    all_layers = list(svd_vecs.keys())
    for seed in range(N_NULL_SEEDS):
        rng = random.Random(seed + 100)
        rand_k = rng.sample(all_layers, min(best_k, len(all_layers)))
        rand_subset = {li: svd_vecs[li] for li in rand_k}
        r = apply_svd_steering(model, tok, test, rand_subset, best_svd_a,
                               f"test_randlayer_s{seed}")
        randlayer_accs.append(r["accuracy"])
        results[r["label"]] = r
    print(f"  Random K layers: mean={np.mean(randlayer_accs):.1f}%  "
          f"std={np.std(randlayer_accs):.1f}%", flush=True)

    # Step 7: McNemar + bootstrap for main methods
    print("\n[STEP 7] Paired statistics...", flush=True)
    stats = {}
    for key, res in summary.items():
        mi = [it["correct"] for it in res["items"]]
        mn = mcnemar_test(base_items, mi)
        bc = bootstrap_ci(base_items, mi)
        stats[key] = {**mn, **bc}
        lo, hi = wilson_ci(res["correct"], res["total"])
        print(f"  {key:<20} acc={res['accuracy']:.1f}%  [{lo},{hi}]  "
              f"Δ={bc['mean_diff_pp']:+.1f}pp  McNemar p={mn['p_value']:.3f}", flush=True)

    # Save
    print("\n[SAVE]", flush=True)
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "items"}
            for k, v in results.items()}
    with open(OUTPUT_DIR / "paper_eval_results.json", "w") as f:
        json.dump(slim, f, indent=2)

    save_stats = {
        "baseline_test_acc": bl_test, "baseline_val_acc": bl_val,
        "best_svd_alpha": best_svd_a, "best_k": best_k,
        "best_md_alpha": best_md_a,
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "items"}
                    for k, v in summary.items()},
        "bootstrap_stats": stats,
        "null_distributions": {
            "random_dirs":    {"mean": np.mean(rand_accs),   "std": np.std(rand_accs),   "values": rand_accs},
            "random_signs":   {"mean": np.mean(sign_accs),   "std": np.std(sign_accs),   "values": sign_accs},
            "wrong_layer":    {"mean": np.mean(wrong_accs),  "std": np.std(wrong_accs),  "values": wrong_accs},
            "random_k_layers":{"mean": np.mean(randlayer_accs), "std": np.std(randlayer_accs), "values": randlayer_accs},
        },
    }
    with open(OUTPUT_DIR / "paper_eval_stats.json", "w") as f:
        json.dump(save_stats, f, indent=2)

    # Final table
    print("\n" + "="*72)
    print("FINAL  (TEST n=400, problems 0-399)")
    print("="*72)
    bl_lo, bl_hi = wilson_ci(test_bl["correct"], test_bl["total"])
    print(f"  {'Baseline':<50} {bl_test:.1f}%  [{bl_lo},{bl_hi}]")
    for key, res in summary.items():
        print_result_row(key, res, base_items)
    print(f"\n  Null distributions (mean ± std):")
    for name, vals in save_stats["null_distributions"].items():
        print(f"    {name:<20} {vals['mean']:.1f}% ± {vals['std']:.1f}%")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
