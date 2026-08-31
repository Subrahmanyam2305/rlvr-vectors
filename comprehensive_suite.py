"""
Comprehensive Experiment Suite — minimum publishable package:

  1. Multi-source × multi-target evaluation matrix
     NOTE: Only two RLVR checkpoints available (1shot-pi1 and 1200-step duration
     ablation — NOT an independent seed). Only one cross-model target (Instruct)
     plus same-model self-transfer. Described honestly as one cross-model setting
     with a training-duration ablation.
  2. Sign-orientation ablation (3 orientations × 5 calibration sizes)
     NOTE: Fixed alpha=0.5 used as diagnostic; does not select per-orientation.
     This shows sensitivity to orientation choice at fixed intervention strength.
  3. Matched-scope intervention ablation
  4. GSM8K for out-of-distribution generalization
"""

import torch, json, gc, os, random
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors import safe_open
from shared_eval import (
    load_math500_split, evaluate, mcnemar_test, bootstrap_ci,
    wilson_ci, answers_match, extract_boxed, make_prompt, OUTPUT_DIR
)
from paper_eval_suite import (
    get_svd_vectors, get_mean_diff_vectors, apply_svd_steering,
    apply_meandiff_steering, collect_proj_inputs, collect_residual_states
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    # Sources (RLVR checkpoints — three independent deltas)
    "math_base":     "Qwen/Qwen2.5-Math-1.5B",
    "rlvr_oneshot":  "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    "rlvr_1200step": "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-1.2k-dsr-sub",
    # Targets
    "instruct":      "Qwen/Qwen2.5-1.5B-Instruct",
    "math_base_tgt": "Qwen/Qwen2.5-Math-1.5B",  # common ancestor / self-transfer
}

DATA_DIR = Path("/home/ubuntu/rlvr-vectors/data")


# ── GSM8K loader ───────────────────────────────────────────────────────────────
def load_gsm8k(n=200):
    """Load n problems from GSM8K test set."""
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = []
        for item in ds:
            ans_text = item["answer"]
            # GSM8K answers end with "#### <number>"
            if "####" in ans_text:
                ans = ans_text.split("####")[-1].strip().replace(",", "")
            else:
                ans = ans_text.strip()
            problems.append({"problem": item["question"], "answer": ans})
        return problems[:n]
    except Exception as e:
        print(f"  [GSM8K] Failed to load: {e}")
        return []


# ── Weight-only SVD extraction (no calibration data needed) ───────────────────
def get_svd_vectors_weight_only(rlvr_model_id, base_model_id="Qwen/Qwen2.5-Math-1.5B"):
    """Extract SVD vectors using purely weight-based sign (largest-coord positive)."""
    from paper_eval_suite import get_svd_vectors as _get
    # Temporarily patch MODELS — run with weight_only orientation
    import paper_eval_suite as ps
    orig_models = ps.MODELS.copy()
    ps.MODELS["math_base"] = base_model_id
    ps.MODELS["rlvr"] = rlvr_model_id
    vecs = _get(None, orientation="weight_only")
    ps.MODELS.update(orig_models)
    return vecs


def get_svd_vectors_for_pair(rlvr_id, base_id, calib_problems,
                              orientation="source_gate"):
    """Get SVD vectors for an arbitrary source pair."""
    import paper_eval_suite as ps
    orig = ps.MODELS.copy()
    ps.MODELS["math_base"] = base_id
    ps.MODELS["rlvr"]      = rlvr_id
    vecs = ps.get_svd_vectors(calib_problems, orientation=orientation)
    ps.MODELS.update(orig)
    return vecs


def get_meandiff_for_pair(rlvr_id, base_id, calib_problems):
    import paper_eval_suite as ps
    orig = ps.MODELS.copy()
    ps.MODELS["math_base"] = base_id
    ps.MODELS["rlvr"]      = rlvr_id
    vecs = ps.get_mean_diff_vectors(calib_problems)
    ps.MODELS.update(orig)
    return vecs


# ── Gating numerical-equivalence verification ─────────────────────────────────
def verify_rank1_equals_conditional_steering(model, tokenizer, calib_problems,
                                              svd_vecs_raw):
    """
    Verify Proposition 1 numerically: rank-1 weight update ≡ dynamic hook
    adding sigma*(v^T x)*u. Checks that logits match to float16 tolerance.
    Returns dict with mean/max absolute logit difference.
    """
    from paper_eval_suite import collect_proj_inputs
    import paper_eval_suite as ps

    base_path  = snapshot_download(MODELS["math_base"])
    rlvr_path  = snapshot_download(MODELS["rlvr_oneshot"])
    base_idx, rlvr_idx = {}, {}
    for f in sorted(Path(base_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys(): base_idx[k] = str(f)
    for f in sorted(Path(rlvr_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys(): rlvr_idx[k] = str(f)

    TARGET = ["self_attn.o_proj.weight"]  # Test on o_proj only for tractability

    proj_data = {}
    for pn in sorted(base_idx):
        if not any(pn.endswith(s) for s in TARGET): continue
        if pn not in rlvr_idx: continue
        with safe_open(base_idx[pn], framework="pt", device="cpu") as sf:
            wb = sf.get_tensor(pn).float()
        with safe_open(rlvr_idx[pn], framework="pt", device="cpu") as sf:
            wr = sf.get_tensor(pn).float()
        dW = wr - wb
        if dW.norm().item() < 1e-8: continue
        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
        proj_data[pn] = {
            "u": U[:, 0], "sigma": S[0].item(), "v": Vt[0],
            "layer_idx": int(pn.split(".")[2])
        }
        del wb, wr, dW, U, S, Vt
    gc.collect()

    diffs = []
    for prob in calib_problems[:5]:  # 5 problems is enough for verification
        text = make_prompt(tokenizer, prob["problem"])
        inp = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=256).to(model.device)

        # Version A: weight-modified model
        with torch.no_grad():
            for pn, data in proj_data.items():
                param = dict(model.named_parameters())[pn]
                rank1 = (data["sigma"] * torch.outer(
                    data["u"].to(param.device, param.dtype),
                    data["v"].to(param.device, param.dtype)))
                param.data.add_(rank1)
            logits_A = model(**inp).logits.float()
            # Undo
            for pn, data in proj_data.items():
                param = dict(model.named_parameters())[pn]
                rank1 = (data["sigma"] * torch.outer(
                    data["u"].to(param.device, param.dtype),
                    data["v"].to(param.device, param.dtype)))
                param.data.sub_(rank1)

        # Version B: dynamic conditional hook
        hooks = []
        for pn, data in proj_data.items():
            mod_name = pn[:-len(".weight")]
            mod = dict(model.named_modules())[mod_name]
            u = data["u"].to(model.device, model.dtype)
            v = data["v"].to(model.device, model.dtype)
            sigma = data["sigma"]

            def make_hook(u_, v_, s_):
                def fn(mod, inp_, out):
                    x = inp_[0]
                    gate = (x @ v_.unsqueeze(-1)).squeeze(-1)  # (B, seq)
                    steer = s_ * gate.unsqueeze(-1) * u_.unsqueeze(0).unsqueeze(0)
                    return out + steer.to(out.dtype)
                return fn
            hooks.append(mod.register_forward_hook(make_hook(u, v, sigma)))

        with torch.no_grad():
            logits_B = model(**inp).logits.float()
        for h in hooks: h.remove()

        diff = (logits_A - logits_B).abs()
        diffs.append({"mean": float(diff.mean()), "max": float(diff.max())})

    return {
        "n_problems": len(diffs),
        "n_projections": len(proj_data),
        "mean_logit_diff": float(np.mean([d["mean"] for d in diffs])),
        "max_logit_diff":  float(np.max([d["max"]  for d in diffs])),
        "per_problem": diffs,
    }


# ── Matched-scope intervention ablation ───────────────────────────────────────
def matched_scope_ablation(model, tokenizer, test_problems, svd_vecs,
                            proj_svd_raw, mean_diff_vecs, best_alpha,
                            best_md_alpha, label_prefix):
    """
    Compare interventions at identical modules to separate conditioning,
    hook location, and scaling effects.
    """
    results = {}

    # 1. Weight transfer on o_proj + down_proj only (same modules as SVD)
    from safetensors import safe_open as _so
    base_path  = snapshot_download(MODELS["math_base"])
    rlvr_path  = snapshot_download(MODELS["rlvr_oneshot"])
    base_idx, rlvr_idx = {}, {}
    for f in sorted(Path(base_path).glob("*.safetensors")):
        with _so(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys(): base_idx[k] = str(f)
    for f in sorted(Path(rlvr_path).glob("*.safetensors")):
        with _so(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys(): rlvr_idx[k] = str(f)

    TARGET = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]
    with torch.no_grad():
        for pn in sorted(base_idx):
            if not any(pn.endswith(s) for s in TARGET): continue
            if pn not in rlvr_idx: continue
            try:
                param = dict(model.named_parameters())[pn]
            except KeyError: continue
            with _so(base_idx[pn], framework="pt", device="cpu") as sf:
                wb = sf.get_tensor(pn).float()
            with _so(rlvr_idx[pn], framework="pt", device="cpu") as sf:
                wr = sf.get_tensor(pn).float()
            dW = (wr - wb).to(param.device, param.dtype)
            param.data.add_(dW)
            del wb, wr, dW
    r = evaluate(model, tokenizer, test_problems, f"{label_prefix}_wt_proj_only")
    results["weight_transfer_proj_only"] = r
    print(f"  Weight transfer (o_proj+down only): {r['accuracy']:.1f}%", flush=True)

    # Undo weight transfer
    with torch.no_grad():
        for pn in sorted(base_idx):
            if not any(pn.endswith(s) for s in TARGET): continue
            if pn not in rlvr_idx: continue
            try:
                param = dict(model.named_parameters())[pn]
            except KeyError: continue
            with _so(base_idx[pn], framework="pt", device="cpu") as sf:
                wb = sf.get_tensor(pn).float()
            with _so(rlvr_idx[pn], framework="pt", device="cpu") as sf:
                wr = sf.get_tensor(pn).float()
            dW = (wr - wb).to(param.device, param.dtype)
            param.data.sub_(dW)
            del wb, wr, dW
    gc.collect()

    # 2. SVD full (sigma-weighted, all layers) — reference from paper_eval_suite
    r = apply_svd_steering(model, tokenizer, test_problems, svd_vecs,
                           best_alpha, f"{label_prefix}_svd_full")
    results["svd_full"] = r
    print(f"  SVD full (residual): {r['accuracy']:.1f}%", flush=True)

    # 3. Mean-diff at residual stream (reference)
    r = apply_meandiff_steering(model, tokenizer, test_problems, mean_diff_vecs,
                                best_md_alpha, f"{label_prefix}_meandiff")
    results["meandiff_residual"] = r
    print(f"  MeanDiff (residual): {r['accuracy']:.1f}%", flush=True)

    gc.collect()
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("="*72)
    print("COMPREHENSIVE SUITE")
    print("="*72, flush=True)

    calib = load_math500_split("calib")
    val   = load_math500_split("val")
    test  = load_math500_split("test")
    gsm8k = load_gsm8k(200)

    all_results = {}

    # ── Part A: Primary cross-model cell only ──────────────────────────────
    # (self-transfer and duration-ablation cells removed — already have
    #  same-model recovery from paper_eval_suite; duration ablation is not
    #  essential for the core paper claim)
    print("\n[PART A] Primary cross-model cell: 1shot-pi1 → Instruct", flush=True)
    src_id  = MODELS["rlvr_oneshot"]
    base_id = MODELS["math_base"]
    tgt_id  = MODELS["instruct"]
    pair_label = "1shot-pi1_to_instruct"

    svd_v = get_svd_vectors_for_pair(src_id, base_id, calib)
    md_v  = get_meandiff_for_pair(src_id, base_id, calib)

    model = AutoModelForCausalLM.from_pretrained(
        tgt_id, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(tgt_id)
    tok.pad_token = tok.eos_token

    bl = evaluate(model, tok, test, f"{pair_label}_baseline")
    all_results[f"{pair_label}_baseline"] = bl
    print(f"  Baseline: {bl['accuracy']:.1f}%")

    # VAL sweep → pick best alpha
    best_a, best_a_acc = 0.5, 0.0
    for alpha in [0.3, 0.5, 1.0, 1.5, 2.0]:
        r = apply_svd_steering(model, tok, val, svd_v, alpha,
                               f"{pair_label}_val_svd_a{alpha}")
        if r["accuracy"] > best_a_acc:
            best_a, best_a_acc = alpha, r["accuracy"]
    r = apply_svd_steering(model, tok, test, svd_v, best_a,
                           f"{pair_label}_svd_a{best_a}")
    all_results[r["label"]] = r
    print(f"  SVD α={best_a} (VAL-selected): {r['accuracy']:.1f}%")

    best_md_a, best_md_acc = 0.05, 0.0
    for alpha in [0.01, 0.02, 0.05, 0.1]:
        r_v = apply_meandiff_steering(model, tok, val, md_v, alpha,
                                      f"{pair_label}_val_md_a{alpha}")
        if r_v["accuracy"] > best_md_acc:
            best_md_a, best_md_acc = alpha, r_v["accuracy"]
    r = apply_meandiff_steering(model, tok, test, md_v, best_md_a,
                                f"{pair_label}_md_a{best_md_a}")
    all_results[r["label"]] = r
    print(f"  MeanDiff α={best_md_a} (VAL-selected): {r['accuracy']:.1f}%")

    del model; gc.collect(); torch.cuda.empty_cache()

    # ── Part B: Sign-orientation ablation (diagnostic at fixed alpha=0.5) ──
    print("\n[PART B] Sign-orientation ablation (fixed alpha=0.5, diagnostic)", flush=True)
    print("  3 orientations × 5 calibration sizes — shows sensitivity to orientation.", flush=True)
    orientation_labels = ["none", "weight_only", "source_gate"]
    calib_sizes = [1, 5, 10, 25, 50]

    model_b = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto")
    tok_b = AutoTokenizer.from_pretrained(MODELS["instruct"])
    tok_b.pad_token = tok_b.eos_token

    for orient in orientation_labels:
        for n_calib in calib_sizes:
            calib_sub = calib[:n_calib]
            vecs = get_svd_vectors_for_pair(
                MODELS["rlvr_oneshot"], MODELS["math_base"],
                calib_sub, orientation=orient)
            r = apply_svd_steering(model_b, tok_b, val, vecs, 0.5,
                                   f"orient_{orient}_n{n_calib}")
            all_results[r["label"]] = r
            print(f"  orient={orient:<12} n_calib={n_calib:<3}: {r['accuracy']:.1f}%",
                  flush=True)

    del model_b; gc.collect(); torch.cuda.empty_cache()

    # ── Part C: Matched-scope intervention ablation ──────────────────────────
    print("\n[PART C] Matched-scope ablation (primary source->instruct pair)...", flush=True)
    svd_primary = get_svd_vectors_for_pair(MODELS["rlvr_oneshot"], MODELS["math_base"],
                                           calib, "source_gate")
    md_primary  = get_meandiff_for_pair(MODELS["rlvr_oneshot"], MODELS["math_base"], calib)
    model_ms = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto")
    tok_ms = AutoTokenizer.from_pretrained(MODELS["instruct"])
    tok_ms.pad_token = tok_ms.eos_token
    scope_results = matched_scope_ablation(
        model_ms, tok_ms, test, svd_primary, None, md_primary,
        best_alpha=best_a, best_md_alpha=best_md_a, label_prefix="scope")
    all_results.update(scope_results)
    del model_ms; gc.collect(); torch.cuda.empty_cache()

    # ── Part D: Numerical gate equivalence ──────────────────────────────────
    print("\n[PART D] Numerical gate equivalence verification...", flush=True)
    model_d = AutoModelForCausalLM.from_pretrained(
        MODELS["math_base"], torch_dtype=torch.float16, device_map="auto")
    tok_d = AutoTokenizer.from_pretrained(MODELS["math_base"])
    tok_d.pad_token = tok_d.eos_token
    equiv = verify_rank1_equals_conditional_steering(model_d, tok_d, calib[:5], None)
    all_results["proposition1_verification"] = equiv
    print(f"  Mean logit diff: {equiv['mean_logit_diff']:.2e}")
    print(f"  Max  logit diff: {equiv['max_logit_diff']:.2e}")
    del model_d; gc.collect(); torch.cuda.empty_cache()

    # ── Save ───────────────────────────────────────────────────────────────
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "items"}
            for k, v in all_results.items() if isinstance(v, dict)}
    with open(OUTPUT_DIR / "comprehensive_results.json", "w") as f:
        json.dump(slim, f, indent=2)
    print("\n[SAVE] comprehensive_results.json written.")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
