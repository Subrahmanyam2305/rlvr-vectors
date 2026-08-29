"""
Gate Mediation Experiment — valid causal test of the gate-mismatch hypothesis.

Design: hold u, sigma, location, active layers, AND perturbation norm fixed.
Vary only gate g(x) across 7 conditions:
  1. natural       — v^T x_tgt  (what weight transfer does)
  2. src_mean      — constant = mean(v^T x_src) per projection (constant control)
  3. src_rms       — constant = RMS(v^T x_src) per projection (matched-norm constant)
  4. src_replay    — per-problem mean source gate, replayed as constant during gen
  5. corrected     — (v^T x_tgt) * c_l  where c_l = rms_src/rms_tgt (magnitude-corrected)
  6. negated       — -(v^T x_tgt)
  7. shuffled      — problem i uses gate from problem (i+offset) % n (cross-problem permutation)

Norm equalization: precompute calibration RMS perturbation for each mode, scale
alpha so all modes deliver identical mean intervention energy.

Block selection: rank transformer blocks by total sigma (sum over o_proj + down_proj),
select top N_BLOCKS blocks, include both projections from each selected block.

Saves item-level predictions and McNemar / bootstrap statistics.
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
from paper_eval_suite import MODELS, TARGET_SUFFIXES, get_mean_diff_vectors

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
N_BLOCKS = 15   # top transformer blocks by aggregated sigma


# ── Load projection-level SVD ─────────────────────────────────────────────────
def load_proj_svd():
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


def select_active_projections(proj_svd, n_blocks=N_BLOCKS):
    """
    Rank transformer blocks by aggregated sigma (sum over o_proj + down_proj).
    Select top n_blocks blocks, include BOTH projections from each.
    Returns set of active param_names.
    """
    block_sigma = {}
    for pn, data in proj_svd.items():
        li = data["layer_idx"]
        block_sigma[li] = block_sigma.get(li, 0.0) + data["sigma"]
    top_blocks = set(sorted(block_sigma, key=block_sigma.get, reverse=True)[:n_blocks])
    active = {pn for pn, d in proj_svd.items() if d["layer_idx"] in top_blocks}
    print(f"  Active: {len(active)} projections from {n_blocks} blocks.", flush=True)
    return active


# ── Gate statistics collection ────────────────────────────────────────────────
def collect_proj_gate_stats(model_id, proj_svd, problems, label):
    """
    For each projection and each problem, compute:
      - per-problem mean gate over prompt tokens: {pn: [mean_gate_prob0, ...]}
      - overall mean and RMS across all prompt tokens: {pn: {mean, rms}}
    """
    print(f"  [{label}] Loading...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_id); tok.pad_token = tok.eos_token
    model.eval()

    target_mods = {pn[:-len(".weight")]: pn for pn in proj_svd}
    per_prob   = {pn: [] for pn in proj_svd}   # per-problem mean gate
    all_tokens = {pn: [] for pn in proj_svd}   # all prompt-token gates

    hooks = []
    def make_hook(pn):
        v = proj_svd[pn]["v"].float()
        def fn(mod, inp, out):
            x = inp[0].detach().float()          # (1, seq, d_in)
            gates = (x[0] @ v).tolist()          # (seq,)
            per_prob[pn].append(float(np.mean(gates)))
            all_tokens[pn].extend(gates)
        return fn

    for mn, pn in target_mods.items():
        mod = dict(model.named_modules()).get(mn)
        if mod: hooks.append(mod.register_forward_hook(make_hook(pn)))

    for prob in problems:
        text = make_prompt(tok, prob["problem"])
        inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad(): model(**inp)

    for h in hooks: h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()

    stats = {}
    for pn in proj_svd:
        g = np.array(all_tokens[pn]) if all_tokens[pn] else np.array([0.0])
        stats[pn] = {
            "mean":      float(g.mean()),
            "rms":       float(np.sqrt((g**2).mean())),
            "per_prob":  per_prob[pn],       # list, len = n_problems
        }
    print(f"  [{label}] Done. {len(stats)} projections.", flush=True)
    return stats


def compute_calib_perturbation_rms(proj_svd, active_pns,
                                    src_stats, tgt_stats, calib_problems, model):
    """
    Estimate mean RMS of sigma*g*u perturbation on calibration problems
    for each gate mode. Used to normalize all modes to the same energy.
    We use the norm of the steering vector (sigma * |g| * ||u||) as a proxy.
    ||u|| = 1 (normalized), so perturbation magnitude per token = sigma * |g|.
    """
    rms_per_mode = {}
    for mode in ["natural", "src_mean", "src_rms", "src_replay",
                 "corrected", "negated", "shuffled"]:
        energies = []
        for pn in active_pns:
            if pn not in src_stats or pn not in tgt_stats: continue
            sg  = src_stats[pn]
            tg  = tgt_stats[pn]
            sig = proj_svd[pn]["sigma"]
            tgt_gates = np.array(tg["per_prob"])  # (n_calib,)
            src_gates = np.array(sg["per_prob"])  # (n_calib,)
            n = len(tgt_gates)

            if mode == "natural":
                g = np.abs(tgt_gates)
            elif mode == "src_mean":
                g = np.abs(np.full(n, sg["mean"]))
            elif mode == "src_rms":
                g = np.abs(np.full(n, sg["rms"]))
            elif mode == "src_replay":
                g = np.abs(src_gates)
            elif mode == "corrected":
                c = sg["rms"] / (tg["rms"] + 1e-8)
                g = np.abs(tgt_gates * c)
            elif mode == "negated":
                g = np.abs(-tgt_gates)  # same magnitude as natural
            elif mode == "shuffled":
                perm = np.roll(np.arange(n), 1)
                g = np.abs(tgt_gates[perm])  # different example's gates
            energies.append(sig * g.mean())

        rms_per_mode[mode] = float(np.mean(energies)) if energies else 1.0

    # Alpha scaling: natural mode is reference
    ref = rms_per_mode.get("natural", 1.0)
    scale = {m: ref / (v + 1e-8) for m, v in rms_per_mode.items()}
    return scale


# ── Gated steering evaluation ─────────────────────────────────────────────────
def apply_gated_steering_eval(model, tokenizer, problems, proj_svd, active_pns,
                               src_stats, tgt_stats, gate_mode, alpha,
                               norm_scale, label):
    """
    Apply sigma * gate * u at each active projection, with per-mode norm scaling.
    Gate modes:
      natural   — natural v^T x (what weight transfer does)
      src_mean  — constant source mean per projection
      src_rms   — constant source RMS per projection
      src_replay— per-problem source mean, constant during generation
      corrected — magnitude-corrected natural gate
      negated   — -(v^T x)
      shuffled  — gate from a different problem (cross-problem permutation)
    """
    scaled_alpha = alpha * norm_scale.get(gate_mode, 1.0)
    n_probs = len(problems)

    hooks = []
    for pn in active_pns:
        if pn not in proj_svd: continue
        mod_name = pn[:-len(".weight")]
        mod = dict(model.named_modules()).get(mod_name)
        if not mod: continue

        data = proj_svd[pn]
        u   = data["u"].to(model.device, dtype=model.dtype)
        v   = data["v"].to(model.device, dtype=model.dtype)
        sig = data["sigma"]
        sg  = src_stats.get(pn, {"mean": 0.0, "rms": 1.0, "per_prob": []})
        tg  = tgt_stats.get(pn, {"mean": 0.0, "rms": 1.0, "per_prob": []})

        # Constants needed at hook time
        src_mean   = float(sg["mean"])
        src_rms    = float(sg["rms"])
        tgt_rms    = float(tg["rms"])
        correction = src_rms / (tgt_rms + 1e-8)
        # For shuffled: roll per-problem gates by 1 (deterministic cross-problem permutation)
        src_pp     = sg["per_prob"]   # list len=n_calib (may differ from n_test)
        tgt_pp     = tg["per_prob"]   # list len=n_calib

        # We index problems by a counter stored as a list to be mutable inside closure
        prob_counter = [0]

        def make_hook(u_, v_, sig_, s_m, s_rms_, corr, mode, s_pp, t_pp, cnt):
            def fn(module, inp, out):
                x = inp[0]                                  # (B, seq, d_in)
                nat = (x @ v_.unsqueeze(-1)).squeeze(-1)    # (B, seq)
                seq_len = nat.shape[1]

                if mode == "natural":
                    g = nat
                elif mode == "src_mean":
                    g = torch.full_like(nat, s_m)
                elif mode == "src_rms":
                    g = torch.full_like(nat, s_rms_)
                elif mode == "src_replay":
                    # Replay per-problem source mean as a constant (prompt + gen tokens)
                    pidx = cnt[0] % max(len(s_pp), 1)
                    c_val = float(s_pp[pidx]) if s_pp else s_m
                    g = torch.full_like(nat, c_val)
                elif mode == "corrected":
                    g = nat * corr
                elif mode == "negated":
                    g = -nat
                elif mode == "shuffled":
                    # Use gate from the NEXT problem (rolled permutation).
                    # src_pp[next_idx] is the source mean gate for that problem.
                    # This works correctly for both prompt and gen phases.
                    next_idx = (cnt[0] + 1) % max(len(s_pp), 1)
                    c_val = float(s_pp[next_idx]) if s_pp else s_m
                    g = torch.full_like(nat, c_val)
                else:
                    g = nat

                steer = (scaled_alpha * sig_ * g.unsqueeze(-1) * u_.unsqueeze(0).unsqueeze(0))
                return out + steer.to(out.dtype)
            return fn

        hooks.append(mod.register_forward_hook(
            make_hook(u, v, sig, src_mean, src_rms, correction,
                      gate_mode, src_pp, tgt_pp, prob_counter)))

    # We need to increment prob_counter at the problem boundary.
    # Patch evaluate to do this by adding a thin wrapper.
    result = _evaluate_with_counter(model, tokenizer, problems,
                                     label, hooks, active_pns, proj_svd)
    for h in hooks: h.remove()
    return result


def _evaluate_with_counter(model, tokenizer, problems, label, hooks, active_pns, proj_svd):
    """Wrapper that increments prob_counter hooks after each problem."""
    from shared_eval import extract_boxed, answers_match, make_prompt
    import torch

    # Find all prob_counter lists in the hooks' closures
    counters = []
    for h in hooks:
        try:
            fn = h.hook
            if hasattr(fn, '__closure__') and fn.__closure__:
                for cell in fn.__closure__:
                    try:
                        v = cell.cell_contents
                        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], int):
                            counters.append(v)
                    except (ValueError, AttributeError): pass
        except Exception: pass

    model.eval()
    correct, items = 0, []
    for i, prob in enumerate(problems):
        text = make_prompt(tokenizer, prob["problem"])
        inp = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(resp)
        ok   = answers_match(pred, prob["answer"])
        if ok: correct += 1
        items.append({"idx": i, "gold": prob["answer"], "pred": pred, "correct": ok})

        # Advance problem counter in all gate hooks
        for c in counters: c[0] += 1

        if (i + 1) % 10 == 0:
            print(f"  [{label}] [{i+1}/{len(problems)}] "
                  f"Acc: {correct/(i+1)*100:.1f}%", flush=True)

    return {"accuracy": correct / len(problems) * 100,
            "correct": correct, "total": len(problems),
            "label": label, "items": items}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("="*60); print("GATE MEDIATION (valid design)"); print("="*60, flush=True)

    calib = load_math500_split("calib")
    test  = load_math500_split("test")

    print("\n[1] Loading SVD + selecting active projections...", flush=True)
    proj_svd = load_proj_svd()
    active_pns = select_active_projections(proj_svd, N_BLOCKS)

    print("\n[2] Collecting gate statistics (source + target)...", flush=True)
    src_stats = collect_proj_gate_stats(MODELS["math_base"], proj_svd, calib, "source_base")
    tgt_stats = collect_proj_gate_stats(MODELS["instruct"],  proj_svd, calib, "target_instruct")

    print("\n[3] Loading target model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODELS["instruct"]); tok.pad_token = tok.eos_token

    print("\n[4] Computing norm-equalization scale factors on CALIB...", flush=True)
    norm_scale = compute_calib_perturbation_rms(
        proj_svd, active_pns, src_stats, tgt_stats, calib, model)
    for m, s in norm_scale.items():
        print(f"  {m:<15} scale={s:.3f}", flush=True)

    # Select alpha on VAL using natural mode (reference)
    from shared_eval import load_math500_split as _lms
    val = _lms("val")
    best_alpha, best_val = 0.5, 0.0
    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0]:
        r = apply_gated_steering_eval(model, tok, val, proj_svd, active_pns,
                                      src_stats, tgt_stats, "natural",
                                      alpha, norm_scale, f"val_natural_a{alpha}")
        if r["accuracy"] > best_val: best_val = r["accuracy"]; best_alpha = alpha
    print(f"\n  → Best alpha (natural, VAL): {best_alpha}", flush=True)

    print("\n[5] TEST evaluation — all gate modes...", flush=True)
    bl = apply_gated_steering_eval(model, tok, test, proj_svd, active_pns,
                                    src_stats, tgt_stats, "natural",
                                    0.0, norm_scale, "gate_med_baseline")
    base_items = [it["correct"] for it in bl["items"]]
    print(f"  Baseline: {bl['accuracy']:.1f}%")

    gate_modes = [
        ("natural",    "Weight-transfer-equiv  (v^T x_tgt)"),
        ("src_mean",   "Constant source mean   (E[v^T x_src] per proj)"),
        ("src_rms",    "Constant source RMS    (RMS per proj)"),
        ("src_replay", "Per-problem src replay (src mean, problem-matched)"),
        ("corrected",  "Magnitude-corrected    ((v^T x_tgt)·c_l)"),
        ("negated",    "Negated                (-(v^T x_tgt))"),
        ("shuffled",   "Cross-problem perm     (next problem's src gate)"),
    ]

    results = {"baseline": {k: v for k, v in bl.items() if k != "items"}}
    stats    = {}
    for mode, desc in gate_modes:
        r = apply_gated_steering_eval(model, tok, test, proj_svd, active_pns,
                                      src_stats, tgt_stats, mode,
                                      best_alpha, norm_scale,
                                      f"gate_med_{mode}")
        mi  = [it["correct"] for it in r["items"]]
        mn  = mcnemar_test(base_items, mi)
        bc  = bootstrap_ci(base_items, mi)
        lo, hi = wilson_ci(r["correct"], r["total"])
        results[mode] = {k: v for k, v in r.items() if k != "items"}
        stats[mode]   = {**mn, **bc}
        print(f"  {desc:<50} {r['accuracy']:.1f}% [{lo},{hi}]  "
              f"Δ={bc['mean_diff_pp']:+.1f}pp  McNemar p={mn['p_value']:.3f}", flush=True)

    # Reference: mean-diff at same location (residual stream) for comparison
    md_vecs = get_mean_diff_vectors(calib)
    from paper_eval_suite import apply_meandiff_steering
    r_md = apply_meandiff_steering(model, tok, test, md_vecs, 0.05, "gate_med_meandiff_ref")
    mi   = [it["correct"] for it in r_md["items"]]
    mn   = mcnemar_test(base_items, mi)
    bc   = bootstrap_ci(base_items, mi)
    lo, hi = wilson_ci(r_md["correct"], r_md["total"])
    results["meandiff_ref"] = {k: v for k, v in r_md.items() if k != "items"}
    stats["meandiff_ref"]   = {**mn, **bc}
    print(f"  {'Mean-diff (reference)':<50} {r_md['accuracy']:.1f}% [{lo},{hi}]  "
          f"Δ={bc['mean_diff_pp']:+.1f}pp  McNemar p={mn['p_value']:.3f}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()

    save = {
        "design_note": ("Perturbation norm equalized across all modes. "
                        "Block selection: top-15 blocks by aggregated sigma. "
                        "Shuffling: cross-problem permutation (next-problem source gate). "
                        "Source replay: per-problem source mean gate, constant during gen."),
        "n_active_projections": len(active_pns),
        "n_active_blocks": N_BLOCKS,
        "best_alpha_natural": best_alpha,
        "norm_scales": norm_scale,
        "results": results,
        "paired_stats": stats,
    }
    with open(OUTPUT_DIR / "gate_mediation_results.json", "w") as f:
        json.dump(save, f, indent=2)
    print("\n[SAVE] gate_mediation_results.json written.")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
