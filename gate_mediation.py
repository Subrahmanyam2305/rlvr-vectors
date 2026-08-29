"""
Gate Mediation Experiment — direct causal test of the gate-mismatch hypothesis.

Design: hold u, sigma, location, active layers, AND perturbation amplitude fixed.
Vary only gate g(x) across 7 conditions:
  1. natural       — v^T x_tgt  (what rank-1 weight transfer does)
  2. src_mean      — constant = mean(v^T x_src) per projection (global constant control)
  3. src_rms       — constant = RMS(v^T x_src) per projection (RMS-constant control)
  4. src_replay    — per-problem source mean gate on the SAME test prompts (oracle)
  5. corrected     — (v^T x_tgt) * c_l  where c_l = rms_src/rms_tgt (magnitude-corrected)
  6. negated       — -(v^T x_tgt)
  7. shuffled      — problem i uses src_replay[perm[i]] (permuted src_replay values only)

Amplitude equalization: each mode's alpha is scaled so the mean |sigma * g| across
active projections on calibration prompts equals that of the natural mode. Described
honestly as "calibration-matched mean perturbation amplitude" (not RMS energy).

src_replay and shuffled both index into test-prompt source gates, so:
  - src_replay[i] = source gate for test problem i (correctly matched)
  - shuffled[i]   = source gate for test problem perm[i] (same values, permuted)
This isolates the effect of problem correspondence while keeping all other factors
(gate type, amplitude, ...) identical between src_replay and shuffled.

Block selection: rank transformer blocks by total sigma (sum over o_proj + down_proj),
select top N_BLOCKS blocks, include BOTH projections from each selected block.

Oracle/transductive note: collecting source gates on test prompts uses test-prompt
inputs (but NOT test labels). This is labelled "oracle" in the output.
"""

import torch, json, gc, os
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from shared_eval import (
    load_math500_split, mcnemar_test, bootstrap_ci,
    wilson_ci, make_prompt, extract_boxed, answers_match, OUTPUT_DIR
)
from paper_eval_suite import MODELS, get_mean_diff_vectors, apply_meandiff_steering

TARGET_SUFFIXES = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
N_BLOCKS = 15   # top transformer blocks by aggregated sigma
PERM_SEED = 42  # fixed seed for shuffled permutation


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
    For each projection and each problem, compute mean gate over prompt tokens.
    Returns {pn: {"mean": float, "rms": float, "per_prob": [float, ...]}}.
    """
    print(f"  [{label}] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_id); tok.pad_token = tok.eos_token
    model.eval()

    target_mods = {pn[:-len(".weight")]: pn for pn in proj_svd}
    per_prob   = {pn: [] for pn in proj_svd}
    all_tokens = {pn: [] for pn in proj_svd}

    def make_hook(pn):
        v_cpu = proj_svd[pn]["v"].float()
        def fn(mod, inp, out):
            x = inp[0].detach().float().cpu()
            v = v_cpu.to(x.device)
            gates = (x[0] @ v).tolist()              # (seq,)
            per_prob[pn].append(float(np.mean(gates)))
            all_tokens[pn].extend(gates)
        return fn

    hooks = []
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
            "mean":     float(g.mean()),
            "rms":      float(np.sqrt((g ** 2).mean())),
            "per_prob": per_prob[pn],
        }
    print(f"  [{label}] Done ({len(stats)} projections).", flush=True)
    return stats


# ── Calibration-matched amplitude scaling ────────────────────────────────────
def compute_mean_abs_amplitude(proj_svd, active_pns, src_calib, tgt_calib):
    """
    Compute mean |sigma * g| across active projections on calibration problems,
    for each gate mode. Used to scale alpha so all modes deliver the same
    calibration-matched mean perturbation amplitude as the natural mode.

    NOTE: this uses prompt-level per-problem mean gates (not token-level).
    Described honestly as "mean perturbation amplitude" in the saved output.
    """
    amp = {}
    for mode in ["natural", "src_mean", "src_rms", "src_replay",
                 "corrected", "negated", "shuffled"]:
        vals = []
        for pn in active_pns:
            if pn not in src_calib or pn not in tgt_calib: continue
            sc  = src_calib[pn]
            tc  = tgt_calib[pn]
            sig = proj_svd[pn]["sigma"]
            tgt_g = np.array(tc["per_prob"])
            src_g = np.array(sc["per_prob"])
            n = len(tgt_g)
            rng = np.random.default_rng(PERM_SEED)
            perm = rng.permutation(n)

            if mode == "natural":
                g = np.abs(tgt_g)
            elif mode == "src_mean":
                g = np.abs(np.full(n, sc["mean"]))
            elif mode == "src_rms":
                g = np.abs(np.full(n, sc["rms"]))
            elif mode == "src_replay":
                g = np.abs(src_g)
            elif mode == "corrected":
                c = sc["rms"] / (tc["rms"] + 1e-8)
                g = np.abs(tgt_g * c)
            elif mode == "negated":
                g = np.abs(tgt_g)           # same magnitude as natural
            elif mode == "shuffled":
                g = np.abs(src_g[perm])     # permuted src_replay values
            vals.append(sig * g.mean())

        amp[mode] = float(np.mean(vals)) if vals else 1.0

    ref = amp.get("natural", 1.0)
    scale = {m: ref / (v + 1e-8) for m, v in amp.items()}
    return scale, amp


# ── Gated steering evaluation ─────────────────────────────────────────────────
def run_gated_eval(model, tokenizer, problems, proj_svd, active_pns,
                   src_test_stats, tgt_calib_stats, src_calib_stats,
                   gate_mode, alpha, amp_scale, test_perm, label):
    """
    Evaluate with gated steering on `problems`. A single shared prob_idx list
    is set to the current problem index before each inference call — no closure
    introspection needed, counters are always correct.

    src_test_stats: source gates collected on the test prompts (oracle access,
                    no labels used). Used for src_replay and shuffled.
    test_perm:      fixed permutation of range(len(problems)) for shuffled mode.
    """
    scaled_alpha = alpha * amp_scale.get(gate_mode, 1.0)
    prob_idx = [0]   # shared mutable counter; set to i before each problem

    hooks = []
    for pn in active_pns:
        if pn not in proj_svd: continue
        mod_name = pn[:-len(".weight")]
        mod = dict(model.named_modules()).get(mod_name)
        if not mod: continue

        data  = proj_svd[pn]
        u     = data["u"].to(model.device, dtype=model.dtype)
        v_cpu = data["v"].float()
        sig   = data["sigma"]

        sc = src_calib_stats.get(pn, {"mean": 0.0, "rms": 1.0})
        tc = tgt_calib_stats.get(pn, {"mean": 0.0, "rms": 1.0})
        st = src_test_stats.get(pn, {"per_prob": []})

        src_mean_val  = float(sc["mean"])
        src_rms_val   = float(sc["rms"])
        correction    = src_rms_val / (float(tc["rms"]) + 1e-8)
        src_test_pp   = st["per_prob"]     # len = n_test
        mode          = gate_mode

        def make_hook(u_, v_c, sig_, s_m, s_rms_, corr, m, s_test_pp, perm, pidx):
            def fn(module, inp, out):
                x   = inp[0]                                        # (B, seq, d_in)
                v_d = v_c.to(x.device, dtype=x.dtype)
                nat = (x @ v_d.unsqueeze(-1)).squeeze(-1)           # (B, seq)
                i   = pidx[0]

                if m == "natural":
                    g = nat
                elif m == "src_mean":
                    g = torch.full_like(nat, s_m)
                elif m == "src_rms":
                    g = torch.full_like(nat, s_rms_)
                elif m == "src_replay":
                    # Per-problem source gate for test problem i (oracle, no labels)
                    c_val = float(s_test_pp[i]) if i < len(s_test_pp) else s_m
                    g = torch.full_like(nat, c_val)
                elif m == "corrected":
                    g = nat * corr
                elif m == "negated":
                    g = -nat
                elif m == "shuffled":
                    # Use src_replay value for permuted problem — same values as
                    # src_replay, only problem correspondence changed
                    j = int(perm[i]) if i < len(perm) else i
                    c_val = float(s_test_pp[j]) if j < len(s_test_pp) else s_m
                    g = torch.full_like(nat, c_val)
                else:
                    g = nat

                steer = scaled_alpha * sig_ * g.unsqueeze(-1) * u_.unsqueeze(0).unsqueeze(0)
                return out + steer.to(out.dtype)
            return fn

        hooks.append(mod.register_forward_hook(
            make_hook(u, v_cpu, sig, src_mean_val, src_rms_val, correction,
                      mode, src_test_pp, test_perm, prob_idx)))

    model.eval()
    correct, items = 0, []
    for i, prob in enumerate(problems):
        prob_idx[0] = i          # set BEFORE inference so all hooks see the right index
        text = make_prompt(tokenizer, prob["problem"])
        inp  = tokenizer(text, return_tensors="pt",
                         truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(resp)
        ok   = answers_match(pred, prob["answer"])
        if ok: correct += 1
        items.append({"idx": i, "gold": prob["answer"], "pred": pred, "correct": ok})
        if (i + 1) % 10 == 0:
            print(f"  [{label}] [{i+1}/{len(problems)}] "
                  f"Acc: {correct/(i+1)*100:.1f}%", flush=True)

    for h in hooks: h.remove()
    return {"accuracy": correct / len(problems) * 100,
            "correct": correct, "total": len(problems),
            "label": label, "items": items}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GATE MEDIATION (valid design)")
    print("=" * 60, flush=True)

    calib = load_math500_split("calib")
    val   = load_math500_split("val")
    test  = load_math500_split("test")
    n_test = len(test)

    print("\n[1] Loading SVD + selecting active projections...", flush=True)
    proj_svd   = load_proj_svd()
    active_pns = select_active_projections(proj_svd, N_BLOCKS)

    print("\n[2] Collecting calibration gate stats (source + target)...", flush=True)
    src_calib = collect_proj_gate_stats(MODELS["math_base"], proj_svd, calib, "src_calib")
    tgt_calib = collect_proj_gate_stats(MODELS["instruct"],  proj_svd, calib, "tgt_calib")

    # Oracle: collect source gates on test prompts (no labels used)
    print("\n[3] Collecting source gates on test prompts (oracle — no labels used)...", flush=True)
    src_test = collect_proj_gate_stats(MODELS["math_base"], proj_svd, test, "src_test_oracle")

    # Fixed permutation for shuffled mode (same values as src_replay, different problem mapping)
    rng       = np.random.default_rng(PERM_SEED)
    test_perm = rng.permutation(n_test)   # shape (n_test,)

    print("\n[4] Computing calibration-matched mean perturbation amplitude...", flush=True)
    amp_scale, raw_amp = compute_mean_abs_amplitude(proj_svd, active_pns, src_calib, tgt_calib)
    for m in sorted(amp_scale):
        print(f"  {m:<15} raw_amp={raw_amp[m]:.4f}  scale={amp_scale[m]:.3f}", flush=True)

    print("\n[5] Loading target model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODELS["instruct"]); tok.pad_token = tok.eos_token

    # Select alpha on VAL using natural mode
    best_alpha, best_val_acc = 0.5, 0.0
    print("\n[6] Alpha selection on VAL (natural mode)...", flush=True)
    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0]:
        r = run_gated_eval(model, tok, val, proj_svd, active_pns,
                           src_test, tgt_calib, src_calib,
                           "natural", alpha, amp_scale, test_perm,
                           f"val_natural_a{alpha}")
        print(f"  alpha={alpha}  val_acc={r['accuracy']:.1f}%", flush=True)
        if r["accuracy"] > best_val_acc:
            best_val_acc = r["accuracy"]; best_alpha = alpha
    print(f"  → Best alpha: {best_alpha}", flush=True)

    print("\n[7] Baseline (alpha=0) on TEST...", flush=True)
    bl = run_gated_eval(model, tok, test, proj_svd, active_pns,
                        src_test, tgt_calib, src_calib,
                        "natural", 0.0, amp_scale, test_perm, "baseline")
    base_correct = [it["correct"] for it in bl["items"]]
    print(f"  Baseline: {bl['accuracy']:.1f}%", flush=True)

    print("\n[8] TEST evaluation — all gate modes...", flush=True)
    gate_modes = [
        ("natural",    "Weight-transfer-equiv  (v^T x_tgt)"),
        ("src_mean",   "Global-constant src    (mean(v^T x_src))"),
        ("src_rms",    "Global-constant RMS    (rms(v^T x_src))"),
        ("src_replay", "Per-problem src oracle (src gate, test-matched)"),
        ("corrected",  "Magnitude-corrected    ((v^T x_tgt)·c_l)"),
        ("negated",    "Negated                (-(v^T x_tgt))"),
        ("shuffled",   "Shuffled src oracle    (src_replay[perm[i]])"),
    ]

    results = {"baseline": {k: v for k, v in bl.items() if k != "items"}}
    stats = {}
    for mode, desc in gate_modes:
        r = run_gated_eval(model, tok, test, proj_svd, active_pns,
                           src_test, tgt_calib, src_calib,
                           mode, best_alpha, amp_scale, test_perm,
                           f"gate_{mode}")
        mi     = [it["correct"] for it in r["items"]]
        mn     = mcnemar_test(base_correct, mi)
        bc     = bootstrap_ci(base_correct, mi)
        lo, hi = wilson_ci(r["correct"], r["total"])
        results[mode] = {k: v for k, v in r.items() if k != "items"}
        results[mode]["item_correct"] = mi
        stats[mode]   = {**mn, **bc}
        print(f"  {desc:<52} {r['accuracy']:.1f}% CI[{lo:.3f},{hi:.3f}]  "
              f"Δ={bc['mean_diff_pp']:+.1f}pp  p={mn['p_value']:.3f}", flush=True)

    # Reference: mean-diff steering at same alpha
    print("\n[9] Mean-diff reference...", flush=True)
    md_vecs = get_mean_diff_vectors(calib)
    r_md   = apply_meandiff_steering(model, tok, test, md_vecs, 0.05, "meandiff_ref")
    mi     = [it["correct"] for it in r_md["items"]]
    mn     = mcnemar_test(base_correct, mi)
    bc     = bootstrap_ci(base_correct, mi)
    lo, hi = wilson_ci(r_md["correct"], r_md["total"])
    results["meandiff_ref"] = {k: v for k, v in r_md.items() if k != "items"}
    results["meandiff_ref"]["item_correct"] = mi
    stats["meandiff_ref"]   = {**mn, **bc}
    print(f"  {'Mean-diff (reference)':<52} {r_md['accuracy']:.1f}% CI[{lo:.3f},{hi:.3f}]  "
          f"Δ={bc['mean_diff_pp']:+.1f}pp  p={mn['p_value']:.3f}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()

    # ── Causal interpretation note ────────────────────────────────────────────
    # src_replay vs shuffled is the cleanest causal contrast: same gate values,
    # same amplitude normalization, only problem correspondence differs.
    # src_replay > shuffled => gate SOURCE matters, not just gate magnitude.
    # Results should be interpreted within the selected 15-block rank-1 subspace only.
    save = {
        "design_note": (
            "Perturbation amplitude equalized via calibration-matched mean |sigma*g| "
            "(not token-level RMS; prompt-level per-problem means used for calibration). "
            "src_replay and shuffled both use test-prompt source gates (oracle, no labels). "
            "shuffled[i] = src_replay[perm[i]]: same values, permuted — isolates "
            "problem correspondence. Block selection: top-15 blocks by aggregated sigma. "
            "Causal claims limited to the 15-block rank-1 intervention subspace."
        ),
        "n_active_projections": len(active_pns),
        "n_active_blocks":      N_BLOCKS,
        "best_alpha_natural":   best_alpha,
        "amplitude_scales":     amp_scale,
        "raw_amplitudes":       raw_amp,
        "perm_seed":            PERM_SEED,
        "results":              results,
        "paired_stats":         stats,
    }
    out_path = OUTPUT_DIR / "gate_mediation_results.json"
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n[SAVE] {out_path}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
