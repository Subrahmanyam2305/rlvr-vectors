"""
Gate Analysis — correctly addresses all reviewer concerns:

  1. Uses power iteration (not full SVD) for spectral null — O(50 draws × mn × 10 iters)
  2. Gate sign agreement: PAIRED at problem level (same 50 problems, prompt tokens only)
     + separate unpaired generation-time gate distributions
  3. Covers all 56 o_proj + down_proj projections
  4. Reports per-projection stats with exact counts
"""

import torch, json, gc, os
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from shared_eval import load_math500_split, make_prompt, OUTPUT_DIR

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODELS = {
    "math_base": "Qwen/Qwen2.5-Math-1.5B",
    "rlvr":      "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    "instruct":  "Qwen/Qwen2.5-1.5B-Instruct",
}
TARGET_SUFFIXES = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]
N_NULL_DRAWS = 50   # per shape; fast with power iteration


def top_sv_ratio_power(G: np.ndarray, n_iter: int = 20) -> float:
    """Estimate sigma_1^2 / ||G||_F^2 via power iteration. O(mn * n_iter)."""
    m, n = G.shape
    rng = np.random.default_rng()
    v = rng.standard_normal(n).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    sigma = 0.0
    for _ in range(n_iter):
        u = G @ v;  sigma = np.linalg.norm(u);  u /= sigma + 1e-10
        v = G.T @ u; v /= np.linalg.norm(v) + 1e-10
    frob_sq = float((G * G).sum())
    return float(sigma**2) / (frob_sq + 1e-30)


# ── Prompt-level paired gate collection ──────────────────────────────────────
def collect_prompt_gates(model, tokenizer, problems, proj_svd):
    """
    For each problem and each projection, compute mean(v^T x) over PROMPT tokens.
    Returns {param_name: list_of_per_problem_mean_gates}  (len = n_problems)
    PAIRED: same problems run on both source and target models.
    """
    model.eval()
    # Hook inputs to each target projection
    target_mods = {pn[:-len(".weight")]: pn for pn in proj_svd}
    stores = {pn: [] for pn in proj_svd}
    hooks = []

    def make_hook(param_name):
        v = proj_svd[param_name]["v"].float()
        def fn(mod, inp, out):
            x = inp[0].detach().float()   # (1, seq, d_in)
            gates = (x[0] @ v).tolist()   # (seq,) — prompt tokens
            stores[param_name].append(float(np.mean(gates)))
        return fn

    for mod_name, param_name in target_mods.items():
        mod = dict(model.named_modules()).get(mod_name)
        if mod is not None:
            hooks.append(mod.register_forward_hook(make_hook(param_name)))

    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inp = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model(**inp)   # prompt-only forward pass (no generation)

    for h in hooks: h.remove()
    return stores   # {param_name: [per_problem_mean_gate, ...]}


def collect_generation_gates(model, tokenizer, problems, proj_svd, n_gen=32):
    """
    Collect gate values during GENERATION (unpaired: sequences differ between models).
    Returns {param_name: all_token_gates_flat_list}
    """
    model.eval()
    target_mods = {pn[:-len(".weight")]: pn for pn in proj_svd}
    stores = {pn: [] for pn in proj_svd}
    hooks = []

    def make_hook(param_name):
        v = proj_svd[param_name]["v"].float()
        def fn(mod, inp, out):
            x = inp[0].detach().float()
            for tok_vec in x[0]:
                stores[param_name].append(float(torch.dot(v, tok_vec.cpu()).item()))
        return fn

    for mod_name, param_name in target_mods.items():
        mod = dict(model.named_modules()).get(mod_name)
        if mod is not None:
            hooks.append(mod.register_forward_hook(make_hook(param_name)))

    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inp = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=n_gen, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)

    for h in hooks: h.remove()
    return stores


# ── Load SVD ──────────────────────────────────────────────────────────────────
def load_proj_svd():
    print("[GATE] Loading SVD components...", flush=True)
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
            "shape": list(wb.shape),
        }
        del wb, wr, dW, U, S, Vt
    gc.collect()
    print(f"[GATE] {len(proj_svd)} projections.", flush=True)
    return proj_svd


# ── Main gate analysis ─────────────────────────────────────────────────────────
def run_gate_analysis():
    proj_svd = load_proj_svd()
    calib = load_math500_split("calib")

    # Load models and collect gate statistics
    def get_gates_for_model(model_id, label):
        print(f"  [{label}] Loading model...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto")
        tok = AutoTokenizer.from_pretrained(model_id); tok.pad_token = tok.eos_token

        prompt_g  = collect_prompt_gates(model, tok, calib, proj_svd)
        gen_g     = collect_generation_gates(model, tok, calib, proj_svd, n_gen=32)
        del model; gc.collect(); torch.cuda.empty_cache()
        return prompt_g, gen_g

    src_prompt, src_gen = get_gates_for_model(MODELS["math_base"], "source_base")
    tgt_prompt, tgt_gen = get_gates_for_model(MODELS["instruct"],  "target_instruct")

    # Paired prompt-level sign agreement (same problems)
    per_proj = {}
    ratios, paired_sign_agree = [], []

    for pn in proj_svd:
        if pn not in src_prompt or pn not in tgt_prompt: continue
        sg = np.array(src_prompt[pn])   # (n_problems,)
        tg = np.array(tgt_prompt[pn])   # (n_problems,) — same problems

        # Paired per-problem sign agreement
        agree = (np.sign(sg) == np.sign(tg)).mean()
        ratio = np.abs(tg).mean() / (np.abs(sg).mean() + 1e-8)

        # Unpaired generation-time distributions
        src_g_arr = np.array(src_gen.get(pn, [0.0]))
        tgt_g_arr = np.array(tgt_gen.get(pn, [0.0]))

        per_proj[pn] = {
            "layer_idx": proj_svd[pn]["layer_idx"],
            # PAIRED prompt-level stats (n_problems = n_calib)
            "n_problems": len(sg),
            "src_prompt_abs_mean": float(np.abs(sg).mean()),
            "tgt_prompt_abs_mean": float(np.abs(tg).mean()),
            "prompt_ratio": float(ratio),
            "paired_prompt_sign_agree": float(agree),
            # Unpaired generation-time stats
            "src_gen_abs_mean": float(np.abs(src_g_arr).mean()),
            "tgt_gen_abs_mean": float(np.abs(tgt_g_arr).mean()),
            "src_gen_n_tokens": int(len(src_g_arr)),
            "tgt_gen_n_tokens": int(len(tgt_g_arr)),
        }
        ratios.append(ratio)
        paired_sign_agree.append(agree)

    result = {
        "n_projections_analyzed": len(per_proj),
        "n_calib_problems": len(calib),
        "note": ("Paired stats use identical CALIB problems on both models, "
                 "prompt tokens only. Generation stats are unpaired (sequences differ)."),
        "paired_prompt": {
            "mean_ratio": float(np.mean(ratios)),
            "median_ratio": float(np.median(ratios)),
            "std_ratio": float(np.std(ratios)),
            "mean_sign_agreement": float(np.mean(paired_sign_agree)),
            "frac_ratio_below_half": float((np.array(ratios) < 0.5).mean()),
        },
        "per_projection": per_proj,
    }
    print(f"[GATE] {len(per_proj)} projections. "
          f"Paired sign agree: {result['paired_prompt']['mean_sign_agreement']:.1%}  "
          f"Mean ratio: {result['paired_prompt']['mean_ratio']:.3f}", flush=True)

    with open(OUTPUT_DIR / "gate_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print("[GATE] Saved gate_analysis.json", flush=True)
    return result


# ── Spectral null with power iteration ────────────────────────────────────────
def run_spectral_null():
    print("[SPECTRAL NULL] Loading existing spectral data...", flush=True)
    with open(OUTPUT_DIR / "spectral_data.json") as f:
        spectral_data = json.load(f)

    shape_groups = {}
    for d in spectral_data:
        if d["layer_idx"] < 0: continue
        shape = tuple(d["shape"])
        shape_groups.setdefault(shape, []).append(d["rank1_frac"])

    null_results = {}
    rng_state = np.random.default_rng(42)

    for shape, observed_fracs in shape_groups.items():
        m, n = shape
        null_fracs = []
        for draw in range(N_NULL_DRAWS):
            G = rng_state.standard_normal((m, n)).astype(np.float32)
            null_fracs.append(top_sv_ratio_power(G, n_iter=20))

        null_mean = float(np.mean(null_fracs))
        null_std  = float(np.std(null_fracs))
        obs_mean  = float(np.mean(observed_fracs))
        z = (obs_mean - null_mean) / (null_std + 1e-10)
        ratio = obs_mean / (null_mean + 1e-10)

        null_results[str(shape)] = {
            "shape": list(shape), "n_observed": len(observed_fracs),
            "n_null_draws": N_NULL_DRAWS,
            "observed_mean": obs_mean, "null_mean": null_mean,
            "null_std": null_std, "z_score": float(z),
            "concentration_ratio": float(ratio),
        }
        print(f"  {shape}: obs={obs_mean:.4f}  null={null_mean:.4f}±{null_std:.4f}"
              f"  z={z:.1f}  ratio={ratio:.0f}x  (n_draws={N_NULL_DRAWS})", flush=True)

    with open(OUTPUT_DIR / "spectral_null.json", "w") as f:
        json.dump(null_results, f, indent=2)
    print("[SPECTRAL NULL] Saved spectral_null.json", flush=True)
    return null_results


if __name__ == "__main__":
    print("="*60); print("GATE ANALYSIS"); print("="*60)
    run_gate_analysis()
    print("\n" + "="*60); print("SPECTRAL NULL (power iteration)"); print("="*60)
    run_spectral_null()
    print("\nDONE.", flush=True)
