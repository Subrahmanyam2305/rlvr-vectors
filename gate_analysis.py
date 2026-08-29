"""
Gate Analysis — regenerated cleanly to address reviewer concerns:

  1. Analyzes all o_proj and down_proj projections (not residual-stream only)
  2. Collects per-token gate values during GENERATION (not just prompt tokens)
  3. Reports per-projection and aggregate statistics with proper counts
  4. Generates shape-matched empirical null for the spectral concentration claim

Outputs:
  outputs/gate_analysis.json       — per-projection gating stats
  outputs/spectral_null.json       — empirical null distributions per shape
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


def collect_proj_inputs_with_generation(model, tokenizer, problems, target_mods,
                                        n_gen_tokens=32):
    """
    Collect input activations to target projections during BOTH prompt processing
    AND the first n_gen_tokens of generation (captures runtime gating).
    Returns {module_name: list_of_token_activation_tensors}
    """
    model.eval()
    stores = {}
    hooks = []

    def make_hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float()   # (1, seq_or_1, d_in)
            for tok_vec in x[0]:          # each token
                stores.setdefault(name, []).append(tok_vec.cpu())
        return fn

    for name, mod in model.named_modules():
        if name in target_mods:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inp = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            # Run generation so we capture actual generation-time activations
            model.generate(**inp, max_new_tokens=n_gen_tokens, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)

    for h in hooks: h.remove()
    return stores  # {name: [tok_vec, ...]}


def compute_gate_statistics(model_name, proj_svd, calib_problems, label):
    """For each projection, compute distribution of v^T x over all tokens/problems."""
    print(f"  Loading {label}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_name); tok.pad_token = tok.eos_token

    target_mods = {n[:-len(".weight")] for n in proj_svd}
    stores = collect_proj_inputs_with_generation(model, tok, calib_problems, target_mods)
    del model; gc.collect(); torch.cuda.empty_cache()

    gate_stats = {}
    for param_name, data in proj_svd.items():
        mod_name = param_name[:-len(".weight")]
        if mod_name not in stores: continue
        v = data["v"].float()  # (d_in,)
        token_vecs = stores[mod_name]  # list of (d_in,) tensors
        if not token_vecs: continue
        X = torch.stack(token_vecs)   # (N_tokens, d_in)
        gates = (X @ v).numpy()       # (N_tokens,)
        gate_stats[param_name] = {
            "n_tokens": len(gates),
            "mean":   float(gates.mean()),
            "std":    float(gates.std()),
            "abs_mean": float(np.abs(gates).mean()),
            "pos_frac": float((gates > 0).mean()),
        }
    return gate_stats


def run_gate_analysis():
    print("[GATE] Loading SVD components...", flush=True)
    base_path  = snapshot_download(MODELS["math_base"])
    rlvr_path  = snapshot_download(MODELS["rlvr"])

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
        layer_idx = int(pn.split(".")[2])
        proj_svd[pn] = {
            "u": U[:, 0].clone(), "sigma": S[0].item(),
            "v": Vt[0].clone(), "layer_idx": layer_idx,
        }
        del wb, wr, dW, U, S, Vt
    gc.collect()
    print(f"[GATE] {len(proj_svd)} projections with SVD.", flush=True)

    calib = load_math500_split("calib")

    src_gates  = compute_gate_statistics(MODELS["math_base"], proj_svd, calib, "source_base")
    tgt_gates  = compute_gate_statistics(MODELS["instruct"],  proj_svd, calib, "target_instruct")

    # Aggregate: ratio |gate_tgt| / |gate_src|, sign agreement
    ratios, sign_agree = [], []
    per_proj = {}
    for pn in proj_svd:
        if pn not in src_gates or pn not in tgt_gates: continue
        sg = src_gates[pn]; tg = tgt_gates[pn]
        if sg["abs_mean"] < 1e-6: continue
        ratio = tg["abs_mean"] / sg["abs_mean"]
        # sign agreement: both pos_frac compared — both > 0.5 or both < 0.5
        src_sign = 1 if sg["pos_frac"] > 0.5 else -1
        tgt_sign = 1 if tg["pos_frac"] > 0.5 else -1
        agree = (src_sign == tgt_sign)
        ratios.append(ratio)
        sign_agree.append(agree)
        per_proj[pn] = {
            "layer_idx": proj_svd[pn]["layer_idx"],
            "src_abs_mean": sg["abs_mean"], "src_n_tokens": sg["n_tokens"],
            "tgt_abs_mean": tg["abs_mean"], "tgt_n_tokens": tg["n_tokens"],
            "ratio": ratio, "sign_agree": agree,
        }

    result = {
        "n_projections_analyzed": len(per_proj),
        "n_source_tokens_avg": np.mean([v["src_n_tokens"] for v in per_proj.values()]),
        "n_target_tokens_avg": np.mean([v["tgt_n_tokens"] for v in per_proj.values()]),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "std_ratio": float(np.std(ratios)),
        "sign_agreement_frac": float(np.mean(sign_agree)),
        "per_projection": per_proj,
    }
    print(f"[GATE] {len(per_proj)} projections. "
          f"Mean ratio: {result['mean_ratio']:.3f}  "
          f"Sign agree: {result['sign_agreement_frac']:.1%}", flush=True)

    with open(OUTPUT_DIR / "gate_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[GATE] Saved gate_analysis.json", flush=True)
    return result


def run_spectral_null():
    """
    Generate shape-matched empirical null distributions for the spectral
    concentration claim (rank-1 fraction vs random Gaussian matrix).
    Uses actual observed shapes rather than assuming 1536x1536 square.
    """
    print("[SPECTRAL NULL] Loading existing spectral data...", flush=True)
    with open(OUTPUT_DIR / "spectral_data.json") as f:
        spectral_data = json.load(f)

    # Group shapes
    shape_groups = {}
    for d in spectral_data:
        if d["layer_idx"] < 0: continue
        shape = tuple(d["shape"])
        shape_groups.setdefault(shape, []).append(d["rank1_frac"])

    null_results = {}
    N_SAMPLES = 1000
    rng = np.random.default_rng(42)

    for shape, observed_fracs in shape_groups.items():
        m, n = shape
        if m * n > 4096 * 4096:  # skip very large matrices
            continue
        rank = min(m, n)

        # Generate null: rank-1 fraction for shape-matched Gaussian matrices
        null_fracs = []
        for _ in range(N_SAMPLES):
            G = rng.standard_normal((m, n)).astype(np.float32)
            # Only need singular values (not full SVD)
            sv = np.linalg.svd(G, compute_uv=False)
            frac = (sv[0]**2) / (sv**2).sum()
            null_fracs.append(float(frac))

        null_mean = float(np.mean(null_fracs))
        null_std  = float(np.std(null_fracs))
        obs_mean  = float(np.mean(observed_fracs))
        z_score   = (obs_mean - null_mean) / (null_std + 1e-10)

        null_results[str(shape)] = {
            "shape": list(shape),
            "n_observed": len(observed_fracs),
            "observed_mean": obs_mean,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z_score,
            "concentration_ratio": obs_mean / (null_mean + 1e-10),
        }
        print(f"  Shape {shape}: obs={obs_mean:.4f}  null={null_mean:.4f}±{null_std:.4f}  "
              f"z={z_score:.1f}  ratio={obs_mean/null_mean:.0f}x", flush=True)

    with open(OUTPUT_DIR / "spectral_null.json", "w") as f:
        json.dump(null_results, f, indent=2)
    print("[SPECTRAL NULL] Saved spectral_null.json", flush=True)
    return null_results


if __name__ == "__main__":
    print("="*60)
    print("GATE ANALYSIS")
    print("="*60)
    run_gate_analysis()

    print("\n" + "="*60)
    print("SPECTRAL NULL DISTRIBUTIONS")
    print("="*60)
    run_spectral_null()

    print("\nDONE.", flush=True)
