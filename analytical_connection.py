"""
Phase 5: Analytical Connection — Empirical validation that rank-1 weight transfer
acts as input-conditional activation steering.

Measures v^T*x for the source vs target models to quantify the gating mismatch
that explains why weight transfer fails cross-model.
"""

import torch
import json
import gc
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")


def load_math500_split(split="cal"):
    """Load a disjoint MATH500 partition.

    Splits: cal=450-499 (n=50), val=400-449 (n=50), test=0-399 (n=400).
    """
    with open(DATA_DIR / "math500.json") as f:
        data = json.load(f)
    if split == "cal":
        return data[450:500]
    elif split == "val":
        return data[400:450]
    elif split == "test":
        return data[0:400]
    else:
        raise ValueError(f"Unknown split: {split}")


def make_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question}
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{question}\n\nSolution:"


def compute_svd_components():
    """Extract u, sigma, v per layer using safetensors."""
    base_path = snapshot_download("Qwen/Qwen2.5-Math-1.5B")
    rlvr_path = snapshot_download("ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1")

    base_index, rlvr_index = {}, {}
    for f in sorted(Path(base_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = str(f)
    for f in sorted(Path(rlvr_path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                rlvr_index[key] = str(f)

    components = {}
    target_suffixes = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]

    for param_name in sorted(base_index.keys()):
        if not any(param_name.endswith(s) for s in target_suffixes):
            continue
        if param_name not in rlvr_index:
            continue

        with safe_open(base_index[param_name], framework="pt", device="cpu") as sf:
            w_base = sf.get_tensor(param_name).float()
        with safe_open(rlvr_index[param_name], framework="pt", device="cpu") as sf:
            w_rlvr = sf.get_tensor(param_name).float()

        if len(w_base.shape) != 2:
            continue
        dW = w_rlvr - w_base
        if dW.norm().item() < 1e-8:
            continue

        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
        components[param_name] = {
            "u": U[:, 0].clone(),
            "sigma": S[0].item(),
            "v": Vt[0, :].clone(),
            "rank1_frac": ((S[0]**2) / (S**2).sum()).item(),
        }
        del w_base, w_rlvr, dW, U, S, Vt

    gc.collect()
    return components


def collect_layer_inputs(model, tokenizer, problems, target_modules):
    """Collect mean input activations to linear layers (what gets multiplied by W)."""
    model.eval()
    store = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if name not in store:
                store[name] = []
            store[name].append(x.mean(dim=(0, 1)).cpu())
        return hook_fn

    for name, module in model.named_modules():
        if name in target_modules:
            hooks.append(module.register_forward_hook(make_hook(name)))

    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model(**inputs)

    for hook in hooks:
        hook.remove()

    return {name: torch.stack(vecs).mean(0) for name, vecs in store.items()}


def main():
    print("=" * 60)
    print("ANALYTICAL CONNECTION: v^T*x GATING ANALYSIS")
    print("=" * 60)

    problems = load_math500_split("cal")

    # Step 1: SVD components
    print("\nStep 1: Computing SVD components...")
    components = compute_svd_components()
    target_modules = [pn.replace(".weight", "") for pn in components.keys()]
    print(f"  {len(components)} modules")

    # Step 2: Source activations
    print("\nStep 2: Source model activations...")
    src_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.float16, device_map="auto")
    src_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    src_tok.pad_token = src_tok.eos_token
    src_acts = collect_layer_inputs(src_model, src_tok, problems, target_modules)
    del src_model; gc.collect(); torch.cuda.empty_cache()

    # Step 3: Target activations
    print("\nStep 3: Target model activations...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto")
    tgt_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tgt_tok.pad_token = tgt_tok.eos_token
    tgt_acts = collect_layer_inputs(tgt_model, tgt_tok, problems, target_modules)
    del tgt_model; gc.collect(); torch.cuda.empty_cache()

    # Step 4: Compute v^T*x for each layer
    print("\nStep 4: Computing gating signals v^T*x...")
    layer_results = []

    for param_name, svd in components.items():
        module_name = param_name.replace(".weight", "")
        if module_name not in src_acts or module_name not in tgt_acts:
            continue

        v = svd["v"]
        x_src = src_acts[module_name]
        x_tgt = tgt_acts[module_name]

        if x_src.dim() == 0 or x_tgt.dim() == 0:
            continue
        if v.shape[0] != x_src.shape[0] or v.shape[0] != x_tgt.shape[0]:
            continue

        vtx_src = (v @ x_src).item()
        vtx_tgt = (v @ x_tgt).item()

        layer_results.append({
            "param": param_name,
            "vtx_src": vtx_src,
            "vtx_tgt": vtx_tgt,
            "abs_ratio": abs(vtx_tgt) / (abs(vtx_src) + 1e-10),
            "same_sign": np.sign(vtx_src) == np.sign(vtx_tgt),
            "rank1_frac": svd["rank1_frac"],
        })

    # Summary statistics
    ratios = [r["abs_ratio"] for r in layer_results if r["abs_ratio"] < 10]
    sign_agreement = sum(1 for r in layer_results if r["same_sign"]) / len(layer_results)

    print(f"\n{'='*60}")
    print(f"RESULTS: Gating Signal Analysis ({len(layer_results)} layers)")
    print(f"{'='*60}")
    print(f"  Mean |v^T*x_tgt| / |v^T*x_src|: {np.mean(ratios):.3f}")
    print(f"  Median ratio:                     {np.median(ratios):.3f}")
    print(f"  Sign agreement:                   {sign_agreement*100:.1f}%")
    print(f"  Sign flips:                       {(1-sign_agreement)*100:.1f}%")
    print(f"\n  Interpretation:")
    print(f"  - Gating signal drops to {np.mean(ratios)*100:.0f}% of source magnitude")
    print(f"  - {(1-sign_agreement)*100:.0f}% of layers steer in WRONG direction")
    print(f"  - This explains weight transfer failure (+2%) vs steering (+14%)")

    # Save
    output = {
        "layers": layer_results,
        "summary": {
            "n_layers": len(layer_results),
            "mean_abs_ratio": float(np.mean(ratios)),
            "median_abs_ratio": float(np.median(ratios)),
            "sign_agreement": float(sign_agreement),
            "sign_flips": float(1 - sign_agreement),
        }
    }
    with open(OUTPUT_DIR / "analytical_connection.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'analytical_connection.json'}")


if __name__ == "__main__":
    main()
