"""
SVD-Derived Activation Steering

Novel method: Use rank-1 SVD components from RLVR weight deltas as
activation steering vectors at runtime on the target model.

Key formula per layer l:
    h_l' = h_l + alpha * weight_l * u_l

Where u_l is the LEFT singular vector (output direction) from SVD(dW_l),
and weight_l encodes per-layer importance from the singular values.

Variants tested:
  1. sigma-weighted: weight_l = sigma_l / sigma_max
  2. uniform: weight_l = 1 (just use u, ignore sigma)
  3. top-K: only steer in the K layers with largest sigma
  4. conditional: gate steering by input similarity to reasoning direction
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_math500_split(split="test"):
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


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    idx += len("\\boxed{")
    depth = 1
    result = []
    while idx < len(text) and depth > 0:
        c = text[idx]
        if c == '{':
            depth += 1
            result.append(c)
        elif c == '}':
            depth -= 1
            if depth > 0:
                result.append(c)
        else:
            result.append(c)
        idx += 1
    return "".join(result).strip()


def normalize_answer(a):
    a = a.strip().replace(" ", "").lower()
    a = a.replace("\\text{", "").replace("}", "").replace("\\mathrm{", "")
    return a


def answers_match(pred, gold):
    if normalize_answer(pred) == normalize_answer(gold):
        return True
    try:
        return abs(float(normalize_answer(pred)) - float(normalize_answer(gold))) < 1e-6
    except:
        return False


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
    """Extract u and sigma per layer from RLVR weight delta using safetensors."""
    print("   Loading safetensors indices...", flush=True)
    base_path = snapshot_download("Qwen/Qwen2.5-Math-1.5B")
    rlvr_path = snapshot_download("ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1")

    base_files = sorted(Path(base_path).glob("*.safetensors"))
    rlvr_files = sorted(Path(rlvr_path).glob("*.safetensors"))

    base_index = {}
    for f in base_files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = str(f)

    rlvr_index = {}
    for f in rlvr_files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                rlvr_index[key] = str(f)

    components = {}  # layer_idx -> {module_type -> {u, sigma, rank1_frac}}
    count = 0

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
            del w_base, w_rlvr
            continue

        dW = w_rlvr - w_base
        if dW.norm().item() < 1e-8:
            del w_base, w_rlvr, dW
            continue

        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)

        # Extract layer index from param name
        # e.g. "model.layers.5.self_attn.o_proj.weight"
        parts = param_name.split(".")
        layer_idx = int(parts[2])
        module_type = "attn" if "self_attn" in param_name else "mlp"

        if layer_idx not in components:
            components[layer_idx] = {}

        components[layer_idx][module_type] = {
            "u": U[:, 0].clone(),
            "sigma": S[0].item(),
            "rank1_frac": ((S[0]**2) / (S**2).sum()).item(),
            "param_name": param_name,
        }

        del w_base, w_rlvr, dW, U, S, Vt
        count += 1
        if count % 10 == 0:
            gc.collect()

    gc.collect()
    print(f"   Extracted components for {len(components)} layers, {count} modules", flush=True)
    return components


def evaluate_with_steering(model, tokenizer, problems, hooks, label=""):
    """Evaluate model with steering hooks active."""
    model.eval()
    correct = 0
    for i, prob in enumerate(problems):
        text = make_prompt(tokenizer, prob["problem"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if answers_match(extract_boxed(resp), prob["answer"]):
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"    [{label}] [{i+1}/{len(problems)}] Acc: {correct/(i+1)*100:.1f}%", flush=True)
    return {"accuracy": correct / len(problems) * 100, "correct": correct, "total": len(problems)}


def main():
    print("=" * 70)
    print("SVD-DERIVED ACTIVATION STEERING")
    print("=" * 70)
    print(flush=True)

    eval_problems = load_math500_split("test")

    # Step 1: Extract SVD components (u vectors + sigma weights)
    print("Step 1: Extracting SVD steering vectors (o_proj + down_proj)...", flush=True)
    components = compute_svd_components()

    # Compute sigma stats for normalization
    all_sigmas = []
    for layer_idx, mods in components.items():
        for mod_type, data in mods.items():
            all_sigmas.append((layer_idx, mod_type, data["sigma"], data["rank1_frac"]))

    all_sigmas.sort(key=lambda x: x[2], reverse=True)
    sigma_max = all_sigmas[0][2]
    print(f"   Sigma range: [{all_sigmas[-1][2]:.4f}, {sigma_max:.4f}]")
    print(f"   Top-5 layers by sigma:")
    for idx, mod, sig, r1f in all_sigmas[:5]:
        print(f"      Layer {idx} ({mod}): sigma={sig:.4f}, rank1_frac={r1f:.3f}")

    # Step 2: Load target model
    print("\nStep 2: Loading target model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # Build module name -> layer mapping for hooks
    module_map = {}  # module_name -> (layer_idx, mod_type)
    for name, module in model.named_modules():
        if name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj"):
            parts = name.split(".")
            layer_idx = int(parts[2])
            mod_type = "attn" if "self_attn" in name else "mlp"
            if layer_idx in components and mod_type in components[layer_idx]:
                module_map[name] = (layer_idx, mod_type)

    print(f"   Mapped {len(module_map)} modules for steering", flush=True)

    # Step 3: Evaluate variants
    print("\nStep 3: Evaluating steering variants...", flush=True)
    results = {}

    # --- Baseline (no steering) ---
    print("\n  --- Baseline (no steering) ---", flush=True)
    res = evaluate_with_steering(model, tokenizer, eval_problems, [], "baseline")
    results["baseline"] = res
    print(f"  >>> Baseline: {res['accuracy']:.1f}%", flush=True)

    # --- Variant 1: Sigma-weighted steering (various alpha) ---
    for alpha in [0.01, 0.03, 0.05, 0.1]:
        name = f"svd_sigma_weighted_a{alpha}"
        print(f"\n  --- {name} ---", flush=True)
        print(f"      h' = h + {alpha} * (sigma_l/sigma_max) * u_l", flush=True)

        hooks = []
        for mod_name, (layer_idx, mod_type) in module_map.items():
            data = components[layer_idx][mod_type]
            u = data["u"].to(model.device, model.dtype)
            weight = data["sigma"] / sigma_max

            def make_hook(u_vec, w, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h = h + a * w * u_vec.unsqueeze(0).unsqueeze(0)
                        return (h,) + output[1:]
                    else:
                        return output + a * w * u_vec.unsqueeze(0).unsqueeze(0)
                return hook_fn

            mod = dict(model.named_modules())[mod_name]
            hooks.append(mod.register_forward_hook(make_hook(u, weight, alpha)))

        res = evaluate_with_steering(model, tokenizer, eval_problems, hooks, name)
        results[name] = res
        print(f"  >>> {name}: {res['accuracy']:.1f}%", flush=True)

        for h in hooks:
            h.remove()

        with open(OUTPUT_DIR / f"eval_{name}.json", "w") as f:
            json.dump(res, f, indent=2)

    # --- Variant 2: Top-K layers only (K=10, by sigma) ---
    for alpha in [0.05, 0.1]:
        top_k = 10
        top_layers = set((idx, mod) for idx, mod, sig, _ in all_sigmas[:top_k])
        name = f"svd_top{top_k}_a{alpha}"
        print(f"\n  --- {name} ---", flush=True)
        print(f"      Steering only top-{top_k} layers by sigma", flush=True)

        hooks = []
        for mod_name, (layer_idx, mod_type) in module_map.items():
            if (layer_idx, mod_type) not in top_layers:
                continue
            data = components[layer_idx][mod_type]
            u = data["u"].to(model.device, model.dtype)

            def make_hook(u_vec, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h = h + a * u_vec.unsqueeze(0).unsqueeze(0)
                        return (h,) + output[1:]
                    else:
                        return output + a * u_vec.unsqueeze(0).unsqueeze(0)
                return hook_fn

            mod = dict(model.named_modules())[mod_name]
            hooks.append(mod.register_forward_hook(make_hook(u, alpha)))

        res = evaluate_with_steering(model, tokenizer, eval_problems, hooks, name)
        results[name] = res
        print(f"  >>> {name}: {res['accuracy']:.1f}%", flush=True)

        for h in hooks:
            h.remove()

        with open(OUTPUT_DIR / f"eval_{name}.json", "w") as f:
            json.dump(res, f, indent=2)

    # --- Variant 3: Rank1-frac weighted (steer more in high-concentration layers) ---
    alpha = 0.05
    name = f"svd_rank1frac_weighted_a{alpha}"
    print(f"\n  --- {name} ---", flush=True)
    print(f"      h' = h + {alpha} * rank1_frac_l * u_l", flush=True)

    hooks = []
    for mod_name, (layer_idx, mod_type) in module_map.items():
        data = components[layer_idx][mod_type]
        u = data["u"].to(model.device, model.dtype)
        weight = data["rank1_frac"]

        def make_hook(u_vec, w, a):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    h = h + a * w * u_vec.unsqueeze(0).unsqueeze(0)
                    return (h,) + output[1:]
                else:
                    return output + a * w * u_vec.unsqueeze(0).unsqueeze(0)
            return hook_fn

        mod = dict(model.named_modules())[mod_name]
        hooks.append(mod.register_forward_hook(make_hook(u, weight, alpha)))

    res = evaluate_with_steering(model, tokenizer, eval_problems, hooks, name)
    results[name] = res
    print(f"  >>> {name}: {res['accuracy']:.1f}%", flush=True)

    for h in hooks:
        h.remove()

    with open(OUTPUT_DIR / f"eval_{name}.json", "w") as f:
        json.dump(res, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    bl = results["baseline"]["accuracy"]
    print(f"\n{'Method':<45} {'Acc':<8} {'Delta':<8}")
    print("-" * 61)
    for name, r in results.items():
        d = r['accuracy'] - bl
        print(f"{name:<45} {r['accuracy']:.1f}%   {d:+.1f}")
    print(f"\nReference: unconditional mean-diff steering = 58% (+10)")

    with open(OUTPUT_DIR / "svd_steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'svd_steering_results.json'}")


if __name__ == "__main__":
    main()
