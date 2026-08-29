"""
Overnight Experiment Suite: SVD-Derived Activation Steering (Comprehensive)

Runs after svd_steering.py finishes. Tests residual-stream steering,
head-to-head vs mean-diff, and conditional variants.
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

OUTPUT_DIR = Path("/home/ubuntu/rlvr-vectors/outputs")
DATA_DIR = Path("/home/ubuntu/rlvr-vectors/data")


def load_math500(n=50):
    with open(DATA_DIR / "math500.json") as f:
        return json.load(f)[:n]


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


def evaluate(model, tokenizer, problems, label=""):
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


def get_svd_per_layer():
    """Get combined SVD steering vector per layer (sum of o_proj + down_proj u vectors)."""
    print("   Computing per-layer SVD steering vectors...", flush=True)
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

    # For residual-stream steering, we need per-LAYER vectors (hidden_size=1536)
    # The o_proj outputs go into the residual stream, so u from o_proj.weight SVD
    # is already in hidden_size space. Same for down_proj.
    layer_vectors = {}  # layer_idx -> {"u_combined": tensor, "sigma_total": float}

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
        u = U[:, 0].clone()
        sigma = S[0].item()

        parts = param_name.split(".")
        layer_idx = int(parts[2])

        if layer_idx not in layer_vectors:
            layer_vectors[layer_idx] = {"vectors": [], "sigmas": []}

        layer_vectors[layer_idx]["vectors"].append(u)
        layer_vectors[layer_idx]["sigmas"].append(sigma)

        del w_base, w_rlvr, dW, U, S, Vt

    gc.collect()

    # Combine: weighted sum of u vectors per layer
    combined = {}
    for layer_idx, data in layer_vectors.items():
        vecs = data["vectors"]
        sigs = data["sigmas"]
        total_sigma = sum(sigs)
        # Weighted combination of u vectors (sigma-weighted)
        combined_u = sum(s * v for s, v in zip(sigs, vecs)) / total_sigma
        combined_u = combined_u / (combined_u.norm() + 1e-8)  # normalize
        combined[layer_idx] = {
            "u": combined_u,
            "sigma": total_sigma,
            "n_components": len(vecs),
        }

    print(f"   Got steering vectors for {len(combined)} layers", flush=True)
    return combined


def get_mean_diff_vectors(calib_problems):
    """Compute mean-difference steering vectors (the method that got 58%)."""
    print("   Computing mean-diff steering vectors...", flush=True)

    # Collect hidden states from source base
    print("      Source model...", flush=True)
    src_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.float16, device_map="auto"
    )
    src_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    src_tok.pad_token = src_tok.eos_token

    src_states = collect_hidden_states(src_model, src_tok, calib_problems)
    del src_model; gc.collect(); torch.cuda.empty_cache()

    # Collect from RLVR model
    print("      RLVR model...", flush=True)
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1", torch_dtype=torch.float16, device_map="auto"
    )
    rlvr_tok = AutoTokenizer.from_pretrained("ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1")
    rlvr_tok.pad_token = rlvr_tok.eos_token

    rlvr_states = collect_hidden_states(rlvr_model, rlvr_tok, calib_problems)
    del rlvr_model; gc.collect(); torch.cuda.empty_cache()

    # Compute difference
    diff_vectors = {}
    for layer_idx in src_states:
        if layer_idx in rlvr_states:
            diff = rlvr_states[layer_idx] - src_states[layer_idx]
            diff_vectors[layer_idx] = diff

    print(f"   Got mean-diff vectors for {len(diff_vectors)} layers", flush=True)
    return diff_vectors


def collect_hidden_states(model, tokenizer, problems):
    """Collect mean hidden state after each transformer layer."""
    model.eval()
    stores = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            mean_h = h.detach().float().mean(dim=(0, 1)).cpu()
            if layer_idx not in stores:
                stores[layer_idx] = []
            stores[layer_idx].append(mean_h)
        return hook_fn

    for name, module in model.named_modules():
        if hasattr(module, 'self_attn') and hasattr(module, 'mlp'):
            # This is a transformer layer
            parts = name.split(".")
            try:
                layer_idx = int(parts[-1])
            except (ValueError, IndexError):
                for p in parts:
                    try:
                        layer_idx = int(p)
                        break
                    except ValueError:
                        continue
                else:
                    continue
            hooks.append(module.register_forward_hook(make_hook(layer_idx)))

    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    return {idx: torch.stack(vecs).mean(0) for idx, vecs in stores.items()}


def run_residual_steering(model, tokenizer, problems, layer_vectors, alpha, label, top_k=None):
    """Apply steering at the residual stream level (after each layer)."""
    hooks = []
    sigma_max = max(d["sigma"] for d in layer_vectors.values())

    # Select which layers to steer
    if top_k:
        sorted_layers = sorted(layer_vectors.items(), key=lambda x: x[1]["sigma"], reverse=True)
        active_layers = set(idx for idx, _ in sorted_layers[:top_k])
    else:
        active_layers = set(layer_vectors.keys())

    for name, module in model.named_modules():
        if not (hasattr(module, 'self_attn') and hasattr(module, 'mlp')):
            continue
        parts = name.split(".")
        layer_idx = None
        for p in parts:
            try:
                layer_idx = int(p)
                break
            except ValueError:
                continue
        if layer_idx is None or layer_idx not in active_layers:
            continue
        if layer_idx not in layer_vectors:
            continue

        data = layer_vectors[layer_idx]
        u = data["u"].to(model.device, model.dtype)
        weight = data["sigma"] / sigma_max

        def make_hook(u_vec, w):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    steer = alpha * w * u_vec
                    h = h + steer.unsqueeze(0).unsqueeze(0)
                    return (h,) + output[1:]
                else:
                    steer = alpha * w * u_vec
                    return output + steer.unsqueeze(0).unsqueeze(0)
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(u, weight)))

    res = evaluate(model, tokenizer, problems, label)

    for h in hooks:
        h.remove()

    return res


def run_mean_diff_steering(model, tokenizer, problems, diff_vectors, alpha, label):
    """Apply mean-difference steering at the residual stream level."""
    hooks = []

    for name, module in model.named_modules():
        if not (hasattr(module, 'self_attn') and hasattr(module, 'mlp')):
            continue
        parts = name.split(".")
        layer_idx = None
        for p in parts:
            try:
                layer_idx = int(p)
                break
            except ValueError:
                continue
        if layer_idx is None or layer_idx not in diff_vectors:
            continue

        diff = diff_vectors[layer_idx].to(model.device, model.dtype)

        def make_hook(d):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    h = h + alpha * d.unsqueeze(0).unsqueeze(0)
                    return (h,) + output[1:]
                else:
                    return output + alpha * d.unsqueeze(0).unsqueeze(0)
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(diff)))

    res = evaluate(model, tokenizer, problems, label)

    for h in hooks:
        h.remove()

    return res


def main():
    print("=" * 70)
    print("OVERNIGHT EXPERIMENT SUITE")
    print("SVD-Derived vs Mean-Diff Activation Steering (Residual Stream)")
    print("=" * 70)
    print(flush=True)

    eval_problems = load_math500(50)
    calib_problems = load_math500(15)

    results = {}

    # ================================================================
    # PART A: Get steering vectors
    # ================================================================
    print("\n[PART A] Computing steering vectors...", flush=True)
    svd_vectors = get_svd_per_layer()
    mean_diff = get_mean_diff_vectors(calib_problems)

    # ================================================================
    # PART B: Load target model
    # ================================================================
    print("\n[PART B] Loading target model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # ================================================================
    # PART C: Baseline
    # ================================================================
    print("\n[PART C] Baseline...", flush=True)
    res = evaluate(model, tokenizer, eval_problems, "baseline")
    results["baseline"] = res
    print(f"  >>> Baseline: {res['accuracy']:.1f}%", flush=True)

    # ================================================================
    # PART D: SVD Residual-Stream Steering (alpha sweep)
    # ================================================================
    print("\n[PART D] SVD steering (residual stream)...", flush=True)
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
        name = f"svd_residual_a{alpha}"
        print(f"\n  --- {name} ---", flush=True)
        res = run_residual_steering(model, tokenizer, eval_problems, svd_vectors, alpha, name)
        results[name] = res
        print(f"  >>> {name}: {res['accuracy']:.1f}%", flush=True)
        with open(OUTPUT_DIR / f"eval_{name}.json", "w") as f:
            json.dump(res, f, indent=2)

    # ================================================================
    # PART E: Mean-Diff Steering (matched alpha sweep for comparison)
    # ================================================================
    print("\n[PART E] Mean-diff steering (residual stream)...", flush=True)
    for alpha in [0.02, 0.05, 0.1, 0.2]:
        name = f"meandiff_residual_a{alpha}"
        print(f"\n  --- {name} ---", flush=True)
        res = run_mean_diff_steering(model, tokenizer, eval_problems, mean_diff, alpha, name)
        results[name] = res
        print(f"  >>> {name}: {res['accuracy']:.1f}%", flush=True)
        with open(OUTPUT_DIR / f"eval_{name}.json", "w") as f:
            json.dump(res, f, indent=2)

    # ================================================================
    # PART F: SVD Top-K Steering (sparse)
    # ================================================================
    print("\n[PART F] SVD top-K layer steering...", flush=True)
    best_svd_alpha = 2.0  # will adjust based on Part D
    for k in [5, 10, 15]:
        name = f"svd_top{k}_residual_a{best_svd_alpha}"
        print(f"\n  --- {name} ---", flush=True)
        res = run_residual_steering(model, tokenizer, eval_problems, svd_vectors, best_svd_alpha, name, top_k=k)
        results[name] = res
        print(f"  >>> {name}: {res['accuracy']:.1f}%", flush=True)
        with open(OUTPUT_DIR / f"eval_{name}.json", "w") as f:
            json.dump(res, f, indent=2)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("OVERNIGHT RESULTS SUMMARY")
    print("=" * 70)
    bl = results["baseline"]["accuracy"]
    print(f"\n{'Method':<45} {'Acc':<8} {'Delta':<8}")
    print("-" * 61)
    for name, r in results.items():
        d = r['accuracy'] - bl
        print(f"{name:<45} {r['accuracy']:.1f}%   {d:+.1f}")

    with open(OUTPUT_DIR / "overnight_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {OUTPUT_DIR / 'overnight_results.json'}")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
