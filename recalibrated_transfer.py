"""
Recalibrated Rank-1 Transfer (Memory-Efficient)

Fix cross-model rank-1 weight transfer by recalibrating v for the target model.

Approaches:
  A) Per-layer scalar correction: c_l = (v^T x_src) / (v^T x_tgt)
  B) Replace v with target-native v' = normalize(x_tgt), scaled to match source magnitude
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


def compute_svd_layer_by_layer():
    """Memory-efficient SVD using safetensors lazy loading."""
    print("   Downloading model metadata...", flush=True)
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

    svd_components = {}
    count = 0
    total_params = [k for k in base_index if "weight" in k and k in rlvr_index]
    print(f"   Processing {len(total_params)} weight tensors...", flush=True)

    for param_name in total_params:
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
        svd_components[param_name] = {
            "u": U[:, 0].clone(),
            "sigma": S[0].item(),
            "v": Vt[0, :].clone(),
            "rank1_frac": ((S[0]**2) / (S**2).sum()).item(),
        }

        del w_base, w_rlvr, dW, U, S, Vt
        count += 1
        if count % 20 == 0:
            gc.collect()
            print(f"   ... processed {count} layers", flush=True)

    gc.collect()
    print(f"   Done: {count} layers with rank-1 components", flush=True)
    return svd_components


def collect_linear_inputs(model, tokenizer, problems, target_modules):
    """Collect mean input activations to linear layers."""
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
        q = prob["problem"] if isinstance(prob, dict) else prob
        text = make_prompt(tokenizer, q)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model(**inputs)

    for hook in hooks:
        hook.remove()

    return {name: torch.stack(vecs).mean(0) for name, vecs in store.items()}


def evaluate_model(model, tokenizer, problems, label=""):
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
    print("RECALIBRATED RANK-1 TRANSFER")
    print("=" * 70)
    print(flush=True)

    calib_problems = load_math500(15)
    eval_problems = load_math500(50)

    # Step 1: SVD layer by layer
    print("Step 1: Computing SVD (layer-by-layer, memory-safe)...", flush=True)
    svd_components = compute_svd_layer_by_layer()
    target_modules = [pn.replace(".weight", "") for pn in svd_components.keys()]
    print(f"   {len(target_modules)} target modules", flush=True)

    # Step 2: Collect activations
    print("\nStep 2: Collecting calibration activations...", flush=True)

    print("   Source (Math-1.5B)...", flush=True)
    src_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.float16, device_map="auto"
    )
    src_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    src_tok.pad_token = src_tok.eos_token
    src_acts = collect_linear_inputs(src_model, src_tok, calib_problems, target_modules)
    print(f"   Source: {len(src_acts)} modules", flush=True)
    del src_model; gc.collect(); torch.cuda.empty_cache()

    print("   Target (Instruct-1.5B)...", flush=True)
    tgt_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )
    tgt_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tgt_tok.pad_token = tgt_tok.eos_token
    tgt_acts = collect_linear_inputs(tgt_model, tgt_tok, calib_problems, target_modules)
    print(f"   Target: {len(tgt_acts)} modules", flush=True)
    del tgt_model; gc.collect(); torch.cuda.empty_cache()

    # Step 3: Compute corrections (store only scalars + small vectors)
    print("\nStep 3: Computing per-layer corrections...", flush=True)

    corrections = {}  # param_name -> scalar correction factor
    replaced_v_info = {}  # param_name -> {v_prime, scale}
    n_sign_flips = 0
    n_total = 0
    corr_values = []

    for param_name, svd in svd_components.items():
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
        n_total += 1

        # Approach A correction factor
        if abs(vtx_tgt) > 1e-8:
            c = vtx_src / vtx_tgt
            corrections[param_name] = c
            if np.sign(vtx_src) != np.sign(vtx_tgt):
                n_sign_flips += 1
            corr_values.append(c)
        else:
            corrections[param_name] = 0.0

        # Approach B: v' = normalized target activation
        v_prime = x_tgt / (x_tgt.norm() + 1e-8)
        vprime_xtgt = (v_prime @ x_tgt).item()
        scale = (vtx_src / vprime_xtgt) if abs(vprime_xtgt) > 1e-8 else 1.0
        replaced_v_info[param_name] = {"v_prime": v_prime, "scale": scale}

    # Free activations
    del src_acts, tgt_acts
    gc.collect()

    print(f"   {n_total} layers with corrections")
    print(f"   Sign flips: {n_sign_flips}/{n_total} ({n_sign_flips/max(n_total,1)*100:.0f}%)")
    if corr_values:
        print(f"   Correction stats: mean={np.mean(corr_values):.2f}, "
              f"median={np.median(corr_values):.2f}, std={np.std(corr_values):.2f}")

    # Step 4: Evaluate
    print("\nStep 4: Evaluating...", flush=True)
    results = {}

    def apply_approach_a(model, alpha=1.0):
        applied = 0
        with torch.no_grad():
            for pname, param in model.named_parameters():
                if pname not in corrections or pname not in svd_components:
                    continue
                c = corrections[pname]
                if abs(c) > 50 or abs(c) < 1e-6:
                    continue
                svd = svd_components[pname]
                u, sigma, v = svd["u"], svd["sigma"], svd["v"]
                if param.shape != (u.shape[0], v.shape[0]):
                    continue
                rank1 = (alpha * c * sigma) * torch.outer(u, v)
                param.data.add_(rank1.to(param.device, param.dtype))
                del rank1
                applied += 1
        return applied

    def apply_approach_b(model, alpha=1.0):
        applied = 0
        with torch.no_grad():
            for pname, param in model.named_parameters():
                if pname not in replaced_v_info or pname not in svd_components:
                    continue
                svd = svd_components[pname]
                rv = replaced_v_info[pname]
                u, sigma = svd["u"], svd["sigma"]
                v_prime, scale = rv["v_prime"], rv["scale"]
                if abs(scale) > 50:
                    continue
                if param.shape != (u.shape[0], v_prime.shape[0]):
                    continue
                rank1 = (alpha * scale * sigma) * torch.outer(u, v_prime)
                param.data.add_(rank1.to(param.device, param.dtype))
                del rank1
                applied += 1
        return applied

    def load_target():
        gc.collect(); torch.cuda.empty_cache()
        m = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
        )
        return m

    # Baseline
    print("\n  --- Baseline ---", flush=True)
    tgt_model = load_target()
    res = evaluate_model(tgt_model, tgt_tok, eval_problems, "baseline")
    results["baseline"] = res
    print(f"  >>> Baseline: {res['accuracy']:.1f}%", flush=True)
    with open(OUTPUT_DIR / "eval_recal_baseline.json", "w") as f:
        json.dump(res, f, indent=2)

    # Approach A alpha=1.0
    print("\n  --- Recalibrated rank-1, alpha=1.0 ---", flush=True)
    del tgt_model; tgt_model = load_target()
    n = apply_approach_a(tgt_model, alpha=1.0)
    print(f"    Applied to {n} layers", flush=True)
    res = evaluate_model(tgt_model, tgt_tok, eval_problems, "recal_1.0")
    results["recalibrated_a1.0"] = res
    print(f"  >>> {res['accuracy']:.1f}%", flush=True)
    with open(OUTPUT_DIR / "eval_recal_a1.json", "w") as f:
        json.dump(res, f, indent=2)

    # Approach A alpha=0.5
    print("\n  --- Recalibrated rank-1, alpha=0.5 ---", flush=True)
    del tgt_model; tgt_model = load_target()
    n = apply_approach_a(tgt_model, alpha=0.5)
    print(f"    Applied to {n} layers", flush=True)
    res = evaluate_model(tgt_model, tgt_tok, eval_problems, "recal_0.5")
    results["recalibrated_a0.5"] = res
    print(f"  >>> {res['accuracy']:.1f}%", flush=True)
    with open(OUTPUT_DIR / "eval_recal_a05.json", "w") as f:
        json.dump(res, f, indent=2)

    # Approach B alpha=1.0
    print("\n  --- v-replaced (u_src, v'_tgt), alpha=1.0 ---", flush=True)
    del tgt_model; tgt_model = load_target()
    n = apply_approach_b(tgt_model, alpha=1.0)
    print(f"    Applied to {n} layers", flush=True)
    res = evaluate_model(tgt_model, tgt_tok, eval_problems, "vrep_1.0")
    results["v_replaced_a1.0"] = res
    print(f"  >>> {res['accuracy']:.1f}%", flush=True)
    with open(OUTPUT_DIR / "eval_recal_vrep.json", "w") as f:
        json.dump(res, f, indent=2)

    # Approach B alpha=0.5
    print("\n  --- v-replaced (u_src, v'_tgt), alpha=0.5 ---", flush=True)
    del tgt_model; tgt_model = load_target()
    n = apply_approach_b(tgt_model, alpha=0.5)
    print(f"    Applied to {n} layers", flush=True)
    res = evaluate_model(tgt_model, tgt_tok, eval_problems, "vrep_0.5")
    results["v_replaced_a0.5"] = res
    print(f"  >>> {res['accuracy']:.1f}%", flush=True)
    with open(OUTPUT_DIR / "eval_recal_vrep05.json", "w") as f:
        json.dump(res, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    bl = results["baseline"]["accuracy"]
    print(f"\n{'Method':<40} {'Acc':<8} {'Delta':<8}")
    print("-" * 56)
    for name, r in results.items():
        d = r['accuracy'] - bl
        print(f"{name:<40} {r['accuracy']:.1f}%   {d:+.1f}")
    print(f"\nReference points:")
    print(f"  Naive rank-1 alpha=2.0:            ~50% (+2)")
    print(f"  Unconditional steering alpha=0.05: ~58% (+10)")

    with open(OUTPUT_DIR / "recalibrated_all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
