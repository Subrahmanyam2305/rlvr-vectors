"""
Activation-space steering: Transfer reasoning by modifying hidden states at runtime.

Computes steering vectors as the mean hidden-state difference between RLVR and
base models, then applies them to the target model during inference.
"""

import torch
import json
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")

MODELS = {
    "base": "Qwen/Qwen2.5-Math-1.5B",
    "rlvr": "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    "target": "Qwen/Qwen2.5-1.5B-Instruct",
}


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


def collect_mean_hidden_states(model, tokenizer, problems):
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
            parts = name.split(".")
            for p in parts:
                try:
                    layer_idx = int(p)
                    hooks.append(module.register_forward_hook(make_hook(layer_idx)))
                    break
                except ValueError:
                    continue

    for prob in problems:
        text = make_prompt(tokenizer, prob["problem"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    return {idx: torch.stack(vecs).mean(0) for idx, vecs in stores.items()}


def evaluate_with_steering(model, tokenizer, problems, steering_vectors, alpha, label=""):
    """Evaluate target model with steering vectors applied at each layer."""
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
        if layer_idx is None or layer_idx not in steering_vectors:
            continue

        sv = steering_vectors[layer_idx].to(model.device, model.dtype)

        def make_hook(s):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    h = h + alpha * s.unsqueeze(0).unsqueeze(0)
                    return (h,) + output[1:]
                return output + alpha * s.unsqueeze(0).unsqueeze(0)
            return hook_fn

        hooks.append(module.register_forward_hook(make_hook(sv)))

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
            print(f"  [{label}] [{i+1}/{len(problems)}] Acc: {correct/(i+1)*100:.1f}%", flush=True)

    for h in hooks:
        h.remove()

    return {"accuracy": correct / len(problems) * 100, "correct": correct,
            "total": len(problems), "name": label}


def main():
    calib_problems = load_math500_split("cal")
    eval_problems = load_math500_split("test")

    print("Step 1: Collecting hidden states from source base model...")
    src_model = AutoModelForCausalLM.from_pretrained(
        MODELS["base"], torch_dtype=torch.float16, device_map="auto")
    src_tok = AutoTokenizer.from_pretrained(MODELS["base"])
    src_tok.pad_token = src_tok.eos_token
    src_states = collect_mean_hidden_states(src_model, src_tok, calib_problems)
    del src_model; gc.collect(); torch.cuda.empty_cache()

    print("Step 2: Collecting hidden states from RLVR model...")
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        MODELS["rlvr"], torch_dtype=torch.float16, device_map="auto")
    rlvr_tok = AutoTokenizer.from_pretrained(MODELS["rlvr"])
    rlvr_tok.pad_token = rlvr_tok.eos_token
    rlvr_states = collect_mean_hidden_states(rlvr_model, rlvr_tok, calib_problems)
    del rlvr_model; gc.collect(); torch.cuda.empty_cache()

    # Compute steering vectors (difference)
    steering_vectors = {}
    for layer_idx in src_states:
        if layer_idx in rlvr_states:
            steering_vectors[layer_idx] = rlvr_states[layer_idx] - src_states[layer_idx]
    print(f"  Computed steering vectors for {len(steering_vectors)} layers")

    print("Step 3: Loading target model and evaluating...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        MODELS["target"], torch_dtype=torch.float16, device_map="auto")
    tgt_tok = AutoTokenizer.from_pretrained(MODELS["target"])
    tgt_tok.pad_token = tgt_tok.eos_token

    results = {}
    for alpha in [0.02, 0.05, 0.1]:
        label = f"steering_a{alpha}"
        print(f"\n  --- alpha={alpha} ---")
        res = evaluate_with_steering(tgt_model, tgt_tok, eval_problems, steering_vectors, alpha, label)
        results[label] = res
        print(f"  >>> {label}: {res['accuracy']:.1f}%")
        with open(OUTPUT_DIR / f"eval_{label}.json", "w") as f:
            json.dump(res, f, indent=2)

    print("\n--- Results ---")
    for name, r in results.items():
        print(f"  {name}: {r['accuracy']:.1f}%")


if __name__ == "__main__":
    main()
