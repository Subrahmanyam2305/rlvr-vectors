"""
Phase 2: Evaluate cross-model transfer of rank-1 RLVR vectors.

Applies extracted rank-1 vectors to the target model (Qwen2.5-1.5B-Instruct)
and evaluates on MATH500 with various alpha scaling factors.
"""

import torch
import json
import gc
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")

MODELS = {
    "math_base": "Qwen/Qwen2.5-Math-1.5B",
    "rlvr_1shot": "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}


def load_math500(n=100):
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
    a = a.replace("\\frac", "").replace("\\dfrac", "")
    return a


def answers_match(pred, gold):
    if not pred or not gold:
        return False
    if normalize_answer(pred) == normalize_answer(gold):
        return True
    try:
        return abs(float(normalize_answer(pred)) - float(normalize_answer(gold))) < 1e-6
    except (ValueError, TypeError):
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


def evaluate(model, tokenizer, problems, label="", max_new_tokens=1024):
    """Evaluate model on math problems."""
    model.eval()
    correct = 0
    start = time.time()

    for i, prob in enumerate(problems):
        text = make_prompt(tokenizer, prob["problem"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(resp)
        if answers_match(pred, prob["answer"]):
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  [{label}] [{i+1}/{len(problems)}] Acc: {correct/(i+1)*100:.1f}%", flush=True)

    elapsed = time.time() - start
    result = {
        "accuracy": correct / len(problems) * 100,
        "correct": correct,
        "total": len(problems),
        "name": label,
        "elapsed": elapsed,
    }
    return result


def load_and_apply_vectors(model, vectors_path, alpha=1.0, rank=1):
    """Apply rank-k vectors to model weights in-place."""
    vectors = torch.load(vectors_path, map_location="cpu")
    applied = 0
    with torch.no_grad():
        for pname, param in model.named_parameters():
            if pname not in vectors:
                continue
            v = vectors[pname]
            u, sigma, vt = v["u"], v["sigma"], v["v"]
            if param.shape != (u.shape[0], vt.shape[0]):
                continue
            rank1 = (alpha * sigma) * torch.outer(u, vt)
            param.data.add_(rank1.to(param.device, param.dtype))
            del rank1
            applied += 1
    print(f"  Applied vectors to {applied} layers (alpha={alpha})")
    return applied


def run_experiment(model_name, label, problems, vectors_path=None, alpha=1.0):
    """Run a single evaluation experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {label}")
    print(f"{'='*60}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if vectors_path:
        load_and_apply_vectors(model, vectors_path, alpha=alpha)

    result = evaluate(model, tokenizer, problems, label)
    print(f"  >>> {label}: {result['accuracy']:.1f}%")

    out_file = OUTPUT_DIR / f"eval_{label.replace(' ', '_').lower()}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    problems = load_math500(100)
    vectors_path = OUTPUT_DIR / "rank1_vectors.pt"

    if not vectors_path.exists():
        print("ERROR: Run extract_vectors.py first to generate rank1_vectors.pt")
        return

    results = {}

    # Baselines
    results["math_base"] = run_experiment(MODELS["math_base"], "math_base", problems)
    results["rlvr_actual"] = run_experiment(MODELS["rlvr_1shot"], "rlvr_1shot_actual", problems)
    results["instruct_base"] = run_experiment(MODELS["instruct"], "instruct_base", problems)

    # Sanity check: rank-1 back to source
    results["sanity"] = run_experiment(
        MODELS["math_base"], "sanity_rank1", problems,
        vectors_path=vectors_path, alpha=1.0
    )

    # Cross-model transfer sweep
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        label = f"transfer_rank1_a{alpha}"
        results[label] = run_experiment(
            MODELS["instruct"], label, problems,
            vectors_path=vectors_path, alpha=alpha
        )

    # Random baseline
    print("\n" + "=" * 60)
    print("Experiment: random_baseline")
    print("=" * 60, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELS["instruct"])
    tokenizer.pad_token = tokenizer.eos_token

    vectors = torch.load(vectors_path, map_location="cpu")
    with torch.no_grad():
        for pname, param in model.named_parameters():
            if pname not in vectors:
                continue
            v = vectors[pname]
            u, sigma, vt = v["u"], v["sigma"], v["v"]
            rand_u = torch.randn_like(u)
            rand_u = rand_u / rand_u.norm() * u.norm()
            rand_v = torch.randn_like(vt)
            rand_v = rand_v / rand_v.norm() * vt.norm()
            rank1 = sigma * torch.outer(rand_u, rand_v)
            param.data.add_(rank1.to(param.device, param.dtype))

    result = evaluate(model, tokenizer, problems, "random_baseline")
    results["random"] = result
    print(f"  >>> random_baseline: {result['accuracy']:.1f}%")
    with open(OUTPUT_DIR / "eval_random_baseline.json", "w") as f:
        json.dump(result, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Experiment':<30} {'Accuracy':<10}")
    print("-" * 40)
    for name, r in results.items():
        print(f"{name:<30} {r['accuracy']:.1f}%")


if __name__ == "__main__":
    main()
