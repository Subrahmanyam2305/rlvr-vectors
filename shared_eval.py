"""
Shared evaluation utilities used by all experiment scripts.

Single, canonical implementation of:
  - Data loading with disjoint splits
  - Answer extraction and normalization
  - Per-problem evaluation with item-level saving
  - Paired bootstrap confidence intervals
"""

import json
import re
import math
import random
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path("/home/ubuntu/rlvr-vectors/data")
OUTPUT_DIR = Path("/home/ubuntu/rlvr-vectors/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data splits (fully disjoint, fixed indices)
#
#  CALIB : 450–499  (50 problems)  → compute steering vectors / mean-diff
#  VAL   : 400–449  (50 problems)  → select alpha / K (hyperparameter tuning)
#  TEST  : 0–399    (400 problems) → final reported numbers
# ──────────────────────────────────────────────────────────────────────────────
CALIB_SLICE = (450, 500)   # inclusive start, exclusive end
VAL_SLICE   = (400, 450)
TEST_SLICE  = (0,   400)


def load_math500_split(split: str, n: int | None = None):
    """
    Load a fixed, named split so calibration never leaks into evaluation.

    split: 'calib' | 'val' | 'test' | 'all'
    n    : optionally limit to first n from that split
    """
    with open(DATA_DIR / "math500.json") as f:
        all_problems = json.load(f)

    assert len(all_problems) >= 500, f"Expected 500 problems, got {len(all_problems)}"

    if split == "calib":
        s, e = CALIB_SLICE
    elif split == "val":
        s, e = VAL_SLICE
    elif split == "test":
        s, e = TEST_SLICE
    elif split == "all":
        s, e = 0, 500
    else:
        raise ValueError(f"Unknown split: {split}")

    problems = all_problems[s:e]
    if n is not None:
        problems = problems[:n]
    return problems


# ──────────────────────────────────────────────────────────────────────────────
# Answer extraction — single canonical implementation
# ──────────────────────────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str:
    """Extract the last \\boxed{...} content, handling nested braces."""
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


def normalize_answer(a: str) -> str:
    """Normalize a math answer string for comparison."""
    a = a.strip()
    # Remove whitespace
    a = a.replace(" ", "")
    # Lowercase
    a = a.lower()
    # Remove common LaTeX wrappers (but NOT closing braces globally — that corrupts nested LaTeX)
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\left|\\right", "", a)
    # Normalize minus sign variants
    a = a.replace("−", "-")  # unicode minus
    return a


def answers_match(pred: str, gold: str) -> bool:
    """Return True if predicted answer matches gold answer."""
    if not pred or not gold:
        return False
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)
    if pred_n == gold_n:
        return True
    # Try numeric comparison
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-6
    except (ValueError, TypeError):
        pass
    # Try fraction: a/b vs decimal
    try:
        if "/" in pred_n:
            num, den = pred_n.split("/", 1)
            pred_val = float(num) / float(den)
            return abs(pred_val - float(gold_n)) < 1e-4
        if "/" in gold_n:
            num, den = gold_n.split("/", 1)
            gold_val = float(num) / float(den)
            return abs(float(pred_n) - gold_val) < 1e-4
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system",
         "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"Please reason step by step, and put your final answer within \\boxed{{}}."
            f"\n\n{question}\n\nSolution:"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation — saves item-level predictions for paired tests
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, problems, label: str = "",
             max_new_tokens: int = 1024, save_items: bool = True) -> dict:
    """
    Evaluate model on a list of problems.

    Returns dict with 'accuracy', 'correct', 'total', 'items' (per-problem).
    Optionally saves results to outputs/items_{label}.json.
    """
    model.eval()
    correct = 0
    items = []

    for i, prob in enumerate(problems):
        text = make_prompt(tokenizer, prob["problem"])
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        pred = extract_boxed(response)
        gold = prob["answer"]
        is_correct = answers_match(pred, gold)
        if is_correct:
            correct += 1

        items.append({
            "idx": i,
            "problem": prob["problem"][:120] + "...",
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response_len": len(response.split()),
            "has_boxed": bool(pred),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{label}] [{i+1}/{len(problems)}] "
                  f"Acc: {correct/(i+1)*100:.1f}%  "
                  f"boxed: {sum(1 for it in items if it['has_boxed'])}/{i+1}",
                  flush=True)

    result = {
        "accuracy": correct / len(problems) * 100,
        "correct": correct,
        "total": len(problems),
        "label": label,
        "items": items,
    }

    if save_items:
        path = OUTPUT_DIR / f"items_{label}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Statistical utilities
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(correct_a: list[bool], correct_b: list[bool],
                 n_boot: int = 10000, alpha: float = 0.05) -> dict:
    """
    Paired bootstrap confidence interval for accuracy_b - accuracy_a.

    correct_a, correct_b: per-item boolean lists (same length, same ordering).
    Returns dict with 'mean_diff', 'ci_low', 'ci_high', 'p_value'.
    """
    assert len(correct_a) == len(correct_b)
    n = len(correct_a)
    diffs = np.array(correct_b, dtype=float) - np.array(correct_a, dtype=float)
    observed_diff = diffs.mean() * 100  # in percentage points

    boot_diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs.append(diffs[idx].mean() * 100)

    boot_diffs = np.array(boot_diffs)
    ci_low = np.percentile(boot_diffs, alpha / 2 * 100)
    ci_high = np.percentile(boot_diffs, (1 - alpha / 2) * 100)

    # p-value: fraction of bootstrap samples with diff <= 0 (one-sided: b > a)
    p_value = (boot_diffs <= 0).mean()

    return {
        "mean_diff_pp": round(float(observed_diff), 2),
        "ci_low_pp": round(float(ci_low), 2),
        "ci_high_pp": round(float(ci_high), 2),
        "p_value": round(float(p_value), 4),
    }


def exact_ci_wilson(correct: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (as percentage points)."""
    if total == 0:
        return (0.0, 100.0)
    z = 1.959964  # z_{0.975}
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return (round((center - half) * 100, 1), round((center + half) * 100, 1))


def print_result_row(label: str, result: dict, baseline_acc: float = None):
    acc = result["accuracy"]
    lo, hi = exact_ci_wilson(result["correct"], result["total"])
    delta_str = ""
    if baseline_acc is not None:
        delta_str = f"  Δ={acc - baseline_acc:+.1f}pp"
    print(f"  {label:<50} {acc:.1f}%  95%CI [{lo:.1f}, {hi:.1f}]{delta_str}")
