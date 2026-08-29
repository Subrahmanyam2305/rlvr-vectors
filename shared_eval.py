"""
Shared evaluation utilities used by all experiment scripts.

Single, canonical implementation of:
  - Data loading with disjoint splits
  - Answer extraction and normalization via math-verify (symbolic equivalence)
  - Per-problem evaluation with item-level saving
  - Paired McNemar test (primary) + bootstrap CIs (effect size)
  - Wilson confidence intervals
"""

import json
import re
import math
import random
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# math-verify for symbolic answer equivalence — hard requirement
try:
    from math_verify import parse as mv_parse, verify as mv_verify
    MATH_VERIFY_AVAILABLE = True
except ImportError as _mv_err:
    raise ImportError(
        "math-verify is required for correct symbolic answer comparison. "
        "Install it with: pip install math-verify\n"
        f"Original error: {_mv_err}"
    ) from _mv_err

DATA_DIR = Path("/home/ubuntu/rlvr-vectors/data")
OUTPUT_DIR = Path("/home/ubuntu/rlvr-vectors/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data splits (fully disjoint, fixed indices)
#
#  CALIB : 450–499  (50 problems)  → steering vector calibration
#  VAL   : 400–449  (50 problems)  → select alpha / K
#  TEST  : 0–399    (400 problems) → final reported numbers
# ──────────────────────────────────────────────────────────────────────────────
CALIB_SLICE = (450, 500)
VAL_SLICE   = (400, 450)
TEST_SLICE  = (0,   400)


def load_math500_split(split: str, n: int | None = None):
    """Load a fixed named split. split: 'calib'|'val'|'test'|'all'"""
    with open(DATA_DIR / "math500.json") as f:
        all_problems = json.load(f)
    assert len(all_problems) >= 500, f"Expected 500 problems, got {len(all_problems)}"
    slices = {"calib": CALIB_SLICE, "val": VAL_SLICE, "test": TEST_SLICE, "all": (0, 500)}
    if split not in slices:
        raise ValueError(f"Unknown split: {split}")
    s, e = slices[split]
    problems = all_problems[s:e]
    return problems[:n] if n is not None else problems


# ──────────────────────────────────────────────────────────────────────────────
# Answer comparison — uses math-verify for symbolic equivalence, falls back
# to string/numeric matching.  One canonical implementation used everywhere.
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


def _normalize_fallback(a: str) -> str:
    """String/numeric normalization as fallback when math-verify fails."""
    a = a.strip()
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\left|\\right", "", a)
    a = a.replace(" ", "").lower().replace("−", "-")
    return a


def answers_match(pred: str, gold: str) -> bool:
    """Return True if predicted answer matches gold answer.
    Uses math-verify (symbolic SymPy equivalence) as primary method,
    falling back to string/numeric comparison."""
    if not pred or not gold:
        return False

    if MATH_VERIFY_AVAILABLE:
        try:
            p_parsed = mv_parse(f"${pred}$")
            g_parsed = mv_parse(f"${gold}$")
            if p_parsed is not None and g_parsed is not None:
                return bool(mv_verify(g_parsed, p_parsed))
        except Exception:
            pass  # fall through to string/numeric

    # Fallback: string / numeric
    pred_n = _normalize_fallback(pred)
    gold_n = _normalize_fallback(gold)
    if pred_n == gold_n:
        return True
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-6
    except (ValueError, TypeError):
        pass
    try:
        if "/" in pred_n:
            num, den = pred_n.split("/", 1)
            return abs(float(num) / float(den) - float(gold_n)) < 1e-4
        if "/" in gold_n:
            num, den = gold_n.split("/", 1)
            return abs(float(pred_n) - float(num) / float(den)) < 1e-4
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
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (f"Please reason step by step, and put your final answer within "
                f"\\boxed{{}}.\n\n{question}\n\nSolution:")


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation — saves item-level predictions
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, problems, label: str = "",
             max_new_tokens: int = 1024, save_items: bool = True) -> dict:
    """Evaluate model on problems. Returns dict with per-item results."""
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
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(response)
        gold = prob["answer"]
        is_correct = answers_match(pred, gold)
        if is_correct:
            correct += 1

        items.append({
            "idx": i,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response_len": len(response.split()),
            "has_boxed": bool(pred),
        })

        if (i + 1) % 10 == 0:
            boxed_rate = sum(1 for it in items if it["has_boxed"]) / len(items)
            print(f"  [{label}] [{i+1}/{len(problems)}] "
                  f"Acc: {correct/(i+1)*100:.1f}%  "
                  f"boxed: {boxed_rate*100:.0f}%", flush=True)

    result = {
        "accuracy": correct / len(problems) * 100,
        "correct": correct,
        "total": len(problems),
        "label": label,
        "boxed_rate": sum(1 for it in items if it["has_boxed"]) / len(problems),
        "items": items,
    }
    if save_items:
        with open(OUTPUT_DIR / f"items_{label}.json", "w") as f:
            json.dump(result, f, indent=2)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────────────────────────────────────

def mcnemar_test(correct_a: list, correct_b: list) -> dict:
    """
    Two-sided exact McNemar test for paired correctness outcomes.
    Primary significance test as recommended by reviewer.

    correct_a, correct_b: per-item boolean lists (same length, same ordering).
    Returns dict with b (a-correct,b-wrong), c (a-wrong,b-correct), p_value.
    """
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)

    # Exact two-sided p-value via binomial
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "test": "mcnemar_exact"}
    # P(X >= max(b,c)) * 2 under H0: X~Binomial(n, 0.5)
    k = max(b, c)
    p = float(2 * binom.sf(k - 1, n, 0.5))
    p = min(p, 1.0)
    return {"b": b, "c": c, "n_discordant": n, "p_value": round(p, 4), "test": "mcnemar_exact"}


def bootstrap_ci(correct_a: list, correct_b: list,
                 n_boot: int = 10000, alpha: float = 0.05) -> dict:
    """
    Paired bootstrap CI for effect size (accuracy_b - accuracy_a) in pp.
    Complementary to McNemar; do not use as primary significance test.
    """
    assert len(correct_a) == len(correct_b)
    n = len(correct_a)
    diffs = np.array(correct_b, dtype=float) - np.array(correct_a, dtype=float)
    observed = diffs.mean() * 100
    rng = np.random.default_rng(42)
    boot = [rng.integers(0, n, size=n) for _ in range(n_boot)]
    boot_diffs = np.array([diffs[idx].mean() * 100 for idx in boot])
    return {
        "mean_diff_pp": round(float(observed), 2),
        "ci_low_pp":  round(float(np.percentile(boot_diffs, alpha/2*100)), 2),
        "ci_high_pp": round(float(np.percentile(boot_diffs, (1-alpha/2)*100)), 2),
    }


def wilson_ci(correct: int, total: int, alpha: float = 0.05) -> tuple:
    """Wilson score interval (percentage points)."""
    if total == 0:
        return (0.0, 100.0)
    z = 1.959964
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2*total)) / denom
    half = (z * math.sqrt(p*(1-p)/total + z**2/(4*total**2))) / denom
    return (round((center-half)*100, 1), round((center+half)*100, 1))


def print_result_row(label: str, result: dict, baseline_items: list = None):
    acc = result["accuracy"]
    lo, hi = wilson_ci(result["correct"], result["total"])
    row = f"  {label:<50} {acc:.1f}%  95%CI [{lo:.1f}, {hi:.1f}]"
    if baseline_items is not None:
        method_items = [it["correct"] for it in result["items"]]
        mn = mcnemar_test(baseline_items, method_items)
        bc = bootstrap_ci(baseline_items, method_items)
        row += (f"  Δ={bc['mean_diff_pp']:+.1f}pp"
                f"  McNemar p={mn['p_value']:.3f}")
    print(row)
