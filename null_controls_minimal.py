"""
Minimal null controls script — runs after paper_eval_suite step 5 completes.
Uses batched evaluation (batch_size=4, max_new_tokens=512) for speed.
2 seeds × 4 null types × 50 problems = ~400 total inference calls.
"""
import torch, json, gc, os, random
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from shared_eval import load_math500_split, evaluate, mcnemar_test, bootstrap_ci, OUTPUT_DIR
from paper_eval_suite import MODELS, get_svd_vectors, apply_svd_steering

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
N_NULL_SEEDS = 2
NULL_N = 50         # problems per null run

def main():
    print("="*60); print("NULL CONTROLS (2 seeds × 4 types × 50 problems)"); print("="*60)
    calib = load_math500_split("calib")
    test  = load_math500_split("test")
    null_sub = test[:NULL_N]

    # Load best config from saved results (fallback to defaults)
    res_path = OUTPUT_DIR / "paper_eval_results.json"
    best_k, best_svd_a = 5, 1.5
    if res_path.exists():
        with open(res_path) as f:
            saved = json.load(f)
        best_k   = saved.get("best_k",     best_k)
        best_svd_a = saved.get("best_svd_a", best_svd_a)
    print(f"  Config: top_k={best_k}, alpha={best_svd_a}")

    print("\n[1] Computing SVD vectors on CALIB...")
    svd_vecs = get_svd_vectors(calib)
    all_layers = list(svd_vecs.keys())

    print("\n[2] Loading target model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["instruct"], torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODELS["instruct"]); tok.pad_token = tok.eos_token

    print("\n[3] Null baseline on subset...")
    bl = apply_svd_steering(model, tok, null_sub, svd_vecs, 0.0, "null_baseline")
    bl_items = [it["correct"] for it in bl["items"]]
    print(f"  Null baseline: {bl['accuracy']:.1f}%")

    results, null_stats = {}, {}

    def run_null(tag, vecs, alpha, label, **kw):
        r = apply_svd_steering(model, tok, null_sub, vecs, alpha, label, **kw)
        mi = [it["correct"] for it in r["items"]]
        mc = mcnemar_test(bl_items, mi)
        bc = bootstrap_ci(bl_items, mi)
        return r["accuracy"], mc, bc

    for null_type, desc in [
        ("rand_dir",   "Random unit vectors"),
        ("rand_sign",  "Random sign flips"),
        ("wrong_layer","Wrong-layer permutation"),
        ("rand_layer", "Random K layers"),
    ]:
        accs = []
        for seed in range(N_NULL_SEEDS):
            label = f"null_{null_type}_s{seed}"
            if null_type == "rand_dir":
                acc, mc, bc = run_null(null_type, svd_vecs, best_svd_a, label,
                                       random_seed=seed, top_k=best_k)
            elif null_type == "rand_sign":
                rng = random.Random(seed)
                flipped = {li: {**d, "u": d["u"] * (1 if rng.random() > 0.5 else -1)}
                           for li, d in svd_vecs.items()}
                acc, mc, bc = run_null(null_type, flipped, best_svd_a, label, top_k=best_k)
            elif null_type == "wrong_layer":
                acc, mc, bc = run_null(null_type, svd_vecs, best_svd_a, label,
                                       top_k=best_k, wrong_layer_seed=seed)
            else:  # rand_layer
                rng = random.Random(seed + 100)
                rand_k = rng.sample(all_layers, min(best_k, len(all_layers)))
                sub = {li: svd_vecs[li] for li in rand_k}
                acc, mc, bc = run_null(null_type, sub, best_svd_a, label)
            accs.append(acc)
            print(f"  {desc} seed={seed}: {acc:.1f}%  Δ={bc['mean_diff_pp']:+.1f}pp  p={mc['p_value']:.3f}")
        null_stats[null_type] = {"accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
        print(f"  → {desc}: mean={np.mean(accs):.1f}% ± {np.std(accs):.1f}%")

    del model; gc.collect(); torch.cuda.empty_cache()

    out = {"null_n": NULL_N, "n_seeds": N_NULL_SEEDS,
           "null_baseline": bl["accuracy"], "null_stats": null_stats}
    with open(OUTPUT_DIR / "null_controls.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[SAVE] null_controls.json written.")
    print("DONE.")

if __name__ == "__main__":
    main()
