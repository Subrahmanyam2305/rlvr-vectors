"""Run only SVD + MeanDiff steering for Part C (steps 3 and 4 already have baseline + weight transfer)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, json, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from shared_eval import load_math500_split, evaluate, OUTPUT_DIR
from paper_eval_suite import apply_svd_steering, apply_meandiff_steering
from comprehensive_suite import get_svd_vectors_for_pair, get_meandiff_for_pair, MODELS

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

INSTRUCT_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    print("="*60)
    print("PART C STEP 4 — SVD + MeanDiff steering only")
    print("="*60, flush=True)

    calib = load_math500_split("calib")
    test  = load_math500_split("test")

    # Extract vectors first (CPU-heavy, no GPU needed)
    print("\n[1/3] Extracting SVD and MeanDiff vectors...", flush=True)
    svd_v = get_svd_vectors_for_pair(MODELS["rlvr_oneshot"], MODELS["math_base"],
                                     calib, "source_gate")
    md_v  = get_meandiff_for_pair(MODELS["rlvr_oneshot"], MODELS["math_base"], calib)
    gc.collect(); torch.cuda.empty_cache()

    # Load model fresh
    print("\n[2/3] Loading Instruct model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_ID, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(INSTRUCT_ID)
    tok.pad_token = tok.eos_token

    # SVD steering
    print("\n[3/3] Evaluating SVD and MeanDiff...", flush=True)
    r_svd = apply_svd_steering(model, tok, test, svd_v, 1.5, "scope_svd_full")
    print(f"  SVD full (α=1.5): {r_svd['accuracy']:.1f}%", flush=True)

    r_md = apply_meandiff_steering(model, tok, test, md_v, 0.01, "scope_meandiff")
    print(f"  MeanDiff (α=0.01): {r_md['accuracy']:.1f}%", flush=True)

    # Load existing results and merge
    out_path = OUTPUT_DIR / "partC_results.json"
    results = {}
    if out_path.exists():
        results = json.load(open(out_path))
    results["svd_full"] = {k: v for k, v in r_svd.items() if k != "items"}
    results["meandiff_residual"] = {k: v for k, v in r_md.items() if k != "items"}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
