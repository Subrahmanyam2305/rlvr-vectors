"""
Phase 3: Compare 1-shot RLVR vectors with full RLVR vectors.

Memory-efficient layer-by-layer comparison using safetensors to avoid OOM.
"""

import torch
import json
import gc
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open

OUTPUT_DIR = Path("outputs")

MODELS = {
    "base": "Qwen/Qwen2.5-Math-1.5B",
    "rlvr_1shot": "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    "rlvr_full": "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-1.2k-dsr-sub",
}


def main():
    print("Downloading models...")
    base_path = snapshot_download(MODELS["base"])
    oneshot_path = snapshot_download(MODELS["rlvr_1shot"])
    full_path = snapshot_download(MODELS["rlvr_full"])

    # Build safetensors indices
    def build_index(model_path):
        idx = {}
        for f in sorted(Path(model_path).glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    idx[key] = str(f)
        return idx

    base_idx = build_index(base_path)
    oneshot_idx = build_index(oneshot_path)
    full_idx = build_index(full_path)

    # Compare layer by layer
    results = []
    common_params = [k for k in base_idx if k in oneshot_idx and k in full_idx and "weight" in k]
    print(f"Comparing {len(common_params)} weight tensors...")

    for param_name in sorted(common_params):
        with safe_open(base_idx[param_name], framework="pt", device="cpu") as sf:
            w_base = sf.get_tensor(param_name).float()

        if len(w_base.shape) != 2:
            del w_base
            continue

        with safe_open(oneshot_idx[param_name], framework="pt", device="cpu") as sf:
            w_oneshot = sf.get_tensor(param_name).float()
        with safe_open(full_idx[param_name], framework="pt", device="cpu") as sf:
            w_full = sf.get_tensor(param_name).float()

        dW_oneshot = w_oneshot - w_base
        dW_full = w_full - w_base

        norm_oneshot = dW_oneshot.norm().item()
        norm_full = dW_full.norm().item()

        if norm_oneshot < 1e-8 or norm_full < 1e-8:
            del w_base, w_oneshot, w_full, dW_oneshot, dW_full
            continue

        # SVD of each
        U1, S1, Vt1 = torch.linalg.svd(dW_oneshot, full_matrices=False)
        U2, S2, Vt2 = torch.linalg.svd(dW_full, full_matrices=False)

        # Cosine similarity of rank-1 components
        u_cos = abs(torch.dot(U1[:, 0], U2[:, 0]).item())
        v_cos = abs(torch.dot(Vt1[0], Vt2[0]).item())

        # Overall delta cosine similarity
        delta_cos = (torch.sum(dW_oneshot * dW_full) / (norm_oneshot * norm_full)).item()

        results.append({
            "name": param_name,
            "u_cosine": u_cos,
            "v_cosine": v_cos,
            "delta_cosine": delta_cos,
            "norm_oneshot": norm_oneshot,
            "norm_full": norm_full,
            "rank1_frac_oneshot": (S1[0]**2 / (S1**2).sum()).item(),
            "rank1_frac_full": (S2[0]**2 / (S2**2).sum()).item(),
        })

        del w_base, w_oneshot, w_full, dW_oneshot, dW_full, U1, S1, Vt1, U2, S2, Vt2
        gc.collect()

    # Summary
    print(f"\n--- Vector Consistency (1-shot vs Full RLVR) ---")
    print(f"Layers compared: {len(results)}")

    u_cosines = [r["u_cosine"] for r in results]
    v_cosines = [r["v_cosine"] for r in results]
    d_cosines = [r["delta_cosine"] for r in results]

    print(f"u (output dir) cosine: mean={np.mean(u_cosines):.3f}, median={np.median(u_cosines):.3f}")
    print(f"v (input dir) cosine:  mean={np.mean(v_cosines):.3f}, median={np.median(v_cosines):.3f}")
    print(f"Full delta cosine:     mean={np.mean(d_cosines):.3f}, median={np.median(d_cosines):.3f}")
    print(f"High agreement (u>0.8): {sum(1 for c in u_cosines if c > 0.8)}/{len(u_cosines)}")

    with open(OUTPUT_DIR / "vector_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'vector_comparison.json'}")


if __name__ == "__main__":
    main()
