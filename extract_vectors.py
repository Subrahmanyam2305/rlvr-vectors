"""
Phase 1: Extract rank-1 vectors from RLVR weight deltas and analyze spectral concentration.

Computes SVD of dW = W_rlvr - W_base for each layer, measuring how much
energy is concentrated in the rank-1 component.
"""

import torch
import json
import gc
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "base": "Qwen/Qwen2.5-Math-1.5B",
    "rlvr": "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
}


def extract_rank1_vectors():
    """Memory-efficient SVD extraction using safetensors layer-by-layer."""
    print("Downloading model files...")
    base_path = snapshot_download(MODELS["base"])
    rlvr_path = snapshot_download(MODELS["rlvr"])

    base_files = sorted(Path(base_path).glob("*.safetensors"))
    rlvr_files = sorted(Path(rlvr_path).glob("*.safetensors"))

    # Build index: param_name -> file path
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

    spectral_data = []
    vectors = {}  # param_name -> {u, sigma, v}
    count = 0

    weight_params = [k for k in base_index if "weight" in k and k in rlvr_index]
    print(f"Processing {len(weight_params)} weight tensors...")

    for param_name in sorted(weight_params):
        with safe_open(base_index[param_name], framework="pt", device="cpu") as sf:
            w_base = sf.get_tensor(param_name).float()
        with safe_open(rlvr_index[param_name], framework="pt", device="cpu") as sf:
            w_rlvr = sf.get_tensor(param_name).float()

        if len(w_base.shape) != 2:
            del w_base, w_rlvr
            continue

        dW = w_rlvr - w_base
        frob_norm = dW.norm().item()
        if frob_norm < 1e-8:
            del w_base, w_rlvr, dW
            continue

        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
        total_energy = (S ** 2).sum().item()

        rank1_frac = (S[0] ** 2).item() / total_energy
        rank4_frac = (S[:4] ** 2).sum().item() / total_energy
        rank8_frac = (S[:8] ** 2).sum().item() / total_energy

        # Parse layer info from param name
        parts = param_name.split(".")
        layer_idx = -1
        layer_type = "other"
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
        if "self_attn" in param_name:
            layer_type = param_name.split("self_attn.")[-1].replace(".weight", "")
        elif "mlp" in param_name:
            layer_type = param_name.split("mlp.")[-1].replace(".weight", "")

        spectral_data.append({
            "name": param_name,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "rank1_frac": rank1_frac,
            "rank4_frac": rank4_frac,
            "rank8_frac": rank8_frac,
            "frobenius_norm": frob_norm,
            "top_singular_value": S[0].item(),
            "shape": list(w_base.shape),
        })

        vectors[param_name] = {
            "u": U[:, 0].clone(),
            "sigma": S[0].item(),
            "v": Vt[0, :].clone(),
        }

        del w_base, w_rlvr, dW, U, S, Vt
        count += 1
        if count % 20 == 0:
            gc.collect()
            print(f"  ... processed {count} layers")

    gc.collect()
    print(f"\nDone: {count} layers with non-trivial deltas")

    # Save spectral data
    with open(OUTPUT_DIR / "spectral_data.json", "w") as f:
        json.dump(spectral_data, f, indent=2)
    print(f"Saved spectral data to {OUTPUT_DIR / 'spectral_data.json'}")

    # Save vectors
    torch.save(vectors, OUTPUT_DIR / "rank1_vectors.pt")
    print(f"Saved rank-1 vectors to {OUTPUT_DIR / 'rank1_vectors.pt'}")

    # Print summary
    fracs = [d["rank1_frac"] for d in spectral_data]
    print(f"\n--- Spectral Summary ---")
    print(f"Layers analyzed: {len(spectral_data)}")
    print(f"Mean rank-1 fraction: {sum(fracs)/len(fracs):.3f}")
    print(f"Max rank-1 fraction: {max(fracs):.3f}")
    print(f"Layers with >30% in rank-1: {sum(1 for f in fracs if f > 0.3)}/{len(fracs)}")
    print(f"Layers with >50% in rank-1: {sum(1 for f in fracs if f > 0.5)}/{len(fracs)}")

    return spectral_data, vectors


if __name__ == "__main__":
    extract_rank1_vectors()
