# From Weight Deltas to Steering Vectors

**Transferring RLVR reasoning to new models via SVD-extracted steering vectors**

This repository contains the code and experimental results for our paper analyzing why RLVR (Reinforcement Learning with Verifiable Rewards) reasoning vectors fail to transfer across language models, and proposing SVD-derived activation steering as an alternative.

## Key Findings

1. **Rank-1 weight transfer = conditional activation steering** — We prove that applying a rank-1 weight delta is mathematically equivalent to input-conditional steering: `y_new = y + σ·(v^T·x)·u`

2. **The gating mechanism breaks cross-model** — The trigger signal `v^T·x` drops to 45.6% magnitude with 12% sign flips on the target model, explaining why weight transfer yields only +1.5 pp while activation steering yields +3.5 pp

3. **SVD-derived steering vectors** — Extracting `u` from weight deltas and applying as activation steering achieves +2.25 pp with sparse top-K selection, without requiring source model inference

## Results Summary

Results on disjoint held-out TEST partition (n=400). None reach statistical significance.

| Method | MATH500 Accuracy | Δ vs Baseline | McNemar *p* |
|--------|:---:|:---:|:---:|
| Target baseline (Qwen2.5-1.5B-Instruct) | 46.25% | — | — |
| Weight transfer (rank-1) | 47.75% | +1.50 pp | 0.539 |
| SVD top-5 steering (ours) | 48.50% | +2.25 pp | 0.298 |
| **Mean-diff activation steering** | **49.75%** | **+3.50 pp** | 0.125 |

**[Read the full paper →](https://subrahmanyam2305.github.io/rlvr-vectors/)**

## Setup

```bash
pip install -r requirements.txt
```

Models are downloaded automatically from HuggingFace Hub.

## Reproducing Experiments

### Phase 1: Spectral Analysis
```bash
python download_math500.py    # Download evaluation data
python extract_vectors.py     # SVD analysis of RLVR weight deltas
```

### Phase 2: Transfer Evaluation
```bash
python eval_weight_transfer.py  # Rank-1 weight transfer experiments
```

### Phase 3: Vector Consistency
```bash
python compare_vectors.py     # Compare 1-shot vs full RLVR vectors
```

### Phase 4: Activation Steering
```bash
python approach_steering.py   # Mean-difference steering
```

### Phase 5: Analytical Validation
```bash
python analytical_connection.py  # Measure v^T*x gating mismatch
```

### Phase 6: Recalibrated Transfer
```bash
python recalibrated_transfer.py  # Attempt to fix weight transfer
```

### Phase 7: SVD-Derived Steering
```bash
python svd_steering.py           # SVD u-vectors as steering (projection level)
python eval_steering_comparison.py  # Full comparison: SVD vs mean-diff (residual stream)
```

## Models Used

| Model | Role |
|-------|------|
| [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | Source base |
| [ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1](https://huggingface.co/ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1) | Source RLVR |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Target |

## Project Structure

```
├── download_math500.py         # Dataset preparation
├── extract_vectors.py          # Phase 1: SVD spectral analysis
├── eval_weight_transfer.py     # Phase 2: Weight transfer evaluation
├── compare_vectors.py          # Phase 3: Vector consistency check
├── approach_steering.py        # Phase 4: Activation steering
├── analytical_connection.py    # Phase 5: Gating signal analysis
├── recalibrated_transfer.py    # Phase 6: Recalibrated weight transfer
├── svd_steering.py             # Phase 7: SVD-derived steering
├── eval_steering_comparison.py # Phase 7: Full SVD vs mean-diff comparison
├── REPORT.md                   # Detailed experimental report
├── docs/                       # GitHub Pages site (arXiv-style paper)
│   ├── index.html
│   ├── paper.md
│   └── figures/
├── requirements.txt
├── data/
│   └── math500.json            # MATH500 evaluation set
└── outputs/
    ├── spectral_data.json      # Rank-1 fractions per layer
    ├── analytical_connection.json
    ├── overnight_results.json  # Final comparison results
    └── eval_*.json             # Individual experiment results
```

## Citation

```bibtex
@article{arunachalam2026weight-deltas-steering,
  title={From Weight Deltas to Steering Vectors: Understanding and Improving Cross-Model Transfer of RLVR Reasoning},
  author={Arunachalam, Subrahmanyam},
  year={2026}
}
```

## License

MIT
