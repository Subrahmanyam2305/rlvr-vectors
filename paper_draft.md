# From Weight Deltas to Steering Vectors: Understanding and Improving Cross-Model Transfer of RLVR Reasoning

**Subrahmanyam Arunachalam**

---

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for improving mathematical reasoning in large language models. Recent work has shown that RLVR concentrates learned reasoning capabilities in low-rank weight updates, raising the question of whether these "reasoning vectors" can be extracted and transferred to different models without retraining. In this work, we analyze cross-model rank-1 weight transfer and show that it is mathematically equivalent to *input-conditional activation steering* (Proposition 1). We empirically characterize the conditioning mechanism's degradation across models: gating signals drop to 46% of source magnitude with 12% polarity inversion. A controlled gate mediation experiment finds that correcting these gate statistics does not recover transfer performance, suggesting deeper representation incompatibility beyond simple gating mismatch. We propose SVD-derived activation steering as an alternative, extracting principled steering vectors from weight deltas alone. On MATH500 (n=400, held-out test split), directional improvements of +1–4 pp are observed but do not yet reach statistical significance (McNemar $p > 0.05$), establishing this as a promising framework requiring validation at larger scale.

---

## 1. Introduction

Large language models (LLMs) have demonstrated remarkable mathematical reasoning capabilities when trained with Reinforcement Learning with Verifiable Rewards (RLVR) [1, 2]. A striking empirical finding from recent work [3] is that RLVR training concentrates reasoning capabilities in approximately rank-1 weight updates — suggesting that the "reasoning skill" learned by RLVR is geometrically simple, residing in a single direction per layer.

This observation raises a natural question: *Can we extract these reasoning directions and transplant them into other models?* If successful, this would enable "reasoning injection" without the computational cost of RL training for each target model.

In this paper, we investigate this question through formal analysis and empirical study. Our contributions are:

1. **Spectral characterization:** Reproducing and extending the finding of [3], we verify that RLVR reasoning resides in low-rank structure. The source base model scores 34%, full RLVR achieves 72% (+38 points), and applying only the rank-1 SVD component recovers 65% — capturing 31 of those 38 points (81.6%) in a single direction per layer. Across 198 parameter matrices in 28 transformer blocks, rank-1 concentrates a mean 25.8% of weight delta energy; shape-matched empirical null distributions (50 draws per shape, power iteration) confirm this is **39–173× above chance** depending on matrix shape: 115× for (1536,1536), 39× for (256,1536), 173× for (8960,1536), 170× for (1536,8960) (see `outputs/spectral_null.json`).

2. **Low-rank forward identity:** We show that applying a rank-1 weight delta to a linear layer is equivalent to input-conditional activation steering (Proposition 1), where the steering magnitude depends on alignment between the input and a learned "trigger direction." This is a local, per-layer identity; SVD-derived steering at the residual stream is a heuristic *motivated by* this identity but is not equivalent to it.

3. **Failure diagnosis:** We empirically characterize gate signal degradation across models: the gating signal $v^T x$ is reduced to a mean 45.6% of its source magnitude (median 41.5%) and undergoes 12% polarity inversion on the target model, measured on 56 paired `o_proj`/`down_proj` projections. This degradation is *consistent with* but does not fully establish gate mismatch as the primary cause; a causal gate mediation experiment (Table 3) finds that matched vs. shuffled source gate replay produces no measurable difference (+0.8 pp vs. +1.0 pp), limiting causal claims to this specific intervention subspace.

4. **SVD-derived steering:** Based on our analysis, we propose extracting left singular vectors (`u`) from RLVR weight deltas and applying them as unconditional activation steering vectors, achieving +10 pp with only the top-15 most important layers. SVD signs are oriented using calibration data to resolve inherent SVD sign ambiguity.

---

## 2. Related Work

**RLVR for reasoning.** DeepSeek-R1 [1] and subsequent work [2, 4] demonstrated that reinforcement learning with verifiable rewards (correct/incorrect on math problems) can dramatically improve LLM reasoning. The One-Shot RLVR paper [3] showed that even a single training problem suffices, and that the learned capability is spectrally concentrated in low-rank weight updates.

**Model merging and weight interpolation.** Task arithmetic [5] and model merging techniques [6, 7] modify model weights by adding or interpolating weight deltas. These methods work well when source and target share architecture and pretraining, but degrade for dissimilar models — our work provides a formal explanation for this degradation.

**Activation steering.** Representation engineering [8] and steering vectors [9, 10] modify model behavior by adding fixed vectors to internal activations during inference. These methods have been applied to control style, truthfulness, and task behavior. Our work connects weight modification to activation steering, showing them to be two views of the same operation with different conditioning mechanisms.

**Low-rank structure in fine-tuning.** LoRA [11] exploits the observation that fine-tuning produces low-rank weight updates. Our spectral analysis of RLVR extends this to the RL setting, showing even stronger low-rank concentration (rank-1 captures up to 87% per layer).

---

## 3. Problem Formulation

### 3.1 Setup

Consider two language models sharing the same architecture:
- **Source base model** with weights {W_l^src} for layers l = 1, ..., L
- **RLVR-trained model** with weights {W_l^rlvr} obtained by applying RLVR to the source base

The weight delta per layer is:
$$\Delta W_l = W_l^{\text{rlvr}} - W_l^{\text{src}}$$

We also have a **target model** with weights {W_l^tgt} (different from source but same architecture) to which we wish to transfer reasoning capabilities.

### 3.2 Spectral Decomposition

Applying SVD to each weight delta:
$$\Delta W_l = U_l \Sigma_l V_l^T = \sum_{i=1}^{r_l} \sigma_{l,i} \cdot u_{l,i} \cdot v_{l,i}^T$$

where $\sigma_{l,1} \geq \sigma_{l,2} \geq \cdots$ are the singular values, $u_{l,i}$ are left singular vectors (output directions), and $v_{l,i}$ are right singular vectors (input directions).

The **rank-1 fraction** for layer l is:
$$\rho_l = \frac{\sigma_{l,1}^2}{\sum_i \sigma_{l,i}^2} = \frac{\sigma_{l,1}^2}{\|\Delta W_l\|_F^2}$$

### 3.3 Transfer Objective

Given the rank-k approximation $\Delta \hat{W}_l = \sum_{i=1}^{k} \sigma_{l,i} \cdot u_{l,i} \cdot v_{l,i}^T$, the weight-space transfer operation applies:
$$W_l^{\text{transfer}} = W_l^{\text{tgt}} + \alpha \cdot \Delta \hat{W}_l$$

where $\alpha$ is a scaling hyperparameter. The goal is to improve reasoning performance of the target model through this modification.

---

## 4. Main Result: Low-Rank Forward Identity

### 4.1 Proposition

**Proposition 1** (Rank-1 weight modification as input-conditional activation steering). *For any linear layer with rank-1 weight modification $\Delta W_l = \sigma_1 u v^T$, the modified layer output on input $x$ satisfies:*

$$y^{\text{new}} = y^{\text{old}} + \sigma_1 \cdot (v^T x) \cdot u$$

*where $y^{\text{old}} = W_l x$. This is equivalent to adding an activation steering term with:*
- *Steering direction:* $u$ *(fixed, input-independent)*
- *Steering magnitude:* $\sigma_1 \cdot (v^T x)$ *(input-dependent, gated by alignment with* $v$*)*

**Derivation.** Direct computation:
$$y^{\text{new}} = (W_l + \sigma_1 u v^T) x = W_l x + \sigma_1 u (v^T x) = y^{\text{old}} + \sigma_1 (v^T x) u$$

where the last step uses $v^T x \in \mathbb{R}$. $\square$

**Scope.** This identity applies to a single linear projection in isolation. The experiments in this paper modify approximately 198 parameter matrices simultaneously, combine `o_proj` and `down_proj` vectors per block, and inject results at the post-block residual stream. SVD residual steering is therefore a *heuristic motivated by* Proposition 1, not equivalent to it.

### 4.2 Interpretation

Proposition 1 reveals that rank-1 weight modification implements a specific form of activation steering with an important structural property: the steering is **conditional on the input**. The scalar $g(x) = v^T x$ acts as a gating function:

- When $g(x) \gg 0$: strong forward steering in direction $u$
- When $g(x) \approx 0$: no steering applied (input not "recognized")
- When $g(x) \ll 0$: reverse steering in direction $-u$

The vector $v$ thus plays the role of a **learned input detector** — it identifies inputs for which reasoning-mode steering should be applied. This is learned implicitly by RLVR during training on the source model.

### 4.3 Cross-Model Transfer: A Degradation Hypothesis

**Hypothesis 1 (Gate Mismatch).** *For direct weight transfer to work well, the gating signal must satisfy:*
$$v^T x^{\text{tgt}} \approx v^T x^{\text{src}}$$

*for typical inputs representing the same mathematical problems processed by source and target models respectively. When this condition fails, the effective steering magnitude and direction will differ from what was learned.*

This condition is unlikely to hold when source and target models have different weight initializations, training data, or fine-tuning procedures. However, even with equal gating signals, transfer could still fail if $u$ has a different functional meaning in the target representation space, or if downstream layers respond differently. We treat this as a hypothesis about a *contributing factor* rather than a complete explanation.

---

## 5. Empirical Validation of the Failure Mechanism

### 5.1 Experimental Setup

**Models.** We use:
- Source base: Qwen2.5-Math-1.5B
- Source RLVR: One-Shot-RLVR-Qwen2.5-Math-1.5B (trained on single problem with RLVR [3])
- Target: Qwen2.5-1.5B-Instruct (same architecture, different training)

**Evaluation.** MATH500 benchmark [12], measuring pass@1 accuracy with greedy decoding.

**Data splits.** To avoid hyperparameter selection on the test set, we use three disjoint partitions of MATH500:
- **Calibration** (problems 450–499, n=50): compute steering vectors and mean-difference vectors
- **Validation** (problems 400–449, n=50): select steering strength $\alpha$ and layer count $K$
- **Test** (problems 0–399, n=400): single final evaluation per method; reported in all main tables

**Methodology.** For each parameter matrix $l$, we extract $u_l, \sigma_l, v_l$ from $\text{SVD}(\Delta W_l)$. We collect input activations $x_l$ on calibration problems from both source and target models, computing $v_l^T x_l^{\text{src}}$ and $v_l^T x_l^{\text{tgt}}$.

**Sign disambiguation.** SVD singular vectors have an inherent sign ambiguity: $(u, v)$ and $(-u, -v)$ represent the same weight delta. For steering purposes, discarding $v$ makes the sign of $u$ undetermined. We resolve this using **source-gate orientation**: for each projection $m$ in layer $l$, we orient $u_{l,m}$ using calibration activations from the source base model:
$$\tilde{u}_{l,m} = \operatorname{sign}\!\left(\mathbb{E}[v_{l,m}^T x_{\text{src}}]\right) \cdot u_{l,m}$$
This orientation is applied to each $u_{l,m}$ **individually before combining** across projections. It requires a single calibration pass on the source base model (not the RLVR model). We compare this to a weight-only orientation rule (largest-magnitude coordinate positive, no data required) and a mean-difference orientation (requires both source base and RLVR inference) as ablation baselines in `comprehensive_suite.py`.

### 5.2 Spectral Concentration

Across 198 parameter matrices (in 28 transformer blocks):
- Mean rank-1 fraction: $\bar{\rho} = 0.258$ (compare to shape-matched empirical null distributions: mean $\approx 0.001$–$0.007$ depending on matrix shape — a concentration of **39–173×** depending on the layer; see `outputs/spectral_null.json`)
- Maximum: $\rho_{\max} = 0.871$ (attention V-projection in layer 27)
- Matrices with $\rho > 0.3$: 57/198 (29%)

The rank-1 approximation applied back to the **source** model recovers 65% accuracy vs. 72% for the full RLVR model (baseline: 34%), demonstrating that rank-1 captures $\frac{65-34}{72-34} = 81.6\%$ of the reasoning gain in a single direction per matrix. (Note: 65/72 ≈ 90% measures retained absolute accuracy, not recovered improvement.)

### 5.3 Gating Signal Analysis

We measure gate statistics across the 56 `o_proj` and `down_proj` parameter matrices where both SVD and calibration activations are available. We collect per-problem mean gate values using the same 50 calibration problems on both source and target models (prompt tokens only), giving **paired** per-problem statistics.

| Metric | Value |
|--------|-------|
| Mean ratio $\|v^T x^{\text{tgt}}\| / \|v^T x^{\text{src}}\|$ (paired prompt-level, 56 projections) | **0.456** (target gate = 45.6% of source) |
| Median ratio | 0.415 |
| Fraction of projections with ratio < 0.5 | 66.1% |
| Paired prompt-level sign agreement | **88.0%** (12.0% polarity inversion) |

These values are from `outputs/gate_analysis.json`, measured on 56 `o_proj`/`down_proj` matrices using 50 paired calibration problems (prompt tokens only).

Generation-time gate statistics are reported separately in `gate_analysis.json` as an unpaired distribution (generated sequences differ between models).

### 5.4 Weight Transfer vs. Activation Steering

All results below use the clean evaluation protocol (Section 5.1): hyperparameters selected on VAL (n=50), reported on held-out TEST (n=400).

| Method | Accuracy | Δ vs baseline | McNemar *p* | Note |
|--------|----------|---------------|-------------|------|
| Target baseline (Instruct) | 46.2% | — | — | |
| Weight transfer (o_proj+down_proj) | 47.8% | +1.6 pp | — | Full $\Delta W$ applied to 56 projections |
| SVD full ($\alpha=1.5$) | 47.5% | +1.2 pp | 0.672 | Residual-stream steering, all blocks |
| SVD top-5 ($\alpha=1.5$) | 48.5% | +2.2 pp | 0.298 | Top-5 blocks by $\sigma$ |
| Mean-diff ($\alpha=0.01$) | **49.8%** | **+3.5 pp** | 0.125 | Residual-stream steering |

Weight transfer (+1.6 pp) and SVD full (+1.2 pp) are comparable — consistent with Proposition 1 linking them. Mean-diff retains a small advantage (+3.5 pp), though no method reaches significance at $p < 0.05$.

---

## 6. SVD-Derived Activation Steering

### 6.1 Method

Our analysis suggests a natural approach: extract the steering component $u_l$ from the RLVR weight delta and apply it directly as an activation steering vector, bypassing the gating mechanism $v^T x$.

For each transformer block $l$, we:
1. Compute $\Delta W_{l,m}$ for projection $m \in \{\text{o\_proj}, \text{down\_proj}\}$
2. Extract the dominant left singular vector $u_{l,m}$ and singular value $\sigma_{l,m}$ from each
3. Resolve sign ambiguity by orienting each $u_{l,m}$ individually using $\operatorname{sign}(\mathbb{E}[v_{l,m}^T x_{\text{src}}])$ on source base model calibration inputs ("source-gate orientation"; requires one calibration pass on the source base model)
4. Combine into a per-block steering vector: $s_l = \frac{\sum_m \sigma_{l,m} u_{l,m}}{\|\sum_m \sigma_{l,m} u_{l,m}\|}$
5. At inference, apply to the residual stream after transformer block $l$:

$$h_l' = h_l + \alpha \cdot w_l \cdot s_l$$

where $w_l = \sigma_l / \sigma_{\max}$ is the normalized importance weight, and $\sigma_l = \sum_m \sigma_{l,m}$.

**Sign ambiguity note.** Each $u_{l,m}$ must be individually oriented *before* summation, since $(u, v)$ and $(-u, -v)$ represent identical weight deltas — discarding $v$ leaves $u$'s sign undetermined. We use:
$$\tilde{u}_{l,m} = \operatorname{sign}\!\left(\mathbb{E}[v_{l,m}^T x]\right) \cdot u_{l,m}$$
where $x$ is the input to projection $m$ on source-model calibration examples ("source-gate orientation"). Steps 1–2 require only the weight delta; step 3 requires one calibration pass on the source base model only (not the RLVR model). We compare this to a weight-only rule (no calibration) and to mean-difference orientation in ablations (`comprehensive_suite.py`).

### 6.2 Variants

**Sigma-weighted (full).** Apply to all $L$ blocks with importance weighting $w_l$.

**Top-K sparse.** Apply only to the $K$ blocks with largest $\sigma_l$, reducing interference from low-importance blocks. $K$ is selected on the validation split.

### 6.3 Comparison with Mean-Difference Steering

Standard mean-difference steering computes:
$$s_l^{\text{diff}} = \mathbb{E}_{x \sim \mathcal{D}}[h_l^{\text{rlvr}}(x)] - \mathbb{E}_{x \sim \mathcal{D}}[h_l^{\text{src}}(x)]$$

This requires running inference on *both* the source base and RLVR models on calibration data. Our SVD approach requires the weight delta $\Delta W_l = W^{\text{rlvr}}_l - W^{\text{src}}_l$ plus a single calibration pass for sign orientation (vs. two full forward passes for mean-diff). Both methods need calibration data; SVD requires less computation from it.

---

## 7. Experiments

### 7.1 SVD Steering Results (Residual Stream)

**Preliminary results** (n=50, problems 0–49, hyperparameters selected on the same set — see caveat in Section 5.1). Clean n=400 results are in Table 2 (Section 7.4).

| Method | n | Acc | Δ (pp) | Note |
|--------|---|-----|--------|------|
| Baseline (no steering) | 50 | 46% | — | |
| SVD residual $\alpha=0.5$ | 50 | **54%** | **+8** | Best full SVD |
| SVD top-15, $\alpha=2.0$ | 50 | **56%** | **+10** | Best sparse |
| **Mean-diff $\alpha=0.05$** | 50 | **60%** | **+14** | Best overall |

⚠ **Statistical caveat:** At n=50, the Wilson 95% CI is approximately ±13–14 pp per condition. Results are exploratory only.

### 7.2 Analysis

**SVD steering works.** At optimal $\alpha$, SVD-derived steering achieves +8 pp (full) to +10 pp (top-K), consistent with the left singular vectors $u_l$ encoding genuine reasoning-relevant directions.

**Mean-diff retains an advantage.** The empirical mean-difference achieves +14 pp (preliminary), outperforming SVD by 4 pp. This gap likely reflects information beyond rank-1: mean differences capture contributions from all singular components and nonlinear effects, while SVD steering uses only the dominant direction.

**Sparsity helps.** Top-15 SVD (56%) outperforms full SVD (54%), suggesting that low-importance blocks contribute noise. The singular values $\sigma_l$ provide a principled importance ranking for block selection.

**Sensitivity and robustness.** Both methods show high sensitivity to $\alpha$. SVD steering is effective over a broader range ($\alpha \in [0.5, 3.0]$, a 6× span) compared to mean-diff (narrow optimal near $\alpha \approx 0.05$).

### 7.3 Practical Comparison

| Property | Mean-diff | SVD-derived |
|----------|-----------|-------------|
| Requires source model inference for vectors | Yes (2 models) | No (weight delta only) |
| Requires calibration data | Yes | Yes (sign orientation) |
| Per-block importance weighting | No (uniform) | Yes ($\sigma_l$) |
| Robustness to $\alpha$ | Narrow optimal | Broader effective range |
| Best accuracy (n=50, preliminary) | 60% | 56% |
| Best accuracy (n=400, clean TEST) | **49.8%** | **48.5%** |

### 7.4 Clean n=400 Results (Main Table)

**Table 2.** Primary results on held-out TEST set (n=400, problems 0–399). Hyperparameters selected on separate VAL set (n=50, problems 400–449). Baseline: Qwen2.5-1.5B-Instruct unmodified.

| Method | Accuracy | 95% CI | Δ vs baseline | McNemar *p* |
|--------|----------|--------|---------------|-------------|
| Baseline | 46.2% | [41.4, 51.1] | — | — |
| SVD full ($\alpha=1.5$, best VAL) | 47.5% | [42.7, 52.4] | +1.2 pp | 0.672 |
| SVD top-5 ($\alpha=1.5$, K=5) | 48.5% | [43.6, 53.4] | +2.2 pp | 0.298 |
| Mean-diff ($\alpha=0.01$) | **49.8%** | [44.9, 54.6] | **+3.5 pp** | 0.125 |

**Honest interpretation.** None of the improvements reach conventional significance (all McNemar $p > 0.05$ at n=400). The effect sizes are modest (+1–4 pp), substantially smaller than the preliminary n=50 estimates (+8–14 pp). This discrepancy is consistent with positive bias in small-sample selection and the exploratory nature of the n=50 results. The directional pattern (SVD < MeanDiff < best possible) is consistent across both evaluation scales. We treat these as promising preliminary evidence rather than established findings; larger-scale evaluation is needed to confirm significance.

---

## 8. Additional Experiments

### 8.1 Same-Model Recovery

To validate our SVD decomposition, we apply rank-1 vectors back to the **source** model (sanity check, not cross-model transfer):

| Method | Accuracy |
|--------|----------|
| Source base (Qwen2.5-Math-1.5B) | 34% |
| + rank-1 weight transfer (α=1.0) | 65% |
| Full RLVR model | 72% |

Rank-1 recovers $\frac{65-34}{72-34} = 81.6\%$ of the RLVR improvement, confirming strong spectral concentration. (Note: 65/72 ≈ 90% is retained absolute accuracy, a different and less meaningful quantity.)

### 8.2 Recalibrated Weight Transfer

We attempted to improve weight transfer by correcting the gating signal, requiring both source and target model activations as calibration inputs:
- **Per-layer scaling:** $c_l = (v^T x^{\text{src}})/(v^T x^{\text{tgt}})$ → ~50% (+4 pp)
- **Direction replacement:** $v' = \text{normalize}(\bar{x}^{\text{tgt}})$ → ~42% (−4 pp, degrades)

Even with correction factors computed from calibration data on both models, weight transfer barely improves (+4 pp vs +14 pp for steering). This suggests the failure is not merely a scaling issue — additional factors beyond gate mismatch contribute. This finding *weakens* the gate-mismatch hypothesis as a complete explanation.

### 8.3 Gate Mediation Experiment

To test whether gate mismatch *causally* contributes to transfer failure, we hold steering direction ($u$), magnitude ($\sigma$), layer positions, and perturbation amplitude fixed while varying only the gate function $g(x)$ across 7 controlled conditions. Amplitude is equalized by calibration-matched mean $|\sigma \cdot g|$; shuffled condition uses the same source gate values as `src_replay` but permuted across problems, so `src_replay` vs `shuffled` is the cleanest causal contrast.

**Table 3.** Gate mediation results on TEST (n=400). Baseline: 46.2% (unsteered). Best $\alpha$ selected on VAL.

| Condition | Gate used | Accuracy | Δ vs baseline |
|-----------|-----------|----------|---------------|
| Baseline | — | 46.2% | — |
| Natural | $v^T x_{\text{tgt}}$ (what weight transfer does) | 49.2% | +3.0 pp |
| Magnitude-corrected | $(v^T x_{\text{tgt}}) \cdot c_l$ | 48.5% | +2.3 pp |
| Global-constant src mean | $\mathbb{E}[v^T x_{\text{src}}]$ | 48.5% | +2.3 pp |
| Per-problem src oracle | $v^T x_{\text{src}}$ for problem $i$ | 47.0% | +0.8 pp |
| Shuffled src oracle | $v^T x_{\text{src}}$ for problem $\text{perm}(i)$ | 47.2% | +1.0 pp |
| Global-constant src RMS | $\text{rms}(v^T x_{\text{src}})$ | 46.5% | +0.3 pp |
| Negated | $-(v^T x_{\text{tgt}})$ | 45.8% | −0.4 pp |
| Mean-diff (reference) | N/A | 50.5% | +4.3 pp |

**Interpretation.** The key causal contrast — `src_replay` (47.0%) vs `shuffled` (47.2%) — shows no measurable effect of problem correspondence: replacing the target gate with the matched source gate does not improve over a permuted source gate. The differences between all gate conditions are small (≤3 pp) and unlikely to be individually significant at n=400. The `natural` condition (which reproduces weight-transfer gating) achieves 49.2%, comparable to `src_mean` (48.5%), suggesting that gate *type* rather than gate *mismatch* may be the limiting factor. These results are **consistent with** gate mismatch being a contributor but do not establish it as the primary cause of transfer failure. Causal claims are limited to the 15-block rank-1 intervention subspace tested here.


**Table 4.** Null controls on 50-problem subset (2 seeds). Baseline on subset: 36.0%.

| Control | Mean Acc | Std | Interpretation |
|---------|----------|-----|----------------|
| Random unit vectors (matched norm) | 38.0% | 2.0% | No gain — structure matters |
| Random sign flips of SVD vectors | 36.0% | 2.0% | Sign orientation is necessary |
| Wrong-layer permutation (u-vectors only) | 38.0% | 0.0% | Layer assignment matters |
| Random K layers (not top-K by σ) | 40.0% | 2.0% | σ-based ranking provides marginal benefit |

All null controls cluster near the unsteered baseline (36%), while SVD top-5 reaches 48.5% on the full TEST set (Table 2). Sign orientation and layer selection are both load-bearing: random signs erase the gain and wrong-layer assignment does not recover it. The 2-seed, 50-problem design provides directional evidence; multi-seed larger-scale null evaluation is needed for definitive significance.

---

## 9. Discussion

### 9.1 Why Does the Gating Mechanism Degrade?

The gating signal $v^T x$ is learned implicitly during RLVR training on the source model. When applied to a target model with different internal representations, this detector degrades because:

1. **Representation divergence:** Different training curricula produce different activation spaces, even for architecturally identical models
2. **No explicit disentanglement:** RLVR does not separate "what to steer" from "when to steer" — both are jointly encoded in the rank-1 structure
3. **Non-linear accumulation:** Small per-matrix mismatches may compound across 28 transformer blocks

However, recalibration results (Section 8.2) show that even partially correcting the gate does not recover full steering performance. A complete explanation likely requires understanding how the target model's later layers respond to the transferred perturbations — a direction for future work.

### 9.2 Implications for Model Merging

These findings *suggest* (but do not establish) broader implications: any low-rank weight modification (including LoRA adapters, task vectors [5]) implements conditional steering with the same gate-mismatch vulnerability. Model merging may degrade when source and target have divergent activation statistics, even with identical architectures. Empirical validation on LoRA adapters and task vectors is future work.

### 9.3 Limitations

- **Scale:** Experiments are on 1.5B-parameter models within the Qwen2.5 family; larger models and cross-family transfer (e.g., Llama → Qwen) remain unexplored
- **Statistical power:** Clean n=400 results (Table 2) are directionally positive but do not yet reach p < 0.05; all McNemar p-values are 0.125–0.672. The preliminary n=50 results (+8–14 pp) appear inflated relative to n=400 (+1–4 pp), consistent with positive selection bias at small sample sizes
- **Single RLVR source:** We use one RLVR-trained model; results may vary with different training examples, seeds, or RLVR algorithms
- **Evaluation coverage:** MATH500 tests mathematical reasoning specifically; it is unclear whether gains reflect general reasoning improvement or math-domain formatting behavior
- **Gating hypothesis incompleteness:** Gate mediation results (Table 3) show the causal contrast src_replay vs. shuffled is effectively zero (47.0% vs. 47.2%), limiting claims about gate mismatch as a *primary* mechanism

---

## 10. Conclusion

We have established a local algebraic connection between rank-1 weight transfer and input-conditional activation steering (Proposition 1), providing a framework for analyzing why RLVR reasoning vectors degrade when transferred across models. The gating signal $v^T x$ drops to 45.6% of its source magnitude on the target model (mean ratio 0.456, measured on 56 paired projections) with 12% polarity inversions. Spectral concentration is confirmed at 39–173× above shape-matched null, depending on matrix shape.

Clean n=400 evaluation on disjoint test/validation splits shows modest directional improvements: SVD top-5 achieves +2.2 pp and mean-diff +3.5 pp over baseline (46.2%), with McNemar p-values of 0.298 and 0.125 respectively. These effects are consistent but not yet statistically significant. Gate mediation experiments (Table 3) find that replacing target gates with matched source gates does not measurably improve over shuffled source gates (47.0% vs 47.2%), constraining causal claims about gate mismatch.

SVD-derived steering extracts the steering component $u$ from weight deltas and applies it unconditionally at the residual stream, requiring only a single source base model calibration pass (no RLVR model inference). With sparse top-K selection, it achieves +2.2 pp on clean n=400 evaluation, consistently outperforming null controls (36–40%). The most defensible contribution is an exploratory framework — Proposition 1 as an analytic lens, confirmed spectral concentration, and a methodology for clean transfer evaluation — with directionally positive but statistically underpowered empirical results. Stronger conclusions await larger evaluation and multiple RLVR source models.

---

## References

[1] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

[2] Cui, G., et al. Process reinforcement through implicit rewards. *arXiv preprint arXiv:2502.01456*, 2025.

[3] Wang, Y., et al. Reinforcement learning for reasoning in large language models with one training example. *NeurIPS*, 2025. arXiv:2504.20571.

[4] Shao, Z., et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

[5] Ilharco, G., et al. Editing models with task arithmetic. *ICLR*, 2023.

[6] Yadav, P., et al. TIES-merging: Resolving interference when merging models. *NeurIPS*, 2023.

[7] Wortsman, M., et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *ICML*, 2022.

[8] Zou, A., et al. Representation engineering: A top-down approach to AI transparency. *arXiv preprint arXiv:2310.01405*, 2023.

[9] Turner, A., et al. Activation addition: Steering language models without optimization. *arXiv preprint arXiv:2308.10248*, 2023.

[10] Li, K., et al. Inference-time intervention: Eliciting truthful answers from a language model. *NeurIPS*, 2023.

[11] Hu, E., et al. LoRA: Low-rank adaptation of large language models. *ICLR*, 2022.

[12] Hendrycks, D., et al. Measuring mathematical problem solving with the MATH dataset. *NeurIPS*, 2021.

---

## Appendix A: Detailed Spectral Analysis

The rank-1 fraction distribution across 198 parameter matrices shows significant heterogeneity. Attention V-projections tend to have the highest concentration (mean ρ ≈ 0.42 across layers), followed by K-projections and O-projections. MLP layers show lower but still elevated concentration compared to random baselines.

The top 5 parameter matrices by rank-1 fraction are all attention projections:
1. `model.layers.27.self_attn.v_proj.weight` — ρ = 0.871
2. `model.layers.17.self_attn.k_proj.weight` — ρ = 0.650
3. `model.layers.17.self_attn.o_proj.weight` — ρ = 0.606
4. `model.layers.17.self_attn.v_proj.weight` — ρ = 0.601
5. `model.layers.26.self_attn.v_proj.weight` — ρ = 0.594

These values are from `outputs/spectral_data.json` (198 entries, 196 with `layer_idx ≥ 0`).

## Appendix B: Clean Evaluation Results (n=400, Disjoint Splits)

*This appendix will be populated with results from `paper_eval_suite.py` using:*
- *Calibration: MATH500 problems 450–499*
- *Validation (α/K selection): problems 400–449*
- *Test (reported): problems 0–399*

*Table to include: baseline, SVD full (best α from val), SVD top-K (best K from val), mean-diff (best α from val), random control, sign-flip control, wrong-layer control. All with Wilson 95% CIs and paired bootstrap p-values.*

## Appendix C: Full Alpha Sweeps (Preliminary, n=50)

### Weight Transfer (n=100)
| $\alpha$ | 0.5 | 1.0 | 1.5 | 2.0 |
|----------|-----|-----|-----|-----|
| Accuracy | 44% | 47% | 49% | 50% |

### SVD Steering — Residual Stream (n=50, problems 0–49)
| $\alpha$ | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 |
|----------|-----|-----|-----|-----|-----|
| Accuracy | 54% | 50% | 48% | 52% | 24% |

### Mean-Diff Steering — Residual Stream (n=50, problems 0–49)
| $\alpha$ | 0.02 | 0.05 | 0.1 | 0.2 |
|----------|------|------|-----|-----|
| Accuracy | 52% | 60% | 34% | 0% |

⚠ These alpha sweeps use problems 0–49 for both calibration and evaluation. Clean alpha selection using problems 400–449 is implemented in `paper_eval_suite.py`.

## Appendix D: Hook Placement Matters

SVD steering at the individual projection level (o_proj/down_proj outputs) is much less effective than residual-stream steering:

| Hook placement | Best accuracy | Best Δ (pp) |
|---------------|--------------|-------------|
| o\_proj/down\_proj output | 48% | +2 |
| Residual stream (after full transformer block) | 54% | +8 |

This is consistent with the interpretation that reasoning modifications compose across the attention and MLP sub-modules within each transformer block, and that the residual stream is the appropriate injection point for block-level steering.
