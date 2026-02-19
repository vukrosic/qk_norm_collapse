# QK-Norm Might Worsen Muon Optimizer LLM Training

> **Note:** This is an exploratory blog post documenting early-stage observations from a single-seed experiment. The findings are directional signals, not established results. We run one seed per condition and train for only 50M tokens (~0.3% of Chinchilla-optimal for a 1.5B model). All claims should be read as "we observed X" rather than "X is true in general."

---

## Key Findings at a Glance

![Key Rank Collapse - Does γ or Normalization Drive It?](./images/2_key_pr_causal.png)

Each attention head in a transformer uses a **128-dimensional space** to represent tokens. Ideally, the model spreads its representations across all 128 dimensions. In practice, many dimensions shrink to near-zero variance - the model "collapses" into a lower-dimensional subspace. The y-axis in the figure above measures how many of those 128 dimensions are actively carrying variance (higher = more dimensions in use). **Important:** near-zero variance doesn't mean a dimension is useless - it could still encode critical features. PR measures geometric spread, not information content.

**QK-Norm** is a technique that does two things to the attention vectors: (1) it **normalizes** them (controls their magnitude), and (2) it multiplies each dimension by a **learnable weight called γ** - essentially giving the model a volume knob for each dimension. We wanted to know: when QK-Norm helps, which part is actually responsible?

If you need explanation of the QK-Norm and γ (gamma) scroll below first.

We trained a **1.5B parameter** (full specs below) language model three times - once with full QK-Norm, once with normalization but the γ knobs locked at 1.0 (so the model can't adjust them), and once with no QK-Norm at all. Here is what we found:

1. **The γ knobs - not the normalization - appear to control how many dimensions carry variance.** The orange line (full QK-Norm) recovers to 65.7 dimensions. The purple line (normalization but γ locked) recovers to only 59.6 - *worse* than the cyan line (no QK-Norm at all, 63.3). Locking the γ knobs makes the model *worse* than not normalizing in this single-seed experiment.

2. **But normalization - not γ - is what makes the model learn better.** Both normalized models (orange and purple) achieve nearly identical loss, and both beat the no-QK-Norm model. The γ knobs don't affect learning speed at all on this small scale, undertrained model.

- It seems that the model still learns to predict next token well, even if some dimensions inside of the heads are becoming near-zero (holding less data).

3. **Every model goes through the same collapse-then-recovery pattern.** All three start with ~87 dimensions in use, crash to ~51–55, then recover. This U-shape appears in all three conditions, suggesting it may be a general feature of early transformer training (though this needs to be investigated further).

4. **We cannot link PR to model quality.** This is the most important caveat: the model with the *fewest* active dimensions (Frozen γ, PR=59.6) achieves the *best* loss (3.614). **Higher PR does not mean a better model in our experiment.** γ controls a geometric property (how evenly variance is spread) that we cannot yet connect to downstream performance. Whether collapsed dimensions represent wasted compute or efficient compression remains an open question - and the answer may well be "the model doesn't need all 128 dimensions."

Below we define every technical concept, then walk through the figures step by step.

---

### Concepts You Need First

#### What is γ (gamma)?

In a transformer attention head, each token is projected into a 128-dimensional **key** vector. **QK-Norm** applies RMSNorm to this vector, which does two things:

1. **Normalize** - divide every element by the root-mean-square of the whole vector, so the vector has a controlled magnitude.
2. **Scale by γ** - multiply each of the 128 elements by its own learnable weight $\gamma_j$.

$$\text{RMSNorm}(\mathbf{x})_j = \gamma_j \cdot \frac{x_j}{\sqrt{\frac{1}{128}\sum_{i=1}^{128} x_i^2}}$$

Think of γ as a **per-dimension volume knob**. Each of the 128 dimensions gets its own knob, initialized to 1.0 (all equal). As training progresses, the model can turn some knobs up (amplify that dimension) and others down (suppress it), giving it direct control over which dimensions matter.

**Concrete example** with a 4-dimensional vector $\mathbf{x} = [2, 4, 6, 8]$:
- RMS = $\sqrt{(4 + 16 + 36 + 64)/4} = \sqrt{30} \approx 5.48$
- After normalization: $[0.365,\ 0.730,\ 1.095,\ 1.461]$ (all values rescaled)
- Now suppose training learns $\gamma = [1.5, 1.0, 0.5, 0.1]$:
  1. Dim 1: $0.365 × 1.5 = 0.548$ ← **amplified**
  2. Dim 2: $0.730 × 1.0 = 0.730$ ← unchanged
  3. Dim 3: $1.095 × 0.5 = 0.548$ ← **dampened**
  4. Dim 4: $1.461 × 0.1 = 0.146$ ← **nearly removed**
- Final: $[0.55,\ 0.73,\ 0.55,\ 0.15]$ - γ has reshaped which dimensions carry information.

#### What is Participation Ratio (PR)?

PR measures **how many of the 128 dimensions carry significant variance** in the key vectors. It answers: "is the model spreading variance across many dimensions, or concentrating it into a few?"

- **PR = 128** → all dimensions contribute equally (maximum diversity, full rank)
- **PR = 1** → strictly means only one singular value is nonzero and all others are exactly zero. In practice, singular values are never exactly zero - they are just very small. So a real model might show PR = 2–3 in a severe collapse, meaning variance is overwhelmingly concentrated in 2–3 dimensions while the remaining ~125 carry negligible (but not zero) variance.
- **PR ≈ 60** (what we observe) → roughly 60 of 128 dimensions carry significant variance

When PR drops, variance concentrates into fewer dimensions - many key dimensions shrink to near-zero variance. This is called **dimensional collapse** (or rank collapse). **Important caveat:** a low-variance dimension is not necessarily a useless dimension. It could still encode subtle but critical distinctions (e.g., binary features). PR measures geometric spread, not information content. Whether collapsed dimensions represent wasted compute or efficient compression remains an open question.

#### The Three Experimental Conditions

We train the same 1.5B model three times, changing only how QK-Norm is configured:

| Condition | Normalization? | γ learned? | Purpose |
|:---|:---:|:---:|:---|
| **Learned γ** | ✅ Yes | ✅ Yes | Full QK-Norm - the real thing |
| **Frozen γ=1** | ✅ Yes | ❌ No (locked at 1.0) | Ablation - has normalization but *not* the learned scaling |
| **No QK-Norm** | ❌ No | ❌ No | Baseline - no normalization at all |

**Why three conditions?** Because QK-Norm does two things at once (normalize + scale by γ). With only two conditions (on/off) we can't tell which part matters. The Frozen γ=1 condition surgically removes γ learning while keeping normalization, letting us isolate the cause.

---

### Figure 1: The Central Result - Does γ or Normalization Drive Rank?

![Key Rank Collapse - Does γ or Normalization Drive It?](./images/2_key_pr_causal.png)

**What this figure shows:** Lower PR (y-axis) possibly indicates more wasted compute or useless dimensions, but this needs further investigation, so let's define it strictly: The y-axis is *Participation Ratio (PR)* is the number of dimensions carrying significant variance (out of 128). Higher means variance is more evenly spread; lower means it is concentrated into fewer dimensions. The x-axis is training progress in millions of tokens.

**Reading it step by step:**

1. **All three models start at PR ≈ 87** (top-left). At random initialization, each of the 128 dimensions contributes roughly equally - the model hasn't learned anything yet.

2. **All three crash to PR ≈ 51–55 by ~8M tokens** (bottom of the U-curve). This is *rank collapse* - the model destroys its initial random structure as it begins learning. This happens regardless of whether QK-Norm is used.

3. **After the collapse floor, the three lines diverge** - this is where the experiment reveals its answer:
   - 🟠 **Learned γ (orange)** recovers the fastest and highest, reaching PR = **65.7**
   - 🔵 **No QK-Norm (cyan)** recovers to PR = **63.3** - surprisingly, the second-best
   - 🟣 **Frozen γ=1 (purple)** recovers the least, only to PR = **59.6**

4. If normalization were the key mechanism for recovery, the two normalized models (Learned γ and Frozen γ) would recover similarly. Instead, removing γ learning (Frozen) makes it *worse* than having no normalization at all. **In this experiment, the learned γ parameter - not the normalization - appears to be what drives rank recovery.** (Caveat: this is a single-seed result; the 6-unit difference could narrow or widen with different seeds.)

### Figure 2: Loss Tells a Different Story

![Training & Validation Loss](./images/1_loss.png)

**What this figure shows:** Training and validation loss (lower = better language modeling) over the same 50M tokens.

**Reading it step by step:**

1. **Both normalized models (Learned γ and Frozen γ) achieve nearly identical loss** - around 3.62 for training, 3.44 for validation. They overlap almost perfectly.

2. **The No QK-Norm model has noticeably higher loss** - 3.68 training, 3.51 validation. Normalization clearly helps the model learn better.

3. **But here's the key insight:** Compare this to Figure 1 above. For *loss*, normalization helps equally whether γ is learned or frozen. For *rank*, only learned γ helps - frozen γ actually hurts. **This suggests loss and rank are driven by different mechanisms:**
   - **Normalization → stabilizes gradients → lowers loss** (doesn't need γ)
   - **Learned γ → selectively amplifies/suppresses dimensions → maintains dimensional diversity** (needs γ)

4. **An open question emerges:** The Frozen γ=1 model has the *lowest* PR (59.6) but achieves the *best* loss (3.614). If the collapsed dimensions contained critical information, we would expect worse loss. This suggests either (a) those dimensions are genuinely redundant at this training stage, (b) 50M tokens is too early for the rank difference to manifest in loss, or (c) PR captures geometric properties that don't directly map to task-relevant information. We cannot yet determine which explanation is correct.

### The Three Main Observations, Explained

> **Observation 1: "γ drives dimensional diversity, normalization drives loss."** These two effects appear independent. Normalization lowers loss equally whether γ is learned or frozen. Learned γ raises PR - but **we cannot yet link higher PR to better model quality** (Frozen γ has the *lowest* PR yet the *best* loss).

> **Observation 2: "PR follows a U-shaped trajectory."** All three conditions show the same collapse-then-recovery arc, suggesting this may be a general feature of early transformer training rather than a QK-Norm artifact.

> **Observation 3: "Normalization without γ appears counterproductive for rank."** Freezing γ at 1 yields *worse* PR than no QK-Norm at all. Hypothesis: normalization constrains keys to a sphere; without γ to selectively stretch dimensions, this constraint limits spectral diversity. This is unverified.

### Additional Observation: Depth-PR Gradient Inversion

> At 500K tokens the standard gradient holds (shallow layers ≈ PR 122, deep layers ≈ 106). By 50M tokens this **inverts** - deeper layers develop *higher* PR. **Caveat:** the two measurements come from separate experiments with slightly different configs; a rigorous analysis would require continuous per-layer tracking within a single run.

---

## 1. Overview

We investigate how **QK-Normalization** affects dimensional collapse in transformer attention heads. We train a **1.5B parameter** Gemma-style LLM for **~49M tokens** under three conditions (Learned γ, Frozen γ=1, No QK-Norm) to disentangle normalization from the learned scale parameter. This is a single-seed ablation - findings are preliminary.

### Architecture & Training Config

| Parameter | Value |
|:---|:---|
| **Model Size** | ~1.58B Parameters |
| $d_\text{model}$ | 2048 |
| Layers | 32 |
| Attention Heads | 16 ($d_k = 128$) |
| KV Heads | 8 (GQA) |
| $d_\text{ff}$ | 8192 |
| **Training Budget** | **~49M Tokens** per condition |
| **Hardware** | NVIDIA H100 80GB |
| Batch Size | 8 |
| Grad Accumulation | 4 |
| Effective Batch | $8 \times 4 \times 2048 = 65{,}536$ tokens/step |
| Optimizer | Muon + AdamW |
| Dataset | FineWeb/Cosmopedia Mix (1B subset) |

---

## 2. Results Summary

| Metric | Learned γ | Frozen γ=1 | No QK-Norm |
|:---|:---:|:---:|:---:|
| **Initial Mean PR** | 86.85 | 86.85 | 86.42 |
| **Minimum Mean PR** | 55.17 (7.9M) | 51.84 (9.8M) | 51.28 (7.9M) |
| **Final Mean PR (49M)** | **65.70** | 59.55 | 63.25 |
| **Final Train Loss** | **3.618** | **3.614** | 3.683 |
| **Final Val Loss** | **3.440** | **3.439** | 3.505 |

## 4. Mathematical Background

### 4.1 QK-Normalization

In standard attention, tokens are projected into queries and keys: $Q = XW_Q,\ K = XW_K$. With QK-Norm, RMSNorm is applied to both Q and K before RoPE:

$$Q = \text{RoPE}\!\left(\text{RMSNorm}(XW_Q)\right), \quad K = \text{RoPE}\!\left(\text{RMSNorm}(XW_K)\right)$$

$$\text{RMSNorm}(\mathbf{x})_j = \frac{\gamma_j \cdot x_j}{\text{RMS}(\mathbf{x})}, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d_k}\sum_{i=1}^{d_k} x_i^2}$$

Each RMSNorm has its own $\gamma \in \mathbb{R}^{128}$, initialized to 1. There are $32 \times 2 = 64$ such vectors (8,192 extra parameters total). As training progresses, individual $\gamma_j$ values diverge - amplifying some dimensions and suppressing others.

RoPE is applied **after** RMSNorm, so γ acts on the pre-positional representation.

### 4.2 Why This Matters for Rank

PR measures how evenly variance is distributed across the 128 key dimensions via SVD singular values $\sigma_1 \geq \cdots \geq \sigma_{128}$:

$$\text{PR}(K) = \frac{\left(\sum_{i=1}^{d_k} \sigma_i\right)^2}{\sum_{i=1}^{d_k} \sigma_i^2}$$

PR = 128 means all dimensions contribute equally; PR = 1 means total collapse. Each condition affects rank differently: without QK-Norm, rank is controlled only through $W_K$; with learned γ, the model has a direct per-dimension knob; with frozen γ=1, normalization constrains geometry without per-dimension flexibility.

## 5. γ Dynamics

The γ coefficient of variation (CV) reveals how the model uses the learnable parameter. High-CV layers (L0: 0.169, L15: 0.155, L27: 0.172) show aggressive dimension differentiation and correlate with higher PR recovery. Low-CV layers (L5–L10: 0.07–0.08) show minimal differentiation.

### 5.1 Final γ Values: What Did the Model Learn?

**γ values are mostly > 1.0 - the model amplifies rather than suppresses.** Global mean γ ≈ 1.19. Only Layer 0 has mean γ < 1.0 (0.913).

| Layer Zone | Layers | Mean γ | Interpretation |
|:---|:---:|:---:|:---|
| **Shallow** | 0–7 | 1.084 | Closest to initialization, Layer 0 is the outlier |
| **Middle** | 8–23 | 1.298 | Strongest amplification |
| **Deep** | 24–31 | 1.238 | Slightly less than middle layers |

**Layer 0 is uniquely asymmetric.** The first 64 dimensions (high-frequency RoPE) have mean γ = 1.017, while the last 64 (low-frequency RoPE) are suppressed to 0.810. All 14 dimensions with γ < 0.7 fall in the last 64. This suggests Layer 0 learns to de-emphasize low-frequency positional information.

**γ differentiation correlates with PR recovery.** Layers with the highest CV tend to show the most distinct PR behavior; layers with nearly uniform γ show minimal dimension differentiation.

**Key limitation:** We only have γ at the final checkpoint. Tracking γ trajectories over training would reveal when dimension selection happens - a clear next step.

## 6. Limitations

1. **Single Seed**: Effect sizes (2–6 PR units) may shift or reverse with different seeds.
2. **Early Training**: 50M tokens is ~0.3% of Chinchilla-optimal; we observe early-phase dynamics only.
3. **Key-Only Analysis**: We probe Keys but not Queries; the effective rank of $QK^T$ may tell a different story.
4. **Muon Optimizer**: Findings may be Muon-specific artifacts.
5. **PR ≠ quality**: Frozen γ has the *lowest* PR but the *best* loss. Higher PR does not imply a better model in our data.
6. **No singular value distributions saved**: We cannot distinguish gradual taper from sharp cutoff.
7. **No downstream evaluation**: We measure loss only, no benchmarks.

### Future Work

- **Multiple seeds** (≥3) for variance estimation - the most important next step.
- **Extend training to 250M–1B+ tokens** to test plateau stability and whether PR/loss dissociation resolves.
- **Measure query PR** and effective rank of $QK^T$ directly.
- **Save full singular value distributions** and **track individual γ trajectories** over training.
- **Pure AdamW baseline** to isolate Muon-specific effects.
- **Downstream benchmark evaluation**.
