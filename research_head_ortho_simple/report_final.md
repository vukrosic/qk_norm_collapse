# HeadOrtho: The "Rank Collapse is Optimal" Hypothesis

## Overview
We hypothesized that the rapid "rank collapse" observed in HeadOrtho (Mode-1) was due to the optimizer exploiting a relaxed constraint: fitting the heads into a shared, lower-dimensional subspace to minimize loss efficiently. 

To test this, we ran **Mode-3 HeadOrtho**, designed to strictly **prevent** this collapse by forcing heads to be orthogonal to *each other*.

## Experimental Results (at ~2.6M Tokens)

| Method | Constraint Type | Rank (PR) | Loss | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Mode 1** | Input-Side Orthogonality (Shared) | **LOW (~14.5)** | **BEST (Low 5.xx)** | **Collapse is Good** |
| **Baseline** | Global Matrix Orthogonality | Medium (~48.9) | Good (5.88) | Standard |
| **Mode 3** | Inter-Head Orthogonality | **HIGH (~51.0)** | **WORST (~10.7)** | **Rank is Bad** |

## Analysis

### 1. Mode 3 Preserves Rank but DESTROYS Performance
As predicted, Mode-3 successfully forced the Participation Ratio (PR) to stay high (rebounding to >50). However, the training loss remained catastrophically high (~10.7) compared to the others (<6.0).

### 2. The Interpretation
This confirms our hypothesis: **The "Rank Collapse" in Mode-1 is not a failure, but a feature.** 
- The model *wants* to align its heads to learn the early features efficiently.
- **Mode-1** permits this alignment (by putting all heads in one vector), accelerating learning.
- **Mode-3** forbids this alignment (by forcing heads to be orthogonal), stalling learning.
- **Baseline** is somewhere in the middle.

### Conclusion
**We should embrace the collapse.** The low-rank structure found by Mode-1 is a more efficient path to the loss minimum. "Head Diversity" (at least in the sense of orthogonal update steps) appears to be harmful for early pretraining.

![3-Way Comparison](results/mode1_v_mode3_v_baseline.png)
