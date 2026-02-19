# Head Orthogonality in Muon Optimization: Experimental Setup

## Abstract
This document outlines the experimental setup used to investigate how different tensor orthogonalization constraints affect the training dynamics of attention heads in Transformer models optimized with Muon. We compare three distinct orthogonalization modes: Standard (Baseline), Input-Shared (Mode 1), and Inter-Head (Mode 3).

---

## 1. Methodology: Three Modes of Orthogonality

We investigate how the shape of the gradient tensor $G$ during the Newton-Schulz iteration affects the learning dynamics of the Key/Query projections ($W_{QK} \in \mathbb{R}^{H \cdot d_k \times d_{model}}$).

### Summary of Dimensional Constraints

The core difference lies in which components are forced into the same "row" (and thus forced to be orthogonal to others) vs. being in the same "vector" (and thus allowed to align).

| Mode | Reshape Target | Hierarchy of Orthogonality | Effective Constraint |
| :--- | :--- | :--- | :--- |
| **Baseline** | $(H \cdot d_k) \times d_{model}$ | Everything $\perp$ Everything | **Hardest constraint.** Maximum head diversity. |
| **Mode 1** | $d_k \times (H \cdot d_{model})$ | Row-Index $\perp$ Row-Index | **Shared subspace.** Heads are horizontally stacked; can collapse into each other. |
| **Mode 3** | $H \times (d_k \cdot d_{model})$ | Head $\perp$ Head | **Independent heads.** Each head is a unit; rows *within* a head can align. |

*   **Baseline** says: "Every single row (parameter vector) of every single head's weight projection matrix must be unique and orthogonal to all others."
*   **Mode 1** says: "The $i$-th row of Head 1 and the $i$-th row of Head 2 are part of the same long vector. They must be orthogonal to any $j$-th row ($i \neq j$), but Head 1 and Head 2 can use the same weights for the $i$-th dimension."
*   **Mode 3** says: "Head 1 as a whole (all its rows combined) must be orthogonal to Head 2 as a whole. Unlike the baseline, it doesn't care if rows *within* Head 1 are similar, only that the 'Head 1 subspace' is orthogonal to 'Head 2 subspace'."


### 1.1 Baseline: Global Matrix Orthogonality
**Standard Muon** treats the entire parameter matrix as a single 2D tensor.
$$G_{\text{base}} \in \mathbb{R}^{(H \cdot d_k) \times d_{model}}$$
where:
*   $H$ is the number of attention heads (e.g., 16).
*   $d_k$ is the dimension of each head (e.g., 128).
*   $d_{model}$ is the input hidden dimension (e.g., 2048).
*   **Each element $W_{QK}^{h,i}$ represents the $i$-th row vector of the QK projection matrix for head $h$.** It has size $d_{model}$.

The optimizer orthogonalizes the rows of this huge matrix.
*   **Constraint**: Every row (dimension of every head) is orthogonalized against every other row in the combined matrix.
*   **Effect**: This enforces a global diversity constraint across all $H \cdot d_k$ output dimensions simultaneously.

**Example ($H=2, d_k=2$):**
Everything must be orthogonal to everything else. Each row is a weight vector of size $d_{model}$.

| Matrix Row | Represents (Weight Vector) | Must be Orthogonal to: | Meaning |
| :--- | :--- | :--- | :--- |
| Row 1 | $W_{QK}^{1,1}$ | Row 2, Row 3, Row 4 | $W_{QK}^{1,1}$ must be different from $W_{QK}^{1,2}$, $W_{QK}^{2,1}$, etc. |
| **Row 2** | $W_{QK}^{1,2}$ | Row 1, Row 3, Row 4 | Features within Head 1 must be orthogonal. |
| **Row 3** | $W_{QK}^{2,1}$ | Row 1, Row 2, Row 4 | Head 2 must be orthogonal to Head 1. |
| **Row 4** | $W_{QK}^{2,2}$ | Row 1, Row 2, Row 3 | Features within Head 2 must be orthogonal. |

**Visual Matrix ($4 \times d_{model}$):**
$$
G_{\text{base}} = 
\begin{bmatrix}
w_{1,1,1} & w_{1,1,2} & \dots & w_{1,1,D} \\
w_{1,2,1} & w_{1,2,2} & \dots & w_{1,2,D} \\
w_{2,1,1} & w_{2,1,2} & \dots & w_{2,1,D} \\
w_{2,2,1} & w_{2,2,2} & \dots & w_{2,2,D}
\end{bmatrix}
\begin{matrix}
\leftarrow W_{QK}^{1,1} \text{ (Head 1, Row 1)} \\
\leftarrow W_{QK}^{1,2} \text{ (Head 1, Row 2)} \\
\leftarrow W_{QK}^{2,1} \text{ (Head 2, Row 1)} \\
\leftarrow W_{QK}^{2,2} \text{ (Head 2, Row 2)}
\end{matrix}
$$
All 4 rows are orthogonalized against each other.

### 1.2 Mode 1: Input-Shared Orthogonality ("HeadOrtho")
We reshape the gradient to separate the "head dimension" from the "feature dimension" and orthogonalize the $d_k$ feature dimensions across a shared input space.
$$G_{\text{mode1}} \in \mathbb{R}^{d_k \times (H \cdot d_{model})}$$
*   **Constraint**: The $i$-th feature dimension is orthogonalized against the $j$-th feature dimension, aggregating the vector across all heads.
*   **Structure**: This treats the $H$ heads as contributing to a single shared $d_k$-dimensional subspace. It does not enforce orthogonality *between* different heads for the same feature dimension.

**Example ($H=2, d_k=2$):**
We reshape to only 2 big rows ($d_k=2$). Each row is a *concatenation* of the vectors for that feature row across all heads.

| Matrix Row | Represents (Concatenated Vector) | Must be Orthogonal to: | Meaning |
| :--- | :--- | :--- | :--- |
| **Row 1** | $W_{QK}^{1,1} \oplus W_{QK}^{2,1}$ | Row 2 | Row 1 subspace must be orthogonal to Row 2 subspace. |
| **Row 2** | $W_{QK}^{1,2} \oplus W_{QK}^{2,2}$ | Row 1 | **$W_{QK}^{1,1}$ and $W_{QK}^{2,1}$ can be identical.** |

**Visual Matrix ($2 \times 2 \cdot d_{model}$):**
$$
G_{\text{mode1}} = 
\begin{bmatrix}
\underbrace{w_{1,1,1} \dots w_{1,1,D}}_{W_{QK}^{1,1} \text{ (Head 1)}} & \mathbf{|} & \underbrace{w_{2,1,1} \dots w_{2,1,D}}_{W_{QK}^{2,1} \text{ (Head 2)}} \\
\underbrace{w_{1,2,1} \dots w_{1,2,D}}_{W_{QK}^{1,2} \text{ (Head 1)}} & \mathbf{|} & \underbrace{w_{2,2,1} \dots w_{2,2,D}}_{W_{QK}^{2,2} \text{ (Head 2)}}
\end{bmatrix}
$$
Only Top Row $\perp$ Bottom Row. Head 1 and Head 2 are just different columns in the same row vector!

### 1.3 Mode 3: Inter-Head Orthogonality
We reshape to force heads to be orthogonal to each other.
$$G_{\text{mode3}} \in \mathbb{R}^{H \times (d_k \cdot d_{model})}$$
*   **Constraint**: The gradient update vector for `Head i` is orthogonalized against `Head j`.
*   **Structure**: This specifically targets the independence of attention heads, treating each head as a unit vector in the parameter space.

**Example ($H=2, d_k=2$):**
We reshape to only 2 big rows (one per Head, $H=2$). Each row is a *concatenation* of all feature rows in that head.

| Matrix Row | Represents (Concatenated Vector) | Must be Orthogonal to: | Meaning |
| :--- | :--- | :--- | :--- |
| **Row 1** | $W_{QK}^{1,1} \oplus W_{QK}^{1,2}$ | Row 2 | **Head 1 must be orthogonal to Head 2.** |
| **Row 2** | $W_{QK}^{2,1} \oplus W_{QK}^{2,2}$ | Row 1 | Features within a head can be anything (as long as orthogonal to H1). |

**Visual Matrix ($2 \times 2 \cdot d_{model}$):**
$$
G_{\text{mode3}} = 
\begin{bmatrix}
\underbrace{w_{1,1,1} \dots w_{1,1,D}}_{W_{QK}^{1,1}} & \mathbf{|} & \underbrace{w_{1,2,1} \dots w_{1,2,D}}_{W_{QK}^{1,2}} \\
\underbrace{w_{2,1,1} \dots w_{2,1,D}}_{W_{QK}^{2,1}} & \mathbf{|} & \underbrace{w_{2,2,1} \dots w_{2,2,D}}_{W_{QK}^{2,2}}
\end{bmatrix}
\begin{matrix}
\leftarrow \text{Head 1 Parameters} \\
\leftarrow \text{Head 2 Parameters}
\end{matrix}
$$
Top Row (Head 1) $\perp$ Bottom Row (Head 2).

---

---

## 2. Experimental Setup

To compare these modes, we conducted preliminary training runs using a 1.5B parameter Language Model.

### 2.1 Model Architecture
*   **Parameters**: ~1.5 Billion
*   **Layers**: 32
*   **Hidden Dimension ($d_{model}$)**: 2048
*   **Attention Heads**: 16
*   **Head Dimension ($d_k$)**: 128
*   **Sequence Length**: 2048

### 2.2 Training Configuration
*   **Dataset**: C4-like pretraining corpus (Blueberry-1B-Pretrain).
*   **Tokens Trained**: 10 Million (Short-scale study).
*   **Batch Size**: 16 (Effective Batch Size ~131k tokens).
*   **Optimizer**: Muon (for 2D parameters) + AdamW (for embeddings/normalization).
*   **Learning Rate**: Muon LR 0.05, AdamW LR 0.005.

### 2.3 Metrics Tracked
We monitor two primary metrics throughout training:
1.  **Training Loss**: Measuring the optimization efficiency.
2.  **Participation Ratio (PR)**: Estimating the effective rank of the Key/Query projection matrices to quantify "Rank Collapse" versus strict orthogonality.

---

## 3. Preliminary Observations (10M Tokens)

While the experiment is short-scale (10M tokens is <1% of typical pretraining), we observed distinct behaviors for each mode:

| Metric | **Mode 1** | **Baseline** | **Mode 3** |
| :--- | :---: | :---: | :---: |
| **Final Loss** | 5.71 | 5.88 | 5.74 |
| **Final PR** | 14.5 | 48.9 | 13.6 |

*   **Mode 1**: exhibited rapid rank reduction early in training.
*   **Baseline**: maintained a higher effective rank throughout.
*   **Mode 3**: initially maintained high rank but showed a trend of rank reduction in later steps.

*Note: These results are from a limited training duration and should be interpreted as initial observations rather than conclusive evidence of long-term model behavior.*
