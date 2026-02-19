# 93% Redundancy: Can Head-Wise Tensor Orthogonalization Accelerate Transformer Training?

## Abstract
Recent optimization breakthroughs like **Muon** have shown that orthogonalizing weight updates can significantly accelerate training. However, standard Muon treats every attention head's parameters as independent, ignoring the fact that heads often learn highly redundant features ("head collapse"). 

In this 24-hour research sprint, we generalize the **Teon** optimizer (Tensorized Muon) to operate across **attention heads**. Our preliminary diagnostic on a 1.5B parameter model reveals a striking **93% alignment** in the input-side singular vectors of different heads, suggesting massive redundancy. We propose **HeadOrtho Muon**, an optimizer that stacks gradients from all heads into a tensor and orthogonalizes them jointly, forcing heads to update in diverse directions. We test whether exploiting this structure can improve training efficiency and prevent collapse.

---

## 1. The Problem: Head Independence is a Myth

In a standard Transformer, we treat the Query (Q) and Key (K) projections of each attention head as separate matrices. The optimizer updates Head 1's weights ($W_{Q}^{(1)}$) independently of Head 2's weights ($W_{Q}^{(2)}$).

But these heads are **not** independent.
1.  **Shared Input**: Every head reads from the exact same residual stream ($X$).
2.  **Head Collapse**: Research consistently shows that many heads end up learning identical or highly correlated attention patterns, wasting capacity.

If Head 1 and Head 2 are trying to learn the same feature, their gradients will be nearly identical. Standard optimizers (Adam, Muon) will just let them collide. We need an optimizer that *knows* about this redundancy and forces them apart.

## 2. The Solution: Tensor Orthogonalization (Teon)

The **Muon** optimizer works by forcing the weight update matrix $G$ to be orthogonal ($G^T G = I$). This ensures the update step preserves the spectral structure of the weights.

The **Teon** (Tensor Muon) paper introduced a powerful generalization: instead of orthogonalizing a 2D matrix, we can orthogonalize a **3D tensor**.
- If we stack correlated matrices (like layers) into a tensor $\mathcal{G} \in \mathbb{R}^{d_{in} \times d_{out} \times K}$.
- And orthogonalize the *entire stack jointy*.
- Theoretical proofs show a convergence speedup of $\sqrt{K}$ **if** the stacked matrices are correlated (i.e., share singular vectors).

**Our Hypothesis**: The most correlated structures in a Transformer aren't layers—they are **attention heads**.

## 3. Novelty: How is this different from Teon?

While Teon introduced the mathematical framework of tensor orthogonalization for neural networks, it applied it exclusively across **layers** (depth).

| Feature | **Teon (Original Paper)** | **HeadOrtho (Our Work)** |
| :--- | :--- | :--- |
| **Tensor Axis** | **Layers** (Depth) | **Attention Heads** (Width) |
| **Why?** | Layers processed sequentially share features. | Heads processed in parallel learn redundant features. |
| **Alignment Source** | Slow residual stream evolution. | **93% Shared Right Singular Vectors** (Diagnostic discovery). |
| **Mechanism** | Orthogonalize $G_{layer}$ tensor. | Orthogonalize $G_{head}$ tensor. |

**Our specific novel contribution** is identifying that **attention heads within a layer** exhibit far stronger structural redundancy (93% alignment in our diagnostic) than layers typically do. Teon never explored head-wise orthogonalization. We hypothesize that this "width-wise" redundancy is a more potent target for tensor optimization than "depth-wise" redundancy.

## 4. The Smoking Gun: 93% Gradient Alignment

Before building the optimizer, we ran a diagnostic on a 1.5B parameter model to verify if head gradients are actually correlated. We computed the gradients for all 16 attention heads and decomposed them using SVD ($G = U \Sigma V^T$).

We measured the **Cosine Similarity** of the singular vectors across different heads:

| Component | Alignment Score | Meaning |
| :--- | :---: | :--- |
| **Left Singular Vectors ($U_{grad}$)** | 0.07 | **Low.** Heads project outputs to different subspaces. |
| **Right Singular Vectors ($V_{grad}$)** | **0.93** | **EXTREME.** Heads are sensitive to the **same input directions**. |

**This 93% number is massive.** It means that structurally, every head is trying to update its weights to respond to the *exact same features* in the residual stream. This is the definition of redundancy.

## 5. The Method: HeadOrtho Muon

Leveraging this 93% alignment, we propose **HeadOrtho Muon**. Instead of optimizing each head $(d_{k} \times d_{model})$ independently, we:

1.  **Stack**: Collect Q gradients from all $H$ heads into a tensor $\mathcal{G} \in \mathbb{R}^{H \times d_k \times d_{model}}$.
2.  **Flatten (Mode-1)**: Reshape into a giant matrix $M \in \mathbb{R}^{d_k \times (d_{model} \cdot H)}$. This effectively concatenates the input dimensions of all heads.
3.  **Orthogonalize**: Apply Newton-Schulz iteration to $M$ to make it semi-orthogonal.
4.  **Unstack**: Reshape back to per-head gradients.

**Why this works**: By orthogonalizing the *joint* matrix $M$, we enforce that the update directions for Head 1 and Head 2 are orthogonal *in the shared input space*. The optimizer literally subtracts the common direction from Head 2 that Head 1 is already covering.

**It forces diversity mathematically.**

## 6. Experimental Setup

We test this on a **1.5B parameter** causal language model (GPT-style).

-   **Model**: $d_{model}=2048$, 16 heads, 32 layers.
-   **Context**: 2048 tokens.
-   **Training**: 20M tokens (~1200 steps).
-   **Baseline**: Standard Muon (per-parameter orthogonalization).
-   **Ortho Mode**: We use Mode-1 based on the 93% $V$-vector alignment (since Mode-1 preserves the $d_k$ mixing while orthogonalizing the shared $d_{model}$ input space).

## 7. Research Questions

1.  **Does HeadOrtho improve validation loss?** (Can we learn faster by removing redundancy?)
2.  **Does it prevent Head Collapse?** (Do the resulting heads have more diverse attention patterns?)
3.  **Is the 93% alignment reduced over time?** (Does the optimizer successfully de-correlate the heads?)

---
*Code and diagnostic scripts are available in `new_research/head_orthogonal/`.*
