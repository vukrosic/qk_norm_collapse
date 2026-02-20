# I Did 3 LLM Training Experiments on Attention Head Orthogonality 🧠

I did 3 LLM training experiments: standard Muon vs Head Orthogonality modes (Input-Shared and Inter-Head). I wanted to see how different optimizer constraints affect the attention heads in a Transformer model.

To give some background: The Muon optimizer fundamentally works by taking a matrix of gradients and forcing its rows to be orthogonal (mathematically perpendicular, meaning they are independent and non-overlapping). By creatively reshaping the gradient tensor before feeding it to Muon, we can change the definition of what a "row" is. We can control exactly *what* is forced to be independent from *what*.

I tested this on the attention projections of a ~650M parameter model. The experiments specifically applied these head orthogonality constraints independently to both the **Query and Key projections ($W_Q$ and $W_K$)**, while letting the Value and Output projections train with standard global Muon. 

Here are the 3 setups I compared:
1. **Standard Muon (Baseline)**: Global orthogonality. Every single dimension of every single head must be independent from everything else. Maximum forced diversity.
2. **Mode 1 (Input-Shared / HeadOrtho)**: Shared subspace. It forces the $i$-th feature dimension to be independent from the $j$-th feature dimension, but it allows identically-indexed features in *different heads* to be exactly the same. Heads can naturally align and collaborate.
3. **Mode 3 (Inter-Head)**: Independent heads. Head 1 as a whole must be orthogonal to Head 2 as a whole. However, the features *within* a single head are free to be perfectly correlated.

### The Results

I tracked the Training Loss (optimization efficiency) and the Participation Ratio (PR), which is a way to measure the "effective rank" of the projection matrices. If PR stays high, the matrix is fully utilizing its diverse capacity. If PR drops, it means the representations are collapsing and correlating.

![Loss and PR comparison up to 300M tokens](results_1b/comparison_300m_tokens.png)

We trained these configurations on a C4-like pretraining corpus for about 300 million tokens. 
*(Note: 300M tokens is not nearly enough for an LLM of this size. We are currently compute constrained, but I'm working to fix it for my future posts! However, this short-scale window still gives us very strong signals regarding optimization dynamics.)*

#### Quantitative Results at 300M Tokens
| Metric | **Mode 1 (Input-Shared)** | **Mode 3 (Inter-Head)** | **Baseline (Standard Muon)** |
| :--- | :---: | :---: | :---: |
| **Final Loss** | **3.073** | 3.100 | 3.096 |
| **Final PR** | 31.6 | 26.1 | **57.2** |

### What the Graphs and Table Tell Us

If we analyze the data above, a few very clear (and somewhat deeply weird) behaviors emerge:

**1. Muon is Very Good at Keeping Rank High (Forced Diversity)**
* The **Baseline (cyan squares)** aggressively fights the model's natural tendency to simplify. By enforcing strict, massive-scale orthogonality across everything, standard Muon proves it is incredibly good at keeping the Participation Ratio high (~57.2). 
* By comparison, when we relax the constraints within the heads (Mode 1 and Mode 3), the model happily throws away its diversity and experiences a devastating rank collapse: Mode 1's PR drops to ~31.6, and Mode 3 drops to ~26.1. 

**2. It's Bizarre, but More Collapsed Heads Give Better Loss!**
* Here is the kicker: despite undergoing a massive rank collapse, **Mode 1 achieves the lowest training loss** (3.073) and descends the fastest. 
* It is weird and totally counter-intuitive! We often assume that preventing rank collapse and forcing diverse representations is objectively good for modern LLMs. But the Baseline, which enforces maximum diversity, struggles slightly over the same period (3.096) and flat-out loses to Mode 1.

**The Takeaway**
The 300M token runs suggest that strictly enforcing parameter-wide global orthogonality (Baseline) artificially sustains diversity at a slight cost to actual loss minimization. 

Instead, relaxing the constraint to let heads naturally share representations across the same feature dimensions (Mode 1)—even though it causes a massive drop in matrix rank—yields a more favorable path for the optimizer. Sometimes, letting the network correlate exactly what it wants to correlate is the best move.

---

# Head Orthogonality in Muon Optimization: Experimental Setup

## Abstract
This document outlines the experimental setup used to investigate how different tensor orthogonalization constraints affect the training dynamics of attention heads in Transformer models optimized with Muon. We compare three distinct orthogonalization modes: Standard (Baseline), Input-Shared (Mode 1), and Inter-Head (Mode 3).

---

To understand these ablations intuitively, you have to look at what the Muon optimizer is fundamentally doing. Muon takes a matrix of gradients and forces its rows to be orthogonal (mathematically perpendicular, meaning they are independent and non-overlapping).

By reshaping the gradient tensor before giving it to Muon, we are changing the definition of a "row." Therefore, we are changing who is forced to be independent from whom.






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

---

## 4. Extended Training Results (~300M Tokens)

To evaluate the constraints over a longer optimization trajectory, we extended the training to approximately 300 Million tokens.

![Loss and PR comparison up to 300M tokens](results_1b/comparison_300m_tokens.png)

### 4.1 Quantitative Results at 300M Tokens

| Metric | **Mode 1 (Input-Shared)** | **Mode 3 (Inter-Head)** | **Baseline (Global)** |
| :--- | :---: | :---: | :---: |
| **Final Loss** | **3.073** | 3.100 | 3.096 |
| **Final PR** | 31.6 | 26.1 | **57.2** |

### 4.2 Analysis and Discussion

1.  **Optimization Efficiency (Loss)**:
    *   **Mode 1** achieves the lowest final loss (3.073) among the three constraints. By forcing the same feature dimension across heads to be an orthogonal unit (and allowing heads to align), we potentially align the optimizer's updates with a more natural parameter subspace.
    *   **Baseline and Mode 3** converge to slightly higher final loss values (3.096 and 3.100 respectively). For Mode 3, strictly separating attention heads might prevent useful correlations between heads that improve learning efficiency.
2.  **Effective Rank (Participation Ratio)**:
    *   **The Baseline** successfully maintains a high matrix diversity (PR ~57.2). By making every row element orthogonal to every other element, it heavily resists the network's natural tendency toward rank collapse.
    *   **Mode 1 and Mode 3** both demonstrate substantial rank collapse, settling at PRs of approximately 31.6 and 26.1, respectively. This confirms the hypotheses drawn from the dimensional constraints:
        *   In **Mode 1**, since heads are horizontally stacked in the same subspace, they can share highly overlapping representations, allowing the rank to drop significantly compared to Baseline.
        *   In **Mode 3**, although entire heads are strictly orthogonal to other heads, the raw features *within* a single head are free to be completely correlated. This leads to an even sharper reduction in rank.

**Conclusion**: The extended 300M-token runs provide compelling evidence that enforcing parameter-wide global orthogonality (Baseline) artificially sustains a high projection matrix rank. By relaxing this constraint to allow structural groupings (Mode 1 and Mode 3), the network undergoes a clear rank collapse. Furthermore, since Mode 1 achieves the lowest loss despite significant rank drop, it suggests that allowing inter-head correlations—while maintaining input-feature diversity—may be a more favorable inductive bias than enforcing strict independence across all parameters.
