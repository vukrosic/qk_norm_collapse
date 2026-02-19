# HeadOrtho: 1M Token Study Results

## Experiment
- **Model**: 1.5B (2048/16 heads), 32 layers
- **Data**: 1M tokens (Blueberry-1B-Pretrain)
- **Method**: HeadOrtho Muon (Mode-1 Tensor Orthogonalization on Q/K) vs Baseline Muon
- **Both runs use QK-Norm**

## Results

| Metric | HeadOrtho | Baseline | Delta |
| :--- | :---: | :---: | :---: |
| **Final Loss** | **7.0592** | 7.0831 | -0.0239 (Better) |
| **Final PR** | 32.1 | **55.4** | -23.3 (More Collapse) |
| **Initial PR** | 85.95 | 85.95 | 0.0 |

## Analysis

**Paradox Confirmed**: HeadOrtho (Mode-1) **accelerates dimensional collapse** (PR drops to 32.1 vs 55.4) yet **improves training loss** (7.05 vs 7.08).

This suggests that "Head Collapse" (redundancy) might be a *feature*, not a bug, for early training efficiency. By forcing `d_k` dimensions to be orthogonal across the combined `d_model * H` input space (Mode-1), we might be efficiently "compressing" the update information, allowing the model to learn faster by focusing on fewer, more important dimensions.

**Next Steps**:
- Try **Mode-3 Orthogonalization** (Orthogonalize Heads directly) to see if we can *prevent* collapse.
- If Mode-3 prevents collapse but *hurts* loss, it would strongly support the "Collapse is Optimal" hypothesis.
