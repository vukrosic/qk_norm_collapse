# Head-wise Tensor Orthogonalization: Final Experiment Plan (V1.2)

**Status:** Ready for Execution
**Method:** Mode-1 HeadOrtho (Data-Driven Decision)

## 1. The Core Insight
We hypothesized that orthogonalizing gradients across attention heads would improve training. [based on what is this hypothesis?]
- **Teon Theory**: Tensor orthogonalization is optimal when singular vectors align.
- **Our Diagnostic (Step 0)**: We measured alignment on this exact model/data.
  - Left singular vectors (U): **0.07** (weak)
  - Right singular vectors (V): **0.93** (STRONG)
  
**Conclusion:** The gradients of different heads share a massive amount of structure in their row-space (V). This strongly validates **Mode-1** orthogonalization (unfolding along the head dimension), exactly as Teon recommended, but for a data-verified reason.

## 2. The Experiment

We will compare:
1. **Baseline**: Standard Muon (reuse existing results if available, else run fresh).
2. **HeadOrtho**: Our method using Mode-1 tensor orthogonalization.

**Hypothesis**: HeadOrtho will exploit the 0.93 alignment to accelerate convergence better than independent per-head orthogonalization (standard Muon).

## 3. Configuration
- **Model**: 1.5B parameters (d=2048, heads=16, layers=32)
- **Data**: 20M tokens (approx 1220 steps)
- **Batch Size**: 1 (grad_accum=8) → effective 16k tokens/step
- **Optimizer**: 
  - `qkvo_proj`: HeadOrtho Muon (Mode-1)
  - Other 2D: Standard Muon
  - 1D/Embed: AdamW

## 4. Execution Steps

Run the following command on your GPU machine:

```bash
# Run both baseline and HeadOrtho
python new_research/head_orthogonal/run_experiment.py --experiment all

# OR if you already have baseline results:
python new_research/head_orthogonal/run_experiment.py --experiment headortho
```

## 5. Success Criteria
Does the **HeadOrtho** val loss curve drop below the **Baseline** curve?
- **Yes**: Discovery! Cross-head structure is exploitable.
- **No**: The alignment (0.93) was "handled" sufficiently by standard Muon.

## 6. Code State
- `optimizers/head_ortho_muon.py`: **Updated to Mode-1**
- `run_experiment.py`: **Updated to Mode-1**
- `step0_diagnostic.py`: **Completed** (Confirmed Mode-1)
