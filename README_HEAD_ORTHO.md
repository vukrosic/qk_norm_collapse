# HeadOrtho Muon: 24-Hour Research Sprint

This experiment tests **Head-wise Tensor Orthogonalization** to improve 1.5B model training.
Based on our diagnostic finding: **Right singular vectors (V) are 93% aligned across heads!**

## 🚀 Quick Start (On GPU Machine)

1.  **Run Experiment:**
    ```bash
    # Runs the HeadOrtho variant (Mode-1)
    python new_research/head_orthogonal/run_experiment.py --experiment headortho
    ```

    *(If you need a baseline comparison, run `python new_research/head_orthogonal/run_experiment.py --experiment all`)*

2.  **Monitor Progress:**
    Logs will print loss every ~50 steps. Total steps = ~1220 (for 20M tokens).

3.  **Output:**
    Results saved to: `new_research/head_orthogonal/results_{TIMESTAMP}/`

## 📄 Documentation

- **[paper.md](new_research/head_orthogonal/paper.md)**: The "Micro-Paper" explaining the hypothesis (93% head alignment) and algorithm.
- **[experiment_plan.md](new_research/head_orthogonal/experiment_plan.md)**: Detailed experimental setup & metrics.
- **[step0_diagnostic.py](new_research/head_orthogonal/step0_diagnostic.py)**: The script that found the 93% alignment.

## 🛠️ Configuration
- **Model**: 1.5B (d=2048, 16 heads)
- **Data**: 20M tokens (16k tokens/step)
- **Optimizer**: HeadOrtho Muon (Mode-1) for Q/K, Standard Muon for others.

**Hypothesis**: HeadOrtho will exploit the massive 93% redundancy to converge faster than independent per-head updates.
