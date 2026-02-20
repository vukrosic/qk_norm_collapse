# QK-Norm Might Worsen Muon Optimizer LLM Training

We train a **1.5B parameter** LLM under three QK-Norm conditions for **~50M tokens** each and measure how the learned γ parameter vs. normalization affects dimensional collapse (Participation Ratio) in attention key vectors.

**→ Full write-up & easy explanations: [`research_qk_collapse/paper.md`](research_qk_collapse/paper.md)**  
**→ PDF Version: [`English`](research_qk_collapse/paper.pdf) | [`Chinese`](research_qk_collapse/paper-ch.pdf)**

| Condition | Final PR (↑) | Train Loss (↓) |
|:---|:---:|:---:|
| Learned γ (full QK-Norm) | **65.7** | 3.618 |
| Frozen γ=1 (norm only) | 59.6 | **3.614** |
| No QK-Norm | 63.3 | 3.683 |

**Key observations:** γ controls dimensional diversity; normalization controls loss — and they are separable. Normalization without γ *hurts* rank more than no normalization at all.

---

## Setup & Reproduce

### 1. Clone & Install

```bash
git clone https://github.com/vukrosic/qk_norm_llm_rank_collapse
cd qk_norm_llm_rank_collapse
pip install -r requirements.txt
```

### 2. Download Dataset

The experiment uses a **1B-token** subset of FineWeb/Cosmopedia. Run the general training script once to download and prepare the data:

Option A: If you train on 40M to 1B tokens, which is this experiment, download the dataset
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 1B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain')
os.makedirs('processed_data/pretrain_1B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_1B')
print('✅ Full Data Ready!')
"
```

Option B: Quick Start (40M Tokens) - for smaller experiments and faster download (I recommend 1B though)
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 40M Token Subset...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain', split='train[:20000]')
os.makedirs('processed_data/speedrun_40M', exist_ok=True)
ds.save_to_disk('processed_data/speedrun_40M')
print('✅ Speedrun Data Ready!')
"
```


This downloads the dataset from HuggingFace, tokenizes it, and saves to `processed_data/`. You can stop it after the data is cached. The prepared data will be at `processed_data/pretrain_1B/` (or auto-detected by the experiment script).

### 3. Run the Experiment

```bash
python research_qk_collapse/qk_norm_25m_study.py
```

This trains 3 models sequentially (~2-3 hours on H100), probes PR every ~2M tokens, and saves results + plots to `research_results/qk_norm_50m_study/`.

**GPU settings** are at the top of the script:
```python
BATCH_SIZE = 8    # ← Lower for smaller GPUs (e.g., 1 for 24GB)
GRAD_ACCUM = 4    # ← Increase proportionally to keep effective batch ~65K tokens
```

You would need to do learning rate ablations if you want to maximize performance / learning.

### 4. View Results

```
research_results/qk_norm_50m_study/
├── QK/results.json          # Learned γ (loss, PR, gamma values)
├── QK_frozen/results.json   # Frozen γ=1
├── NoQK/results.json        # No QK-Norm
├── 1_loss.png               # Loss curves
└── 2_key_pr_causal.png      # PR trajectory (main figure)
```

To regenerate plots from existing results: `python research_qk_collapse/split_panels.py`

---

## HeadOrtho Muon Experiment

This experiment tests **Head-wise Tensor Orthogonalization** to improve model training, leveraging the finding that right singular vectors are highly aligned across heads.

**→ Full write-up & easy explanations: [`research_head_ortho_simple/paper.md`](research_head_ortho_simple/paper.md)**  
**→ PDF Version: [`English`](research_head_ortho_simple/paper.pdf) | [`Chinese`](research_head_ortho_simple/paper-ch.pdf)**

### Run Experiment

```bash
# Runs the HeadOrtho variant (Mode-1)
python new_research/head_orthogonal/run_experiment.py --experiment headortho
```
*(If you need a baseline comparison, run `--experiment all` instead)*

Logs will print loss every ~50 steps. Results are saved to `new_research/head_orthogonal/results_{TIMESTAMP}/`.

---

## License

MIT
