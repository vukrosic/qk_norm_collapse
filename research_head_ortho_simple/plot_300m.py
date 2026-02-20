import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_300m_comparison():
    res_dir = Path("/root/attention-keys-analysis/research_head_ortho_simple/results_1b")
    mode1_file = res_dir / "1b_full_mode1" / "results.json"
    mode3_file = res_dir / "1b_full_mode3" / "results.json"
    base_file = res_dir / "1b_full_baseline" / "results.json"
    
    data = {}
    for name, fpath in [("Mode 1 (Input-Shared)", mode1_file), ("Mode 3 (Inter-Head)", mode3_file), ("Baseline (Global)", base_file)]:
        if fpath.exists():
            with open(fpath, 'r') as f:
                data[name] = json.load(f)
        else:
            print(f"Warning: Missing {name} at {fpath}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        ax.grid(True, alpha=0.15, color='#484f58')

    colors = {'Mode 1 (Input-Shared)': '#a855f7', 'Mode 3 (Inter-Head)': '#facc15', 'Baseline (Global)': '#22d3ee'}
    styles = {'Mode 1 (Input-Shared)': 'o', 'Mode 3 (Inter-Head)': '^', 'Baseline (Global)': 's'}

    # Loss
    ax = axes[0]
    for name, d in data.items():
        tokens = [t/1e6 for t, l in zip(d["tokens"], d["train_loss"]) if not np.isnan(l)]
        loss = [l for l in d["train_loss"] if not np.isnan(l)]
        ax.plot(tokens, loss, color=colors[name], linewidth=2, label=name, marker=styles[name], markersize=4)
    
    ax.set_xlabel('Tokens (Millions)')
    ax.set_ylabel('Training Loss')
    ax.set_title('Loss Comparison (Up to ~300M Tokens)')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')

    # PR
    ax = axes[1]
    for name, d in data.items():
        tokens = [t/1e6 for t in d["tokens"]]
        pr = d["mean_pr"]
        ax.plot(tokens, pr, color=colors[name], linewidth=2.5, label=name, marker=styles[name], markersize=4)

    ax.set_xlabel('Tokens (Millions)')
    ax.set_ylabel('Mean Participation Ratio')
    ax.set_title('Rank Collapse: Dimensional vs Head Orthogonality')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')

    plt.tight_layout(pad=3.0)
    out_path = res_dir / "comparison_300m_tokens.png"
    fig.savefig(out_path, dpi=150, facecolor='#0d1117')
    print(f"Final 3-way plot saved to {out_path}")

if __name__ == "__main__":
    plot_300m_comparison()
