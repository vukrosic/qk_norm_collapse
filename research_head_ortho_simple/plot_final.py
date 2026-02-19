import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_comparison():
    res_dir = Path("research_head_ortho_simple/results")
    ho_file = res_dir / "1b_1m_HeadOrtho" / "results.json"
    base_file = res_dir / "1b_1m_Baseline" / "results.json"
    
    if not ho_file.exists() or not base_file.exists():
        print("Missing result files.")
        return

    with open(ho_file, 'r') as f:
        ho_data = json.load(f)
    with open(base_file, 'r') as f:
        base_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        ax.grid(True, alpha=0.15, color='#484f58')

    ho_color = '#a855f7' # purple
    base_color = '#22d3ee' # cyan

    # Loss
    ax = axes[0]
    ho_tokens = [t/1e6 for t, l in zip(ho_data["tokens"], ho_data["train_loss"]) if not np.isnan(l)]
    ho_loss = [l for l in ho_data["train_loss"] if not np.isnan(l)]
    base_tokens = [t/1e6 for t, l in zip(base_data["tokens"], base_data["train_loss"]) if not np.isnan(l)]
    base_loss = [l for l in base_data["train_loss"] if not np.isnan(l)]

    ax.plot(ho_tokens, ho_loss, color=ho_color, linewidth=2, label='HeadOrtho (Mode-1)', marker='o', markersize=4)
    ax.plot(base_tokens, base_loss, color=base_color, linewidth=2, label='Baseline Muon', marker='s', markersize=4)
    ax.set_xlabel('Tokens (Millions)')
    ax.set_ylabel('Training Loss')
    ax.set_title('10M Token Training: Loss Comparison')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')

    # PR
    ax = axes[1]
    ax.plot([t/1e6 for t in ho_data["tokens"]], ho_data["mean_pr"], color=ho_color, linewidth=2.5, label='HeadOrtho', marker='o', markersize=4)
    ax.plot([t/1e6 for t in base_data["tokens"]], base_data["mean_pr"], color=base_color, linewidth=2.5, label='Baseline', marker='s', markersize=4)
    ax.set_xlabel('Tokens (Millions)')
    ax.set_ylabel('Mean Participation Ratio')
    ax.set_title('Rank Collapse: HeadOrtho vs Baseline')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')

    plt.tight_layout(pad=3.0)
    out_path = res_dir / "final_10m_comparison.png"
    fig.savefig(out_path, dpi=150, facecolor='#0d1117')
    print(f"Final plot saved to {out_path}")

if __name__ == "__main__":
    plot_comparison()
