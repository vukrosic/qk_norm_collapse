import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_1b_comparison():
    res_dir = Path("research_head_ortho_simple/results_1b")
    modes = ["baseline", "mode1", "mode3"]
    
    data = {}
    for mode in modes:
        fpath = res_dir / f"1b_full_{mode}" / "results.json"
        if fpath.exists():
            with open(fpath, 'r') as f:
                data[mode] = json.load(f)
        else:
            print(f"Waiting for {mode} data...")

    if not data:
        print("No results found yet.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        ax.grid(True, alpha=0.15, color='#484f58')

    colors = {'mode1': '#a855f7', 'mode3': '#facc15', 'baseline': '#22d3ee'}
    
    # Loss
    ax = axes[0]
    for mode, d in data.items():
            d_content = data[mode]
            if "train_loss" in d_content:
                tokens = [t/1e6 for t in d_content["tokens"]]
                ax.plot(tokens, d_content["train_loss"], label=mode, linewidth=2)
    
    ax.set_xlabel('Tokens (Millions)')
    ax.set_ylabel('Training Loss')
    ax.set_title('1B Token Run: Loss Comparison')
    ax.legend()

    # PR
    ax = axes[1]
    for mode, d in data.items():
        d_content = data[mode]
        if "mean_pr" in d_content:
            tokens = [t/1e6 for t in d_content["tokens"]]
            ax.plot(tokens, d_content["mean_pr"], label=mode, linewidth=2.5)

    ax.set_xlabel('Tokens (Millions)')
    ax.set_ylabel('Mean Participation Ratio')
    ax.set_title('1B Token Run: Rank Dynamics')
    ax.legend()

    plt.tight_layout()
    out_path = res_dir / "comparison_1b.png"
    fig.savefig(out_path, dpi=150, facecolor='#0d1117')
    print(f"Updated plot saved to {out_path}")

if __name__ == "__main__":
    plot_1b_comparison()
