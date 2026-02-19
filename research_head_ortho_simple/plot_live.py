import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def plot_live(run_dir):
    run_path = Path(run_dir)
    results_file = run_path / "results.json"
    
    if not results_file.exists():
        print(f"No results found at {results_file}")
        return

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading results: {e}")
        return

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss
    ax = axes[0]
    tokens = [t/1000 for t in data["tokens"]]
    loss = data["train_loss"]
    ax.plot(tokens, loss, marker='o', label='Train Loss')
    ax.set_xlabel('Tokens (K)')
    ax.set_ylabel('Loss')
    ax.set_title(f'Live Training Loss ({data.get("run_name", "Unknown")})')
    ax.grid(True)
    
    # PR
    ax = axes[1]
    pr = data["mean_pr"]
    ax.plot(tokens, pr, marker='o', color='orange', label='Mean PR')
    ax.set_xlabel('Tokens (K)')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('Live Rank Collapse')
    ax.grid(True)
    
    plt.tight_layout()
    out_file = run_path / "live_plot.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-detect latest
        base_dir = Path("research_head_ortho_simple/results")
        if base_dir.exists():
            runs = [d for d in base_dir.iterdir() if d.is_dir()]
            if runs:
                latest_run = max(runs, key=lambda d: d.stat().st_mtime)
                print(f"Plotting latest run: {latest_run}")
                plot_live(latest_run)
            else:
                print("No runs found.")
    else:
        plot_live(sys.argv[1])
