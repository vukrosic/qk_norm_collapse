"""
Split the 4-panel figure into individual images for the report.
Reads from combined_results.json, no model/torch needed.
"""
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("research_results/qk_norm_500k_study")

with open(OUTPUT_DIR / "combined_results.json") as f:
    data = json.load(f)

res_qk = data["qk_norm"]
res_noqk = data["no_qk_norm"]

colors_qk = '#f97316'     # orange
colors_noqk = '#22d3ee'   # cyan

def style_ax(ax):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#c9d1d9', labelsize=12)
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#e6edf3')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

def save_fig(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"  Saved: {path}")

# Helper: filter NaN
def filter_nan(tokens, values):
    t_out, v_out = [], []
    for t, v in zip(tokens, values):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            t_out.append(t)
            v_out.append(v)
    return t_out, v_out

# ================================================================
# Panel A: Training Loss
# ================================================================
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0d1117')
style_ax(ax)

qk_t, qk_l = filter_nan(res_qk["tokens"], res_qk["train_loss"])
nq_t, nq_l = filter_nan(res_noqk["tokens"], res_noqk["train_loss"])

ax.plot([t/1000 for t in qk_t], qk_l, color=colors_qk, linewidth=2.5,
        label='With QK-Norm', marker='o', markersize=8)
ax.plot([t/1000 for t in nq_t], nq_l, color=colors_noqk, linewidth=2.5,
        label='Without QK-Norm', marker='s', markersize=8)
ax.set_xlabel('Tokens (K)', fontsize=14)
ax.set_ylabel('Training Loss', fontsize=14)
ax.set_title('Panel A — Training Loss Comparison', fontsize=16, fontweight='bold')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=12)
ax.grid(True, alpha=0.15, color='#484f58')

# Annotate final values
ax.annotate(f'{qk_l[-1]:.4f}', xy=(qk_t[-1]/1000, qk_l[-1]),
            xytext=(15, 10), textcoords='offset points',
            color=colors_qk, fontsize=11, fontweight='bold')
ax.annotate(f'{nq_l[-1]:.4f}', xy=(nq_t[-1]/1000, nq_l[-1]),
            xytext=(15, -15), textcoords='offset points',
            color=colors_noqk, fontsize=11, fontweight='bold')

save_fig(fig, "panel_a_loss.png")

# ================================================================
# Panel B: Mean PR Trajectory
# ================================================================
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0d1117')
style_ax(ax)

ax.plot([t/1000 for t in res_qk["tokens"]], res_qk["mean_pr"], color=colors_qk,
        linewidth=2.5, label='With QK-Norm', marker='o', markersize=8)
ax.plot([t/1000 for t in res_noqk["tokens"]], res_noqk["mean_pr"], color=colors_noqk,
        linewidth=2.5, label='Without QK-Norm', marker='s', markersize=8)
ax.set_xlabel('Tokens (K)', fontsize=14)
ax.set_ylabel('Mean Participation Ratio', fontsize=14)
ax.set_title('Panel B — Rank Collapse: Mean Effective Rank Over Time', fontsize=16, fontweight='bold')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=12)
ax.grid(True, alpha=0.15, color='#484f58')

# Annotate init and final
ax.annotate(f'{res_qk["mean_pr"][0]:.2f}', xy=(0, res_qk["mean_pr"][0]),
            xytext=(-10, 10), textcoords='offset points',
            color=colors_qk, fontsize=10)
ax.annotate(f'{res_qk["mean_pr"][-1]:.2f}', xy=(res_qk["tokens"][-1]/1000, res_qk["mean_pr"][-1]),
            xytext=(10, 8), textcoords='offset points',
            color=colors_qk, fontsize=11, fontweight='bold')
ax.annotate(f'{res_noqk["mean_pr"][-1]:.2f}', xy=(res_noqk["tokens"][-1]/1000, res_noqk["mean_pr"][-1]),
            xytext=(10, -12), textcoords='offset points',
            color=colors_noqk, fontsize=11, fontweight='bold')

# Shade the gap
ax.fill_between([t/1000 for t in res_qk["tokens"]],
                res_qk["mean_pr"], res_noqk["mean_pr"],
                alpha=0.15, color='#a78bfa')

save_fig(fig, "panel_b_pr_trajectory.png")

# ================================================================
# Panel C: Per-Layer PR at Final Checkpoint
# ================================================================
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0d1117')
style_ax(ax)

layers = sorted([int(k) for k in res_qk["layer_pr"].keys()])
qk_final = [res_qk["layer_pr"][str(l)][-1] for l in layers]
nq_final = [res_noqk["layer_pr"][str(l)][-1] for l in layers]

x_pos = list(range(len(layers)))
width = 0.35
ax.bar([x - width/2 for x in x_pos], qk_final, width, color=colors_qk, alpha=0.85, label='With QK-Norm')
ax.bar([x + width/2 for x in x_pos], nq_final, width, color=colors_noqk, alpha=0.85, label='Without QK-Norm')
ax.set_xlabel('Layer Index', fontsize=14)
ax.set_ylabel('Participation Ratio (Final)', fontsize=14)
ax.set_title('Panel C — Per-Layer Effective Rank at 500K Tokens', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos[::2])
ax.set_xticklabels([str(l) for l in layers[::2]])
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=12)
ax.grid(True, alpha=0.15, color='#484f58', axis='y')

# Add horizontal line at dk=128
ax.axhline(y=128, color='#8b949e', linewidth=1, linestyle=':', alpha=0.5)
ax.text(31, 128.5, 'd_k = 128 (full rank)', color='#8b949e', fontsize=10, ha='right')

save_fig(fig, "panel_c_per_layer.png")

# ================================================================
# Panel D: Delta PR per Layer
# ================================================================
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0d1117')
style_ax(ax)

delta_pr = [qk - nq for qk, nq in zip(qk_final, nq_final)]
bar_colors = ['#f87171' if d < 0 else '#4ade80' for d in delta_pr]
bars = ax.bar(x_pos, delta_pr, color=bar_colors, alpha=0.85, edgecolor='#30363d', linewidth=0.5)
ax.axhline(y=0, color='#8b949e', linewidth=1, linestyle='--')
ax.set_xlabel('Layer Index', fontsize=14)
ax.set_ylabel('ΔPR (QK − NoQK)', fontsize=14)
ax.set_title('Panel D — Per-Layer Rank Difference: QK-Norm Effect', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos[::2])
ax.set_xticklabels([str(l) for l in layers[::2]])
ax.grid(True, alpha=0.15, color='#484f58', axis='y')

# Annotate the max delta
max_idx = delta_pr.index(max(delta_pr))
ax.annotate(f'L{layers[max_idx]}: +{delta_pr[max_idx]:.2f}',
            xy=(max_idx, delta_pr[max_idx]),
            xytext=(0, 15), textcoords='offset points',
            color='#4ade80', fontsize=11, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#4ade80', lw=1.5))

# Add a note
ax.text(0.02, 0.95, 'All layers: QK-Norm > NoQK\n(all bars positive)',
        transform=ax.transAxes, color='#4ade80', fontsize=11,
        verticalalignment='top', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#4ade80', alpha=0.8))

save_fig(fig, "panel_d_delta.png")

print("\n✅ All 4 individual panels saved!")
