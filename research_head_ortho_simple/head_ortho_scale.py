"""
HeadOrtho Rank Collapse Study (1M tokens)
=========================================
Trains a 1B parameter LLM for 1M tokens with and without HeadOrtho Muon,
measuring Participation Ratio (effective rank) and loss at regular intervals.

Research Question: Does Head-Wise Tensor Orthogonalization prevent dimensional collapse?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import time
import gc
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.getcwd())

from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
# We will use HeadOrthoMuon from optimizers
try:
    from optimizers.head_ortho_muon import HeadOrthoMuon
except ImportError:
    # If not found, we implement a simple version here or fail
    print("Error: could not import HeadOrthoMuon from optimizers.head_ortho_muon")
    sys.exit(1)

from optimizers.muon import Muon
from svd_probe import RankProbe
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from train_llm import prepare_datasets


# ============================================================
# Configuration
# ============================================================
TARGET_TOKENS = 10_000_000
PROBE_EVERY_STEPS = 20       # Probe every 20 micro-batches
BATCH_SIZE = 16              # Utilize memory
GRAD_ACCUM = 4               # Effective batch = 16 * 4 = 64 * 2048 = 131k tokens per step
SEED = 42
OUTPUT_DIR = Path("research_head_ortho_simple/results")
DATASET_PATH = "processed_data/pretrain_1B"


def setup_head_ortho_optimizer(model: nn.Module, config: LLMConfig, use_head_ortho: bool = True):
    """
    Setup optimizer with optional head-wise tensor orthogonalization.
    If use_head_ortho=True, Q/K projections in qkvo_proj use HeadOrthoMuon.
    """
    head_ortho_params = []
    muon_params = []
    adamw_params = []
    
    # We use Muon for 2D params, AdamW for others
    # HeadOrtho applies only to qkvo_proj if enabled
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.ndim == 2 and 'token_embedding' not in name and 'norm' not in name:
            if use_head_ortho and 'qkvo_proj' in name:
                head_ortho_params.append((name, param))
            else:
                muon_params.append(param)
        else:
            adamw_params.append(param)
    
    optimizers = []
    
    # HeadOrtho optimizer for qkvo_proj params
    if head_ortho_params:
        head_ortho_group = {
            "params": [p for _, p in head_ortho_params],
            "lr": 0.05, # Muon LR (Increased for larger batch)
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "ortho_mode": 1,  # mode-1
            "head_info": {
                "n_heads": config.n_heads,
                "n_kv_heads": config.n_kv_heads if config.n_kv_heads else config.n_heads,
                "d_k": config.d_model // config.n_heads,
                "d_model": config.d_model,
                "q_size": config.d_model,
                "kv_size": (config.n_kv_heads if config.n_kv_heads else config.n_heads) * (config.d_model // config.n_heads),
            }
        }
        optimizers.append(HeadOrthoMuon([head_ortho_group]))
    
    # Standard Muon for other 2D params
    if muon_params:
        optimizers.append(Muon(muon_params, lr=0.05, momentum=0.95))
    
    # AdamW for 1D/embedding params
    optimizers.append(torch.optim.AdamW(
        adamw_params,
        lr=0.005, # AdamW LR
        weight_decay=0.0, # minimal decay
        fused=torch.cuda.is_available()
    ))
    
    return optimizers


def run_experiment(use_head_ortho: bool) -> dict:
    """
    Train a 1B model for 1M tokens with or without HeadOrtho.
    Returns a dict with tokens, val_loss, mean_pr, and per-layer PR.
    """
    tag = "HeadOrtho" if use_head_ortho else "Baseline"
    run_name = f"1b_1m_{tag}"
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {run_name} (use_head_ortho={use_head_ortho})")
    print(f"{'='*70}")

    # ---- Config ----
    config = LLMConfig()
    config.use_qk_norm = True # Enable QK Norm for both to isolate HeadOrtho effect
    config.train_tokens = TARGET_TOKENS
    config.batch_size = BATCH_SIZE
    config.gradient_accumulation_steps = GRAD_ACCUM
    config.compile_model = True # Try compiling for speed
    config.gradient_checkpointing = True

    device = torch.device('cuda')

    # ---- Data ----
    data_cfg = DataConfig(dataset_path=DATASET_PATH, seq_length=config.max_seq_len)
    tokenizer = setup_tokenizer(data_cfg)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    # Fixed eval batch for PR measurement (same across both runs via seed)
    torch.manual_seed(SEED + 999)  # separate seed for eval batch
    eval_loader = DataLoader(val_ds, batch_size=4, shuffle=True)
    eval_batch = next(iter(eval_loader))["input_ids"].to(device)

    # ---- Model ----
    torch.manual_seed(SEED)
    model = MinimalLLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    if config.compile_model:
        try:
            model = torch.compile(model)
            print("  Model compiled.")
        except:
            print("  Compilation failed, using eager.")

    # ---- Probe ----
    dk = config.d_model // config.n_heads  # 128
    probe = RankProbe(model, device, eval_batch=eval_batch)

    # ---- Optimizer ----
    optimizers = setup_head_ortho_optimizer(model, config, use_head_ortho=use_head_ortho)

    # ---- Results storage ----
    results = {
        "run_name": run_name,
        "use_head_ortho": use_head_ortho,
        "tokens": [],
        "steps": [],
        "train_loss": [],
        "mean_pr": [],
        "layer_pr": {},  # layer_idx -> list of PR values
    }

    # ---- Create output dir ----
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training Loop ----
    model.train()
    tokens_seen = 0
    step = 0
    running_loss = 0.0
    loss_count = 0
    pbar = tqdm(total=TARGET_TOKENS, desc=run_name, unit="tok")

    # Initial probe at step 0
    print("  Running initial probe (step 0)...")
    model.eval()
    try:
        pr_dict = probe.run_probe()
    except Exception as e:
        print(f"  Probe failed: {e}")
        pr_dict = {}
        
    model.train()
    if pr_dict:
        mean_pr = np.mean([v["k_pr"] for v in pr_dict.values()])
        results["tokens"].append(0)
        results["steps"].append(0)
        results["train_loss"].append(float('nan'))  # no loss at step 0
        results["mean_pr"].append(float(mean_pr))
        for l_idx, l_metrics in pr_dict.items():
            key = str(l_idx)
            if key not in results["layer_pr"]:
                results["layer_pr"][key] = []
            results["layer_pr"][key].append(float(l_metrics["k_pr"]))
        print(f"  Step 0 | PR={mean_pr:.2f}")

    while tokens_seen < TARGET_TOKENS:
        for batch in train_loader:
            if tokens_seen >= TARGET_TOKENS:
                break

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                scaled_loss = loss / config.gradient_accumulation_steps

            scaled_loss.backward()

            # Track raw loss
            running_loss += loss.item()
            loss_count += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()

            batch_tokens = x.numel()
            tokens_seen += batch_tokens
            step += 1
            pbar.update(batch_tokens)

            # ---- Probe every N steps ----
            if step % PROBE_EVERY_STEPS == 0:
                model.eval()
                try:
                    pr_dict = probe.run_probe()
                except:
                    pr_dict = {}
                model.train()

                if pr_dict:
                    mean_pr = np.mean([v["k_pr"] for v in pr_dict.values()])
                    avg_loss = running_loss / max(loss_count, 1)

                    results["tokens"].append(int(tokens_seen))
                    results["steps"].append(int(step))
                    results["train_loss"].append(float(avg_loss))
                    results["mean_pr"].append(float(mean_pr))

                    for l_idx, l_metrics in pr_dict.items():
                        key = str(l_idx)
                        if key not in results["layer_pr"]:
                            results["layer_pr"][key] = []
                        results["layer_pr"][key].append(float(l_metrics["k_pr"]))

                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "PR": f"{mean_pr:.1f}"
                    })

                    running_loss = 0.0
                    loss_count = 0

                    # Save intermediate results
                    with open(run_dir / "results.json", "w") as f:
                        json.dump(results, f, indent=2)

    pbar.close()
    print(f"  ✅ {run_name} complete: {tokens_seen:,} tokens, {step} steps")

    # Final save
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Cleanup
    del model, optimizers, probe, eval_batch
    torch.cuda.empty_cache()
    gc.collect()

    return results


def plot_results(res_ho: dict, res_base: dict):
    """Generate comprehensive comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    colors_ho = '#a855f7'     # purple
    colors_base = '#22d3ee'   # cyan

    # ================================================================
    # Panel 1: Training Loss
    # ================================================================
    ax = axes[0, 0]
    # Filter NaN from step-0
    ho_tokens = [t for t, l in zip(res_ho["tokens"], res_ho["train_loss"]) if not np.isnan(l)]
    ho_loss = [l for l, _ in zip(res_ho["train_loss"], res_ho["tokens"]) if not np.isnan(l)]
    base_tokens = [t for t, l in zip(res_base["tokens"], res_base["train_loss"]) if not np.isnan(l)]
    base_loss = [l for l, _ in zip(res_base["train_loss"], res_base["tokens"]) if not np.isnan(l)]

    ax.plot([t/1000 for t in ho_tokens], ho_loss, color=colors_ho, linewidth=2, label='HeadOrtho', marker='o', markersize=4)
    ax.plot([t/1000 for t in base_tokens], base_loss, color=colors_base, linewidth=2, label='Baseline', marker='s', markersize=4)
    ax.set_xlabel('Tokens (K)')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, color='#484f58')

    # ================================================================
    # Panel 2: Mean Participation Ratio
    # ================================================================
    ax = axes[0, 1]
    ax.plot([t/1000 for t in res_ho["tokens"]], res_ho["mean_pr"], color=colors_ho, linewidth=2.5, label='HeadOrtho', marker='o', markersize=4)
    ax.plot([t/1000 for t in res_base["tokens"]], res_base["mean_pr"], color=colors_base, linewidth=2.5, label='Baseline', marker='s', markersize=4)
    ax.set_xlabel('Tokens (K)')
    ax.set_ylabel('Mean Participation Ratio')
    ax.set_title('Rank Collapse: Mean Effective Rank')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, color='#484f58')

    # ================================================================
    # Panel 3: Per-layer PR at FINAL checkpoint
    # ================================================================
    ax = axes[1, 0]
    n_layers = len(res_ho["layer_pr"])
    layers = sorted([int(k) for k in res_ho["layer_pr"].keys()])

    ho_final_pr = [res_ho["layer_pr"][str(l)][-1] for l in layers]
    base_final_pr = [res_base["layer_pr"][str(l)][-1] for l in layers]

    x_pos = np.arange(len(layers))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, ho_final_pr, width, color=colors_ho, alpha=0.85, label='HeadOrtho')
    bars2 = ax.bar(x_pos + width/2, base_final_pr, width, color=colors_base, alpha=0.85, label='Baseline')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Participation Ratio (Final)')
    ax.set_title(f'Per-Layer Effective Rank at {TARGET_TOKENS//1000}K Tokens')
    ax.set_xticks(x_pos[::4])
    ax.set_xticklabels([str(l) for l in layers[::4]])
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(True, alpha=0.15, color='#484f58', axis='y')

    # ================================================================
    # Panel 4: PR Delta (HeadOrtho minus Baseline)
    # ================================================================
    ax = axes[1, 1]
    delta_pr = [ho - base for ho, base in zip(ho_final_pr, base_final_pr)]
    bar_colors = ['#f87171' if d < 0 else '#4ade80' for d in delta_pr]
    ax.bar(x_pos, delta_pr, color=bar_colors, alpha=0.85)
    ax.axhline(y=0, color='#8b949e', linewidth=1, linestyle='--')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('ΔPR (HeadOrtho − Baseline)')
    ax.set_title('Per-Layer Rank Difference')
    ax.set_xticks(x_pos[::4])
    ax.set_xticklabels([str(l) for l in layers[::4]])
    ax.grid(True, alpha=0.15, color='#484f58', axis='y')

    plt.tight_layout(pad=2.0)
    plot_path = OUTPUT_DIR / "head_ortho_rank_collapse.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"  📈 Main comparison plot saved: {plot_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("  HEAD ORTHO RANK COLLAPSE STUDY — 1B Model × 1M Tokens")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Run 1: With HeadOrtho
    print("\n\n🔬 PHASE 1: Training WITH HeadOrtho...")
    res_ho = run_experiment(use_head_ortho=True)

    # Run 2: Without HeadOrtho (Baseline)
    print("\n\n🔬 PHASE 2: Training WITHOUT HeadOrtho (Baseline)...")
    res_base = run_experiment(use_head_ortho=False)

    # Plot
    print("\n\n📊 PHASE 3: Generating plots...")
    plot_results(res_ho, res_base)

    elapsed = time.time() - start_time
    print(f"\n\n✅ STUDY COMPLETE in {elapsed/60:.1f} minutes")
    print(f"   Results: {OUTPUT_DIR}/")
