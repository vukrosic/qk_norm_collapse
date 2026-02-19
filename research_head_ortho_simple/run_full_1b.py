import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import time
import gc
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.getcwd())

from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from optimizers.head_ortho_muon import HeadOrthoMuon
from optimizers.muon import Muon
from svd_probe import RankProbe
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from train_llm import prepare_datasets

# ============================================================
# Configuration
# ============================================================
TARGET_TOKENS = 300_000_000
PROBE_EVERY_STEPS = 100      # Probe every 100 steps
BATCH_SIZE = 32
GRAD_ACCUM = 4               # Effective batch = 16 * 4 * 2048 = 131,072 tokens
SEED = 42
DATASET_PATH = "processed_data/pretrain_1B"
RESULTS_DIR = Path("research_head_ortho_simple/results_1b")

def setup_optimizer(model: nn.Module, config: LLMConfig, mode: str):
    head_ortho_params = []
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.ndim == 2 and 'token_embedding' not in name and 'norm' not in name:
            if 'qkvo_proj' in name:
                if mode == 'baseline':
                    muon_params.append(param)
                else:
                    head_ortho_params.append((name, param))
            else:
                muon_params.append(param)
        else:
            adamw_params.append(param)
    
    optimizers = []
    
    if head_ortho_params:
        ortho_mode = 1 if mode == 'mode1' else 3
        head_ortho_group = {
            "params": [p for _, p in head_ortho_params],
            "lr": 0.05,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "ortho_mode": ortho_mode,
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
    
    if muon_params:
        optimizers.append(Muon(muon_params, lr=0.05, momentum=0.95))
    
    optimizers.append(torch.optim.AdamW(
        adamw_params, lr=0.005, weight_decay=0.01, fused=True
    ))
    
    return optimizers

def run_1b_experiment(mode: str):
    run_name = f"1b_full_{mode}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 STARTING 1B TOKEN RUN: {run_name}")
    
    config = LLMConfig()
    config.use_qk_norm = True
    config.train_tokens = TARGET_TOKENS
    config.batch_size = BATCH_SIZE
    config.gradient_accumulation_steps = GRAD_ACCUM
    config.compile_model = True
    config.gradient_checkpointing = True
    
    device = torch.device('cuda')
    
    # Data
    data_cfg = DataConfig(dataset_path=DATASET_PATH, seq_length=config.max_seq_len)
    tokenizer = setup_tokenizer(data_cfg)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    torch.manual_seed(SEED + 999)
    eval_loader = DataLoader(val_ds, batch_size=4, shuffle=True)
    eval_batch = next(iter(eval_loader))["input_ids"].to(device)
    
    # Model
    torch.manual_seed(SEED)
    model = MinimalLLM(config).to(device)
    if config.compile_model:
        model = torch.compile(model)
        
    probe = RankProbe(model, device, eval_batch=eval_batch)
    optimizers = setup_optimizer(model, config, mode)
    
    results = {
        "mode": mode,
        "tokens": [],
        "train_loss": [],
        "mean_pr": []
    }
    
    # Training Loop
    model.train()
    tokens_seen = 0
    step = 0
    running_loss = 0.0
    loss_count = 0
    pbar = tqdm(total=TARGET_TOKENS, desc=run_name, unit="tok")
    
    while tokens_seen < TARGET_TOKENS:
        for batch in train_loader:
            if tokens_seen >= TARGET_TOKENS: break
            
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
                scaled_loss = loss / config.gradient_accumulation_steps
            
            scaled_loss.backward()
            running_loss += loss.item()
            loss_count += 1
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()
            
            tokens_seen += x.numel()
            step += 1
            pbar.update(x.numel())
            
            if step % (PROBE_EVERY_STEPS * GRAD_ACCUM) == 0:
                model.eval()
                with torch.no_grad():
                    pr_dict = probe.run_probe()
                    mean_pr = np.mean([v["k_pr"] for v in pr_dict.values()])
                model.train()
                
                avg_loss = running_loss / max(loss_count, 1)
                results["tokens"].append(tokens_seen)
                results["train_loss"].append(avg_loss)
                results["mean_pr"].append(float(mean_pr))
                
                with open(run_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2)
                
                running_loss = 0.0
                loss_count = 0
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "PR": f"{mean_pr:.1f}"})

    pbar.close()
    print(f"✅ Finished {run_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['baseline', 'mode1', 'mode3'], required=True)
    args = parser.parse_args()
    
    run_1b_experiment(args.mode)
