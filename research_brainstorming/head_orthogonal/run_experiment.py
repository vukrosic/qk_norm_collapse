"""
Training script for HeadOrtho vs baseline comparison.

Runs 3 experiments:
  1. Muon baseline (with QK Norm)
  2. HeadOrtho Muon (with QK Norm)  
  3. HeadOrtho Muon (without QK Norm)

Usage:
  python new_research/head_orthogonal/run_experiment.py --experiment all
  python new_research/head_orthogonal/run_experiment.py --experiment baseline
  python new_research/head_orthogonal/run_experiment.py --experiment headortho
  python new_research/head_orthogonal/run_experiment.py --experiment headortho_no_qknorm
"""

import argparse
import sys
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from optimizers.muon import Muon
from optimizers.head_ortho_muon import HeadOrthoMuon
from training.trainer import train_model, warmup_compiled_kernels
from utils.helpers import set_seed


def setup_head_ortho_optimizer(model: nn.Module, config: LLMConfig, use_head_ortho: bool = True):
    """
    Setup optimizer with optional head-wise tensor orthogonalization.
    
    If use_head_ortho=True, Q/K projections in qkvo_proj use HeadOrthoMuon.
    All other 2D params use standard Muon. 1D/embedding params use AdamW.
    """
    head_ortho_params = []
    muon_params = []
    adamw_params = []
    
    use_muon = getattr(config, 'use_muon', True)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if (use_muon and param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name):
            
            if use_head_ortho and 'qkvo_proj' in name:
                # This is a merged QKVO projection — needs head info
                head_ortho_params.append((name, param))
            else:
                muon_params.append(param)
        else:
            adamw_params.append(param)
    
    print(f"\n  HeadOrtho parameters: {sum(p.numel() for _, p in head_ortho_params):,}")
    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")
    
    optimizers = []
    
    # HeadOrtho optimizer for qkvo_proj params
    if head_ortho_params:
        head_ortho_group = {
            "params": [p for _, p in head_ortho_params],
            "lr": config.muon_lr,
            "momentum": config.muon_momentum,
            "nesterov": True,
            "ns_steps": 5,
            "ortho_mode": 1,  # mode-1 chosen based on diagnostic (0.93 alignment)
            "head_info": {
                "n_heads": config.n_heads,
                "n_kv_heads": config.n_kv_heads if config.n_kv_heads else config.n_heads,
                "d_k": config.d_k,
                "d_model": config.d_model,
                "q_size": config.d_model,
                "kv_size": (config.n_kv_heads if config.n_kv_heads else config.n_heads) * config.d_k,
            }
        }
        optimizers.append(HeadOrthoMuon([head_ortho_group]))
    
    # Standard Muon for other 2D params
    if muon_params:
        optimizers.append(Muon(muon_params, lr=config.muon_lr, momentum=config.muon_momentum))
    
    # AdamW for 1D/embedding params
    optimizers.append(torch.optim.AdamW(
        adamw_params,
        lr=config.adamw_lr,
        weight_decay=config.weight_decay,
        fused=torch.cuda.is_available()
    ))
    
    return optimizers


def run_single_experiment(
    exp_name: str,
    config: LLMConfig,
    train_loader,
    val_loader,
    use_head_ortho: bool,
    output_dir: str,
):
    """Run a single training experiment."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  HeadOrtho: {use_head_ortho}")
    print(f"  QK Norm: {config.use_qk_norm}")
    print(f"  Model: d={config.d_model}, heads={config.n_heads}, layers={config.n_layers}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with fixed seed
    set_seed(42)
    model = MinimalLLM(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Save initial state for reset after warmup
    initial_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Compile
    if config.compile_model:
        print("Compiling model...")
        orig_model = model
        try:
            model = torch.compile(model)
            warmup_compiled_kernels(model, config, train_loader, device, num_steps=3)
            orig_model.load_state_dict(initial_model_state)
            print("Compiled and reset")
        except Exception as e:
            print(f"Compilation failed: {e}, using eager mode")
            model = orig_model
            model.load_state_dict(initial_model_state)
    
    del initial_model_state
    torch.cuda.empty_cache()
    
    # Setup optimizers
    optimizers = setup_head_ortho_optimizer(model, config, use_head_ortho=use_head_ortho)
    
    # Setup schedulers
    tokens_per_opt = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
    total_steps = config.train_tokens // tokens_per_opt
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    schedule_type = getattr(config, 'schedule_type', 'cosine')
    
    schedulers = []
    for optimizer in optimizers:
        if schedule_type == 'cosine':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule_type == 'linear':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return max(0.1, 1.0 - progress)
        else:  # constant
            def lr_lambda(current_step, warmup=warmup_steps):
                return current_step / warmup if current_step < warmup else 1.0
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))
    
    # Reset seed
    set_seed(42)
    
    # Train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    results = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        early_stopper=None,
        output_dir=output_dir,
        extra_config={
            "experiment_name": exp_name,
            "use_head_ortho": use_head_ortho,
            "use_qk_norm": config.use_qk_norm,
        },
        log_every=getattr(config, 'log_every', 100),
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="HeadOrtho Experiment")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "baseline", "headortho", "headortho_no_qknorm"],
                       help="Which experiment to run")
    parser.add_argument("--train_tokens", type=int, default=None,
                       help="Override number of training tokens (default: use config)")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to preprocessed dataset")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # ============================
    # Setup data (shared across experiments)
    # ============================
    from train_llm import prepare_datasets
    from configs.dataset_config import DataConfig
    from data.loader import setup_tokenizer
    from torch.utils.data import DataLoader
    import numpy as np
    import random
    
    # Use the default 1.5B config
    base_config = LLMConfig()
    if args.train_tokens is not None:
        base_config.train_tokens = args.train_tokens
    if args.batch_size is not None:
        base_config.batch_size = args.batch_size
    
    # Set eval milestones based on token count
    tokens_per_opt = base_config.batch_size * base_config.max_seq_len * base_config.gradient_accumulation_steps
    est_total_steps = base_config.train_tokens // tokens_per_opt
    
    if est_total_steps <= 600:
        base_config.eval_milestones = (0, 50, 100, 150, 200, 300, 400, 500)
        base_config.log_every = 50
        base_config.eval_every = None
    elif est_total_steps <= 1500:
        base_config.eval_milestones = (0, 100, 250, 500, 750, 1000)
        base_config.log_every = 100
        base_config.eval_every = None
    elif est_total_steps <= 6000:
        base_config.eval_milestones = (0, 500, 1000, 2000, 3000, 4000, 5000)
        base_config.log_every = 250
        base_config.eval_every = None
    else:
        base_config.eval_milestones = (0, 1000, 5000, 10000, 20000, 30000, 40000, 50000)
        base_config.log_every = 1000
        base_config.eval_every = None
    
    # Calculate needed docs
    avg_tokens_per_doc = 1000
    safety_factor = 2.0
    num_docs = max(100, int((base_config.train_tokens / avg_tokens_per_doc) * safety_factor))
    
    data_cfg = DataConfig(
        dataset_path=args.dataset_path if args.dataset_path else "auto",
        seq_length=base_config.max_seq_len,
        num_samples=num_docs,
        cache_dir="./hf_cache",
    )
    
    tokenizer = setup_tokenizer(data_cfg)
    base_config.vocab_size = tokenizer.vocab_size
    
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    
    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    loader_args = dict(
        batch_size=base_config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    
    # ============================
    # Run experiments
    # ============================
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"new_research/head_orthogonal/results_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = {
        "baseline": {"use_head_ortho": False, "use_qk_norm": True},
        "headortho": {"use_head_ortho": True, "use_qk_norm": True},
        "headortho_no_qknorm": {"use_head_ortho": True, "use_qk_norm": False},
    }
    
    if args.experiment != "all":
        experiments = {args.experiment: experiments[args.experiment]}
    
    all_results = {}
    for exp_name, exp_cfg in experiments.items():
        # Clone config for this run
        config = LLMConfig()
        config.train_tokens = base_config.train_tokens
        config.batch_size = base_config.batch_size
        config.vocab_size = base_config.vocab_size
        config.use_qk_norm = exp_cfg["use_qk_norm"]
        config.compile_model = True
        config.eval_milestones = base_config.eval_milestones
        config.log_every = base_config.log_every
        config.eval_every = base_config.eval_every
        
        exp_output = str(results_dir / exp_name)
        
        results = run_single_experiment(
            exp_name=exp_name,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            use_head_ortho=exp_cfg["use_head_ortho"],
            output_dir=exp_output,
        )
        
        all_results[exp_name] = {
            "final_val_loss": results["final_metrics"]["val_loss"],
            "final_val_ppl": results["final_metrics"]["val_perplexity"],
            "final_val_acc": results["final_metrics"]["val_accuracy"],
            "training_time": results["training_time"],
            "steps": results["steps"],
        }
        
        print(f"\n  {exp_name}: Val Loss={results['final_metrics']['val_loss']:.4f}, "
              f"Val PPL={results['final_metrics']['val_perplexity']:.2f}")
    
    # ============================
    # Summary
    # ============================
    print(f"\n{'='*70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for name, res in all_results.items():
        print(f"  {name:30s} | Val Loss: {res['final_val_loss']:.4f} | "
              f"Val PPL: {res['final_val_ppl']:.2f} | "
              f"Time: {res['training_time']/60:.1f}min")
    print(f"{'='*70}")
    
    # Save summary
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
