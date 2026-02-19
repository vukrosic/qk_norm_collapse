#!/usr/bin/env python3
"""
QK-Norm Rank Collapse — Causal Study (Simplified)

3 conditions, 1 metric (Key PR), 5 plots.
  A) QK-Norm, learned γ    — full effect
  B) QK-Norm, frozen γ=1   — normalization only
  C) No QK-Norm             — baseline
"""
import torch
import torch.nn.functional as F
import sys, json, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.llm import MinimalLLM
from configs.llm_config_1b import LLMConfig1B
from training.trainer import setup_muon_optimizer
from research.svd_probe import RankProbe
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from train_llm import prepare_datasets

# ============================================================
TARGET_TOKENS = 50_000_000
PROBE_EVERY = 120
BATCH_SIZE = 8           # H100 80GB — 16 OOMs with compile, 8 is safe
GRAD_ACCUM = 4           # effective batch = 8*4*2048 = 65,536 tok/step
SEED = 42
OUT = Path("research_results/qk_norm_50m_study")
DATA = "processed_data/pretrain_1B"
# ============================================================


def run(use_qk_norm, freeze_gamma=False):
    tag = "QK_frozen" if freeze_gamma else ("QK" if use_qk_norm else "NoQK")
    print(f"\n{'='*60}\n  RUN: {tag}\n{'='*60}")

    config = LLMConfig1B()
    config.use_qk_norm = use_qk_norm
    config.train_tokens = TARGET_TOKENS
    config.batch_size = BATCH_SIZE
    config.gradient_accumulation_steps = GRAD_ACCUM
    config.compile_model = True            # H100 benefits from torch.compile
    config.gradient_checkpointing = True    # needed for 1.5B even on 80GB with compile
    device = torch.device('cuda')

    # Data — the 1B dataset only has a 'train' split, so we carve out val ourselves
    data_cfg = DataConfig(dataset_path=DATA, seq_length=config.max_seq_len)
    tokenizer = setup_tokenizer(data_cfg)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    torch.manual_seed(SEED + 999)
    eval_batch = next(iter(val_loader))["input_ids"].to(device)

    # Model
    torch.manual_seed(SEED)
    model = MinimalLLM(config).to(device)
    raw_model = model  # keep ref for probe/gamma access before compile wraps it
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Batch size: {BATCH_SIZE}, Grad accum: {GRAD_ACCUM}, "
          f"Effective batch: {BATCH_SIZE * GRAD_ACCUM * config.max_seq_len:,} tok/step")

    # Freeze γ for condition B (must happen before compile)
    if freeze_gamma:
        for block in raw_model.transformer_blocks:
            if hasattr(block.attention.k_norm, 'weight'):
                block.attention.k_norm.weight.requires_grad_(False)
            if hasattr(block.attention.q_norm, 'weight'):
                block.attention.q_norm.weight.requires_grad_(False)
        print("  🔒 γ frozen at 1.0")

    # Probe uses hooks on transformer_blocks — set up on raw model
    probe = RankProbe(raw_model, device, eval_batch)
    optimizers = setup_muon_optimizer(model, config)

    # Compile for H100 speedup (after probe/freeze setup)
    if config.compile_model:
        print("  ⚡ Compiling model with torch.compile...")
        model = torch.compile(model)

    res = {"tag": tag, "tokens": [], "loss": [], "val_loss": [], "mean_pr": [], "layer_pr": {}, "gamma_cv": {}}
    run_dir = OUT / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initial probe
    pr = probe.run_probe()
    _record(res, pr, 0, 0, float('nan'), float('nan'))

    # Train
    model.train()
    tokens = 0; step = 0; rloss = 0.0; lcount = 0
    pbar = tqdm(total=TARGET_TOKENS, desc=tag, unit="tok")

    while tokens < TARGET_TOKENS:
        for batch in train_loader:
            if tokens >= TARGET_TOKENS:
                break
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
                (loss / GRAD_ACCUM).backward()

            rloss += loss.item(); lcount += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for opt in optimizers:
                    opt.step(); opt.zero_grad()

            tokens += x.numel(); step += 1; pbar.update(x.numel())

            if step % PROBE_EVERY == 0:
                model.eval()
                pr = probe.run_probe()
                # Val loss (4 batches)
                vloss = 0.0; vn = 0
                with torch.no_grad():
                    for vb in list(val_loader)[:4]:
                        vx = vb["input_ids"].to(device); vy = vb["labels"].to(device)
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            vlogits = model(vx)
                            vsl = torch.full_like(vy, -100); vsl[:, :-1] = vy[:, 1:]
                            vl = F.cross_entropy(vlogits.view(-1, config.vocab_size), vsl.view(-1), ignore_index=-100)
                        vloss += vl.item(); vn += 1
                val_l = vloss / max(vn, 1)
                model.train()
                avg = rloss / max(lcount, 1)
                _record(res, pr, tokens, step, avg, val_l)
                pbar.set_postfix(loss=f"{avg:.3f}", vl=f"{val_l:.3f}", pr=f"{res['mean_pr'][-1]:.1f}")
                rloss = 0.0; lcount = 0
                with open(run_dir / "results.json", "w") as f:
                    json.dump(res, f)

    pbar.close()

    # Final gamma save
    if use_qk_norm and not freeze_gamma:
        res["gamma_final"] = {}
        for i, block in enumerate(raw_model.transformer_blocks):
            if hasattr(block.attention.k_norm, 'weight'):
                res["gamma_final"][str(i)] = block.attention.k_norm.weight.detach().float().cpu().tolist()

    with open(run_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)

    del model, raw_model, optimizers, probe, eval_batch
    torch.cuda.empty_cache(); gc.collect()
    return res


def _record(res, pr, tokens, step, avg_loss, val_loss):
    mean_pr = np.mean([v["k_pr"] for v in pr.values()])
    res["tokens"].append(int(tokens))
    res["loss"].append(float(avg_loss) if not np.isnan(avg_loss) else None)
    res["val_loss"].append(float(val_loss) if not np.isnan(val_loss) else None)
    res["mean_pr"].append(float(mean_pr))

    for l, m in pr.items():
        k = str(l)
        res["layer_pr"].setdefault(k, []).append(float(m["k_pr"]))
        if m.get("gamma_k") and "cv" in m["gamma_k"]:
            res["gamma_cv"].setdefault(k, []).append(float(m["gamma_k"]["cv"]))

    vl = f" | Val {val_loss:.4f}" if not np.isnan(val_loss) else ""
    print(f"  Step {step:>6d} | {tokens:>10,} tok | Loss {avg_loss:.4f}{vl} | PR {mean_pr:.1f}")


# ============================================================
# Plotting — 5 panels
# ============================================================
CA, CB, CC = '#f97316', '#a855f7', '#22d3ee'
LO = dict(facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')

def fig():
    f, a = plt.subplots(figsize=(12, 7))
    f.patch.set_facecolor('#0d1117'); a.set_facecolor('#161b22')
    a.tick_params(colors='#c9d1d9'); a.xaxis.label.set_color('#c9d1d9')
    a.yaxis.label.set_color('#c9d1d9'); a.title.set_color('#e6edf3')
    for s in a.spines.values(): s.set_color('#30363d')
    return f, a

def sax(a):
    a.set_facecolor('#161b22'); a.tick_params(colors='#c9d1d9')
    a.xaxis.label.set_color('#c9d1d9'); a.yaxis.label.set_color('#c9d1d9')
    a.title.set_color('#e6edf3')
    for s in a.spines.values(): s.set_color('#30363d')

def M(t): return [x/1e6 for x in t]

def save(f, name):
    p = OUT / name; f.savefig(p, dpi=180, bbox_inches='tight', facecolor='#0d1117')
    plt.close(f); print(f"  📈 {p}")


def plot(a, b, c):
    conds = [(a, "Learned γ", CA), (b, "Frozen γ=1", CB), (c, "No QK-Norm", CC)]

    # 1. Loss (train solid, val dashed)
    f, ax = fig()
    for r, lb, co in conds:
        t = [x for x, l in zip(r["tokens"], r["loss"]) if l]; ls = [l for l in r["loss"] if l]
        if t: ax.plot(M(t), ls, color=co, lw=2, label=f'{lb} (train)')
        vt = [x for x, l in zip(r["tokens"], r["val_loss"]) if l]; vls = [l for l in r["val_loss"] if l]
        if vt: ax.plot(M(vt), vls, color=co, lw=1.5, ls='--', alpha=0.7, label=f'{lb} (val)')
    ax.set_xlabel('Tokens (M)'); ax.set_ylabel('Loss'); ax.set_title('Train & Validation Loss')
    ax.legend(**LO, fontsize=8); ax.grid(True, alpha=0.15, color='#484f58'); save(f, "1_loss.png")

    # 2. Key PR — THE ANSWER
    f, ax = fig()
    for r, lb, co in conds:
        ax.plot(M(r["tokens"]), r["mean_pr"], color=co, lw=2.5, label=lb, marker='o', ms=2)
    ax.set_xlabel('Tokens (M)'); ax.set_ylabel('Mean Key PR')
    ax.set_title('Key Rank Collapse — Does γ or Normalization Drive It?')
    ax.legend(**LO); ax.grid(True, alpha=0.15, color='#484f58')
    ax.text(0.02, 0.02, 'A≠B≈C → γ drives it\nA≈B≠C → normalization\nAll differ → both',
            transform=ax.transAxes, fontsize=9, color='#8b949e', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#30363d', alpha=0.8))
    save(f, "2_key_pr_causal.png")

    # 3. Per-layer bars at final checkpoint
    layers = sorted(int(k) for k in a["layer_pr"])
    f, ax = fig((14, 7))
    w = 0.25; x = np.arange(len(layers))
    for i, (r, lb, co) in enumerate(conds):
        pr = [r["layer_pr"][str(l)][-1] for l in layers]
        ax.bar(x + (i-1)*w, pr, w, color=co, alpha=0.85, label=lb)
    ax.set_xlabel('Layer'); ax.set_ylabel('Key PR'); ax.set_title('Per-Layer Key PR at Final Checkpoint')
    ax.set_xticks(x[::4]); ax.set_xticklabels([str(l) for l in layers[::4]])
    ax.legend(**LO); ax.grid(True, alpha=0.15, color='#484f58', axis='y'); save(f, "3_per_layer.png")

    # 4. Causal decomposition: Δ(A-B) vs Δ(B-C)
    f, ax = fig((14, 7))
    apr = [a["layer_pr"][str(l)][-1] for l in layers]
    bpr = [b["layer_pr"][str(l)][-1] for l in layers]
    cpr = [c["layer_pr"][str(l)][-1] for l in layers]
    d_ab = [x-y for x, y in zip(apr, bpr)]
    d_bc = [x-y for x, y in zip(bpr, cpr)]
    ax.bar(x-0.15, d_ab, 0.3, color=CA, alpha=0.85, label='Δ(A−B): γ effect')
    ax.bar(x+0.15, d_bc, 0.3, color=CB, alpha=0.85, label='Δ(B−C): norm effect')
    ax.axhline(0, color='#8b949e', ls='--')
    ax.set_xlabel('Layer'); ax.set_ylabel('ΔPR'); ax.set_title('Causal Decomposition per Layer')
    ax.set_xticks(x[::4]); ax.set_xticklabels([str(l) for l in layers[::4]])
    ax.legend(**LO); ax.grid(True, alpha=0.15, color='#484f58', axis='y')
    ax.text(0.02, 0.95, f'Mean |γ effect|: {np.mean(np.abs(d_ab)):.2f}\nMean |norm effect|: {np.mean(np.abs(d_bc)):.2f}',
            transform=ax.transAxes, fontsize=10, color='#c9d1d9', va='top',
            bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#30363d'))
    save(f, "4_causal_decomposition.png")

    # 5. γ-CV evolution (condition A only)
    if a.get("gamma_cv"):
        f, ax = fig()
        sample = [0, 8, 15, 23, 30]
        cm = plt.cm.plasma
        for i, l in enumerate(sample):
            cvs = a["gamma_cv"].get(str(l), [])
            if cvs:
                t = a["tokens"][:len(cvs)]
                ax.plot(M(t), cvs, lw=2, label=f'L{l}', color=cm(i/(len(sample)-1)), marker='o', ms=2)
        ax.set_xlabel('Tokens (M)'); ax.set_ylabel('γ-CV (std/mean)')
        ax.set_title('γ Non-Uniformity Over Training (Condition A)')
        ax.legend(**LO, fontsize=9); ax.grid(True, alpha=0.15, color='#484f58')
        save(f, "5_gamma_cv.png")


# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("  QK-NORM CAUSAL STUDY — 3 CONDITIONS × 50M TOKENS")
    print("="*60)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ra = run(use_qk_norm=True, freeze_gamma=False)
    rb = run(use_qk_norm=True, freeze_gamma=True)
    rc = run(use_qk_norm=False)

    print("\n📊 Plotting...")
    plot(ra, rb, rc)

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed/60:.1f} min — results in {OUT}/")

    with open(OUT / "combined.json", "w") as f:
        json.dump({"A": ra, "B": rb, "C": rc, "elapsed_s": elapsed}, f, indent=2)
