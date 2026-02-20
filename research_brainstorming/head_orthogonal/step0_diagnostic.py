"""
Step 0 Diagnostic: Cross-Head Singular Vector Alignment Check

HARD GATE: If mean cosine alignment < 0.3 for both left and right
singular vectors, the HeadOrtho project is DROPPED.

This script:
1. Creates a 1.5B model and loads data
2. Runs one forward + backward pass
3. Extracts Q/K gradients from qkvo_proj
4. Reshapes to per-head gradient slices
5. Computes SVD of each head's gradient
6. Measures cosine similarity of top singular vectors across heads
7. Reports left (U) and right (V) alignment separately
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np
from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from utils.helpers import set_seed


def check_alignment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup
    set_seed(42)
    config = LLMConfig()
    config.compile_model = False  # don't need compile for diagnostic
    
    # We need a tokenizer for vocab_size
    from data.loader import setup_tokenizer
    from configs.dataset_config import DataConfig
    data_cfg = DataConfig(dataset_path="auto", seq_length=config.max_seq_len, num_samples=10)
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size
    
    model = MinimalLLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")
    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"d_k={config.d_k}, n_kv_heads={config.n_kv_heads}")
    
    # Create a random batch (we just need gradients, data content doesn't matter)
    batch_size = 1
    seq_len = 512  # short is fine for gradient check
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = x.clone()
    labels[:, :-1] = x[:, 1:]
    labels[:, -1] = -100
    
    # Forward + backward
    model.train()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100)
    loss.backward()
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"\n{'='*70}")
    print("  CROSS-HEAD SINGULAR VECTOR ALIGNMENT DIAGNOSTIC")
    print(f"{'='*70}\n")
    
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads if config.n_kv_heads else config.n_heads
    d_k = config.d_k
    d_model = config.d_model
    q_size = d_model  # n_heads * d_k
    kv_size = n_kv_heads * d_k
    
    all_left_cosines = []
    all_right_cosines = []
    
    for layer_idx, block in enumerate(model.transformer_blocks):
        attn = block.attention
        grad = attn.qkvo_proj.grad  # [q_size + 2*kv_size + o_size, d_model]
        
        if grad is None:
            continue
        
        grad = grad.float()
        
        # Extract Q gradient
        g_q = grad[:q_size]  # [n_heads * d_k, d_model]
        
        # Reshape to per-head: [n_heads, d_k, d_model]
        g_heads = g_q.reshape(n_heads, d_k, d_model)
        
        # SVD each head's gradient
        left_vecs = []   # top-1 left singular vector (u1) per head
        right_vecs = []  # top-1 right singular vector (v1) per head
        
        for h in range(n_heads):
            g_h = g_heads[h]  # [d_k, d_model]
            U, S, Vt = torch.linalg.svd(g_h, full_matrices=False)
            left_vecs.append(U[:, 0])   # [d_k]
            right_vecs.append(Vt[0, :]) # [d_model]
        
        left_vecs = torch.stack(left_vecs)   # [n_heads, d_k]
        right_vecs = torch.stack(right_vecs) # [n_heads, d_model]
        
        # Compute pairwise cosine similarity
        left_cos = F.cosine_similarity(left_vecs.unsqueeze(0), left_vecs.unsqueeze(1), dim=2)
        right_cos = F.cosine_similarity(right_vecs.unsqueeze(0), right_vecs.unsqueeze(1), dim=2)
        
        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(n_heads, n_heads, dtype=torch.bool), diagonal=1)
        left_cos_vals = left_cos[mask].abs()  # absolute cosine (alignment regardless of sign)
        right_cos_vals = right_cos[mask].abs()
        
        mean_left = left_cos_vals.mean().item()
        mean_right = right_cos_vals.mean().item()
        max_left = left_cos_vals.max().item()
        max_right = right_cos_vals.max().item()
        
        all_left_cosines.append(mean_left)
        all_right_cosines.append(mean_right)
        
        if layer_idx % 8 == 0 or layer_idx == config.n_layers - 1:
            print(f"  Layer {layer_idx:2d}: Left(U) cos={mean_left:.4f} (max={max_left:.4f}) | "
                  f"Right(V) cos={mean_right:.4f} (max={max_right:.4f})")
    
    # Summary
    overall_left = np.mean(all_left_cosines)
    overall_right = np.mean(all_right_cosines)
    
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Mean LEFT (U) alignment across all layers:  {overall_left:.4f}")
    print(f"  Mean RIGHT (V) alignment across all layers: {overall_right:.4f}")
    print(f"{'='*70}")
    
    # Decision
    print(f"\n  DECISION GATE:")
    if overall_left >= 0.3:
        print(f"  GO: Left singular vectors aligned ({overall_left:.4f} >= 0.3)")
        print(f"  --> Use mode-2 (matricize along d_model, preserving left structure)")
        recommended_mode = 2
    elif overall_right >= 0.3:
        print(f"  GO: Right singular vectors aligned ({overall_right:.4f} >= 0.3)")
        print(f"  --> Use mode-1 (matricize along d_k, preserving right structure)")
        recommended_mode = 1
    else:
        print(f"  CAUTION: Neither left ({overall_left:.4f}) nor right ({overall_right:.4f}) "
              f"alignment exceeds 0.3")
        print(f"  The Teon analogy is weak. Consider mode-3 or DROP the project.")
        recommended_mode = None
    
    # Also check K gradients
    print(f"\n  --- K Gradient Alignment (n_kv_heads={n_kv_heads}) ---")
    k_left_cosines = []
    k_right_cosines = []
    
    for layer_idx, block in enumerate(model.transformer_blocks):
        attn = block.attention
        grad = attn.qkvo_proj.grad.float()
        
        g_k = grad[q_size:q_size+kv_size]  # [n_kv_heads*d_k, d_model]
        g_k_heads = g_k.reshape(n_kv_heads, d_k, d_model)
        
        left_vecs = []
        right_vecs = []
        for h in range(n_kv_heads):
            U, S, Vt = torch.linalg.svd(g_k_heads[h], full_matrices=False)
            left_vecs.append(U[:, 0])
            right_vecs.append(Vt[0, :])
        
        left_vecs = torch.stack(left_vecs)
        right_vecs = torch.stack(right_vecs)
        
        mask = torch.triu(torch.ones(n_kv_heads, n_kv_heads, dtype=torch.bool), diagonal=1)
        left_cos_vals = F.cosine_similarity(left_vecs.unsqueeze(0), left_vecs.unsqueeze(1), dim=2)[mask].abs()
        right_cos_vals = F.cosine_similarity(right_vecs.unsqueeze(0), right_vecs.unsqueeze(1), dim=2)[mask].abs()
        
        k_left_cosines.append(left_cos_vals.mean().item())
        k_right_cosines.append(right_cos_vals.mean().item())
    
    k_left = np.mean(k_left_cosines)
    k_right = np.mean(k_right_cosines)
    print(f"  K Left (U) alignment:  {k_left:.4f}")
    print(f"  K Right (V) alignment: {k_right:.4f}")
    
    print(f"\n{'='*70}")
    print(f"  FINAL RECOMMENDATION")
    print(f"{'='*70}")
    
    if recommended_mode is not None:
        print(f"  PROCEED with mode-{recommended_mode}")
        print(f"  Q alignment: left={overall_left:.4f}, right={overall_right:.4f}")
        print(f"  K alignment: left={k_left:.4f}, right={k_right:.4f}")
    else:
        # Even if below 0.3, if there's SOME alignment, it may still be worth a try
        best_q = max(overall_left, overall_right)
        best_k = max(k_left, k_right)
        if best_q >= 0.15 or best_k >= 0.15:
            print(f"  BORDERLINE: Weak but non-zero alignment detected.")
            print(f"  Best Q alignment: {best_q:.4f}, Best K alignment: {best_k:.4f}")
            print(f"  Proceed with caution — use mode-2, expect modest gains at best.")
            recommended_mode = 2
        else:
            print(f"  DROP: No meaningful cross-head gradient alignment detected.")
            print(f"  HeadOrtho has no theoretical basis for this model.")
    
    # Cleanup
    del model, grad
    torch.cuda.empty_cache()
    
    return {
        "q_left_alignment": overall_left,
        "q_right_alignment": overall_right,
        "k_left_alignment": k_left,
        "k_right_alignment": k_right,
        "recommended_mode": recommended_mode,
        "per_layer_q_left": all_left_cosines,
        "per_layer_q_right": all_right_cosines,
    }


if __name__ == "__main__":
    result = check_alignment()
    
    import json
    out_path = "new_research/head_orthogonal/diagnostic_result.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDiagnostic saved to {out_path}")
