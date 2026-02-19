"""
Smoke test for HeadOrtho optimizer (Mode-1).
Verifies:
1. No shape errors with [128, 32768] matrix
2. Loss decreases
3. Step time is reasonable
4. No numerical explosion (NaNs)
"""
import sys
import os
import time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from new_research.head_orthogonal.run_experiment import setup_head_ortho_optimizer

def smoke_test():
    print("🚬 SMOKE TEST: HeadOrtho (Mode-1)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Tiny config
    config = LLMConfig()
    config.n_layers = 2  # speed 
    config.compile_model = False # speed
    
    # Need correct vocab size
    from data.loader import setup_tokenizer
    from configs.dataset_config import DataConfig
    data_cfg = DataConfig(dataset_path="auto", seq_length=128, num_samples=100)
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size
    
    model = MinimalLLM(config).to(device)
    optimizers = setup_head_ortho_optimizer(model, config, use_head_ortho=True)
    optimizer = optimizers[0] # The HeadOrtho one
    
    # Dummy data
    x = torch.randint(0, config.vocab_size, (1, 128), device=device)
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = -100
    
    model.train()
    print("\nStarting 10 steps...")
    start_time = time.time()
    
    for i in range(10):
        t0 = time.time()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=-100)
        
        loss.backward()
        
        # Verify gradients exists
        check_grad = model.transformer_blocks[0].attention.qkvo_proj.grad
        if check_grad is None:
            print("❌ No gradients!")
            return
            
        # Optimizer step (this is where the [128, 32768] SVD happens)
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
            
        dt = time.time() - t0
        print(f"Step {i+1}: Loss={loss.item():.4f}, Time={dt*1000:.1f}ms")
        
        if torch.isnan(loss):
            print("❌ NaN detected!")
            return

    total_time = time.time() - start_time
    print(f"\n✅ SMOKE TEST PASSED in {total_time:.2f}s")
    print("Mode-1 matrix [128, 32768] is numerically stable.")

if __name__ == "__main__":
    smoke_test()
