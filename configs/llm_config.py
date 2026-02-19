from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LLMConfig:
    # Model architecture (~1.5B Params)
    d_model: int = 2048       
    n_heads: int = 16         # d_k = 128
    n_layers: int = 12
    d_ff: int = 8192         
    
    # GQA parameters
    n_kv_heads: int = 8      
    
    # Data params
    # WARNING: If you change max_seq_len, you MUST re-run data preparation!
    max_seq_len: int = 2048
    vocab_size: int = 49152  
    use_qk_norm: bool = True
    use_muon: bool = True
    
    # Training
    compile_model: bool = True
    gradient_checkpointing: bool = True
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    train_tokens: int = 100000000  # 100M tokens
    
    # Learning Rate
    muon_lr: float = 0.02  # Slight increase for larger batch
    muon_momentum: float = 0.95
    adamw_lr: float = 0.003
    warmup_ratio: float = 0.01
    schedule_type: str = "cosine"

    # Evaluation
    eval_every: Optional[int] = None
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None
    
    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
