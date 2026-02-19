"""
HeadOrtho Muon: Head-wise Tensor Orthogonalization for Muon Optimizer

Inspired by Teon (Tensorized Orthonormalization Beyond Layer-Wise Muon),
which stacks gradients across *layers* into a tensor and orthogonalizes.

Our idea: stack Q/K gradients across *attention heads* into a tensor and
orthogonalize, enforcing cross-head independence/orthogonality to prevent
attention head collapse.

For Q projection gradient G_Q ∈ R^{(H*d_k) × d_model}:
  1. Reshape to tensor:  T ∈ R^{d_k × d_model × H}
  2. Mode-1 matricize:   M_1(T) ∈ R^{d_k × (d_model * H)}
  3. Ortho:              O = UV^T of M_1(T)
  4. Fold back:          R^{d_k × d_model × H}
  5. Reshape to matrix:  R^{(H*d_k) × d_model}
"""

import torch
import torch.nn.functional as F
from optimizers.muon import zeropower_polar_express


class HeadOrthoMuon(torch.optim.Optimizer):
    """
    Muon optimizer with head-wise tensor orthogonalization for Q/K projections.
    
    For parameters identified as Q/K projections (via head_info metadata),
    the gradient is reshaped into a tensor [d_k, d_model, H], mode-1 matricized
    to [d_k, d_model*H], orthogonalized, and folded back.
    
    For all other 2D parameters, standard Muon orthogonalization is applied.
    
    Args:
        params: iterable of parameters or dicts. Each param group can have:
            - head_info: dict with keys:
                - 'n_heads': number of attention heads for this param
                - 'd_k': head dimension
                - 'param_type': 'q' or 'k' (which part of qkvo_proj)
                - 'q_size': size of Q projection 
                - 'kv_size': size of K/V projection
            If head_info is not provided, standard Muon is used.
        lr: learning rate
        momentum: momentum coefficient
        nesterov: use Nesterov momentum
        ns_steps: number of Newton-Schulz / Polar Express iteration steps
        ortho_mode: which tensor mode to orthogonalize along (1 or 2)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=5, ortho_mode=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                       ns_steps=ns_steps, ortho_mode=ortho_mode)
        super().__init__(params, defaults)

    def _head_ortho(self, g, n_heads, d_k, d_model, ns_steps, mode=1):
        """
        Apply tensor orthogonalization across attention heads.
        
        g: gradient tensor of shape [n_heads * d_k, d_model]
        n_heads: number of attention heads
        d_k: head dimension
        d_model: model dimension
        mode: which mode to matricize (1 or 2)
        
        Returns: orthogonalized gradient of same shape
        """
        # Step 1: Reshape to tensor [d_k, d_model, H]
        # g is [H*d_k, d_model], first reshape to [H, d_k, d_model]
        g_tensor = g.reshape(n_heads, d_k, d_model)  # [H, d_k, d_model]
        
        if mode == 1:
            # Mode-1 matricization: unfold along d_k dimension
            # Result: [d_k, d_model * H]
            g_mat = g_tensor.permute(1, 2, 0).reshape(d_k, d_model * n_heads)
            
            # Orthogonalize
            g_ortho_mat = zeropower_polar_express(g_mat, steps=ns_steps)
            
            # Fold back: [d_k, d_model, H] → [H, d_k, d_model] → [H*d_k, d_model]
            g_ortho = g_ortho_mat.reshape(d_k, d_model, n_heads).permute(2, 0, 1).reshape(n_heads * d_k, d_model)
            
        elif mode == 2:
            # Mode-2 matricization: unfold along d_model dimension
            # Result: [d_model, d_k * H]
            g_mat = g_tensor.permute(2, 1, 0).reshape(d_model, d_k * n_heads)
            
            # Orthogonalize
            g_ortho_mat = zeropower_polar_express(g_mat, steps=ns_steps)
            
            # Fold back: [d_model, d_k, H] → [H, d_k, d_model] → [H*d_k, d_model]
            g_ortho = g_ortho_mat.reshape(d_model, d_k, n_heads).permute(2, 1, 0).reshape(n_heads * d_k, d_model)
            
        elif mode == 3:
            # Mode-3 matricization: unfold along heads dimension
            # Result: [H, d_k * d_model]
            g_mat = g_tensor.reshape(n_heads, d_k * d_model)
            
            # Orthogonalize
            g_ortho_mat = zeropower_polar_express(g_mat, steps=ns_steps)
            
            # Fold back: [H, d_k * d_model] → [H, d_k, d_model] → [H*d_k, d_model]
            g_ortho = g_ortho_mat.reshape(n_heads, d_k, d_model).reshape(n_heads * d_k, d_model)
        
        else:
            raise ValueError(f"Invalid ortho_mode: {mode}, must be 1, 2, or 3")
            
        return g_ortho

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            head_info = group.get("head_info", None)
            ns_steps = group["ns_steps"]
            ortho_mode = group.get("ortho_mode", 1)
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                if head_info is not None and p.ndim == 2:
                    # This is a qkvo_proj parameter — apply head ortho to Q and K parts
                    n_heads = head_info["n_heads"]
                    n_kv_heads = head_info["n_kv_heads"]
                    d_k = head_info["d_k"]
                    d_model = head_info["d_model"]
                    q_size = head_info["q_size"]
                    kv_size = head_info["kv_size"]
                    
                    # Split gradient into Q, K, V, O parts
                    # qkvo_proj shape: [q_size + 2*kv_size + o_size, d_model]
                    g_q = g[:q_size]           # [H*d_k, d_model]
                    g_k = g[q_size:q_size+kv_size]  # [n_kv_heads*d_k, d_model]
                    g_rest = g[q_size+kv_size:]      # V + O parts
                    
                    # Apply head-wise tensor orthogonalization to Q
                    g_q_ortho = self._head_ortho(g_q, n_heads, d_k, d_model, ns_steps, mode=ortho_mode)
                    
                    # Apply head-wise tensor orthogonalization to K
                    g_k_ortho = self._head_ortho(g_k, n_kv_heads, d_k, d_model, ns_steps, mode=ortho_mode)
                    
                    # Standard Muon ortho for V + O
                    g_rest_ortho = zeropower_polar_express(g_rest, steps=ns_steps)
                    
                    # Concatenate back
                    g = torch.cat([g_q_ortho, g_k_ortho, g_rest_ortho], dim=0)
                    
                elif p.ndim == 2:
                    # Standard Muon for other 2D params
                    g = zeropower_polar_express(g, steps=ns_steps)
                
                g = g.to(p.dtype)
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
