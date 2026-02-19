import torch
import numpy as np


def compute_pr(K):
    """Participation Ratio of matrix K [tokens, dk]."""
    K = K.to(torch.float32)
    try:
        s = torch.linalg.svdvals(K).cpu().numpy()
    except RuntimeError:
        return None
    return float((np.sum(s)**2) / np.sum(s**2))


def extract_gamma_stats(norm_module):
    """Extract γ non-uniformity stats. Returns None if no γ (Identity)."""
    if not hasattr(norm_module, 'weight'):
        return None
    gamma = norm_module.weight.detach().float()
    g_mean = float(gamma.mean())
    g_std = float(gamma.std())
    return {
        "mean": g_mean,
        "std": g_std,
        "cv": g_std / abs(g_mean) if abs(g_mean) > 1e-8 else 0.0,
        "min": float(gamma.min()),
        "max": float(gamma.max()),
        "values": gamma.cpu().numpy().tolist(),
    }


class RankProbe:
    """Probes key representations and γ parameters."""
    def __init__(self, model, device, eval_batch):
        self.model = model
        self.device = device
        self.eval_batch = eval_batch

    def run_probe(self):
        """Returns dict: layer_idx -> {k_pr, gamma_k}"""
        self.model.eval()
        hooks = []
        captured_ks = {}

        def get_hook(layer_idx):
            def hook(module, input, output):
                captured_ks[layer_idx] = output.detach()
            return hook

        for i, block in enumerate(self.model.transformer_blocks):
            hooks.append(block.attention.k_norm.register_forward_hook(get_hook(i)))

        with torch.no_grad():
            _ = self.model(self.eval_batch.to(self.device))

        for h in hooks:
            h.remove()

        layer_metrics = {}
        for layer_idx, K_BTHD in captured_ks.items():
            B, T, H, D = K_BTHD.shape
            head_prs = []
            for h in range(H):
                pr = compute_pr(K_BTHD[:, :, h, :].reshape(-1, D))
                if pr is not None:
                    head_prs.append(pr)

            if head_prs:
                block = self.model.transformer_blocks[layer_idx]
                layer_metrics[layer_idx] = {
                    "k_pr": float(np.mean(head_prs)),
                    "gamma_k": extract_gamma_stats(block.attention.k_norm),
                }

        return layer_metrics
