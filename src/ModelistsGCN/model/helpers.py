import numpy as np
import torch
import random
from typing import Tuple
import torch.nn.functional as F
from sklearn.decomposition import PCA

def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic seeds across Python, NumPy, and PyTorch (including CUDA) 
    and forces deterministic CUDA/cuDNN behavior for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)  # For full determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
@torch.no_grad()
def pick_cutoffs_similarity(
    raw_features: torch.Tensor,
    pca_dim: int = 16,
    fallback_percentiles=(15, 85),
)-> Tuple[float, float]:
    """
    Projects features into PCA space, computes pairwise cosine similarities, 
    and picks data-driven low/high similarity cutoffs (percentile-based) 
    for negative/positive thresholds.
    """
    device = raw_features.device

    d_x = raw_features.shape[1]
    pca_dim = min(pca_dim, d_x)
    x_np = raw_features.detach().cpu().numpy()
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=42)
    x_pca = pca.fit_transform(x_np)
    x_pca = torch.from_numpy(x_pca).to(device=device, dtype=torch.float32)
    x_pca = F.normalize(x_pca, dim=1)

    sim = torch.mm(x_pca, x_pca.t())      # (N, N) in [-1, 1]
    n = sim.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    vals = sim[iu[0], iu[1]].detach().cpu().numpy()

    lo, hi = np.percentile(vals, [fallback_percentiles[0], fallback_percentiles[1]])
    neg_cut, pos_cut = float(lo), float(hi)

    if not (neg_cut < pos_cut):
        # if distribution is degenerate, widen a bit
        eps = 1e-3
        neg_cut, pos_cut = min(neg_cut, pos_cut - eps), max(pos_cut, neg_cut + eps)

    return neg_cut, pos_cut
