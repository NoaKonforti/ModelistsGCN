from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from typing import Tuple, Sequence
import numpy as np
import torch

def evaluate_unsupervised(
    embeddings: torch.Tensor,
    cluster_labels: Sequence[int]
) -> float:
    embeddings_np = embeddings.detach().cpu().numpy()
    score = silhouette_score(embeddings_np, cluster_labels)
    return score

def evaluate_with_ground_truth(
    cluster_labels: Sequence[int],
    true_labels: Sequence[int]
) -> Tuple[float, float]:
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    return ari, nmi

