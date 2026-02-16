from __future__ import annotations

from typing import Any, Tuple, Dict, Sequence
from torch import nn
from torch_geometric.data import Data
import torch
from .helpers import set_seed, pick_cutoffs_similarity


def train(
    model: nn.Module,
    G: Data,
    modelists: torch.Tensor,
    num_clusters: int,
    num_modelists_clusters: int,
    *,

    lr: float = 1e-2,
    prop_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    pull_weight: float = 0.6,
    num_epochs: int = 20,
    seed: int = 5507,
    alpha=0.8, 
    max_iter=5,
    prop_beta = 0.03,
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Trains ModelistsGCN end-to-end by iteratively computing embeddings, 
    initializing a GMM from modelists, generating soft labels via propagation, 
    and optimizing a weighted combination of propagation, contrastive, and GMM-pull losses.
    """
    print("\n=== Training ===")

    set_seed(seed)
    model = model.to(G.x.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    neg_cut, pos_cut = pick_cutoffs_similarity(G.x, pca_dim=2)
    
    for epoch in range(num_epochs):
        if epoch%5 == 0:
            print(f"epoch {epoch}")
            
        model.train()
        optimizer.zero_grad()
        z = model(G)

        model.initialize_GMM_from_modelists(z, modelists, num_clusters, num_modelists_clusters)       

        gmm_posteriors = model.compute_gmm_posteriors(z)
        soft_modelists,_ = model.label_propagation_weights(G, modelists, num_clusters, alpha=alpha, max_iter=max_iter)
        confidence = soft_modelists.max(dim=1).values
        soft_mask = (modelists != -1) & (confidence > (1 / num_clusters + prop_beta))
        prop_loss = model.loss_propagation(gmm_posteriors, soft_modelists, soft_mask)
    
        contrast_loss = model.loss_feature_modelist_contrastive(
            z, G.x, modelists, modelist_mask=(modelists != -1), edge_index=G.edge_index, feature_sim_thresh=(neg_cut, pos_cut)
        )

        pull_loss = model.loss_gmm(z, epoch, num_epochs/10)

        total_loss = (
            prop_weight * prop_loss
            + contrastive_weight * contrast_loss
            + pull_weight * pull_loss
        )
 
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():    
        z = model(G)
      
    return model, z
