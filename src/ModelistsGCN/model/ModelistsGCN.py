import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.stats import chi2
import math
from torch_geometric.data import Data


class ModelistsGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, num_clusters):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.layers.append(GCNConv(hidden_dim, latent_dim))
        self.classifier_head = nn.Linear(latent_dim, num_clusters) 
        self.num_clusters = num_clusters
        
    def initialize_GMM_from_modelists(self, z, labels, num_classes, n_known_clusters, distance_threshold=25.0):
        """
        Initializes a tied-covariance sklearn GMM by computing “known” component means 
        from labeled modelists and selecting additional “unknown” component means 
        from a fallback GMM using PCA-space filtering (density + distance).
        """
        device = z.device
        dtype = z.dtype
        latent_dim = z.shape[1]
    
        # ---- PCA helper (project any (N,d) torch tensor to PCA space) ----
        PCA_DIM = 2  
        z_np_full = z.detach().cpu().numpy()
        pca = PCA(n_components=PCA_DIM, svd_solver="auto", random_state=42)
        z_pca_np = pca.fit_transform(z_np_full)  # (n_nodes, P)
        # quick projector for any (M,d) tensor:
        def _to_pca_space(x_t: torch.Tensor) -> torch.Tensor:
            x_np = x_t.detach().cpu().numpy()
            x_pca = pca.transform(x_np)  # (M, P)
            return torch.from_numpy(x_pca).to(device=device, dtype=torch.float32)
    
        # --- modelists ---
        known_mask = (labels != -1)
        known_labels = labels[known_mask]
        known_z = z[known_mask]
    
        means = []
        variances = []
    
        if known_labels.numel() > 0:
            unique_known = known_labels.unique()
            for cluster_id in unique_known:
                cluster_z = known_z[known_labels == cluster_id]
                mu = cluster_z.mean(dim=0)
                var = cluster_z.var(dim=0, unbiased=True)
                means.append(mu)
                variances.append(var)
    
        means = torch.stack(means) if len(means) else torch.empty(0, latent_dim, device=device, dtype=dtype)
        variances = torch.stack(variances) if len(variances) else torch.empty(0, latent_dim, device=device, dtype=dtype)
    
        # --- distance threshold handling in PCA space ---
        effective_threshold = float(distance_threshold)
        if means.size(0) >= 2:
            means_pca = _to_pca_space(means)  # (K_known, P)
            dists_pca = torch.cdist(means_pca, means_pca)  # (K_known, K_known)
            tri_mask = torch.triu(torch.ones_like(dists_pca, dtype=torch.bool), diagonal=1)
            upper_triangle = dists_pca[tri_mask]
            if upper_triangle.numel() > 0:
                min_sep = float(upper_triangle.min().item())
                effective_threshold = min_sep
    
        # --- fallback GMM just to propose unknown MEANS (same as before) ---
        extra_components = num_classes * 2
        fallback_gmm = GaussianMixture(
            n_components=extra_components,
            covariance_type='tied',
            reg_covar=1e-6,
            random_state=42
        )
        fallback_gmm.fit(z_np_full)
        fallback_means = torch.tensor(fallback_gmm.means_, device=z.device, dtype=torch.float32)
    
        # ---- compute candidates in PCA space for distance checks ----
        means_pca = _to_pca_space(means) if means.numel() > 0 else torch.empty(0, PCA_DIM, device=device)
        fallback_means_pca = _to_pca_space(fallback_means)
    
        Z_pca = torch.from_numpy(z_pca_np).to(device=device, dtype=torch.float32)  # all cells in PCA space

        # =========================
        # OUTLIER / DENSITY FILTER  
        # =========================     
        C_np = fallback_means_pca.detach().cpu().numpy()
        Z_np = Z_pca.detach().cpu().numpy()
        lw = LedoitWolf().fit(Z_np)
        mu = lw.location_
        VI = np.linalg.inv(lw.covariance_)                 
        diff = C_np - mu
        m2 = np.einsum('ij,jk,ik->i', diff, VI, diff)      
        alpha = 0.997                                      
        m2_cut = chi2.ppf(alpha, df=fallback_means_pca.shape[1])
        mask_maha = (m2 <= m2_cut)
        
        with torch.no_grad():
            k = 50  
            D_cand = torch.cdist(fallback_means_pca, Z_pca)                 # (n_cand, n_cells)
            kth_cand = torch.topk(D_cand, k=k, largest=False).values[:, -1] # (n_cand,)
        
            Dz = torch.cdist(Z_pca, Z_pca)
            Dz.fill_diagonal_(float('inf'))
            kth_cells = torch.topk(Dz, k=k, largest=False).values[:, -1]    # (n_cells,)
            knn_thresh = torch.quantile(kth_cells, 0.95)                    
        
        mask_knn = (kth_cand <= knn_thresh).cpu().numpy()
        
        dense_idx = np.where(mask_maha & mask_knn)[0].tolist()
        if len(dense_idx) == 0:  # safety fallback
            dense_idx = list(range(fallback_means_pca.shape[0]))
        
        # =========================
        # DISTANCE-THRESHOLD FILTER (in PCA space) on the dense candidates
        # =========================
        if means_pca.size(0) > 0:
            D = torch.cdist(fallback_means_pca[dense_idx], means_pca)  # (n_dense, K_known)
            ok = (D > effective_threshold).all(dim=1)                  # (n_dense,)
            kept_indices = [dense_idx[j] for j in torch.nonzero(ok, as_tuple=True)[0].tolist()]
            if len(kept_indices) == 0:
                kept_indices = dense_idx
        else:
            kept_indices = dense_idx
            
        needed = num_classes - n_known_clusters
 
        if len(kept_indices) < max(needed, 1):
            if means_pca.size(0) > 0:
                print("not enough kept_indices")
                # use dense set if available, else all
                if len(dense_idx) > 0:
                    print("dense_idx > 0 -> take from center distribution")
                    pool_idx = torch.as_tensor(dense_idx, device=fallback_means_pca.device, dtype=torch.long)
                    D = torch.cdist(fallback_means_pca[pool_idx].float(), means_pca.float())
                else:
                    pool_idx = torch.arange(fallback_means_pca.size(0), device=fallback_means_pca.device)
                    D = torch.cdist(fallback_means_pca[pool_idx].float(), means_pca.float())
                min_d = D.min(dim=1).values  # smaller = more "central" to known means
                order_local = torch.argsort(min_d, descending=False)  # center-first
                order_global = pool_idx[order_local].tolist()
            else:
                # no known means: "center" = smallest mean distance to others
                D = torch.cdist(fallback_means_pca.float(), fallback_means_pca.float())
                D.fill_diagonal_(0)
                mean_d = D.mean(dim=1)
                order_global = torch.argsort(mean_d, descending=False).tolist()
        
            # top-up only as many as still needed, avoid duplicates
            still = max(needed, 1) - len(kept_indices)
            kept_set = set(kept_indices)
            topup = []
            for gi in order_global:
                if gi not in kept_set:
                    topup.append(int(gi))
                    kept_set.add(int(gi))
                if len(topup) >= still:
                    break
            kept_indices = kept_indices + topup
    
        # ---- Greedy farthest-first on PCA space ----
        def select_diverse_components_PCA(fallback_means_p, kept_idx, needed, modelist_means_p):
            device_local = fallback_means_p.device
            kept_idx_t = torch.as_tensor(kept_idx, device=device_local, dtype=torch.long)
            kept_p = fallback_means_p[kept_idx_t]  # (M, P)
            if needed <= 0:
                return []
            if kept_p.size(0) == 0:
                raise ValueError("No candidate means available to initialize unknown clusters.")
            if kept_p.size(0) == 1:
                return [kept_idx_t[0].item()]
    
            anchors = modelist_means_p.float() if modelist_means_p is not None else torch.empty(0, kept_p.size(1), device=device_local)
            rem_mask = torch.ones(kept_p.size(0), dtype=torch.bool, device=device_local)
            selected = []
    
            def pick_farthest_from_anchors():
                remaining = kept_p[rem_mask].float()
                if anchors.numel() > 0:
                    D = torch.cdist(remaining, anchors)
                    scores = D.min(dim=1).values
                else:
                    if remaining.size(0) == 1:
                        scores = torch.tensor([float('inf')], device=device_local)
                    else:
                        D_all = torch.cdist(remaining, remaining)
                        D_all.fill_diagonal_(0)
                        scores = D_all.mean(dim=1)
                best_local = torch.argmax(scores).item()
                rem_idx_all = torch.arange(kept_p.size(0), device=device_local)[rem_mask]
                return rem_idx_all[best_local].item()
    
            while len(selected) < needed and rem_mask.any():
                nxt = pick_farthest_from_anchors()
                selected.append(kept_idx_t[nxt].item())
                anchors = torch.cat([anchors, kept_p[nxt:nxt+1].float()], dim=0)
                rem_mask[nxt] = False
    
            if len(selected) < needed:
                extras = torch.arange(kept_p.size(0), device=device_local)[rem_mask].tolist()
                selected.extend([kept_idx_t[k].item() for k in extras[: needed - len(selected)]])
            return selected
    
        selected_indices = select_diverse_components_PCA(
            fallback_means_p=fallback_means_pca,
            kept_idx=kept_indices,
            needed=needed,
            modelist_means_p=means_pca
        )
        unknown_means = fallback_means[selected_indices]  # (K_unknown, d)  # keep original-space params for the GMM
    
        self.known_gmm_means = means.detach().cpu()
        self.unknown_gmm_means = unknown_means.detach().cpu()
    
        
        final_means = torch.cat([means, unknown_means], dim=0).detach().cpu().numpy().astype(np.float64)

        
        S = torch.zeros(latent_dim, latent_dim, device=device, dtype=torch.float64)
        denom = 0  
        for idx, cid in enumerate(unique_known):
            cz = known_z[known_labels == cid]
            n_i = cz.shape[0]
            if n_i >= 2:
                cz64 = cz.to(torch.float64)
                mu_i = means[idx].to(torch.float64)
                centered = cz64 - mu_i
                S += centered.T @ centered                    
                denom += (n_i - 1)
        
        use_pooled = denom >= max(latent_dim, 2)             
        
        if use_pooled:
            cov_tied = S / denom                             
        else:
            # fallback: global covariance with shrinkage (on CPU/np)
            z_np = z.detach().cpu().numpy().astype(np.float64)
            try:
                cov_tied = torch.from_numpy(LedoitWolf().fit(z_np).covariance_)
            except Exception:
                cov_tied = torch.from_numpy(np.cov(z_np, rowvar=False))
        
        cov_tied = cov_tied.to(torch.float64).to(device)
        
        cov_tied = 0.5 * (cov_tied + cov_tied.T)
        w, V = torch.linalg.eigh(cov_tied)                    # float64
        d = cov_tied.shape[0]
        trace = torch.trace(cov_tied)
        scale = (trace / max(d, 1)).item() if torch.isfinite(trace) else 1.0
        floor = max(1e-6 * scale, 1e-12)                     
        w_clipped = torch.clamp(w, min=floor)
        cov_spd = (V * w_clipped) @ V.T
        cov_spd = 0.5 * (cov_spd + cov_spd.T)
        
        final_cov_tied = cov_spd.detach().cpu().numpy().astype(np.float64)
            
        # --- Create sklearn GMM with tied covariance and overwrite params ---
        from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
        self.num_clusters = num_classes
        self.gmm = GaussianMixture(n_components=num_classes, covariance_type='tied', random_state=42)
        self.gmm.fit(np.random.randn(10, latent_dim))  
    
        self.gmm.means_ = np.asarray(final_means, dtype=np.float64) # (K, d)
        self.gmm.covariances_ = final_cov_tied       # (d, d)
        self.gmm.precisions_cholesky_ = _compute_precision_cholesky(self.gmm.covariances_, 'tied')


    def forward(self, G):
        """
        Runs the GCN encoder on the input graph.
        """
        x, edge_index = G.x, G.edge_index
        edge_weight = getattr(G, "weight", None)
        if edge_weight is None and getattr(G, "edge_attr", None) is not None:
            edge_weight = G.edge_attr

        for conv in self.layers[:-1]:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            x = F.dropout(x, p=0.3, training=self.training)

        z = self.layers[-1](x, edge_index, edge_weight=edge_weight)
        return z

    def fit(self, G, modelists, num_clusters, num_modelists_clusters, **kwargs):
        from .train import train
        out = train(
            self, G, modelists,
            num_clusters=num_clusters,
            num_modelists_clusters=num_modelists_clusters,
            **kwargs
        )
        return out 


    def predict(self, G: Data):
        """
        Runs the model in eval mode to compute node embeddings and returns 
        GMM-predicted cluster labels for all nodes.
        """
        self.eval()
        with torch.no_grad():
            embeddings = self(G)
            embeddings_np = embeddings.detach().cpu().numpy()

            cluster_labels = self.gmm.predict(embeddings_np)

            return cluster_labels


    def compute_gmm_posteriors(self, z):
        """
        Computes soft posterior responsibilities for each node under the current 
        tied-covariance GMM (in torch) using Gaussian log-likelihood + mixture weights.
        """
        assert hasattr(self, 'gmm'), "GMM not initialized."
    
        device = z.device
        dtype  = z.dtype
        N, D   = z.shape
    
        means_np = self.gmm.means_          # (K, D)
        cov_np   = self.gmm.covariances_    # (D, D)
        w_np     = self.gmm.weights_        # (K,)
    
        means = torch.as_tensor(means_np, device=device, dtype=dtype)     # (K, D)
        cov   = torch.as_tensor(cov_np,   device=device, dtype=dtype)     # (D, D)
        weights = torch.as_tensor(w_np,   device=device, dtype=dtype)     # (K,)
    
        prec        = torch.inverse(cov)          # (D, D)
        log_det_cov = torch.logdet(cov)          # scalar
    
        diff = z.unsqueeze(1) - means.unsqueeze(0)   # (N, 1, D) - (1, K, D) -> (N, K, D)
        maha = torch.einsum("nkd,df,nkf->nk", diff, prec, diff)  # (N, K)
    
        const = D * math.log(2 * math.pi)
        log_likelihoods = -0.5 * (maha + log_det_cov + const)   # (N, K)
    
        log_prior = torch.log(weights + 1e-10)                  # (K,)
        log_joint = log_likelihoods + log_prior.unsqueeze(0)    # (N, K)
    
        log_post = log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)  # (N, K)
        responsibilities = torch.exp(log_post)                                  # (N, K)
    
        return responsibilities


    def loss_gmm(self, embeddings, epoch=None, update_freq=10, threshold=20):
        """
        Computes a “pull” loss that encourages embeddings to be close (cosine-wise) 
        to their assigned GMM component mean in normalized space.
        """
        mu_k = torch.tensor(self.gmm.means_, dtype=torch.float32, device=embeddings.device)
        z_norm = F.normalize(embeddings, dim=1)
        mu_norm = F.normalize(mu_k, dim=1)

        similarity = torch.mm(z_norm, mu_norm.t())  # (n_nodes, n_clusters)
        assigned_clusters = similarity.argmax(dim=1)  # (n_nodes,)    
        target_means = mu_norm[assigned_clusters]  # (n_nodes, latent_dim)

        cos_pos = F.cosine_similarity(z_norm, target_means, dim=1)  # in [-1,1]
        gmm_pull_loss = (1 - cos_pos).mean()
        
        return gmm_pull_loss
    
    def label_propagation(self, G, labels, num_classes, alpha=0.6, max_iter=1):
        """
        Performs iterative label propagation using a dense adjacency matrix 
        (with added self-loops) to diffuse one-hot labels across the graph.
        """
        y = F.one_hot(labels.clamp(min=0), num_classes=num_classes).float()  # One-hot only for labeled
        y[labels == -1] = 0.0  # Unknown = 0 vector
    
        y_orig = y.clone()
        adj = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes).squeeze(0)
        adj = adj + torch.eye(adj.size(0), device=adj.device)  # NEW
        degree = adj.sum(dim=1)
        degree[degree == 0] = 1  # Prevent divide by zero
        D_inv = torch.diag(1 / degree)
        L = torch.mm(D_inv, adj)
        
        for _ in range(max_iter):
            y = alpha * torch.mm(L, y) + (1 - alpha) * y_orig
            
        i, j = G.edge_index
   
        return y  
    
    
    def label_propagation_weights(self, G, labels, num_classes,
                              alpha=0.8, max_iter=5,
                              add_self_loops=True, self_loop_weight=None,
                              scale='max',   
                              clamp_seeds=True,
                              renorm_rows=True):
        """
        Performs weighted label propagation using graph edge weights (with self-loops), 
        with scaling, row renormalization, and clamping of seed labels.
        """
        N = G.num_nodes
        device = G.edge_index.device
    
        y = F.one_hot(labels.clamp(min=0), num_classes=num_classes).float().to(device)

        uniform_prior = torch.full((N, num_classes), 1.0 / num_classes, device=device)
        y[labels == -1] = uniform_prior[labels == -1]

        y_orig = y.clone()
        seed_mask = (labels != -1)

        i, j = G.edge_index
        W = torch.zeros((N, N), dtype=torch.float32, device=device)
        if getattr(G, "edge_weight", None) is not None:
            W[i, j] = G.edge_weight.to(device)
        else:
            W[i, j] = 1.0
    
        if add_self_loops:
            if self_loop_weight is None:
                self_loop_weight = 1
            W = W.clone()
            W = W + torch.eye(N, device=device) * self_loop_weight
    
        if scale == 'max':
            denom = W.abs().max().clamp_min(1e-12)
            W = W / denom
        elif isinstance(scale, (float, int)):
            W = W / float(scale)

        L = W

        for _ in range(max_iter):
            y = alpha * (L @ y) + (1 - alpha) * y_orig
            if renorm_rows:
                y = y / (y.sum(dim=1, keepdim=True) + 1e-6)
            if clamp_seeds:
                y[seed_mask] = y_orig[seed_mask]
        return y, L
       

    def loss_propagation(self, logits, soft_labels, mask):
        """
        Computes cross-entropy loss between predicted logits and propagated soft labels.
        """
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            logits      = logits[mask]
            soft_labels = soft_labels[mask]
    
        if logits.numel() == 0:
            return logits.new_tensor(0.0, requires_grad=True)
    
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_labels * log_probs).sum(dim=1).mean()
        return loss
    
    def loss_feature_modelist_contrastive(self, z, raw_features, labels, 
                                          modelist_mask, edge_index, feature_sim_thresh=0.8, 
                                          temperature=0.5):
        """
        Builds positive/negative pairs using feature-space similarity (PCA+cosine) 
        and modelist label agreement, then applies a BCE contrastive objective on embedding similarity.
        """
        device = z.device
    
        with torch.no_grad():
            d_x = raw_features.shape[1]
            pca_dim_x = min(16, d_x)
            x_np = raw_features.detach().cpu().numpy()
    
            pca_x = PCA(n_components=pca_dim_x, svd_solver="auto", random_state=42)
            x_pca_np = pca_x.fit_transform(x_np)
            x_pca = torch.from_numpy(x_pca_np).to(device=device, dtype=torch.float32)
            x_pca = F.normalize(x_pca, dim=1)
    
            # Pairwise cosine similarities in PCA(feature) space
            sim_x = torch.mm(x_pca, x_pca.t())  # (N, N)
            N = sim_x.shape[0]
            iu = torch.triu_indices(N, N, offset=1)
            vals = sim_x[iu[0], iu[1]].detach().cpu()
    
            # Thresholds
            if isinstance(feature_sim_thresh, (tuple, list)) and len(feature_sim_thresh) == 2:
                neg_thresh = float(feature_sim_thresh[0])
                pos_thresh = float(feature_sim_thresh[1])
            else:
                pos_thresh = float(feature_sim_thresh)
                # data-adaptive neg threshold as a reasonable default (Q25)
                neg_thresh = float(torch.quantile(vals, 0.25).item())
    
            # masks from feature similarity
            pos_mask_features = (sim_x >= pos_thresh)
            neg_mask_features = (sim_x <= neg_thresh)
    
            # label-based positives (modelists)
            same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))
            known_mask = modelist_mask.unsqueeze(0) & modelist_mask.unsqueeze(1)
            not_unknown = (labels != -1).unsqueeze(0)
            pos_mask_labels = same_label & known_mask & not_unknown
    
            pos_mask = (pos_mask_features | pos_mask_labels)
            neg_mask = neg_mask_features & (~pos_mask)  # if it's positive by label, keep it positive
    

        N, d_z = z.shape
        pca_dim_z = min(2, d_z)
    
        z_mean = z.mean(dim=0, keepdim=True)               # (1, d_z)
        z_centered = z - z_mean                             # (N, d_z)
    
        U, S, Vh = torch.linalg.svd(z_centered, full_matrices=False)  # Vh: (d_z, d_z)
        V = Vh.transpose(0, 1)                                         # (d_z, d_z)
    
        W = V[:, :pca_dim_z]                                           # (d_z, pca_dim_z)
    
        z_proj = z_centered @ W                                        # (N, pca_dim_z)
        z_proj = F.normalize(z_proj, dim=1)
    
        sim_z = torch.mm(z_proj, z_proj.t()) / temperature             # (N, N)

        with torch.no_grad():
            target = torch.zeros_like(sim_z)
            target[pos_mask] = 1.0
    
            valid = pos_mask | neg_mask
    
            eye = torch.eye(sim_z.size(0), device=sim_z.device, dtype=torch.bool)
            valid = valid & (~eye)
    
        if valid.sum() == 0:
            return sim_z.new_tensor(0.0, requires_grad=True)
    
        loss = F.binary_cross_entropy_with_logits(sim_z[valid], target[valid])
    
        return loss



        