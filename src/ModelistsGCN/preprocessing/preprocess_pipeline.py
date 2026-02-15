from __future__ import annotations

from typing import Any, Dict, Tuple
import torch
from torch_geometric.data import Data
from . import io_utils
from . import features
from . import modelists
from . import graph


def preprocessing(cfg: Dict[str, Any]) -> Tuple[Data, torch.LongTensor]:
    print("=== Preprocessing ===")
    expr = io_utils.load_expr(cfg["expression_csv"])
    markers = io_utils.load_markers(cfg["markers_csv"])
    
    # ===================================
    # Features preprocessing
    # ===================================
    if cfg.get("morpho_features_csv") is not None:

        if cfg.get("centroids_csv") is not None:
           raise ValueError(
               "If you provide 'morpho_features_csv', you must also provide 'centroids_csv' "
               "because centroids are not computed from segmentation in this mode."
           )

        print("Using user-provided morphology + centroids CSVs...")
        morpho_df = io_utils.load_morpho_df(cfg["morpho_features_csv"])
        centroids_df = io_utils.load_centroids_df(cfg["centroids_csv"])
    
        # ensure same cells
        common = morpho_df.index.intersection(centroids_df.index)
        if len(common) == 0:
            raise ValueError("No overlapping cellIDs between morpho_features_csv and centroids_csv")
        morpho_df = morpho_df.loc[common]
        centroids_df = centroids_df.loc[common]
    else:
        if "segmentation_npy" not in cfg:
           raise ValueError(
               "In order to calculate morphological features, you must also provide 'segmentation_npy' "
           )
        print("Computing morphology features from segmentation...")
        seg  = io_utils.load_seg(cfg["segmentation_npy"])
        morpho_features = features.compute_morpho_fetures(seg, voxel_size=cfg["features"].get("voxel_size",(1.0,1.0,1.0)))
        centroids_df, morpho_df = features.split_morpho_centroids(morpho_features)
    
    expr_norm = features.scale_features(expr)
    genes_to_drop = features.get_least_variable_genes(expr_norm)
    expr = features.drop_genes(expr, genes_to_drop)
    featurs_df = features.merge_morpho_expression(morpho_df, expr)
    featurs_df = features.scale_features(featurs_df)
    
    proximity_df = features.compute_centroid_proximity_distances(centroids_df)
    adj_matrix, weighted_adj_matrix = features.compute_proximity_adjacency(proximity_df, factor_exp=cfg["features"].get("factor_exp",1), cell_thresh=cfg["features"].get("cell_thresh",20))
    
    # ===================================
    # Modelists cell selection
    # ===================================
    
    modelist_dict, cfg2 = modelists.run_modelists_from_cfg(
        expr=expr,
        marker_df=markers,
        cfg=cfg,
        max_tries=6,
        verbose=True
    )
    print("Final modelists cfg:", cfg2["modelists"])
    
    # ===================================
    # Creat graph
    # ===================================
    adj_int, feat_int = graph.harmonize_and_intersect(weighted_adj_matrix, featurs_df)
    nx_graph = graph.build_networkx_graph(adj_int, feat_int)
    G = graph.convert_to_pyg_graph(nx_graph, feat_int)
    G = graph.refine_features(G, morpho_df.columns.tolist(), cfg["graph"].get("DROP_FEATURES", ['edge_index', 'edge_attr', 'weight', 'num_nodes', 'cell_ids', 'x']))
    if getattr(G, "weight", None) is None:
        G = graph.adjust_G(G)
    else:
        G.x = G.x.float()
        G.weight = G.weight.float()

    labels, _, _, _ = graph.build_cluster_labels_tensor(G, modelist_dict)
    
    return G, labels
