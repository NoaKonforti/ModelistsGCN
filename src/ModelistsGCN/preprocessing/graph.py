import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from scipy import sparse
import pandas as pd
from typing import Any, Dict, Iterable, List, Tuple, Optional
from torch_geometric.data import Data


def harmonize_and_intersect(
    weighted_adj_matrix: pd.DataFrame,
    features_df: pd.DataFrame,
    id_col: str="cellID"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns multiple DataFrames by keeping only shared cellIDs 
    and ensuring consistent ordering.
    """
    # Set index to CellID for features
    if id_col in features_df.columns:
        features_df = features_df.set_index(id_col)

    # Make sure both sides use the SAME dtype (float is OK here)
    features_df.index = features_df.index.astype(str)#(float)

    adj = weighted_adj_matrix.copy()
    adj.index = adj.index.astype(str)#(float)
    adj.columns = adj.columns.astype(str)#(float)

    # Intersect IDs
    common_ids = features_df.index.intersection(adj.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping cellIDs between features and adjacency!")

    # Subset both
    features_df = features_df.loc[common_ids]
    adj = adj.loc[common_ids, common_ids]

    # Final sanity
    assert features_df.index.equals(adj.index), "Index mismatch after subsetting!"
    return adj, features_df


# =====================================
# Build NetworkX Graph
# =====================================
# def build_networkx_graph(weighted_adj_matrix, node_features_df):
#     graph = nx.Graph()

#     # Add nodes with features
#     for node_id, features in node_features_df.iterrows():
#         graph.add_node(node_id, **features.to_dict())

#     # Add weighted edges
#     for u in range(len(weighted_adj_matrix)):
#         for v in range(len(weighted_adj_matrix)):
#             if weighted_adj_matrix.iloc[u,v] > 0 :
#                 graph.add_edge(weighted_adj_matrix.index[u], weighted_adj_matrix.index[v], weight=weighted_adj_matrix.iloc[u,v])#weight=weighted_adj_matrix[u,v])

#     return graph

def build_networkx_graph(
        weighted_adj_matrix: pd.DataFrame, 
        node_features_df: pd.DataFrame,
        undirected: bool=True, 
        drop_self_loops: bool=True
)-> nx.Graph:
    """
    Constructs a weighted NetworkX graph from an adjacency matrix, 
    adding node features and efficiently creating edges only for nonzero connections.
    """
    A = weighted_adj_matrix
    nodes = A.index.to_numpy()

    G = nx.Graph() if undirected else nx.DiGraph()
    G.add_nodes_from((n, attrs) for n, attrs in node_features_df.to_dict(orient="index").items())

    # Convert to sparse (CSR) then iterate only non-zeros
    csr = sparse.csr_matrix(A.to_numpy(copy=False))

    if drop_self_loops:
        csr.setdiag(0)
        csr.eliminate_zeros()

    if undirected:
        csr = sparse.triu(csr, k=1)

    r, c = csr.nonzero()
    w = csr.data  # aligns with nonzero entries for CSR/COO conversions, but safest as COO:
    coo = csr.tocoo()
    edges = [(nodes[i], nodes[j], float(wk)) for i, j, wk in zip(coo.row, coo.col, coo.data)]
    G.add_weighted_edges_from(edges, weight="weight")

    return G

def convert_to_pyg_graph(
        nx_graph: nx.Graph, 
        node_features_df: pd.DataFrame
)-> Data:

    """
    Converts a NetworkX graph into a PyTorch Geometric graph object, 
    attaching node features, edge weights, and cellID metadata.
    """
    G = from_networkx(nx_graph)
    G.x = torch.tensor(node_features_df.values, dtype=torch.float)
    G.edge_attr = torch.tensor([d['weight'] for _, _, d in nx_graph.edges(data=True)], dtype=torch.float)
    cell_id_list = list(nx_graph.nodes) 
    #G.cell_ids = torch.tensor(cell_id_list)
    #G.cell_ids = list(cell_id_list)
    G.cell_ids = cell_id_list

    return G

# =====================================
# Feature Refinement for G.x
# =====================================

def refine_features(
        G: Data,
        MORPHOLOGY_FEATURES: pd.DataFrame,
        DROP_FEATURES: Optional[List[str]]
)-> Data:
    """
    Filters and reorders the final feature matrix before graph construction.
    """
    all_features = list(G.keys())

    # things that are NOT features:
    PROTECTED_ATTRS = {"y", "y_mapped", "cell_ids"}

    remaining_features = [
        feat for feat in all_features
        if feat not in MORPHOLOGY_FEATURES
        and feat not in DROP_FEATURES
        and feat not in PROTECTED_ATTRS
    ]

    feature_names = MORPHOLOGY_FEATURES + sorted(remaining_features)

    features = torch.stack([getattr(G, name) for name in feature_names], dim=1)
    G.x = features

    # Remove only feature attrs, keep y / cell_ids
    for name in feature_names:
        delattr(G, name)

    return G

# =====================================
# Adjust Graph Object
# =====================================
def adjust_G(G: Data)-> Data:
    """
    Ensures edge attributes match the bidirectional edge index representation 
    by duplicating weights when needed.
    """
    if G.edge_index.shape[1] == 2 * G.edge_attr.shape[0]:
        G.edge_attr = torch.cat([G.edge_attr, G.edge_attr], dim=0)
        print("Edge attributes duplicated for bidirectional edges.")
    else:
        print("Edge attributes match bidirectional edge_index.")
    print(f"# Edges: {G.edge_index.shape[1]}, Edge Attrs: {G.edge_attr.shape[0]}")
    return G

# =====================================
# Modelists to graph
# =====================================
def _id_to_key(x: Any) -> str:
    """
    Normalizes cell identifiers into a consistent string format to enable reliable 
    matching between graph nodes and external data sources.
    """
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            # If cell_id is stored weirdly (e.g., vector), fall back to repr
            return repr(x.detach().cpu().tolist())
    # Normalize strings: strip whitespace
    if isinstance(x, str):
        return x.strip()
    return str(x)


def build_cluster_labels_tensor(
    G,
    modelist_dict: Dict[str, Iterable[Any]],
    *,
    missing_value: int = -1,
    cluster_order: Optional[List[str]] = None,
    on_overlap: str = "first",  # "first" | "last" | "error"
) -> Tuple[torch.LongTensor, Dict[str, int], Dict[str, List[Any]], Dict[str, List[str]]]:
    """
    Creates a tensor of cluster labels for graph nodes based on modelist cell assignments, 
    handling missing IDs and resolving overlaps.
    """

    if not hasattr(G, "cell_ids"):
        raise AttributeError("G must have attribute `cell_ids` aligned with nodes.")
    if not hasattr(G, "num_nodes") or G.num_nodes is None:
        raise AttributeError("G must have `num_nodes` set.")

    # 1) Node lookup: cell_id_key -> node_index
    graph_keys = [_id_to_key(cid) for cid in G.cell_ids]
    lookup = {k: i for i, k in enumerate(graph_keys)}  # if duplicates exist, last wins

    # 2) Decide cluster ordering and ids
    names = cluster_order if cluster_order is not None else list(modelist_dict.keys())
    cluster_to_id = {name: idx for idx, name in enumerate(names)}

    # 3) Build labels
    labels = torch.full((int(G.num_nodes),), int(missing_value), dtype=torch.long)

    missing: Dict[str, List[Any]] = {}
    seen_by_key: Dict[str, str] = {}        # cell_key -> cluster_name assigned
    overlaps: Dict[str, List[str]] = {}     # cell_key -> [cluster_names...]

    for cname in names:
        cid_list = modelist_dict.get(cname, [])
        cid_id = cluster_to_id[cname]

        for cid in cid_list:
            k = _id_to_key(cid)
            node_idx = lookup.get(k)

            if node_idx is None:
                missing.setdefault(cname, []).append(cid)
                continue

            prev = seen_by_key.get(k)
            if prev is not None and prev != cname:
                overlaps.setdefault(k, [prev])
                if overlaps[k][-1] != cname:
                    overlaps[k].append(cname)

                if on_overlap == "error":
                    raise ValueError(f"Cell id {k} appears in multiple clusters: {overlaps[k]}")
                if on_overlap == "first":
                    continue
                # on_overlap == "last" -> overwrite

            labels[node_idx] = cid_id
            seen_by_key[k] = cname

    return labels, cluster_to_id, missing, overlaps