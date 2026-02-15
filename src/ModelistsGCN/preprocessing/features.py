from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple
from skimage import measure
from skimage.measure import regionprops
import numpy as np
import pandas as pd
from scipy.spatial import distance


# ===================================
# Compute morphological features
# ===================================
def compute_morpho_fetures(
    matrix: np.ndarray,
    labels: Optional[Iterable[int]] = None,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[int, Dict[str, Any]]:
    """
    Computes per-cell 3D morphological properties directly from
    a labeled segmentation using voxel statistics
    and marching-cubes surface reconstruction.
    """
    if matrix.ndim != 3:
        raise ValueError(f"Expected 3D matrix, got shape={matrix.shape}")

    # IMPORTANT: spacing order must match matrix axis order (z, y, x)
    vz, vy, vx = map(float, voxel_size)
    voxel_volume = vz * vy * vx

    max_label = int(matrix.max())
    if max_label <= 0:
        return {}

    if labels is None:
        labels_arr = np.unique(matrix)
        labels_arr = labels_arr[labels_arr > 0].astype(np.int32)
    else:
        labels_arr = np.asarray([int(x) for x in labels], dtype=np.int32)
        labels_arr = labels_arr[labels_arr > 0]

    if labels_arr.size == 0:
        return {}

    z_idx, y_idx, x_idx = np.nonzero(matrix)
    lab = matrix[z_idx, y_idx, x_idx].astype(np.int32)

    size = max_label + 1

    counts = np.bincount(lab, minlength=size).astype(np.float64)
    area = counts * voxel_volume

    sum_z = np.bincount(lab, weights=z_idx.astype(np.float64), minlength=size)
    sum_y = np.bincount(lab, weights=y_idx.astype(np.float64), minlength=size)
    sum_x = np.bincount(lab, weights=x_idx.astype(np.float64), minlength=size)

    with np.errstate(divide="ignore", invalid="ignore"):
        cz = sum_z / counts
        cy = sum_y / counts
        cx = sum_x / counts

    # bbox (index-space)
    inf = np.iinfo(np.int32).max
    min_z = np.full(size, inf, dtype=np.int32)
    min_y = np.full(size, inf, dtype=np.int32)
    min_x = np.full(size, inf, dtype=np.int32)
    max_z = np.full(size, -1, dtype=np.int32)
    max_y = np.full(size, -1, dtype=np.int32)
    max_x = np.full(size, -1, dtype=np.int32)

    z_i32 = z_idx.astype(np.int32)
    y_i32 = y_idx.astype(np.int32)
    x_i32 = x_idx.astype(np.int32)

    np.minimum.at(min_z, lab, z_i32)
    np.minimum.at(min_y, lab, y_i32)
    np.minimum.at(min_x, lab, x_i32)
    np.maximum.at(max_z, lab, z_i32)
    np.maximum.at(max_y, lab, y_i32)
    np.maximum.at(max_x, lab, x_i32)

    def _marching_cubes_surface_area(mask: np.ndarray) -> float:

        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(np.float32),
                level=0.5,
                spacing=(vz, vy, vx),
            )
        except Exception:
            return 0.0

        triangles = verts[faces]
        vec1 = triangles[:, 1] - triangles[:, 0]
        vec2 = triangles[:, 2] - triangles[:, 0]
        cross_prod = np.cross(vec1, vec2)
        return float(0.5 * np.sum(np.linalg.norm(cross_prod, axis=1)))

    def _elongation_regionprops(mask: np.ndarray) -> float:

        props = regionprops(mask.astype(np.uint8))
        if not props:
            return 0.0
        major = float(getattr(props[0], "major_axis_length", 0.0) or 0.0)
        try:
            minor = float(getattr(props[0], "minor_axis_length", 0.0) or 0.0)
        except ValueError:
            return 0.0
        return 0.0 if major == 0.0 else float(1.0 - minor / major)

    def _sphericity(volume: float, surface_area_: float) -> float:
        if surface_area_ <= 0.0:
            return 0.0
        return float((np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / surface_area_)

    def _roundness(volume: float, surface_area_: float, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> float:
        if surface_area_ <= 0.0:
            return 0.0

        dz = float(z1 - z0) * vz
        dy = float(y1 - y0) * vy
        dx = float(x1 - x0) * vx
        prod = dz * dy * dx
        if prod <= 0.0:
            return 0.0
        return float(volume / (surface_area_ * (prod ** (1.0 / 3.0))))

    results: Dict[int, Dict[str, Any]] = {}

    Z, Y, X = matrix.shape

    for L in labels_arr.tolist():
        if L <= 0 or L >= size or counts[L] <= 0:
            continue

        z0, z1 = int(min_z[L]), int(max_z[L])
        y0, y1 = int(min_y[L]), int(max_y[L])
        x0, x1 = int(min_x[L]), int(max_x[L])

        # Crop to tight bbox
        sub = matrix[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        sub_mask = (sub == L)

        # Surface area needs background around object if it touches bbox edge: pad by 1 voxel of zeros
        sub_mask_pad = np.pad(sub_mask, 1, mode="constant", constant_values=False)

        vol = float(area[L])  # your code calls this "area" but uses it as volume in formulas
        sa = _marching_cubes_surface_area(sub_mask_pad)
        centroid = (float(cz[L]), float(cy[L]), float(cx[L]))

        sph = _sphericity(vol, sa)
        rnd = _roundness(vol, sa, z0, z1, y0, y1, x0, x1)

        # regionprops on the unpadded cropped mask (translation doesn't matter)
        elong = _elongation_regionprops(sub_mask)

        results[int(L)] = {
            "area": vol,
            "surface_area": sa,
            "centroid": centroid,
            "sphericity": sph,
            "roundness": rnd,
            "elongation_ratio": elong,
        }
        
    return results

# ===================================
# Split morpho fetures and centroid coords
# ===================================
def split_morpho_centroids(
    features: Dict[int, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates centroid coordinates from other morphological features 
    and converts them into two aligned DataFrames indexed by cellID.
    """
    df = pd.DataFrame.from_dict(features, orient="index")
    df.index.name = "cellID"

    # make index consistent with the rest of your pipeline (string IDs) :contentReference[oaicite:1]{index=1}
    df = df.reset_index()
    df["cellID"] = df["cellID"].astype(str)
    df = df.set_index("cellID")

    # expand centroid tuple -> centroid_z/y/x (same logic you already had) :contentReference[oaicite:2]{index=2}
    if "centroid" in df.columns:
        centroid_df = pd.DataFrame(
            df["centroid"].tolist(),
            index=df.index,
            columns=["centroid_z", "centroid_y", "centroid_x"],
        )
        df = df.drop(columns=["centroid"]).join(centroid_df)

    # ensure numeric
    for c in ["centroid_z", "centroid_y", "centroid_x"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    centroid_cols = [c for c in df.columns if c.lower().startswith("centroid_")]
    centroids_df = df[centroid_cols].copy()
    morpho_df = df.drop(columns=centroid_cols).copy()

    return centroids_df, morpho_df

# ===================================
# Gene filtering utilities
# ===================================
def get_least_variable_genes(
    norm_expr: pd.DataFrame,
    drop_percent: float = 0.05,
) -> list[str]:
    """
    Identifies genes with low variability across cells to 
    remove uninformative expression features.
    """
    gene_means = norm_expr.mean(axis=0)
    gene_vars = norm_expr.var(axis=0)
    ratio = gene_vars / gene_means
    least_variable = ratio.nsmallest(int(drop_percent * len(ratio)))

    return least_variable.index.tolist()

def drop_genes(expr_df: pd.DataFrame, genes_to_drop: Iterable[str]) -> pd.DataFrame:
    """
    Removes specified genes from the expression matrix.
    """
    return expr_df.drop(columns=genes_to_drop)

# ===================================
# Merge + scale utilities
# ===================================
def merge_morpho_expression(
    morpho_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    *,
    index_name: str = "cellID",
) -> pd.DataFrame:
    """
    Concatenates morphological features with gene expression data by
    matching cells using their shared cellID index.
    """
    if morpho_df.index.name != index_name:
        morpho_df = morpho_df.copy()
        morpho_df.index.name = index_name
    if expr_df.index.name != index_name:
        expr_df = expr_df.copy()
        expr_df.index.name = index_name

    merged = morpho_df.join(expr_df, how="inner")
    return merged

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes feature values across cells using z-score normalization.
    """
    return (df - df.mean()) / df.std(ddof=0)

# =====================================
# Proximity adjacency utilities
# =====================================
def compute_centroid_proximity_distances(centroids_df: pd.DataFrame)->pd.DataFrame:
    """
    Computes pairwise Euclidean distances between cell centroids 
    to quantify spatial proximity.
    """
    # Extract labels and coordinates
    labels = centroids_df.index.to_numpy()
    coords = centroids_df[["centroid_z", "centroid_y", "centroid_x"]].to_numpy()

    # Compute full pairwise distance matrix
    distance_matrix = distance.cdist(coords, coords, metric='euclidean')
    proximity_df = pd.DataFrame(
        distance_matrix,
        index=labels,
        columns=labels
    )
    proximity_df.index.name = "cellID"
    proximity_df.columns.name = "cellID"
    return proximity_df

def compute_proximity_adjacency(
    proximity_df: pd.DataFrame,
    *,
    cell_thresh: float=20,
    factor_exp=1
)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds adjacency and weighted adjacency matrices by connecting nearby cells 
    and assigning weights based on inverse distance.
    """
    threshold = cell_thresh * factor_exp

    adj_matrix = (proximity_df <= threshold).astype(float)
    weights = np.where(proximity_df > 0, 1 / proximity_df, 0)
    weighted_adj = adj_matrix * weights

    return adj_matrix, pd.DataFrame(weighted_adj, index=proximity_df.index, columns=proximity_df.columns)
