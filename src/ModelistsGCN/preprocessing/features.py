from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple
from skimage import measure
from skimage.measure import regionprops
import numpy as np
import pandas as pd
from scipy.spatial import distance
from skimage.measure import perimeter
from skimage.morphology import convex_hull_image
from scipy import ndimage as ndi

# ===================================
# Compute morphological features
# ===================================

def compute_morpho_features(
    matrix: np.ndarray,
    labels: Optional[Iterable[int]] = None,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-cell morphological features from a labeled segmentation.

    Supports:
    - 3D segmentations with shape (z, y, x)
    - 2D segmentations with shape (y, x)
    - 2D segmentations stored as shape (1, y, x)

    Notes:
    - Label 0 is treated as background
    - For 3D, the original logic is preserved
    - For 2D, 3D-only features are replaced with 2D-appropriate shape descriptors
    """

    if matrix.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D matrix, got shape={matrix.shape}")

    is_2d = (matrix.ndim == 2) or (matrix.ndim == 3 and matrix.shape[0] == 1)

    if is_2d:
        return _compute_morpho_features_2d(matrix, labels, voxel_size)
    else:
        return _compute_morpho_features_3d(matrix, labels, voxel_size)



def _compute_morpho_features_3d(
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

def _compute_morpho_features_2d(
    matrix: np.ndarray,
    labels: Optional[Iterable[int]] = None,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[int, Dict[str, Any]]:
    """
    2D implementation with voxel-size-aware shape features.

    For 2D:
    - area is scaled by vy * vx
    - perimeter is scaled using spacing=(vy, vx)
    - centroid is returned in physical coordinates: (centroid_y, centroid_x)
    """
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise ValueError(f"Expected 2D or single-slice matrix, got shape={matrix.shape}")
        matrix = matrix[0]

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix.shape}")

    #_, vy, vx = map(float, voxel_size)
    if len(voxel_size) == 2:
        vy, vx = map(float, voxel_size)
    elif len(voxel_size) == 3:
        _, vy, vx = map(float, voxel_size)
    else:
        raise ValueError(f"voxel_size must have length 2 or 3, got {voxel_size}")

    pixel_area = vy * vx

    matrix = np.asarray(matrix)
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

    y_idx, x_idx = np.nonzero(matrix)
    lab = matrix[y_idx, x_idx].astype(np.int32)

    size = max_label + 1
    counts = np.bincount(lab, minlength=size).astype(np.float64)

    sum_y = np.bincount(lab, weights=y_idx.astype(np.float64), minlength=size)
    sum_x = np.bincount(lab, weights=x_idx.astype(np.float64), minlength=size)

    with np.errstate(divide="ignore", invalid="ignore"):
        cy = sum_y / counts
        cx = sum_x / counts

    obj_slices = ndi.find_objects(matrix)
    results: Dict[int, Dict[str, Any]] = {}

    for L in labels_arr.tolist():
        if L <= 0 or L >= size or counts[L] <= 0:
            continue

        slc = obj_slices[L - 1]
        if slc is None:
            continue

        sub = matrix[slc]
        mask = sub == L
        if not np.any(mask):
            continue

        area_val = float(counts[L] * pixel_area)
        centroid = (
            float(cy[L] * vy),
            float(cx[L] * vx),
        )
        def _physical_perimeter_2d(mask: np.ndarray, vy: float, vx: float) -> float:
            padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
        
            # boundaries between rows -> horizontal edge length = vx
            y_edges = np.diff(padded, axis=0) != 0
        
            # boundaries between columns -> vertical edge length = vy
            x_edges = np.diff(padded, axis=1) != 0
        
            return float(y_edges.sum() * vx + x_edges.sum() * vy)
        
        #perim = float(perimeter(mask, spacing=(vy, vx)))
        perim = _physical_perimeter_2d(mask, vy, vx)

        convex_mask = convex_hull_image(mask)
        #convex_perim = float(perimeter(convex_mask, spacing=(vy, vx)))
        convex_perim = _physical_perimeter_2d(convex_mask, vy, vx)

        circularity = 0.0
        if perim > 0.0:
            circularity = float(4.0 * np.pi * area_val / (perim ** 2))

        roughness = 0.0
        if convex_perim > 0.0:
            roughness = float(perim / convex_perim)

        props = regionprops(mask.astype(np.uint8), spacing=(vy, vx))
        if props:
            major = float(getattr(props[0], "major_axis_length", 0.0) or 0.0)
            minor = float(getattr(props[0], "minor_axis_length", 0.0) or 0.0)
            elong = 0.0 if major == 0.0 else float(1.0 - minor / major)
        else:
            elong = 0.0

        results[int(L)] = {
            "area": area_val,
            "perimeter": perim,
            "centroid": centroid,
            "circularity": circularity,
            "roughness": roughness,
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
        centroid_lengths = df["centroid"].dropna().apply(len).unique()

        if len(centroid_lengths) != 1:
            raise ValueError(
                f"Inconsistent centroid dimensions found: {centroid_lengths}"
            )

        centroid_dim = centroid_lengths[0]

        if centroid_dim == 3:
            centroid_cols = ["centroid_z", "centroid_y", "centroid_x"]
        elif centroid_dim == 2:
            centroid_cols = ["centroid_y", "centroid_x"]
        else:
            raise ValueError(
                f"Centroid must have length 2 or 3, got length {centroid_dim}"
            )

        centroid_df = pd.DataFrame(
            df["centroid"].tolist(),
            index=df.index,
            columns=centroid_cols,
        )

        df = df.drop(columns=["centroid"]).join(centroid_df)
        
    # ensure numeric
    for c in centroid_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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

    if {"centroid_z", "centroid_y", "centroid_x"}.issubset(centroids_df.columns):
        coord_cols = ["centroid_z", "centroid_y", "centroid_x"]
    elif {"centroid_y", "centroid_x"}.issubset(centroids_df.columns):
        coord_cols = ["centroid_y", "centroid_x"]
    else:
        raise ValueError(
            "centroids_df must contain either "
            "['centroid_z', 'centroid_y', 'centroid_x'] or "
            "['centroid_y', 'centroid_x']"
        )

    coords = centroids_df[coord_cols].to_numpy()

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
