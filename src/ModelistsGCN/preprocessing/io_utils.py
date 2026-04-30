import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

# ===================================
# Loads
# ===================================
def load_expr(expr_path:str)-> pd.DataFrame:
    expr = pd.read_csv(expr_path, index_col=0)
    expr.index = expr.index.astype(str)
    expr.index.name = "cellID"
    return expr

def load_seg(seg_path:str)-> np.ndarray:
    seg = np.load(seg_path)
    return seg

def load_markers(markers_path:str)-> pd.DataFrame:
    markers = pd.read_csv(markers_path)
    return markers

def load_morpho_df(path: str) -> pd.DataFrame:
    morpho = pd.read_csv(path)
    if "cellID" not in morpho.columns:
        raise ValueError(f"morpho_features_csv must contain 'cellID'. Got: {list(morpho.columns)}")
    morpho["cellID"] = morpho["cellID"].astype(str)
    morpho = morpho.set_index("cellID")
    morpho.index.name = "cellID"
    return morpho

def load_centroids_df(path: str) -> pd.DataFrame:
    centroids = pd.read_csv(path)

    cols = set(centroids.columns)

    has_3d = {"cellID", "centroid_z", "centroid_y", "centroid_x"}.issubset(cols)
    has_2d = {"cellID", "centroid_y", "centroid_x"}.issubset(cols)

    if not (has_3d or has_2d):
        raise ValueError(
            "centroids_csv must contain either:\n"
            "['cellID', 'centroid_z', 'centroid_y', 'centroid_x'] (3D)\n"
            "or\n"
            "['cellID', 'centroid_y', 'centroid_x'] (2D)\n"
            f"Got: {list(centroids.columns)}"
        )
    centroids["cellID"] = centroids["cellID"].astype(str)
    centroids = centroids.set_index("cellID")
    centroids.index.name = "cellID"


    # determine which columns to keep
    if has_3d:
        coord_cols = ["centroid_z", "centroid_y", "centroid_x"]
    else:
        coord_cols = ["centroid_y", "centroid_x"]

    # convert to numeric safely
    for c in coord_cols:
        centroids[c] = pd.to_numeric(centroids[c], errors="coerce")

    centroids = centroids[coord_cols]
    return centroids