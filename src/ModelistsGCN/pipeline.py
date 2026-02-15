from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pandas as pd

from .preprocessing.preprocess_pipeline import preprocessing  
from .model.ModelistsGCN import ModelistsGCN
from .model.helpers import set_seed

def run(cfg: Dict[str, Any]):
    """
    End-to-end ModelistsGCN pipeline.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing data paths + hyperparameters.
        Must include at least: cfg["expression_csv"], cfg["markers_csv"], cfg["num_clusters"] 
        and cfg["segmentation_npy"] or cfg["morpho_features_csv"] and cfg["centroids_csv"]

    Returns
    -------
    model : Any
        Trained model object returned by model.train.train(...)
    pred : pd.DataFrame
        Clustering output
    """

    if "num_clusters" not in cfg:
        raise KeyError("cfg must include 'num_clusters' (int).")

    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    seed = training_cfg.get("seed", 5507)
    set_seed(seed) 
    # === preprocessing ===
    G, modelists = preprocessing(cfg)

    # compute number of known/modelist clusters (excluding -1)
    unique_labs = modelists[modelists != -1].unique()
    num_modelists_clusters = int(unique_labs.numel())

    # === build model ===    
    model = ModelistsGCN(
        input_dim=G.num_node_features,
        hidden_dims=model_cfg.get("hidden_dims", (64,)),
        latent_dim=model_cfg.get("latent_dim", 16),
        num_clusters=int(cfg["num_clusters"]),
        )
    # === training ===
    
    model, z = model.fit(
        G=G,
        modelists=modelists,
        num_clusters=int(cfg["num_clusters"]),
        num_modelists_clusters=num_modelists_clusters,
        lr=training_cfg.get("lr", 1e-2),
        prop_weight=training_cfg.get("alpha", 1.0),
        contrastive_weight=training_cfg.get("gamma", 1.0),
        pull_weight=training_cfg.get("beta", 0.6),
        num_epochs=training_cfg.get("epochs", 20),
        seed=training_cfg.get("seed", 5507),
    )
    
    print("\n=== Clustering ===")
    labels = model.predict(G)
    labels_df = pd.DataFrame({"pred": labels}, index=pd.Index(G.cell_ids, name="cellID"))

    print(labels_df.head())
    
    return model, labels_df if cfg.get("return_model", True) else labels_df


