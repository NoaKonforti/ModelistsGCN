
from collections import Counter
import math
from typing import Dict, List, Sequence, Tuple, Any
import pandas as pd
import copy


def filter_duplicates(
    modelist_dict: Dict[str, Sequence[str]],
    min_modelists: int = 7
)->Dict[str, List[str]]:
    """
    Removes cells that appear as modelists in multiple cell types 
    to ensure each modelist uniquely represents a single cluster.
    """
    # Flatten all cell IDs and count their occurrences
    all_ids = [cell_id for ids in modelist_dict.values() for cell_id in ids]
    id_counts = Counter(all_ids)

    filtered = {}
    for cell_type, cell_ids in modelist_dict.items():
        # Keep only cell IDs that appear exactly once globally
        unique = [c for c in cell_ids if id_counts[c] == 1]
        if len(unique) < min_modelists:
            print(f"Removing {cell_type} after duplicate filtering: only {len(unique)} unique modelists remain")
            continue
        filtered[cell_type] = unique
        filtered[cell_type] = filtered[cell_type]

    return filtered

def find_modelists(
    expr: pd.DataFrame,
    marker_df: pd.DataFrame,     
    quantile_thresh: float = 0.9,
    min_modelists: int = 7,
    min_gene_fraction: float = 0.8,
    max_other_gene_fraction: float = 0.0
) -> Dict[str, List[str]]:
    """
    Identifies high-confidence anchor cells (“modelists”) for each cell type 
    based on strong marker gene expression and low expression of other cell-type markers.
    """
    
    modelist_dict = {}
    marker_dict = marker_df.groupby("CellType")["Marker"].apply(list).to_dict()

    gene_detection_rate = (expr).sum() / len(expr)

    valid_genes_global = set(gene_detection_rate[gene_detection_rate > 0.1].index)

    all_marker_genes = set(marker_df["Marker"]) & set(expr.columns) & valid_genes_global

    for cell_type, markers in marker_dict.items():

        valid_genes = [g for g in markers if g in all_marker_genes]


        # === Positive selection ===
        subset = expr[valid_genes]
        thresholds = subset.quantile(quantile_thresh)

        thresholds = thresholds[thresholds > 2] #only above 2 expression
        subset = subset[thresholds.index]


        above_threshold = subset.ge(thresholds)
        high_expr_counts = above_threshold.sum(axis=1)


        required_count = math.ceil(min_gene_fraction * len(thresholds))

        
        qualifying_cells = high_expr_counts[(high_expr_counts >= required_count) & (high_expr_counts >= 1)]


        # === Negative selection: suppress expression of other markers ===
        other_genes = list(all_marker_genes - set(valid_genes))

        global_other_means = expr[other_genes].mean()

        candidate_expr = expr.loc[qualifying_cells.index, global_other_means.index]

        tolerance = 0.5           # max allowed distance between candidate and negative mean
        max_mean_for_tolerance = 1.0  # only apply tolerance if negative mean <= 1

        base_low = candidate_expr.le(global_other_means)

        diff = (candidate_expr - global_other_means).abs()

        within_tolerance = (diff.le(tolerance)) & (global_other_means.le(max_mean_for_tolerance))
        is_low_expr = base_low | within_tolerance

        # 4. Keep cells that underexpress most of the other genes
        min_required_low = int((1 - max_other_gene_fraction) * len(global_other_means))

        low_expr_counts = is_low_expr.sum(axis=1)
        final_cells = low_expr_counts[low_expr_counts >= min_required_low].index

        selected = final_cells
        modelist_dict[cell_type] = selected.tolist()
    modelist_dict = filter_duplicates(modelist_dict, min_modelists)
        
    return modelist_dict


# ============================================================
# Report modelists summary (works with your existing output)
# ============================================================
def report_modelists(modelist_dict: Dict[str, List[str]], *, verbose: bool = True) -> Tuple[bool, Dict[str, int]]:
    """
    Print how many modelists were found per cell type.
    """
    counts = {ct: len(ids) for ct, ids in modelist_dict.items()}
    has_any = any(n > 0 for n in counts.values())

    if verbose:
        if not counts:
            print("No modelists found (modelist_dict is empty).")
        else:
            print("Modelists per cell type:")
            for ct, n in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {'-' if n > 0 else '⚠️'} {ct}: {n}")

    return has_any, counts


# ============================================================
# Wrapper: run functions using cfg["modelists"], with auto-relax if needed
# ============================================================
def build_relax_schedule(mcfg):
    q0 = float(mcfg["quantile_thresh"])
    g0 = float(mcfg["min_gene_fraction"])
    o0 = float(mcfg["max_other_gene_fraction"])

    return [
        {"min_gene_fraction": max(0.0, g0 - 0.05)},
        {"min_gene_fraction": max(0.0, g0 - 0.10)},
        {"max_other_gene_fraction": min(0.5, o0 + 0.05)},
        {"max_other_gene_fraction": min(0.5, o0 + 0.10)},
        {"min_gene_fraction": max(0.0, g0 - 0.15)},
        {"quantile_thresh": max(0.5, q0 - 0.03)},
        {"quantile_thresh": max(0.5, q0 - 0.07)},
    ]

def run_modelists_from_cfg(
    expr,
    marker_df,
    cfg: Dict[str, Any],
    *,
    min_types_with_modelists: int = 1,
    max_tries: int = 6,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:

    updated_cfg = copy.deepcopy(cfg)
    mcfg = updated_cfg.get("modelists", {})

    # ensure defaults exist (no behavior change if already provided)
    mcfg.setdefault("quantile_thresh", 0.9)
    mcfg.setdefault("min_modelists", 7)
    mcfg.setdefault("min_gene_fraction", 0.8)
    mcfg.setdefault("max_other_gene_fraction", 0.0)


    relax_steps = build_relax_schedule(mcfg)


    last_modelist_dict: Dict[str, List[str]] = {}

    for attempt in range(1, max_tries + 1):
        if verbose:
            print(f"\nModelists attempt {attempt}/{max_tries}")
            print("Using modelists cfg:", mcfg)

        # Run YOUR function exactly as-is
        last_modelist_dict = find_modelists(
            expr=expr,
            marker_df=marker_df,
            quantile_thresh=float(mcfg["quantile_thresh"]),
            min_modelists=int(mcfg["min_modelists"]),
            min_gene_fraction=float(mcfg["min_gene_fraction"]),
            max_other_gene_fraction=float(mcfg["max_other_gene_fraction"]),
        )

        has_any, counts = report_modelists(last_modelist_dict, verbose=verbose)
        n_types = sum(1 for n in counts.values() if n > 0)
        
        if "min_types_with_modelists" not in cfg["modelists"]:
            min_types_with_modelists = 1
        else:
            min_types_with_modelists = cfg["modelists"]["min_types_with_modelists"]
        
        if has_any and n_types >= min_types_with_modelists:
            if verbose:
                print(f"Success: found modelists for {n_types} cell types.")
            return last_modelist_dict, updated_cfg

        # No/insufficient modelists → relax thresholds (if we still can)
        if attempt <= len(relax_steps):
            step = relax_steps[attempt - 1]

            # Apply step but keep within sensible bounds
            if "quantile_thresh" in step:
                mcfg["quantile_thresh"] = max(0.60, min(0.95, step["quantile_thresh"]))
            if "min_gene_fraction" in step:
                mcfg["min_gene_fraction"] = max(0.40, min(0.95, step["min_gene_fraction"]))
            if "max_other_gene_fraction" in step:
                mcfg["max_other_gene_fraction"] = max(0.0, min(0.50, step["max_other_gene_fraction"]))
            if "min_modelists" in step:
                mcfg["min_modelists"] = int(max(1, step["min_modelists"]))

            if verbose:
                print("No/insufficient modelists — relaxing cfg to:", step)
        else:
            if verbose:
                print("Exhausted relaxation schedule.")
            break

    return last_modelist_dict, updated_cfg