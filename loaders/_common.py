"""Dataset directory and a validation helper shared by the loaders.

Each family downloads into its own subdirectory of the dataset directory:
planetoid/, heterophilic_graphs/, wikipedia/, city_networks/, ogb/, reddit/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

_DEFAULT_DATASETS_DIR = Path.home() / ".cache" / "graco" / "datasets"


def datasets_dir(override: Optional[str] = None) -> str:
    """Dataset directory: override if given, else GRACO_DATASETS_DIR, else
    ~/.cache/graco/datasets.
    """
    if override is not None:
        return str(override)
    env = os.environ.get("GRACO_DATASETS_DIR")
    if env:
        return env
    return str(_DEFAULT_DATASETS_DIR)


def validate_data(data: Data) -> dict:
    """Report the properties the coarseners assume, without enforcing any.

    Returns
    -------
    dict
        connected, n_components : connectivity of the graph.
        no_self_loops, n_self_loops : self-loops.
        is_undirected : whether the adjacency is symmetric.
        train_mask_shape, val_mask_shape, test_mask_shape : when the mask exists.
        masks_disjoint : whether the three masks share no node, on the first
        split; present when all three exist.
    """
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    n_components, _ = connected_components(A, directed=False)
    diag_nz = int((A.diagonal() != 0).sum())
    is_symmetric = (A != A.T).nnz == 0
    out = {
        "connected": n_components == 1,
        "n_components": int(n_components),
        "no_self_loops": diag_nz == 0,
        "n_self_loops": diag_nz,
        "is_undirected": bool(is_symmetric),
    }
    for name in ("train_mask", "val_mask", "test_mask"):
        m = getattr(data, name, None)
        if m is not None:
            out[f"{name}_shape"] = tuple(m.shape)
    tr, va, te = (getattr(data, n, None) for n in ("train_mask", "val_mask", "test_mask"))
    if tr is not None and va is not None and te is not None:
        tr1 = tr[:, 0] if tr.ndim == 2 else tr
        va1 = va[:, 0] if va.ndim == 2 else va
        te1 = te[:, 0] if te.ndim == 2 else te
        overlap = (tr1 & va1).any() | (tr1 & te1).any() | (va1 & te1).any()
        out["masks_disjoint"] = not bool(overlap.item())
    return out


def _split_mask(mask, split_idx: int):
    """Column split_idx of a (N, n_splits) mask; a 1D mask is returned as is."""
    if mask is None:
        return None
    if mask.ndim == 1:
        return mask
    return mask[:, split_idx]
