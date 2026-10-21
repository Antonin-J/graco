"""Laplacians and degrees.

Three conventions, with A the adjacency and D its degrees:

    combinatorial          L = D - A
    normalized             L = D^{-1/2} (D - A) D^{-1/2}
    normalized_self_loop   L = (D+I)^{-1/2} (D - A) (D+I)^{-1/2}

Nothing here checks that the graph is undirected and without self-loops; the
loaders ensure it.
"""

from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch

from graco.core.conversions import data_to_scipy_csr_adjacency, scipy_csr_to_torch_csr

LaplacianKind = Literal["combinatorial", "normalized", "normalized_self_loop"]


def degrees(A: sp.csr_matrix) -> np.ndarray:
    """Row sums of a scipy CSR adjacency, float64."""
    return np.asarray(A.sum(axis=1)).ravel().astype(np.float64, copy=False)


def shifted_degrees(deg: np.ndarray, laplacian_kind: LaplacianKind) -> np.ndarray:
    """Degrees inside the normalisation: D + 1 for normalized_self_loop, D for
    normalized, ones for combinatorial, which leaves Q binary.
    """
    if laplacian_kind == "normalized_self_loop":
        return deg + 1.0
    elif laplacian_kind == "normalized":
        return deg
    elif laplacian_kind == "combinatorial":
        return np.ones_like(deg)
    else:
        raise ValueError(f"unknown laplacian_kind {laplacian_kind!r}")


def normalize_laplacian_sp(
    A: sp.csr_matrix, deg: np.ndarray, laplacian_kind: LaplacianKind
) -> sp.csr_matrix:
    """The chosen Laplacian as scipy CSR, from an adjacency and its degrees.

    Local variation calls it at every level.
    """
    L_comb = (sp.diags(deg) - A).tocsr()
    if laplacian_kind == "combinatorial":
        return L_comb
    D_N = shifted_degrees(deg, laplacian_kind)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(D_N > 0, D_N ** -0.5, 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return (D_inv_sqrt @ L_comb @ D_inv_sqrt).tocsr()


def build_laplacian(
    data, laplacian_kind: LaplacianKind, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """The chosen Laplacian of a PyG Data, torch sparse CSR.

    float32 by default, as in PyG; ask float64 where precision matters, e.g. for
    a full eigendecomposition.
    """
    if laplacian_kind not in ("combinatorial", "normalized", "normalized_self_loop"):
        raise ValueError(
            "laplacian_kind must be combinatorial / normalized / normalized_self_loop, "
            f"got {laplacian_kind!r}"
        )
    A = data_to_scipy_csr_adjacency(data)
    deg = degrees(A)
    L = normalize_laplacian_sp(A, deg, laplacian_kind)
    return scipy_csr_to_torch_csr(L).to(dtype)
