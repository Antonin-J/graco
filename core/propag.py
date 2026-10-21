"""Propagation matrices of a GCN layer X to S X W, on torch sparse CSR.

On the original graph, mode of build_S:

    self_loop_adj         (D+I)^{-1/2} (A+I) (D+I)^{-1/2}, as in Kipf and Welling
    normalized_adj        D^{-1/2} A D^{-1/2}
    combinatorial_tuned   I - L / N, with L = D - A

On the coarsened graph, mode of build_S_coarsen:

    MP                    S_c = P S Q (Joly and Keriven, NeurIPS 2024)
    kw_on_Ac              build_S applied to the coarsened graph
"""

from __future__ import annotations

from typing import Literal

import torch

from graco.core.coarsening import Coarsening
from graco.core.conversions import (
    csr_from_triplets,
    data_to_torch_csr_adjacency,
    extract_triplets,
    sparse_matmul,
)
from graco.core.q import degrees_torch

OriginalMode = Literal["self_loop_adj", "normalized_adj", "combinatorial_tuned"]
CoarsenedMode = Literal["MP", "kw_on_Ac"]


def build_S(
    data,
    mode: OriginalMode = "self_loop_adj",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Propagation matrix S of a graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph without self-loops.
    mode : {"self_loop_adj", "normalized_adj", "combinatorial_tuned"}, default "self_loop_adj"
        Formula, see the module docstring.
    dtype : torch.dtype, default torch.float32

    Returns
    -------
    torch.Tensor, sparse CSR (N, N)
    """
    A = data_to_torch_csr_adjacency(data, dtype=dtype)
    deg = degrees_torch(A)
    N = A.shape[0]
    device = A.device

    if mode == "self_loop_adj":
        d_inv_sqrt = (deg + 1.0).pow(-0.5)
        return _scale_with_diagonal_injection(
            A, d_inv_sqrt, diag_inject_value=torch.ones(N, dtype=d_inv_sqrt.dtype, device=device),
        )
    elif mode == "normalized_adj":
        d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        return _scale_edges(A, d_inv_sqrt)
    elif mode == "combinatorial_tuned":
        # S = I - L/N = (NI - D + A)/N
        # Off-diag: A_ij / N. Diag: (N - d_i)/N.
        diag_vals = (N - deg) / N
        return _scale_and_set_diagonal(A, edge_scale=1.0 / N, diag_values=diag_vals)
    else:
        raise ValueError(
            f"mode must be self_loop_adj / normalized_adj / combinatorial_tuned, got {mode!r}"
        )


def build_S_coarsen(
    coarsening: Coarsening,
    S_original: torch.Tensor,
    mode: CoarsenedMode = "MP",
    original_mode: OriginalMode = "self_loop_adj",
) -> torch.Tensor:
    """Propagation matrix of the coarsened graph.

    Parameters
    ----------
    coarsening : Coarsening
    S_original : torch.Tensor, sparse CSR (N, N)
        Propagation matrix of the original graph, read by "MP" only.
    mode : {"MP", "kw_on_Ac"}, default "MP"
        "MP" returns S_c = P S Q, often asymmetric even when S is symmetric.
        "kw_on_Ac" returns build_S on the coarsened graph.
    original_mode : str, default "self_loop_adj"
        Formula of build_S under "kw_on_Ac".

    Returns
    -------
    torch.Tensor, sparse CSR (n, n)
    """
    if mode == "kw_on_Ac":
        return build_S(coarsening.coarsened, original_mode)
    if mode != "MP":
        raise ValueError(f"mode must be MP or kw_on_Ac, got {mode!r}")
    # P S Q, MKL case in needs_coo
    return sparse_matmul(sparse_matmul(coarsening.P, S_original), coarsening.Q)


def _scale_edges(A: torch.Tensor, d_inv_sqrt: torch.Tensor) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2}, with the sparsity of A."""
    crow = A.crow_indices()
    col = A.col_indices()
    vals = A.values()
    row = torch.repeat_interleave(
        torch.arange(A.shape[0], dtype=torch.int64, device=A.device),
        crow[1:] - crow[:-1],
    )
    new_vals = vals * d_inv_sqrt[row] * d_inv_sqrt[col]
    return torch.sparse_csr_tensor(crow, col, new_vals, size=A.shape)


def _scale_with_diagonal_injection(
    A: torch.Tensor,
    d_inv_sqrt: torch.Tensor,
    diag_inject_value: torch.Tensor,
) -> torch.Tensor:
    """D^{-1/2} (A + diag(diag_inject_value)) D^{-1/2}, for A without self-loops."""
    N = A.shape[0]
    device = A.device

    row, col, vals = extract_triplets(A)
    a_vals = vals * d_inv_sqrt[row] * d_inv_sqrt[col]

    diag_arange = torch.arange(N, dtype=torch.int64, device=device)
    diag_vals = diag_inject_value * d_inv_sqrt * d_inv_sqrt

    all_rows = torch.cat([row, diag_arange])
    all_cols = torch.cat([col, diag_arange])
    all_vals = torch.cat([a_vals, diag_vals.to(dtype=a_vals.dtype)])

    return csr_from_triplets(all_rows, all_cols, all_vals, N, N)


def _scale_and_set_diagonal(
    A: torch.Tensor,
    edge_scale: float,
    diag_values: torch.Tensor,
) -> torch.Tensor:
    """edge_scale A + diag(diag_values), for A without self-loops."""
    N = A.shape[0]
    device = A.device

    row, col, vals = extract_triplets(A)
    a_vals = vals * edge_scale

    diag_arange = torch.arange(N, dtype=torch.int64, device=device)

    all_rows = torch.cat([row, diag_arange])
    all_cols = torch.cat([col, diag_arange])
    all_vals = torch.cat([a_vals, diag_values.to(dtype=a_vals.dtype)])

    return csr_from_triplets(all_rows, all_cols, all_vals, N, N)
