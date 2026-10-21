"""Lifting helpers on torch sparse CSR: degrees, coarsened adjacency and the
degree-normalised Q. See Coarsening for the notation.
"""

from __future__ import annotations

import numpy as np
import torch

from graco.core.conversions import (
    csr_from_triplets, extract_triplets, sparse_add, sparse_matmul,
)
from graco.core.laplacian import LaplacianKind, shifted_degrees


def degrees_torch(A: torch.Tensor) -> torch.Tensor:
    """Weighted row sums of a torch sparse CSR adjacency."""
    if A.layout != torch.sparse_csr:
        raise TypeError(f"A must be torch.sparse_csr, got {A.layout}")
    row, _, vals = extract_triplets(A)
    deg = torch.zeros(A.shape[0], dtype=vals.dtype, device=A.device)
    deg.scatter_add_(0, row, vals)
    return deg


def coarsened_adjacency(
    A: torch.Tensor,
    Q_binary: torch.Tensor,
    *,
    delete_diagonal: bool = True,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Coarsened adjacency A_c = Q_b^T A Q_b, torch sparse CSR.

    Parameters
    ----------
    A : torch.Tensor, sparse CSR (N, N)
    Q_binary : torch.Tensor, sparse CSR (N, n)
        Binary indicator of the partition.
    delete_diagonal : bool, default True
        If True, drop the diagonal of A_c, i.e. the edge weight inside each
        supernode.
    symmetrize : bool, default True
        If True, return (A_c + A_c^T) / 2.

    Returns
    -------
    torch.Tensor, sparse CSR (n, n)
    """
    if A.layout != torch.sparse_csr:
        raise TypeError(f"A must be torch.sparse_csr, got {A.layout}")
    if Q_binary.layout != torch.sparse_csr:
        raise TypeError(f"Q_binary must be torch.sparse_csr, got {Q_binary.layout}")
    if A.device != Q_binary.device:
        raise ValueError(
            f"device mismatch: A on {A.device}, Q_binary on {Q_binary.device}"
        )

    # The transpose is CSC, which torch multiplies with CSR natively, faster
    # than converting it first.
    Ac = sparse_matmul(sparse_matmul(Q_binary.t(), A), Q_binary)  # Q^T A Q, MKL case in needs_coo

    if delete_diagonal:
        Ac = _drop_diagonal(Ac)
    if symmetrize:
        # Addition needs both operands in CSR; CSR + CSC is unsupported.
        Ac = sparse_add(Ac, Ac.t().to_sparse_csr()) * 0.5  # (Ac + Ac^T) / 2, MKL case in needs_coo
        if Ac.layout != torch.sparse_csr:
            Ac = Ac.to_sparse_csr()
    return Ac


def Q_normalized_from_partition(
    Q_binary: torch.Tensor,
    deg_orig: torch.Tensor,
    deg_coarsen: torch.Tensor,
    laplacian_kind: LaplacianKind,
) -> torch.Tensor:
    """Degree-normalised lifting Q[i, c] = sqrt(D_i / D_c) Q_binary[i, c].

    D is deg_orig for the original nodes and deg_coarsen for the supernodes,
    plus one under "normalized_self_loop". Under "combinatorial" Q = Q_binary.
    A supernode of degree zero gets a zero column.
    """
    if Q_binary.layout != torch.sparse_csr:
        raise TypeError(f"Q_binary must be torch.sparse_csr, got {Q_binary.layout}")
    N, n = Q_binary.shape
    if deg_orig.shape != (N,):
        raise ValueError(f"deg_orig shape {tuple(deg_orig.shape)} != ({N},)")
    if deg_coarsen.shape != (n,):
        raise ValueError(f"deg_coarsen shape {tuple(deg_coarsen.shape)} != ({n},)")
    device = Q_binary.device
    dtype = Q_binary.values().dtype

    D_N_np = shifted_degrees(deg_orig.detach().cpu().numpy(), laplacian_kind)
    D_n_np = shifted_degrees(deg_coarsen.detach().cpu().numpy(), laplacian_kind)
    sqrt_D_N = torch.from_numpy(np.sqrt(D_N_np)).to(dtype=dtype, device=device)
    with np.errstate(divide="ignore"):
        sqrt_inv_D_n_np = np.where(D_n_np > 0, D_n_np ** -0.5, 0.0)
    sqrt_inv_D_n = torch.from_numpy(sqrt_inv_D_n_np).to(dtype=dtype, device=device)

    crow = Q_binary.crow_indices()
    col_idx = Q_binary.col_indices()
    vals = Q_binary.values()
    row_idx = torch.repeat_interleave(
        torch.arange(N, dtype=torch.int64, device=device), crow[1:] - crow[:-1]
    )
    new_vals = vals * sqrt_D_N[row_idx] * sqrt_inv_D_n[col_idx]
    # Same sparsity as Q_binary, so crow and col_idx carry over.
    return torch.sparse_csr_tensor(crow, col_idx, new_vals, size=(N, n))


def _drop_diagonal(M: torch.Tensor) -> torch.Tensor:
    """Drop the diagonal entries of a torch sparse CSR matrix."""
    row, col, vals = extract_triplets(M)
    keep = row != col
    return csr_from_triplets(
        row[keep], col[keep], vals[keep], M.shape[0], M.shape[1]
    )
