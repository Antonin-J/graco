"""Conversions between PyG Data, scipy CSR and torch sparse CSR, and the
triplet helpers that edit a torch CSR tensor without going through COO.

scipy is used where torch has no equivalent: the eigensolver of eig.py and
the internals of local variation. Outputs stay on the device of the inputs.
"""

import numpy as np
import scipy.sparse as sp
import torch


def data_to_scipy_csr_adjacency(data) -> sp.csr_matrix:
    """Scipy CSR adjacency of a PyG Data, float64, with unit weights when edge_weight is absent."""
    edge_index = data.edge_index.detach().cpu().numpy()
    num_nodes = int(data.num_nodes)
    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is None:
        w = np.ones(edge_index.shape[1], dtype=np.float64)
    else:
        w = edge_weight.detach().cpu().numpy().astype(np.float64, copy=False)
    return sp.csr_matrix(
        (w, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )


def scipy_csr_to_torch_csr(
    M: sp.csr_matrix, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Convert scipy CSR to torch.sparse_csr_tensor. Copies the three arrays."""
    M = M.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(M.indptr.astype(np.int64)),
        torch.from_numpy(M.indices.astype(np.int64)),
        torch.from_numpy(M.data.astype(np.float64)).to(dtype),
        size=M.shape,
    )


def torch_csr_to_scipy_csr(T: torch.Tensor) -> sp.csr_matrix:
    """Convert a torch sparse CSR tensor to scipy CSR."""
    if T.layout != torch.sparse_csr:
        raise TypeError(f"expected torch.sparse_csr layout, got {T.layout}")
    return sp.csr_matrix(
        (
            T.values().detach().cpu().numpy(),
            T.col_indices().detach().cpu().numpy(),
            T.crow_indices().detach().cpu().numpy(),
        ),
        shape=tuple(T.shape),
    )


def csr_from_triplets(
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    N_rows: int,
    N_cols: int,
) -> torch.Tensor:
    """Torch sparse CSR tensor from unsorted (rows, cols, vals) triplets.

    Sorts the triplets by (row, col) and counts the entries of each row.
    Duplicate (row, col) pairs are not summed, so the caller merges them first.
    """
    device = rows.device
    if rows.shape != cols.shape or rows.shape != vals.shape:
        raise ValueError(
            f"shape mismatch: rows {rows.shape}, cols {cols.shape}, vals {vals.shape}"
        )
    rows_long = rows.to(torch.int64)
    cols_long = cols.to(torch.int64)
    order = torch.argsort(rows_long * N_cols + cols_long)
    s_rows = rows_long[order]
    s_cols = cols_long[order]
    s_vals = vals[order]
    counts = torch.bincount(s_rows, minlength=N_rows)
    crow = torch.zeros(N_rows + 1, dtype=torch.int64, device=device)
    crow[1:] = torch.cumsum(counts, dim=0)
    return torch.sparse_csr_tensor(crow, s_cols, s_vals, size=(N_rows, N_cols))


def extract_triplets(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(row, col, val) of a torch sparse CSR tensor, rows rebuilt from crow_indices."""
    if M.layout != torch.sparse_csr:
        raise TypeError(f"expected torch.sparse_csr layout, got {M.layout}")
    crow = M.crow_indices()
    col = M.col_indices()
    vals = M.values()
    counts = crow[1:] - crow[:-1]
    row = torch.repeat_interleave(
        torch.arange(M.shape[0], dtype=torch.int64, device=M.device), counts
    )
    return row, col, vals


def needs_coo(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True for two sparse operands on CPU, which then go through COO.

    On CPU the CSR sum and product run in MKL, which never frees the memory
    of the result it builds (still the case in torch 2.9.1), so a job that
    repeats them is killed for host memory. Builds without MKL (macOS, ARM)
    raise instead. COO gives the same result, a few times slower. On GPU the
    CSR path is cuSPARSE and stays.
    """
    sparse = (torch.sparse_csr, torch.sparse_csc)
    return a.layout in sparse and b.layout in sparse and a.device.type == "cpu"


def sparse_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a + b, through COO when needs_coo."""
    if not needs_coo(a, b):
        return a + b
    return (a.to_sparse_coo() + b.to_sparse_coo()).coalesce().to_sparse_csr()


def sparse_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b, through COO when needs_coo."""
    if not needs_coo(a, b):
        return a @ b
    return torch.sparse.mm(a.to_sparse_coo(), b.to_sparse_coo()).coalesce().to_sparse_csr()


def data_to_torch_csr_adjacency(data, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Torch sparse CSR adjacency of a PyG Data, with unit weights when edge_weight is absent.

    float32 by default, as in PyG; ask float64 where precision matters.
    """
    ei = data.edge_index
    device = ei.device
    N = int(data.num_nodes)
    ew = getattr(data, "edge_weight", None)
    if ew is None:
        vals = torch.ones(ei.shape[1], dtype=dtype, device=device)
    else:
        vals = ew.to(dtype).to(device)
    return csr_from_triplets(ei[0], ei[1], vals, N, N)
