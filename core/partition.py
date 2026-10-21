"""Partition dataclass and the pseudo-inverse of a lifting with one nonzero per row.

Every coarsener returns a Partition: one supernode id per original node, with
contiguous ids in [0, n_clusters). Its binary lifting Q_combi, of shape
(N, n_clusters), has a single 1 per row. Q^T Q is then the diagonal of the
cluster sizes, and P = pinv(Q) = (Q^T Q)^{-1} Q^T averages each cluster.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class Partition:
    """Supernode of every original node, the output of every coarsener.

    Raises ValueError unless the ids cover [0, n_clusters), each at least once;
    relabel_contiguous fixes them.
    """

    assignment: np.ndarray
    n_clusters: int
    _cluster_sizes_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.assignment, np.ndarray):
            self.assignment = np.asarray(self.assignment)
        if self.assignment.ndim != 1:
            raise ValueError(f"assignment must be 1D, got shape {self.assignment.shape}")
        if self.assignment.dtype.kind not in ("i", "u"):
            self.assignment = self.assignment.astype(np.int64, copy=False)
        if self.assignment.dtype != np.int64:
            self.assignment = self.assignment.astype(np.int64, copy=False)
        if int(self.assignment.min()) < 0:
            raise ValueError(f"assignment has negative ids: min={int(self.assignment.min())}")
        if int(self.assignment.max()) >= self.n_clusters:
            raise ValueError(
                f"assignment max={int(self.assignment.max())} >= n_clusters={self.n_clusters}"
            )
        # contiguous: every cluster id in [0, n_clusters) must be used at least once
        used = np.unique(self.assignment)
        if used.shape[0] != self.n_clusters:
            raise ValueError(
                f"assignment has {used.shape[0]} distinct ids but n_clusters={self.n_clusters}; "
                "ids must be contiguous (relabel before constructing Partition)"
            )

    @property
    def N(self) -> int:
        """Number of original nodes."""
        return int(self.assignment.shape[0])

    @property
    def cluster_sizes(self) -> np.ndarray:
        """Number of original nodes per cluster, shape (n_clusters,). Cached."""
        if self._cluster_sizes_cache is None:
            self._cluster_sizes_cache = np.bincount(self.assignment, minlength=self.n_clusters)
        return self._cluster_sizes_cache

    def to_Q_combi(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Binary lifting matrix as torch sparse CSR, shape (N, n_clusters), one 1 per row."""
        N, n = self.N, self.n_clusters
        crow = torch.arange(N + 1, dtype=torch.int64)  # one nonzero per row
        col = torch.from_numpy(self.assignment.astype(np.int64, copy=False))
        vals = torch.ones(N, dtype=dtype)
        return torch.sparse_csr_tensor(crow, col, vals, size=(N, n))


def relabel_contiguous(raw_assignment: np.ndarray) -> tuple[np.ndarray, int]:
    """Remap arbitrary integer ids to contiguous ids 0..n-1.

    Returns
    -------
    assignment : np.ndarray of int64
    n_clusters : int
    """
    raw_assignment = np.asarray(raw_assignment)
    used, inverse = np.unique(raw_assignment, return_inverse=True)
    return inverse.astype(np.int64), int(used.shape[0])


def pinv_well_partitioned(Q: sp.csr_matrix) -> sp.csr_matrix:
    """Closed-form pseudo-inverse of a lifting with one nonzero per row, scipy CSR.

    Q^T Q is then diagonal with entries s_c = sum_{i in c} Q[i, c]^2, so
    pinv(Q) = diag(1 / s_c) Q^T. For a binary Q, s_c is the cluster size and P
    averages each cluster. An empty column gives a zero row of P.
    """
    Q = Q.tocsr()
    QtQ_diag = np.asarray(Q.multiply(Q).sum(axis=0)).ravel()
    QtQ_diag = np.where(QtQ_diag > 0, QtQ_diag, 1.0)
    return (sp.diags(1.0 / QtQ_diag) @ Q.T).tocsr()


def pinv_well_partitioned_torch(Q: torch.Tensor) -> torch.Tensor:
    """Torch sparse CSR version of pinv_well_partitioned."""
    if Q.layout != torch.sparse_csr:
        raise TypeError(f"expected torch.sparse_csr, got {Q.layout}")
    N, n_cols = Q.shape
    vals = Q.values()
    col_idx = Q.col_indices()
    crow = Q.crow_indices()
    device, dtype = vals.device, vals.dtype

    # diag(Q^T Q): sum of the squared values of each column of Q.
    diag = torch.zeros(n_cols, dtype=dtype, device=device)
    diag.scatter_add_(0, col_idx, vals * vals)
    diag_safe = torch.where(diag > 0, diag, torch.ones_like(diag))

    # P = diag(1/diag) Q^T has the nonzeros of Q^T, with rescaled values.
    new_vals = vals / diag_safe[col_idx]
    row_idx_Q = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.int64), crow[1:] - crow[:-1]
    )
    # P is (n_cols, N): its rows are the cluster ids, its columns the nodes.
    from graco.core.conversions import csr_from_triplets
    return csr_from_triplets(col_idx.to(torch.int64), row_idx_Q, new_vals, n_cols, N)


def partition_from_Q(Q) -> "Partition":
    """Partition of a lifting with one nonzero per row, scipy or torch CSR.

    Ids are relabelled to be contiguous, so empty columns are dropped.
    """
    if isinstance(Q, torch.Tensor):
        if Q.layout != torch.sparse_csr:
            raise TypeError(f"expected torch.sparse_csr or scipy CSR, got {Q.layout}")
        if Q.crow_indices().shape[0] - 1 != Q.shape[0]:
            raise RuntimeError("Q does not look like a row-stochastic lifting")
        col_idx = Q.col_indices().detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        Q_sp = Q.tocsr()
        if Q_sp.indptr.shape[0] - 1 != Q_sp.shape[0]:
            raise RuntimeError("Q does not look like a row-stochastic lifting")
        col_idx = Q_sp.indices.copy().astype(np.int64)
    assignment, n_clusters = relabel_contiguous(col_idx)
    return Partition(assignment, n_clusters)


def sanity_check_partition_Q(partition: "Partition", Q: sp.csr_matrix) -> dict:
    """Consistency checks of a (Partition, Q) pair, run by coarseners with validate=True.

    Returns
    -------
    dict
        qp_idempotent_err : max |(QP)^2 - QP| with P = pinv(Q).
        pinv_err : max |QPQ - Q|.
        cluster_sizes_min, cluster_sizes_max, cluster_sizes_mean : cluster sizes.
        n_singletons : number of clusters of size one.
    """
    sizes = np.asarray(Q.multiply(Q).sum(axis=0)).ravel()
    P_mp = (sp.diags(1.0 / np.where(sizes > 0, sizes, 1.0)) @ Q.T).tocsr()
    QP = (Q @ P_mp).tocsr()
    QPQP = QP @ QP
    diff_idemp = (QPQP - QP).tocsr()
    err_idemp = float(np.abs(diff_idemp.data).max()) if diff_idemp.nnz > 0 else 0.0
    QPQ = Q @ P_mp @ Q
    diff_pinv = (QPQ - Q).tocsr()
    err_pinv = float(np.abs(diff_pinv.data).max()) if diff_pinv.nnz > 0 else 0.0
    return {
        "qp_idempotent_err": err_idemp,
        "pinv_err": err_pinv,
        "cluster_sizes_min": int(partition.cluster_sizes.min()),
        "cluster_sizes_max": int(partition.cluster_sizes.max()),
        "cluster_sizes_mean": float(partition.cluster_sizes.mean()),
        "n_singletons": int((partition.cluster_sizes == 1).sum()),
    }
