"""Metrics of the partition and of P and Q, with no Laplacian.

max_acc, cluster_size_distribution, check_coarsening, granular_ball_quality,
partition_density_quality and modularity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from graco.core.coarsening import Coarsening
from graco.core.conversions import (
    data_to_torch_csr_adjacency, extract_triplets, sparse_matmul,
)


def max_acc(coarsening: Coarsening, original, *, mask: Optional[torch.Tensor] = None) -> dict:
    """Accuracy ceiling of any prediction constant on each supernode.

    Each supernode predicts the majority label of its members, the best a
    constant prediction can do; max_acc is the accuracy of that prediction (Joly
    and Keriven, NeurIPS 2024).

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
        Must carry y.
    mask : torch.Tensor of bool (N,), optional
        Extra subset of nodes to score.

    Returns
    -------
    dict
        all : accuracy over all nodes.
        train, val, test : accuracy over each split mask present, a list with
        one value per split for (N, n_splits) masks.
        masked : accuracy over mask, when given.
    """
    data = original
    y = data.y
    if y is None:
        raise ValueError("max_acc requires original.y to be set")
    assignment = coarsening.partition.assignment
    if not isinstance(assignment, torch.Tensor):
        assignment = torch.as_tensor(assignment, dtype=torch.long)
    if assignment.shape[0] != y.shape[0]:
        raise ValueError(
            f"assignment ({assignment.shape[0]}) != y ({y.shape[0]}); "
            "Coarsening and labels are inconsistent"
        )

    n_classes = int(y.max().item()) + 1
    n_clusters = coarsening.n

    counts = torch.zeros(n_clusters, n_classes, dtype=torch.long, device=y.device)
    counts.index_add_(0, assignment, torch.nn.functional.one_hot(y, num_classes=n_classes))
    majority = counts.argmax(dim=1)         # (n_clusters,)
    y_pred = majority[assignment]           # (N,)

    out = {"all": float((y_pred == y).float().mean().item())}
    for name in ("train", "val", "test"):
        m = getattr(data, f"{name}_mask", None)
        if m is None:
            continue
        if m.ndim == 2:
            out[name] = [float((y_pred[m[:, s]] == y[m[:, s]]).float().mean().item())
                         for s in range(m.shape[1])]
        else:
            out[name] = float((y_pred[m] == y[m]).float().mean().item())
    if mask is not None:
        out["masked"] = float((y_pred[mask] == y[mask]).float().mean().item())
    return out


def cluster_size_distribution(coarsening: Coarsening) -> dict:
    """Cluster sizes and their summary statistics.

    Returns
    -------
    dict
        n_clusters : number of clusters.
        sizes : tensor (n_clusters,) of sizes.
        min, max, mean, std, median : statistics of the sizes.
        n_singletons : number of clusters of size one.
        max_size_ratio : max size over mean size, 1.0 when balanced.
    """
    a = coarsening.partition.assignment
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a, dtype=torch.long)
    n_clusters = coarsening.n
    sizes = torch.bincount(a, minlength=n_clusters).to(torch.long)
    sizes_f = sizes.to(torch.float64)
    mean = float(sizes_f.mean().item())
    return {
        "n_clusters": int(n_clusters),
        "sizes": sizes,
        "min": int(sizes.min().item()),
        "max": int(sizes.max().item()),
        "mean": mean,
        "std": float(sizes_f.std(unbiased=False).item()),
        "median": float(sizes_f.median().item()),
        "n_singletons": int((sizes == 1).sum().item()),
        "max_size_ratio": float(sizes.max().item()) / mean if mean > 0 else float("nan"),
    }


def check_coarsening(
    coarsening: Coarsening,
    *,
    atol: float = 1e-5,
    check_qp_idempotent: bool = False,
    qp_mode: str = "auto",
    n_test_vectors: int = 5,
    n_dense_max: int = 10_000,
    seed: int = 42,
) -> dict:
    """Check that P is a generalised inverse of Q and that Q has one nonzero per row.

    PQ = I_n implies that QP is a projection, (QP)^2 = Q (PQ) P = QP, so the
    (n, n) product PQ is checked by default. The idempotency of QP itself is
    opt-in, as a guard against numerical drift.

    Parameters
    ----------
    coarsening : Coarsening
    atol : float, default 1e-5
        Tolerance of every check.
    check_qp_idempotent : bool, default False
        If True, also check (QP)^2 = QP.
    qp_mode : {"auto", "frobenius", "stochastic"}, default "auto"
        "frobenius" computes ||(QP)^2 - QP||_F on a dense (N, N) QP.
        "stochastic" takes the largest entry of ((QP)^2 - QP) X for
        n_test_vectors random columns X, with matrix products only. "auto"
        picks "frobenius" when N <= n_dense_max.
    n_test_vectors : int, default 5
    n_dense_max : int, default 10000
    seed : int, default 42
        Seed of X.

    Returns
    -------
    dict
        pq_is_identity : whether max |PQ - I_n| < atol.
        max_pq_minus_eye : max |PQ - I_n|.
        q_well_partitioned : whether every row of Q has exactly one nonzero.
        max_row_nnz, min_row_nnz : extreme nonzero counts of the rows of Q.
        qp_is_idempotent : whether the idempotency score is below atol, with
        check_qp_idempotent.
        qp_idempotency_score : the score of qp_mode, with check_qp_idempotent.
        qp_mode_used : "frobenius" or "stochastic", with check_qp_idempotent.
    """
    P, Q = coarsening.P, coarsening.Q
    n, N = coarsening.n, coarsening.N

    # PQ is (n, n), densified for the comparison with I_n.
    PQ = sparse_matmul(P, Q)  # P Q, MKL case in needs_coo
    PQ_dense = PQ.to_dense() if PQ.layout == torch.sparse_csr else PQ
    pq_minus_eye = (PQ_dense - torch.eye(n, dtype=PQ_dense.dtype, device=PQ_dense.device)).abs().max().item()

    # Q row nnz
    if Q.layout == torch.sparse_csr:
        row_nnz = Q.crow_indices().diff()
    else:
        row_nnz = (Q.abs() > 0).sum(dim=1)
    max_row_nnz = int(row_nnz.max().item())
    min_row_nnz = int(row_nnz.min().item())

    out = {
        "pq_is_identity": bool(pq_minus_eye < atol),
        "max_pq_minus_eye": float(pq_minus_eye),
        "q_well_partitioned": bool(max_row_nnz == 1 and min_row_nnz == 1),
        "max_row_nnz": max_row_nnz,
        "min_row_nnz": min_row_nnz,
    }

    if check_qp_idempotent:
        if qp_mode not in ("auto", "frobenius", "stochastic"):
            raise ValueError(f"qp_mode must be 'auto'/'frobenius'/'stochastic', got {qp_mode!r}")
        mode = qp_mode
        if mode == "auto":
            mode = "frobenius" if N <= n_dense_max else "stochastic"

        if mode == "frobenius":
            # Exact, on a dense (N, N) QP.
            QP = sparse_matmul(Q, P)  # Q P, MKL case in needs_coo
            QP = QP.to_dense() if QP.layout == torch.sparse_csr else QP
            diff = QP @ QP - QP
            score = float(torch.linalg.matrix_norm(diff, ord="fro").item())
        else:
            # Largest entry of (QPQP - QP) X, with matrix products only.
            dtype = P.values().dtype if P.layout == torch.sparse_csr else P.dtype
            g = torch.Generator().manual_seed(seed)
            X = torch.randn(N, n_test_vectors, generator=g, dtype=dtype)
            QPX = Q @ (P @ X)
            QPQPX = Q @ (P @ QPX)
            score = float((QPQPX - QPX).abs().max().item())
        out["qp_is_idempotent"] = bool(score < atol)
        out["qp_idempotency_score"] = score
        out["qp_mode_used"] = mode
    return out


def granular_ball_quality(
    adj: list[set], members: set, use_transitivity: bool = True,
) -> float:
    """Quality of one granular ball, as in GBGC (Xia et al., IJCAI 2025).

        quality = E_in / N_in
                  + [ sum_{i!=j!=k} A_ij A_jk A_ki ] / [ sum_{i!=j!=k} A_ij A_ik ]

    with E_in and N_in the edges and nodes inside the ball. The paper names the
    first term the average degree and writes E/N. The code released by the
    same group computes E/N, so we keep the formula. The second
    term is the global transitivity of the ball, not the mean of the local
    clustering coefficients.

    Parameters
    ----------
    adj : list of set of int
        adj[v] is the set of neighbours of v.
    members : set of int
        Nodes of the ball.
    use_transitivity : bool, default True
        If False, drop the second term, as the follow-up of GBGC does (Wang et
        al., arXiv 2026).

    Returns
    -------
    float
        0.0 for a ball of fewer than two nodes.
    """
    n = len(members)
    if n < 2:
        return 0.0

    internal = {v: adj[v] & members for v in members}
    internal_deg = {v: len(s) for v, s in internal.items()}
    e_internal = sum(internal_deg.values()) / 2
    density = e_internal / n
    if not use_transitivity:
        return float(density)

    triples = sum(d * (d - 1) for d in internal_deg.values())
    if triples == 0:
        return float(density)

    # adj[u] & adj[v] & members is (adj[u] & members) & (adj[v] & members), so
    # the intersections taken for the degrees answer the triangle count too.
    numerator = 0
    for u in members:
        iu = internal[u]
        for v in iu:
            if v <= u:
                continue
            numerator += 6 * len(iu & internal[v])
    numerator = numerator // 3                                              # each triangle counted 3 edge-times

    clustering = numerator / max(triples, 1)
    return float(density + clustering)


def partition_density_quality(coarsening: Coarsening, original) -> dict:
    """granular_ball_quality of every cluster, on the original graph, and its summary.

    Tells whether a partition made dense, clique-like clusters or sparse,
    chain-like ones.

    Returns
    -------
    dict
        mean, max, min : statistics over the clusters of two nodes or more,
        0.0 when there is none.
        per_cluster : np.ndarray (n_clusters,) of the qualities, 0.0 for a
        singleton.
        n_singleton : number of clusters of size one.
    """
    data = original
    N = data.num_nodes
    ei = data.edge_index.cpu().numpy()
    adj: list[set] = [set() for _ in range(N)]
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)

    assignment = coarsening.partition.assignment
    n_clusters = coarsening.partition.n_clusters

    cluster_members: dict[int, set] = {}
    for i, c in enumerate(assignment):
        cluster_members.setdefault(int(c), set()).add(int(i))

    per_cluster = np.zeros(n_clusters, dtype=np.float64)
    n_singleton = 0
    for c in range(n_clusters):
        members = cluster_members.get(c, set())
        if len(members) <= 1:
            n_singleton += 1
            per_cluster[c] = 0.0
        else:
            per_cluster[c] = granular_ball_quality(adj, members)

    non_singleton_mask = np.array(
        [len(cluster_members.get(c, set())) > 1 for c in range(n_clusters)]
    )
    non_singleton_vals = per_cluster[non_singleton_mask]
    return {
        "mean": float(non_singleton_vals.mean()) if non_singleton_vals.size else 0.0,
        "max": float(non_singleton_vals.max()) if non_singleton_vals.size else 0.0,
        "min": float(non_singleton_vals.min()) if non_singleton_vals.size else 0.0,
        "per_cluster": per_cluster,
        "n_singleton": int(n_singleton),
    }


def modularity(coarsening: Coarsening, original) -> float:
    """Newman modularity of the partition (Newman and Girvan, Physical Review E 2004).

        Mod = (1 / 2m) sum_{i, j} [ A_ij - d_i d_j / (2m) ] delta(c_i, c_j)
            = sum_c [ e_c / m - (D_c / 2m)^2 ]

    with m the total edge weight, e_c the weight of the edges inside cluster c
    and D_c the sum of the degrees of its nodes. Mod lies in [-1/2, 1]; Mod > 0
    means denser clusters than the degree-preserving null model. Leiden and
    Louvain maximise it.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data

    Returns
    -------
    float
    """
    data = original
    N = data.num_nodes
    A = data_to_torch_csr_adjacency(data, dtype=torch.float64)              # (N, N) sparse CSR
    row, col, val = extract_triplets(A)                                     # each edge appears twice (i,j) and (j,i)

    assignment = coarsening.partition.assignment
    if not isinstance(assignment, torch.Tensor):
        assignment = torch.as_tensor(assignment, dtype=torch.long)
    n_clusters = coarsening.partition.n_clusters

    two_m = float(val.sum().item())                                         # 2m = sum A_ij
    if two_m <= 0.0:
        return 0.0
    m = two_m / 2.0

    # Edges inside each cluster (undirected count : each internal edge appears
    # twice in the row/col symmetric list, so we divide by 2 at the end).
    row_l = row.to(torch.long)
    col_l = col.to(torch.long)
    same = (assignment[row_l] == assignment[col_l])
    internal_edge_weight = torch.zeros(n_clusters, dtype=torch.float64)
    intra_c = assignment[row_l][same]
    intra_v = val[same].to(torch.float64)
    internal_edge_weight.scatter_add_(0, intra_c, intra_v)
    internal = internal_edge_weight / 2.0                                   # each edge counted twice

    deg = torch.zeros(N, dtype=torch.float64)
    deg.scatter_add_(0, row_l, val.to(torch.float64))

    D_c = torch.zeros(n_clusters, dtype=torch.float64)
    D_c.scatter_add_(0, assignment, deg)

    mod = (internal / m - (D_c / (2.0 * m)).pow(2)).sum()
    return float(mod.item())
