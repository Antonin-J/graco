"""Optimal transport between the original and the coarsened graph.

wasserstein_feature_distance compares the features as distributions, by
entropic Sinkhorn (OTCoarsening). gromov_wasserstein_distortion compares S
with its block-mean approximation, in closed form (GW coarsening).

The three d_gw_* share the proximal solver of SGWL (Xu et al., NeurIPS 2019),
graco.algorithms.gw_sgwl.d_gw_proximal, and differ in the cost matrices C_s
on G and C_t on G_c:

    d_gw_paper   C_s = (A + A^T) / 2, C_t = diag(p_t): n disconnected nodes,
                 the target of the SGWL partitioning. It reads only G and n,
                 so it measures how well G splits into n parts.
    d_gw_op      C_s = S = (D+I)^{-1/2} (A+I) (D+I)^{-1/2}, C_t = P S Q:
                 distortion of the propagation operator. C_t can be
                 asymmetric, which GW allows.
    d_gw_struct  C_s = S, C_t the same normalisation applied to A_c, the
                 coarsened adjacency without its diagonal: distortion of the
                 coarsened graph taken as a graph of its own.

Their marginals are deg^b / sum(deg^b) with b = 0.001, the default of SGWL,
close to uniform. Their transport plan is dense, of shape (N, n), as the one
of wasserstein_feature_distance: a GPU is advised.
"""

from __future__ import annotations

from typing import Optional

import torch

from torch_geometric.data import Data as PyGData

from graco.algorithms.gw_sgwl import d_gw_proximal
from graco.core.conversions import (
    csr_from_triplets, data_to_torch_csr_adjacency, extract_triplets,
)
from graco.core.coarsening import Coarsening
from graco.core.propag import build_S
from graco.core.q import degrees_torch


def _degree_marginal(A: torch.Tensor, b: float = 0.001) -> torch.Tensor:
    """Marginal deg^b / sum(deg^b), mixed with a small uniform mass so no node has zero mass."""
    N = A.shape[0]
    deg = A.sum(dim=1, keepdim=True).to_dense().squeeze(1)
    p = torch.pow(deg.clamp(min=0.0), b)
    total = p.sum()
    if total.item() == 0.0:
        p = torch.ones_like(p) / N
    else:
        p = p / total
    # Uniform floor: mix 99.99% degree-based + 0.01% uniform to guarantee p_i > 0.
    p = 0.9999 * p + 0.0001 / N
    return p.unsqueeze(1)


def _A_norm_paper(A: torch.Tensor) -> torch.Tensor:
    """(A + A^T) / 2, also for a directed A, with the mirrored entries summed."""
    N = A.shape[0]
    row, col, val = extract_triplets(A)
    scaled = val / 2.0
    all_row = torch.cat([row, col])
    all_col = torch.cat([col, row])
    all_val = torch.cat([scaled, scaled])
    key = all_row * N + all_col
    unique_key, inverse = torch.unique(key, return_inverse=True)
    coalesced_val = torch.zeros(
        unique_key.numel(), dtype=all_val.dtype, device=all_val.device,
    )
    coalesced_val.scatter_add_(0, inverse, all_val)
    new_row = unique_key // N
    new_col = unique_key % N
    return csr_from_triplets(new_row, new_col, coalesced_val, N, N)


def _diag_target(p_t: torch.Tensor) -> torch.Tensor:
    """diag(p_t) as torch sparse CSR."""
    n = p_t.shape[0]
    ids = torch.arange(n, dtype=torch.long, device=p_t.device)
    return csr_from_triplets(ids, ids, p_t[:, 0].clone(), n, n)


def _coarsened_adj_no_diag(coarsening: Coarsening, original) -> torch.Tensor:
    """Q_b^T A Q_b for the binary indicator Q_b, without its diagonal, one entry per pair of supernodes."""
    n = coarsening.n
    A = data_to_torch_csr_adjacency(original)
    row, col, val = extract_triplets(A)
    assignment = torch.as_tensor(coarsening.partition.assignment, dtype=torch.long, device=row.device)
    c_row = assignment[row]
    c_col = assignment[col]
    mask = c_row != c_col                                  # drop self-cluster
    c_row = c_row[mask]; c_col = c_col[mask]; c_val = val[mask]

    key = c_row * n + c_col
    unique_key, inverse = torch.unique(key, return_inverse=True)
    coalesced_val = torch.zeros(
        unique_key.numel(), dtype=c_val.dtype, device=c_val.device,
    )
    coalesced_val.scatter_add_(0, inverse, c_val)
    new_row = unique_key // n
    new_col = unique_key % n
    return csr_from_triplets(new_row, new_col, coalesced_val, n, n)


def d_gw_paper(coarsening: Coarsening, original) -> float:
    """GW discrepancy between (A + A^T) / 2 and n disconnected nodes; reads only G and n."""
    A = data_to_torch_csr_adjacency(original)
    C_s = _A_norm_paper(A)
    p_s = _degree_marginal(A)

    n = coarsening.n
    p_t = torch.ones((n, 1), dtype=p_s.dtype, device=p_s.device) / n
    C_t = _diag_target(p_t)
    return d_gw_proximal(C_s, C_t, p_s, p_t)


def d_gw_op(coarsening: Coarsening, original) -> float:
    """GW discrepancy between S on G and P S Q on the coarsened graph."""
    A = data_to_torch_csr_adjacency(original)
    C_s = build_S(original, mode="self_loop_adj", dtype=torch.float32)
    p_s = _degree_marginal(A)

    # C_t = P S Q, sparse: entry (k, l) is nonzero when supernodes k and l touch.
    P = coarsening.P                                    # (n, N)
    Q = coarsening.Q                                    # (N, n)
    C_t = P @ (C_s @ Q)                                 # (n, n) CSR
    n = C_t.shape[0]

    # Marginal of the coarsened side from the row sums of |PSQ|, with the same
    # uniform floor as _degree_marginal.
    deg_t = C_t.abs().sum(dim=1, keepdim=True).to_dense().squeeze(1)
    p = torch.pow(deg_t.clamp(min=0.0), 0.001)
    total = p.sum()
    p = p / total if total.item() > 0.0 else torch.ones_like(p) / n
    p = 0.9999 * p + 0.0001 / n
    p_t = p.unsqueeze(1)

    return d_gw_proximal(C_s, C_t, p_s, p_t)


def d_gw_struct(coarsening: Coarsening, original) -> float:
    """GW discrepancy between S on G and the same normalisation of A_c, without its diagonal."""
    A = data_to_torch_csr_adjacency(original)
    C_s = build_S(original, mode="self_loop_adj", dtype=torch.float32)
    p_s = _degree_marginal(A)

    A_c = _coarsened_adj_no_diag(coarsening, original)
    # Wrap A_c as a PyG Data so we can reuse `build_S` for the coarsened side.
    row_c, col_c, val_c = extract_triplets(A_c)
    edge_index_c = torch.stack([row_c, col_c], dim=0)
    data_c = PyGData(
        edge_index=edge_index_c, edge_weight=val_c, num_nodes=A_c.shape[0],
    )
    C_t = build_S(data_c, mode="self_loop_adj", dtype=torch.float32)
    p_t = _degree_marginal(A_c)

    return d_gw_proximal(C_s, C_t, p_s, p_t)


def wasserstein_feature_distance(
    coarsening: Coarsening,
    original,
    x: Optional[torch.Tensor] = None,
    *,
    p: int = 2,
    gamma: float = 0.1,
    weight_source: str = "uniform",
    n_sinkhorn_iters: int = 5000,
    tol: float = 1e-4,
    return_plan: bool = False,
    eps: float = 1e-12,
) -> dict:
    """Entropic Wasserstein distance between the original and the coarsened features.

        M[i, j] = ||X[i] - X_c[j]||_2^p,   X_c = P X
        W       = <T, M> - gamma H(T),     H(T) = - sum T (log T - 1)

    with T the entropic optimal plan between the marginals a on the nodes and b
    on the supernodes, found by log-domain Sinkhorn. OTCoarsening (Ma and Chen,
    AAAI 2021) trains on this loss with features from a GNN encoder; we
    evaluate it on P X for any coarsening. Sinkhorn stops once the supernode
    marginal is met within tol, or after n_sinkhorn_iters iterations;
    plan_marginals_err tells how far it got.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    x : torch.Tensor (N,) or (N, F), optional
        Features; None uses original.x.
    p : {1, 2}, default 2
        Exponent of the cost.
    gamma : float, default 0.1
        Entropic regularisation.
    weight_source : {"uniform", "degrees"}, default "uniform"
        a is uniform in both cases. b is uniform under "uniform", and
        proportional to the cluster sizes under "degrees".
    n_sinkhorn_iters : int, default 5000
        Maximum number of Sinkhorn iterations.
    tol : float, default 1e-4
        Sinkhorn stops once ||T^T 1 - b||_1 falls below it.
    return_plan : bool, default False
        If True, also return T.
    eps : float, default 1e-12
        Floor of the marginals before the log.

    Returns
    -------
    dict
        wasserstein : W.
        transport_cost : <T, M>.
        entropy_reg : gamma H(T).
        plan_marginals_err : ||T^T 1 - b||_1, on the supernode side.
        transport_plan : T (N, n) if return_plan, else None.
    """
    if weight_source not in ("uniform", "degrees"):
        raise ValueError(f"weight_source must be 'uniform' or 'degrees', got {weight_source!r}")
    if p not in (1, 2):
        raise ValueError(f"p must be 1 or 2, got {p}")
    if x is None:
        x = original.x
    if x is None:
        raise ValueError("wasserstein_feature_distance needs x (or original.x).")
    if x.ndim == 1:
        x = x.unsqueeze(1)
    dtype = x.dtype

    N, n = coarsening.N, coarsening.n
    X = x.to(dtype)
    P_red = coarsening.P.to(dtype)
    X_c = P_red @ X                                                # (n, d)

    # M[i, j] = ||X[i] - X_c[j]||^p, from ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b,
    # with no (N, n, d) intermediate.
    X_norm2  = (X * X).sum(dim=1)                                  # (N,)
    Xc_norm2 = (X_c * X_c).sum(dim=1)                              # (n,)
    sq = X_norm2.unsqueeze(1) + Xc_norm2.unsqueeze(0) - 2.0 * (X @ X_c.transpose(0, 1))
    sq = sq.clamp(min=0.0)
    M = sq if p == 2 else sq.sqrt()                                # (N, n)

    if weight_source == "uniform":
        a = torch.full((N,), 1.0 / N, dtype=dtype, device=X.device)
        b = torch.full((n,), 1.0 / n, dtype=dtype, device=X.device)
    else:
        assignment = coarsening.partition.assignment
        if not isinstance(assignment, torch.Tensor):
            assignment = torch.as_tensor(assignment, dtype=torch.long)
        cluster_sizes = torch.bincount(assignment.to(torch.long), minlength=n).to(
            dtype=dtype, device=X.device
        )
        b = cluster_sizes / cluster_sizes.sum()
        a = torch.full((N,), 1.0 / N, dtype=dtype, device=X.device)

    # Log-domain Sinkhorn: logsumexp updates of log_u and log_v, stable for any
    # scale of M and gamma.
    log_K = -M / gamma                                              # (N, n)
    log_a = a.clamp(min=eps).log()
    log_b = b.clamp(min=eps).log()
    log_u = torch.zeros(N, dtype=dtype, device=X.device)
    log_v = torch.zeros(n, dtype=dtype, device=X.device)
    for it in range(n_sinkhorn_iters):
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        # The supernode marginal, every 50 steps, ends the loop once met.
        if it % 50 == 49:
            col = torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0) + log_v
            if float((col.exp() - b).abs().sum().item()) < tol:
                break
    log_Pg = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)         # (N, n)
    Pg = log_Pg.exp()

    transport_cost = float((Pg * M).sum().item())
    # gamma H(T), with H(T) = -sum T (log T - 1).
    entropy_reg = float(-gamma * (Pg * (log_Pg - 1.0)).sum().item())
    # The last update fits the node marginal, so the supernode one tells how far it got.
    plan_marginals_err = float((Pg.sum(dim=0) - b).abs().sum().item())
    out = {
        "wasserstein":        transport_cost - entropy_reg,
        "transport_cost":     transport_cost,
        "entropy_reg":        entropy_reg,
        "plan_marginals_err": plan_marginals_err,
        "transport_plan":     Pg if return_plan else None,
    }
    return out


def gromov_wasserstein_distortion(
    coarsening: Coarsening,
    original,
    *,
    pairwise_kind: Optional[str] = None,
    weight_source: str = "uniform",
) -> dict:
    """GW distortion of a partition, from GW coarsening (Taveras, Segarra and Uribe, ICASSP 2026).

        GW^2 = || (S - P C^T S C P^T) * (mu mu^T)^{1/2} ||_F^2

    in the paper's notation, where their P is our binary indicator, C its
    mu-weighted column normalisation and mu a distribution on the nodes.
    P C^T S C P^T replaces each block of S between two clusters by its
    mu-weighted mean. With T[k, l] that mean and mu_k the mass of cluster k,
    Steiner's identity gives

        GW^2 = sum_{i,j} mu_i mu_j S_ij^2 - sum_{k,l} mu_k mu_l T[k, l]^2

    computed in O(nnz(S) + n^2) with no transport solver.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    pairwise_kind : {"adjacency", "normalized_adjacency", "gcn_propagation"}, optional
        S = A, D^{-1/2} A D^{-1/2} or (D+I)^{-1/2} (A+I) (D+I)^{-1/2}. None
        picks the one matching coarsening.laplacian_kind: combinatorial,
        normalized and normalized_self_loop in that order.
    weight_source : {"uniform", "degrees"}, default "uniform"
        mu uniform, or proportional to the degrees.

    Returns
    -------
    dict
        gw_distortion_sq : GW^2.
        gw_distortion : its square root.
        pairwise_kind_used : the S used.
    """
    if weight_source not in ("uniform", "degrees"):
        raise ValueError(f"weight_source must be 'uniform' or 'degrees', got {weight_source!r}")

    if pairwise_kind is None:
        pairwise_kind = {
            "combinatorial":        "adjacency",
            "normalized":           "normalized_adjacency",
            "normalized_self_loop": "gcn_propagation",
        }[coarsening.laplacian_kind]
    if pairwise_kind not in ("adjacency", "normalized_adjacency", "gcn_propagation"):
        raise ValueError(
            f"pairwise_kind must be adjacency / normalized_adjacency / gcn_propagation, "
            f"got {pairwise_kind!r}"
        )

    if pairwise_kind == "adjacency":
        S = data_to_torch_csr_adjacency(original, dtype=torch.float64)
    else:
        mode = "normalized_adj" if pairwise_kind == "normalized_adjacency" else "self_loop_adj"
        S = build_S(original, mode=mode, dtype=torch.float64)

    rows_S, cols_S, vals_S = extract_triplets(S)
    N = coarsening.N
    n = coarsening.n

    assignment = coarsening.partition.assignment
    if not isinstance(assignment, torch.Tensor):
        assignment = torch.as_tensor(assignment, dtype=torch.long)
    else:
        assignment = assignment.to(torch.long)

    if weight_source == "uniform":
        mu = torch.full((N,), 1.0 / N, dtype=torch.float64)
    else:
        A = data_to_torch_csr_adjacency(original, dtype=torch.float64)
        deg = degrees_torch(A)
        total = deg.sum().clamp(min=1e-12)
        mu = deg / total

    # Mass of each cluster, sum_{i in k} mu_i.
    mu_per_cluster = torch.zeros(n, dtype=torch.float64)
    mu_per_cluster.scatter_add_(0, assignment, mu)

    # Block weights mu_k mu_l.
    block_weights = mu_per_cluster.unsqueeze(1) * mu_per_cluster.unsqueeze(0)         # (n, n)

    # Block sums sum_{i in k, j in l} mu_i mu_j S_ij, by scatter_add on the
    # flattened (k, l) index.
    rows_S_long = rows_S.to(torch.long)
    cols_S_long = cols_S.to(torch.long)
    weighted_vals = vals_S.to(torch.float64) * mu[rows_S_long] * mu[cols_S_long]
    flat_kl = assignment[rows_S_long] * n + assignment[cols_S_long]
    block_sum_flat = torch.zeros(n * n, dtype=torch.float64)
    block_sum_flat.scatter_add_(0, flat_kl, weighted_vals)
    block_sum = block_sum_flat.reshape(n, n)

    # GW^2 = sum mu_i mu_j S_ij^2 - sum block_sum^2 / block_weights (Steiner).
    fro_S_sq_weighted = float((weighted_vals * vals_S.to(torch.float64)).sum().item())
    block_weights_safe = block_weights.clamp(min=1e-15)
    aggregated = float((block_sum * block_sum / block_weights_safe).sum().item())

    gw_sq = fro_S_sq_weighted - aggregated
    gw_sq = max(gw_sq, 0.0)                       # clamp float noise
    return {
        "gw_distortion_sq":    gw_sq,
        "gw_distortion":       gw_sq ** 0.5,
        "pairwise_kind_used":  pairwise_kind,
    }
