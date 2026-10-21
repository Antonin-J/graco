"""Gromov-Wasserstein graph coarsening, GPC and KGPC (Taveras et al., ICASSP 2026).

A graph is a measure network (S, mu), and merging two nodes distorts it by a
closed-form amount in the Gromov-Wasserstein geometry. GPC merges the pair
of least distortion, one pair at a time; KGPC computes the distortion of
every pair and runs k-means on its rows. Our port adds choices of S beyond
the adjacency and the signless Laplacian, a degree-normalized lifting, and
a torch k-means. Both build dense N x N matrices, which limits them to
graphs of moderate size.
Reference code: github.com/ctaveras1999/graph-coarsening
Details in the method catalogue of the GraCo paper (GW GPC / KGPC).
"""

from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from graco.algorithms.common import resolve_target
from graco.core.conversions import (
    csr_from_triplets, data_to_torch_csr_adjacency,
)
from graco.core.laplacian import LaplacianKind, build_laplacian
from graco.core.partition import Partition, relabel_contiguous
from graco.core.propag import build_S as build_propagation_S


SKind = Literal["adjacency", "signless_laplacian", "L_combi",
                "L_norm_self_loop", "propagation"]
PiKind = Literal["combi", "normalized"]
ScMode = Literal["mp_guarantees", "psp"]


def coarsen_gw_gpc(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    S_kind: SKind = "adjacency",
    pi_kind: PiKind = "combi",
    sc_mode: ScMode = "psp",
    mu: Optional[torch.Tensor] = None,
    laplacian_kind: LaplacianKind = "normalized_self_loop",
    seed: int = 14,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Merge, one pair at a time, the two nodes of least distortion.

    Give exactly one of r and n_clusters. GPC costs O(N^4).

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph, read through its adjacency.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    S_kind : str, default "adjacency"
        Matrix S of the measure network: "adjacency" A and
        "signless_laplacian" D + A (paper), "L_combi" D - A,
        "L_norm_self_loop" the self-loop normalized Laplacian, or
        "propagation" the GCN propagation.
    pi_kind : {"combi", "normalized"}, default "combi"
        Lifting inside the distortion: the binary one (paper) or the
        degree-normalized one.
    sc_mode : {"psp", "mp_guarantees"}, default "psp"
        Coarse operator held in info["S_c"]: P S P^T (paper) or P S Q
        (Joly et al., NeurIPS 2024). The partition does not depend on it.
    mu : torch.Tensor, optional
        Measure on the nodes, uniform by default (paper).
    laplacian_kind : str, default "normalized_self_loop"
        With "normalized_self_loop", the degrees of the normalized lifting
        count a self-loop.
    seed : int, default 14
        Unused, since GPC is deterministic. Kept for the common signature.
    device : torch.device, optional
        Device of the computation, that of data.x by default, else the CPU.
    progress : bool, default False
        Show a progress bar over the merges.

    Returns
    -------
    partition : Partition
    info : dict
        method : "gw".
        variant : "gpc".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        S_kind, pi_kind, sc_mode, laplacian_kind : the options used.
        total_time_s : wall time of the call.
        device : device of the computation.
        seed : the seed passed.
        S_c_psp, S_c_psq : P S P^T and P S Q, torch sparse CSR.
        S_c : the one of the two that sc_mode selects.
        notes : a summary of the options.
    """
    t_total = time.perf_counter()

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)
    dev = device or (data.x.device if data.x is not None else torch.device("cpu"))

    S0 = _build_S(data, S_kind, laplacian_kind).to(dev).to(torch.float32)     # (N, N) dense
    if mu is None:
        mu = torch.ones(N, dtype=torch.float32, device=dev) / N
    else:
        mu = mu.to(dev).to(torch.float32)

    # Level-l state (shrinks after each merge).
    S_cur = S0.clone()                                                        # (n_cur, n_cur)
    mu_cur = mu.clone()                                                       # (n_cur,)
    # cluster_of[orig_i] = current-level index of the super-node containing i.
    cluster_of = torch.arange(N, dtype=torch.long, device=dev)
    n_cur = N

    iterator = tqdm(range(N - n_target), disable=not progress, desc="GW GPC")
    for _ in iterator:
        d_cur = _current_row_mass(S_cur, laplacian_kind) if pi_kind == "normalized" else None

        # Find lowest-cost pair (i, j) at this level.
        i, j = _find_best_pair(S_cur, mu_cur, pi_kind, d_cur)

        # Merge j into i, then drop j.
        S_cur = _merge_S_psp(S_cur, mu_cur, i, j)
        mu_cur = _merge_mu(mu_cur, i, j)
        cluster_of[cluster_of == j] = i
        # Compact indices > j down by 1.
        cluster_of = torch.where(cluster_of > j, cluster_of - 1, cluster_of)
        n_cur -= 1

    # Extract Partition from cluster_of.
    assignment, n_actual = relabel_contiguous(cluster_of.cpu().numpy())
    partition = Partition(assignment, n_actual)

    info = _make_info("gpc", partition, S0, r_req, N, S_kind, pi_kind, sc_mode,
                      laplacian_kind, seed, dev, t_total)
    return partition, info


def coarsen_gw_kgpc(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    S_kind: SKind = "adjacency",
    pi_kind: PiKind = "combi",
    sc_mode: ScMode = "psp",
    mu: Optional[torch.Tensor] = None,
    laplacian_kind: LaplacianKind = "normalized_self_loop",
    kmeans_backend: Literal["torch", "sklearn"] = "torch",
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 100,
    seed: int = 14,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Cluster with k-means the rows of the matrix of pair distortions.

    Give exactly one of r and n_clusters. Building the dense N x N matrix of
    distortions costs O(N^3); each of its rows is one point of the k-means.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph, read through its adjacency.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    S_kind : str, default "adjacency"
        Matrix S of the measure network: "adjacency" A and
        "signless_laplacian" D + A (paper), "L_combi" D - A,
        "L_norm_self_loop" the self-loop normalized Laplacian, or
        "propagation" the GCN propagation.
    pi_kind : {"combi", "normalized"}, default "combi"
        Lifting inside the distortion: the binary one (paper) or the
        degree-normalized one.
    sc_mode : {"psp", "mp_guarantees"}, default "psp"
        Coarse operator held in info["S_c"]: P S P^T (paper) or P S Q
        (Joly et al., NeurIPS 2024). The partition does not depend on it.
    mu : torch.Tensor, optional
        Measure on the nodes, uniform by default (paper).
    laplacian_kind : str, default "normalized_self_loop"
        With "normalized_self_loop", the degrees of the normalized lifting
        count a self-loop.
    kmeans_backend : {"torch", "sklearn"}, default "torch"
        "torch" runs k-means++ and Lloyd on the device, with one start;
        "sklearn" is the call of the released code.
    kmeans_n_init : int, default 10
        Number of k-means starts, sklearn only.
    kmeans_max_iter : int, default 100
        Maximum number of Lloyd iterations.
    seed : int, default 14
        Seed of the k-means.
    device : torch.device, optional
        Device of the computation, that of data.x by default, else the CPU.
    progress : bool, default False
        Unused. Kept for the common signature.

    Returns
    -------
    partition : Partition
    info : dict
        method : "gw".
        variant : "kgpc".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        S_kind, pi_kind, sc_mode, laplacian_kind : the options used.
        total_time_s : wall time of the call.
        device : device of the computation.
        seed : the seed passed.
        S_c_psp, S_c_psq : P S P^T and P S Q, torch sparse CSR.
        S_c : the one of the two that sc_mode selects.
        notes : a summary of the options.
        H_time_s, kmeans_time_s : wall time of the distortion matrix and of
            the k-means.
        kmeans_backend, kmeans_n_init : the options used.
    """
    t_total = time.perf_counter()

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)
    dev = device or (data.x.device if data.x is not None else torch.device("cpu"))

    S0 = _build_S(data, S_kind, laplacian_kind).to(dev).to(torch.float32)
    if mu is None:
        mu = torch.ones(N, dtype=torch.float32, device=dev) / N
    else:
        mu = mu.to(dev).to(torch.float32)

    d0 = _current_row_mass(S0, laplacian_kind) if pi_kind == "normalized" else None

    # Distortion of every pair, H of shape (N, N).
    t_H = time.perf_counter()
    H = _build_pairwise_distortion_matrix(S0, mu, pi_kind, d0, progress=progress)
    H_time_s = time.perf_counter() - t_H

    # k-means on rows of H.
    t_km = time.perf_counter()
    if kmeans_backend == "sklearn":
        H_np = H.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=int(n_target), random_state=int(seed),
                        n_init=int(kmeans_n_init), max_iter=int(kmeans_max_iter))
        labels = kmeans.fit_predict(H_np).astype(np.int64)
    elif kmeans_backend == "torch":
        labels = _torch_kmeans(H, K=int(n_target), n_iter=int(kmeans_max_iter),
                                seed=int(seed)).cpu().numpy().astype(np.int64)
    else:
        raise ValueError(
            f"kmeans_backend must be 'sklearn' or 'torch', got {kmeans_backend!r}"
        )
    kmeans_time_s = time.perf_counter() - t_km

    assignment, n_actual = relabel_contiguous(labels)
    partition = Partition(assignment, n_actual)

    info = _make_info("kgpc", partition, S0, r_req, N, S_kind, pi_kind, sc_mode,
                      laplacian_kind, seed, dev, t_total,
                      extra={
                          "H_time_s": H_time_s,
                          "kmeans_time_s": kmeans_time_s,
                          "kmeans_backend": kmeans_backend,
                          # The torch backend makes a single start.
                          "kmeans_n_init": int(kmeans_n_init) if kmeans_backend == "sklearn" else 1,
                      })
    return partition, info


def _torch_kmeans(
    H: torch.Tensor, K: int, n_iter: int = 100, seed: int = 14,
) -> torch.Tensor:
    """K-means on the device, one start, k-means++ init, Lloyd iterations."""
    N = H.shape[0]
    dev = H.device
    g = torch.Generator(device=dev).manual_seed(seed)

    # k-means++ init.
    first = torch.randint(0, N, (1,), generator=g, device=dev)
    centers_idx = [int(first.item())]
    d_min_sq = torch.cdist(H, H[first]).squeeze(1) ** 2                       # (N,)
    for _ in range(K - 1):
        probs = d_min_sq / (d_min_sq.sum() + 1e-30)
        next_idx = int(torch.multinomial(probs, 1, generator=g).item())
        centers_idx.append(next_idx)
        d_new_sq = torch.cdist(H, H[next_idx:next_idx + 1]).squeeze(1) ** 2
        d_min_sq = torch.minimum(d_min_sq, d_new_sq)
    centers = H[torch.tensor(centers_idx, device=dev)].clone()                # (K, N)

    # Lloyd iterations.
    labels_prev = torch.full((N,), -1, dtype=torch.long, device=dev)
    for _ in range(n_iter):
        dists = torch.cdist(H, centers)                                       # (N, K)
        labels = dists.argmin(dim=1)                                          # (N,)
        if torch.equal(labels, labels_prev):
            break
        labels_prev = labels
        # Vectorised mean update via scatter_add.
        centers = torch.zeros_like(centers)
        counts = torch.zeros(K, dtype=H.dtype, device=dev)
        centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, N), H)
        counts.scatter_add_(0, labels, torch.ones(N, dtype=H.dtype, device=dev))
        counts = counts.clamp(min=1)
        centers = centers / counts.unsqueeze(1)
    return labels


def _build_S(data, S_kind: SKind, laplacian_kind: LaplacianKind) -> torch.Tensor:
    """Dense symmetric N x N matrix S of the chosen kind."""
    N = int(data.num_nodes)

    if S_kind == "adjacency":
        A_csr = data_to_torch_csr_adjacency(data, dtype=torch.float32)
        return A_csr.to_dense()

    if S_kind == "signless_laplacian":
        # D + A.
        A_csr = data_to_torch_csr_adjacency(data, dtype=torch.float32)
        A_dense = A_csr.to_dense()
        deg = A_dense.sum(dim=1)
        return A_dense + torch.diag(deg)

    if S_kind == "L_combi":
        L = build_laplacian(data, "combinatorial", dtype=torch.float32)
        return L.to_dense()

    if S_kind == "L_norm_self_loop":
        L = build_laplacian(data, "normalized_self_loop", dtype=torch.float32)
        return L.to_dense()

    if S_kind == "propagation":
        # D_hat^{-1/2} A_hat D_hat^{-1/2}, the GCN propagation.
        S = build_propagation_S(data, mode="self_loop_adj", dtype=torch.float32)
        return S.to_dense()

    raise ValueError(f"Unknown S_kind {S_kind!r}")


def _current_row_mass(S: torch.Tensor, laplacian_kind: LaplacianKind) -> torch.Tensor:
    """Row mass of |S|, plus one for the self-loop kind, as the degrees of the normalized lifting."""
    row_mass = S.abs().sum(dim=1)
    if laplacian_kind == "normalized_self_loop":
        return row_mass + 1.0
    return row_mass


def _pair_distortion_matrix(
    S: torch.Tensor, mu: torch.Tensor,
    pi_kind: PiKind, deg: Optional[torch.Tensor],
) -> torch.Tensor:
    """Distortion of every pair as an N x N matrix, +inf on the diagonal.

    The pair distortion of the released code, for all pairs at once, through
    sum_k mu_k (S[i,k] - S[j,k])^2 = R[i] + R[j] - 2 G[i,j], with
    R = (S * S) mu and G = S diag(mu) S^T.
    """
    N = S.shape[0]
    dev = S.device

    # mu_bar, mu_hat and the scale
    mu_bar = 0.5 * (mu[:, None] + mu[None, :])                                # (N, N)
    mu_hat_sq = mu[:, None] * mu[None, :]                                     # (N, N)
    mu_hat = torch.sqrt(mu_hat_sq)
    scale = mu_hat_sq / (2 * mu_bar ** 2 + 1e-30)

    # Vectorised inner products
    R_mu = (S * S) @ mu                                                       # (N,) R_mu[i] = sum mu_k S[i,k]^2
    G_mu = S @ (mu[:, None] * S.transpose(0, 1))                              # (N, N) G_mu[i,j] = sum mu_k S[i,k] S[j,k]

    diag_S = torch.diag(S)                                                    # (N,) S[i, i]
    a = diag_S[:, None].expand(N, N)                                          # (N, N) a[i,j] = S[i,i]
    b = diag_S[None, :].expand(N, N)                                          # (N, N) b[i,j] = S[j,j]
    c = S                                                                     # c[i,j] = S[i,j]

    # Rows S[i] and S[j] on the nodes k other than i and j.
    if pi_kind == "combi":
        sum_all = R_mu[:, None] + R_mu[None, :] - 2 * G_mu                    # sum over all k
        # Subtract k = i and k = j contributions
        contrib_i = mu[:, None] * (a - c) ** 2                                # mu_i (S[i,i] - S[i,j])^2
        contrib_j = mu[None, :] * (c - b) ** 2                                # mu_j (S[i,j] - S[j,j])^2
        off_sum = sum_all - contrib_i - contrib_j
        dist_others = 2 * mu_bar * off_sum
    else:                                                                     # "normalized"
        alpha = deg[:, None]                                                   # (N, 1)
        beta = deg[None, :]                                                    # (1, N)
        gamma = alpha + beta + 1e-30                                           # (N, N)
        sqrt_a = torch.sqrt(alpha)
        sqrt_b = torch.sqrt(beta)
        sqrt_ab = sqrt_a * sqrt_b

        sum_all = beta * R_mu[:, None] + alpha * R_mu[None, :] - 2 * sqrt_ab * G_mu
        # k = i: mu_i (sqrt(beta) S[i,i] - sqrt(alpha) S[j,i])^2 = mu_i (sqrt(beta) a - sqrt(alpha) c)^2
        # k = j: mu_j (sqrt(beta) S[i,j] - sqrt(alpha) S[j,j])^2 = mu_j (sqrt(beta) c - sqrt(alpha) b)^2
        contrib_i = mu[:, None] * (sqrt_b * a - sqrt_a * c) ** 2
        contrib_j = mu[None, :] * (sqrt_b * c - sqrt_a * b) ** 2
        off_sum = sum_all - contrib_i - contrib_j
        dist_others = (2 * mu_bar / gamma) * off_sum

    # Cross entries, (x_i - y_i)^2 + (x_j - y_j)^2 with S symmetric.
    # x = S[i], y = S[j] ; x_i = S[i,i] = a ; y_i = S[j,i] = c
    # x_j = S[i,j] = c   ; y_j = S[j,j] = b
    dist_cross = (mu[:, None] * (a - c)) ** 2 + (mu[None, :] * (c - b)) ** 2

    # Self weights, (x_i - y_j)^2 = (S[i,i] - S[j,j])^2 = (a - b)^2.
    dist_self = (mu_hat * (a - b)) ** 2 / 2

    H = scale * (dist_others + dist_cross + dist_self)
    # Diag = +inf to exclude "merge with self" from argmin.
    idx = torch.arange(N, device=dev)
    H[idx, idx] = float("inf")
    # Symmetrise to guard against floating-point asymmetry from ops above.
    H = 0.5 * (H + H.transpose(0, 1))
    return H


def _find_best_pair(
    S: torch.Tensor, mu: torch.Tensor,
    pi_kind: PiKind, deg: Optional[torch.Tensor],
) -> tuple[int, int]:
    """Pair (i, j), i < j, of least distortion."""
    H = _pair_distortion_matrix(S, mu, pi_kind, deg)
    N = H.shape[0]
    flat = int(H.argmin().item())
    i, j = flat // N, flat % N
    return (i, j) if i < j else (j, i)


def _build_pairwise_distortion_matrix(
    S: torch.Tensor, mu: torch.Tensor,
    pi_kind: PiKind, deg: Optional[torch.Tensor],
    progress: bool = False,
) -> torch.Tensor:
    """Pair distortions with a zero diagonal, the input of the k-means of KGPC."""
    _ = progress                                                              # kept for API parity
    H = _pair_distortion_matrix(S, mu, pi_kind, deg)
    # Replace +inf diagonal by 0 for k-means input.
    N = H.shape[0]
    idx = torch.arange(N, device=H.device)
    H[idx, idx] = 0.0
    return H


def _merge_S_psp(S: torch.Tensor, mu: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """S after merging j into i: mass-weighted average of rows and columns i and j, then j dropped.

    With t = mu_i / (mu_i + mu_j), row and column i become t (row i) +
    (1 - t) (row j), and S[i, i] becomes (mu_i^2 S[i,i] + mu_i mu_j (S[i,j] +
    S[j,i]) + mu_j^2 S[j,j]) / (mu_i + mu_j)^2. Call it before _merge_mu, on
    the masses before the merge.
    """
    n = S.shape[0]
    mu_i, mu_j = mu[i], mu[j]
    mu_sum = mu_i + mu_j
    theta = mu_i / mu_sum
    S_new = S.clone()
    new_row = theta * S[i] + (1.0 - theta) * S[j]
    new_diag = (mu_i * mu_i * S[i, i] + mu_i * mu_j * (S[i, j] + S[j, i])
                + mu_j * mu_j * S[j, j]) / (mu_sum * mu_sum)
    S_new[i, :] = new_row
    S_new[:, i] = new_row
    S_new[i, i] = new_diag
    idx = torch.arange(n, device=S.device)
    keep = idx != j
    return S_new[keep][:, keep]


def _merge_mu(mu: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Masses after merging j into i."""
    mu_new = mu.clone()
    mu_new[i] = mu[i] + mu[j]
    idx = torch.arange(mu.shape[0], device=mu.device)
    return mu_new[idx != j]


def _compute_S_c(
    S: torch.Tensor, partition: Partition, pi_kind: PiKind,
    laplacian_kind: LaplacianKind,
) -> tuple[torch.Tensor, torch.Tensor]:
    """P S Q and P S P^T of the final partition, as torch sparse CSR."""
    dev = S.device
    N, K = S.shape[0], partition.n_clusters
    assign = torch.from_numpy(partition.assignment).to(dev).to(torch.long)    # (N,)

    if pi_kind == "combi":
        cluster_sizes = torch.zeros(K, dtype=torch.float32, device=dev)
        cluster_sizes.scatter_add_(0, assign, torch.ones(N, dtype=torch.float32, device=dev))
        cluster_sizes = cluster_sizes.clamp(min=1)
        # Q_combi values : 1 at (i, assign[i])
        # P_combi values : 1/|k| at (assign[i], i)
        q_vals = torch.ones(N, dtype=torch.float32, device=dev)
        p_vals = 1.0 / cluster_sizes[assign]                                  # (N,)
    else:
        # Q_norm[i, k] = sqrt(d_hat[i] / d_hat_c[k]) using d_hat^(0)
        d_hat = _current_row_mass(S, laplacian_kind)
        d_hat_c = torch.zeros(K, dtype=torch.float32, device=dev)
        d_hat_c.scatter_add_(0, assign, d_hat)
        d_hat_c = d_hat_c.clamp(min=1e-30)
        q_vals = torch.sqrt(d_hat / d_hat_c[assign])                          # (N,)
        # P_norm = Q_norm^T (Q_norm has orthonormal columns)
        p_vals = q_vals.clone()

    # Q[i, assign[i]] = q_vals[i], one nonzero per row.
    rows_Q = torch.arange(N, dtype=torch.long, device=dev)
    Q_sp = csr_from_triplets(rows_Q, assign, q_vals, N_rows=N, N_cols=K)      # (N, K)

    # P sparse CSR (K, N) : P[assign[i], i] = p_vals[i], one nonzero per col.
    P_sp = csr_from_triplets(assign, rows_Q, p_vals, N_rows=K, N_cols=N)      # (K, N)

    S_c_psq_dense = P_sp @ (S @ Q_sp)                                         # (K, K) dense
    S_c_psp_dense = P_sp @ (S @ P_sp.transpose(0, 1))                         # (K, K) dense
    S_c_psq = _dense_to_sparse_csr(S_c_psq_dense)
    S_c_psp = _dense_to_sparse_csr(S_c_psp_dense)

    return S_c_psq, S_c_psp


def _dense_to_sparse_csr(M: torch.Tensor, threshold: float = 1e-12) -> torch.Tensor:
    """Sparse CSR copy of M, without its entries below threshold."""
    mask = M.abs() > threshold
    rows_c, cols_c = mask.nonzero(as_tuple=True)
    vals = M[mask]
    return csr_from_triplets(rows_c, cols_c, vals, N_rows=M.shape[0], N_cols=M.shape[1])


def _make_info(
    variant: str, partition: Partition, S0: torch.Tensor, r_req: float, N: int,
    S_kind: SKind, pi_kind: PiKind, sc_mode: ScMode, laplacian_kind: LaplacianKind,
    seed: int, dev: torch.device, t_total: float, extra: Optional[dict] = None,
) -> dict:
    """info dict shared by GPC and KGPC."""
    n_actual = partition.n_clusters
    S_c_psq, S_c_psp = _compute_S_c(S0, partition, pi_kind, laplacian_kind)
    info = {
        "method": "gw",
        "variant": variant,
        "r_requested": r_req,
        "r_actual": 1.0 - n_actual / N,
        "n_clusters": int(n_actual),
        "S_kind": S_kind,
        "pi_kind": pi_kind,
        "sc_mode": sc_mode,
        "laplacian_kind": laplacian_kind,
        "total_time_s": time.perf_counter() - t_total,
        "device": str(dev),
        "seed": seed,
        # Both conventions are kept; S_c is the one sc_mode selects.
        "S_c_psq": S_c_psq.detach().cpu(),
        "S_c_psp": S_c_psp.detach().cpu(),
        "S_c": (S_c_psq if sc_mode == "mp_guarantees" else S_c_psp).detach().cpu(),
        "notes": (
            f"GW {variant.upper()} : S_kind={S_kind}, pi_kind={pi_kind}. "
            f"The partition does not depend on sc_mode; info['S_c'] uses "
            f"sc_mode={sc_mode}."
        ),
    }
    if extra:
        info.update(extra)
    return info
