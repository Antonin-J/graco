"""Local variation coarsening (Loukas, JMLR 2019).

Each level scores candidate contraction sets by how much merging them
perturbs the first K eigenvectors of the Laplacian, then contracts the
cheapest disjoint sets, until n_target supernodes remain. "edges" contracts
pairs, "neighborhood" closed neighborhoods. Our port adds the Laplacian
family of laplacian_kind, two reductions between levels, a per-level budget,
and an absorption pass that lets a free node join a cluster when a level
stalls. A separate paper by the author of this repository documents the
Delta-Laplacian extension and the two reductions at more length.
Reference code: github.com/loukasa/graph-coarsening
Details in the method catalogue of the GraCo paper (Local Variation).
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Callable, Literal, Optional

import numpy as np
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

from graco.algorithms.common import resolve_target
from graco.core.conversions import (
    data_to_scipy_csr_adjacency,
    scipy_csr_to_torch_csr,
)
from graco.core.eig import smallest_eigenpairs
from graco.core.laplacian import (
    LaplacianKind,
    build_laplacian,
    degrees,
    normalize_laplacian_sp,
    shifted_degrees,
)
from graco.core.partition import (
    Partition,
    partition_from_Q,
    sanity_check_partition_Q,
)
from graco.algorithms.local_variation.cost import (
    quickspectral_costs_all_edges,
    row_norms_squared,
    induced_dense,
    spectral_cost_neighborhood,
)
from graco.algorithms.local_variation.matching import (
    greedy_match_edges,
    greedy_match_neighborhood,
)

log = logging.getLogger(__name__)


def coarsen_local_variation(
    data,
    r: float | None = None,
    n_clusters: int | None = None,
    *,
    method: Literal["edges", "neighborhood"] = "edges",
    laplacian_kind: LaplacianKind = "normalized_self_loop",
    reduction: Literal["global", "per_level"] = "global",
    cost_norm: Literal["fro", "spectral"] = "fro",
    K: int = 100,
    n_e_percent: float = 0.1,
    n_e_adaptive: bool = True,
    max_levels: int = 1000,
    max_level_r: float = 0.9999,
    delta: float = 0.0,
    max_memory_gb: float = 5.0,
    seed: int | None = 14,
    on_level: Optional[Callable[[int, np.ndarray, dict], None]] = None,
    progress: bool = False,
    validate: bool = False,
    # Absorption fallback (edges only)
    absorption_on_stall: bool = True,
    stall_threshold: float = 0.5,
    absorption_cost_factor: float = 2.0,
    absorption_max_cluster_size: int = 10,
) -> tuple[Partition, dict]:
    """Coarsen by local variation, one level of contractions at a time.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Connected graph, read through its adjacency only.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    method : {"edges", "neighborhood"}, default "edges"
        Contraction sets: pairs joined by an edge, or closed neighborhoods.
    laplacian_kind : str, default "normalized_self_loop"
        Laplacian of the cost and of the eigenvectors, "combinatorial" being
        Loukas' own; his algorithm is "combinatorial" with "per_level".
    reduction : {"global", "per_level"}, default "global"
        How the basis is reduced from one level to the next, the projector of
        the cost following it. "global" averages with the masses of the
        original nodes, so the levels compose into Q^+, the reduction of the
        Coarsening: it is the only weighted average per level whose product
        over the levels is Q^+.
        "per_level" takes the pseudo-inverse of each level, orthogonal at each
        level; their product is info["P_chain"]. Under "combinatorial",
        "per_level" is Loukas' algorithm.
    cost_norm : {"fro", "spectral"}, default "fro"
        Norm of the cost of a set of more than two nodes: Frobenius, as in
        Loukas' code, or spectral, Eq. (6) of his paper. Both agree on pairs.
    K : int, default 100
        Number of smallest eigenpairs preserved. Loukas' code uses 10.
    n_e_percent : float, default 0.1
        Per-level budget, as a fraction of the nodes a level may remove.
    n_e_adaptive : bool, default True
        If True, the budget is a fraction of the current size, so later
        levels contract less. If False, it is a fraction of N at every level.
    max_levels : int, default 1000
        Maximum number of levels.
    max_level_r : float, default 0.9999
        Largest fraction of the current nodes one level may remove.
    delta : float, default 0.0
        Added to the diagonal of the Laplacian before the eigensolve.
    max_memory_gb : float, default 5.0
        Rough cap on the memory of the edge costs ("edges" only), computed by
        blocks of edges under it. The costs do not depend on the cap.
    seed : int or None, default 14
        Seed of the ARPACK starting vector. None draws it at random, so two
        calls can differ by a few contractions.
    on_level : callable, optional
        Called after each level with (level, assignment, match_info), the
        assignment being the partition reached so far.
    progress : bool, default False
        Show a progress bar over the levels.
    validate : bool, default False
        Check the partition against the composed lifting, in info["sanity"].
    absorption_on_stall : bool, default True
        Run the absorption pass when a level stalls ("edges" only).
    stall_threshold : float, default 0.5
        A level stalls when its disjoint pass removes less than this fraction
        of its budget.
    absorption_cost_factor : float, default 2.0
        An enlarged cluster is accepted if its cost is at most this factor
        times the worst cost accepted at the level.
    absorption_max_cluster_size : int, default 10
        Largest cluster the absorption pass may build.

    Returns
    -------
    partition : Partition
    info : dict
        method : "local_variation".
        variant : the contraction family, "edges" or "neighborhood".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        n_levels : number of levels run.
        K, laplacian_kind, reduction, cost_norm, seed : the options used.
        level_times_s : wall time of each level.
        eig_time_s : wall time of the first eigensolve.
        total_time_s : wall time of the call.
        stalled : True if the descent ended more than 5% above n_target.
        levels : per-level matching counts (see matching.py).
        R : the K eigenvectors of the first level, as a numpy array.
        P_chain : under "per_level", the pseudo-inverses of the levels
            composed over the descent, torch sparse CSR. At Delta = I it is
            the reduction of Loukas. None under "global", whose reduction is
            Q^+.
        sanity : the partition check, only when validate is True.
    """
    t_total = time.perf_counter()

    if method not in ("edges", "neighborhood"):
        raise ValueError(f"method must be edges or neighborhood, got {method!r}")
    if reduction not in ("global", "per_level"):
        raise ValueError(f"reduction must be global or per_level, got {reduction!r}")
    if cost_norm not in ("fro", "spectral"):
        raise ValueError(f"cost_norm must be fro or spectral, got {cost_norm!r}")
    spectral = cost_norm == "spectral"

    A_orig = data_to_scipy_csr_adjacency(data)
    # The cost assumes a single null eigenvector, hence a connected graph.
    from scipy.sparse.csgraph import connected_components
    if connected_components(A_orig, directed=False, return_labels=False) > 1:
        warnings.warn(
            "Local variation expects a connected graph; the input has several components. "
            "The spectral cost may be wrong. Use preprocess_graph(data, keep_lcc=True).",
            stacklevel=2,
        )
    N = A_orig.shape[0]
    n_target, r_req = resolve_target(N, r, n_clusters)
    r = min(r_req, 0.9999)
    # A fixed budget is a fraction of the original size.
    n_e_fixed = max(1, int(n_e_percent * N)) if not n_e_adaptive else None

    # First level: eigenpairs of the original Laplacian.
    t_eig = time.perf_counter()
    L_torch = build_laplacian(data, laplacian_kind)
    vals_t, vecs_t = smallest_eigenpairs(L_torch, K, laplacian_kind, delta=delta,
                                         v0_seed=seed)
    eig_time_s = time.perf_counter() - t_eig

    vals = vals_t.numpy().astype(np.float64)
    R = vecs_t.numpy().astype(np.float64)
    # 1e-6 masks the null eigenvector, whose eigenvalue ARPACK returns as
    # noise, and keeps every useful one.
    mask = vals < 1e-6
    vals_safe = np.where(mask, 1.0, vals)
    lsinv = np.where(mask, 0.0, vals_safe ** -0.5)
    B = R * lsinv[None, :]               # (N, K)
    A_cost = B

    A_inter = A_orig.tocsr().copy()
    # Masses of the current nodes, the sums of the original shifted degrees,
    # read by the global reduction and its projector.
    mu = shifted_degrees(degrees(A_orig), laplacian_kind)
    n_cur = N
    Q_acc: sp.csr_matrix | None = None
    coarsening_chain: list[sp.csr_matrix] = []
    per_level_pinvs: list[sp.csr_matrix] = []
    level_times: list[float] = []
    level_infos: list[dict] = []

    iterator = tqdm(range(1, max_levels + 1), disable=not progress,
                    desc=f"local_variation[{method}]")
    for level in iterator:
        if n_cur <= n_target:
            break
        t_level = time.perf_counter()

        deg = degrees(A_inter)
        D_N = shifted_degrees(deg, laplacian_kind)
        mu_cost = mu if reduction == "global" else None

        n_e_level = max(1, int(n_e_percent * n_cur)) if n_e_adaptive else n_e_fixed
        # Integer room, n_cur - n_target: the float ceil((1 - n_target/n_cur) * n_cur)
        # can round one node too high.
        room = min(n_cur - n_target, int(np.ceil(max_level_r * n_cur)))
        n_target_removed = max(1, min(n_e_level, room))

        if method == "edges":
            edges_i, edges_j = _upper_triangular_edges(A_inter)
            if edges_i.shape[0] == 0:
                log.debug(f"level {level}: no edges left, stopping")
                break
            row_norm2 = row_norms_squared(A_cost)
            w_edges = np.asarray(A_inter[edges_i, edges_j]).ravel()
            costs = quickspectral_costs_all_edges(
                A_cost, edges_i, edges_j, deg, row_norm2, w_edges, D_N,
                max_memory_gb=max_memory_gb, mu=mu_cost,
            )
            coarsening_list, match_info = greedy_match_edges(
                costs, edges_i, edges_j, n_target_removed=n_target_removed,
                absorption_on_stall=absorption_on_stall,
                stall_threshold=stall_threshold,
                absorption_cost_factor=absorption_cost_factor,
                absorption_max_cluster_size=absorption_max_cluster_size,
                A_inter=A_inter,
                A_cost=A_cost,
                deg=deg,
                dh=D_N,
                mu=mu_cost,
                spectral=spectral,
            )
        else:  # neighborhood
            neighborhoods = _build_neighborhoods(A_inter)
            if not neighborhoods:
                log.debug(f"level {level}: no neighborhoods, stopping")
                break
            indptr = A_inter.indptr.astype(np.int64)
            indices = A_inter.indices.astype(np.int64)
            wdata = A_inter.data.astype(np.float64)
            pos = np.full(n_cur, -1, dtype=np.int64)

            def recost(nodes):
                W_sub = induced_dense(indptr, indices, wdata, nodes, pos)
                return spectral_cost_neighborhood(nodes, W_sub, A_cost, deg, D_N,
                                                  mu_cost, spectral)

            costs = np.array([recost(nbh) for nbh in neighborhoods], dtype=np.float64)
            coarsening_list, match_info = greedy_match_neighborhood(
                costs, neighborhoods, n_target_removed, recost,
            )

        if not coarsening_list:
            log.debug(f"level {level}: no admissible contraction, stopping")
            break

        intermediate_Q_combi = _coarsening_list_to_Q(coarsening_list, n_cur)
        n_next = intermediate_Q_combi.shape[1]

        # A_c = Q^T A Q without its diagonal, symmetrized so the eigenvalues
        # stay real. A mask drops the diagonal in one pass, where setdiag
        # would go through the lil format.
        A_c = (intermediate_Q_combi.T @ A_inter @ intermediate_Q_combi).tocoo()
        off = A_c.row != A_c.col
        A_next = sp.csr_matrix((A_c.data[off], (A_c.row[off], A_c.col[off])),
                               shape=A_c.shape)
        A_next.eliminate_zeros()
        A_next = ((A_next + A_next.T) * 0.5).tocsr()

        deg_next = degrees(A_next)
        D_n = shifted_degrees(deg_next, laplacian_kind)

        # Reduction of this level. Both weigh each node by w and average it
        # in its supernode, between the two normalizations: w = mu, the masses,
        # under global, so the steps compose into the pinv of the composed
        # lifting without forming it, and w = D_N under per_level, the pinv of
        # the level's own lifting.
        mu_next = np.asarray(intermediate_Q_combi.T @ mu).ravel()
        if reduction == "global":
            step = (sp.diags(np.sqrt(D_n) / mu_next)
                    @ intermediate_Q_combi.T
                    @ sp.diags(mu / np.sqrt(D_N)))
        else:
            S_lvl = np.asarray(intermediate_Q_combi.T @ D_N).ravel()
            step = (sp.diags(np.sqrt(D_n) / S_lvl)
                    @ intermediate_Q_combi.T
                    @ sp.diags(np.sqrt(D_N)))
        B = np.asarray(step @ B)
        mu = mu_next

        # Next cost basis, from the eigendecomposition of B^T L_next B.
        if n_next > 1:
            L_norm_next = normalize_laplacian_sp(A_next, deg_next, laplacian_kind)
            product = np.asarray((B.T @ L_norm_next @ B))
            d, V = np.linalg.eigh(product)
            mask2 = d < 1e-6  # same threshold as at the first level
            d_safe = np.where(mask2, 1.0, d)
            dinvsqrt = np.where(mask2, 0.0, d_safe ** -0.5)
            A_cost = (B @ V) * dinvsqrt[None, :] @ V.T
        else:
            A_cost = B

        coarsening_chain.append(intermediate_Q_combi)
        if reduction == "per_level":
            per_level_pinvs.append(step.tocsr())

        if on_level is not None:
            # The partition reached so far, composed level by level.
            Q_acc = (intermediate_Q_combi if Q_acc is None
                     else Q_acc @ intermediate_Q_combi)
            acc_coo = Q_acc.tocoo()
            assignment = np.empty(N, dtype=np.int64)
            assignment[acc_coo.row] = acc_coo.col
            on_level(level, assignment, match_info)

        A_inter = A_next
        n_cur = n_next
        level_times.append(time.perf_counter() - t_level)
        match_info["n_target_removed"] = n_target_removed
        match_info["n_cur_after"] = n_cur
        level_infos.append(match_info)
        log.debug(
            f"level {level}: n = {n_cur}, accepted_disjoint={match_info['n_disjoint_accepted']}, "
            f"accepted_absorption={match_info['n_absorption_accepted']}, dt={level_times[-1]:.2f}s"
        )

    Q_lifting_combi = sp.eye(N, format="csr")
    for IQ in coarsening_chain:
        Q_lifting_combi = Q_lifting_combi @ IQ
    P_chain = None
    if reduction == "per_level":
        P_chain = sp.eye(N, format="csr")
        for IP in per_level_pinvs:
            P_chain = IP @ P_chain
        P_chain = scipy_csr_to_torch_csr(P_chain.tocsr())

    partition = partition_from_Q(Q_lifting_combi)
    n_clusters = partition.n_clusters
    r_actual = 1.0 - n_clusters / N
    stalled = (n_clusters > int(np.ceil((1.0 - r) * N)) * 1.05)

    info: dict = {
        "r_requested": r_req,
        "r_actual": r_actual,
        "n_clusters": n_clusters,
        "n_target": n_target,
        "n_levels": len(level_times),
        "K": K,
        "method": "local_variation",
        "variant": method,
        "seed": seed,
        "laplacian_kind": laplacian_kind,
        "reduction": reduction,
        "cost_norm": cost_norm,
        "level_times_s": level_times,
        "eig_time_s": eig_time_s,
        "total_time_s": time.perf_counter() - t_total,
        "stalled": stalled,
        "levels": level_infos,
        "R": R,
        "P_chain": P_chain,
    }

    if stalled:
        warnings.warn(
            f"Local variation stalled: r_requested={r:.3f}, r_actual={r_actual:.3f}, "
            f"n_clusters={n_clusters} (target ~{int(np.ceil((1.0 - r) * N))})",
            stacklevel=2,
        )

    if validate:
        info["sanity"] = sanity_check_partition_Q(partition, Q_lifting_combi)

    return partition, info


def _upper_triangular_edges(A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges (i < j) as two int64 arrays."""
    coo = sp.triu(A, k=1).tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64)


def _build_neighborhoods(A: sp.csr_matrix) -> list[np.ndarray]:
    """Per node, the closed neighborhood [i, neighbors(i)] as a sorted int64 array."""
    N = A.shape[0]
    A = A.tocsr()
    out: list[np.ndarray] = []
    for i in range(N):
        s, e = A.indptr[i], A.indptr[i + 1]
        nbrs = A.indices[s:e]
        if nbrs.shape[0] == 0:
            continue
        nodes = np.unique(np.concatenate([[i], nbrs])).astype(np.int64)
        if nodes.shape[0] >= 2:
            out.append(nodes)
    return out


def _coarsening_list_to_Q(
    coarsening_list: list[np.ndarray], n_cur: int
) -> sp.csr_matrix:
    """Binary lifting of one level: a cluster per accepted group, a singleton per other node."""
    cluster_id = -np.ones(n_cur, dtype=np.int64)
    next_id = 0
    for group in coarsening_list:
        cluster_id[group] = next_id
        next_id += 1
    survivors = np.where(cluster_id < 0)[0]
    for u in survivors:
        cluster_id[u] = next_id
        next_id += 1
    rows = np.arange(n_cur)
    cols = cluster_id
    data = np.ones(n_cur, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_cur, next_id))




