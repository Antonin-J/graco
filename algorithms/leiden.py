"""Leiden community detection (Traag et al., Sci. Rep. 2019).

Leiden maximizes the modularity of a partition at a given resolution, and
each community becomes a supernode. We call igraph and leidenalg with the
Reichardt-Bornholdt modularity. Leiden takes a resolution and no number of
communities, so we run a binary search on the resolution, on a log scale,
until the count is within n_clusters_tol of the target. Otherwise the
closest partition met is returned.
Details in the method catalogue of the GraCo paper (Leiden).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from graco.algorithms.common import resolve_target
from graco.core.partition import Partition, relabel_contiguous


def coarsen_leiden(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    seed: int = 14,
    resolution: Optional[float] = None,
    n_clusters_tol: int = 1,
    max_search_iter: int = 50,
    time_budget_s: Optional[float] = None,
) -> tuple[Partition, dict]:
    """Partition the graph into Leiden communities, one supernode each.

    Give exactly one of r and n_clusters. With resolution set, the search is
    skipped and the number of communities is whatever Leiden returns.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph, read through its edge_index and its edge_weight if present.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    seed : int, default 14
        Seed passed to leidenalg.
    resolution : float, optional
        Run Leiden once at this resolution, with no search.
    n_clusters_tol : int, default 1
        The search stops as soon as the number of communities is within this
        many of the target. Leiden rarely hits an arbitrary count exactly.
    max_search_iter : int, default 50
        Maximum number of binary search steps.
    time_budget_s : float, optional
        Wall-clock budget of the search. It returns the closest partition met
        so far when the next call would overrun.

    Returns
    -------
    partition : Partition
    info : dict
        method : "leiden".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        final_resolution : resolution of the returned partition.
        final_call_time_s : wall time of the call that produced it.
        n_search_iters : number of Leiden calls in the search.
        search_stopped_on_budget : True if time_budget_s stopped the search.
        n_clusters_tol : the tolerance used.
        target_match_exact : True if n_clusters equals n_target.
        target_match_within_tol : True if n_clusters is within the tolerance.
        total_time_s : wall time of the call.
        seed : the seed used.
    """
    t0 = time.perf_counter()
    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    # igraph and leidenalg are GPL and optional, so they are imported here and
    # the package imports without them.
    try:
        import igraph as ig
        import leidenalg  # noqa: F401
    except ImportError as e:
        raise ImportError('leiden needs igraph and leidenalg: pip install -e ".[leiden]"') from e

    # PyG stores both (i, j) and (j, i); keep i < j so igraph sees each edge once.
    ei = data.edge_index
    src = ei[0]
    dst = ei[1]
    upper = src < dst
    edges = torch.stack([src[upper], dst[upper]], dim=1).cpu().numpy()
    ew = getattr(data, "edge_weight", None)
    weights = ew[upper].cpu().numpy() if ew is not None else np.ones(len(edges))

    G = ig.Graph(n=N, edges=edges, directed=False)
    G.es["weight"] = weights

    # The search reruns Leiden in full, so total_time_s counts every call;
    # final_call_time_s is the one that produced the returned partition.
    if resolution is not None:
        t_call = time.perf_counter()
        partition_obj = _leiden_at_resolution(G, resolution, seed)
        final_call_time = time.perf_counter() - t_call
        n_clusters_raw = len(partition_obj)
        final_resolution = resolution
        n_search_iters = 0
        stopped_on_budget = False
        target_match_exact = (n_clusters_raw == n_target)
    else:
        (partition_obj, final_resolution, n_search_iters, final_call_time,
         stopped_on_budget) = _leiden_binary_search(
            G, n_target, max_search_iter, seed, n_clusters_tol, time_budget_s
        )
        n_clusters_raw = len(partition_obj)
        target_match_exact = (n_clusters_raw == n_target)

    assignment = np.zeros(N, dtype=np.int64)
    for cid, nodes in enumerate(partition_obj):
        assignment[nodes] = cid
    assignment, n_clusters = relabel_contiguous(assignment)
    partition = Partition(assignment, n_clusters)

    info = {
        "method": "leiden",
        "r_requested": r_req,
        "r_actual": 1.0 - n_clusters / N,
        "n_clusters": n_clusters,
        "n_target": n_target,
        "total_time_s": time.perf_counter() - t0,
        "seed": seed,
        "final_resolution": final_resolution,
        "final_call_time_s": final_call_time,
        "n_search_iters": n_search_iters,
        "search_stopped_on_budget": stopped_on_budget,
        "n_clusters_tol": int(n_clusters_tol),
        "target_match_exact": target_match_exact,
        "target_match_within_tol": abs(n_clusters - n_target) <= n_clusters_tol,
    }
    return partition, info


def _leiden_at_resolution(G: ig.Graph, resolution: float, seed: int):
    """One leidenalg call at the given resolution."""
    import leidenalg
    return leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights=G.es["weight"] if G.ecount() > 0 else None,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed,
    )


def _leiden_binary_search(
    G: ig.Graph, n_target: int, max_iter: int, seed: int, n_clusters_tol: int = 1,
    time_budget_s: Optional[float] = None,
) -> tuple[object, float, int, float, bool]:
    """Binary search on the resolution, on a log scale over [1e-3, 1e4], to reach n_target."""
    lo, hi = 1e-3, 1e4
    best_partition = None
    best_diff = float("inf")
    best_res = 1.0
    best_time = 0.0
    t_start = time.perf_counter()
    last_time = 0.0

    for it in range(1, max_iter + 1):
        # Stop if the next call, priced at the last one, would overrun the budget.
        if (time_budget_s is not None and best_partition is not None
                and time.perf_counter() - t_start + last_time > time_budget_s):
            return best_partition, best_res, it - 1, best_time, True
        mid = np.sqrt(lo * hi)              # midpoint on a log scale
        t_call = time.perf_counter()
        part = _leiden_at_resolution(G, mid, seed)
        t_call = time.perf_counter() - t_call
        last_time = t_call
        k = len(part)
        diff = abs(k - n_target)

        if diff < best_diff:
            best_diff = diff
            best_partition = part
            best_res = mid
            best_time = t_call

        if diff <= n_clusters_tol:
            return best_partition, best_res, it, best_time, False
        elif k < n_target:
            lo = mid                        # more clusters need a higher resolution
        else:
            hi = mid

        if hi / lo < 1.001:                 # resolution range converged
            return best_partition, best_res, it, best_time, False

    return best_partition, best_res, max_iter, best_time, False
