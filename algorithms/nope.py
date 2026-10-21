"""NOPE, non-selfish pair merging (Bai et al., ICML 2026).

Greedy pair merging. An edge (p, q) is scored by the interference of its
merge on the neighbourhood, i.e. how much the similarities of the
neighbours to p and to q differ, weighted by the harmonic mean of the two
sizes. The cheapest pair merges first, from a priority queue. Three
variants: "nope" computes the index of the paper; "nope_star" is NOPE*, the
faster relaxation of the paper under local isotropy, as in the released
merge_relax.py; "nope_repo" is the default routine of the released code,
which computes the index on the common neighbours only. Our port purges
dead queue entries, which bounds the memory of deep reductions without
changing any merge.
Reference code: github.com/dazonglian/NOPE-main
Details in the method catalogue of the GraCo paper (NOPE).
"""

from __future__ import annotations

import heapq
import time
from typing import Literal, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from graco.algorithms.common import resolve_target
from graco.core.conversions import data_to_torch_csr_adjacency
from graco.core.laplacian import build_laplacian
from graco.core.partition import Partition, relabel_contiguous
from graco.core.propag import build_S


Variant = Literal["nope", "nope_repo", "nope_star"]
SimilarityKind = Literal[
    "euclidean", "propagation", "L_combi", "L_norm_self_loop", "adjacency",
]

_WEIGHT_EPS = 1e-8


def coarsen_nope(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    variant: Variant = "nope",
    similarity_kind: SimilarityKind = "euclidean",
    heap_purge: bool = True,
    heap_purge_factor: float = 2.0,
    seed: int = 14,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Merge, one pair at a time, the edge whose merge least disturbs its neighbourhood.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Undirected graph with node features data.x.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    variant : {"nope", "nope_repo", "nope_star"}, default "nope"
        Score of a pair. "nope" is the index of the paper, "nope_star" NOPE*,
        its faster relaxation (paper and released code), and "nope_repo" the
        default routine of the released code, restricted to the common
        neighbours.
    similarity_kind : str, default "euclidean"
        Features whose inner products the score compares: "euclidean" (paper)
        uses X, "propagation" S X, "L_combi" (D - A) X, "L_norm_self_loop"
        the self-loop normalized Laplacian times X, and "adjacency" A X.
    heap_purge : bool, default True
        Rebuild the queue without its dead entries whenever it has grown by
        heap_purge_factor. No merge changes. False keeps the released code,
        whose queue keeps its dead entries.
    heap_purge_factor : float, default 2.0
        Growth of the queue allowed between two rebuilds.
    seed : int, default 14
        Unused, since NOPE is deterministic given its input. Kept for the
        common signature.
    progress : bool, default False
        Show a progress bar over the merges.

    Returns
    -------
    partition : Partition
    info : dict
        method : "nope".
        variant, similarity_kind, heap_purge : the options used.
        heap_purges : number of queue rebuilds.
        heap_peak_entries : largest queue size.
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        seed : the seed passed.
        backend : "numpy".
        total_time_s : wall time of the call.
        notes : a summary of the options.
    """
    if variant not in ("nope", "nope_repo", "nope_star"):
        raise ValueError(
            f"variant must be nope / nope_repo / nope_star, got {variant!r}"
        )
    valid_kinds = ("euclidean", "propagation", "L_combi", "L_norm_self_loop", "adjacency")
    if similarity_kind not in valid_kinds:
        raise ValueError(
            f"similarity_kind must be one of {valid_kinds}, got {similarity_kind!r}"
        )

    t_total = time.perf_counter()

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)
    if data.x is None:
        raise ValueError("NOPE needs node features (data.x is None).")
    # No device kwarg: the merge loop scores one pair at a time and reads a
    # scalar back for the queue, so a GPU would only add a synchronisation.
    X_sim = _build_X_sim(data, similarity_kind)                                # (N, d) numpy

    adj = _build_adj_list_of_sets(data, N)
    sizes = np.ones(N, dtype=np.float32)                                       # mu = 1 initially
    sum_sq = _build_sum_sq(X_sim, adj, N) if variant == "nope_repo" else None

    # Union-find for final partition extraction. parent[i] = current super-node id.
    parent = np.arange(N, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    version = np.zeros(N, dtype=np.int64)

    # Score every edge once, as (p, k) with k > p.
    heap: list = []
    for p in range(N):
        neighbors = [k for k in adj[p] if k > p]
        if not neighbors:
            continue
        _score_and_push_batch(
            heap=heap, p=p, neighbors=neighbors,
            X_sim=X_sim, sizes=sizes, sum_sq=sum_sq, adj=adj,
            variant=variant, version=version,
        )

    # An entry touching a merged node goes stale, yet stays in the queue until
    # popped. The purge drops only what the pop-time check would discard, and
    # bounds the queue at heap_purge_factor times its live size.
    def purge(h: list) -> list:
        return [e for e in h
                if alive[e[3]] and alive[e[4]]
                and version[e[3]] == e[1] and version[e[4]] == e[2]]

    purge_at = max(1024, int(heap_purge_factor * len(heap))) if heap_purge else None
    n_purges = 0
    heap_peak = len(heap)

    # Merge loop
    n_cur = N
    iterator = tqdm(total=N - n_target, disable=not progress, desc=f"NOPE[{variant}]")
    while n_cur > n_target and heap:
        entry = heapq.heappop(heap)
        score, tag_p, tag_q, p, q = entry
        # Skip an entry whose endpoints changed since it was pushed.
        if not alive[p] or not alive[q]:
            continue
        if version[p] != tag_p or version[q] != tag_q:
            continue

        # Merge q into p.
        _merge_into(p, q, adj, sizes, sum_sq, X_sim, variant)
        alive[q] = False
        parent[q] = p
        version[p] += 1
        n_cur -= 1
        iterator.update(1)

        # Rescore the pairs of p; the version bump made its old entries stale.
        neighbors = sorted(adj[p])
        if neighbors:
            _score_and_push_batch(
                heap=heap, p=p, neighbors=neighbors,
                X_sim=X_sim, sizes=sizes, sum_sq=sum_sq, adj=adj,
                variant=variant, version=version,
            )

        if len(heap) > heap_peak:
            heap_peak = len(heap)
        if purge_at is not None and len(heap) > purge_at:
            heap = purge(heap)
            heapq.heapify(heap)
            purge_at = max(1024, int(heap_purge_factor * len(heap)))
            n_purges += 1
    iterator.close()

    # Compact union-find to get contiguous cluster ids.
    for i in range(N):
        parent[i] = _find(parent, i)
    assignment, n_actual = relabel_contiguous(parent)
    partition = Partition(assignment, n_actual)

    info = {
        "method": "nope",
        "variant": variant,
        "similarity_kind": similarity_kind,
        "heap_purge": heap_purge,
        "heap_purges": n_purges,
        "heap_peak_entries": heap_peak,
        "r_requested": r_req,
        "r_actual": 1.0 - n_actual / N,
        "n_clusters": int(n_actual),
        "seed": seed,
        "backend": "numpy",
        "total_time_s": time.perf_counter() - t_total,
        "notes": (
            f"NOPE variant={variant!r}, similarity_kind={similarity_kind!r}. "
            f"weight = (s_p s_q) / (s_p + s_q + 1e-8), as in the paper and the "
            f"released code."
        ),
    }
    return partition, info


def _build_X_sim(data, similarity_kind: SimilarityKind) -> np.ndarray:
    """Features of the chosen similarity_kind, as a numpy array for the scalar merge loop."""
    dtype = torch.float32
    X = data.x.to(dtype)
    if similarity_kind == "euclidean":
        # Copy: merging rewrites the super-node row in place.
        return X.cpu().numpy().copy()
    if similarity_kind == "propagation":
        return (build_S(data, mode="self_loop_adj", dtype=dtype) @ X).cpu().numpy()
    if similarity_kind == "adjacency":
        return (data_to_torch_csr_adjacency(data, dtype=dtype) @ X).cpu().numpy()
    if similarity_kind == "L_combi":
        return (build_laplacian(data, "combinatorial", dtype=dtype) @ X).cpu().numpy()
    # L_norm_self_loop
    return (build_laplacian(data, "normalized_self_loop", dtype=dtype) @ X).cpu().numpy()


def _build_adj_list_of_sets(data, N: int) -> list[set]:
    """CPU dict-of-sets adjacency from data.edge_index (undirected)."""
    adj: list[set] = [set() for _ in range(N)]
    ei = data.edge_index.cpu().numpy()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _build_sum_sq(X_sim: np.ndarray, adj: list[set], N: int) -> np.ndarray:
    """sum_sq[i] = sum over k in N_i of <x_i, x_k>^2. Only for variant='nope_repo'."""
    sum_sq = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if not adj[i]:
            continue
        nb = np.fromiter(sorted(adj[i]), dtype=np.int64, count=len(adj[i]))
        dots = X_sim[nb] @ X_sim[i]
        sum_sq[i] = np.sum(dots * dots)
    return sum_sq


def _find(parent: np.ndarray, i: int) -> int:
    """Root of i in the union-find, with path compression."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _merge_into(
    p: int, q: int,
    adj: list[set],
    sizes: np.ndarray,
    sum_sq: Optional[np.ndarray],
    X_sim: np.ndarray,
    variant: Variant,
) -> None:
    """Merge q into p in place: adjacency, sizes, features and, for nope_repo, sum_sq."""
    incremental = variant == "nope_repo" and sum_sq is not None
    if incremental:
        old_p, old_q = set(adj[p]), set(adj[q])
        x_p_old, x_q_old = X_sim[p].copy(), X_sim[q].copy()

    # Neighbors of q, minus p (self-loop after fold).
    new_from_q = adj[q] - {p}
    for k in new_from_q:
        adj[k].discard(q)
        adj[k].add(p)
    adj[p].discard(q)
    adj[p] |= new_from_q
    adj[q] = set()                                                             # dead

    # The super-node carries the size-weighted mean of its members, which is
    # what P X gives downstream. Every later score involving p reads this row,
    # so it must be updated before sum_sq is refreshed. Uses the OLD sizes.
    s_p = float(sizes[p])
    s_q = float(sizes[q])
    X_sim[p] = (X_sim[p] * s_p + X_sim[q] * s_q) / (s_p + s_q)

    sizes[p] = sizes[p] + sizes[q]

    if incremental:
        # Updated in place as the released merge.py does, float32 drift
        # included. The cache can turn negative, as in the released code.
        nb = np.fromiter(adj[p], dtype=np.int64, count=len(adj[p]))
        if nb.size:
            V = X_sim[nb]
            dots_new = V @ X_sim[p]
            upd = dots_new * dots_new
            dots_p = V @ x_p_old
            dots_q = V @ x_q_old
            was_p = np.fromiter((k in old_p for k in nb), dtype=bool, count=nb.size)
            was_q = np.fromiter((k in old_q for k in nb), dtype=bool, count=nb.size)
            upd[was_p] -= dots_p[was_p] * dots_p[was_p]
            upd[was_q] -= dots_q[was_q] * dots_q[was_q]
            sum_sq[nb] += upd
            sum_sq[p] = np.sum(dots_new * dots_new)
        else:
            sum_sq[p] = 0.0


def _score_and_push_batch(
    *,
    heap: list,
    p: int,
    neighbors: list[int],
    X_sim: np.ndarray,
    sizes: np.ndarray,
    sum_sq: Optional[np.ndarray],
    adj: list[set],
    variant: Variant,
    version: np.ndarray,
) -> None:
    """Score (p, q) for every q in neighbors and push it onto the queue."""
    if not neighbors:
        return
    tag_p = int(version[p])
    for q in neighbors:
        tag_q = int(version[q])
        score = _score_pair(p, q, X_sim, sizes, sum_sq, adj, variant)
        heapq.heappush(heap, (score, tag_p, tag_q, p, q))


def _score_pair(
    p: int, q: int,
    X_sim: np.ndarray,
    sizes: np.ndarray,
    sum_sq: Optional[np.ndarray],
    adj: list[set],
    variant: Variant,
) -> float:
    """Score of the pair (p, q) under the chosen variant."""
    s_p = float(sizes[p])
    s_q = float(sizes[q])
    weight = (s_p * s_q) / (s_p + s_q + _WEIGHT_EPS)

    if variant == "nope_star":
        # NOPE*: weight |N_r| ||x_p - x_q||^2, with |N_r| = |N_p u N_q| - 2.
        union_len = len(adj[p] | adj[q])
        n_r = max(union_len - 2, 0)
        if n_r == 0:
            return 0.0
        diff = X_sim[p] - X_sim[q]
        return weight * n_r * float(np.sum(diff * diff))

    if variant == "nope":
        # Index of the paper: weight times the sum over k in N_p u N_q, minus
        # p and q, of (s_kp - s_kq)^2.
        union = (adj[p] | adj[q]) - {p, q}
        if not union:
            return 0.0
        idx = np.fromiter(union, dtype=np.int64, count=len(union))
        X_u = X_sim[idx]                                                        # (|u|, d)
        diff = X_u @ X_sim[p] - X_u @ X_sim[q]
        return weight * float(np.sum(diff * diff))

    # nope_repo: the intersection-based approximation of the released merge.py.
    inter = (adj[p] & adj[q]) - {p, q}
    inter_term = 0.0
    if inter:
        idx = np.fromiter(inter, dtype=np.int64, count=len(inter))
        X_i = X_sim[idx]
        diff = X_i @ X_sim[p] - X_i @ X_sim[q]
        inter_term = float(np.sum(diff * diff))
    assert sum_sq is not None
    return weight * (float(sum_sq[p]) + float(sum_sq[q]) + inter_term)
