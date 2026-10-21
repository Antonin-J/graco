"""Granular-ball graph coarsening, GBGC (Xia et al., IJCAI 2025).

Covers the graph with breadth-first balls of about sqrt(N) nodes, then
splits each ball in two, by breadth-first search from its two nodes of
highest internal degree, while the quality of the children (average degree
plus transitivity) exceeds that of the parent. Each ball becomes a
supernode. The paper takes no target size, so we stop the splits once
n_target balls exist; the initial count is a floor no target can go below.
The authors released no code for GBGC. Their follow-up paper, an
efficient and scalable granular-ball coarsening for large-scale node
classification (Wang et al., arXiv 2026), comes with code
(github.com/Wangwangguanguan/1). It departs
from GBGC on three points, each a switch whose default is GBGC's:
init "bfs" or "metis" (the follow-up starts from METIS), accept "sum" or
"mean", use_transitivity True or False.
Details in the method catalogue of the GraCo paper (GBGC).
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from graco.algorithms.common import resolve_target
from graco.core.partition import Partition, relabel_contiguous
from graco.metrics.structural import granular_ball_quality as _quality


def coarsen_gbgc(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    init: str = "bfs",
    accept: str = "sum",
    use_transitivity: bool = True,
    quality_split_factor: float = 1.0,
    max_split_iter: Optional[int] = None,
    seed: int = 14,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Split granular balls while the split raises their quality, one supernode per ball.

    Give at most one of r and n_clusters. With neither, GBGC runs until no ball
    splits, and the count is whatever comes out.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Undirected graph, read through its edge_index only.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    init : {"bfs", "metis"}, default "bfs"
        Initial balls: breadth-first balls from the nodes of highest degree
        (GBGC), or ceil(sqrt(N)) METIS parts (follow-up).
    accept : {"sum", "mean"}, default "sum"
        A split is accepted if the sum (GBGC) or the mean (follow-up) of the
        two child qualities exceeds the quality of the parent.
    use_transitivity : bool, default True
        Quality as average degree plus transitivity (GBGC), or the average
        degree alone (follow-up). The transitivity is the only part that is
        not linear in the graph size.
    quality_split_factor : float, default 1.0
        Factor on the quality of the parent in the acceptance test, 1 in the
        paper.
    max_split_iter : int, optional
        Cap on the number of splits, N by default.
    seed : int, default 14
        Unused, since GBGC draws nothing at random. Kept for the common
        signature.
    progress : bool, default False
        Show a progress bar over the splits.

    Returns
    -------
    partition : Partition
    info : dict
        method : "gbgc".
        r_requested, r_actual : requested and obtained reduction ratio; the
            requested one is the obtained one when no target is given.
        n_clusters : number of supernodes returned.
        init, accept, use_transitivity : the options used.
        quality_split_factor_used : the factor used.
        n_splits_performed : number of accepted splits.
        seed : the seed passed.
        total_time_s : wall time of the call.
        notes : a summary of the method.
    """
    t_total = time.perf_counter()

    N = int(data.num_nodes)
    if r is None and n_clusters is None:
        # Natural convergence, no target
        n_target = None
        r_req = None
    else:
        n_target, r_req = resolve_target(N, r, n_clusters)

    adj = _build_adj(data, N)
    if init not in {"bfs", "metis"}:
        raise ValueError(f"init must be 'bfs' or 'metis', got {init!r}")
    if accept not in {"sum", "mean"}:
        raise ValueError(f"accept must be 'sum' or 'mean', got {accept!r}")

    deg = np.array([len(s) for s in adj], dtype=np.int64)
    if max_split_iter is None:
        max_split_iter = N                                                   # effectively unlimited

    if n_target is None:
        # No target: run at the given factor.
        assignment, factor_used, n_splits = _run_gbgc(
            adj, deg, N, quality_split_factor, max_split_iter, seed, progress,
            init, accept, use_transitivity,
        )
        n_actual = int(np.unique(assignment).shape[0])
    else:
        # The split budget stops at n_target from below. If the criterion
        # stops first, return what it reached and warn.
        assignment, factor_used, n_splits = _run_gbgc(
            adj, deg, N, quality_split_factor, max_split_iter, seed, progress,
            init, accept, use_transitivity, n_target=n_target,
        )
        n_actual = int(np.unique(assignment).shape[0])

        gap = abs(n_actual - n_target) / max(1, n_target)
        if gap > 0.05:
            warnings.warn(
                f"GBGC's natural quality-driven convergence gave {n_actual} clusters "
                f"but {n_target} were requested ({gap*100:.1f}% gap). r_actual will "
                f"differ from r_requested. Not all reduction ratios are reachable "
                f"under the split criterion in force (accept={accept!r}), and "
                f"init={init!r} fixes the floor on the achievable count.",
                stacklevel=2,
            )

    partition = Partition(assignment, n_actual)

    info = {
        "method": "gbgc",
        "r_requested": r_req if r_req is not None else 1.0 - n_actual / N,
        "r_actual": 1.0 - n_actual / N,
        "n_clusters": int(n_actual),
        "init": init,
        "accept": accept,
        "use_transitivity": bool(use_transitivity),
        "quality_split_factor_used": float(factor_used),
        "n_splits_performed": int(n_splits),
        "seed": seed,
        "total_time_s": time.perf_counter() - t_total,
        "notes": (
            "GBGC top-down granular-ball splitting. `quality_split_factor` is "
            "used as given (paper convention = 1.0) ; a requested target is "
            "reached by stopping the split loop, and cannot go below the "
            "initial ball count."
        ),
    }
    return partition, info


def _run_gbgc(
    adj: list[set], deg: np.ndarray, N: int,
    factor: float, max_split_iter: int, seed: int, progress: bool,
    init: str = "bfs", accept: str = "sum", use_transitivity: bool = True,
    n_target: Optional[int] = None,
) -> tuple[np.ndarray, float, int]:
    """Initial balls, then the split loop, stopped at n_target when one is given."""
    if init == "metis":
        ball_id = _initial_balls_metis(adj, N)
    else:
        ball_id = _initial_balls(adj, deg, N)
    ball_members = _members_from_ball_id(ball_id)

    # One accepted split adds one ball, so the budget is n_target minus the
    # count the init produced, which only the metis init fixes in advance. A
    # budget of 0 means the floor already sits above the target.
    if n_target is not None:
        max_split_iter = min(max_split_iter, max(0, n_target - len(ball_members)))

    queue = deque(ball_members.keys())
    next_ball_id = max(ball_members.keys()) + 1
    n_splits = 0

    iterator = tqdm(total=max_split_iter, disable=not progress, desc="GBGC split")
    while queue and n_splits < max_split_iter:
        bid = queue.popleft()
        members = ball_members[bid]
        if len(members) < 4:                             # too small to split meaningfully
            continue

        q_parent = _quality(adj, members, use_transitivity)
        result = _try_split(adj, members)
        if result is None:
            continue
        set_A, set_B = result
        q_A = _quality(adj, set_A, use_transitivity)
        q_B = _quality(adj, set_B, use_transitivity)
        q_children = q_A + q_B if accept == "sum" else 0.5 * (q_A + q_B)
        if q_children > factor * q_parent:
            new_A = next_ball_id
            new_B = next_ball_id + 1
            next_ball_id += 2
            for v in set_A:
                ball_id[v] = new_A
            for v in set_B:
                ball_id[v] = new_B
            ball_members[new_A] = set_A
            ball_members[new_B] = set_B
            del ball_members[bid]
            queue.append(new_A)
            queue.append(new_B)
            n_splits += 1
            iterator.update(1)
    iterator.close()

    assignment, _ = relabel_contiguous(ball_id)
    return assignment, factor, n_splits


def _initial_balls(
    adj: list[set], deg: np.ndarray, N: int,
) -> np.ndarray:
    """Breadth-first balls of ceil(sqrt(N)) nodes, each grown from the free node of highest degree.

    On a heavy-tailed graph the hubs empty their neighbourhoods first, so
    later balls come out small and far more numerous than sqrt(N).
    """
    target = int(np.ceil(np.sqrt(N)))
    ball_id = -np.ones(N, dtype=np.int64)
    remaining = np.ones(N, dtype=bool)
    next_id = 0

    # A node only ever leaves, so the free node of highest degree is the next
    # one of a single sorted order. Ties go to the lowest index, as a stable
    # sort gives the released code.
    order = np.lexsort((np.arange(N), -deg))
    p = 0
    while True:
        while p < N and not remaining[order[p]]:
            p += 1
        if p == N:
            break
        # BFS from centre until |ball| >= target OR no more reachable nodes
        ball = _bfs_cover(adj, int(order[p]), target, remaining)
        for v in ball:
            ball_id[v] = next_id
            remaining[v] = False
        next_id += 1

    return ball_id


def _initial_balls_metis(adj: list[set], N: int) -> np.ndarray:
    """Exactly ceil(sqrt(N)) initial balls, cut by METIS, as the follow-up does."""
    import pymetis

    target = int(np.ceil(np.sqrt(N)))
    if target >= N:
        return np.arange(N, dtype=np.int64)
    _, parts = pymetis.part_graph(target, adjacency=[sorted(s) for s in adj])
    ball_id, _ = relabel_contiguous(np.asarray(parts, dtype=np.int64))
    return ball_id


def _bfs_cover(
    adj: list[set], center: int, target_size: int, mask: np.ndarray,
) -> set:
    """Breadth-first ball from center over the free nodes, up to target_size nodes."""
    ball = {center}
    frontier = deque([center])
    while frontier and len(ball) < target_size:
        v = frontier.popleft()
        for k in adj[v]:
            if mask[k] and k not in ball:
                ball.add(k)
                frontier.append(k)
                if len(ball) >= target_size:
                    break
    return ball


def _try_split(
    adj: list[set], members: set,
) -> Optional[tuple[set, set]]:
    """Split a ball in two by simultaneous breadth-first search from its two nodes of highest internal degree.

    The degree inside the ball is the one the follow-up code takes; GBGC does
    not say which. Returns None when no split exists.
    """
    if len(members) < 2:
        return None
    # Degree inside the ball, not the degree in the whole graph.
    degs_in = [(len(adj[v] & members), v) for v in members]
    degs_in.sort(reverse=True)
    v_A = degs_in[0][1]
    v_B = None
    for _, cand in degs_in[1:]:
        if cand != v_A:
            v_B = cand
            break
    if v_B is None:
        return None

    # Both searches advance one layer at a time; a node joins the side that
    # reaches it first, A on ties.
    label = {v_A: 'A', v_B: 'B'}
    frontier_A = deque([v_A])
    frontier_B = deque([v_B])

    while frontier_A or frontier_B:
        # One layer of A
        next_A = deque()
        while frontier_A:
            v = frontier_A.popleft()
            for k in adj[v]:
                if k in members and k not in label:
                    label[k] = 'A'
                    next_A.append(k)
        # One layer of B
        next_B = deque()
        while frontier_B:
            v = frontier_B.popleft()
            for k in adj[v]:
                if k in members and k not in label:
                    label[k] = 'B'
                    next_B.append(k)
        frontier_A = next_A
        frontier_B = next_B

    # Members neither search reached, disconnected inside the ball, go to A.
    for v in members:
        if v not in label:
            label[v] = 'A'

    set_A = {v for v, lab in label.items() if lab == 'A'}
    set_B = {v for v, lab in label.items() if lab == 'B'}
    if not set_A or not set_B:
        return None
    return set_A, set_B


def _build_adj(data, N: int) -> list[set]:
    """Undirected adjacency as a list of sets."""
    adj: list[set] = [set() for _ in range(N)]
    ei = data.edge_index.cpu().numpy()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _members_from_ball_id(ball_id: np.ndarray) -> dict[int, set]:
    """Node sets of each ball."""
    m: dict[int, set] = {}
    for v, b in enumerate(ball_id.tolist()):
        m.setdefault(int(b), set()).add(int(v))
    return m
