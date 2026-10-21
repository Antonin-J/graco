"""Scalable topology-preserving graph coarsening, STPGC (Wu et al., ICML 2026).

Removes the nodes and edges a neighbour dominates (their closed
neighbourhood sits inside the neighbour's), which keeps the graph homotopy
equivalent, and cones low-degree nodes to create more dominance. Once this
exact phase stalls, a relaxed phase lets a few neighbours fall outside the
dominator, more of them over time, until the target is met. Each removed
node joins its dominator. With label_guide, as in the released code,
neighbours of the same label are tried first and edges between nodes of
the same label are kept; this reads the labels of the training split, of
the validation split too under "train_val", and of every node under "all".
Five departures from the released code: we
leave out the DropEdge it applies to the coarse graph; we keep the small
components it deletes; we read the labels of the chosen split only, where
the released code reads every label (label_split="all"); a coned node
merges into its dominator; theta1 follows a law in N fitted to their
per-dataset values.
Its two speed-ups, a work-list across the primitives and a single coning,
are our defaults, through incremental and coning_once.
Reference code: github.com/BITNEO/STPGC
Details in the method catalogue of the GraCo paper (STPGC).
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Literal, Optional

import numpy as np
import torch

from graco.algorithms.common import resolve_target
from graco.core.partition import Partition, relabel_contiguous


LabelSplit = Literal["train_val", "train_only", "all"]


def coarsen_stpgc(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    label_guide: bool = True,
    label_split: LabelSplit = "train_val",
    theta1: Optional[int] = None,
    theta2_ratio: float = 0.01,
    incremental: bool = True,
    coning_once: bool = True,
    max_outer_iter: int = 1000,
    seed: int = 14,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Collapse dominated nodes and edges, then relax dominance until the target is met.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph, read through its edge_index, and through y and its split masks
        when label_guide is True.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    label_guide : bool, default True
        Try neighbours of the same label first and keep same-label edges, as
        the paper and the released code do. False uses the topology alone.
    label_split : {"train_val", "train_only", "all"}, default "train_val"
        Labels available to label_guide: training and validation, training
        only, or every node as in the released code. Only "all" reads labels
        outside the training and validation splits; it exists to measure what
        they bring. The two others need data.train_mask.
    theta1 : int, optional
        Degree threshold of the dominance checks, theta1_rule(N) by default.
    theta2_ratio : float, default 0.01
        The relaxed phase loosens dominance once a round removes fewer than
        this fraction of N nodes, 1% in the paper.
    incremental : bool, default True
        Each primitive checks only what the previous one touched, as the
        released code does. False checks every candidate every round.
    coning_once : bool, default True
        Cone once, when the exact phase stalls, then switch to the relaxed
        phase, as the paper's text and the released code do. False cones again
        on every stall.
    max_outer_iter : int, default 1000
        Maximum number of rounds per phase.
    seed : int, default 14
        Unused, since STPGC is deterministic. Kept for the common signature.
    progress : bool, default False
        Print the counts of each round.

    Returns
    -------
    partition : Partition
    info : dict
        method : "stpgc".
        label_guide, label_split, theta2_ratio, incremental, coning_once : the
            options used.
        theta1, theta1_final : threshold at the start and at the end, the
            relaxed phase raising it.
        exact_rounds, relaxed_rounds : rounds of each phase.
        final_tolerance : misses allowed at the end of the relaxed phase.
        target_reached : True if n_clusters <= n_target.
        hit_max_iter : True if max_outer_iter stopped a phase.
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        seed : the seed passed.
        total_time_s : wall time of the call.
        notes : a summary of the method.
    """
    t_total = time.perf_counter()

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    rng = np.random.default_rng(seed)

    labels = _extract_labels(data, N, label_guide, label_split)
    adj = _build_adj_with_self_loops(data, N)
    degree = np.array([len(a) for a in adj], dtype=np.int32)
    alive = np.ones(N, dtype=bool)
    parent = np.arange(N, dtype=np.int64)

    if theta1 is None:
        theta1 = theta1_rule(N)
    theta2 = max(1, int(theta2_ratio * N))

    state = _StpgcState(
        adj=adj,
        degree=degree,
        alive=alive,
        parent=parent,
        labels=labels,
        theta1=int(theta1),
        theta2=theta2,
        n_target=n_target,
        n_alive=N,
        rng=rng,
        progress=progress,
        incremental=bool(incremental),
    )

    # Exact phase: dominance removals without tolerance, until they stall or
    # the target is met.
    exact_rounds = 0
    relaxed_rounds = 0

    for _ in range(max_outer_iter):
        if state.n_alive <= n_target:
            break
        n_before = state.n_alive

        _strong_collapse(state, relaxed=False)
        if state.n_alive <= n_target:
            break

        _edge_collapse(state)
        state.first_sweep = False
        if state.n_alive <= n_target:
            break

        # Both collapses stalled: cone.
        if state.n_alive == n_before:
            _neighborhood_coning(state)
            state.after_coning = True
            if coning_once or state.n_alive == n_before:
                # The exact phase is over.
                break

        exact_rounds += 1

    # Relaxed phase: the tolerance grows by one whenever a round removes fewer
    # than theta2 nodes.
    if state.n_alive > n_target:
        state.tolerance = 0
        no_progress_streak = 0
        for _ in range(max_outer_iter):
            if state.n_alive <= n_target:
                break
            n_before = state.n_alive
            _strong_collapse(state, relaxed=True)
            if state.n_alive <= n_target:
                break
            _edge_collapse(state)
            if n_before - state.n_alive < theta2:
                # Raise theta1 too, so wider neighbourhoods get checked.
                state.tolerance += 1
                state.theta1 += 1
                if state.n_alive == n_before:
                    no_progress_streak += 1
                else:
                    no_progress_streak = 0
                # Stop after max(theta1, 20) raises in a row with no removal.
                if no_progress_streak > max(theta1, 20):
                    break
            else:
                no_progress_streak = 0
            relaxed_rounds += 1

    for i in range(N):
        parent[i] = _find(parent, i)
    assignment, n_actual = relabel_contiguous(parent)
    partition = Partition(assignment, n_actual)

    r_actual = 1.0 - n_actual / N
    # Falling short of the target takes the safety stop of the relaxed phase.
    target_reached = (n_actual <= n_target)
    hit_max_iter = (exact_rounds + relaxed_rounds >= 2 * max_outer_iter - 1)

    info = {
        "method": "stpgc",
        "label_guide": label_guide,
        "label_split": label_split,
        "theta1": int(theta1),
        "theta1_final": int(state.theta1),  # bumped during Approximate phase
        "theta2_ratio": theta2_ratio,
        "incremental": bool(incremental),
        "coning_once": bool(coning_once),
        "exact_rounds": exact_rounds,
        "relaxed_rounds": relaxed_rounds,
        "final_tolerance": state.tolerance,
        "target_reached": bool(target_reached),
        "hit_max_iter": bool(hit_max_iter),
        "r_requested": r_req,
        "r_actual": r_actual,
        "n_clusters": int(n_actual),
        "seed": seed,
        "total_time_s": time.perf_counter() - t_total,
        "notes": (
            "Homotopy equivalence holds in the exact phase only. With label_guide, "
            "the labels of the training split, of validation under 'train_val' "
            "and of every node under 'all', guide the coarsening; only 'all' "
            "reads test labels."
        ),
    }
    return partition, info


class _StpgcState:
    """State shared by the three primitives."""

    __slots__ = (
        "adj", "degree", "alive", "parent", "labels",
        "theta1", "theta2", "n_target", "n_alive",
        "tolerance", "rng", "progress",
        "incremental", "first_sweep", "after_coning",
        "edge_worklist", "strong_worklist",
    )

    def __init__(self, *, adj, degree, alive, parent, labels,
                 theta1, theta2, n_target, n_alive, rng, progress,
                 incremental=False):
        self.adj = adj
        self.degree = degree
        self.alive = alive
        self.parent = parent
        self.labels = labels
        self.theta1 = theta1
        self.theta2 = theta2
        self.n_target = n_target
        self.n_alive = n_alive
        self.tolerance = 0
        self.rng = rng
        self.progress = progress
        # Work-list filter of the released code: off on the first sweep, and
        # off for good once coning has fired.
        self.incremental = incremental
        self.first_sweep = True
        self.after_coning = False
        self.edge_worklist: set = set()
        self.strong_worklist: set = set()


def theta1_rule(N: int) -> int:
    """theta1 as min(100, max(15, round(N^(1/3)))).

    A law in N fitted to the values of the released code: 15 for Cora and
    Citeseer, 25 for dblp, 50 for ogbn-arxiv, 100 for ogbn-products.
    """
    return min(100, max(15, round(N ** (1 / 3))))


def _build_adj_with_self_loops(data, N: int) -> list[set]:
    """Closed neighbourhoods, a list of sets each holding its own node."""
    adj: list[set] = [set() for _ in range(N)]
    for i in range(N):
        adj[i].add(i)
    if data.edge_index is not None:
        ei = data.edge_index.cpu().numpy()
        for u, v in zip(ei[0].tolist(), ei[1].tolist()):
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
    return adj


def _extract_labels(data, N: int, label_guide: bool, label_split: LabelSplit) -> np.ndarray:
    """Labels as int64, -1 outside the chosen split: training (and validation under "train_val"), every node under "all", none without label_guide."""
    if not label_guide:
        return np.full(N, -1, dtype=np.int64)
    if getattr(data, "y", None) is None:
        return np.full(N, -1, dtype=np.int64)

    y = data.y.cpu().numpy()
    if y.ndim > 1:
        y = y.reshape(-1)
    labels = y.astype(np.int64)
    if label_split == "all":
        return labels

    # Keep the labels of the split, so the nodes outside every split stay unlabeled.
    keep = _bool_mask(getattr(data, "train_mask", None), N)
    if keep is None:
        raise ValueError(
            f'label_split="{label_split}" needs data.train_mask; pass '
            'label_guide=False, or label_split="all" to read every label.'
        )
    keep = keep.copy()
    if label_split == "train_val":
        val_mask = _bool_mask(getattr(data, "val_mask", None), N)
        if val_mask is not None:
            keep |= val_mask
    labels[~keep] = -1
    return labels


def _bool_mask(mask, N: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if m.ndim == 2:
        m = m[:, 0]  # multi-split masks : take split 0
    if m.dtype != bool:
        m = m.astype(bool)
    if m.shape[0] != N:
        return None
    return m


def _strong_collapse(state: _StpgcState, *, relaxed: bool) -> None:
    """Remove every node a neighbour dominates, with up to state.tolerance misses when relaxed."""
    tolerance = state.tolerance if relaxed else 0
    N = len(state.adj)
    gated = state.incremental and not state.first_sweep and not state.after_coning

    # Every alive node once, then the requeued ones.
    queue = deque(i for i in range(N) if state.alive[i])
    in_queue = np.zeros(N, dtype=bool)
    in_queue[list(queue)] = True

    while queue:
        if state.n_alive <= state.n_target:
            return
        u = queue.popleft()
        in_queue[u] = False
        if gated and u not in state.strong_worklist:
            continue
        if not state.alive[u]:
            continue
        if state.degree[u] > state.theta1:
            continue

        Nu = state.adj[u]
        # Same-label neighbours first, u excluded.
        ordered = _order_neighbors_by_label(Nu, u, state.labels)

        for v in ordered:
            if v == u or not state.alive[v]:
                continue
            if state.degree[v] + tolerance < state.degree[u]:
                continue
            if _is_subset_with_tolerance(Nu, state.adj[v], tolerance):
                # v dominates u: merge u into v.
                _merge_node_into(state, u, v, in_queue, queue)
                if state.incremental and not state.first_sweep:
                    for w in Nu:
                        if w != u and state.alive[w]:
                            state.edge_worklist.add(w)
                break

    if state.incremental:
        state.strong_worklist = set()


def _is_subset_with_tolerance(Nu: set, Nv: set, tolerance: int) -> bool:
    """True if at most tolerance elements of Nu fall outside Nv."""
    if tolerance == 0:
        return Nu <= Nv
    return len(Nu - Nv) <= tolerance


def _order_neighbors_by_label(Nu: set, u: int, labels: np.ndarray) -> list[int]:
    """Neighbours of u, those sharing its label first."""
    lu = labels[u]
    if lu < 0:
        return [v for v in Nu if v != u]
    same, other = [], []
    for v in Nu:
        if v == u:
            continue
        if labels[v] == lu:
            same.append(v)
        else:
            other.append(v)
    return same + other


def _merge_node_into(state: _StpgcState, u: int, v: int,
                     in_queue: np.ndarray, queue: deque) -> None:
    """Delete u into v, update the degrees and requeue the neighbours."""
    Nu = state.adj[u]
    state.alive[u] = False
    state.parent[u] = v

    # Every alive neighbour of u loses its edge to u. The other edges of u
    # already sit in N[v], since v dominates u.
    for w in Nu:
        if w == u or not state.alive[w]:
            continue
        if u in state.adj[w]:
            state.adj[w].discard(u)
            state.degree[w] -= 1
        if not in_queue[w]:
            queue.append(w)
            in_queue[w] = True
    state.degree[u] = 0
    state.n_alive -= 1


def _edge_collapse(state: _StpgcState) -> None:
    """Remove every edge (x, y) whose closed neighbourhood sits inside that of a third node."""
    N = len(state.adj)
    gated = state.incremental and not state.first_sweep

    # Each undirected edge once.
    seen = set()
    queue: deque[tuple[int, int]] = deque()
    for u in range(N):
        if not state.alive[u]:
            continue
        for v in state.adj[u]:
            if v == u or not state.alive[v]:
                continue
            e = (u, v) if u < v else (v, u)
            if e not in seen:
                seen.add(e)
                queue.append(e)
    in_queue = seen  # alias : set of edges currently in the queue

    while queue:
        e = queue.popleft()
        in_queue.discard(e)
        x, y = e
        if not state.alive[x] or not state.alive[y]:
            continue
        if y not in state.adj[x]:
            continue
        if state.degree[x] + state.degree[y] > 2 * state.theta1:
            continue
        if gated and (x not in state.edge_worklist or y not in state.edge_worklist):
            continue
        # Same-label edges are kept.
        lx, ly = state.labels[x], state.labels[y]
        if lx >= 0 and ly >= 0 and lx == ly:
            continue

        if _is_edge_dominated(state, x, y):
            # Delete the edge (x, y).
            state.adj[x].discard(y)
            state.adj[y].discard(x)
            state.degree[x] -= 1
            state.degree[y] -= 1
            # Requeue the edges around x and y, whose dominance may have changed.
            for w in state.adj[x]:
                if w == x or not state.alive[w]:
                    continue
                e2 = (x, w) if x < w else (w, x)
                if e2 not in in_queue:
                    in_queue.add(e2)
                    queue.append(e2)
                if state.incremental:
                    state.strong_worklist.add(w)
            for w in state.adj[y]:
                if w == y or not state.alive[w]:
                    continue
                e2 = (y, w) if y < w else (w, y)
                if e2 not in in_queue:
                    in_queue.add(e2)
                    queue.append(e2)
                if state.incremental:
                    state.strong_worklist.add(w)

    if state.incremental:
        state.edge_worklist = set()


def _is_edge_dominated(state: _StpgcState, x: int, y: int) -> bool:
    """True if some alive v outside {x, y} has N[x, y] inside N[v]."""
    common = (state.adj[x] & state.adj[y]) - {x, y}
    n_common = len(common)
    if n_common == 0:
        return False
    if n_common == 1:
        return True
    # Highest-degree candidate first, as the released code does.
    for v in sorted(common, key=lambda z: -state.degree[z]):
        if state.degree[v] < n_common:
            return False
        if common - {v} <= state.adj[v]:
            return True
    return False


def _neighborhood_coning(state: _StpgcState) -> None:
    """Add the edges that make a low-degree node dominated, then remove it into its dominator."""
    N = len(state.adj)
    heap = []
    for i in range(N):
        if state.alive[i] and 2 < state.degree[i] <= state.theta1:
            heapq.heappush(heap, (state.degree[i], i))

    while heap:
        if state.n_alive <= state.n_target:
            return
        deg_k, k = heapq.heappop(heap)
        if not state.alive[k] or deg_k != state.degree[k]:
            continue
        if state.degree[k] <= 2 or state.degree[k] > state.theta1:
            continue

        # Alive neighbours of k (excluding k itself).
        others = [n for n in state.adj[k] if n != k and state.alive[n]]
        if len(others) < 2:
            continue

        # Same-label neighbours first.
        others = _order_neighbors_by_label(set(others), k, state.labels)

        found = False
        insert_edges: list[tuple[int, int]] = []
        dominating = -1
        for i in range(len(others)):
            candidate = others[i]
            insert_here: list[tuple[int, int]] = []
            ok = True
            for j in range(len(others)):
                if j == i:
                    continue
                target = others[j]
                if target in state.adj[candidate]:
                    continue
                # A missing edge is accepted if it would be dominated.
                if _is_edge_dominated(state, candidate, target):
                    insert_here.append((candidate, target))
                else:
                    ok = False
                    break
            if ok:
                found = True
                insert_edges = insert_here
                dominating = candidate
                break

        if not found:
            continue

        # Insert the edges, then merge k into its dominator.
        for (a, b) in insert_edges:
            if b in state.adj[a]:
                continue
            state.adj[a].add(b)
            state.adj[b].add(a)
            state.degree[a] += 1
            state.degree[b] += 1
        # Re-queue any affected nodes.
        merged_neighbors = list(state.adj[k])
        _merge_node_into(state, k, dominating, in_queue=np.zeros(N, dtype=bool), queue=deque())
        for w in merged_neighbors:
            if state.alive[w] and 2 < state.degree[w] <= state.theta1:
                heapq.heappush(heap, (state.degree[w], w))
        if state.alive[dominating]:
            heapq.heappush(heap, (state.degree[dominating], dominating))


def _find(parent: np.ndarray, i: int) -> int:
    """Root of i in the union-find, with path compression."""
    root = i
    while parent[root] != root:
        root = parent[root]
    # Compress.
    j = i
    while parent[j] != root:
        nxt = parent[j]
        parent[j] = root
        j = nxt
    return root
