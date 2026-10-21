"""Greedy matching of local variation, one level at a time.

greedy_match_edges accepts pairs cheapest first, each node in at most one
pair, as Loukas' code does. When this disjoint pass removes less than
stall_threshold of the level budget, an absorption pass, our addition, lets
a free node join a cluster if the enlarged cluster costs at most
absorption_cost_factor times the worst accepted pair, up to
absorption_max_cluster_size nodes. greedy_match_neighborhood is Algorithm 2
of Loukas for closed neighborhoods: a candidate that meets a taken node is
pruned to its free nodes, costed again and put back.
"""

import heapq
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from graco.algorithms.local_variation.cost import spectral_cost_neighborhood


def greedy_match_edges(
    costs: np.ndarray,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    n_target_removed: int,
    *,
    max_take: Optional[int] = None,
    # Absorption pass, run only when the disjoint pass stalls.
    absorption_on_stall: bool = False,
    stall_threshold: float = 0.5,
    absorption_cost_factor: float = 2.0,
    absorption_max_cluster_size: int = 10,
    A_inter: Optional[sp.csr_matrix] = None,
    A_cost: Optional[np.ndarray] = None,
    deg: Optional[np.ndarray] = None,
    dh: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
    spectral: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """Accept pairs cheapest first, then absorb free nodes if the level stalls.

    The absorption pass needs A_inter, A_cost, deg and dh to compute the cost
    of each enlarged cluster, and mu and spectral to read it with the
    projector and the norm of the descent (see spectral_cost_neighborhood).

    Returns
    -------
    coarsening_list : list of np.ndarray
        Node ids of each group to merge.
    info : dict
        n_disjoint_accepted, n_absorption_accepted : pairs and absorptions.
        n_absorption_skipped_too_costly, n_absorption_skipped_cluster_full :
            absorptions refused for cost or size.
        n_removed : nodes removed by the level.
        n_visited : candidates read by the disjoint pass.
        total_cost : sum of the accepted pair costs.
        max_cluster_size : largest group.
        stalled_pass1 : True if the disjoint pass stalled.
    """
    M = costs.shape[0]
    if M == 0:
        return [], _empty_info()

    if absorption_on_stall and (A_inter is None or A_cost is None
                                or deg is None or dh is None):
        raise ValueError("absorption_on_stall=True requires A_inter, A_cost, deg, dh")

    N = int(max(edges_i.max(), edges_j.max())) + 1
    order = np.argsort(costs)

    # Disjoint pass.
    node_cluster = -np.ones(N, dtype=np.int64)
    cluster_members: list[list[int]] = []
    cluster_cost: list[float] = []
    n_removed_disjoint = 0
    n_visited = 0
    limit = max_take if max_take is not None else M

    for idx in order:
        n_visited += 1
        i, j = int(edges_i[idx]), int(edges_j[idx])
        if node_cluster[i] >= 0 or node_cluster[j] >= 0:
            continue
        cid = len(cluster_members)
        node_cluster[i] = cid
        node_cluster[j] = cid
        cluster_members.append([i, j])
        cluster_cost.append(float(costs[idx]))
        n_removed_disjoint += 1
        if n_removed_disjoint >= n_target_removed or len(cluster_members) >= limit:
            break

    n_absorption_accepted = 0
    n_skip_costly = 0
    n_skip_size = 0

    # Absorption pass.
    stalled = (n_removed_disjoint < stall_threshold * n_target_removed) if n_target_removed > 0 else False
    if absorption_on_stall and stalled and cluster_cost:
        max_accepted_cost = max(cluster_cost)
        cost_ceiling = absorption_cost_factor * max_accepted_cost
        remaining_target = n_target_removed - n_removed_disjoint

        for idx in order:
            if n_absorption_accepted >= remaining_target:
                break
            i, j = int(edges_i[idx]), int(edges_j[idx])
            ci, cj = node_cluster[i], node_cluster[j]
            if ci >= 0 and cj >= 0:
                continue  # both assigned (same or different clusters): no-op
            if ci < 0 and cj < 0:
                continue  # both free: the disjoint pass stopped on its budget

            free_node = i if ci < 0 else j
            host_cid = cj if ci < 0 else ci
            host_members = cluster_members[host_cid]

            if len(host_members) >= absorption_max_cluster_size:
                n_skip_size += 1
                continue

            extended = host_members + [free_node]
            nodes_arr = np.array(extended, dtype=np.int64)
            W_sub = np.asarray(A_inter[nodes_arr, :][:, nodes_arr].toarray())
            cost_recomputed = float(
                spectral_cost_neighborhood(nodes_arr, W_sub, A_cost, deg, dh, mu, spectral)
            )

            if cost_recomputed > cost_ceiling:
                n_skip_costly += 1
                continue

            cluster_members[host_cid].append(free_node)
            node_cluster[free_node] = host_cid
            n_absorption_accepted += 1

    # Unassigned nodes stay out; main.py makes them singletons.
    coarsening_list = [np.array(m, dtype=np.int64) for m in cluster_members]

    cluster_sizes = [len(m) for m in cluster_members]
    info = {
        "n_disjoint_accepted": len(cluster_members) - n_absorption_accepted,
        "n_absorption_accepted": n_absorption_accepted,
        "n_absorption_skipped_too_costly": n_skip_costly,
        "n_absorption_skipped_cluster_full": n_skip_size,
        "n_removed": n_removed_disjoint + n_absorption_accepted,
        "n_visited": n_visited,
        "total_cost": sum(cluster_cost),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "stalled_pass1": stalled,
    }
    return coarsening_list, info


def greedy_match_neighborhood(
    costs: np.ndarray,
    neighborhoods: list[np.ndarray],
    n_target_removed: int,
    recost: Callable[[np.ndarray], float],
) -> tuple[list[np.ndarray], dict]:
    """Algorithm 2 of Loukas (2019), within the level budget.

    Candidates come out cheapest first. One that meets a taken node is pruned
    to its free nodes, costed again with recost and put back, unless a single
    node is left. As in his code (contract_variation_linear), a free candidate
    that would overshoot the budget is dropped, and ties go to the candidate
    inserted first. A neighborhood of nc nodes removes nc - 1. If nothing
    fits, the cheapest candidate is taken anyway, so the level still makes
    progress.
    """
    M = len(neighborhoods)
    if M == 0:
        return [], _empty_info()
    if costs.shape[0] != M:
        raise ValueError(f"costs length {costs.shape[0]} != neighborhoods count {M}")

    heap = [(float(costs[k]), k, neighborhoods[k]) for k in range(M)]
    heapq.heapify(heap)
    seq = M
    max_id = max(int(nbh.max()) for nbh in neighborhoods)
    marked = np.zeros(max_id + 1, dtype=bool)

    coarsening_list: list[np.ndarray] = []
    cluster_cost = []
    n_removed = 0
    n_visited = 0
    n_pruned = 0

    while heap and n_removed < n_target_removed:
        cost, _, nbh = heapq.heappop(heap)
        n_visited += 1
        hit = marked[nbh]
        if hit.any():
            rest = nbh[~hit]
            if rest.shape[0] > 1:
                heapq.heappush(heap, (float(recost(rest)), seq, rest))
                seq += 1
                n_pruned += 1
            continue
        nc = nbh.shape[0]
        if n_removed + (nc - 1) > n_target_removed:
            continue
        marked[nbh] = True
        coarsening_list.append(nbh)
        cluster_cost.append(cost)
        n_removed += nc - 1

    if not coarsening_list:
        k = int(np.argmin(costs))
        coarsening_list.append(neighborhoods[k])
        cluster_cost.append(float(costs[k]))
        n_removed += neighborhoods[k].shape[0] - 1

    sizes = [c.shape[0] for c in coarsening_list]
    info = _empty_info()
    info.update({
        "n_disjoint_accepted": len(coarsening_list),
        "n_removed": n_removed,
        "n_visited": n_visited,
        "n_pruned": n_pruned,
        "total_cost": sum(cluster_cost),
        "max_cluster_size": max(sizes) if sizes else 0,
    })
    return coarsening_list, info


def _empty_info() -> dict:
    return {
        "n_disjoint_accepted": 0,
        "n_absorption_accepted": 0,
        "n_absorption_skipped_too_costly": 0,
        "n_absorption_skipped_cluster_full": 0,
        "n_removed": 0,
        "n_visited": 0,
        "total_cost": 0.0,
        "max_cluster_size": 0,
        "stalled_pass1": False,
    }
