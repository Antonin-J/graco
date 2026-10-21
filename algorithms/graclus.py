"""Graclus heavy-edge matching (Dhillon et al., TPAMI 2007).

Each level matches every node with at most one neighbour, along the heaviest
edges, so a level at most halves the graph. We call the matching of PyG
(torch_geometric.nn.pool.graclus, backed by torch-cluster), without the
kernel k-means refinement of the paper, and repeat it until n_target is
reached. The last level can overshoot, so r_actual can exceed r_requested.
Details in the method catalogue of the GraCo paper (Graclus).
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import torch
from torch_geometric.nn.pool import graclus

from graco.algorithms.common import resolve_target
from graco.core.conversions import (
    data_to_torch_csr_adjacency,
    extract_triplets,
)
from graco.core.partition import Partition, relabel_contiguous
from graco.core.q import coarsened_adjacency, degrees_torch


def coarsen_graclus(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    seed: int = 14,
    max_levels: int = 50,
) -> tuple[Partition, dict]:
    """Coarsen by repeated Graclus heavy-edge matching.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with no isolated node, read through its adjacency only.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    seed : int, default 14
        Seeds the global torch generator, from which the torch-cluster kernel
        draws its node visit order. Without it the matching changes at every
        call.
    max_levels : int, default 50
        Maximum number of matching levels.

    Returns
    -------
    partition : Partition
    info : dict
        method : "graclus".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        n_levels : number of matching levels run.
        total_time_s : wall time of the call.
        seed : the seed used.
    """
    t0 = time.perf_counter()
    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    # On the host: the matching is fast there, and every level stays on one device.
    current_A = data_to_torch_csr_adjacency(data).cpu()      # (N, N) sparse CSR
    # The heavy-edge matching leaves an isolated node unclustered.
    n_isolated = int((degrees_torch(current_A) == 0).sum())
    if n_isolated > 0:
        warnings.warn(
            f"Graclus does not handle isolated nodes; got {n_isolated}. "
            f"Use preprocess_graph(data, keep_lcc=True).",
            stacklevel=2,
        )
    # cumulative_assignment[i] = current cluster id of original node i
    cumulative_assignment = torch.arange(N, dtype=torch.int64, device=current_A.device)

    n_cur = N
    n_levels = 0

    # The torch_cluster kernel has no seed argument: it permutes the nodes with
    # the global torch RNG to fix the greedy visit order. Seed it here, inside a
    # fork so we do not disturb the caller's stream.
    fork_devices = [current_A.device] if current_A.device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)

        while n_cur > n_target and n_levels < max_levels:
            row, col, weight = extract_triplets(current_A)
            edge_index = torch.stack([row, col])

            # One cluster id per current node, not contiguous.
            cluster_torch = graclus(edge_index, weight=weight, num_nodes=n_cur)

            unique_ids, new_ids = torch.unique(cluster_torch, return_inverse=True)
            n_next = int(unique_ids.shape[0])

            if n_next == n_cur:
                # Fixed point: no edge left to match.
                break

            cumulative_assignment = new_ids[cumulative_assignment]

            # A_c = Q_b^T A Q_b with the binary Q of this level.
            Q_intermediate = Partition(new_ids.cpu().numpy(), n_next).to_Q_combi()
            current_A = coarsened_adjacency(
                current_A, Q_intermediate, delete_diagonal=True, symmetrize=True
            )

            n_cur = n_next
            n_levels += 1

    assignment = cumulative_assignment.cpu().numpy()
    assignment, n_clusters = relabel_contiguous(assignment)
    partition = Partition(assignment, n_clusters)

    info = {
        "method": "graclus",
        "r_requested": r_req,
        "r_actual": 1.0 - n_clusters / N,
        "n_clusters": n_clusters,
        "n_target": n_target,
        "n_levels": n_levels,
        "total_time_s": time.perf_counter() - t0,
        "seed": seed,
    }
    return partition, info
