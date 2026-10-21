"""METIS graph partitioning (Karypis and Kumar, 1997).

Splits the graph into n parts with few cut edges, and each part becomes a
supernode. We call the reference C library through the pymetis binding.
METIS returns fewer non-empty parts than requested when n is close to N, so
r_actual can exceed r_requested; a warning says so.
An asymmetric input is symmetrized, and an isolated node gets a self-loop.
Details in the method catalogue of the GraCo paper (METIS).
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pymetis

from graco.algorithms.common import resolve_target
from graco.core.partition import Partition, relabel_contiguous


def coarsen_metis(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    seed: int = 14,
) -> tuple[Partition, dict]:
    """Partition the graph with METIS and merge each part into a supernode.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph, read through its edge_index only.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) parts.
    n_clusters : int in [1, N], optional
        Target number of parts.
    seed : int, default 14
        Seed passed to METIS.

    Returns
    -------
    partition : Partition
    info : dict
        method : "metis".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of non-empty parts returned.
        n_target, n_clusters_requested : number of parts asked of METIS.
        n_isolated_self_loops : isolated nodes patched with a self-loop.
        edgecuts : number of cut edges, as METIS reports it.
        total_time_s : wall time of the call.
        seed : the seed used.
    """
    t0 = time.perf_counter()
    N = int(data.num_nodes)
    n_clusters_requested, r_req = resolve_target(N, r, n_clusters)

    # pymetis segfaults on an asymmetric adjacency, so symmetrize first.
    from torch_geometric.utils import to_undirected
    ei_sym = to_undirected(data.edge_index, num_nodes=N)
    if ei_sym.shape[1] != data.edge_index.shape[1]:
        warnings.warn(
            "METIS requires an undirected graph and pymetis segfaults otherwise; "
            "symmetrizing the input. Use preprocess_graph(data, symmetrize=True).",
            stacklevel=2,
        )
    # pymetis reads either a list of neighbour lists or the two CSR arrays, and
    # takes the same route through METIS for both. The lists cost a python
    # object per node and one per edge, so we pass the arrays.
    ei = ei_sym.detach().cpu().numpy()
    keep = ei[0] != ei[1]                       # METIS forbids self-loops
    src = ei[0][keep].astype(np.int64)
    dst = ei[1][keep].astype(np.int64)

    # Isolated nodes get a dummy self-loop. The loader keeps the largest
    # component everywhere but the OGB graphs, and ogbn-products has some.
    degrees = np.bincount(src, minlength=N)
    isolated = np.flatnonzero(degrees == 0)
    n_isolated_self_loops = int(isolated.size)
    if n_isolated_self_loops:
        src = np.concatenate([src, isolated])
        dst = np.concatenate([dst, isolated])
        degrees = np.bincount(src, minlength=N)

    order = np.lexsort((dst, src))
    adjncy = dst[order]
    xadj = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(degrees, out=xadj[1:])

    # pymetis takes the seed through Options, not as a kwarg of part_graph.
    options = pymetis.Options(seed=seed)
    edgecuts, parts = pymetis.part_graph(
        n_clusters_requested,
        adjacency=pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy),
        options=options,
    )
    assignment = np.asarray(parts, dtype=np.int64)

    # METIS may leave some ids of [0, n_requested) unused.
    assignment, n_clusters = relabel_contiguous(assignment)

    if n_isolated_self_loops > 0:
        warnings.warn(
            f"METIS got {n_isolated_self_loops} isolated node(s); patched with "
            f"self-loops. Expected zero if loader enforces LCC.",
            stacklevel=2,
        )
    gap = abs(n_clusters - n_clusters_requested) / max(1, n_clusters_requested)
    if gap > 0.05:
        warnings.warn(
            f"METIS returned {n_clusters} non-empty clusters but {n_clusters_requested} "
            f"were requested ({gap*100:.1f}% gap). r_actual will differ from r_requested.",
            stacklevel=2,
        )

    partition = Partition(assignment, n_clusters)
    info = {
        "method": "metis",
        "r_requested": r_req,
        "r_actual": 1.0 - n_clusters / N,
        "n_clusters": n_clusters,
        "n_target": n_clusters_requested,
        "n_clusters_requested": n_clusters_requested,
        "n_isolated_self_loops": n_isolated_self_loops,
        "edgecuts": int(edgecuts),
        "total_time_s": time.perf_counter() - t0,
        "seed": seed,
    }
    return partition, info
