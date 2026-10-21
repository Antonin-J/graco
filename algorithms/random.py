"""Random partition baseline.

Each node joins a supernode drawn uniformly at random, with no use of the
graph or the features. The default draw leaves some supernodes empty, so
r_actual exceeds r_requested; force_equal_size reaches the target exactly.
Details in the method catalogue of the GraCo paper (Random).
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from graco.algorithms.common import resolve_target
from graco.core.partition import Partition, relabel_contiguous


def coarsen_random(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    seed: int = 14,
    force_equal_size: bool = False,
) -> tuple[Partition, dict]:
    """Assign each node to a supernode drawn at random.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph, read for its number of nodes only.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    seed : int, default 14
        Seed of the torch generator.
    force_equal_size : bool, default False
        If True, shuffle the nodes and cut them into n chunks whose sizes
        differ by at most one. If False, draw each node's supernode uniformly,
        which leaves some supernodes empty.

    Returns
    -------
    partition : Partition
    info : dict
        method : "random".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        force_equal_size : the option used.
        total_time_s : wall time of the call.
        seed : the seed used.
    """
    t0 = time.perf_counter()
    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    gen = torch.Generator().manual_seed(seed)

    if force_equal_size:
        perm = torch.randperm(N, generator=gen)
        # Spread the remainder one node at a time over the first clusters.
        # Putting it all in the last cluster would leave it huge when
        # chunk_size is 1.
        chunk_size = N // n_target
        remainder = N - chunk_size * n_target
        sizes = torch.full((n_target,), chunk_size, dtype=torch.int64)
        sizes[:remainder] += 1
        cluster_ids = torch.repeat_interleave(
            torch.arange(n_target, dtype=torch.int64), sizes
        )
        assignment_torch = torch.empty(N, dtype=torch.int64)
        assignment_torch[perm] = cluster_ids
    else:
        assignment_torch = torch.randint(0, n_target, (N,), generator=gen, dtype=torch.int64)

    # Partition stores np.int64.
    assignment_np = assignment_torch.cpu().numpy()

    # The draw can miss some cluster ids, likely when n_target is close to N.
    assignment_np, n_clusters = relabel_contiguous(assignment_np)
    partition = Partition(assignment_np, n_clusters)

    info = {
        "method": "random",
        "r_requested": r_req,
        "r_actual": 1.0 - n_clusters / N,
        "n_clusters": n_clusters,
        "n_target": n_target,
        "total_time_s": time.perf_counter() - t0,
        "seed": seed,
        "force_equal_size": force_equal_size,
    }
    return partition, info
