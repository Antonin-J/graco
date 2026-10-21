"""Kcenter coarsening by farthest-point sampling (Gonzalez, TCS 1985).

Picks n pivots by farthest-point sampling on the propagated features
H = S^K X, then assigns every node to its nearest pivot, and each pivot
with its nodes becomes a supernode. Gonzalez picks centres in a metric
space; the propagation and the assignment make it a coarsener. It uses
neither labels nor training, unlike the Kcenter of graph condensation.
Details in the method catalogue of the GraCo paper (Kcenter).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from graco.algorithms.common import (
    nearest_pivot_assignment,
    resolve_target,
    sgc_propagate,
)
from graco.core.conversions import data_to_torch_csr_adjacency
from graco.core.laplacian import LaplacianKind
from graco.core.partition import Partition, relabel_contiguous


def coarsen_kcenter(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    sgc_hops: int = 2,
    laplacian_kind: LaplacianKind = "normalized_self_loop",
    seed: int = 14,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Pick pivots by farthest-point sampling, then assign each node to the nearest.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with node features data.x.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    sgc_hops : int, default 2
        Number K of propagation steps in H = S^K X. 0 uses the raw features.
    laplacian_kind : str, default "normalized_self_loop"
        Recorded in info only. The propagation always uses the self-loop
        normalized adjacency.
    seed : int, default 14
        Seed of the first pivot, drawn at random.
    device : torch.device, optional
        Device of the computation, data.x.device by default.
    progress : bool, default False
        Show a progress bar over the pivots.

    Returns
    -------
    partition : Partition
    info : dict
        method : "kcenter".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of pivots asked for.
        sgc_hops, laplacian_kind : the options used.
        embed_time_s, fps_time_s, voronoi_time_s : wall time of the
            propagation, of the pivot selection and of the assignment.
        total_time_s : wall time of the call.
        device : device of the computation.
        seed : the seed used.
    """
    t_total = time.perf_counter()
    if data.x is None:
        raise ValueError("KCenter needs node features (data.x).")

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    dev = device or data.x.device
    X = data.x.to(dtype=torch.float32, device=dev)

    # H = S^K X.
    t_embed = time.perf_counter()
    if sgc_hops > 0:
        A = data_to_torch_csr_adjacency(data, dtype=torch.float32).to(dev)
        H = sgc_propagate(X, A, sgc_hops)
    else:
        H = X
    embed_time_s = time.perf_counter() - t_embed

    t_fps = time.perf_counter()
    pivots = _farthest_point_greedy(H, n_target, seed, progress=progress)
    fps_time_s = time.perf_counter() - t_fps

    # Two pivots with the same embedding share one cell, so relabel.
    t_voronoi = time.perf_counter()
    assignment_raw = nearest_pivot_assignment(H, pivots).cpu().numpy().astype(np.int64)
    assignment, n_actual = relabel_contiguous(assignment_raw)
    voronoi_time_s = time.perf_counter() - t_voronoi

    partition = Partition(assignment, n_actual)

    info = {
        "method": "kcenter",
        "r_requested": r_req,
        "r_actual": 1.0 - n_actual / N,
        "n_clusters": int(n_actual),
        "n_target": int(n_target),
        "sgc_hops": int(sgc_hops),
        "laplacian_kind": laplacian_kind,
        "embed_time_s": embed_time_s,
        "fps_time_s": fps_time_s,
        "voronoi_time_s": voronoi_time_s,
        "total_time_s": time.perf_counter() - t_total,
        "device": str(dev),
        "seed": seed,
    }
    return partition, info


def _farthest_point_greedy(
    H: torch.Tensor,
    K: int,
    seed: int,
    progress: bool = False,
) -> torch.Tensor:
    """Indices of K rows of H picked by farthest-point sampling in L2."""
    N = H.shape[0]
    K = min(K, N)
    dev = H.device

    g = torch.Generator(device=dev).manual_seed(seed)
    first = int(torch.randint(0, N, (1,), generator=g, device=dev).item())
    pivots = [first]
    min_dists = torch.cdist(H, H[first:first + 1]).squeeze(1)   # (N,)

    iterator = tqdm(range(K - 1), disable=not progress, desc="KCenter FPS")
    for _ in iterator:
        next_pivot = int(torch.argmax(min_dists).item())
        pivots.append(next_pivot)
        new_dists = torch.cdist(H, H[next_pivot:next_pivot + 1]).squeeze(1)
        min_dists = torch.minimum(min_dists, new_dists)
    return torch.tensor(pivots, dtype=torch.long, device=dev)
