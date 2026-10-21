"""Universal Graph Coarsening, UGC (Kataria et al., NeurIPS 2024).

Each node gets the augmented vector [(1 - alpha) X_i, alpha A_i], hashed
along random projections. A node joins the bin it falls in most often, and
each bin becomes a supernode. A binary search on the bin width reaches the
target. The two halves of the projection are computed apart, so the
dense N x (F + N) augmented matrix is never formed.
Reference code: github.com/katariaMohit/UGC-Universal-Graph-Coarsening
Details in the method catalogue of the GraCo paper (UGC).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from graco.algorithms.common import resolve_target
from graco.core.conversions import data_to_torch_csr_adjacency
from graco.core.partition import Partition, relabel_contiguous


def coarsen_ugc(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    alpha: Optional[float] = None,
    bin_width: Optional[float] = None,
    n_projectors: int = 500,
    projectors_distribution: str = "uniform",
    add_adj_to_features: bool = True,
    seed: int = 14,
    bin_search_tol: float = 0.05,
    bin_search_max_iter: int = 30,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Coarsen by hashing the augmented node vectors into bins.

    Give exactly one of r and n_clusters, unless bin_width is set: the bin
    width then fixes the number of supernodes, and r and n_clusters are
    ignored.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with node features data.x, and labels data.y when alpha is None.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    alpha : float in [0, 1], optional
        Weight of the adjacency against the features in the augmented
        vector. None computes the heterophily factor of the paper from the
        labels of all nodes, test nodes included, as the paper does (details
        in the GraCo paper). The released code fixes 0.7.
    bin_width : float, optional
        Width of the hashing bins. None runs a binary search on it.
    n_projectors : int, default 500
        Number of random projections (paper and released code).
    projectors_distribution : {"uniform", "normal"}, default "uniform"
        Law of the projection entries. "uniform" (released code) draws from
        U(0, 1); "normal" is the law the paper assumes.
    add_adj_to_features : bool, default True
        Append alpha A to (1 - alpha) X in the augmented vector (paper).
    seed : int, default 14
        Seed of the projections and of the bin offsets (released code).
    bin_search_tol : float, default 0.05
        The search stops when the relative gap to the target number of
        supernodes falls below this value.
    bin_search_max_iter : int, default 30
        Maximum number of binary search steps.
    progress : bool, default False
        Print one line per binary search step.

    Returns
    -------
    partition : Partition
    info : dict
        method : "ugc".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_clusters_requested : number asked for, -1 when bin_width is set.
        alpha, bin_width : the values used.
        n_projectors, projectors_distribution,
            add_adj_to_features : the options used.
        n_bin_search_iters : number of binary search steps run.
        total_time_s : wall time of the call.
        seed : the seed used.
    """
    t0 = time.perf_counter()
    if projectors_distribution not in ("uniform", "normal"):
        raise ValueError(f"projectors_distribution must be uniform / normal, got {projectors_distribution!r}")
    if data.x is None:
        raise ValueError("UGC needs node features (data.x).")

    N = int(data.num_nodes)
    if bin_width is None:
        n_clusters_requested, r_req = resolve_target(N, r, n_clusters)
    else:
        n_clusters_requested = -1                            # unused
        r_req = r if r is not None else float("nan")

    # Heterophily factor: the fraction of edges whose endpoints have different
    # labels.
    if alpha is None:
        if getattr(data, "y", None) is None:
            raise ValueError("alpha=None requires data.y to auto-compute the heterophily factor.")
        src, dst = data.edge_index[0], data.edge_index[1]
        alpha = float((data.y[src] != data.y[dst]).float().mean().item())

    torch.manual_seed(seed)
    X = data.x.to(torch.float32)
    dev = X.device
    F = X.shape[1]
    A = (
        data_to_torch_csr_adjacency(data, dtype=torch.float32).to(dev)
        if add_adj_to_features else None
    )
    feat_size = F + (N if add_adj_to_features else 0)

    # Projection matrix Wl of shape (n_proj, F + N). The generator must live on
    # the device of the tensor, or torch refuses the draw on GPU.
    g = torch.Generator(device=dev).manual_seed(seed)
    if projectors_distribution == "normal":
        Wl = torch.randn(n_projectors, feat_size, generator=g, dtype=torch.float32, device=dev)
    else:
        Wl = torch.empty(n_projectors, feat_size, dtype=torch.float32, device=dev)
        Wl.uniform_(0, 1, generator=g)

    # (1 - alpha) X Wl_X^T + alpha A Wl_A^T, so the (N, F + N) matrix is never formed.
    Wl_X = Wl[:, :F]                                          # (n_proj, F)
    Bin = (1.0 - alpha) * (X @ Wl_X.transpose(0, 1))          # (N, n_proj)
    if add_adj_to_features:
        Wl_A = Wl[:, F:]                                       # (n_proj, N)
        Bin = Bin + alpha * (A @ Wl_A.transpose(0, 1))

    # Offsets b ~ U(-bw, bw) as in the released code, drawn once and rescaled
    # for each width, so the search works on a fixed objective.
    bias_g = torch.Generator(device=dev).manual_seed(seed + 1)
    bias_unit = torch.rand(n_projectors, generator=bias_g, dtype=torch.float32, device=dev) * 2 - 1

    def _partition_for_bw(bw: float) -> tuple[np.ndarray, int]:
        """Partition and its size for one bin width."""
        bias_vec = bias_unit * bw
        hashed = torch.floor((Bin + bias_vec.unsqueeze(0)) / bw)
        cluster_codes = torch.mode(hashed, dim=1).values.to(torch.int64)
        assignment, n = relabel_contiguous(cluster_codes.cpu().numpy())
        return assignment, n

    n_search_iters = 0
    if bin_width is not None:
        assignment, n_clusters = _partition_for_bw(float(bin_width))
        bw_used = float(bin_width)
    else:
        bw_used, assignment, n_clusters, n_search_iters = _bin_width_search(
            n_clusters_requested, _partition_for_bw,
            tol=bin_search_tol, max_iter=bin_search_max_iter, progress=progress,
        )

    if n_clusters < 1:
        raise RuntimeError("UGC produced 0 clusters: bin_width is likely too large.")

    partition = Partition(assignment, n_clusters)
    info = {
        "method": "ugc",
        "r_requested": r_req,
        "r_actual": 1.0 - n_clusters / N,
        "n_clusters": n_clusters,
        "n_clusters_requested": n_clusters_requested,
        "alpha": float(alpha),
        "bin_width": bw_used,
        "n_projectors": int(n_projectors),
        "projectors_distribution": projectors_distribution,
        "add_adj_to_features": bool(add_adj_to_features),
        "n_bin_search_iters": int(n_search_iters),
        "total_time_s": time.perf_counter() - t0,
        "seed": seed,
    }
    return partition, info


def _bin_width_search(
    n_clusters_target: int,
    partition_fn,
    *,
    tol: float,
    max_iter: int,
    progress: bool,
) -> tuple[float, np.ndarray, int, int]:
    """Binary search on the bin width, on a log scale, to reach the target number of bins."""
    # A width of 1e-6 gives many bins, a width of 10 a single one.
    lo, hi = 1e-6, 10.0
    assignment_lo, n_lo = partition_fn(lo)
    assignment_hi, n_hi = partition_fn(hi)
    if n_lo <= n_clusters_target:
        return lo, assignment_lo, n_lo, 0
    if n_hi >= n_clusters_target:
        return hi, assignment_hi, n_hi, 0

    best = (lo, assignment_lo, n_lo)
    best_gap = abs(n_lo - n_clusters_target)
    for it in range(max_iter):
        mid = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))     # midpoint on a log scale
        assignment, n = partition_fn(mid)
        gap = abs(n - n_clusters_target)
        if gap < best_gap:
            best_gap = gap
            best = (mid, assignment, n)
        if progress:
            print(f"[UGC bin_search iter {it:2d}] bw={mid:.4e}, n={n} (target {n_clusters_target})", flush=True)
        if best_gap / max(1, n_clusters_target) < tol:
            return mid, assignment, n, it + 1
        if n > n_clusters_target:                                # too many bins, widen them
            lo = mid
        else:
            hi = mid
    return best[0], best[1], best[2], max_iter
