"""Helpers shared by the coarsening algorithms.

resolve_target turns r or n_clusters into a number of supernodes.
sgc_propagate smooths the features as SGC does, for Kcenter, Herding and
ConvMatch. nearest_pivot_assignment is the Voronoi assignment of Kcenter and
Herding, computed by row blocks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from graco.core.conversions import csr_from_triplets, extract_triplets, sparse_add


def resolve_target(
    N: int,
    r: Optional[float],
    n_clusters: Optional[int],
) -> tuple[int, float]:
    """Turn r or n_clusters, exactly one of them, into (n_target, r)."""
    if (r is None) == (n_clusters is None):
        raise ValueError(
            "Specify exactly one of `r` (float in (0,1)) or `n_clusters` (int in [1, N]); "
            f"got r={r}, n_clusters={n_clusters}"
        )
    if r is not None:
        if not (0.0 < r < 1.0):
            raise ValueError(f"r must be in (0, 1), got {r}")
        # Round first, so the float noise of (1 - r) N cannot add a supernode.
        n_target = max(1, int(np.ceil(round((1.0 - r) * N, 6))))
        return n_target, r
    # n_clusters route
    if not (1 <= n_clusters <= N):
        raise ValueError(f"n_clusters must be in [1, {N}], got {n_clusters}")
    return int(n_clusters), 1.0 - n_clusters / N


def sgc_propagate(X: torch.Tensor, A: torch.Tensor, K: int) -> torch.Tensor:
    """Return S^K X with S = D^{-1/2} (A + I) D^{-1/2}, the SGC propagation."""
    if K == 0:
        return X.clone()

    N = X.shape[0]
    dev = X.device
    dtype = A.values().dtype

    eye_ids = torch.arange(N, dtype=torch.long, device=dev)
    I_csr = csr_from_triplets(eye_ids, eye_ids,
                              torch.ones(N, dtype=dtype, device=dev), N, N)
    A_hat = sparse_add(A, I_csr)  # A + I, MKL case in needs_coo

    # keepdim is the only form of sum that CSR supports.
    deg = A_hat.sum(dim=1, keepdim=True).to_dense().squeeze(1)
    dinvsqrt = 1.0 / torch.sqrt(deg.clamp(min=1e-12))

    # Scale rows and columns on the triplets, then rebuild.
    row, col, val = extract_triplets(A_hat)
    s_vals = val * dinvsqrt[row] * dinvsqrt[col]
    S_hat = csr_from_triplets(row, col, s_vals, N, N)

    H = X.clone()
    for _ in range(K):
        H = S_hat @ H
    return H


def nearest_pivot_assignment(
    H: torch.Tensor,
    pivots: torch.Tensor,
    max_elements: int = 2 ** 26,
) -> torch.Tensor:
    """Index in [0, K) of the closest pivot, in L2, for every row of H.

    One torch.cdist would allocate the full (N, K) distance matrix, which grows
    as N^2 at a fixed ratio. We walk H by row blocks of at most max_elements
    entries instead; the result is the same.
    """
    N = H.shape[0]
    P = H[pivots]
    K = P.shape[0]
    chunk = max(1, max_elements // max(1, K))

    out = torch.empty(N, dtype=torch.long, device=H.device)
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        out[start:stop] = torch.cdist(H[start:stop], P).argmin(dim=1)
    return out
