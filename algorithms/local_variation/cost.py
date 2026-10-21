"""Contraction costs of local variation.

quickspectral_costs_all_edges scores every candidate pair, by blocks of
edges, and spectral_cost_neighborhood one candidate set of any size. Both measure the
energy, in the local Laplacian, of what the local projector discards when a
set C of nc nodes is merged:

    L_C   = diag(2 deg - W 1) - W              external weights doubled
    Pibot = I - v o' / (o'v),   v = sqrt(dh[C])
    cost  = || B' Dh^-1/2 L_C Dh^-1/2 B || / (nc - 1),   B = Pibot A_cost[C]

with dh = shifted_degrees(deg, laplacian_kind). The projector is the block on C
of the reduction the descent carries: o = v under the per-level reduction, an
orthogonal projector, and o = mu[C] / v under the global one, mu being the
masses of the current nodes (the sums of the original dh). The norm is
Frobenius, the one of Loukas' code, or spectral, Eq. (6) of his paper. dh = 1
with o = v gives the cost of Loukas' code. For a pair the residual has rank
one, both norms agree, and

    cost(i, j) = |u' Dh^-1/2 L_C Dh^-1/2 u| ||sqrt(dh_j) b_i - sqrt(dh_i) b_j||^2
    u          = (mu_j / sqrt(dh_j), -mu_i / sqrt(dh_i)) / (mu_i + mu_j)

with mu = dh under the per-level reduction. It reads (1/2) (d_i + d_j)
||b_i - b_j||^2 at dh = 1. A_cost is the dense (N, K) basis,
R diag(lambda^-1/2) at the first level.
"""

import numpy as np
from numba import jit


def row_norms_squared(A_cost: np.ndarray) -> np.ndarray:
    """Squared L2 norm of each row of A_cost, reused for every edge."""
    return np.einsum("ij,ij->i", A_cost, A_cost)


def quickspectral_costs_all_edges(
    A_cost: np.ndarray,
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    deg: np.ndarray,
    row_norm2: np.ndarray,
    w: np.ndarray,
    dh: np.ndarray,
    max_memory_gb: float = 5.0,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """Pair cost of every candidate edge, by blocks of edges.

    The rank-one residual lets the costs come out of array arithmetic on two
    (m, K) blocks, where the neighborhood family needs one call per candidate.
    The edges go through in blocks small enough for the temporaries to stay
    under about max_memory_gb; each cost only reads its own edge, so the
    blocks give the same costs as a single pass.

    Parameters
    ----------
    A_cost   : (N, K) float, the spectral basis matrix B = R diag(lambda^{-1/2}).
    edges_i  : (M,) int, source node of each candidate edge.
    edges_j  : (M,) int, target node of each candidate edge (with i < j typically).
    deg      : (N,) float, degree of each node in the current intermediate graph.
    row_norm2: (N,) float, output of row_norms_squared(A_cost).
    w        : (M,) float, weight of each candidate edge.
    dh       : (N,) float, shifted_degrees(deg, laplacian_kind).
    max_memory_gb : float, default 5.0
        Rough cap on the temporaries of one block.
    mu       : (N,) float, optional, the masses of the current nodes. None
               projects orthogonally (per-level reduction), else with the
               masses (global reduction).

    Returns
    -------
    costs : (M,) float, one cost per candidate edge.
    """
    M, K = edges_i.shape[0], A_cost.shape[1]
    # per edge: its two rows of A_cost (2K floats) and about twelve scalar temporaries, 8 bytes each
    block = max(1, int(max_memory_gb * 1e9 // (8 * (2 * K + 12))))
    costs = np.empty(M, dtype=np.float64)
    for start in range(0, M, block):
        e_i, e_j = edges_i[start:start + block], edges_j[start:start + block]
        w_b = w[start:start + block]
        d_i, d_j = deg[e_i], deg[e_j]
        dh_i, dh_j = dh[e_i], dh[e_j]
        dot = np.einsum("mk,mk->m", A_cost[e_i], A_cost[e_j])
        root = np.sqrt(dh_i * dh_j)
        residual2 = dh_j * row_norm2[e_i] + dh_i * row_norm2[e_j] - 2.0 * root * dot
        if mu is None:
            quad = (dh_j * (2.0 * d_i - w_b) / dh_i
                    + dh_i * (2.0 * d_j - w_b) / dh_j
                    + 2.0 * w_b) / (dh_i + dh_j) ** 2
        else:
            m_i, m_j = mu[e_i], mu[e_j]
            quad = (((2.0 * d_i - w_b) * m_j ** 2 + (2.0 * d_j - w_b) * m_i ** 2
                     + 2.0 * w_b * m_i * m_j)
                    / (dh_i * dh_j * (m_i + m_j) ** 2))
        costs[start:start + block] = np.abs(quad) * residual2
    return costs


@jit(nopython=True, cache=True)
def induced_dense(indptr, indices, data, nodes, pos):
    """Dense induced submatrix W[nodes][:, nodes], from the CSR arrays.

    `pos` is an (N,) scratch of -1 the kernel restores on exit, so it is
    allocated once per level instead of once per neighborhood.
    """
    nc = nodes.shape[0]
    W = np.zeros((nc, nc))
    for a in range(nc):
        pos[nodes[a]] = a
    for a in range(nc):
        v = nodes[a]
        for k in range(indptr[v], indptr[v + 1]):
            p = pos[indices[k]]
            if p >= 0:
                W[a, p] = data[k]
    for a in range(nc):
        pos[nodes[a]] = -1
    return W


@jit(nopython=True, cache=True)
def spectral_cost_neighborhood(
    nodes: np.ndarray,
    W_restrict: np.ndarray,
    A_cost: np.ndarray,
    deg: np.ndarray,
    dh: np.ndarray,
    mu=None,
    spectral: bool = False,
) -> float:
    """Generic spectral cost for a subgraph of nc >= 2 nodes (jit per call).

    At dh = 1 and mu = None this is subgraph_cost of Loukas' code, division by
    (nc - 1) included.

    Parameters
    ----------
    nodes      : (nc,) int, the subgraph node indices.
    W_restrict : (nc, nc) dense weight matrix W[nodes, :][:, nodes].
                 Caller materialises it from the sparse adjacency once.
    A_cost     : (N, K) float spectral basis.
    deg        : (N,) float degree of nodes in the current graph.
    dh         : (N,) float, shifted_degrees(deg, laplacian_kind).
    mu         : (N,) float, optional, the masses of the current nodes. None
                 projects orthogonally, else with the masses.
    spectral   : bool, default False
                 Spectral norm instead of Frobenius.
    """
    nc = nodes.shape[0]
    if nc == 1:
        return 1e7  # singleton has no contraction value
    ones = np.ones(nc)
    v = np.sqrt(dh[nodes])
    if mu is None:
        o = v
    else:
        o = mu[nodes] / v
    Pibot = np.eye(nc) - np.outer(v, o) / np.dot(o, v)
    B = Pibot @ A_cost[nodes]                            # (nc, K)
    inv = 1.0 / v
    L_local = ((np.diag(2.0 * deg[nodes] - W_restrict @ ones) - W_restrict)
               * np.outer(inv, inv))                     # (nc, nc)
    if not spectral:
        return np.linalg.norm(B.T @ L_local @ B) / (nc - 1)
    # the nonzero eigenvalues of B' L B are those of L^1/2 B B' L^1/2, so the
    # smaller of the two matrices is decomposed
    if nc >= B.shape[1]:
        return np.linalg.eigvalsh(B.T @ L_local @ B)[-1] / (nc - 1)
    w, U = np.linalg.eigh(L_local)
    Y = (U * np.sqrt(np.maximum(w, 0.0))) @ U.T @ B
    return np.linalg.eigvalsh(Y @ Y.T)[-1] / (nc - 1)
