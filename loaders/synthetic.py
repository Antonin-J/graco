"""Synthetic graph generators, built with networkx.

make_sbm, make_geometric, make_erdos_renyi, make_barabasi_albert and
make_watts_strogatz return an undirected PyG Data without self-loops or edge
weights. A disconnected draw is connected by one edge from each smaller
component to the largest. features="smooth_random" adds features
X ~ N(0, L^+), as in the synthetic experiments of FGC (Kumar et al., ICML
2023).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def make_sbm(
    n_communities: int = 5,
    n_per_community: Union[int, Sequence[int]] = 100,
    p_in: float = 1.0,
    p_out: float = 0.02,
    *,
    n_nodes: Optional[int] = None,
    features: Optional[str] = None,
    n_features: int = 10,
    seed: int = 42,
) -> Data:
    """Stochastic block model with n_communities blocks.

    Community sizes come from n_per_community, as one size for every block or
    one size per block, or from n_nodes, split as evenly as possible with the
    remainder going to the first blocks: n_communities=3, n_nodes=500 gives
    [167, 167, 166].

    Parameters
    ----------
    n_communities : int, default 5
    n_per_community : int or sequence of int, default 100
    n_nodes : int, optional
        Total number of nodes; give it or n_per_community, not both.
    p_in, p_out : float, default 1.0 and 0.02
        Edge probability inside a block and between two blocks.
    features : {None, "smooth_random"}, default None
    n_features : int, default 10
    seed : int, default 42

    Returns
    -------
    torch_geometric.data.Data
    """
    if n_nodes is not None:
        if not isinstance(n_per_community, int) or n_per_community != 100:
            raise ValueError("give either n_nodes or n_per_community, not both.")
        if n_nodes < n_communities:
            raise ValueError(
                f"n_nodes={n_nodes} is below n_communities={n_communities}."
            )
        base, rest = divmod(n_nodes, n_communities)
        sizes = [base + 1] * rest + [base] * (n_communities - rest)
    elif isinstance(n_per_community, int):
        sizes = [n_per_community] * n_communities
    else:
        sizes = list(n_per_community)
        if len(sizes) != n_communities:
            raise ValueError(
                f"n_per_community has {len(sizes)} sizes for "
                f"n_communities={n_communities}."
            )
    probs = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs, seed=seed, directed=False)
    G = _ensure_connected_and_relabel(G)
    return _nx_to_pyg(G, features=features, n_features=n_features, seed=seed)


def make_geometric(
    n_nodes: int = 1000,
    radius: float = 0.1,
    *,
    features: Optional[str] = None,
    n_features: int = 10,
    seed: int = 42,
) -> Data:
    """Random geometric graph: n_nodes uniform points in the unit square, joined when closer than radius."""
    G = nx.random_geometric_graph(n_nodes, radius, seed=seed)
    G = _ensure_connected_and_relabel(G)
    return _nx_to_pyg(G, features=features, n_features=n_features, seed=seed)


def make_erdos_renyi(
    n_nodes: int = 1000,
    p: float = 0.01,
    *,
    features: Optional[str] = None,
    n_features: int = 10,
    seed: int = 42,
) -> Data:
    """Erdős-Rényi graph G(n, p): each edge drawn independently with probability p."""
    G = nx.gnp_random_graph(n_nodes, p, seed=seed, directed=False)
    G = _ensure_connected_and_relabel(G)
    return _nx_to_pyg(G, features=features, n_features=n_features, seed=seed)


def make_barabasi_albert(
    n_nodes: int = 1000,
    m: int = 5,
    *,
    features: Optional[str] = None,
    n_features: int = 10,
    seed: int = 42,
) -> Data:
    """Barabási-Albert graph: each new node attaches to m nodes, with probability proportional to their degree."""
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    G = _ensure_connected_and_relabel(G)
    return _nx_to_pyg(G, features=features, n_features=n_features, seed=seed)


def make_watts_strogatz(
    n_nodes: int = 1000,
    k: int = 6,
    p_rewire: float = 0.1,
    *,
    features: Optional[str] = None,
    n_features: int = 10,
    seed: int = 42,
) -> Data:
    """Watts-Strogatz graph: a ring where each node has k neighbours, each edge rewired with probability p_rewire."""
    G = nx.watts_strogatz_graph(n_nodes, k, p_rewire, seed=seed)
    G = _ensure_connected_and_relabel(G)
    return _nx_to_pyg(G, features=features, n_features=n_features, seed=seed)


def _ensure_connected_and_relabel(G: nx.Graph) -> nx.Graph:
    """Join each smaller component to the largest by one edge, then relabel the nodes 0..N-1."""
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        components.sort(key=len, reverse=True)
        main = next(iter(components[0]))
        for comp in components[1:]:
            G.add_edge(main, next(iter(comp)))
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")
    return G


def _nx_to_pyg(
    G: nx.Graph,
    *,
    features: Optional[str],
    n_features: int,
    seed: int,
) -> Data:
    """networkx graph to PyG Data, with the optional features."""
    data = from_networkx(G)
    if features is None:
        return data
    if features == "smooth_random":
        data.x = _smooth_random_features(data, n_features=n_features, seed=seed)
        return data
    raise ValueError(f"features must be None or 'smooth_random', got {features!r}")


def _smooth_random_features(data: Data, *, n_features: int, seed: int) -> torch.Tensor:
    """Features X = L^{+1/2} Y with Y standard normal, so X ~ N(0, L^+).

    Dense eigendecomposition in O(N^3), run in float64 since 1/sqrt(lambda)
    amplifies the error on small eigenvalues; X is float32.
    """
    from graco.core.laplacian import build_laplacian
    L = build_laplacian(data, "combinatorial", dtype=torch.float64).to_dense()
    eigvals, eigvecs = torch.linalg.eigh(L)
    zero = eigvals.abs() < 1e-7
    inv_sqrt = torch.where(zero, torch.zeros_like(eigvals), eigvals.clamp(min=1e-7).rsqrt())
    L_inv_sqrt = ((eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T).to(torch.float32)

    g = torch.Generator().manual_seed(seed)
    Y = torch.randn(data.num_nodes, n_features, generator=g, dtype=torch.float32)
    return L_inv_sqrt @ Y
