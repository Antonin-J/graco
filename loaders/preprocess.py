"""Put a PyG graph in the shape the coarseners expect.

The coarseners assume an undirected graph without self-loops, and local
variation also needs a connected one. The built-in loaders call
preprocess_graph too, so a graph of your own goes through the same steps.
"""

from __future__ import annotations

from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import to_undirected


def preprocess_graph(
    data: Data,
    *,
    symmetrize: bool = True,
    remove_self_loops: bool = True,
    keep_lcc: bool = True,
) -> Data:
    """Strip self-loops, symmetrise, then keep the largest connected component.

    Each step is idempotent, so a clean graph comes back unchanged. Connectivity
    is judged on the undirected graph. An edge_weight or edge_attr follows its
    edges; symmetrising averages the two directions of an edge.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Modified in place by the first two steps.
    symmetrize : bool, default True
        If True, add the reverse of every edge.
    remove_self_loops : bool, default True
        If True, drop the self-loops.
    keep_lcc : bool, default True
        If True, keep the largest connected component.

    Returns
    -------
    torch_geometric.data.Data
    """
    keys = [k for k in ("edge_weight", "edge_attr") if getattr(data, k, None) is not None]
    attrs = [data[k] for k in keys]
    if remove_self_loops:
        keep = data.edge_index[0] != data.edge_index[1]
        data.edge_index = data.edge_index[:, keep]
        attrs = [a[keep] for a in attrs]
    if symmetrize:
        if attrs:
            data.edge_index, attrs = to_undirected(
                data.edge_index, attrs, num_nodes=data.num_nodes, reduce="mean"
            )
        else:
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    for k, a in zip(keys, attrs):
        data[k] = a
    if keep_lcc:
        data = LargestConnectedComponents()(data)
    return data
