"""Assemble a Coarsening from the partition of any coarsener.

Example
-------
    from graco.algorithms.local_variation.main import coarsen_local_variation
    from graco.core.builders import build_coarsening

    partition, info = coarsen_local_variation(data, r=0.5)
    coarsening = build_coarsening(
        data, partition,
        method="local_variation",
        r_requested=0.5,
        P_chain=info["P_chain"],
        extras={"info": info},
    )
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch_geometric.data import Data

from graco.core.coarsening import Coarsening
from graco.core.conversions import (
    data_to_torch_csr_adjacency,
    extract_triplets,
)
from graco.core.laplacian import LaplacianKind
from graco.core.partition import Partition, pinv_well_partitioned_torch
from graco.core.q import (
    Q_normalized_from_partition,
    coarsened_adjacency,
    degrees_torch,
)


def build_coarsening(
    data,
    partition: Partition,
    *,
    laplacian_kind: LaplacianKind = "normalized_self_loop",
    method: str = "unknown",
    r_requested: Optional[float] = None,
    P_chain: Optional[torch.Tensor] = None,
    reduction: Literal["pinv", "chain"] = "pinv",
    extras: Optional[dict] = None,
    delete_diagonal: bool = True,
    dtype: torch.dtype = torch.float32,        # PyG / training convention
) -> Coarsening:
    """Build the Coarsening of a partition.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Original graph, undirected and without self-loops.
    partition : Partition
        Partition of the nodes of data.
    laplacian_kind : {"combinatorial", "normalized", "normalized_self_loop"}, default "normalized_self_loop"
        Degree normalisation of Q, see Coarsening.
    method : str, default "unknown"
        Name stored in the Coarsening.
    r_requested : float, optional
        Stored as is; None stores r_actual.
    P_chain : torch.Tensor, optional
        Reduction of local variation under reduction="per_level", chained over
        its levels, (n, N), cast to dtype. It belongs to the Laplacian the
        descent ran on, and becomes P only with reduction="chain".
    reduction : {"pinv", "chain"}, default "pinv"
        Which reduction becomes P: the pseudo-inverse of Q, or P_chain. Every
        metric reads P, so it follows this choice.
    extras : dict, optional
        Stored as is.
    delete_diagonal : bool, default True
        If True, drop the diagonal of A_c, i.e. the edge weight inside each
        supernode.
    dtype : torch.dtype, default torch.float32
        dtype of A_c, x_c, Q and P.

    Returns
    -------
    Coarsening
        A_c = Q_b^T A Q_b with Q_b the binary indicator, symmetrised. Q is built
        from the degrees of A and A_c, P = pinv(Q) or P_chain, and x_c = P x.
    """
    if data.num_nodes != partition.N:
        raise ValueError(
            f"data.num_nodes ({data.num_nodes}) != partition.N ({partition.N})"
        )
    if reduction not in ("pinv", "chain"):
        raise ValueError(f"reduction must be pinv or chain, got {reduction!r}")
    if reduction == "chain" and P_chain is None:
        raise ValueError("reduction='chain' needs P_chain")
    if P_chain is not None:
        P_chain = P_chain.to(dtype)
        if P_chain.layout != torch.sparse_csr:
            P_chain = P_chain.to_sparse_csr()
        if tuple(P_chain.shape) != (partition.n_clusters, partition.N):
            raise ValueError(f"P_chain has shape {tuple(P_chain.shape)}, expected "
                             f"({partition.n_clusters}, {partition.N})")

    # Build everything in dtype from the start, so nothing is cast later.
    Q_binary = partition.to_Q_combi(dtype=dtype)             # (N, n) sparse CSR
    A_orig = data_to_torch_csr_adjacency(data, dtype=dtype)  # (N, N) sparse CSR

    A_c = coarsened_adjacency(
        A_orig, Q_binary, delete_diagonal=delete_diagonal, symmetrize=True
    )
    deg_orig = degrees_torch(A_orig)
    deg_c = degrees_torch(A_c)
    Q = Q_normalized_from_partition(Q_binary, deg_orig, deg_c, laplacian_kind)
    P = pinv_well_partitioned_torch(Q) if reduction == "pinv" else P_chain

    # Coarsened features x_c = P x.
    x_c = (P @ data.x.to(dtype)) if data.x is not None else None

    # The coarsened Data carries no labels and no masks.
    Ac_row, Ac_col, Ac_vals = extract_triplets(A_c)
    coarsened = Data(
        x=x_c,
        edge_index=torch.stack([Ac_row, Ac_col]),
        edge_weight=Ac_vals,
        num_nodes=partition.n_clusters,
    )

    r_actual = 1.0 - partition.n_clusters / partition.N
    r_req = r_actual if r_requested is None else r_requested

    return Coarsening(
        coarsened=coarsened,
        partition=partition,
        Q=Q,
        P=P,
        P_chain=P_chain,
        method=method,
        r_requested=r_req,
        r_actual=r_actual,
        laplacian_kind=laplacian_kind,
        extras=extras if extras is not None else {},
    )


def convert_laplacian_kind_coarsening(
    coarsening: Coarsening,
    original,
    new_laplacian_kind: LaplacianKind,
) -> Coarsening:
    """Rebuild Q and P of a coarsening under another Laplacian convention.

    The partition is reused, so the coarsener does not run again.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
        The original graph, needed for the degrees.
    new_laplacian_kind : {"combinatorial", "normalized", "normalized_self_loop"}

    Returns
    -------
    Coarsening
        New Q and P = pinv(Q), with the degrees of A_c taken with its diagonal
        dropped. P_chain is dropped, since it belongs to the Laplacian of the
        descent. Every other field is the input's, including coarsened, whose
        x_c still comes from the old P. The input itself is returned when the
        convention does not change.
    """
    if new_laplacian_kind == coarsening.laplacian_kind:
        return coarsening

    dtype = coarsening.Q.values().dtype if coarsening.Q.layout == torch.sparse_csr else coarsening.Q.dtype
    Q_binary = coarsening.partition.to_Q_combi(dtype=dtype)
    # Recompute A and A_c for their degrees.
    A_orig = data_to_torch_csr_adjacency(original, dtype=dtype)
    A_c = coarsened_adjacency(A_orig, Q_binary, delete_diagonal=True, symmetrize=True)
    deg_orig = degrees_torch(A_orig)
    deg_c = degrees_torch(A_c)
    new_Q = Q_normalized_from_partition(Q_binary, deg_orig, deg_c, new_laplacian_kind)
    new_P = pinv_well_partitioned_torch(new_Q)

    return Coarsening(
        coarsened=coarsening.coarsened,
        partition=coarsening.partition,
        Q=new_Q,
        P=new_P,
        P_chain=None,
        method=coarsening.method,
        r_requested=coarsening.r_requested,
        r_actual=coarsening.r_actual,
        laplacian_kind=new_laplacian_kind,
        extras=coarsening.extras,
    )
