"""Coarsening dataclass, returned by graco.core.builders.build_coarsening."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from graco.core.laplacian import LaplacianKind
from graco.core.partition import Partition


@dataclass
class Coarsening:
    """Coarse side of a coarsening.

    The original graph is not stored. Metrics that compare both graphs take it
    as an argument, e.g. rsa(coarsening, original), so a large original can be
    freed once the coarsening is built.

    Attributes
    ----------
    coarsened : torch_geometric.data.Data
        Coarsened graph: weighted adjacency A_c and features x_c = P x, with no
        labels and no masks.
    partition : Partition
        Supernode of each original node. partition.to_Q_combi() gives the
        binary indicator of the partition.
    Q : torch.Tensor, sparse CSR (N, n)
        Lifting, Q[i, c] = sqrt(D_i / D_c) for node i in supernode c. D is the
        degree, plus one under "normalized_self_loop"; Q is binary under
        "combinatorial".
    P : torch.Tensor, sparse CSR (n, N)
        Reduction, the pseudo-inverse of Q in closed form, or P_chain when
        built with reduction="chain".
    P_chain : torch.Tensor or None
        Reduction of local variation under reduction="per_level", the
        pseudo-inverses of its levels composed, which the partition alone
        cannot rebuild. It belongs to the Laplacian of the descent; at
        Delta = I it is the reduction Loukas uses (JMLR 2019, Eq. 2). None
        for other methods and for the global variant, whose reduction is Q^+.
    method : str
        Name of the coarsener.
    r_requested, r_actual : float
        Requested and obtained reduction ratio.
    laplacian_kind : str
        Laplacian convention behind Q.
    extras : dict
        Whatever the caller attached, e.g. the info dict of the coarsener.
    N, n : int
        Number of original nodes and of supernodes.
    """

    coarsened: "torch_geometric.data.Data"
    partition: Partition

    Q: torch.Tensor                                   # degree-normalised lifting
    P: torch.Tensor                                   # pinv(Q), or P_chain on demand
    P_chain: Optional[torch.Tensor] = None            # multilevel chain, local variation only

    method: str = "unknown"
    r_requested: float = 0.0
    r_actual: float = 0.0
    laplacian_kind: LaplacianKind = "normalized_self_loop"

    extras: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.partition.N

    @property
    def n(self) -> int:
        return self.partition.n_clusters

    def __repr__(self) -> str:
        return (
            f"Coarsening(method={self.method!r}, N={self.N}, n={self.n}, "
            f"r_requested={self.r_requested:.3f}, r_actual={self.r_actual:.3f}, "
            f"laplacian_kind={self.laplacian_kind!r}, "
            f"has_P_chain={self.P_chain is not None})"
        )
