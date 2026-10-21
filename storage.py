"""Save and reload a coarsening as its partition alone.

A saved coarsening is a directory with two files:

    partition.npz   the assignment (int64) and n_clusters (int)
    meta.json       parameters, dataset, timing, peak memory, versions

Q, P and the coarsened graph are not written. load_coarsening rebuilds them
with build_coarsening, which is deterministic given the graph, the partition
and the laplacian_kind. The save is much smaller than the matrices, the same
partition can be rebuilt under another laplacian_kind, and the file does not
depend on the torch or PyG version.

The one reduction a partition cannot rebuild is P_chain, the reduction of
local variation under reduction="per_level", composed over its levels. It has
the support of Q^T, one entry
per node, so partition.npz stores it as p_chain_values, the entry of each
node, when the coarsening has one.

Bump BUILD_VERSION on any change to what build_coarsening returns
for a given partition. Older saves then refuse to load, while partition.npz
stays readable with np.load.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from graco.core.builders import build_coarsening
from graco.core.coarsening import Coarsening
from graco.core.partition import Partition

BUILD_VERSION = 2
META_NAME = "meta.json"
PARTITION_NAME = "partition.npz"


def pyg_version() -> Optional[str]:
    """Installed torch_geometric version, or None."""
    try:
        import torch_geometric
        return torch_geometric.__version__
    except Exception:
        return None


def save_coarsening(
    coarsening: Coarsening,
    *,
    dir_: Path,
    dataset: str,
    algo: str,
    algo_params: dict,                       # algo-specific extras (K, cost, ...); no laplacian_kind here
    laplacian_kind_coarsen: str = "normalized_self_loop",
    laplacian_kind_build: Optional[str] = None,
    r_requested: Optional[float] = None,
    n_clusters_requested: Optional[int] = None,
    seed: int = 14,
    coarsening_time_s: float = 0.0,
    coarsening_peak_mem_MB: float = 0.0,
) -> Path:
    """Write partition.npz and meta.json for one coarsening, overwriting them.

    Metrics are not saved, so they can be recomputed on their own. A P_chain
    goes into partition.npz as one value per node.

    Parameters
    ----------
    coarsening : Coarsening
    dir_ : path
        Target directory, created if needed.
    dataset, algo : str
        Names recorded in meta.json.
    algo_params : dict
        Keyword arguments of the coarsener, without laplacian_kind.
    laplacian_kind_coarsen : str, default "normalized_self_loop"
        Laplacian passed to the coarsener; only the spectral ones read it.
    laplacian_kind_build : str, optional
        Laplacian passed to build_coarsening, which sets Q; None takes the
        one the coarsening was built with.
    r_requested, n_clusters_requested : optional
        The target asked for.
    seed : int, default 14
    coarsening_time_s, coarsening_peak_mem_MB : float, default 0.0
        Cost of the coarsener, recorded as given.

    Returns
    -------
    Path
        dir_.
    """
    if laplacian_kind_build is None:
        laplacian_kind_build = coarsening.laplacian_kind
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=True)

    arrays = dict(
        assignment=coarsening.partition.assignment,
        n_clusters=np.array(coarsening.partition.n_clusters, dtype=np.int64),
    )
    if coarsening.P_chain is not None:
        arrays["p_chain_values"] = p_chain_values(coarsening)
    np.savez(dir_ / PARTITION_NAME, **arrays)

    meta = {
        "dataset": dataset,
        "algo": algo,
        "algo_params": algo_params,
        "laplacian_kind_coarsen": laplacian_kind_coarsen,
        "laplacian_kind_build": laplacian_kind_build,
        "r_requested": r_requested,
        "n_clusters_requested": n_clusters_requested,
        "seed": seed,
        "N": int(coarsening.N),
        "n": int(coarsening.n),
        "r_actual": float(coarsening.r_actual),
        "coarsening_time_s": float(coarsening_time_s),
        "coarsening_peak_mem_MB": float(coarsening_peak_mem_MB),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "torch_geometric_version": pyg_version(),
        "build_version": BUILD_VERSION,
    }
    with open(dir_ / META_NAME, "w") as f:
        json.dump(meta, f, indent=2)
    return dir_


def p_chain_values(coarsening: Coarsening) -> np.ndarray:
    """The entry of P_chain in the column of each node, on the row of its supernode."""
    coo = coarsening.P_chain.to_sparse_coo().coalesce()
    rows, cols = coo.indices().cpu().numpy()
    vals = coo.values().cpu().numpy()
    assignment = np.asarray(coarsening.partition.assignment)
    if cols.shape[0] != coarsening.N or not np.array_equal(rows, assignment[cols]):
        raise ValueError("P_chain does not have the support of Q^T")
    out = np.empty(coarsening.N, dtype=vals.dtype)
    out[cols] = vals
    return out


def load_meta(dir_: Path) -> dict:
    """Read meta.json of a saved coarsening."""
    with open(Path(dir_) / META_NAME) as f:
        return json.load(f)


def load_coarsening(dir_: Path, data: Data, reduction: str = "pinv") -> Coarsening:
    """Reload a saved partition and rebuild its Coarsening with build_coarsening.

    Q and P are rebuilt under laplacian_kind_build, or "normalized_self_loop"
    when meta.json lacks it. A saved P_chain comes back only when
    laplacian_kind_build is the Laplacian of the descent. Raises ValueError
    when the save was made under another BUILD_VERSION.

    Parameters
    ----------
    dir_ : path
    data : torch_geometric.data.Data
        The graph the partition was computed on.
    reduction : {"pinv", "chain"}, default "pinv"
        Passed to build_coarsening: "chain" makes the saved P_chain the P.

    Returns
    -------
    Coarsening
    """
    dir_ = Path(dir_)
    meta = load_meta(dir_)

    # Saves made before the rename carry the key benchmark_build_version.
    saved = meta.get("build_version", meta.get("benchmark_build_version", 0))
    if saved != BUILD_VERSION:
        raise ValueError(
            f"build_version mismatch in {dir_}: saved={saved}, "
            f"current={BUILD_VERSION}. This save comes from another version of "
            f"GraCo: coarsen again and save. The partition stays readable in "
            f"{PARTITION_NAME}."
        )

    with np.load(dir_ / PARTITION_NAME) as npz:
        assignment = npz["assignment"]
        n_clusters = int(npz["n_clusters"])
        values = npz["p_chain_values"] if "p_chain_values" in npz.files else None
    partition = Partition(assignment=assignment, n_clusters=n_clusters)

    # laplacian_kind_build sets Q; laplacian_kind_coarsen is already baked into
    # the partition.
    laplacian_kind_build = meta.get("laplacian_kind_build", "normalized_self_loop")
    P_chain = None
    if values is not None and laplacian_kind_build == meta.get("laplacian_kind_coarsen"):
        N = assignment.shape[0]
        index = torch.from_numpy(np.stack([np.asarray(assignment, dtype=np.int64),
                                           np.arange(N, dtype=np.int64)]))
        P_chain = torch.sparse_coo_tensor(index, torch.from_numpy(values),
                                           (n_clusters, N)).to_sparse_csr()
    return build_coarsening(
        data, partition,
        laplacian_kind=laplacian_kind_build,
        method=meta["algo"],
        r_requested=meta.get("r_requested"),
        P_chain=P_chain,
        reduction=reduction,
    )
