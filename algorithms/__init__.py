"""Coarsening algorithms, one module each, behind one contract.

    coarsen_<name>(data, r=None, n_clusters=None, *, seed=14, **kwargs)
        returns (Partition, info)

Give exactly one of r, the target reduction ratio in (0, 1), or n_clusters,
the target number of supernodes in [1, N]. A ratio r asks for
n = ceil((1 - r) N) supernodes.

Every info dict carries these keys:
    method : name of the method.
    r_requested : requested reduction ratio.
    r_actual : obtained reduction ratio, 1 - n_clusters / N.
    n_clusters : number of supernodes returned.
    total_time_s : wall time of the call.
    seed : the seed used, for every method that takes one.

COARSENING_ALGOS maps a name to its function. Graclus joins it only when
torch-cluster is installed.
"""

from __future__ import annotations

from typing import Callable

from graco.algorithms.convmatch import coarsen_convmatch
from graco.algorithms.fgc import coarsen_fgc
from graco.algorithms.gbgc import coarsen_gbgc
from graco.algorithms.gw import coarsen_gw_gpc, coarsen_gw_kgpc
from graco.algorithms.gw_sgwl import coarsen_gw_sgwl
from graco.algorithms.herding import coarsen_herding
from graco.algorithms.ipe import coarsen_ipe
from graco.algorithms.kcenter import coarsen_kcenter
from graco.algorithms.leiden import coarsen_leiden
from graco.algorithms.local_variation.main import coarsen_local_variation
from graco.algorithms.metis import coarsen_metis
from graco.algorithms.nope import coarsen_nope
from graco.algorithms.otcoarsening import coarsen_otcoarsening
from graco.algorithms.random import coarsen_random
from graco.algorithms.stpgc import coarsen_stpgc
from graco.algorithms.ugc import coarsen_ugc


COARSENING_ALGOS: dict[str, Callable] = {
    "local_variation": coarsen_local_variation,
    "metis": coarsen_metis,
    "leiden": coarsen_leiden,
    "random": coarsen_random,
    "ugc": coarsen_ugc,
    "convmatch": coarsen_convmatch,
    "kcenter": coarsen_kcenter,
    "herding": coarsen_herding,
    "fgc": coarsen_fgc,
    "ipe": coarsen_ipe,
    "gw_gpc": coarsen_gw_gpc,
    "gw_kgpc": coarsen_gw_kgpc,
    "nope": coarsen_nope,
    "gbgc": coarsen_gbgc,
    "otcoarsening": coarsen_otcoarsening,
    "sgwl": coarsen_gw_sgwl,
    "stpgc": coarsen_stpgc,
}

# Graclus needs `torch-cluster`, a compiled extension. Skip it if missing.
# Probing torch_cluster directly is the only reliable test: importing
# `torch_geometric.nn.pool.graclus` succeeds without it and only raises when
# the function is called, which would blow up a sweep halfway through.
try:
    import torch_cluster  # noqa: F401
except ImportError:
    pass
else:
    from graco.algorithms.graclus import coarsen_graclus
    COARSENING_ALGOS["graclus"] = coarsen_graclus


def get_coarsening_algo(name: str) -> Callable:
    """Return the coarsening function registered under name."""
    if name == "graclus" and name not in COARSENING_ALGOS:
        raise KeyError('graclus needs torch-cluster: pip install -e ".[graclus]"')
    if name not in COARSENING_ALGOS:
        raise KeyError(
            f"Unknown coarsening algorithm {name!r}. "
            f"Available: {sorted(COARSENING_ALGOS)}"
        )
    return COARSENING_ALGOS[name]
