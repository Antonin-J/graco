"""GraCo: a graph coarsening toolkit.

The names needed to load a graph, coarsen it, build the coarsening and save
it. Metrics and refinements stay in their subpackages, graco.metrics and
graco.refinement.
"""

from graco.algorithms import (
    COARSENING_ALGOS,
    coarsen_convmatch,
    coarsen_fgc,
    coarsen_gbgc,
    coarsen_gw_gpc,
    coarsen_gw_kgpc,
    coarsen_gw_sgwl,
    coarsen_herding,
    coarsen_ipe,
    coarsen_kcenter,
    coarsen_leiden,
    coarsen_local_variation,
    coarsen_metis,
    coarsen_nope,
    coarsen_otcoarsening,
    coarsen_random,
    coarsen_stpgc,
    coarsen_ugc,
    get_coarsening_algo,
)
from graco.core.builders import build_coarsening
from graco.core.coarsening import Coarsening
from graco.core.partition import Partition
from graco.loaders import get_dataset, list_datasets
from graco.storage import load_coarsening, save_coarsening

__all__ = [
    "COARSENING_ALGOS",
    "Coarsening",
    "Partition",
    "build_coarsening",
    "coarsen_convmatch",
    "coarsen_fgc",
    "coarsen_gbgc",
    "coarsen_gw_gpc",
    "coarsen_gw_kgpc",
    "coarsen_gw_sgwl",
    "coarsen_herding",
    "coarsen_ipe",
    "coarsen_kcenter",
    "coarsen_leiden",
    "coarsen_local_variation",
    "coarsen_metis",
    "coarsen_nope",
    "coarsen_otcoarsening",
    "coarsen_random",
    "coarsen_stpgc",
    "coarsen_ugc",
    "get_coarsening_algo",
    "get_dataset",
    "list_datasets",
    "load_coarsening",
    "save_coarsening",
]

# Graclus needs torch-cluster; exported only when the registry has it.
if "graclus" in COARSENING_ALGOS:
    coarsen_graclus = COARSENING_ALGOS["graclus"]
    __all__.append("coarsen_graclus")
