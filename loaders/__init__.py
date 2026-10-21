"""Dataset loaders and synthetic graph generators.

get_dataset(name) loads any registered node classification dataset,
list_datasets() lists them. preprocess_graph puts a graph of your own in the
shape the coarseners expect. The make_* functions generate synthetic graphs.
Datasets are downloaded under GRACO_DATASETS_DIR, by default
~/.cache/graco/datasets.
"""

from graco.loaders._common import datasets_dir, validate_data
from graco.loaders.node_classification import (
    load_city,
    load_heterophilic,
    load_ogb,
    load_planetoid,
    load_wikipedia,
)
from graco.loaders.preprocess import preprocess_graph
from graco.loaders.registry import get_dataset, list_datasets
from graco.loaders.synthetic import (
    make_barabasi_albert,
    make_erdos_renyi,
    make_geometric,
    make_sbm,
    make_watts_strogatz,
)

__all__ = [
    # registry
    "get_dataset",
    "list_datasets",
    # preprocessing (bring-your-own-graph)
    "preprocess_graph",
    # node-classif loaders (direct access)
    "load_city",
    "load_heterophilic",
    "load_ogb",
    "load_planetoid",
    "load_wikipedia",
    # synthetic generators
    "make_barabasi_albert",
    "make_erdos_renyi",
    "make_geometric",
    "make_sbm",
    "make_watts_strogatz",
    # helpers
    "datasets_dir",
    "validate_data",
]
