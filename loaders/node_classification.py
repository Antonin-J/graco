"""Loaders of the node classification datasets, one per family.

Planetoid has one split, the public split of Yang et al. (ICML 2016).
Heterophilous and WikipediaNetwork ship ten splits as (N, 10) masks. OGB has
one fixed split, turned into masks. CityNetwork and Reddit have one split.
get_dataset in registry.py dispatches to these functions.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import (
    HeterophilousGraphDataset,
    Planetoid,
    WikipediaNetwork,
)
from graco.loaders._common import _split_mask, datasets_dir
from graco.loaders.preprocess import preprocess_graph

# Since PyTorch 2.6, torch.load defaults to weights_only=True, which refuses
# the PyG classes stored in processed datasets, e.g. by OGB. Allow them; older
# PyG versions lack these classes and skip this.
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except (ImportError, AttributeError):
    pass

try:
    from torch_geometric.datasets import CityNetwork
    _HAS_CITY = True
except ImportError:
    _HAS_CITY = False

try:
    from ogb.nodeproppred import PygNodePropPredDataset
    _HAS_OGB = True
except ImportError:
    _HAS_OGB = False


_PLANETOID_NAMES = {"Cora", "CiteSeer", "PubMed"}


def load_planetoid(
    name: str,
    *,
    keep_lcc: bool = True,
    root: Optional[str] = None,
    pyg_split: str = "public",
) -> Data:
    """Cora, CiteSeer or PubMed. pyg_split picks the PyG split: "public", "full" or "geom-gcn"."""
    path = os.path.join(datasets_dir(root), "planetoid")
    data = Planetoid(root=path, name=name, split=pyg_split)[0]
    return preprocess_graph(data, keep_lcc=keep_lcc)


_HETERO_NAMES = {"Amazon-ratings", "Roman-empire", "Minesweeper", "Tolokers", "Questions"}


def load_heterophilic(
    name: str,
    *,
    keep_lcc: bool = True,
    split: Union[int, str] = 0,
    root: Optional[str] = None,
) -> Data:
    """Amazon-ratings, Roman-empire, Minesweeper, Tolokers or Questions.

    split=int keeps one of the ten splits; split="all" keeps the (N, 10) masks.
    """
    path = os.path.join(datasets_dir(root), "heterophilic_graphs")
    data = HeterophilousGraphDataset(root=path, name=name)[0]
    data = preprocess_graph(data, keep_lcc=keep_lcc)
    return _select_split(data, split)


_WIKI_NAMES = {"chameleon", "squirrel"}


def load_wikipedia(
    name: str,
    *,
    keep_lcc: bool = True,
    split: Union[int, str] = 0,
    root: Optional[str] = None,
) -> Data:
    """chameleon or squirrel, with the geom_gcn preprocessing and its ten splits."""
    path = os.path.join(datasets_dir(root), "wikipedia")
    data = WikipediaNetwork(root=path, name=name, geom_gcn_preprocess=True)[0]
    # chameleon and squirrel ship directed and with self-loops, and pymetis
    # segfaults on an asymmetric adjacency.
    data = preprocess_graph(data, keep_lcc=keep_lcc)
    return _select_split(data, split)


_WIKI_FILTERED_NAMES = {"chameleon-filtered", "squirrel-filtered"}


def load_wikipedia_filtered(
    name: str,
    *,
    keep_lcc: bool = True,
    split: Union[int, str] = 0,
    root: Optional[str] = None,
) -> Data:
    """chameleon or squirrel as filtered by Platonov et al. (ICLR 2023), with their ten splits.

    The filtering removes the duplicated nodes of the geom_gcn version, which
    put copies of a test node in the training set. The npz files come from
    the repository of the paper, the source PyG reads its Heterophilous
    datasets from.
    """
    import numpy as np
    from torch_geometric.data import download_url

    stem = name.replace("-filtered", "_filtered")
    path = os.path.join(datasets_dir(root), "wikipedia_filtered")
    npz = os.path.join(path, stem + ".npz")
    if not os.path.exists(npz):
        download_url(f"https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{stem}.npz", path)
    # plain arrays, so no pickle
    raw = np.load(npz, allow_pickle=False)
    data = Data(
        x=torch.from_numpy(raw["node_features"]).float(),
        y=torch.from_numpy(raw["node_labels"]).long(),
        edge_index=torch.from_numpy(raw["edges"]).long().t().contiguous(),
        train_mask=torch.from_numpy(raw["train_masks"]).bool().t().contiguous(),
        val_mask=torch.from_numpy(raw["val_masks"]).bool().t().contiguous(),
        test_mask=torch.from_numpy(raw["test_masks"]).bool().t().contiguous(),
    )
    data = preprocess_graph(data, keep_lcc=keep_lcc)
    return _select_split(data, split)


# The names PyG accepts for CityNetwork; Los Angeles is "la".
_CITY_NAMES = {"paris", "london", "shanghai", "la"}


def load_city(
    name: str,
    *,
    keep_lcc: bool = True,
    root: Optional[str] = None,
) -> Data:
    """paris, london, shanghai or la, from the CityNetwork datasets of PyG."""
    if not _HAS_CITY:
        raise ImportError(
            "torch_geometric.datasets.CityNetwork not available in this PyG version."
        )
    path = os.path.join(datasets_dir(root), "city_networks")
    data = CityNetwork(root=path, name=name)[0]
    return preprocess_graph(data, keep_lcc=keep_lcc)


_OGB_NAMES = {"ogbn-arxiv", "ogbn-products"}


def load_ogb(
    name: str,
    *,
    keep_lcc: bool = True,
    root: Optional[str] = None,
) -> Data:
    """ogbn-arxiv or ogbn-products. Needs the ogb extra.

    The fixed split of OGB, given as index lists, becomes train_mask, val_mask
    and test_mask.
    """
    if not _HAS_OGB:
        raise ImportError("ogb package not installed (pip install ogb).")
    import torch
    path = os.path.join(datasets_dir(root), "ogb")
    ds = PygNodePropPredDataset(name=name, root=path)
    data = ds[0]
    split_idx = ds.get_idx_split()
    N = data.num_nodes
    for split_name in ("train", "valid", "test"):
        idx = split_idx[split_name]
        mask = torch.zeros(N, dtype=torch.bool)
        mask[idx] = True
        attr = "val_mask" if split_name == "valid" else f"{split_name}_mask"
        setattr(data, attr, mask)
    if data.y.dim() == 2 and data.y.shape[1] == 1:
        data.y = data.y.squeeze(1)
    # OGB graphs ship directed, and pymetis segfaults on an asymmetric adjacency.
    return preprocess_graph(data, keep_lcc=keep_lcc)


def load_reddit(
    name: str,
    *,
    keep_lcc: bool = True,
    root: Optional[str] = None,
) -> Data:
    """The Reddit dataset of PyG (Hamilton et al., NeurIPS 2017); name must be "reddit"."""
    from torch_geometric.datasets import Reddit
    if name != "reddit":
        raise ValueError(f"load_reddit expects 'reddit', got {name!r}")
    path = os.path.join(datasets_dir(root), "reddit")
    data = Reddit(root=path)[0]
    return preprocess_graph(data, keep_lcc=keep_lcc)


def _select_split(data: Data, split: Union[int, str]) -> Data:
    """Keep column split of the (N, n_splits) masks; "all" keeps them whole."""
    if split == "all":
        return data
    if not isinstance(split, int):
        raise ValueError(f"split must be int or 'all', got {split!r}")
    for attr in ("train_mask", "val_mask", "test_mask"):
        m = getattr(data, attr, None)
        if m is not None and m.ndim == 2:
            setattr(data, attr, _split_mask(m, split))
    return data


# Name to (family, loader), read by registry.py.
_LOADERS = {
    **{n: ("planetoid",    load_planetoid)    for n in _PLANETOID_NAMES},
    **{n: ("heterophilic", load_heterophilic) for n in _HETERO_NAMES},
    **{n: ("wikipedia",    load_wikipedia)    for n in _WIKI_NAMES},
    **{n: ("wikipedia",    load_wikipedia_filtered) for n in _WIKI_FILTERED_NAMES},
    **{n: ("city",         load_city)         for n in _CITY_NAMES},
    **{n: ("ogb",          load_ogb)          for n in _OGB_NAMES},
    "reddit": ("reddit", load_reddit),
}
