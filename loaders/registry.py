"""get_dataset, the single entry point of the node classification datasets.

    data = get_dataset("Cora")                     # largest component, split 0
    data = get_dataset("Roman-empire", split=3)    # fourth of ten splits
    data = get_dataset("ogbn-arxiv", root="/path/to/datasets")

Synthetic generators are not registered; call graco.loaders.synthetic.
"""

from __future__ import annotations

from typing import Optional, Union

from torch_geometric.data import Data

from graco.loaders.node_classification import _LOADERS


def get_dataset(
    name: str,
    *,
    keep_lcc: bool = True,
    split: Union[int, str] = 0,
    root: Optional[str] = None,
) -> Data:
    """Load a registered node classification dataset by name.

    Names follow torch_geometric and OGB, case included:

        Planetoid              "Cora", "CiteSeer", "PubMed"
        Heterophilous          "Amazon-ratings", "Roman-empire", "Minesweeper",
                               "Tolokers", "Questions"
        WikipediaNetwork       "chameleon", "squirrel"
        Platonov et al. 2023   "chameleon-filtered", "squirrel-filtered"
        CityNetwork            "paris", "london", "shanghai", "la"
        OGB                    "ogbn-arxiv", "ogbn-products"
        Reddit                 "reddit"

    Parameters
    ----------
    name : str
    keep_lcc : bool, default True
        If True, keep the largest connected component. Local variation needs a
        connected graph.
    split : int or "all", default 0
        For the ten-split datasets (Heterophilous, WikipediaNetwork and its
        filtered version), the column of the masks to keep; "all" keeps the
        (N, 10) masks. Ignored elsewhere.
    root : str, optional
        Dataset directory. None reads GRACO_DATASETS_DIR, then the default.

    Returns
    -------
    torch_geometric.data.Data
        Undirected, without self-loops, with train_mask, val_mask and test_mask.
    """
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}. Known: {sorted(_LOADERS)}")
    family, loader = _LOADERS[name]

    # Single-split families take no split argument.
    if family in ("heterophilic", "wikipedia"):
        return loader(name, keep_lcc=keep_lcc, split=split, root=root)
    return loader(name, keep_lcc=keep_lcc, root=root)


def list_datasets() -> dict:
    """Registered dataset names, grouped by family."""
    out: dict = {}
    for name, (family, _) in _LOADERS.items():
        out.setdefault(family, []).append(name)
    return {fam: sorted(names) for fam, names in out.items()}
