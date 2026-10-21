"""OTCoarsening on Cora and the nodes it leaves uncovered: Table 8 of the GraCo paper (App. C.3).

For r = 0.5, 0.7 and 0.9, under the shipped defaults, four readings of the
same trained pooling S:

    argmax alone       each node to its largest column of S; an uncovered node,
                       whose row of S is zero, lands in supernode 0
    attach             uncovered nodes join the supernode of the nearest covered one
    isolate            each uncovered node becomes a supernode of its own
    isolate_recursive  isolate levels stacked, the level nearest the request kept

Purity is max_acc over all nodes; largest is the size of the largest supernode.

Run:  python -m graco.reproduce.otcoarsening_cora [--device cuda]
Output: results/otcoarsening_Cora.json
"""

from __future__ import annotations

import argparse

import numpy as np

from graco.algorithms.otcoarsening import coarsen_otcoarsening
from graco.core.builders import build_coarsening
from graco.core.partition import Partition, relabel_contiguous
from graco.loaders import get_dataset
from graco.metrics import max_acc
from graco.reproduce.scaling import default_results_dir, pick_device, write_json

ratios = (0.5, 0.7, 0.9)


def purity(data, partition) -> float:
    return max_acc(build_coarsening(data, partition), data)["all"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, the device of the paper table) or cuda")
    dev = pick_device(ap.parse_args().device)
    data = get_dataset("Cora")
    rows = []
    for r in ratios:
        part_attach, info = coarsen_otcoarsening(data, r=r, uncovered="attach",
                                                 return_soft_s=True, device=dev)
        S = info["S_soft"].to_dense().cpu()
        part_argmax = Partition(*relabel_contiguous(S.argmax(dim=1).numpy()))
        part_isolate, info_isolate = coarsen_otcoarsening(data, r=r, uncovered="isolate",
                                                          device=dev)
        part_rec, info_rec = coarsen_otcoarsening(data, r=r, device=dev)
        row = {
            "r": r,
            "n_asked": int(info["n_target"]),
            "argmax_n": int(part_argmax.n_clusters),
            "argmax_purity": purity(data, part_argmax),
            "argmax_largest": int(np.bincount(part_argmax.assignment).max()),
            "attach_purity": purity(data, part_attach),
            "attach_largest": int(np.bincount(part_attach.assignment).max()),
            "isolate_uncovered": int(info_isolate["n_uncovered"]),
            "isolate_n": int(part_isolate.n_clusters),
            "recursive_levels": int(info_rec["n_levels"]),
            "recursive_n": int(part_rec.n_clusters),
            "recursive_purity": purity(data, part_rec),
        }
        rows.append(row)
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()), flush=True)

    out = default_results_dir() / "otcoarsening_Cora.json"
    write_json(out, {"N": int(data.num_nodes), "rows": rows})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
