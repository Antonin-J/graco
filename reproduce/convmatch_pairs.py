"""ConvMatch under its three pairs: Table 10 of the GraCo paper (App. C.5).

Cora and PubMed at r = 0.7 and 0.9. Each pair runs with the exact cost
(ConvMatch) and the bound (A-ConvMatch), with the candidates of the paper
(kNN) and with the edges added (kNN + edges). Each row is scored by the CME
of Dickens et al. (convolution_matching_error, mode="dickens", in thousands),
the RSA on 100 eigenvectors and max_acc in %, all under the
normalized_self_loop Laplacian.

Run:  python -m graco.reproduce.convmatch_pairs --dataset Cora
Output: results/convmatch_pairs_<dataset>.json
"""

from __future__ import annotations

import argparse
import time

from graco.algorithms.convmatch import coarsen_convmatch
from graco.core.builders import build_coarsening
from graco.loaders import get_dataset
from graco.metrics import convolution_matching_error, max_acc, rsa
from graco.reproduce.scaling import default_results_dir, pick_device, write_json

laplacian_kind = "normalized_self_loop"
ratios = (0.7, 0.9)
methods = ("exact", "approximate")
candidates = ("knn", "knn+edges")
pairs = ("dickens", "binary", "normalized")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = get_dataset(args.dataset)
    dev = pick_device(args.device)
    rows = []
    for method in methods:
        for cand in candidates:
            for pair in pairs:
                for r in ratios:
                    t0 = time.perf_counter()
                    partition, _ = coarsen_convmatch(
                        data, r=r, method=method, pair=pair, merge_candidates=cand,
                        device=dev, progress=False,
                    )
                    co = build_coarsening(data, partition, laplacian_kind=laplacian_kind,
                                          method="convmatch", r_requested=r)
                    row = {
                        "dataset": args.dataset, "r": r, "method": method,
                        "candidates": cand, "pair": pair,
                        "n_clusters": int(partition.n_clusters),
                        "cme_thousands": convolution_matching_error(
                            co, data, mode="dickens")["total"] / 1e3,
                        "rsa": rsa(co, data, laplacian_kind=laplacian_kind, K=100),
                        "max_acc_pct": 100 * max_acc(co, data)["all"],
                        "time_s": time.perf_counter() - t0,
                    }
                    rows.append(row)
                    print(f"{method:11s} {cand:9s} {pair:10s} r={r}  "
                          f"CME={row['cme_thousands']:.1f}  RSA={row['rsa']:.2f}  "
                          f"acc={row['max_acc_pct']:.1f}", flush=True)

    out = default_results_dir() / f"convmatch_pairs_{args.dataset}.json"
    write_json(out, {"laplacian_kind": laplacian_kind, "rows": rows})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
