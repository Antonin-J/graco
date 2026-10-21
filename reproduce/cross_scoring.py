"""Every method on one dataset, scored under four metrics from different papers: Table 4 of the GraCo paper (App. B.5).

Each method runs once at its published defaults, then the same Coarsening is
read by rsa, mean_REE, convolution_matching_error and max_acc. The paper runs
it on Cora and PubMed at r = 0.7.

One Laplacian for every row. The coarseners do not agree on theirs (FGC
descends on the combinatorial one, most others on the normalised one), so a
column only compares across methods once build_coarsening and the metrics are
pinned to a single convention.

Run:  python -m graco.reproduce.cross_scoring --dataset Cora [--r 0.7]
      [--algo metis,leiden] [--skip-done]
Output: results/cross_scoring_<dataset>_r070/<algo>.json, one row per method,
        and results/cross_scoring_<dataset>_r070.json merging them.

Each row is written as "killed" before its method starts, so a method that
never returns still leaves a row.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import torch

from graco.algorithms import COARSENING_ALGOS
from graco.core.builders import build_coarsening
from graco.reproduce.scaling import (EXTRA_KWARGS, default_results_dir,
                                        device_kwargs, pick_device, write_json)
from graco.loaders import get_dataset
from graco.metrics import convolution_matching_error, max_acc, mean_REE, rsa

laplacian_kind = "normalized_self_loop"
rsa_k = 100
ree_m = 100

# STPGC runs on the topology alone here: its default reads training and
# validation labels, which max_acc would then score.
label_free = {"stpgc": {"label_guide": False}}


def score_one(data, dataset: str, r: float, algo: str, seed: int, dev) -> dict:
    """Coarsen once, score four times. Never raises: failures go in the row."""
    row = {"dataset": dataset, "r": r, "algo": algo, "seed": seed,
           "laplacian_kind": laplacian_kind, "status": "ok", "error": None,
           "n_clusters": None, "r_actual": None, "time_s": None,
           "rsa": None, "mean_REE": None, "cme": None,
           "max_acc": None, "max_acc_test": None}
    fn = COARSENING_ALGOS[algo]
    kw = {**device_kwargs(fn, dev), **EXTRA_KWARGS.get(algo, {}), **label_free.get(algo, {})}

    t0 = time.perf_counter()
    try:
        partition, _ = fn(data, r=r, seed=seed, **kw)
    except (MemoryError, torch.cuda.OutOfMemoryError):
        row["status"] = "oom"
        row["error"] = traceback.format_exc(limit=1)
        return row
    except Exception:
        row["status"] = "failed"
        row["error"] = traceback.format_exc(limit=3)
        return row
    row["time_s"] = time.perf_counter() - t0
    row["n_clusters"] = int(partition.n_clusters)
    row["r_actual"] = 1.0 - partition.n_clusters / data.num_nodes

    co = build_coarsening(data, partition, method=algo, r_requested=r,
                          laplacian_kind=laplacian_kind)
    metrics = {
        "rsa": lambda: rsa(co, data, laplacian_kind=laplacian_kind, K=rsa_k),
        "mean_REE": lambda: mean_REE(co, data, laplacian_kind=laplacian_kind, m=ree_m),
        "cme": lambda: convolution_matching_error(co, data)["total"],
    }
    for name, compute in metrics.items():
        try:
            row[name] = float(compute())
        except Exception as exc:
            row["status"] = "scored_partially"
            row["error"] = f"{name}: {type(exc).__name__}: {str(exc)[:200]}"
    try:
        m = max_acc(co, data)
        row["max_acc"] = float(m["all"])
        row["max_acc_test"] = float(m["test"]) if "test" in m else None
    except Exception as exc:
        row["status"] = "scored_partially"
        row["error"] = f"max_acc: {type(exc).__name__}: {str(exc)[:200]}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--r", type=float, default=0.7)
    ap.add_argument("--algo", default=None, help="comma list, default every method")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()

    tag = f"cross_scoring_{args.dataset}_r{int(round(args.r * 100)):03d}"
    out_dir = default_results_dir()
    raw_dir = out_dir / tag
    raw_dir.mkdir(parents=True, exist_ok=True)

    algos = args.algo.split(",") if args.algo else sorted(COARSENING_ALGOS)
    dev = pick_device(args.device)
    data = get_dataset(args.dataset)
    print(f"{args.dataset} N={data.num_nodes}, r={args.r}, {len(algos)} methods on {dev}", flush=True)

    for algo in algos:
        path = raw_dir / f"{algo}.json"
        if args.skip_done and path.exists():
            print(f"{algo}: done already", flush=True)
            continue
        write_json(path, {"dataset": args.dataset, "r": args.r, "algo": algo,
                          "status": "killed",
                          "error": "never returned: timeout or crash"})
        row = score_one(data, args.dataset, args.r, algo, args.seed, dev)
        write_json(path, row)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{algo}: {row['status']}  n={row['n_clusters']}  rsa={row['rsa']}  "
              f"REE={row['mean_REE']}  cme={row['cme']}  max_acc={row['max_acc']}",
              flush=True)

    rows = {}
    for p in sorted(raw_dir.glob("*.json")):
        rows[p.stem] = json.loads(p.read_text())
    write_json(out_dir / f"{tag}.json", rows)
    print(f"merged {len(rows)} rows into {tag}.json", flush=True)


if __name__ == "__main__":
    main()
