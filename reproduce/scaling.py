"""One coarsener, one dataset, one ratio: a cell of Table 2 of the GraCo paper.

    python -m graco.reproduce.scaling --dataset Cora --r 0.9 --algo metis

run_scaling.py calls it once per method, each in its own process, so a method
that runs out of memory or overruns its budget costs its own row only. Every
method runs at its published defaults; tuning is the job of sbm_tuning.py. The
row lands in results/raw/<dataset>_r<r>_<algo>.json, and collect_scaling.py
merges the rows of a tier.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import resource
import time
import traceback
from pathlib import Path

import torch

from graco.algorithms import COARSENING_ALGOS
from graco.core.builders import build_coarsening
from graco.loaders import get_dataset
from graco.metrics import max_acc, rsa


def default_results_dir() -> Path:
    """Where results land: GRACO_RESULTS_DIR if set, else results/ next to the scripts."""
    env = os.environ.get("GRACO_RESULTS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "results"


# Dataset, ratio, tier name, RSA eigenvector count (None skips the RSA) and
# per-method timeout in seconds. The ratios keep every coarsened graph between
# 250 and 2500 supernodes, so what varies down the table is the input size.
TIERS = [
    ("Cora",          0.9,   "small",      100,  1 * 3600),   # N =    2485, n =  249
    ("PubMed",        0.9,   "medium",     100,  1 * 3600),   # N =   19717, n = 1972
    ("ogbn-arxiv",    0.99,  "large",      100,  3 * 3600),   # N =  169343, n = 1694
    ("ogbn-products", 0.999, "very large", None, 12 * 3600),  # N = 2449029, n = 2450
]


# Rows beyond the registry defaults: a published variant picked by a keyword,
# or another setting worth a row, each with its own file.
VARIANTS: dict[str, tuple[str, dict]] = {
    "nope_star":       ("nope",      {"variant": "nope_star", "progress": True}),
    # The NOPE routine of the released code.
    "nope_repo":       ("nope",      {"variant": "nope_repo", "progress": True}),
    "ipe_ingc":        ("ipe",       {"variant": "INGC"}),
    "fgc_sparsity":    ("fgc",       {"variant": "sparsity"}),
    # GBGC cannot go below its initial ball count, which the breadth-first init
    # of the paper keeps above what the tiers ask. The follow-up of the same
    # group initialises with METIS, which reaches every tier. Both get a row.
    "gbgc_metis":      ("gbgc",      {"init": "metis"}),
    "convmatch_exact": ("convmatch", {"method": "exact"}),
    # The second contraction family of local variation, slower: it builds a
    # dense (nc, nc) matrix per neighborhood.
    "local_variation_neighborhood": ("local_variation",
                                     {"method": "neighborhood"}),
    # The descent of the JMLR paper, the combinatorial Laplacian with the
    # per-level reduction; the default rows run the global reduction under the
    # self-loop normalization. All four are built and scored under the
    # normalized lifting, like every other row.
    "local_variation_combi": ("local_variation",
                              {"laplacian_kind": "combinatorial",
                               "reduction": "per_level"}),
    "local_variation_neighborhood_combi": ("local_variation",
                                           {"laplacian_kind": "combinatorial",
                                            "reduction": "per_level",
                                            "method": "neighborhood"}),
    # The device axis: these three also run on the host, so the table carries
    # both devices.
    "convmatch_cpu":   ("convmatch", {}),
    "kcenter_cpu":     ("kcenter",   {}),
    "herding_cpu":     ("herding",   {}),
    # SGWL the other way round: its default row stays on the host, this one
    # takes the device.
    "sgwl_gpu":        ("sgwl",      {}),
    # ConvMatch under the ogbn-products parameters of the released code: two
    # hops, a smaller candidate pool, ten PCA dimensions.
    "convmatch_dickens_products": ("convmatch", {"num_hops": 2,
                                                 "nearest_neighbors_keep_rate": 0.001,
                                                 "knn_embedding_space_dim": 10}),
}

# Variants that only run on the listed tiers.
VARIANT_TIERS: dict[str, set[str]] = {
    "convmatch_dickens_products": {"ogbn-products"},
}

# Keyword arguments added to a registry default. progress=True makes NOPE
# report on stderr how far it got, should it overrun its budget.
EXTRA_KWARGS: dict[str, dict] = {
    "nope": {"progress": True},
}

# Rows that stay on the host whatever --device says.
FORCE_CPU = {"convmatch_cpu", "kcenter_cpu", "herding_cpu", "sgwl"}


def write_json(path: Path, obj) -> None:
    """Write obj as JSON atomically, through a temporary file and os.replace.

    A worker killed during the write, or two runs writing the same file, then
    never leave a truncated row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def max_rss_MB() -> float:
    """Process high-water resident set, in MB. Linux reports ru_maxrss in kB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def machine_info() -> dict:
    """Where the row was measured: host, CPU model, cores on the host and allocated, torch threads."""
    cpu_model = None
    try:
        with open("/proc/cpuinfo") as f:
            cpu_model = next((line.split(":", 1)[1].strip() for line in f
                              if line.startswith("model name")), None)
    except OSError:
        pass
    allocated = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                 else os.cpu_count())
    return {
        "cpu_model": cpu_model,
        "cpus_on_host": os.cpu_count(),
        "cpus_allocated": allocated,
        "torch_threads": torch.get_num_threads(),
    }


def pick_device(name: str | None = None) -> torch.device:
    """The requested device when available, else the host; None picks cuda when available."""
    if not name:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def device_kwargs(fn, dev) -> dict:
    """{"device": dev} when fn takes one, {} otherwise.

    The coarseners that accept a device otherwise run on data.x.device, the host,
    even when a GPU was reserved.
    """
    if dev is None:
        return {}
    return {"device": dev} if "device" in inspect.signature(fn).parameters else {}


def budget_kwargs(fn, budget) -> dict:
    """{"time_budget_s": budget} when fn takes one, {} otherwise.

    Leiden is the coarsener that does: it stops its resolution search on the
    budget and returns its best partition, where a killed run writes no row.
    """
    if budget is None:
        return {}
    return {"time_budget_s": budget} if "time_budget_s" in inspect.signature(fn).parameters else {}


def run_one(dataset: str, r: float, algo: str, *, seed: int = 14,
            time_budget_s: float | None = None,
            rsa_k: int | None = None, device: str | None = None,
            keep_lcc: bool = True, checkpoint=None) -> dict:
    """Coarsen once and score: time, peak memory, max_acc and the RSA.

    Never raises; a failure comes back in the row with its status. checkpoint,
    if given, receives the row before the RSA, so a run killed during the RSA
    keeps its time, memory and max_acc.
    """
    row: dict = {
        "dataset": dataset, "r": r, "algo": algo, "seed": seed,
        "status": "ok", "error": None, "device": None,
        **machine_info(), "gpu_model": None,
        "n_nodes": None, "n_clusters": None, "r_actual": None,
        "time_s": None,
        "search_iters": None, "search_final_call_s": None,
        "search_stopped_on_budget": None,
        "n_levels": None, "sgcn_time_s": None,
        "rss_baseline_MB": None, "rss_peak_MB": None,
        "gpu_peak_MB": None, "gpu_reserved_MB": None, "keep_lcc": None,
        "max_acc": None, "max_acc_test": None,
        "rsa": None, "rsa_K": rsa_k, "rsa_time_s": None, "rsa_error": None,
    }
    base, variant_kw = VARIANTS.get(algo, (algo, {}))
    variant_kw = {**EXTRA_KWARGS.get(algo, {}), **variant_kw}
    if base not in COARSENING_ALGOS:
        row["status"] = "unavailable"
        row["error"] = f"{algo} not in COARSENING_ALGOS (missing optional dependency?)"
        return row

    # None leaves the device to the coarsener; a named device falls back to the
    # host when absent.
    dev = pick_device(device) if device else None
    if algo in FORCE_CPU:
        dev = torch.device("cpu")
    # Every tier on its largest connected component.
    data = get_dataset(dataset, keep_lcc=keep_lcc)
    row["n_nodes"] = int(data.num_nodes)
    row["keep_lcc"] = keep_lcc

    # Memory of the interpreter, torch and the loaded graph: the floor of every
    # row.
    row["rss_baseline_MB"] = max_rss_MB()

    # Reset the device peak, in case run_one runs twice in one process.
    if dev is not None and dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    fn = COARSENING_ALGOS[base]

    extra = device_kwargs(fn, dev)
    extra.update(budget_kwargs(fn, time_budget_s))
    row["device"] = str(dev) if extra else "cpu"
    if row["device"].startswith("cuda"):
        row["gpu_model"] = torch.cuda.get_device_name(dev)

    t0 = time.perf_counter()
    try:
        out = fn(data, r=r, seed=seed, **extra, **variant_kw)
    except MemoryError:
        row["status"] = "oom"
        row["error"] = "host allocation refused"
        return row
    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
        row["error"] = "cuda out of memory"
        return row
    except Exception:
        row["status"] = "failed"
        row["error"] = traceback.format_exc(limit=3)
        return row

    partition, info = out
    row["time_s"] = time.perf_counter() - t0
    # Leiden and UGC search a resolution and a bin width to reach n_target.
    row["search_iters"] = info.get("n_search_iters", info.get("n_bin_search_iters"))
    row["search_final_call_s"] = info.get("final_call_time_s")
    row["search_stopped_on_budget"] = info.get("search_stopped_on_budget")
    # Levels and kNN time of ConvMatch.
    row["n_levels"] = info.get("n_levels")
    row["sgcn_time_s"] = info.get("sgcn_time_s")
    # Host: peak resident set of the process. Device: peak allocated and peak
    # reserved by the caching allocator; the second has to fit on the card.
    row["rss_peak_MB"] = max_rss_MB()
    if dev is not None and dev.type == "cuda":
        row["gpu_peak_MB"] = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        row["gpu_reserved_MB"] = torch.cuda.max_memory_reserved(dev) / (1024 ** 2)
    row["n_clusters"] = int(partition.n_clusters)
    row["r_actual"] = 1.0 - partition.n_clusters / data.num_nodes

    try:
        co = build_coarsening(data, partition, method=base, r_requested=r)
        m = max_acc(co, data)
        row["max_acc"] = float(m["all"])
        if "test" in m:
            row["max_acc_test"] = float(m["test"])
    except Exception:
        row["status"] = "scored_partially"
        row["error"] = traceback.format_exc(limit=3)
        return row

    if checkpoint is not None:
        checkpoint(row)

    # The RSA needs the K smallest eigenpairs of L from ARPACK; a failure is
    # recorded in the row.
    if rsa_k:
        t0 = time.perf_counter()
        try:
            row["rsa"] = float(rsa(co, data, K=rsa_k))
        except Exception as exc:
            row["rsa_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["rsa_time_s"] = time.perf_counter() - t0

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--rsa-k", type=int, default=None,
                    help="Preserved eigenvector count for the RSA constant. "
                         "Omit to skip RSA entirely.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--time-budget", type=float, default=None,
                    help="wall-clock budget handed to coarseners that accept one")
    ap.add_argument("--whole-graph", action="store_true",
                    help="Measure on the graph as shipped instead of its "
                         "largest connected component.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # One file per method under raw/, so concurrent runs never write the same
    # file; collect_scaling.py merges them into one file per tier.
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{args.dataset}_r{args.r}_{args.algo}.json"

    def write(row: dict) -> None:
        write_json(path, row)

    row = run_one(args.dataset, args.r, args.algo,
                  seed=args.seed, rsa_k=args.rsa_k, device=args.device,
                  time_budget_s=args.time_budget,
                  keep_lcc=not args.whole_graph, checkpoint=write)
    write(row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
