"""Hyperparameter sensitivity on the three-community SBM: Tables 15 and 16 of the GraCo paper (App. E.2).

    python -m graco.reproduce.sbm_tuning
    python -m graco.reproduce.sbm_tuning --reuse fgc,ipe

Three blocks. First, every coarsener at its published default, the reference
points. Second, numeric grids for the methods below a purity of 0.90 that
have a numeric knob: STPGC, UGC, FGC and IPE, fixed in the code. NOPE is
below 0.90 too but has no such knob, and its readings go to the third block.
The list of methods below 0.90 is printed as a check. Third, the readings
the source leaves open: the paper and its released code disagree, the paper
publishes two variants, or it does not say. Our own extensions are not
swept.

The seed is never a tuned axis. STPGC runs label-free throughout, since its
default reads labels and on this graph the labels are the planted
communities. The iteration counts of FGC and IPE are in the grids, and the
best configuration at the count their authors publish is reported apart.

The graph and its features are those of sbm_viz.py, which reads the JSON
written here to draw its tuned panels. --reuse copies the named grids from
the JSON of a previous run, for when their code has not changed since; the
FGC and IPE grids take most of the time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from graco.algorithms import COARSENING_ALGOS
from graco.algorithms.common import resolve_target
from graco.algorithms.convmatch import coarsen_convmatch
from graco.algorithms.fgc import coarsen_fgc
from graco.algorithms.gbgc import coarsen_gbgc
from graco.algorithms.gw import coarsen_gw_gpc, coarsen_gw_kgpc
from graco.algorithms.gw_sgwl import coarsen_gw_sgwl
from graco.algorithms.ipe import coarsen_ipe
from graco.algorithms.nope import coarsen_nope
from graco.algorithms.otcoarsening import coarsen_otcoarsening
from graco.algorithms.stpgc import coarsen_stpgc
from graco.algorithms.ugc import coarsen_ugc
from graco.reproduce.scaling import (default_results_dir, device_kwargs,
                                        pick_device)
from graco.loaders import make_sbm

R = 0.9
DEVICE = pick_device()
SHORT_OF = 0.90          # below this at default, a method gets a numeric grid

# Label-free, for the reason in the module docstring.
STPGC_BASE = dict(label_guide=False)


def build_sbm():
    data = make_sbm(n_communities=3, n_nodes=500, p_in=0.12, p_out=0.005,
                    features="smooth_random", n_features=10, seed=0)
    data.y = data.block
    return data, data.block.numpy()


def purity(assignment, comm) -> float:
    a = np.asarray(assignment)
    return sum(np.bincount(comm[a == c]).max() for c in np.unique(a)) / len(a)


def score(fn, data, comm, **kw):
    """Purity and supernode count, or (None, None) on failure; the device goes to the coarseners that take one."""
    try:
        p, info = fn(data, r=R, **device_kwargs(fn, DEVICE), **kw)
        return purity(p.assignment, comm), info["n_clusters"]
    except Exception:
        return None, None


def block_defaults(data, comm) -> dict:
    """Every coarsener at its published default. The reference points."""
    print(f"\n{'=' * 70}\n# defaults\n{'=' * 70}", flush=True)
    out = {}
    for algo in sorted(COARSENING_ALGOS):
        kw = dict(STPGC_BASE) if algo == "stpgc" else {}
        t0 = time.perf_counter()
        pu, n = score(COARSENING_ALGOS[algo], data, comm, **kw)
        out[algo] = {"default_purity": pu, "default_n": n,
                     "default_kwargs": kw or None}
        print(f"  {algo:<14} purity={'FAIL' if pu is None else f'{pu:.4f}'}  "
              f"n={n}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    base = out.get("random", {}).get("default_purity")
    short = sorted(a for a, v in out.items()
                   if v["default_purity"] is not None
                   and v["default_purity"] < SHORT_OF and a != "random")
    if base is not None:
        print(f"\n  random baseline {base:.4f}", flush=True)
    print(f"  below {SHORT_OF}: {', '.join(short) or 'none'}", flush=True)
    return out


def sweep(name, fn, data, comm, default_kw, grid, published=None):
    """Sweep grid, and report the best configuration overall and the best under published.

    Configs are ranked on purity discounted by the supernodes they give up,

        purity * min(1, n / n_target)

    so a run that returns a fifth of the supernodes asked keeps a fifth of its
    purity. Purity alone rewards collapsed runs, since fewer supernodes are
    easier to keep pure; the undiscounted best is reported beside.

    published is a predicate on the config, true when its iteration count is the
    one of the source paper. Reported apart, so a method held back by its
    published budget can be told from one held back by its weights.
    """
    n_target = resolve_target(data.num_nodes, R, None)[0]
    pu, n = score(fn, data, comm, **default_kw)
    print(f"\n## {name}  ({len(grid)} configs)", flush=True)
    print(f"  default {default_kw or '{}'}: purity={pu} n={n}", flush=True)

    def discounted(pu_, n_):
        return pu_ * min(1.0, n_ / n_target) if n_ else pu_

    t0 = time.perf_counter()
    best = (-1.0, None, None, None)            # discounted, purity, n, kwargs
    best_raw = (-1.0, None, None)              # purity, n, kwargs
    best_pub = (-1.0, None, None, None)
    for kw in grid:
        pu2, n2 = score(fn, data, comm, **kw)
        if pu2 is None:
            continue
        disc = discounted(pu2, n2)
        if disc > best[0]:
            best = (disc, pu2, n2, kw)
        if pu2 > best_raw[0]:
            best_raw = (pu2, n2, kw)
        if published is not None and published(kw) and disc > best_pub[0]:
            best_pub = (disc, pu2, n2, kw)

    took = time.perf_counter() - t0
    if best[3] is None:
        print(f"  tuned: every config failed  ({took:.0f}s)", flush=True)
        return {"default_purity": pu, "default_n": n,
                "tuned_purity": None, "tuned_n": None, "tuned_kwargs": None,
                "n_configs": len(grid)}

    print(f"  tuned: purity={best[1]:.4f} n={best[2]} "
          f"(discounted {best[0]:.4f}) at {best[3]}  ({took:.0f}s)", flush=True)
    print(f"  best undiscounted: purity={best_raw[0]:.4f} n={best_raw[1]} "
          f"at {best_raw[2]}", flush=True)
    if published is not None and best_pub[3] is not None:
        print(f"  best at the published iteration count: purity={best_pub[1]:.4f} "
              f"n={best_pub[2]} at {best_pub[3]}", flush=True)
    return {"default_purity": pu, "default_n": n,
            "tuned_purity": best[1], "tuned_n": best[2], "tuned_kwargs": best[3],
            "tuned_discounted": best[0],
            "raw_best_purity": best_raw[0], "raw_best_n": best_raw[1],
            "raw_best_kwargs": best_raw[2],
            "published_iters_purity": best_pub[1] if best_pub[3] else None,
            "published_iters_kwargs": best_pub[3],
            "n_configs": len(grid)}


def block_grids(data, comm, reused: dict) -> dict:
    """Numeric grids for the methods whose default falls short; the grids in reused are copied."""
    print(f"\n{'=' * 70}\n# numeric grids\n{'=' * 70}", flush=True)
    out = {}

    def sweep_or_reuse(key, name, *args, **kw):
        if key in reused:
            print(f"\n## {name}  kept from the previous sbm_tuning.json", flush=True)
            return reused[key]
        return sweep(name, *args, **kw)

    out["stpgc"] = sweep_or_reuse(
        "stpgc", "STPGC", coarsen_stpgc, data, comm, dict(STPGC_BASE),
        [dict(STPGC_BASE, theta1=t, theta2_ratio=t2)
         for t, t2 in itertools.product([2, 3, 5, 8, 10, 15, 20, 30],
                                        [0.005, 0.01, 0.05])])

    out["ugc"] = sweep_or_reuse(
        "ugc", "UGC", coarsen_ugc, data, comm, dict(),
        [dict(add_adj_to_features=a, n_projectors=p, projectors_distribution=d)
         for a, p, d in itertools.product([True, False], [100, 500, 2000],
                                          ["uniform", "normal"])])

    # NOPE has no numeric knob of its own; its readings are swept in the third
    # block.

    # FGC: the grid spans the range of their notebook, alpha up to 500 with F/2
    # on gamma or lam, and n_iters, 10 being the notebook's count. Both solvers.
    # The step is swept on "metrics" only: 1e-5 as in its notebook, 1/K as in
    # the paper, 1e-3 between. "sparsity" keeps the step of its own notebook.
    K = resolve_target(data.num_nodes, R, None)[0]
    weights = list(itertools.product(
        [1., 10., 100., 500.], [5., 100., 500.],
        [0.5, 1., 2., 5., 10., 50., 100., 500.], [10, 50, 200]))
    out["fgc"] = sweep_or_reuse(
        "fgc", "FGC", coarsen_fgc, data, comm, dict(),
        [dict(variant="metrics", lr=lr, alpha=a, gamma=g, lam=l, n_iters=ni)
         for lr in [1e-5, 1e-3, 1.0 / K] for a, g, l, ni in weights]
        + [dict(variant="sparsity", alpha=a, gamma=g, lam=l, n_iters=ni)
           for a, g, l, ni in weights],
        published=lambda kw: kw["n_iters"] == (10 if kw["variant"] == "metrics" else 50))

    # IPE: t_iter is the analogous knob, 2000 for SINGC and 20 outer steps of
    # 100 inner ones for INGC, as in the paper. The step gets 1/K as for FGC;
    # 1e-5 stays out, at which INGC does not move on this graph.
    out["ipe"] = sweep_or_reuse(
        "ipe", "IPE", coarsen_ipe, data, comm, dict(),
        [dict(variant=v, lr=lr, alpha=a, lam=l, k_grassmann=k, t_iter=t)
         for v, lr, a, l, k, t in itertools.product(
             ["SINGC", "INGC"], [1e-3, 1e-4, 1.0 / K], [0.1, 1., 10., 100.],
             [0.1, 1., 100.], [10, 50, 100], [None, 5000])],
        published=lambda kw: kw["t_iter"] is None)

    return out


def block_readings(data, comm) -> dict:
    """Sweep the switches where the source leaves a choice, each value one of its readings.

    A switch belongs here when the method cannot run without picking one: the
    paper and its released code disagree, the paper publishes two variants, or
    the paper does not say. Our own extensions stay out; NOPE's similarity_kind
    is the test case, since only "euclidean" is theirs. These methods already do
    well by default, so the question is whether the reading we ship costs
    anything, and whether the ones that miss the requested count can reach it.
    """
    print(f"\n{'=' * 70}\n# readings the source leaves open\n{'=' * 70}",
          flush=True)
    out = {}

    # The paper forms no hard partition, so a node its operator leaves
    # uncovered has to go somewhere: the three readings are forced. cut and
    # zero_row_filter set the paper against its released code.
    out["otcoarsening"] = sweep(
        "OTCoarsening", coarsen_otcoarsening, data, comm, dict(),
        [dict(uncovered=u, cut=c, zero_row_filter=z)
         for u, c, z in itertools.product(
             ["attach", "isolate", "isolate_recursive"],
             ["raw", "relu"], ["none", "repo"])])

    # Two published variants: the GBGC paper against the later work of the same
    # authors, which changed the initialisation, the acceptance rule, and
    # dropped the transitivity term.
    out["gbgc"] = sweep(
        "GBGC", coarsen_gbgc, data, comm, dict(),
        [dict(init=i, accept=a, use_transitivity=t)
         for i, a, t in itertools.product(["bfs", "metis"], ["sum", "mean"],
                                          [True, False])])

    # The exact cost and its bound, both in the paper. pair and
    # merge_candidates stay out: only their default is the paper's.
    out["convmatch"] = sweep(
        "ConvMatch", coarsen_convmatch, data, comm, dict(),
        [dict(method=m) for m in ["approximate", "exact"]])

    # Three readings of their interference index: the paper, and the two
    # routines of the released code.
    out["nope_variant"] = sweep(
        "NOPE variant", coarsen_nope, data, comm, dict(),
        [dict(variant=v) for v in ["nope", "nope_repo", "nope_star"]])

    # The adjacency against the signless Laplacian, both in the paper; the other
    # values of S_kind, pi_kind and sc_mode are ours and stay out.
    for name, fn, key in (("GW-GPC", coarsen_gw_gpc, "gw_gpc"),
                          ("GW-KGPC", coarsen_gw_kgpc, "gw_kgpc")):
        out[key] = sweep(
            name, fn, data, comm, dict(),
            [dict(S_kind=s) for s in ["adjacency", "signless_laplacian"]])

    # Neither of their stopping rules targets a ratio: the paper recurses to a
    # fixed depth, the repository stops on a size threshold. "count" is ours,
    # to compare at a fixed n.
    out["sgwl"] = sweep(
        "SGWL", coarsen_gw_sgwl, data, comm, dict(),
        [dict(stop_rule=s) for s in ["count", "size"]])

    # Their notebook puts F/2 on gamma, the appendix of their paper puts it on
    # lambda, with the same three values otherwise.
    out["fgc_weights_source"] = sweep(
        "FGC weights_source", coarsen_fgc, data, comm, dict(),
        [dict(weights_source=w) for w in ["code", "paper"]])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", default="",
                        help="grids copied from the previous results/sbm_tuning.json, "
                             "comma-separated among stpgc, ugc, fgc, ipe")
    args = parser.parse_args()
    path = default_results_dir() / "sbm_tuning.json"
    keys = [k for k in args.reuse.split(",") if k]
    reused = {}
    if keys:
        previous = json.loads(path.read_text())["methods"]
        reused = {k: previous[k] for k in keys}

    data, comm = build_sbm()
    print(f"SBM N={data.num_nodes} F={data.x.shape[1]} r={R}", flush=True)

    blob = {"r": R}
    blob["defaults"] = block_defaults(data, comm)
    blob["methods"] = block_grids(data, comm, reused)
    blob["readings"] = block_readings(data, comm)

    # sbm_viz.py reads this back, so its tuned panels use the settings of the
    # table.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n# wrote {path}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
