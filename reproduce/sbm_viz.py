"""Coarsened SBM per method: Figure 1 (r = 0.7) and Figures 2 and 3 (r = 0.5, 0.9) of the GraCo paper.

    python -m graco.reproduce.sbm_viz

A 5x4 grid per ratio. The first cell is the SBM, nodes coloured by community.
Each other cell is the coarsened graph of one method: one node per supernode
at the centroid of its members, sized by their count, coloured by its majority
community and greyed the more it mixes; the edges are A_c.

Two sets per ratio: every method at its published default, and a tuned set
where the methods tuned by sbm_tuning.py run its best configuration, read
from results/sbm_tuning.json, and carry a star. The features are
smooth_random, X ~ N(0, L^+), as in the synthetic experiments of FGC. Ten of
the methods do not read X: Random, METIS, Graclus, Leiden, local variation,
both GW, SGWL, GBGC and STPGC.

Output: results/sbm_coarse_{default,tuned}_r{050,070,090}.png
"""
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from graco.reproduce.scaling import (default_results_dir, device_kwargs,
                                        pick_device)

OUT = default_results_dir()
OUT.mkdir(parents=True, exist_ok=True)

from graco.loaders import make_sbm
from torch_geometric.utils import to_networkx
from graco.algorithms.random import coarsen_random
from graco.algorithms.metis import coarsen_metis
from graco.algorithms.leiden import coarsen_leiden
from graco.algorithms.local_variation.main import coarsen_local_variation
from graco.algorithms.fgc import coarsen_fgc
from graco.algorithms.ipe import coarsen_ipe
from graco.algorithms.ugc import coarsen_ugc
from graco.algorithms.kcenter import coarsen_kcenter
from graco.algorithms.herding import coarsen_herding
from graco.algorithms.convmatch import coarsen_convmatch
from graco.algorithms.nope import coarsen_nope
from graco.algorithms.gw import coarsen_gw_gpc, coarsen_gw_kgpc
from graco.algorithms.gw_sgwl import coarsen_gw_sgwl
from graco.algorithms.otcoarsening import coarsen_otcoarsening
from graco.algorithms.gbgc import coarsen_gbgc
from graco.algorithms.stpgc import coarsen_stpgc
try:
    from torch_cluster import graclus_cluster  # the kernel the wrapper calls
    from graco.algorithms.graclus import coarsen_graclus
    HAS_GRACLUS = True
except Exception:
    HAS_GRACLUS = False

data = make_sbm(n_communities=3, n_nodes=500, p_in=0.12, p_out=0.005,
                features="smooth_random", n_features=10, seed=0)
N = data.num_nodes
comm = data.block.numpy()
data.y = data.block
G = to_networkx(data, to_undirected=True)
layout = nx.spring_layout(G, seed=1, k=0.15)
pos = np.array([layout[i] for i in range(N)])
base = np.array([[0.85, 0.20, 0.20], [0.20, 0.65, 0.25], [0.20, 0.40, 0.85]])
grey = np.array([0.6, 0.6, 0.6])

# (label, function, kwargs, key in results/sbm_tuning.json). Every entry runs
# at its published default; the tuned set swaps in what sbm_tuning.py found.
methods = [("Random", coarsen_random, {}, None), ("METIS", coarsen_metis, {}, None),
           ("Graclus", coarsen_graclus if HAS_GRACLUS else None, {}, None)]
# Panels in the family order of the paper (Sec. 4).
methods += [
    ("Leiden", coarsen_leiden, {}, None),
    ("UGC", coarsen_ugc, {}, "ugc"),
    ("GW-KGPC", coarsen_gw_kgpc, {}, None),
    ("Kcenter", coarsen_kcenter, {}, None), ("Herding", coarsen_herding, {}, None),
    ("OTCoarsen", coarsen_otcoarsening, {}, None),
    ("Local Var.", coarsen_local_variation, {"progress": False}, None),
    ("ConvMatch", coarsen_convmatch, {}, None),
    ("NOPE", coarsen_nope, {}, None),   # the cost of the paper
    ("NOPE*", coarsen_nope, {"variant": "nope_star"}, None),
    ("GW-GPC", coarsen_gw_gpc, {}, None),
    ("STPGC", coarsen_stpgc, {"label_guide": False}, "stpgc"),
    ("SGWL", coarsen_gw_sgwl, {}, None),
    ("GBGC", coarsen_gbgc, {}, None),
    ("FGC", coarsen_fgc, {}, "fgc"),
    ("IPE", coarsen_ipe, {}, "ipe"),   # Grassmann term with the sign of the derivative
]


def load_tuned():
    """Best kwargs per method, as sbm_tuning.py measured them. {} when absent."""
    path = OUT / "sbm_tuning.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    out = {}
    for key, entry in blob.get("methods", {}).items():
        kw = entry.get("tuned_kwargs")
        if isinstance(kw, str):        # json.dumps(default=str) on an odd value
            continue
        if kw:
            out[key] = kw
    return out

def coarse(assignment):
    a = np.asarray(assignment); sids = np.unique(a)
    remap = {s: i for i, s in enumerate(sids)}; n = len(sids)
    cent = np.zeros((n, 2)); size = np.zeros(n); col = np.zeros((n, 3)); tot = pure = 0
    for s in sids:
        m = np.where(a == s)[0]; i = remap[s]
        cent[i] = pos[m].mean(0); size[i] = len(m)
        h = np.bincount(comm[m], minlength=3); p = h.max() / h.sum()
        col[i] = p * base[h.argmax()] + (1 - p) * grey; tot += len(m); pure += p * len(m)
    Ac = {}
    for u, v in G.edges():
        x, y = remap[a[u]], remap[a[v]]
        if x != y:
            k = (min(x, y), max(x, y)); Ac[k] = Ac.get(k, 0) + 1
    return cent, size, col, Ac, pure / tot

def frame(ax):
    # same padded window on every panel, so a big supernode on the edge is not cut
    lo, hi = pos.min(0), pos.max(0)
    pad = 0.12 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])

def draw(ax, cent, size, col, Ac):
    for (x, y), w in Ac.items():
        ax.plot([cent[x, 0], cent[y, 0]], [cent[x, 1], cent[y, 1]],
                color="0.5", lw=min(2.5, 0.12 * w), alpha=0.25, zorder=1)
    ax.scatter(cent[:, 0], cent[:, 1], s=12 + 7 * size, c=col, zorder=2, linewidths=0)
    frame(ax)

DEVICE = pick_device()


def panel_grid(r, tuned_map):
    """One 5x4 grid at ratio r; tuned_map holds the tuned kwargs, empty for the default set."""
    fig, axes = plt.subplots(5, 4, figsize=(4 * 3.4, 5 * 3.0)); axes = axes.ravel()
    # cell 0 : original
    ax = axes[0]
    for u, v in G.edges():
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], color="0.5",
                lw=0.3, alpha=0.08, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=14, c=base[comm], zorder=2, linewidths=0)
    frame(ax)
    ax.set_title(f"Original  N={N}", fontsize=12, fontweight="bold"); ax.axis("off")

    for ax, (name, fn, kw, key) in zip(axes[1:], methods):
        if fn is None:
            ax.text(0.5, 0.5, "Graclus\n(needs torch-cluster)", fontsize=8,
                    color="0.65", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off"); continue
        use = dict(kw)
        marked = ""
        if key and key in tuned_map:
            use.update(tuned_map[key])
            marked = "*"
        try:
            part, info = fn(data, r=r, **device_kwargs(fn, DEVICE), **use)
            cent, size, col, Ac, mp = coarse(part.assignment)
            draw(ax, cent, size, col, Ac)
            ax.set_title(f"{name}{marked}  n={part.n_clusters}  pur={mp:.2f}", fontsize=12)
        except Exception as e:
            ax.set_title(f"{name}{marked}: FAIL", fontsize=12)
            ax.text(0.5, 0.5, str(e)[:42], fontsize=5, ha="center", va="center",
                    transform=ax.transAxes)
        ax.axis("off")
    for ax in axes[1 + len(methods):]:
        ax.axis("off")

    fig.tight_layout()
    tag = "tuned" if tuned_map else "default"
    out = OUT / f"sbm_coarse_{tag}_r{int(r * 100):03d}.png"
    fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
    print("saved", out, flush=True)


def main():
    print(f"coarsening on {DEVICE}", flush=True)
    tuned = load_tuned()
    if tuned:
        print(f"tuned settings read for {sorted(tuned)}", flush=True)
    else:
        print("no results/sbm_tuning.json: the tuned panels would repeat the "
              "default ones, so only the default figures are written", flush=True)

    for r in [0.5, 0.7, 0.9]:
        panel_grid(r, {})
        if tuned:
            panel_grid(r, tuned)
    print("done")


if __name__ == "__main__":
    main()
