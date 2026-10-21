"""Coarsened random geometric graph per method: Figure 5 of the GraCo paper (App. E.3), and the image of the README.

    python -m graco.reproduce.rgg_viz

500 points uniform in the unit square, joined when closer than 0.08 (mean
degree about 9), features X ~ N(0, L^+) as for the SBM. No method sees the
positions. Each panel colours the points by supernode, shades the convex hull
of each supernode and draws its inner edges in its colour; two adjacent
supernodes never share a colour. Crosses mark the centroids. The title gives
the Voronoi agreement vor, the share of points whose nearest centroid is the
centroid of their own supernode (1 for a partition that is exactly the
Voronoi diagram of its centroids), and conn, the share of supernodes that are
connected in the graph. Every method at its published default, but UGC at
alpha = 0.5 and STPGC without labels: their defaults read the labels, which
this graph has not. Graclus needs torch-cluster.

Output: results/rgg_coarse_r{070,090}.png, results/rgg_tweet_r090.png (the
original and five methods), results/rgg_readme_r090.png (the same, lighter,
for the README) and results/rgg_summary.json (n, vor and conn of each method
at both ratios).
"""
import json
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

from graco.reproduce.scaling import default_results_dir

OUT = default_results_dir()
OUT.mkdir(parents=True, exist_ok=True)

from graco.loaders.synthetic import make_geometric
from graco.loaders.preprocess import preprocess_graph
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
    from graco.algorithms.graclus import coarsen_graclus
except Exception:
    coarsen_graclus = None

data = preprocess_graph(make_geometric(n_nodes=500, radius=0.08, features="smooth_random", n_features=10, seed=0))
pos = data.pos.numpy()
N = data.num_nodes
ei = data.edge_index.numpy()

methods = [
    ("Random", coarsen_random, {}), ("METIS", coarsen_metis, {}), ("Graclus", coarsen_graclus, {}),
    ("Leiden", coarsen_leiden, {}), ("UGC", coarsen_ugc, {"alpha": 0.5}), ("GW-KGPC", coarsen_gw_kgpc, {}),
    ("Kcenter", coarsen_kcenter, {}), ("Herding", coarsen_herding, {}),
    ("OTCoarsen", coarsen_otcoarsening, {}), ("Local Var.", coarsen_local_variation, {"progress": False}),
    ("ConvMatch", coarsen_convmatch, {}), ("NOPE", coarsen_nope, {}),
    ("NOPE*", coarsen_nope, {"variant": "nope_star"}), ("GW-GPC", coarsen_gw_gpc, {}),
    ("STPGC", coarsen_stpgc, {"label_guide": False}), ("SGWL", coarsen_gw_sgwl, {}),
    ("GBGC", coarsen_gbgc, {}), ("FGC", coarsen_fgc, {}), ("IPE", coarsen_ipe, {}),
]
tweet_methods = ["METIS", "Graclus", "Leiden", "Local Var.", "ConvMatch"]


def voronoi_agreement(a):
    labels, a = np.unique(a, return_inverse=True)
    cent = np.stack([pos[a == k].mean(0) for k in range(len(labels))])
    _, nearest = cKDTree(cent).query(pos)
    return float((nearest == a).mean()), cent, a


def connected_share(a, n):
    """Share of the supernodes that are connected in the graph."""
    parent = np.arange(N)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for u, v in zip(ei[0], ei[1]):
        if a[u] == a[v]:
            parent[find(u)] = find(v)
    roots = np.array([find(i) for i in range(N)])
    return float(np.mean([len(set(roots[a == k])) == 1 for k in range(n)]))


def coarse_colors(a, n):
    """Greedy colouring of A_c, so two adjacent supernodes never share a colour."""
    nbrs = [set() for _ in range(n)]
    for u, v in zip(a[ei[0]], a[ei[1]]):
        if u != v:
            nbrs[u].add(v)
    palette = plt.cm.tab10(np.arange(10))
    colour = -np.ones(n, dtype=int)
    for k in np.argsort([-len(s) for s in nbrs]):
        used = {colour[j] for j in nbrs[k]}
        colour[k] = next(c for c in range(n) if c not in used)
    return np.array([palette[c % 10] for c in colour])


def draw(ax, a, cent):
    colors = coarse_colors(a, len(cent))
    ax.plot(pos[ei, 0], pos[ei, 1], color="0.85", lw=0.2, zorder=0)
    for k in range(len(cent)):
        pts = pos[a == k]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=colors[k], alpha=0.18,
                        lw=0, zorder=0)
            except Exception:
                pass
    inside = a[ei[0]] == a[ei[1]]
    for u, v in zip(ei[0][inside], ei[1][inside]):
        ax.plot(pos[[u, v], 0], pos[[u, v], 1], color=colors[a[u]], lw=0.4, zorder=0.5)
    ax.scatter(pos[:, 0], pos[:, 1], s=9, c=colors[a], linewidths=0, zorder=1)
    ax.scatter(cent[:, 0], cent[:, 1], s=10, c="k", marker="x", linewidths=0.6, zorder=2)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.axis("off")


def draw_original(ax, title, fontsize):
    ax.plot(pos[ei, 0], pos[ei, 1], color="0.6", lw=0.3, zorder=0)
    ax.scatter(pos[:, 0], pos[:, 1], s=9, c="0.2", linewidths=0, zorder=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=fontsize, fontweight="bold")


def run_all(r):
    """Coarsen once per method at ratio r, so the grid and the small figures share the partitions."""
    res = {}
    for name, fn, kw in methods:
        if fn is None:
            res[name] = None
            print(f"r={r} {name}: needs torch-cluster", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            part, _ = fn(data, r=r, **kw)
            agree, cent, a = voronoi_agreement(np.asarray(part.assignment))
            conn = connected_share(a, len(cent))
            res[name] = dict(a=a, cent=cent, n=len(cent), vor=agree, conn=conn)
            print(f"r={r} {name}: n={len(cent)} vor={agree:.2f} conn={conn:.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
        except Exception as e:
            res[name] = None
            print(f"r={r} {name}: FAIL {str(e)[:200]}", flush=True)
    return res


def panel_grid(res, r):
    fig, axes = plt.subplots(5, 4, figsize=(4 * 3.2, 5 * 3.4)); axes = axes.ravel()
    draw_original(axes[0], f"Original  N={N}, E={ei.shape[1] // 2}", 11)
    for ax, (name, _, _) in zip(axes[1:], methods):
        x = res[name]
        if x is None:
            ax.set_title(f"{name}: FAIL", fontsize=11); ax.axis("off"); continue
        draw(ax, x["a"], x["cent"])
        ax.set_title(f"{name}  n={x['n']}  vor={x['vor']:.2f}  conn={x['conn']:.2f}", fontsize=10)
    for ax in axes[1 + len(methods):]:
        ax.axis("off")
    fig.tight_layout()
    out = OUT / f"rgg_coarse_r{int(r * 100):03d}.png"
    fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
    print("saved", out, flush=True)


def small_panel(res, r):
    """Original and five methods, names only: a large one for the release post, a light one for the README."""
    fig, axes = plt.subplots(2, 3, figsize=(3 * 4.0, 2 * 4.2)); axes = axes.ravel()
    draw_original(axes[0], "Original graph", 14)
    for ax, name in zip(axes[1:], tweet_methods):
        x = res[name]
        if x is None:
            ax.set_title(f"{name}: FAIL", fontsize=14); ax.axis("off"); continue
        draw(ax, x["a"], x["cent"])
        ax.set_title(name.replace("Local Var.", "Local variation"), fontsize=14)
    fig.tight_layout()
    tag = f"r{int(r * 100):03d}"
    fig.savefig(OUT / f"rgg_tweet_{tag}.png", dpi=200, bbox_inches="tight")
    readme = OUT / f"rgg_readme_{tag}.png"
    fig.savefig(readme, dpi=100, bbox_inches="tight"); plt.close(fig)
    # 256 colours is enough for flat fills and thin lines, and divides the file size by about three
    Image.open(readme).convert("RGB").quantize(256).save(readme, optimize=True)
    print("saved", OUT / f"rgg_tweet_{tag}.png", "and", readme, flush=True)


if __name__ == "__main__":
    summary = {"N": N, "E": ei.shape[1] // 2}
    for r in (0.7, 0.9):
        res = run_all(r)
        summary[str(r)] = {k: (None if v is None else {m: float(v[m]) for m in ("n", "vor", "conn")})
                           for k, v in res.items()}
        panel_grid(res, r)
        if r == 0.9:
            small_panel(res, r)
    (OUT / "rgg_summary.json").write_text(json.dumps(summary, indent=1))
    print("saved", OUT / "rgg_summary.json", flush=True)
