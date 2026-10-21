"""IPE on the three-community SBM with its Grassmann rank set to 3: Figure 4 of the GraCo paper (App. E.2).

The SBM of sbm_viz.py, at r = 0.9, i.e. 50 supernodes requested. IPE runs
the best raw configuration of the SBM grid of sbm_tuning.py (SINGC, lr 0.02,
alpha 0.1, lam 0.1) with its Grassmann rank set from 10 to 3. It then returns
three community-pure supernodes; the script checks it over five seeds and
draws the default seed next to the original graph.

Run:  python -m graco.reproduce.ipe_k_effect
Output: results/ipe_k_effect.png and results/ipe_k_effect.json
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

from graco.algorithms.ipe import coarsen_ipe
from graco.loaders import make_sbm
from graco.reproduce.scaling import default_results_dir, write_json

r = 0.9
k_grassmann = 3
ipe_kwargs = dict(variant="SINGC", lr=0.02, alpha=0.1, lam=0.1)
seeds = (14, 0, 1, 2, 3)
colors = np.array([[0.85, 0.20, 0.20], [0.20, 0.65, 0.25], [0.20, 0.40, 0.85]])


def main() -> None:
    data = make_sbm(n_communities=3, n_nodes=500, p_in=0.12, p_out=0.005,
                    features="smooth_random", n_features=10, seed=0)
    comm = data.block.numpy()
    runs = []
    for seed in seeds:
        partition, _ = coarsen_ipe(data, r=r, k_grassmann=k_grassmann, seed=seed,
                                   progress=False, **ipe_kwargs)
        a = partition.assignment
        purity = sum(np.bincount(comm[a == s]).max() for s in np.unique(a)) / len(comm)
        runs.append({"seed": seed, "n_clusters": int(partition.n_clusters),
                     "purity": float(purity), "assignment": a})
        print(f"seed={seed}  n={partition.n_clusters}  purity={purity:.2f}", flush=True)

    G = to_networkx(data, to_undirected=True)
    layout = nx.spring_layout(G, seed=1, k=0.15)
    pos = np.array([layout[i] for i in range(data.num_nodes)])
    a = runs[0]["assignment"]
    ids = np.unique(a)
    cent = np.array([pos[a == s].mean(0) for s in ids])
    size = np.array([(a == s).sum() for s in ids])
    col = np.array([colors[np.bincount(comm[a == s], minlength=3).argmax()] for s in ids])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.2, 5.6))
    nx.draw_networkx_edges(G, pos, ax=ax0, width=0.1, alpha=0.15)
    ax0.scatter(pos[:, 0], pos[:, 1], s=12, c=colors[comm], linewidths=0)
    ax0.set_title(f"Original  N={data.num_nodes}, 3 communities")
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            ax1.plot(*zip(cent[x], cent[y]), color="0.85", lw=3, zorder=1)
    ax1.scatter(cent[:, 0], cent[:, 1], s=12 + 7 * size, c=col, zorder=2, linewidths=0)
    ax1.set_title(f"IPE, Grassmann rank = {k_grassmann} (number of communities)\n"
                  f"n={runs[0]['n_clusters']} supernodes, purity = {runs[0]['purity']:.2f}")
    for ax in (ax0, ax1):
        ax.set_xlim(*ax0.get_xlim()); ax.set_ylim(*ax0.get_ylim()); ax.axis("off")
    fig.suptitle("IPE on the SBM, its Grassmann rank set to the number of communities")
    fig.tight_layout()

    out = default_results_dir()
    fig.savefig(out / "ipe_k_effect.png", dpi=200)
    write_json(out / "ipe_k_effect.json", {
        "r": r, "k_grassmann": k_grassmann, "ipe_kwargs": ipe_kwargs,
        "runs": [{k: v for k, v in run.items() if k != "assignment"} for run in runs],
    })
    print(f"wrote {out / 'ipe_k_effect.png'}")


if __name__ == "__main__":
    main()
