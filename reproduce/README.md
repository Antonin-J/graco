# Reproducing the GraCo paper

One script per table or figure. Run each from the repository root; outputs
land in `reproduce/results/`, or wherever `GRACO_RESULTS_DIR` points. Install
GraCo with `pip install -e .`: after a plain `pip install .`, `reproduce/results/`
sits inside site-packages, unless `GRACO_RESULTS_DIR` is set. `sbm_viz`,
`ipe_k_effect` and `rgg_viz` draw with matplotlib, which the `viz` extra installs:
`pip install -e ".[viz]"`. Table, figure and appendix numbers refer to the
current version of the paper.

| Paper | Command | Cost |
|---|---|---|
| Table 2: runtime and memory of every method over four tiers (Sec. 2.2) | `python -m graco.reproduce.run_scaling`, then `python -m graco.reproduce.collect_scaling` | GPU; hours, with a budget of 1 h per method on Cora and PubMed, 3 h on ogbn-arxiv, 12 h on ogbn-products |
| One cell of Table 2 | `python -m graco.reproduce.scaling --dataset Cora --r 0.9 --algo metis` | minutes on Cora |
| Table 4: four metrics on every method, Cora and PubMed (App. B.5) | `python -m graco.reproduce.cross_scoring --dataset Cora`, and the same with `--dataset PubMed` | minutes on Cora, about 3 h on PubMed |
| Figures 1 to 3: coarsened SBM per method at r = 0.7, 0.5 and 0.9 (Sec. 5, App. E.1) | `python -m graco.reproduce.sbm_tuning`, then `python -m graco.reproduce.sbm_viz` | sbm_tuning: about 15 h on a GPU, for the FGC and IPE grids, which `--reuse fgc,ipe` copies from a previous run; sbm_viz: minutes |
| Tables 15 and 16: hyperparameter sensitivity on the SBM and its grids (App. E.2) | `python -m graco.reproduce.sbm_tuning` | as above |
| Figure 4: IPE with Grassmann rank 3 on the SBM (App. E.2) | `python -m graco.reproduce.ipe_k_effect` | about a minute |
| Figure 5 and Table 17: coarsened random geometric graph per method, with its Voronoi agreement and connectivity (App. E.3), and the image of the README | `python -m graco.reproduce.rgg_viz` | a few minutes on CPU |
| Table 8: OTCoarsening and its uncovered nodes on Cora (App. C.3) | `python -m graco.reproduce.otcoarsening_cora`, `--device cuda` for a GPU | about 30 min on CPU |
| Table 10: ConvMatch under its three pairs (App. C.5) | `python -m graco.reproduce.convmatch_pairs --dataset Cora`, and the same with `--dataset PubMed` | about a minute on Cora; GPU advised on PubMed |
| Table 12: SGWL under its two stopping rules (App. C.8) | `python -m graco.reproduce.sgwl_stop` | about a minute |

Every method runs at the defaults of its paper or of its released code,
except in `sbm_tuning`, whose point is to tune them. The NOPE* cell of
ogbn-products in Table 2 overran the 12 h budget of `run_scaling`; it comes
from the one-cell command, which has no budget. Datasets download on first use into
`GRACO_DATASETS_DIR`, by default `~/.cache/graco/datasets`.
The RSA calls ARPACK from a random start, so its last digit can move between
runs.
