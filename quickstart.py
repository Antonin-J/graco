"""Minimal end-to-end example.

Load a graph, coarsen it with local variation, then score that same
coarsening under metrics from different papers: the RSA constant (Loukas,
JMLR 2019), the convolution matching error of ConvMatch (Dickens et al., WWW
Companion 2024) and the label ceiling max_acc. Any coarsening pairs with any
metric.

Run:  python -m graco.quickstart
"""

from graco.loaders import get_dataset
from graco.algorithms.local_variation.main import coarsen_local_variation
from graco.core.builders import build_coarsening
from graco.metrics.spectral import rsa
from graco.metrics.energy import convolution_matching_error
from graco.metrics.structural import max_acc


def main():
    # 1. Load a graph. Loaders return a PyG Data already made undirected and
    #    reduced to its largest connected component (the preprocessing every
    #    coarsening algorithm expects).
    data = get_dataset("Cora")
    print(f"Cora: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.x.shape[1]} features")

    # 2. Coarsen. Every coarsener takes (data, r) and returns (Partition, info);
    #    here local variation at r = 0.5.
    partition, info = coarsen_local_variation(data, r=0.5, progress=False)
    print(f"Local variation r=0.5: {partition.n_clusters} supernodes "
          f"(r_actual={info['r_actual']:.3f})")

    # 3. Build the coarsening object (P, Q, coarsened graph) from the partition.
    coarsening = build_coarsening(data, partition, method="local_variation", r_requested=0.5)

    # 4. Score the SAME coarsening under metrics from different papers.
    #    Pair metrics take the original graph explicitly (the Coarsening does
    #    not keep it), so a large original can be freed after building.
    print(f"RSA              (Loukas 2019) : {rsa(coarsening, data):.4f}")
    print(f"ConvMatch error  (Dickens 2024): {convolution_matching_error(coarsening, data)['total']:.1f}")
    print(f"max_acc  (label majority bound): {max_acc(coarsening, data)['all']:.4f}")


if __name__ == "__main__":
    main()
