"""Coarsening metrics.

structural : check_coarsening, cluster_size_distribution, max_acc,
             modularity, granular_ball_quality, partition_density_quality
spectral   : rsa, rsa_features, mean_REE, spectral_distance
energy     : dirichlet_energy, dirichlet_energy_error, hyperbolic_error,
             reconstruction_error, fgc_loss, convolution_matching_error,
             inner_product_error, ipe_grassmann_loss,
             neighborhood_interference_index
transport  : wasserstein_feature_distance, gromov_wasserstein_distortion,
             d_gw_paper, d_gw_op, d_gw_struct

A metric that compares the two graphs takes the Coarsening first and the
original graph second, e.g. rsa(coarsening, original). Each docstring names
the paper the metric comes from.
"""

from graco.metrics.energy import (
    convolution_matching_error,
    dirichlet_energy,
    dirichlet_energy_error,
    fgc_loss,
    hyperbolic_error,
    inner_product_error,
    ipe_grassmann_loss,
    neighborhood_interference_index,
    reconstruction_error,
)
from graco.metrics.spectral import mean_REE, rsa, rsa_features, spectral_distance
from graco.metrics.structural import (
    check_coarsening,
    cluster_size_distribution,
    granular_ball_quality,
    max_acc,
    modularity,
    partition_density_quality,
)
from graco.metrics.transport import (
    d_gw_op,
    d_gw_paper,
    d_gw_struct,
    gromov_wasserstein_distortion,
    wasserstein_feature_distance,
)

__all__ = [
    # structural
    "check_coarsening",
    "cluster_size_distribution",
    "granular_ball_quality",
    "max_acc",
    "modularity",
    "partition_density_quality",
    # spectral
    "rsa",
    "rsa_features",
    "mean_REE",
    "spectral_distance",
    # energy
    "dirichlet_energy",
    "dirichlet_energy_error",
    "hyperbolic_error",
    "reconstruction_error",
    "fgc_loss",
    "convolution_matching_error",
    "inner_product_error",
    "ipe_grassmann_loss",
    "neighborhood_interference_index",
    # transport
    "wasserstein_feature_distance",
    "gromov_wasserstein_distortion",
    "d_gw_paper",
    "d_gw_op",
    "d_gw_struct",
]
