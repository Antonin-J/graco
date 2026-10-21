"""Refinement of the reduction matrix P of an existing coarsening.

The partition, and so Q, stays fixed; only P changes. Following Joly et al. (NeurIPS 2025), P
can be sought in three nested sets, E3 in E2 in E1:

    E1  {P : (QP)^2 = QP}, QP a projection. Holds P_opt = L_c^+ Q^T L with
        L_c = Q^T L Q, computed in closed form by
        graco.refinement.e1.compute_p_opt_e1_rsa.
    E2  the generalised inverses of Q, {P_MP + M (I - Q P_MP)} with
        P_MP = pinv(Q); refine_p_e2 runs gradient descent on the dense M.
    E3  the P with the support of Q^T and a fixed weighted row sum, which
        holds P_MP and the P of local variation; refine_p_e3 runs projected
        gradient descent on the N stored entries.

The default loss is the RSA constant on the K first eigenvectors of L,
built by make_rsa_loss; any differentiable loss_fn(P, Q) can replace it.
"""

from __future__ import annotations

from graco.refinement.e1 import compute_p_opt_e1_rsa
from graco.refinement.e2 import refine_p_e2
from graco.refinement.e3 import refine_p_e3

__all__ = ["compute_p_opt_e1_rsa", "refine_p_e2", "refine_p_e3"]
