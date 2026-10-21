"""Closed-form reduction P_opt = L_c^+ Q^T L, with L_c = Q^T L Q.

Joly et al. (NeurIPS 2025) derive P_opt from an
unconstrained problem in which the preserved subspace R does not appear,
and expects it to reach a better RSA than P_MP. P_opt lies in E1 but not in
E2, as it is not full rank. It is dense in general: L and Q stay sparse,
while L_c, its pseudo-inverse, in O(n^3), and P_opt are dense.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from graco.core.coarsening import Coarsening
from graco.core.conversions import sparse_matmul
from graco.core.laplacian import LaplacianKind, build_laplacian
from graco.refinement._common import (
    make_rsa_loss,
    replace_P_in_coarsening,
)


def compute_p_opt_e1_rsa(
    coarsening: Coarsening,
    original,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    device: Optional[torch.device] = None,
    report_rsa: bool = True,
    K: int = 100,
) -> tuple[Coarsening, dict]:
    """Replace P by P_opt = L_c^+ Q^T L.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.
    device : torch.device, optional
        None uses the device of coarsening.Q.
    report_rsa : bool, default True
        If True, also compute the RSA of P_opt on K eigenvectors, one more
        ARPACK call.
    K : int, default 100
        Dimension of the subspace of that RSA.

    Returns
    -------
    coarsening : Coarsening
        Copy with P = P_opt, method suffixed with "_e1_opt".
    info : dict
        method : "compute_p_opt_e1_rsa".
        admissible_set : "E1".
        laplacian_kind : the convention used.
        device : the device used.
        total_time_s : wall time of the closed form.
        notes : a reminder of the formula.
        K, rsa_e1_optimum : with report_rsa, K and the RSA of P_opt.
    """
    t_total = time.perf_counter()

    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    dev = torch.device(device) if device is not None else coarsening.Q.device

    Q = coarsening.Q.to(torch.float64).to(dev)                               # (N, n) CSR
    L = build_laplacian(original, laplacian_kind, dtype=torch.float64).to(dev)

    # L and Q stay CSR; only L_c (n, n) and P_opt (n, N) are dense.
    Qt = Q.transpose(0, 1).to_sparse_csr()                                   # (n, N)
    L_c = sparse_matmul(Qt, sparse_matmul(L, Q)).to_dense()  # Q^T L Q, MKL case in needs_coo
    P_opt_64 = torch.linalg.pinv(L_c) @ sparse_matmul(Qt, L)  # (n, N)

    P_opt = P_opt_64.to(torch.float32)
    Q_f32 = Q.to(torch.float32)

    info: dict = {
        "method": "compute_p_opt_e1_rsa",
        "admissible_set": "E1",
        "laplacian_kind": laplacian_kind,
        "device": str(dev),
        "total_time_s": time.perf_counter() - t_total,
        "notes": (
            "Closed-form P_opt = L_c^+ Q^T L. In E1 but not in E2 in general, "
            "as it is not full rank. Dense."
        ),
    }

    if report_rsa:
        rsa_fn = make_rsa_loss(coarsening, original, K, laplacian_kind=laplacian_kind, device=dev)
        with torch.no_grad():
            rsa_e1 = float(rsa_fn(P_opt, Q_f32).item())
        info["K"] = int(K)
        info["rsa_e1_optimum"] = rsa_e1

    coarsening_refined = replace_P_in_coarsening(
        coarsening, P_opt.detach(), method_suffix="e1_opt",
    )
    return coarsening_refined, info
