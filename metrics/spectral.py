"""Spectral metrics: rsa, rsa_features, mean_REE and spectral_distance."""

from __future__ import annotations

from typing import Optional

import torch

from graco.core.coarsening import Coarsening
from graco.core.eig import preserved_eigenpairs, smallest_eigenpairs
from graco.core.laplacian import LaplacianKind, build_laplacian
from graco.metrics._common import (
    SpectralCache,
    ZERO_EIG_TOL,
)


def rsa(
    coarsening: Coarsening,
    original,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    K: Optional[int] = None,
    _cache: Optional[SpectralCache] = None,
) -> float:
    """Restricted spectral approximation constant (Loukas, JMLR 2019), exact.

        epsilon = sup_{x in R, ||x||_L = 1} ||x - Q P x||_L
                = || L^{1/2} (I - QP) R Lambda_K^{-1/2} ||_2

    R holds the K eigenvectors of L with the smallest eigenvalues and Lambda_K
    those eigenvalues. Eigenvalues below 1e-7, e.g. the constant eigenvector of
    the combinatorial Laplacian, are dropped from R. Computed in float64 as
    sqrt(lambda_max(M^T L M)) with M = (I - QP) R Lambda_K^{-1/2}.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.
    K : int in (0, N], optional
        Dimension of R; None uses min(100, N - 1). A fixed K keeps the RSA
        comparable across reduction ratios; K = coarsening.n tests as many
        eigenvectors as there are supernodes.
    _cache : SpectralCache, optional
        Eigenpairs of L, ascending, at least K of them (eigvecs and
        inv_sqrt_eigvals); skips ARPACK when given.

    Returns
    -------
    float
    """
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    N = int(original.num_nodes)
    if K is None:
        K = min(100, N - 1)
    if K <= 0 or K > N:
        raise ValueError(f"K must be in (0, N]; got K={K}, N={N}")

    # Exact form: rsa = ||L^{1/2} (I - QP) R_dot||_2, R_dot = R Lambda_K^{-1/2}.
    # With M = (I - QP) R_dot of shape (N, K), ||A||_2^2 = lambda_max(A^T A) and
    # L^{1/2} L^{1/2} = L give the same value as
    #     rsa^2 = lambda_max(M^T L M)
    # L^{1/2} needs the full eigh of L (N x N), this needs K eigenpairs, a sparse
    # L and the eigvalsh of a (K, K) matrix.
    dev = coarsening.P.device
    L = build_laplacian(original, laplacian_kind, dtype=torch.float64)
    if _cache is not None:
        R = _cache["eigvecs"][:, :K]
        inv_sqrt_K = _cache["inv_sqrt_eigvals"][:K]
    else:
        eigvals, R = preserved_eigenpairs(L, K, laplacian_kind)
        inv_sqrt_K = torch.where(
            eigvals.abs() < ZERO_EIG_TOL,
            torch.zeros_like(eigvals),
            eigvals.clamp(min=ZERO_EIG_TOL).rsqrt(),
        )
    R_dot = (R * inv_sqrt_K.unsqueeze(0)).to(dtype=torch.float64, device=dev)  # (N, K)

    P = coarsening.P.to(dtype=torch.float64)
    Q = coarsening.Q.to(dtype=torch.float64)
    M = R_dot - Q @ (P @ R_dot)                                                # (N, K)
    G = M.transpose(0, 1) @ (L.to(dev) @ M)                                    # (K, K)
    return float(torch.linalg.eigvalsh(G)[-1].clamp(min=0).sqrt().item())


def rsa_features(
    coarsening: Coarsening,
    original,
    x: Optional[torch.Tensor] = None,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    skip_zero_norm: bool = True,
    norm: str = "L",
) -> dict:
    """RSA restricted to the feature columns: how well QP preserves each feature.

        epsilon_j = ||x_j - Q P x_j|| / ||x_j||

    for each column x_j of x, with the norm ||v||_L = sqrt(v^T L v) or the
    Euclidean norm. Nothing dense of size (N, N) is built.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    x : torch.Tensor (N, F), optional
        Features; None uses original.x.
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.
    skip_zero_norm : bool, default True
        If True, drop the columns of zero norm, e.g. a constant column under
        "L". If False they count as 0.
    norm : {"L", "l2"}, default "L"

    Returns
    -------
    dict
        max, mean : max and mean of epsilon_j over the kept columns, NaN when
        none is kept.
        n_used : number of columns kept.
        n_zero_norm : number of columns dropped.
        norm : the norm used.
    """
    if norm not in ("L", "l2"):
        raise ValueError(f"norm must be 'L' or 'l2', got {norm!r}")
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    if x is None:
        x = original.x
    if x is None:
        raise ValueError("rsa_features needs a feature matrix x (or original.x).")

    P = coarsening.P
    Q = coarsening.Q

    PX = P @ x                                                                          # (n, F)
    QPX = Q @ PX                                                                        # (N, F)
    diff = x - QPX                                                                      # (N, F)

    if norm == "L":
        L = build_laplacian(original, laplacian_kind, dtype=x.dtype)
        L_diff = L @ diff                                                               # (N, F)
        L_x = L @ x                                                                     # (N, F)
        num_sq = (diff * L_diff).sum(dim=0).clamp(min=0)                                # (F,)
        den_sq = (x * L_x).sum(dim=0).clamp(min=0)                                      # (F,)
    else:                                                                               # "l2"
        num_sq = (diff * diff).sum(dim=0)                                               # (F,) = ||x - QPx||^2
        den_sq = (x * x).sum(dim=0)                                                     # (F,) = ||x||^2

    keep = den_sq > 1e-12 if skip_zero_norm else torch.ones_like(den_sq, dtype=torch.bool)
    n_zero = int((~keep).sum().item())
    if not keep.any():
        return {
            "max": float("nan"), "mean": float("nan"),
            "n_used": 0, "n_zero_norm": n_zero, "norm": norm,
        }

    num = num_sq[keep].sqrt()
    den = den_sq[keep].sqrt()
    # Columns of zero norm, kept when skip_zero_norm is False, count as 0.
    eps = torch.where(den > 1e-6, num / den, torch.zeros_like(num))
    return {
        "max": float(eps.max().item()),
        "mean": float(eps.mean().item()),
        "n_used": int(keep.sum().item()),
        "n_zero_norm": n_zero,
        "norm": norm,
    }


def mean_REE(
    coarsening: Coarsening,
    original,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    m: int = 100,
    skip_trivial: bool = True,
    which: str = "smallest",
    eigvals_L: Optional[torch.Tensor] = None,
) -> float:
    """Mean relative eigenvalue error between L and L_c.

        REE = (1/m) sum_{i=1..m} |lambda_c_i - lambda_i| / lambda_i

    L_c is the Laplacian of the coarsened graph A_c. which picks the end of the
    spectrum. "smallest" compares the m smallest eigenvalues, the low
    frequencies a coarsening is meant to preserve, as in Loukas (JMLR 2019).
    "largest" compares the m largest ones. FGC (Kumar et al., ICML 2023), UGC
    (Kataria et al., NeurIPS 2024) and IPE (Cohen et al., NeurIPS 2025) define
    REE on the "top" or "dominant" eigenvalues, after Loukas. The released
    code of FGC and UGC takes the largest ones.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    laplacian_kind : str, optional
        Laplacian convention for L and L_c; None uses coarsening.laplacian_kind.
    m : int, default 100
        Number of eigenvalues, clamped to min(N, n) - 1.
    skip_trivial : bool, default True
        With which="smallest", clamp both spectra to 1e-7, so an eigenvalue
        null in both spectra adds 0 to the sum.
    which : {"smallest", "largest"}, default "smallest"
    eigvals_L : torch.Tensor, optional
        Smallest eigenvalues of L, ascending, at least m of them; None solves
        for them. Read with which="smallest".

    Returns
    -------
    float
    """
    if which not in ("smallest", "largest"):
        raise ValueError(f"which must be 'smallest' or 'largest', got {which!r}")
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    L = build_laplacian(original, laplacian_kind, dtype=torch.float64)
    L_c = build_laplacian(coarsening.coarsened, laplacian_kind, dtype=torch.float64)
    n_actual = min(L.shape[0], L_c.shape[0]) - 1
    m_eff = min(m, n_actual)
    if m_eff < 1:
        return 0.0
    if which == "largest":
        # No null eigenvalue at this end, so a plain ARPACK "LM" call does, as
        # in the UGC code.
        import numpy as np
        import scipy.sparse.linalg as spla
        from graco.core.conversions import torch_csr_to_scipy_csr
        top_L = np.sort(spla.eigsh(torch_csr_to_scipy_csr(L), k=m_eff,
                                   which="LM", return_eigenvectors=False))
        top_Lc = np.sort(spla.eigsh(torch_csr_to_scipy_csr(L_c), k=m_eff,
                                    which="LM", return_eigenvectors=False))
        return float(np.mean(np.abs(top_Lc - top_L) / top_L))

    if eigvals_L is not None and eigvals_L.shape[0] >= m_eff:
        vals_L = eigvals_L[:m_eff].to(torch.float64).cpu()
    else:
        vals_L, _ = smallest_eigenpairs(L, m_eff, laplacian_kind)
    vals_Lc, _ = smallest_eigenpairs(L_c, m_eff, laplacian_kind)
    if skip_trivial:
        # Clamp both spectra, so a pair of null eigenvalues adds 0.
        vals_L = vals_L.clamp(min=1e-7)
        vals_Lc = vals_Lc.clamp(min=1e-7)
    return float(((vals_L - vals_Lc).abs() / vals_L).mean().item())


def spectral_distance(
    coarsening: Coarsening,
    original,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
) -> float:
    """Euclidean distance between the n smallest eigenvalues of L and L_c.

        SD = sqrt( sum_{i=1..n} (lambda_i - lambda_c_i)^2 )

    with both spectra sorted ascending, as in GBGC (Xia et al., IJCAI 2025),
    after Wilson and Zhu (Pattern Recognition 2008). GBGC compares the i-th
    eigenvalues of the two graphs, so i stops at the size n of the coarsened
    graph and the N - n largest eigenvalues of L are left out. Two dense
    eigendecompositions, O(N^3) time and O(N^2) memory.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    laplacian_kind : str, optional
        None uses coarsening.laplacian_kind.

    Returns
    -------
    float
    """
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind

    L = build_laplacian(original, laplacian_kind, dtype=torch.float64)
    L_c = build_laplacian(coarsening.coarsened, laplacian_kind, dtype=torch.float64)
    eigvals_L = torch.linalg.eigvalsh(L.to_dense()).sort().values
    eigvals_Lc = torch.linalg.eigvalsh(L_c.to_dense()).sort().values

    k = min(eigvals_L.shape[0], eigvals_Lc.shape[0])
    return float(torch.linalg.vector_norm(eigvals_L[:k] - eigvals_Lc[:k]).item())
