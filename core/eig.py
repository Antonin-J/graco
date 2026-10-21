"""Smallest eigenpairs of a sparse Laplacian, with scipy ARPACK.

ARPACK finds extreme eigenvalues best, so the spectrum is transformed first.
The combinatorial Laplacian, whose spectrum grows with the maximal degree,
uses shift-invert. The normalised Laplacians, whose spectrum lies in [0, 2],
use the reflection omega I - L with omega = 3. Shift-invert falls back to the
reflection when its LU factorisation runs out of memory. We use scipy over
torch.lobpcg, which was slower on CPU and less stable on ill-conditioned
Laplacians.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from graco.core.conversions import torch_csr_to_scipy_csr
from graco.core.laplacian import LaplacianKind


def smallest_eigenpairs(
    L: torch.Tensor,
    k: int,
    laplacian_kind: LaplacianKind,
    delta: float = 0.0,
    v0_seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The k smallest eigenpairs of L, float64, sorted ascending.

    Parameters
    ----------
    L : torch.Tensor, sparse CSR (N, N)
        Laplacian, as built by graco.core.laplacian.build_laplacian.
    k : int in (0, N)
        Number of eigenpairs.
    laplacian_kind : {"combinatorial", "normalized", "normalized_self_loop"}
        Convention L was built with; it picks the solver.
    delta : float, default 0.0
        Added to the diagonal of L before the solve. The returned eigenvalues
        include the shift.
    v0_seed : int, optional
        Seed of the ARPACK starting vector. None draws it at random, so two calls
        can return slightly different vectors.

    Returns
    -------
    eigvals : torch.Tensor (k,)
    eigvecs : torch.Tensor (N, k)
    """
    if L.layout != torch.sparse_csr:
        raise TypeError(f"L must be torch.sparse_csr, got layout {L.layout}")
    if k <= 0 or k >= L.shape[0]:
        raise ValueError(f"k must be in (0, N); got k={k}, N={L.shape[0]}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")

    L_sp = torch_csr_to_scipy_csr(L)
    if delta > 0:
        L_sp = L_sp + delta * sp.eye(L_sp.shape[0], format="csr")

    v0 = (None if v0_seed is None
          else np.random.default_rng(v0_seed).standard_normal(L_sp.shape[0]))

    if laplacian_kind == "combinatorial":
        try:
            vals, vecs = _shift_invert(L_sp, k, v0=v0)
        except MemoryError:
            vals, vecs = _reflect(L_sp, k, omega=2.0 * L_sp.diagonal().max() + 1.0,
                                  v0=v0)
    elif laplacian_kind in ("normalized", "normalized_self_loop"):
        vals, vecs = _reflect(L_sp, k, omega=3.0, v0=v0)
    else:
        raise ValueError(f"unknown laplacian_kind {laplacian_kind!r}")

    return (
        torch.from_numpy(vals.astype(np.float64)),
        torch.from_numpy(vecs.astype(np.float64)),
    )


def preserved_eigenpairs(
    L: torch.Tensor,
    k: int,
    laplacian_kind: LaplacianKind,
    v0_seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The k smallest eigenpairs of L for the RSA, float64, ascending.

    ARPACK while k is small next to N. From k = N/4 on, the (N, k) matrices the
    RSA works with already hold N^2/4 entries, so a dense eigh costs the same
    order of memory, and it also covers k = N, which ARPACK refuses.
    """
    N = L.shape[0]
    if 4 * k < N:
        return smallest_eigenpairs(L, k, laplacian_kind, v0_seed=v0_seed)
    vals, vecs = torch.linalg.eigh(L.to(torch.float64).to_dense())
    return vals[:k], vecs[:, :k]


def _shift_invert(L: sp.csr_matrix, k: int, v0=None, sigma: float = -1e-3):
    """The k smallest eigenpairs, ARPACK on (L - sigma I)^{-1}."""
    shifted = (L - sigma * sp.eye(L.shape[0], format="csr")).tocsc()
    # Factorise here to pick the ordering: MMD_AT_PLUS_A suits a symmetric
    # matrix, scipy's default COLAMD assumes an unsymmetric one.
    lu = spla.splu(shifted, permc_spec="MMD_AT_PLUS_A")
    OPinv = spla.LinearOperator(L.shape, matvec=lu.solve, dtype=np.float64)
    vals, vecs = spla.eigsh(L, k=k, sigma=sigma, which="LM", v0=v0, OPinv=OPinv)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def _reflect(L: sp.csr_matrix, k: int, omega: float, v0=None):
    """The k smallest eigenpairs, ARPACK on omega I - L; omega must bound the largest eigenvalue."""
    M = omega * sp.eye(L.shape[0], format="csr") - L
    mu, vecs = spla.eigsh(M, k=k, which="LM", v0=v0)
    vals = omega - mu
    order = np.argsort(vals)
    return vals[order], vecs[:, order]
