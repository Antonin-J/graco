"""Helpers shared by the metrics, not part of the public API.

_spectral_cache runs the full eigendecomposition L = U Lambda U^T once, in
O(N^3), and stores only the eigenvalues and U. L^{1/2} X and L^{+1/2} X are
then applied in factor form, U (f(lambda) * (U^T X)), so no other dense
(N, N) matrix is built.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import torch

from graco.core.laplacian import LaplacianKind, build_laplacian

ZERO_EIG_TOL = 1e-7


class SpectralCache(TypedDict):
    """Eigendecomposition U Lambda U^T of L, with sqrt and inverse sqrt of the eigenvalues."""
    eigvals: torch.Tensor       # (N,)
    eigvecs: torch.Tensor       # (N, N)
    sqrt_eigvals: torch.Tensor  # (N,) sqrt(lambda), zero on the null space
    inv_sqrt_eigvals: torch.Tensor  # (N,) 1/sqrt(lambda), zero on the null space


def _spectral_cache(
    graph,
    laplacian_kind: LaplacianKind = "combinatorial",
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> SpectralCache:
    """Full eigendecomposition of L, float64 by default.

    float64 because 1/sqrt(lambda) amplifies the error on near-null
    eigenvalues, which float32 leaves large enough to swamp the RSA.
    Eigenvalues below 1e-7 in absolute value, e.g. the constant eigenvector of
    the combinatorial Laplacian, get zero in both sqrt and inverse sqrt, so the
    metrics built on them ignore the null space.
    """
    L_torch = build_laplacian(graph, laplacian_kind, dtype=dtype)
    if device is not None:
        L_torch = L_torch.to(device)
    eigvals, eigvecs = torch.linalg.eigh(L_torch.to_dense())

    zero_mask = eigvals.abs() < ZERO_EIG_TOL
    sqrt_eigvals = torch.where(zero_mask, torch.zeros_like(eigvals), eigvals.clamp(min=0).sqrt())
    inv_sqrt_eigvals = torch.where(
        zero_mask,
        torch.zeros_like(eigvals),
        eigvals.clamp(min=ZERO_EIG_TOL).rsqrt(),
    )
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "sqrt_eigvals": sqrt_eigvals,
        "inv_sqrt_eigvals": inv_sqrt_eigvals,
    }


def apply_L_sqrt(cache: SpectralCache, X: torch.Tensor) -> torch.Tensor:
    """L^{1/2} X in factor form, X dense (N, k)."""
    U = cache["eigvecs"]
    sqrt_lam = cache["sqrt_eigvals"]
    return U @ (sqrt_lam.unsqueeze(1) * (U.T @ X))


def apply_L_inv_sqrt(cache: SpectralCache, X: torch.Tensor) -> torch.Tensor:
    """L^{+1/2} X in factor form, X dense (N, k)."""
    U = cache["eigvecs"]
    inv_sqrt_lam = cache["inv_sqrt_eigvals"]
    return U @ (inv_sqrt_lam.unsqueeze(1) * (U.T @ X))


def _to_dense_NN(M: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Dense torch (N, N) copy of a torch CSR, dense or scipy sparse matrix."""
    if isinstance(M, torch.Tensor):
        if M.layout == torch.sparse_csr:
            return M.to_dense().to(dtype=dtype)
        return M.to(dtype=dtype)
    import scipy.sparse as sp
    if sp.issparse(M):
        from graco.core.conversions import scipy_csr_to_torch_csr
        return scipy_csr_to_torch_csr(M.tocsr()).to_dense().to(dtype=dtype)
    raise TypeError(f"Cannot convert {type(M).__name__} to dense torch tensor")
