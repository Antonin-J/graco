"""Helpers shared by the E1, E2 and E3 refinements: the RSA loss and its sparse autograd.

A loss is any callable loss_fn(P, Q) returning a differentiable scalar, with
P (n, N) the reduction being refined and Q (N, n) the fixed lifting. The
eigenpairs and the (K, K) end of the RSA loss run in float64, since
1/sqrt(lambda) amplifies the float32 error on near-null eigenvalues; the
refinement loops stay in float32.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from graco.core.coarsening import Coarsening
from graco.core.eig import preserved_eigenpairs
from graco.core.laplacian import LaplacianKind, build_laplacian


ZERO_EIG_TOL = 1e-7

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_R_and_L(
    coarsening: Coarsening,
    original,
    K: int,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    device: Optional[torch.device] = None,
    v0_seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The scaled preserved subspace and the Laplacian for the RSA loss.

    v0_seed seeds the ARPACK starting vector; None draws it at random.

    Returns
    -------
    R_dot : torch.Tensor (N, K), float32
        The K first eigenvectors of L times diag(1/sqrt(lambda)), zero on the
        null space.
    L : torch.Tensor (N, N), sparse CSR float64
    eigvals : torch.Tensor (K,), float32
        The K smallest eigenvalues.
    """
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    dev = device or coarsening.Q.device

    L = build_laplacian(original, laplacian_kind, dtype=torch.float64)
    N = L.shape[0]
    if K > N:
        raise ValueError(f"K={K} exceeds N={N}")
    eigvals_64, eigvecs_64 = preserved_eigenpairs(L, K, laplacian_kind, v0_seed=v0_seed)

    inv_sqrt_lam = torch.where(
        eigvals_64.abs() < ZERO_EIG_TOL,
        torch.zeros_like(eigvals_64),
        eigvals_64.clamp(min=ZERO_EIG_TOL).rsqrt(),
    )
    R_dot_64 = eigvecs_64 * inv_sqrt_lam.unsqueeze(0)

    return (
        R_dot_64.to(dtype=torch.float32, device=dev),
        L.to(dev),
        eigvals_64.to(torch.float32),
    )


class CsrFromValues(torch.autograd.Function):
    """CSR tensor built from its values, whose backward reads the gradient of the stored entries only.

    torch.sparse_csr_tensor's own backward goes through a dense (n, N) buffer.
    """

    @staticmethod
    def forward(ctx, crow, col, values, size):
        ctx.save_for_backward(crow, col)
        return torch.sparse_csr_tensor(crow, col, values.detach(), size=size)

    @staticmethod
    def backward(ctx, grad):
        crow, col = ctx.saved_tensors
        same_support = (
            grad.layout == torch.sparse_csr
            and torch.equal(grad.crow_indices(), crow)
            and torch.equal(grad.col_indices(), col)
        )
        if same_support:
            return None, None, grad.values(), None
        rows = torch.repeat_interleave(
            torch.arange(crow.numel() - 1, device=crow.device), crow.diff().long(),
        )
        return None, None, grad.to_dense()[rows, col.long()], None


class CsrMatmul(torch.autograd.Function):
    """P X for a CSR P given by its arrays, with the gradient of P on its stored values only.

    Torch's backward forms grad_out X^T, a dense (n, N) matrix, then keeps the
    entries of P. Entry (i, j) only needs <grad_out[i], X[j]>, taken here for
    every stored entry at once.
    """

    @staticmethod
    def forward(ctx, crow, col, values, X):
        n = crow.numel() - 1
        rows = torch.repeat_interleave(torch.arange(n, device=X.device), crow.diff().long())
        ctx.save_for_backward(rows, col.long(), values, X)
        P = torch.sparse_csr_tensor(crow, col, values.detach(), size=(n, X.shape[0]))
        return P @ X

    @staticmethod
    def backward(ctx, grad_out):
        rows, col, values, X = ctx.saved_tensors
        grad_values = grad_X = None
        if ctx.needs_input_grad[2]:
            grad_values = (grad_out[rows] * X[col]).sum(1)                   # (nnz,)
        if ctx.needs_input_grad[3]:
            grad_X = torch.zeros_like(X).index_add_(0, col, values.unsqueeze(1) * grad_out[rows])
        return None, None, grad_values, grad_X


def rsa_diff(
    P: torch.Tensor, Q: torch.Tensor, L: torch.Tensor, R_dot: torch.Tensor,
) -> torch.Tensor:
    """Differentiable RSA on the preserved subspace.

        epsilon = || L^{1/2} (I - QP) R_dot ||_2 = sqrt(lambda_max(M^T L M)),
        M = (I - QP) R_dot

    Only (N, K) and (K, K) matrices are formed, the (K, K) one in float64. A
    CSR P goes through CsrMatmul, so its gradient stays on the stored entries.
    """
    if P.layout == torch.sparse_csr:
        PR = CsrMatmul.apply(P.crow_indices(), P.col_indices(), P.values(), R_dot)  # (n, K)
    else:
        PR = P @ R_dot                                                              # (n, K)
    M = R_dot - Q @ PR                                                              # (N, K)
    M_64 = M.to(torch.float64)
    G = M_64.transpose(0, 1) @ (L @ M_64)                                   # (K, K)
    return torch.linalg.eigvalsh(G)[-1].clamp(min=0).sqrt().to(M.dtype)


def make_rsa_loss(
    coarsening: Coarsening,
    original,
    K: int,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    device: Optional[torch.device] = None,
    v0_seed: Optional[int] = None,
) -> LossFn:
    """RSA loss_fn(P, Q) of a coarsening, with the K eigenpairs computed once.

    v0_seed seeds the ARPACK starting vector; None draws it at random.

        >>> loss_fn = make_rsa_loss(coarsening, original, K=100)
        >>> refined, info = refine_p_e2(coarsening, original, loss_fn=loss_fn)
    """
    R_dot, L, _ = build_R_and_L(
        coarsening, original, K, laplacian_kind=laplacian_kind, device=device,
        v0_seed=v0_seed,
    )

    def _rsa_loss(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        return rsa_diff(P, Q, L, R_dot)

    _rsa_loss.__name__ = f"rsa_loss_K{K}"
    return _rsa_loss


def replace_P_in_coarsening(
    coarsening: Coarsening,
    P_new: torch.Tensor,
    *,
    method_suffix: str = "refined",
) -> Coarsening:
    """Copy of the coarsening with P replaced, as sparse CSR, and method suffixed with _<method_suffix>.

    Everything else is shared with the input, including coarsened, whose x_c
    still comes from the old P.
    """
    from dataclasses import replace

    P_sparse = P_new.to_sparse_csr() if P_new.layout != torch.sparse_csr else P_new
    return replace(
        coarsening,
        P=P_sparse,
        method=f"{coarsening.method}_{method_suffix}",
    )
