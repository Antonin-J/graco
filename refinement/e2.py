"""Refinement of P inside E2, the generalised inverses of Q.

    E2 = { P_MP + M (I - Q P_MP) : M in R^(n x N) },   P_MP = pinv(Q)

Every P in E2 satisfies Q P Q = Q. Gradient descent runs on the dense M,
from M = 0, i.e. P = P_MP. For the RSA loss on an eigenspace of L, the loss
is convex in M (Joly et al., NeurIPS 2025). With l1_lambda > 0 the loss gains lambda ||P||_1,
the sparse variant of the paper, and the entries of the final P below
l1_threshold are zeroed, as in the paper. M and P are dense (n, N).
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from tqdm.auto import tqdm

from graco.core.coarsening import Coarsening
from graco.core.laplacian import LaplacianKind
from graco.refinement._common import (
    LossFn,
    make_rsa_loss,
    replace_P_in_coarsening,
)


def refine_p_e2(
    coarsening: Coarsening,
    original,
    *,
    loss_fn: Optional[LossFn] = None,
    K: int = 100,
    laplacian_kind: Optional[LaplacianKind] = None,
    n_iters: int = 200,
    lr: float = 1e-2,
    optim_name: str = "SGD",
    l1_lambda: float = 0.0,
    l1_threshold: float = 1e-3,
    device: Optional[torch.device] = None,
    progress: bool = False,
    seed: int = 42,
    keep_loss_history: bool = True,
) -> tuple[Coarsening, dict]:
    """Refine P inside E2 by gradient descent on M.

    Parameters
    ----------
    coarsening : Coarsening
        Its P is P_MP, the starting point.
    original : torch_geometric.data.Data
    loss_fn : callable, optional
        loss_fn(P, Q), differentiable in P, called with a dense P and a dense
        Q. None uses the RSA on K eigenvectors (make_rsa_loss).
    K : int, default 100
        Dimension of the default RSA loss.
    laplacian_kind : str, optional
        Laplacian of the default RSA loss; None uses coarsening.laplacian_kind.
    n_iters : int, default 200
    lr : float, default 1e-2
    optim_name : {"SGD", "Adam", "RMSprop"}, default "SGD"
        SGD runs with momentum 0.9. Joly et al. (NeurIPS 2025) found Adam less stable here.
    l1_lambda : float, default 0.0
        Weight of ||P||_1, the sum of |P_ij|. 0 disables it.
    l1_threshold : float, default 1e-3
        With l1_lambda > 0, entries of the final P below it are zeroed.
    device : torch.device, optional
        None uses the device of coarsening.Q.
    progress : bool, default False
        Show a progress bar.
    seed : int, default 42
        Seed of the ARPACK starting vector of the default RSA loss, the only
        random draw of the refinement. Unused with a custom loss_fn.
    keep_loss_history : bool, default True

    Returns
    -------
    coarsening : Coarsening
        Copy with the refined P, method suffixed with "_e2".
    info : dict
        method : "refine_p_e2".
        admissible_set : "E2".
        loss_fn_name : name of the loss, "rsa_loss_K<K>" for the default.
        K : the K of the default RSA loss, None for a custom loss.
        n_iters, lr, optim_name, seed : the arguments used.
        laplacian_kind : the convention used.
        loss_init, loss_final : loss at P_MP and at the returned P.
        loss_improvement : loss_init - loss_final.
        loss_history : loss at each iteration, None if keep_loss_history=False.
        device : the device used.
        total_time_s : wall time of the call.
        notes : a reminder of the parametrisation.
    l1_lambda : the L1 weight used.
    l1_history : ||P||_1 at each iteration, None if l1_lambda = 0 or
    keep_loss_history=False.
    nnz_final : number of entries of the returned P above 1e-12 in
    absolute value.
    sparsity : fraction of zero entries of the returned P.
    """
    t_total = time.perf_counter()

    dev = torch.device(device) if device is not None else coarsening.Q.device

    Q = coarsening.Q.to_dense().to(torch.float32).to(dev)                   # (N, n), for loss_fn
    Q_csr = coarsening.Q.to(torch.float32).to(dev)                          # (N, n)
    P_MP = coarsening.P.to(torch.float32).to(dev)                           # (n, N) CSR
    N, n = Q.shape

    if loss_fn is None:
        loss_fn = make_rsa_loss(coarsening, original, K, laplacian_kind=laplacian_kind, device=dev,
                                v0_seed=seed)

    M = torch.zeros(n, N, dtype=torch.float32, device=dev, requires_grad=True)

    if optim_name == "Adam":
        opt = torch.optim.Adam([M], lr=lr)
    elif optim_name == "SGD":
        opt = torch.optim.SGD([M], lr=lr, momentum=0.9)
    elif optim_name == "RMSprop":
        opt = torch.optim.RMSprop([M], lr=lr)
    else:
        raise ValueError(f"optim_name must be Adam/SGD/RMSprop, got {optim_name!r}")

    with torch.no_grad():
        P0 = P_MP + M - (M @ Q_csr) @ P_MP
        loss_init = float(loss_fn(P0, Q).item())

    loss_history: list[float] = []
    l1_history: list[float] = [] if l1_lambda > 0 else None
    iterator = tqdm(range(n_iters), disable=not progress, desc="E2 refine")
    for _ in iterator:
        opt.zero_grad()
        # P_MP + M (I - Q P_MP) expanded, so no N x N matrix is formed
        P = P_MP + M - (M @ Q_csr) @ P_MP                                  # (n, N)
        loss_data = loss_fn(P, Q)
        loss = loss_data
        if l1_lambda > 0:
            l1_term = P.abs().sum()
            loss = loss + l1_lambda * l1_term
        loss.backward()
        opt.step()
        if keep_loss_history:
            loss_history.append(float(loss_data.item()))
            if l1_lambda > 0:
                l1_history.append(float(l1_term.item()))
        if progress:
            iterator.set_postfix_str(f"loss={loss_data.item():.4f}")

    with torch.no_grad():
        P_final = P_MP + M - (M @ Q_csr) @ P_MP
        # Final thresholding of the L1 variant, as in the paper.
        if l1_lambda > 0:
            mask = P_final.abs() >= l1_threshold
            P_final = P_final * mask
        loss_final = float(loss_fn(P_final, Q).item())
        nnz_final = int((P_final.abs() > 1e-12).sum().item())
        sparsity = 1.0 - nnz_final / P_final.numel()

    coarsening_refined = replace_P_in_coarsening(
        coarsening, P_final.detach(), method_suffix="e2",
    )

    info = {
        "method": "refine_p_e2",
        "admissible_set": "E2" if l1_lambda == 0 else "E2, then thresholded",
        "loss_fn_name": getattr(loss_fn, "__name__", "custom"),
        "K": int(K) if getattr(loss_fn, "__name__", "").startswith("rsa_loss") else None,
        "n_iters": int(n_iters),
        "lr": float(lr),
        "optim_name": optim_name,
        "laplacian_kind": laplacian_kind or coarsening.laplacian_kind,
        "loss_init": loss_init,
        "loss_final": loss_final,
        "loss_improvement": loss_init - loss_final,
        "loss_history": loss_history if keep_loss_history else None,
        "l1_lambda": float(l1_lambda),
        "l1_history": l1_history if keep_loss_history else None,
        "nnz_final": nnz_final,
        "sparsity": sparsity,
        "seed": seed,
        "device": str(dev),
        "total_time_s": time.perf_counter() - t_total,
        "notes": (
            "E2: P = P_MP + M (I - Q P_MP), M dense (n, N), from M = 0."
        ),
    }
    return coarsening_refined, info
