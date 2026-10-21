"""Refinement of P inside E3, by projected gradient descent.

    E3 = { P : supp(P) = supp(Q^T),  sum_k P[i, k] / Delta(k) = 1 / Delta_c(i) for every i }

following Joly et al. (NeurIPS 2025). Delta is the diagonal of the Delta-Laplacian
L = Delta L_combi Delta: I for "combinatorial", D^{-1/2} for "normalized",
(D+I)^{-1/2} for "normalized_self_loop", and Delta_c the same on the
coarsened graph. Under "combinatorial" the rows of P sum to one, as for the
P of local variation. P_MP lies in E3.

P has one stored entry per original node, at the row of its cluster, so the
free parameter mu has N entries and P stays sparse CSR throughout. Each step
is a gradient step on mu, then the projection onto E3: mu is clamped to
1e-6 and each row rescaled to its target sum. At high reduction ratios the
projection can undo the step and the loss can end above that of P_MP, as
the paper also reports; a smaller lr, or E2, helps.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from tqdm.auto import tqdm

from graco.core.coarsening import Coarsening
from graco.core.laplacian import LaplacianKind
from graco.refinement._common import (
    CsrFromValues,
    LossFn,
    make_rsa_loss,
    replace_P_in_coarsening,
)


def refine_p_e3(
    coarsening: Coarsening,
    original,
    *,
    loss_fn: Optional[LossFn] = None,
    K: int = 100,
    laplacian_kind: Optional[LaplacianKind] = None,
    n_iters: int = 200,
    lr: float = 5e-2,
    optim_name: str = "SGD",
    auto_lr_decay: bool = False,
    auto_lr_decay_window: int = 20,
    auto_lr_decay_factor: float = 0.5,
    auto_lr_decay_min: float = 1e-6,
    device: Optional[torch.device] = None,
    progress: bool = False,
    seed: int = 42,
    keep_loss_history: bool = True,
) -> tuple[Coarsening, dict]:
    """Refine P inside E3 by projected gradient descent on its N stored entries.

    Parameters
    ----------
    coarsening : Coarsening
        Its P is P_MP, which gives the support and the starting values.
    original : torch_geometric.data.Data
    loss_fn : callable, optional
        loss_fn(P, Q), differentiable in P, called with a sparse CSR P and a
        sparse CSR Q. None uses the RSA on K eigenvectors (make_rsa_loss).
    K : int, default 100
        Dimension of the default RSA loss.
    laplacian_kind : str, optional
        Sets Delta, hence the projection, and the Laplacian of the default
        loss; None uses coarsening.laplacian_kind.
    n_iters : int, default 200
    lr : float, default 5e-2
        As in Joly et al. (NeurIPS 2025).
    optim_name : {"SGD", "Adam", "RMSprop"}, default "SGD"
        SGD runs with momentum 0.9; the paper found Adam less stable here.
    auto_lr_decay : bool, default False
        If True, multiply lr by auto_lr_decay_factor after
        auto_lr_decay_window iterations without improvement, down to
        auto_lr_decay_min.
    auto_lr_decay_window : int, default 20
    auto_lr_decay_factor : float, default 0.5
    auto_lr_decay_min : float, default 1e-6
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
        Copy with the refined P, method suffixed with "_e3".
    info : dict
        method : "refine_p_e3".
        admissible_set : "E3".
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
        sparse_P_storage : True.
        n_free_params_mu : N.
        auto_lr_decay : the option used.
        n_lr_decays_performed : number of lr decays.
    """
    t_total = time.perf_counter()

    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    dev = torch.device(device) if device is not None else coarsening.Q.device

    Q = coarsening.Q.to(torch.float32).to(dev)                              # (N, n) CSR
    P_MP = coarsening.P.to(torch.float32).to(dev)                           # (n, N) CSR, init = P_MP
    n_clusters, N = P_MP.shape

    # cluster_of[j] = cluster of node j (i.e., row of P where col j is nonzero)
    rows = torch.repeat_interleave(
        torch.arange(n_clusters, device=dev), P_MP.crow_indices().diff().long(),
    )
    cols = P_MP.col_indices().long()
    cluster_of = torch.full((N,), -1, dtype=torch.long, device=dev)
    cluster_of[cols] = rows
    if (cluster_of < 0).any():
        raise ValueError(
            "coarsening.P has columns with no nonzero, so the coarsening is "
            "not well-partitioned (every original node must belong to a cluster)."
        )

    # Fixed CSR structure of P.
    # For CSR row-major, col_indices must be grouped by row and increasing
    # inside it. Torch's backward returns the gradient permuted within a row
    # whose columns are out of order, without raising.
    sort_idx = torch.argsort(cluster_of * N + torch.arange(N, device=dev))  # (N,)
    col_indices_csr = sort_idx.to(torch.int32)                               # (N,) col of each nonzero
    # crow_indices: cumulative count of nonzeros per row.
    counts = torch.bincount(cluster_of, minlength=n_clusters)               # (n,)
    crow_indices = torch.zeros(n_clusters + 1, dtype=torch.int32, device=dev)
    crow_indices[1:] = counts.cumsum(0).to(torch.int32)

    # Delta, the diagonal scaling of laplacian_kind.
    delta_original, delta_coarsened = _build_delta(coarsening, original, laplacian_kind, dev)

    # mu starts at the values of P_MP.
    mu_init = torch.zeros(N, dtype=torch.float32, device=dev)
    mu_init[cols] = P_MP.values()                                            # (N,)
    mu = mu_init.clone().detach().requires_grad_(True)

    if optim_name == "Adam":
        opt = torch.optim.Adam([mu], lr=lr)
    elif optim_name == "SGD":
        opt = torch.optim.SGD([mu], lr=lr, momentum=0.9)
    elif optim_name == "RMSprop":
        opt = torch.optim.RMSprop([mu], lr=lr)
    else:
        raise ValueError(f"optim_name must be Adam/SGD/RMSprop, got {optim_name!r}")

    if loss_fn is None:
        loss_fn = make_rsa_loss(coarsening, original, K, laplacian_kind=laplacian_kind, device=dev,
                                v0_seed=seed)

    def build_P_sparse(mu_vec: torch.Tensor) -> torch.Tensor:
        """P as sparse CSR from mu, in row-major order."""
        values_csr = mu_vec[sort_idx]                                        # (N,) reordered
        return CsrFromValues.apply(
            crow_indices, col_indices_csr, values_csr, (n_clusters, N),
        )

    def project_e3(mu_vec: torch.Tensor) -> torch.Tensor:
        """Clamp mu to 1e-6, then rescale each row to sum_k P[i, k] / Delta(k) = 1 / Delta_c(i)."""
        with torch.no_grad():
            mu_pos = mu_vec.clamp(min=1e-6)
            weighted = mu_pos / delta_original
            cluster_sums = torch.zeros(n_clusters, dtype=torch.float32, device=dev)
            cluster_sums.scatter_add_(0, cluster_of, weighted)
            cluster_sums = cluster_sums.clamp(min=1e-8)
            target = 1.0 / delta_coarsened.clamp(min=1e-12)
            scale = target / cluster_sums                                    # (n,)
            return mu_pos * scale[cluster_of]

    # Start inside E3.
    with torch.no_grad():
        mu.data.copy_(project_e3(mu.data))
        P0 = build_P_sparse(mu)
        loss_init = float(loss_fn(P0, Q).item())

    loss_history: list[float] = []
    n_lr_decays = 0
    best_loss = float("inf")
    epochs_since_improvement = 0
    iterator = tqdm(range(n_iters), disable=not progress, desc="E3 refine")
    for _ in iterator:
        opt.zero_grad()
        P = build_P_sparse(mu)
        loss = loss_fn(P, Q)
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu.data.copy_(project_e3(mu.data))
        loss_val = float(loss.item())
        if keep_loss_history:
            loss_history.append(loss_val)
        if auto_lr_decay:
            if loss_val < best_loss - 1e-8:
                best_loss = loss_val
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= auto_lr_decay_window:
                current_lr = opt.param_groups[0]["lr"]
                new_lr = max(auto_lr_decay_min, current_lr * auto_lr_decay_factor)
                if new_lr < current_lr:
                    for group in opt.param_groups:
                        group["lr"] = new_lr
                    n_lr_decays += 1
                    epochs_since_improvement = 0
        if progress:
            iterator.set_postfix_str(f"loss={loss_val:.4f}")

    with torch.no_grad():
        P_final = build_P_sparse(mu)
        loss_final = float(loss_fn(P_final, Q).item())

    coarsening_refined = replace_P_in_coarsening(
        coarsening, P_final.detach(), method_suffix="e3",
    )

    info = {
        "method": "refine_p_e3",
        "admissible_set": "E3",
        "loss_fn_name": getattr(loss_fn, "__name__", "custom"),
        "K": int(K) if loss_fn.__name__.startswith("rsa_loss") else None,
        "n_iters": int(n_iters),
        "lr": float(lr),
        "optim_name": optim_name,
        "laplacian_kind": laplacian_kind,
        "loss_init": loss_init,
        "loss_final": loss_final,
        "loss_improvement": loss_init - loss_final,
        "loss_history": loss_history if keep_loss_history else None,
        "seed": seed,
        "device": str(dev),
        "sparse_P_storage": True,
        "n_free_params_mu": int(N),
        "auto_lr_decay": bool(auto_lr_decay),
        "n_lr_decays_performed": int(n_lr_decays),
        "total_time_s": time.perf_counter() - t_total,
        "notes": (
            "E3: supp(P) = supp(Q^T) and sum_k P[i, k] / Delta(k) = 1 / Delta_c(i), "
            "projected after each step. P stored as sparse CSR, N nonzeros."
        ),
    }
    return coarsening_refined, info


def _build_delta(
    coarsening: Coarsening,
    original,
    laplacian_kind: LaplacianKind,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Delta on the original nodes and Delta_c on the supernodes, float32."""
    from graco.core.conversions import data_to_torch_csr_adjacency
    from graco.core.q import degrees_torch

    if laplacian_kind == "combinatorial":
        return (
            torch.ones(coarsening.N, dtype=torch.float32, device=device),
            torch.ones(coarsening.n, dtype=torch.float32, device=device),
        )

    A = data_to_torch_csr_adjacency(original, dtype=torch.float32).to(device)
    A_c = data_to_torch_csr_adjacency(coarsening.coarsened, dtype=torch.float32).to(device)
    deg = degrees_torch(A).to(device)
    deg_c = degrees_torch(A_c).to(device)

    if laplacian_kind == "normalized":
        return (deg.clamp(min=1e-12).rsqrt(), deg_c.clamp(min=1e-12).rsqrt())
    if laplacian_kind == "normalized_self_loop":
        return ((deg + 1.0).rsqrt(), (deg_c + 1.0).rsqrt())
    raise ValueError(f"Unsupported laplacian_kind={laplacian_kind!r}")
