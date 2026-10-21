# Ported from GraphCoarsening/Featured-Graph-Coarsening (Apache-2.0).
#
# The two solvers follow their two notebooks. The metrics solver
# (FGC_experiment.ipynb) is numpy there and torch here. The sparsity solver
# (Node_classification_FGC_with_sparsity.ipynb) is already torch there, so our
# version stays close to it: same gradient terms, same step k, same L1 row
# normalisations. On top of the port we expose both solvers behind one
# variant argument, return the hard partition by default, and add the
# Delta-Laplacian family. See THIRD_PARTY_NOTICES.md.
"""Featured Graph Coarsening, FGC (Kumar et al., ICML 2023).

Learns a soft loading matrix C and coarse features X_tilde, minimizing the
Dirichlet energy of X_tilde on C^T L C, a log-det term against collapse,
the reconstruction error of X and a group penalty on C. Each node then
joins the supernode of its largest loading, so the returned partition is
the argmax of C. The released code ships two solvers, both behind variant.
Our port adds the Delta-Laplacian family and an optional ridge.
Reference code: github.com/GraphCoarsening/Featured-Graph-Coarsening
Details in the method catalogue of the GraCo paper (FGC).
"""

from __future__ import annotations

import time
import warnings
from typing import Literal, Optional

import numpy as np
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

from graco.algorithms.common import resolve_target
from graco.core.laplacian import LaplacianKind, build_laplacian
from graco.core.partition import Partition, relabel_contiguous, sanity_check_partition_Q


def coarsen_fgc(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    variant: Literal["metrics", "sparsity"] = "metrics",
    weights_source: Literal["code", "paper"] = "code",
    n_iters: Optional[int] = None,
    inner_c_steps: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    lam: Optional[float] = None,
    lr: Optional[float] = None,
    C_init_density: float = 0.15,
    ridge: float = 0.0,
    hermitian_solve: bool = True,
    return_soft: bool = False,
    laplacian_kind: LaplacianKind = "combinatorial",
    seed: int = 14,
    device: Optional[torch.device] = None,
    progress: bool = False,
    validate: bool = False,
) -> tuple[Partition, dict]:
    """Learn a soft loading matrix C, then assign each node to its largest loading.

    Give exactly one of r and n_clusters.

    FGC has many hyperparameters. We report the two configurations we use, and a
    few hints on which knob to turn.
      * Three-community SBM, 500 nodes, r = 0.9: variant="metrics", alpha=1,
        gamma=5, lam=10, n_iters=200.
      * Cora, r = 0.9: variant="metrics", gamma=5, n_iters=50.
    Hints: more outer iterations usually help. Lower lam when supernodes come
    back empty. gamma guards against collapse, but a high value keeps C close
    to its initialisation. A larger lr tends to collapse C onto fewer supernodes.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with node features data.x.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    variant : {"metrics", "sparsity"}, default "metrics"
        Solver of the released code. "metrics" produces the tables of the
        paper: no spectral term, L2-normalized coarse features, dense Gaussian
        init, step 1e-5, 100 steps on C per iteration. "sparsity" is the
        node classification solver: spectral term beta, L1-normalized
        features, sparse init, step k, one step on C per iteration.
    weights_source : {"code", "paper"}, default "code"
        For "metrics" only. The released code puts F/2 on gamma and 500 on
        lam, F being the number of features; the paper reports the same run
        with the two swapped.
    n_iters : int, optional
        Outer iterations, 10 for "metrics" and 50 for "sparsity" by default.
    inner_c_steps : int, optional
        Gradient steps on C per iteration, 100 for "metrics" and 1 for
        "sparsity" by default.
    alpha : float, optional
        Weight of the feature reconstruction, 500 ("metrics") or 100
        ("sparsity") by default.
    beta : float, optional
        Weight of the spectral term, 0 ("metrics") or 0.01 ("sparsity") by
        default. It is not in the objective of the paper.
    gamma : float, optional
        Weight of the log-det term, F/2 or 500 ("metrics", by
        weights_source) or 100 ("sparsity") by default.
    lam : float, optional
        Weight of the group penalty on C, 500 or F/2 ("metrics", by
        weights_source) or 100 ("sparsity") by default.
    lr : float, optional
        Gradient step, 1e-5 ("metrics") or k ("sparsity", the 1/L of the
        released code with L = 1/k) by default.
    C_init_density : float, default 0.15
        Density of the sparse init ("sparsity" only).
    ridge : float, default 0.0
        0 solves the two inner systems by pseudo-inverse, as the released
        code. A positive value adds ridge I and solves directly; this is not
        in the paper.
    hermitian_solve : bool, default True
        Compute the pseudo-inverse by eigendecomposition, which the symmetry
        of the matrix allows, instead of a full SVD.
    return_soft : bool, default False
        Warn that info["C_soft"] is no coarsening. C_soft is returned either
        way; the partition is always the argmax.
    laplacian_kind : str, default "combinatorial"
        Laplacian of the objective, D - A in the released code.
    seed : int, default 14
        Seed of torch, numpy and the init.
    device : torch.device, optional
        Device of the computation, data.x.device by default.
    progress : bool, default False
        Show a progress bar over the outer iterations.
    validate : bool, default False
        Check the partition against its lifting, in info["sanity"].

    Returns
    -------
    partition : Partition
    info : dict
        method : "fgc".
        variant, weights_source, laplacian_kind : the options used.
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        n_iters, inner_c_steps, alpha, beta, gamma, lambda, C_init_density,
            ridge, hermitian_solve, lr, return_soft : the values used.
        loss_history : objective after each outer iteration.
        total_time_s : wall time of the call.
        device : device of the computation.
        seed : the seed used.
        C_soft, X_tilde : the learned matrices, which reproduce the metrics
            of the paper; C_soft is no coarsening.
        notes : a reminder of the above.
        sanity : the partition check, only when validate is True.
    """
    t_total = time.perf_counter()
    if data.x is None:
        raise ValueError("FGC needs node features (data.x).")

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)
    K = n_target

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = device or data.x.device
    X = data.x.to(dtype=torch.float32, device=dev)                 # (N, F)
    F_dim = int(X.shape[1])

    L = build_laplacian(data, laplacian_kind, dtype=torch.float32).to(dev)

    # Defaults of the two solvers of the released code.
    if variant == "metrics":
        # Code and paper agree on the three values, and disagree on whether
        # gamma or lam carries F/2.
        if alpha is None:
            alpha = 500.0
        if weights_source == "code":
            if lam is None:
                lam = 500.0
            if gamma is None:
                gamma = F_dim / 2.0
        elif weights_source == "paper":
            if gamma is None:
                gamma = 500.0
            if lam is None:
                lam = F_dim / 2.0
        else:
            raise ValueError(
                f"weights_source must be 'code' or 'paper', got {weights_source!r}"
            )
        if beta is None:
            beta = 0.0
        if n_iters is None:
            n_iters = 10
        if inner_c_steps is None:
            inner_c_steps = 100
        if lr is None:
            lr = 1e-5
        xtilde_l2 = True
        dense_init = True
        # The notebook inverts with np.linalg.pinv, which cuts at 1e-15 of the
        # largest eigenvalue. Torch's own default cuts far higher in float32.
        pinv_rtol = 1e-15
    elif variant == "sparsity":
        if alpha is None:
            alpha = 100.0
        if gamma is None:
            gamma = 100.0
        if lam is None:
            lam = 100.0
        if beta is None:
            # The value of the single run of the notebook, inside its sweep.
            beta = 0.01
        if n_iters is None:
            n_iters = 50
        if inner_c_steps is None:
            inner_c_steps = 1
        if lr is None:
            lr = float(K)                # step 1/L with the notebook's L = 1/k
        xtilde_l2 = False
        dense_init = False
        # This notebook calls torch.linalg.pinv on float32 with no cutoff, so
        # torch's default one.
        pinv_rtol = None
    else:
        raise ValueError(f"variant must be 'metrics' or 'sparsity', got {variant!r}")

    thresh = 1e-10

    # Initial C and X_tilde.
    if dense_init:
        g = torch.Generator(device="cpu").manual_seed(seed)
        C = torch.clamp(torch.randn(N, K, generator=g), min=thresh).to(dev)             # (N, K)
        gx = torch.Generator(device="cpu").manual_seed(seed + 1)
        X_tilde = torch.randn(K, F_dim, generator=gx).to(dev)                           # (K, F)
    else:
        C = _init_C_sparse_random(N, K, density=C_init_density, seed=seed, device=dev)
        X_tilde = _init_C_sparse_random(K, F_dim, density=C_init_density, seed=seed + 1, device=dev)

    J = torch.full((K, K), 1.0 / K, dtype=torch.float32, device=dev)          # J = 1_K 1_K^T / K
    ones_KK = torch.ones((K, K), dtype=torch.float32, device=dev)              # for L1,2 approx grad

    loss_hist: list[float] = []

    iterator = tqdm(range(n_iters), disable=not progress, desc=f"FGC[{variant}]")
    for it in iterator:
        # Gradient steps on C.
        for _ in range(inner_c_steps):
            LC = L @ C                                             # (N, K) sparse @ dense
            CtLC = C.transpose(0, 1) @ LC                           # (K, K)
            # (C^T L C + J)^+ (L C)^T, by pseudo-inverse unless ridge > 0.
            inv_CtLC_J_LCt = _fgc_solve(CtLC + J, LC.transpose(0, 1), ridge,
                                        hermitian_solve, pinv_rtol)            # (K, N)
            LC_inv = inv_CtLC_J_LCt.transpose(0, 1)                            # (N, K)

            Xt_XtT = X_tilde @ X_tilde.transpose(0, 1)                          # (K, K)
            CXt_minus_X = C @ X_tilde - X                                       # (N, F)

            grad_C = (-2.0 * gamma * LC_inv                                     # log-det
                      + alpha * (CXt_minus_X @ X_tilde.transpose(0, 1))         # feature reconstruction
                      + 2.0 * (LC @ Xt_XtT)                                     # Dirichlet
                      + lam * (C @ ones_KK))                                    # L1,2 group sparsity
            if beta > 0.0:
                grad_C = grad_C + 2.0 * beta * (LC @ CtLC)                      # spectral term (sparsity variant)

            C = torch.clamp(C - lr * grad_C, min=0.0)                           # ReLU non-neg
            C = torch.clamp(C, min=thresh)                                      # avoid exact 0
            C = C / C.abs().sum(dim=1, keepdim=True).clamp(min=1e-12)           # L1 row-norm (both variants)

        # Closed-form X_tilde = ((2/alpha) C^T L C + C^T C)^{-1} C^T X, then
        # row normalization.
        Ct = C.transpose(0, 1)                                                  # (K, N)
        CtLC = Ct @ (L @ C)                                                     # (K, K), current C
        A = (2.0 / alpha) * CtLC + Ct @ C                                       # (K, K)
        X_tilde = _fgc_solve(A, Ct @ X, ridge, hermitian_solve, pinv_rtol)      # (K, F)
        if xtilde_l2:                                                           # "metrics": L2
            X_tilde = X_tilde / X_tilde.norm(dim=1, keepdim=True).clamp(min=1e-12)
        else:                                                                   # "sparsity": L1
            X_tilde = X_tilde / X_tilde.abs().sum(dim=1, keepdim=True).clamp(min=1e-12)

        loss_hist.append(_fgc_loss(C, X_tilde, X, L, J, alpha, beta, gamma, lam))

    # Each node joins the supernode of its largest loading.
    assignment_raw = C.argmax(dim=1).cpu().numpy().astype(np.int64)
    assignment, n_actual = relabel_contiguous(assignment_raw)
    partition = Partition(assignment, n_actual)

    if return_soft:
        warnings.warn(
            "FGC return_soft=True: info['C_soft'] is the soft matrix C, which the "
            "paper's metrics (REE/DE/HE) use. The returned Partition is its argmax.",
            stacklevel=2,
        )

    info = {
        "method": "fgc",
        "variant": variant,
        "r_requested": r_req,
        "r_actual": 1.0 - n_actual / N,
        "n_clusters": int(n_actual),
        "n_target": int(K),
        "laplacian_kind": laplacian_kind,
        "n_iters": int(n_iters),
        "inner_c_steps": int(inner_c_steps),
        "weights_source": weights_source,
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "lambda": float(lam),
        "C_init_density": float(C_init_density),
        "ridge": float(ridge),
        "hermitian_solve": bool(hermitian_solve),
        "lr": float(lr),
        "return_soft": bool(return_soft),
        "loss_history": loss_hist,
        "total_time_s": time.perf_counter() - t_total,
        "device": str(dev),
        "seed": seed,
        # C_soft and X_tilde reproduce the metrics of the paper; neither is a
        # coarsening.
        "C_soft": C.detach().cpu(),
        "X_tilde": X_tilde.detach().cpu(),
        "notes": (
            "The paper's metrics (REE, DE, HE) use C_soft, with L_c = C_soft^T L C_soft. "
            "C_soft is no coarsening; the returned partition is the argmax of C."
        ),
    }

    if n_actual < K:
        warnings.warn(
            f"FGC returned {n_actual} clusters (< target {K}): some rows of C share "
            f"their argmax; consider a higher gamma or more iterations.",
            stacklevel=2,
        )

    if validate:
        Q_sp = _torch_csr_partition_to_scipy(partition)
        info["sanity"] = sanity_check_partition_Q(partition, Q_sp)

    return partition, info


def _fgc_solve(M: torch.Tensor, b: torch.Tensor, ridge: float,
               hermitian: bool = True, rtol: Optional[float] = 1e-15) -> torch.Tensor:
    """Solve M X = b by pseudo-inverse with relative cutoff rtol, or directly on M + ridge I."""
    if ridge > 0.0:
        eye = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
        return torch.linalg.solve(M + ridge * eye, b)
    try:
        return torch.linalg.pinv(M, hermitian=hermitian, rtol=rtol) @ b
    except torch.linalg.LinAlgError as exc:
        raise torch.linalg.LinAlgError(
            f"pinv did not converge on a ({M.shape[0]}, {M.shape[1]}) matrix built from C. "
            "This happens once C collapses onto a few columns, which makes it singular. "
            "Lower lr or lam, raise the log-det weight (gamma in FGC, alpha in IPE), "
            "or set ridge > 0, a Tikhonov term that is not in the original paper."
        ) from exc


def _fgc_loss(
    C: torch.Tensor, X_tilde: torch.Tensor, X: torch.Tensor, L: torch.Tensor,
    J: torch.Tensor, alpha: float, beta: float, gamma: float, lam: float,
) -> float:
    """FGC objective, with the beta term of the "sparsity" solver."""
    LC = L @ C
    CtLC = C.transpose(0, 1) @ LC
    dirichlet = float(torch.trace(X_tilde.transpose(0, 1) @ CtLC @ X_tilde).item())
    sign, logabs = torch.linalg.slogdet(CtLC + J)
    logdet = float(logabs.item()) if sign.item() > 0 else float("-inf")
    recon = float((0.5 * alpha) * ((C @ X_tilde - X) ** 2).sum().item())
    l12 = float((0.5 * lam) * ((C.sum(dim=1)) ** 2).sum().item())
    spec = float((0.5 * beta) * (CtLC ** 2).sum().item())
    return dirichlet - gamma * logdet + recon + l12 + spec


def _init_C_sparse_random(
    rows: int, cols: int, density: float, seed: int, device: torch.device,
) -> torch.Tensor:
    """Dense copy of a sparse N(0, 1) matrix of the given density, drawn as the sparsity notebook does."""
    rng = np.random.default_rng(seed)
    C_sp = sp.random(rows, cols, density=density, random_state=rng,
                     data_rvs=lambda size: rng.standard_normal(size), format="csr")
    C_dense = np.asarray(C_sp.toarray(), dtype=np.float32)
    return torch.from_numpy(C_dense).to(device)


def _torch_csr_partition_to_scipy(partition: Partition) -> sp.csr_matrix:
    """Binary lifting of the partition, as scipy CSR."""
    Q_torch = partition.to_Q_combi(dtype=torch.float32)
    return sp.csr_matrix((
        Q_torch.values().detach().cpu().numpy(),
        Q_torch.col_indices().detach().cpu().numpy(),
        Q_torch.crow_indices().detach().cpu().numpy(),
    ), shape=Q_torch.shape)
