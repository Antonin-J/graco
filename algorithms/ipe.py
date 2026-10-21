"""Inner product preservation, IPE (Cohen and Talmon, NeurIPS 2025).

Learns a soft loading matrix C whose span is pulled towards the first k
eigenvectors of the Laplacian, a similarity on the Grassmann manifold, with
the group penalty and log-det term of FGC. INGC adds the inner product
error of the features and projects C on hard assignments every c_iter
steps; SINGC drops that error and projects once, at the end. The returned
partition is the argmax of C. The authors' repository is empty, so the port
follows the paper. The Grassmann term follows the derivative of the
objective, whose sign differs from the printed one. The port also caps k at
k_grassmann and renormalizes the rows of C after each step.
Details in the method catalogue of the GraCo paper (IPE).
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
from graco.algorithms.fgc import _fgc_solve
from graco.core.eig import smallest_eigenpairs
from graco.core.laplacian import LaplacianKind, build_laplacian
from graco.core.partition import Partition, relabel_contiguous


def coarsen_ipe(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    variant: Literal["SINGC", "INGC"] = "SINGC",
    lr: Optional[float] = None,
    alpha: float = 100.0,
    beta: float = 100.0,
    lam: float = 100.0,
    t_iter: Optional[int] = None,
    c_iter: int = 100,
    k_grassmann: int = 100,
    correct_grassmann_sign: bool = True,
    ridge: float = 0.0,
    laplacian_kind: LaplacianKind = "combinatorial",
    seed: int = 14,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Learn a soft loading matrix C, then assign each node to its largest loading.

    Give exactly one of r and n_clusters.

    IPE has many hyperparameters. We report the three configurations we use, and a
    few hints on which knob to turn.
      * Three-community SBM, 500 nodes, r = 0.9: variant="SINGC", lr=1e-3,
        alpha=1.0, lam=0.1, k_grassmann=50, t_iter=5000.
      * Same SBM, Grassmann rank set to its 3 communities (App. E of the
        GraCo paper): variant="SINGC", lr=1/K=0.02, alpha=0.1, lam=0.1,
        k_grassmann=3.
      * Cora, r = 0.5: variant="SINGC", alpha=1e4.
    Hints: a k_grassmann close to the number of classes pulls the partition
    towards them, with fewer supernodes than requested. A higher alpha keeps
    supernodes from emptying, a higher lam empties them. Keep lr small.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph; INGC also reads the node features data.x.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    variant : {"SINGC", "INGC"}, default "SINGC"
        SINGC drops the inner product error and projects C once. INGC keeps
        it and projects C every c_iter steps.
    lr : float, optional
        Gradient step, 1e-4 (SINGC) or 1e-5 (INGC) by default, the values of
        the paper on Cora. Small graphs need a larger step.
    alpha : float, default 100.0
        Weight of the log-det term.
    beta : float, default 100.0
        Weight of the Grassmann term, INGC only; SINGC gives it weight 1.
    lam : float, default 100.0
        Weight of the group penalty on C.
    t_iter : int, optional
        Number of steps for SINGC, 2000 by default, or of projections for
        INGC, 20 by default.
    c_iter : int, default 100
        INGC steps between two projections.
    k_grassmann : int, default 100
        Cap on the number k of eigenvectors. The paper takes k = n.
    correct_grassmann_sign : bool, default True
        True follows the derivative of the objective. False keeps the sign
        printed in the paper.
    ridge : float, default 0.0
        0 inverts C^T L C + J by pseudo-inverse, as FGC's released code. A
        positive value adds ridge I; this is not in the paper.
    laplacian_kind : str, default "combinatorial"
        Laplacian of the objective.
    seed : int, default 14
        Seed of torch, numpy and the init.
    device : torch.device, optional
        Device of the computation, that of data.x by default, else the CPU.
    progress : bool, default False
        Show a progress bar over the iterations.

    Returns
    -------
    partition : Partition
    info : dict
        method : "ipe".
        variant, laplacian_kind : the options used.
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        lr, alpha, beta, lambda, correct_grassmann_sign, ridge, t_iter,
            c_iter : the values used, c_iter None for SINGC.
        eig_time_s : wall time of the eigensolve.
        loss_history : sampled values of the objective. The descent follows
            the gradient of App. A.4 of the paper, which weighs the group and
            log-det terms by one half.
        total_time_s : wall time of the call.
        device : device of the computation.
        seed : the seed used.
        C_soft : the learned C before projection, which reproduces the
            metrics of the paper; it is no coarsening. Under INGC, which
            projects C periodically, it is the last projected C.
        notes : a reminder of the above.
    """
    t_total = time.perf_counter()

    if variant not in ("SINGC", "INGC"):
        raise ValueError(f"variant must be 'SINGC' or 'INGC', got {variant!r}")
    if variant == "INGC" and data.x is None:
        raise ValueError("INGC needs node features (data.x).")

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)
    K = n_target

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = device or (data.x.device if data.x is not None else torch.device("cpu"))
    L = build_laplacian(data, laplacian_kind, dtype=torch.float32).to(dev)

    # First k eigenvectors of L, by ARPACK on CPU. The paper takes k = n; the
    # cap acts only when n exceeds k_grassmann.
    t_eig = time.perf_counter()
    k_eig = min(k_grassmann, K, N - 1)
    _, U_k = smallest_eigenpairs(L.cpu(), k_eig, laplacian_kind)             # scipy ARPACK on CPU
    U_k = U_k.to(dtype=torch.float32, device=dev)                            # (N, k_eig)
    eig_time_s = time.perf_counter() - t_eig

    # U_k (U_k^T C), so the (N, N) projector is never formed.
    U_kT = U_k.transpose(0, 1).contiguous()                                  # (K, N)

    # Sparse non-negative init, as FGC's sparsity solver: a Gaussian one leaves
    # C^T L C rank-deficient.
    rng = np.random.default_rng(seed)
    C_sp = sp.random(N, K, density=0.25, random_state=rng,
                     data_rvs=lambda size: np.abs(rng.standard_normal(size)),
                     format="csr")
    C = torch.from_numpy(C_sp.toarray().astype(np.float32)).to(dev)
    C = torch.clamp(C, min=1e-3)                                             # non-neg floor

    ones_KK = torch.ones((K, K), dtype=torch.float32, device=dev)
    J = torch.full((K, K), 1.0 / K, dtype=torch.float32, device=dev)

    # Defaults of the paper.
    if t_iter is None:
        t_iter = 2000 if variant == "SINGC" else 20

    if lr is None:
        # Values of the paper on Cora; small graphs need a larger step.
        lr = 1e-4 if variant == "SINGC" else 1e-5

    # -1 follows the derivative of the objective, +1 the sign printed in the paper.
    gsign = -1.0 if correct_grassmann_sign else 1.0

    loss_hist: list[float] = []

    if variant == "SINGC":
        # SINGC projects C once, at the end. Clamping and L1 row normalization
        # after each step keep C^T L C + J invertible, as in FGC.
        iterator = tqdm(range(t_iter), disable=not progress, desc="IPE SINGC")
        for it in iterator:
            grad = _grad_singc(C, L, U_k, U_kT, J, ones_KK, alpha, lam, gsign, ridge)
            C = torch.clamp(C - lr * grad, min=1e-10)                          # non-neg
            row_l1 = C.sum(dim=1, keepdim=True).clamp(min=1e-12)
            C = C / row_l1                                                     # row-norm L1
            if it % max(1, t_iter // 50) == 0:                                # about 50 samples
                loss_hist.append(_loss_singc(C, L, U_k, J, alpha, lam, ridge))

        C_soft = C.clone()
        C_hard = _hardmax(C)

    else:  # INGC
        # INGC: c_iter steps between two projections.
        X = data.x.to(dtype=torch.float32, device=dev)                        # (N, p)
        XtLX = X.transpose(0, 1) @ (L @ X)                                    # (p, p) precompute
        # X_c = 0 silences the inner product term until the first projection
        # gives a usable X_c.
        X_c = torch.zeros(K, X.shape[1], dtype=torch.float32, device=dev)

        outer_bar = tqdm(range(t_iter), disable=not progress, desc="IPE INGC")
        for t in outer_bar:
            for _ in range(c_iter):
                grad = _grad_ingc(C, X_c, X, XtLX, L, U_k, U_kT, J, ones_KK,
                                  alpha, beta, lam, gsign, ridge)
                # Same rescaling as SINGC.
                C = torch.clamp(C - lr * grad, min=1e-10)
                row_l1 = C.sum(dim=1, keepdim=True).clamp(min=1e-12)
                C = C / row_l1
            C = _hardmax(C)                                                   # periodic projection
            X_c = _pinv_hardmax(C) @ X                                        # cheap since C is 1-hot
            loss_hist.append(_loss_ingc(C, X_c, X, XtLX, L, U_k, J,
                                        alpha, beta, lam, ridge))

        C_soft = C.clone()                                                     # already Hardmaxed
        C_hard = C

    # Each node joins the supernode of its largest loading.
    assignment_raw = C_hard.argmax(dim=1).cpu().numpy().astype(np.int64)
    assignment, n_actual = relabel_contiguous(assignment_raw)
    partition = Partition(assignment, n_actual)

    info = {
        "method": "ipe",
        "variant": variant,
        "r_requested": r_req,
        "r_actual": 1.0 - n_actual / N,
        "n_clusters": int(n_actual),
        "n_target": int(K),
        "laplacian_kind": laplacian_kind,
        "lr": float(lr),
        "alpha": float(alpha),
        "beta": float(beta),
        "lambda": float(lam),
        "correct_grassmann_sign": bool(correct_grassmann_sign),
        "ridge": float(ridge),
        "t_iter": int(t_iter),
        "c_iter": int(c_iter) if variant == "INGC" else None,
        "eig_time_s": eig_time_s,
        "loss_history": loss_hist,
        "total_time_s": time.perf_counter() - t_total,
        "device": str(dev),
        "seed": seed,
        # C_soft reproduces the metrics of the paper; it is no coarsening.
        "C_soft": C_soft.detach().cpu(),
        "notes": (
            "The paper's metrics (REE, RE, HE, IPE) use C_soft, with "
            "L_c = C_soft^T L C_soft. C_soft is no coarsening; the returned "
            "partition is the argmax of C."
        ),
    }

    if n_actual < K:
        warnings.warn(
            f"IPE ({variant}) returned {n_actual} clusters (< target {K}): some Hardmax "
            f"rows collided. Try {'a higher alpha or k_grassmann' if variant == 'SINGC' else 'a higher beta'}, "
            f"or more iterations.",
            stacklevel=2,
        )

    return partition, info


def _grad_singc(
    C: torch.Tensor, L: torch.Tensor,
    U_k: torch.Tensor, U_kT: torch.Tensor,
    J: torch.Tensor, ones_KK: torch.Tensor,
    alpha: float, lam: float, gsign: float, ridge: float,
) -> torch.Tensor:
    """SINGC gradient, gsign = -1 for the derivative of the objective, +1 for the printed sign."""
    grassmann = gsign * 2.0 * (U_k @ (U_kT @ C))                  # (N, K)
    l12 = lam * (C @ ones_KK)                                      # (N, K)
    LC = L @ C                                                     # (N, K)
    CtLC_J = C.transpose(0, 1) @ LC + J                            # (K, K)
    # Pseudo-inverse, as FGC's released code; an empty column of C makes the
    # matrix singular.
    logdet_term = _fgc_solve(CtLC_J.transpose(0, 1), LC.transpose(0, 1), ridge).transpose(0, 1)
    return grassmann + l12 - alpha * logdet_term


def _grad_ingc(
    C: torch.Tensor, X_c: torch.Tensor, X: torch.Tensor, XtLX: torch.Tensor,
    L: torch.Tensor, U_k: torch.Tensor, U_kT: torch.Tensor,
    J: torch.Tensor, ones_KK: torch.Tensor,
    alpha: float, beta: float, lam: float, gsign: float, ridge: float,
) -> torch.Tensor:
    """INGC gradient, without the lambda C term printed beside the group term (App. A.4 of the paper)."""
    LC = L @ C                                                     # (N, K)
    CtLC = C.transpose(0, 1) @ LC                                  # (K, K)

    grassmann = gsign * 2.0 * beta * (U_k @ (U_kT @ C))           # (N, K)

    # IPE term : M = XtLX - X_c^T C^T L C X_c, shape (p, p).
    M = XtLX - X_c.transpose(0, 1) @ (CtLC @ X_c)                  # (p, p) (X_c is (K, p))
    # -4 L C X_c M X_c^T ; note L C X_c = LC @ X_c (N, p).
    LCXc = LC @ X_c                                                # (N, p)
    ipe = -4.0 * (LCXc @ M @ X_c.transpose(0, 1))                  # (N, K)

    l12 = lam * (C @ ones_KK)                                      # (N, K)

    # Hardmax leaves empty clusters, so C^T L C + J is often singular here.
    logdet_term = _fgc_solve((CtLC + J).transpose(0, 1), LC.transpose(0, 1), ridge).transpose(0, 1)

    return grassmann + ipe + l12 - alpha * logdet_term


def _loss_singc(
    C: torch.Tensor, L: torch.Tensor,
    U_k: torch.Tensor, J: torch.Tensor,
    alpha: float, lam: float, ridge: float,
) -> float:
    """SINGC objective, with the Grassmann term at weight 1."""
    CtLC = C.transpose(0, 1) @ (L @ C)
    grass = float(torch.trace(U_k.transpose(0, 1) @ C @ C.transpose(0, 1) @ U_k).item())
    l12 = float(((C.sum(dim=1)) ** 2).sum().item())
    eye = torch.eye(CtLC.shape[0], dtype=C.dtype, device=C.device)
    sign, logabs = torch.linalg.slogdet(CtLC + J + ridge * eye)
    logdet = float(logabs.item()) if sign.item() > 0 else float("-inf")
    return -grass + lam * l12 - alpha * logdet


def _loss_ingc(
    C: torch.Tensor, X_c: torch.Tensor, X: torch.Tensor, XtLX: torch.Tensor,
    L: torch.Tensor, U_k: torch.Tensor, J: torch.Tensor,
    alpha: float, beta: float, lam: float, ridge: float,
) -> float:
    """INGC objective; an empty cluster makes the log-det -inf unless ridge > 0."""
    K = C.shape[1]
    CtLC = C.transpose(0, 1) @ (L @ C)
    diff = XtLX - X_c.transpose(0, 1) @ (CtLC @ X_c)
    ipe = float((diff * diff).sum().item())
    grass = float(torch.trace(U_k.transpose(0, 1) @ C @ C.transpose(0, 1) @ U_k).item())
    l12 = float(((C.sum(dim=1)) ** 2).sum().item())
    reg_mat = CtLC + J + ridge * torch.eye(K, dtype=C.dtype, device=C.device)
    sign, logabs = torch.linalg.slogdet(reg_mat)
    logdet = float(logabs.item()) if sign.item() > 0 else float("-inf")
    return ipe - beta * grass + lam * l12 - alpha * logdet


def _hardmax(C: torch.Tensor) -> torch.Tensor:
    """Each row of C becomes one-hot at its argmax, the first index on ties."""
    K = C.shape[1]
    argmax = C.argmax(dim=1)                                       # (N,)
    C_hard = torch.zeros_like(C)
    C_hard.scatter_(1, argmax.unsqueeze(1), 1.0)
    return C_hard


def _pinv_hardmax(C_hard: torch.Tensor) -> torch.Tensor:
    """Pseudo-inverse diag(1/|cluster|) C^T of a one-hot C, kept dense."""
    cluster_sizes = C_hard.sum(dim=0).clamp(min=1)                 # (K,) # nodes per cluster
    return C_hard.transpose(0, 1) / cluster_sizes.unsqueeze(1)     # (K, N)
