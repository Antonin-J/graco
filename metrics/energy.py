"""Metrics built on energies, Laplacian quadratic forms and graph convolutions,
and the losses of FGC, IPE and NOPE.

Most papers write C for our lifting Q and C^+ for our reduction P, so their
coarsened features C^+ X are our P X and their C^T L C our Q^T L Q.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from graco.core.coarsening import Coarsening
from graco.core.conversions import data_to_torch_csr_adjacency, sparse_matmul
from graco.core.laplacian import LaplacianKind, build_laplacian
from graco.core.propag import build_S
from graco.core.q import degrees_torch
from graco.metrics._common import _spectral_cache


def dirichlet_energy_error(
    coarsening: Coarsening,
    original,
    x: Optional[torch.Tensor] = None,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    mode: str = "relative",
) -> float:
    """Relative change of the Dirichlet energy under coarsening.

        E_G  = tr(x^T L x)
        E_Gc = tr((P x)^T L_c (P x))

    with L_c the Laplacian of the coarsened graph. mode="relative" returns
    |E_G - E_Gc| / E_G; mode="log" returns |log(E_G / E_Gc)|, the definition of
    IPE (Cohen et al., NeurIPS 2025). For the combinatorial Laplacian,
    L_c = Q^T L Q, so for x in the subspace R of the RSA constant epsilon
    (Loukas, JMLR 2019), (1 - eps) ||x||_L <= ||P x||_Lc <= (1 + eps) ||x||_L
    and the relative error is at most 2 eps + eps^2. Features usually have
    components outside R, where no such bound holds.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    x : torch.Tensor (N,) or (N, F), optional
        Signal or features; None uses original.x.
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.
    mode : {"relative", "log"}, default "relative"

    Returns
    -------
    float
        NaN when E_G is zero; inf in log mode when E_Gc is zero.
    """
    if mode not in ("relative", "log"):
        raise ValueError(f"mode must be 'relative' or 'log', got {mode!r}")
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    if x is None:
        x = original.x
    if x is None:
        raise ValueError("dirichlet_energy_error needs x (or original.x).")
    if x.ndim == 1:
        x = x.unsqueeze(1)

    P = coarsening.P.to(x.dtype)
    E_G  = dirichlet_energy(original,  x,       laplacian_kind=laplacian_kind)
    E_Gc = dirichlet_energy(coarsening.coarsened, P @ x,   laplacian_kind=laplacian_kind)

    if E_G < 1e-15:
        return float("nan")
    if mode == "relative":
        return abs(E_G - E_Gc) / E_G
    if E_Gc < 1e-15:
        return float("inf")
    return abs(math.log(E_G / E_Gc))


def reconstruction_error(
    coarsening: Coarsening,
    original,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    normalize: bool = True,
) -> float:
    """Frobenius distance between L and the lifted Laplacian P^T Q^T L Q P.

        normalize=True    ||L - P^T Q^T L Q P||_F / ||L||_F
        normalize=False   ||L - P^T Q^T L Q P||_F^2

    Used by FGC (Kumar et al., ICML 2023) and IPE (Cohen et al., NeurIPS 2025).
    No (N, N) matrix is built: with L_c = Q^T L Q and G = P P^T,

        ||L - P^T L_c P||_F^2 = ||L||_F^2 - 2 <L_c, P L P^T> + tr(L_c G L_c G)

    and the last two terms are sparse (n, n) products, in float64.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.
    normalize : bool, default True

    Returns
    -------
    float
        NaN when ||L||_F is zero and normalize=True.
    """
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    # float64: the three terms nearly cancel when the error is small.
    L = build_laplacian(original, laplacian_kind, dtype=torch.float64)
    P = coarsening.P.to(torch.float64)
    Q = coarsening.Q.to(torch.float64)
    Pt = P.transpose(0, 1).to_sparse_csr()

    def inner(a, b):
        # through COO: csr * csr gave wrong values on a transposed CSR
        return float((a.to_sparse_coo().coalesce() * b.to_sparse_coo().coalesce()).coalesce().values().sum().item())

    L_c = sparse_matmul(sparse_matmul(Q.transpose(0, 1), L), Q)       # Q^T L Q, MKL case in needs_coo
    G = sparse_matmul(P, Pt)
    fro_L_sq = inner(L, L)
    cross = inner(L_c, sparse_matmul(sparse_matmul(P, L), Pt))
    # L_c and G are symmetric, so tr(L_c G L_c G) = <L_c G, G L_c>
    lift_sq = inner(sparse_matmul(L_c, G), sparse_matmul(G, L_c))

    fro_sq = max(fro_L_sq - 2 * cross + lift_sq, 0.0)
    if not normalize:
        return fro_sq
    if fro_L_sq < 1e-15:
        return float("nan")
    return (fro_sq / fro_L_sq) ** 0.5


def dirichlet_energy(
    graph,
    x: Optional[torch.Tensor] = None,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
) -> float:
    """Dirichlet energy tr(x^T L x) of a signal on one graph.

        tr(x^T L x) = (1/2) sum_{(i,j) in E} w_ij ||x_i - x_j||^2

    On the coarsened graph, dirichlet_energy(coarsening.coarsened) gives the
    energy FGC (Kumar et al., ICML 2023) reports for G_c.

    Parameters
    ----------
    graph : torch_geometric.data.Data
    x : torch.Tensor (N,) or (N, F), optional
        None uses graph.x.
    laplacian_kind : str, optional
        None uses "combinatorial"; pass coarsening.laplacian_kind to match a
        Coarsening.

    Returns
    -------
    float
    """
    if laplacian_kind is None:
        laplacian_kind = "combinatorial"
    if x is None:
        x = graph.x
    if x is None:
        raise ValueError("dirichlet_energy needs x (or graph.x).")
    if x.ndim == 1:
        x = x.unsqueeze(1)
    L = build_laplacian(graph, laplacian_kind, dtype=x.dtype)
    Lx = L @ x
    return float((x * Lx).sum().item())


def hyperbolic_error(
    coarsening: Coarsening,
    original,
    x: Optional[torch.Tensor] = None,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
) -> float:
    """Hyperbolic error between L and the lifted Laplacian L_l = P^T Q^T L Q P.

        HE = arccosh( 1 + ||(L - L_l) X||_F^2 ||X||_F^2
                          / ( 2 tr(X^T L X) tr(X^T L_l X) ) )

    From Bravo Hermsdorff and Gunderson (NeurIPS 2019), used by FGC (Kumar et
    al., ICML 2023), UGC (Kataria et al., NeurIPS 2024) and IPE (Cohen et al.,
    NeurIPS 2025). L_l is applied to X as a chain of sparse products.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    x : torch.Tensor (N,) or (N, F), optional
        Signal or features; None uses original.x.
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.

    Returns
    -------
    float
        NaN when either trace is zero.
    """
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    if x is None:
        x = original.x
    if x is None:
        raise ValueError("hyperbolic_error needs x (or original.x).")
    if x.ndim == 1:
        x = x.unsqueeze(1)

    L = build_laplacian(original, laplacian_kind, dtype=x.dtype)
    P, Q = coarsening.P, coarsening.Q
    # L_l x = P^T Q^T L (Q (P x)), right to left.
    Px = P @ x
    QPx = Q @ Px
    LQPx = L @ QPx
    QtLQPx = Q.transpose(0, 1) @ LQPx
    L_l_x = P.transpose(0, 1) @ QtLQPx                                  # (N, F)

    Lx = L @ x                                                           # (N, F)
    diff_X = Lx - L_l_x                                                  # (L - L_l) X

    fro_diff_sq = float((diff_X * diff_X).sum().item())
    fro_X_sq    = float((x * x).sum().item())
    tr_XtLX     = float((x * Lx).sum().item())
    tr_XtLlX    = float((x * L_l_x).sum().item())

    if tr_XtLX < 1e-15 or tr_XtLlX < 1e-15:
        return float("nan")
    arg = 1.0 + fro_diff_sq * fro_X_sq / (2.0 * tr_XtLX * tr_XtLlX)
    # arccosh requires arg >= 1; clamp to handle float noise
    arg = max(arg, 1.0)
    return float(math.acosh(arg))


def inner_product_error(
    coarsening: Coarsening,
    original,
    x: Optional[torch.Tensor] = None,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    normalize: bool = False,
) -> float:
    """Inner product error of IPE (Cohen et al., NeurIPS 2025).

        IPE = ||X^T L X - X_c^T L_c X_c||_F^2,   X_c = P X,  L_c = Q^T L Q
            = ||X^T (L - P^T Q^T L Q P) X||_F^2

    How well the (F, F) matrix of Laplacian inner products between feature
    columns survives coarsening.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    x : torch.Tensor (N,) or (N, F), optional
        Signal or features; None uses original.x.
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.
    normalize : bool, default False
        If True, divide by ||X^T L X||_F^2.

    Returns
    -------
    float
    """
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    if x is None:
        x = original.x
    if x is None:
        raise ValueError("inner_product_error needs x (or original.x).")
    if x.ndim == 1:
        x = x.unsqueeze(1)

    L = build_laplacian(original, laplacian_kind, dtype=x.dtype)
    P = coarsening.P.to(x.dtype)
    Q = coarsening.Q.to(x.dtype)

    LX = L @ x                                                # (N, F)
    XtLX = x.transpose(0, 1) @ LX                             # (F, F)

    X_c = P @ x                                               # (n, F)
    LcXc = sparse_matmul(sparse_matmul(Q.transpose(0, 1), L), Q) @ X_c  # Q^T L Q X_c, MKL case in needs_coo
    XctLcXc = X_c.transpose(0, 1) @ LcXc                      # (F, F)

    diff = XtLX - XctLcXc
    ipe_sq = float((diff * diff).sum().item())
    if not normalize:
        return ipe_sq
    denom = float((XtLX * XtLX).sum().item())
    if denom < 1e-15:
        return float("nan")
    return ipe_sq / denom


def ipe_grassmann_loss(
    coarsening: Coarsening,
    original,
    x: Optional[torch.Tensor] = None,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    beta: float = 1.0,
    lambda_reg: float = 0.0,
    alpha: float = 1.0,
    K: int = 100,
) -> dict:
    """Objective of IPE (Cohen et al., NeurIPS 2025) on a fixed coarsening.

        loss = ||X^T L X - X_c^T L_c X_c||_F^2          inner product error
               - beta tr(U_K U_K^T Q Q^T)               Grassmann similarity
               + lambda ||Q^T||_{1,2}^2                  group sparsity
               - alpha log det+(Q^T L Q)                pseudo log-determinant

    U_K holds the K eigenvectors of L with the smallest eigenvalues, and
    tr(U_K U_K^T Q Q^T) = ||U_K^T Q||_F^2, so no (N, N) product is built; U_K
    itself comes from a full eigendecomposition of L, in O(N^3).
    ||Q^T||_{1,2}^2 sums the squared l1 norms of the rows of Q, and log det+
    sums the log of the eigenvalues of Q^T L Q above 1e-7. This evaluates the objective on
    the output of any coarsener; IPE minimises it.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    x : torch.Tensor (N,) or (N, F), optional
        Signal or features; None uses original.x.
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.
    beta, lambda_reg, alpha : float, default 1.0, 0.0 and 1.0
        Weights of the last three terms.
    K : int, default 100
        Number of eigenvectors in U_K, clamped to N - 1.

    Returns
    -------
    dict
        total : sum of the four terms.
        ipe_term, grassmann_term, group_sparsity, logdet_term : the terms,
        weights included.
    """
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    if x is None:
        x = original.x
    if x is None:
        raise ValueError("ipe_grassmann_loss needs x (or original.x).")
    if x.ndim == 1:
        x = x.unsqueeze(1)
    dtype = torch.float64                                          # eigh + log det need precision

    L = build_laplacian(original, laplacian_kind, dtype=dtype)
    Q = coarsening.Q.to(dtype)
    P = coarsening.P.to(dtype)
    X = x.to(dtype)

    # Inner product error, computed here to share L.
    LX = L @ X
    XtLX = X.transpose(0, 1) @ LX
    Xc = P @ X
    QtLQ_csr = sparse_matmul(sparse_matmul(Q.transpose(0, 1), L), Q)  # Q^T L Q, MKL case in needs_coo
    QtLQ = QtLQ_csr.to_dense() if QtLQ_csr.layout == torch.sparse_csr else QtLQ_csr
    LcXc = QtLQ @ Xc
    XctLcXc = Xc.transpose(0, 1) @ LcXc
    ipe_diff = XtLX - XctLcXc
    ipe_term = float((ipe_diff * ipe_diff).sum().item())

    # Grassmann similarity tr(Uk Uk^T Q Q^T) = ||Uk^T Q||_F^2.
    cache = _spectral_cache(original, laplacian_kind, dtype=dtype)
    K_eff = min(K, cache["eigvecs"].shape[1] - 1)
    Uk = cache["eigvecs"][:, :K_eff]                               # (N, K_eff)
    # (Uk^T Q)^T = Q^T Uk, a sparse times dense product.
    UktQ_T = Q.transpose(0, 1) @ Uk                                # (n, K_eff)
    grassmann_score = float((UktQ_T * UktQ_T).sum().item())        # ||Uk^T Q||_F^2
    grassmann_term = -beta * grassmann_score

    # Group sparsity ||Q^T||_{1,2}^2 = sum_i (sum_j |Q_ij|)^2, as in fgc_loss.
    if Q.layout == torch.sparse_csr:
        Q_coo = Q.to_sparse_coo().coalesce()
        rows = Q_coo.indices()[0]
        row_l1 = torch.zeros(coarsening.N, dtype=dtype, device=Q.device)
        row_l1.scatter_add_(0, rows, Q_coo.values().abs())
    else:
        row_l1 = Q.abs().sum(dim=1)
    group_sparsity = lambda_reg * float((row_l1 ** 2).sum().item())

    # Pseudo log-determinant of Q^T L Q, over the eigenvalues above 1e-7.
    eigvals_Lc = torch.linalg.eigvalsh(QtLQ)
    pos = eigvals_Lc[eigvals_Lc > 1e-7]
    logdet_val = float(pos.log().sum().item()) if pos.numel() > 0 else float("-inf")
    logdet_term = -alpha * logdet_val

    total = ipe_term + grassmann_term + group_sparsity + logdet_term
    return {
        "total":          float(total),
        "ipe_term":       float(ipe_term),
        "grassmann_term": float(grassmann_term),
        "group_sparsity": float(group_sparsity),
        "logdet_term":    float(logdet_term),
    }


def fgc_loss(
    coarsening: Coarsening,
    original,
    x: Optional[torch.Tensor] = None,
    *,
    laplacian_kind: Optional[LaplacianKind] = None,
    gamma: float = 1.0,
    alpha: float = 1.0,
    lambda_reg: float = 0.0,
) -> dict:
    """Objective of FGC (Kumar et al., ICML 2023) on a fixed coarsening.

        loss = - gamma log det+(L_c)                   spectral term
               + tr(X_t^T L_c X_t)                      Dirichlet energy on G_c
               + (alpha / 2) ||Q X_t - X||_F^2          feature reconstruction
               + (lambda / 2) ||Q^T||_{1,2}^2            group sparsity

    with L_c = Q^T L Q, X_t = P X, and ||Q^T||_{1,2}^2 the sum of the squared
    l1 norms of the rows of Q. The paper writes log det(L_c + J), with J the rank-one term on the
    null direction of L_c. We sum the log of the eigenvalues of L_c above 1e-7,
    which is the same value when L_c has a single null direction, the case of
    a connected original graph. This evaluates the objective on the output of
    any coarsener; FGC minimises it.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    x : torch.Tensor (N,) or (N, F), optional
        Signal or features; None uses original.x.
    laplacian_kind : str, optional
        Laplacian convention; None uses coarsening.laplacian_kind.
    gamma, alpha, lambda_reg : float, default 1.0, 1.0 and 0.0
        Weights of the spectral, reconstruction and sparsity terms.

    Returns
    -------
    dict
        total : sum of the four terms.
        logdet_term, dirichlet_term, recon_term, group_sparsity : the terms,
        weights included.
    """
    if laplacian_kind is None:
        laplacian_kind = coarsening.laplacian_kind
    if x is None:
        x = original.x
    if x is None:
        raise ValueError("fgc_loss needs x (or original.x).")
    if x.ndim == 1:
        x = x.unsqueeze(1)
    dtype = torch.float64                                       # log det needs precision

    L = build_laplacian(original, laplacian_kind, dtype=dtype)
    Q = coarsening.Q.to(dtype)
    P = coarsening.P.to(dtype)
    X = x.to(dtype)

    n = coarsening.n
    L_c = sparse_matmul(sparse_matmul(Q.transpose(0, 1), L), Q)  # Q^T L Q, MKL case in needs_coo
    if L_c.layout == torch.sparse_csr:
        L_c = L_c.to_dense()

    # Pseudo log-determinant over the eigenvalues above 1e-7, equal to
    # log det(L_c + J) of the paper when L_c has a single null direction.
    eigvals_Lc = torch.linalg.eigvalsh(L_c)
    pos = eigvals_Lc[eigvals_Lc > 1e-7]
    logdet_val = float(pos.log().sum().item()) if pos.numel() > 0 else float("-inf")

    X_tilde = P @ X                                              # (n, F)
    LcXt = L_c @ X_tilde
    dirichlet_term = float((X_tilde * LcXt).sum().item())

    QXt = Q @ X_tilde                                            # (N, F)
    recon_term = 0.5 * alpha * float(((QXt - X) ** 2).sum().item())

    # ||Q^T||_{1,2}^2 = sum_i (sum_j |Q_ij|)^2, the definition of FGC.
    if Q.layout == torch.sparse_csr:
        Q_coo = Q.to_sparse_coo().coalesce()
        rows = Q_coo.indices()[0]
        row_l1 = torch.zeros(coarsening.N, dtype=dtype, device=Q.device)
        row_l1.scatter_add_(0, rows, Q_coo.values().abs())
    else:
        row_l1 = Q.abs().sum(dim=1)
    group_sparsity = 0.5 * lambda_reg * float((row_l1 ** 2).sum().item())

    logdet_term = -gamma * logdet_val
    total = logdet_term + dirichlet_term + recon_term + group_sparsity
    return {
        "total": float(total),
        "logdet_term": float(logdet_term),
        "dirichlet_term": float(dirichlet_term),
        "recon_term": float(recon_term),
        "group_sparsity": float(group_sparsity),
    }


def convolution_matching_error(
    coarsening: Coarsening,
    original,
    x: Optional[torch.Tensor] = None,
    *,
    mode: str = "pair",
) -> dict:
    """Error of one GCN convolution under coarsening, from ConvMatch (Dickens et al., WWW Companion 2024).

        error = || Q S' P X - S X ||_{1,1},   S = (D+I)^{-1/2} (A+I) (D+I)^{-1/2}

    mode="pair" takes the P and Q of the coarsening, with S' = P S Q, so the
    lifted output is (QP) S (QP) X. mode="dickens" follows the paper as written:
    S' = Sigma^{-1/2} Q_b^T (A+I) Q_b Sigma^{-1/2} with Q_b the binary indicator
    and Sigma = Q_b^T (D+I) Q_b the summed degrees, applied to the mean features
    of each supernode and lifted by Q_b. Details in the method catalogue of the
    GraCo paper (ConvMatch).

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
    x : torch.Tensor (N,) or (N, F), optional
        Signal or features; None uses original.x.
    mode : {"pair", "dickens"}, default "pair"

    Returns
    -------
    dict
        total : the L1 error summed over the feature columns.
        mean : its mean over the columns.
        max : the largest column error.
    """
    if mode not in ("pair", "dickens"):
        raise ValueError(f"mode must be 'pair' or 'dickens', got {mode!r}")
    if x is None:
        x = original.x
    if x is None:
        raise ValueError("convolution_matching_error needs x (or original.x).")
    if x.ndim == 1:
        x = x.unsqueeze(1)

    S = build_S(original, mode="self_loop_adj", dtype=x.dtype)
    target = S @ x                                   # (N, F)

    if mode == "pair":
        P = coarsening.P.to(x.dtype)
        Q = coarsening.Q.to(x.dtype)
        lifted = Q @ (P @ (S @ (Q @ (P @ x))))
    else:
        assign = torch.as_tensor(coarsening.partition.assignment, dtype=torch.long,
                                 device=x.device)
        n = int(coarsening.partition.n_clusters)
        A = data_to_torch_csr_adjacency(original, dtype=x.dtype).to(x.device)
        d_hat = degrees_torch(A).to(x.dtype) + 1.0
        size = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(
            0, assign, torch.ones_like(d_hat))
        sig_isqrt = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(
            0, assign, d_hat).rsqrt()
        means = torch.zeros(n, x.shape[1], dtype=x.dtype, device=x.device).index_add_(
            0, assign, x) / size.unsqueeze(1)
        y = (means * sig_isqrt.unsqueeze(1))[assign]   # Q_b Sigma^{-1/2} X'
        y = A @ y + y                                  # (A + I)
        h = torch.zeros_like(means).index_add_(0, assign, y) * sig_isqrt.unsqueeze(1)
        lifted = h[assign]

    per_col = (lifted - target).abs().sum(dim=0)     # (F,) L1 per feature column
    return {
        "total": float(per_col.sum().item()),
        "mean":  float(per_col.mean().item()),
        "max":   float(per_col.max().item()),
    }


def neighborhood_interference_index(
    coarsening: Coarsening,
    original,
    *,
    similarity_kind: str = "euclidean",
    dtype: torch.dtype = torch.float32,
) -> float:
    """Interference of NOPE (Bai et al., ICML 2026), summed over the pairs inside each cluster.

        I_pq    = w sum_{k in (N_p u N_q) minus {p, q}} (<x_k, x_p> - <x_k, x_q>)^2
        I_total = sum_c sum_{p < q in c} I_pq

    with w = 1/2, the weight s_p s_q / (s_p + s_q) of NOPE for two singletons,
    and x_i the row i of X_sim built from original.x by similarity_kind. As in
    the released code, p and q leave the union of their neighbourhoods. Lower
    means the partition groups nodes that see their neighbours alike, which
    NOPE minimises greedily. Each cluster is scored on the nodes adjacent to
    it, one p at a time against all its q > p.

    Parameters
    ----------
    coarsening : Coarsening
    original : torch_geometric.data.Data
        Must carry x.
    similarity_kind : str, default "euclidean"
        "euclidean" X_sim = X, the paper's choice; "propagation"
        (D+I)^{-1/2} (A+I) (D+I)^{-1/2} X; "L_combi" (D - A) X;
        "L_norm_self_loop" the normalized_self_loop Laplacian times X;
        "adjacency" A X.
    dtype : torch.dtype, default torch.float32

    Returns
    -------
    float
    """
    valid_kinds = ("euclidean", "propagation", "L_combi", "L_norm_self_loop", "adjacency")
    if similarity_kind not in valid_kinds:
        raise ValueError(
            f"similarity_kind must be one of {valid_kinds}, got {similarity_kind!r}"
        )

    data = original
    if getattr(data, "x", None) is None:
        raise ValueError("original.x is None; the NOPE metric needs node features")
    X = data.x.to(dtype)

    if similarity_kind == "euclidean":
        X_sim = X
    elif similarity_kind == "propagation":
        X_sim = build_S(data, mode="self_loop_adj", dtype=dtype) @ X
    elif similarity_kind == "adjacency":
        X_sim = data_to_torch_csr_adjacency(data, dtype=dtype) @ X
    elif similarity_kind == "L_combi":
        X_sim = build_laplacian(data, "combinatorial", dtype=dtype) @ X
    else:                                                                     # L_norm_self_loop
        X_sim = build_laplacian(data, "normalized_self_loop", dtype=dtype) @ X

    A_csr = data_to_torch_csr_adjacency(data, dtype=dtype)
    crow = A_csr.crow_indices()
    col = A_csr.col_indices()

    assignment = coarsening.partition.assignment                             # np.ndarray (N,)
    cluster_members: dict[int, list[int]] = {}
    for i, c in enumerate(assignment):
        cluster_members.setdefault(int(c), []).append(int(i))

    total = 0.0
    weight = 0.5                                                             # harmonic(1, 1)

    for members in cluster_members.values():
        m = len(members)
        if m < 2:
            continue
        nbrs = [col[crow[v]:crow[v + 1]] for v in members]
        K = torch.cat(nbrs).unique()                                         # nodes adjacent to the cluster
        if K.numel() == 0:
            continue
        pos = torch.full((data.num_nodes,), -1, dtype=torch.long)
        pos[K] = torch.arange(K.numel())
        B = torch.zeros(m, K.numel(), dtype=torch.bool)                      # B[a, j]: members[a] ~ K[j]
        for a, nb in enumerate(nbrs):
            B[a, pos[nb]] = True
        S = X_sim[K] @ X_sim[members].T                                      # S[j, a] = <x_K[j], x_members[a]>
        col_of = pos[torch.as_tensor(members)]                               # column of each member in K, -1 if none
        for a in range(m - 1):
            q = torch.arange(a + 1, m)
            U = B[a] | B[q]                                                  # union of neighbourhoods, (m - a - 1, |K|)
            if col_of[a] >= 0:
                U[:, col_of[a]] = False                                      # p leaves the union
            inside = col_of[q] >= 0
            U[inside.nonzero().squeeze(1), col_of[q][inside]] = False        # so does each q
            D = (S[:, a].unsqueeze(0) - S[:, q].T) ** 2
            total += weight * float((D * U).sum().item())

    return total
