"""OTCoarsening (Ma and Chen, AAAI 2021).

Trains, without labels, a GCN that scores the nodes, keeps the m best as
coarse nodes, and pools the graph through an operator S supported on their
one-hop neighbourhoods. The loss is the Sinkhorn transport between the
features and their decoded coarse version. Each node then joins the column
of its largest weight in S. A node more than one hop from every picked node
has no weight at all, and uncovered decides its supernode. The paper forms
no hard partition. Where the paper and the released code disagree (score input,
column slice, zero-row filter, fixed Sinkhorn steps, entropic term), we
follow the paper, except for the squared score; score_square, cut and
zero_row_filter select between the two. Dense in N, so for graphs of
moderate size.
Reference code: github.com/matenure/OTCoarsening
Details in the method catalogue of the GraCo paper (OTCoarsening).
"""

from __future__ import annotations

import time
import warnings
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from graco.algorithms.common import resolve_target
from graco.core.conversions import (
    data_to_scipy_csr_adjacency, data_to_torch_csr_adjacency,
)
from graco.core.partition import Partition, relabel_contiguous


def coarsen_otcoarsening(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    hidden_dim: int = 64,
    n_epochs: int = 100,
    lr: float = 1e-3,
    sinkhorn_iters: int = 30,
    gamma: float = 1.0,
    p_power: int = 2,
    score_square: bool = True,
    cut: Literal["raw", "relu"] = "raw",
    zero_row_filter: Literal["none", "repo"] = "none",
    seed: int = 14,
    uncovered: Literal["attach", "isolate", "isolate_recursive"] = "isolate_recursive",
    r_per_level: float = 0.5,
    return_soft_s: bool = False,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> tuple[Partition, dict]:
    """Train the pooling operator S, then assign each node to its largest weight.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with node features data.x.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    hidden_dim : int, default 64
        Hidden size of the GCN encoder and decoder.
    n_epochs : int, default 100
        Training epochs.
    lr : float, default 1e-3
        Adam step, as in the released code.
    sinkhorn_iters : int, default 30
        Sinkhorn steps, a fixed number as in the paper.
    gamma : float, default 1.0
        Entropic weight of the transport, as in the released code. It carries
        the units of the cost.
    p_power : int, default 2
        Power p of the cost ||x_i - x_c_j||_2^p, the Euclidean norm raised to p.
    score_square : bool, default True
        Square the score before the sigmoid, as the released code does and the
        paper does not.
    cut : {"raw", "relu"}, default "raw"
        Column weights of S: the m best scores (paper), or the shifted scores of
        the released code, which leave the last column near zero.
    zero_row_filter : {"none", "repo"}, default "none"
        Rows of the transport cost: all of them (paper), or those the released
        get_nonzero_rows returns.
    seed : int, default 14
        Seed of the weights, as in the released code.
    uncovered : {"isolate_recursive", "attach", "isolate"}, default "isolate_recursive"
        Supernode of a node with no weight in S. "isolate" gives it its own.
        "isolate_recursive" does the same, then coarsens the coarse graph again,
        as the released MultiLayerCoarsening does, and keeps the level nearest
        the target. "attach" gives it the supernode of the nearest covered node,
        and keeps the count of the argmax.
    r_per_level : float, default 0.5
        Reduction asked at each level of "isolate_recursive", the ratio of the
        released code. A level delivers less, since isolated nodes give part of
        it back.
    return_soft_s : bool, default False
        Also return S, the pooling operator of the paper, with its zero rows,
        as info["S_soft"], and the mask info["uncovered"]. Single-level modes
        only.
    device : torch.device, optional
        Device of the computation, that of data.x by default.
    progress : bool, default False
        Show a progress bar over the epochs.

    Returns
    -------
    partition : Partition
    info : dict
        method : "otcoarsening".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        uncovered_mode : the mode used.
        n_uncovered : nodes with no weight in S, summed over the kept levels.
        seed : the seed used.
        total_time_s : wall time of the call.
        notes : a summary of the method.
        Single level: hidden_dim, n_epochs, lr, sinkhorn_iters, gamma, p_power,
            score_square, cut, zero_row_filter (the options used), device,
            final_loss, loss_history, and S_soft, uncovered when asked.
        "isolate_recursive": r_per_level, n_levels (levels kept), n_levels_run,
            levels (per-level trace).
    """
    t_total = time.perf_counter()

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    if data.x is None:
        raise ValueError("OTCoarsening requires data.x (node features).")

    if uncovered == "isolate_recursive":
        if return_soft_s:
            warnings.warn(
                "return_soft_s is dropped under uncovered='isolate_recursive': "
                "the stack holds one S per level, and their product is not the "
                "operator the returned partition comes from, since the chain "
                "composes argmaxes. Ask a single level, uncovered='attach' or "
                "'isolate', to get one.",
                stacklevel=2,
            )
        return coarsen_recursive(
            data, n_target=n_target, r_req=r_req, r_per_level=r_per_level,
            hidden_dim=hidden_dim, n_epochs=n_epochs, lr=lr,
            sinkhorn_iters=sinkhorn_iters, gamma=gamma, p_power=p_power,
            score_square=score_square, cut=cut, zero_row_filter=zero_row_filter,
            seed=seed, device=device, progress=progress, t_total=t_total,
        )

    dev = torch.device(device) if device is not None else data.x.device
    torch.manual_seed(seed)

    # Dense adjacency and its normalization.
    A_csr = data_to_torch_csr_adjacency(data, dtype=torch.float32).to(dev)
    A_dense = A_csr.to_dense()                                             # (N, N)
    A_hat = A_dense + torch.eye(N, device=dev, dtype=torch.float32)
    d_hat = A_hat.sum(dim=1)
    d_inv_sqrt = torch.where(d_hat > 0, d_hat.pow(-0.5), torch.zeros_like(d_hat))
    A_norm = d_inv_sqrt.unsqueeze(1) * A_hat * d_inv_sqrt.unsqueeze(0)     # D_hat^-1/2 A_hat D_hat^-1/2

    X = data.x.to(dev).to(torch.float32)
    F_in = X.shape[1]

    model = _OTCoarseningModel(F_in, hidden_dim, N, n_target,
                                sinkhorn_iters=sinkhorn_iters,
                                gamma=gamma, p_power=p_power,
                                score_square=score_square, cut=cut,
                                zero_row_filter=zero_row_filter).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    iterator = tqdm(range(n_epochs), disable=not progress, desc="OTCoarsen")
    for epoch in iterator:
        model.train()
        opt.zero_grad()
        loss, _ = model(X, A_dense, A_norm)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if progress:
            iterator.set_postfix_str(f"loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, S = model(X, A_dense, A_norm)                                    # (N, m) row-normalized

    # Each node joins the column of its largest weight in S.
    assignment_raw = S.argmax(dim=1).cpu().numpy().astype(np.int64)
    uncovered_mask = (S.sum(dim=1) == 0).cpu().numpy()
    if uncovered == "attach":
        assignment_raw = complete_uncovered(
            assignment_raw, uncovered_mask, data_to_scipy_csr_adjacency(data)
        )
    elif uncovered == "isolate":
        n_unc = int(uncovered_mask.sum())
        assignment_raw[uncovered_mask] = assignment_raw.max() + 1 + np.arange(n_unc)
    else:
        raise ValueError(f"uncovered must be 'attach' or 'isolate', got {uncovered!r}")
    assignment, n_actual = relabel_contiguous(assignment_raw)
    partition = Partition(assignment, n_actual)

    info = {
        "method": "otcoarsening",
        "r_requested": r_req,
        "r_actual": 1.0 - n_actual / N,
        "n_clusters": int(n_actual),
        "n_target": int(n_target),
        "hidden_dim": int(hidden_dim),
        "n_epochs": int(n_epochs),
        "lr": float(lr),
        "sinkhorn_iters": int(sinkhorn_iters),
        "gamma": float(gamma),
        "p_power": int(p_power),
        "score_square": bool(score_square),
        "cut": cut,
        "zero_row_filter": zero_row_filter,
        "seed": seed,
        "n_uncovered": int(uncovered_mask.sum()),
        "uncovered_mode": uncovered,
        "device": str(dev),
        "total_time_s": time.perf_counter() - t_total,
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "loss_history": losses,
        "notes": (
            "OTCoarsening = differentiable pooling with Sinkhorn OT loss "
            "(Ma & Chen, AAAI 2021). Partition extracted by a row-wise argmax "
            "on S, uncovered nodes handled by uncovered_mode. "
            "Dense O(N^2) memory."
        ),
    }
    if return_soft_s:
        info["S_soft"] = S.detach().cpu().to_sparse_csr()
        info["uncovered"] = torch.from_numpy(uncovered_mask)
    return partition, info


def coarsen_recursive(data, *, n_target, r_req, r_per_level, t_total, **kw):
    """Stack "isolate" levels and keep the one whose count comes nearest n_target."""
    from graco.core.builders import build_coarsening

    n_orig = int(data.num_nodes)
    chain = np.arange(n_orig, dtype=np.int64)
    current = data
    levels = []
    # Levels only shrink, so the nearest to the target is the running best or
    # the one just produced. Two compositions are enough to keep.
    best_chain, best_n, best_level = None, None, 0

    while current.num_nodes > n_target and len(levels) < 20:
        n_in = int(current.num_nodes)
        # Ask for r_per_level, unless that would overshoot: the last level aims
        # at n_target so the stack lands near the count that was requested.
        n_level = max(n_target, int(np.ceil(n_in * (1.0 - r_per_level))))
        part, info = coarsen_otcoarsening(
            current, n_clusters=n_level, uncovered="isolate", **kw
        )
        if part.n_clusters >= n_in:
            break
        chain = part.assignment[chain]
        levels.append({
            "n_in": n_in,
            "n_asked": int(n_level),
            "n_out": int(part.n_clusters),
            "n_uncovered": int(info["n_uncovered"]),
            "final_loss": info["final_loss"],
        })
        if best_chain is None or abs(part.n_clusters - n_target) < abs(best_n - n_target):
            best_chain, best_n, best_level = chain.copy(), int(part.n_clusters), len(levels)
        current = build_coarsening(
            current, part, method="otcoarsening", r_requested=1.0 - n_level / n_in
        ).coarsened

    # Halving a graph already near the target overshoots it, so the last level
    # run is not always the best one. Keep the level whose count comes closest
    # instead, which costs one extra chain in memory and nothing else.
    if best_chain is not None:
        chain = best_chain
    levels_kept = best_level

    assignment, n_actual = relabel_contiguous(chain)
    partition = Partition(assignment, n_actual)
    info = {
        "method": "otcoarsening",
        "r_requested": r_req,
        "r_actual": 1.0 - n_actual / n_orig,
        "n_clusters": int(n_actual),
        "n_target": int(n_target),
        "uncovered_mode": "isolate_recursive",
        "r_per_level": float(r_per_level),
        "n_levels": levels_kept,
        "n_levels_run": len(levels),
        "levels": levels,
        "n_uncovered": sum(l["n_uncovered"] for l in levels[:levels_kept]),
        "seed": kw.get("seed"),
        "total_time_s": time.perf_counter() - t_total,
        "notes": (
            "OTCoarsening stacked as their MultiLayerCoarsening does, one "
            "isolate level per step, until the requested supernode count is "
            "reached. info['levels'] holds the per-level trace."
        ),
    }
    return partition, info


def complete_uncovered(assignment, uncovered, adj):
    """Give every uncovered node the label of the nearest covered one.

    Breadth-first from the covered set, the lowest-indexed neighbour winning
    ties; a component with no covered node becomes one new supernode.
    """
    out = assignment.copy()
    seen = ~uncovered
    indptr, indices = adj.indptr, adj.indices
    while not seen.all():
        reached, labels = [], []
        for v in np.flatnonzero(~seen):
            nb = indices[indptr[v]:indptr[v + 1]]
            nb = nb[seen[nb]]
            if nb.size:
                reached.append(v)
                labels.append(out[nb.min()])
        if not reached:
            break
        out[np.array(reached)] = np.array(labels)
        seen[np.array(reached)] = True

    if not seen.all():
        from scipy.sparse.csgraph import connected_components
        rest = np.flatnonzero(~seen)
        n_comp, comp = connected_components(adj[rest][:, rest], directed=False)
        out[rest] = out.max() + 1 + comp
    return out


class _OTCoarseningModel(nn.Module):
    """Encoder, scorer and decoder of the paper, all dense."""
    def __init__(
        self, F_in: int, hidden: int, N: int, m: int,
        *,
        sinkhorn_iters: int, gamma: float, p_power: int,
        score_square: bool = True, cut: str = "raw",
        zero_row_filter: str = "none",
    ):
        super().__init__()
        self.F_in = F_in
        self.hidden = hidden
        self.N = N
        self.m = m
        self.sinkhorn_iters = sinkhorn_iters
        self.gamma = gamma
        self.p_power = p_power
        self.score_square = score_square
        self.cut = cut
        self.zero_row_filter = zero_row_filter

        # One score per node.
        self.gcn_att = _DenseGCNConv(F_in, 1)
        # encoder
        self.embed_enc = _DenseGCNConv(F_in, hidden)
        # The decoder maps back to the input dimension.
        self.embed_dec = _DenseGCNConv(hidden, F_in)

    def forward(self, X: torch.Tensor, A: torch.Tensor, A_norm: torch.Tensor):
        """Return the transport loss and S."""
        N, m = self.N, self.m

        # The paper takes sigmoid(A_hat X W_alpha); the released code squares
        # the pre-activation, see score_square.
        score = self.gcn_att(X, A_norm).squeeze(-1)                         # (N,)
        alpha = torch.sigmoid(score ** 2 if self.score_square else score)   # (N,)

        _, top_idx = torch.topk(alpha, m)                                   # (m,)

        if self.cut == "raw":
            # S = row-normalized A_hat_s * (1 alpha_s^T), on the picked columns.
            S_un = A_norm[:, top_idx] * alpha[top_idx].unsqueeze(0)         # (N, m)
            S = S_un / S_un.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            # cut_alpha_vec of the released code: every score shifted by the m-th largest.
            alpha_sorted, _ = torch.sort(alpha, descending=True)
            cut_alpha = F.relu(alpha + 1e-7 - alpha_sorted[m - 1])          # (N,)
            S_dense = A_norm * cut_alpha.unsqueeze(0)                       # (N, N)
            row_sums = S_dense.sum(dim=1, keepdim=True).clamp(min=1e-8)
            S = (S_dense / row_sums)[:, top_idx]                            # (N, m)

        # Encoder, with the ReLU of the released code.
        Z = F.relu(self.embed_enc(X, A_norm))                               # (N, hidden)

        # Z_c = S^T Z.
        Z_c = S.transpose(0, 1) @ Z                                         # (m, hidden)

        # A_c = S^T A S.
        A_c = S.transpose(0, 1) @ A @ S                                     # (m, m)
        # Normalize A_c for the decoder GCN
        A_c_hat = A_c + torch.eye(m, device=A.device, dtype=A.dtype)
        d_c = A_c_hat.sum(dim=1)
        d_c_inv_sqrt = torch.where(d_c > 0, d_c.pow(-0.5), torch.zeros_like(d_c))
        A_c_norm = d_c_inv_sqrt.unsqueeze(1) * A_c_hat * d_c_inv_sqrt.unsqueeze(0)

        # Decoder without a final ReLU, as in the released code, where one can
        # zero X_c.
        X_c = self.embed_dec(Z_c, A_c_norm)                                 # (m, F_in)

        # The paper's cost spans every row; the released code filters both sides
        # through get_nonzero_rows first, see zero_row_filter.
        if self.zero_row_filter == "repo":
            X_use, Xc_use = repo_nonzero_rows(X), repo_nonzero_rows(X_c)
            if X_use.shape[0] == 0 or Xc_use.shape[0] == 0:
                X_use, Xc_use = X, X_c
        else:
            X_use, Xc_use = X, X_c
        M = _pairwise_dist_pow(X_use, Xc_use, p=self.p_power)

        # Sinkhorn iterations
        loss = _sinkhorn_loss(M, self.gamma, self.sinkhorn_iters)

        return loss, S


class _DenseGCNConv(nn.Module):
    """Dense GCN layer, A_norm X W + b, as PyG's DenseGCNConv."""
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_ch, out_ch))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        out = A_norm @ X @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


def repo_nonzero_rows(M: torch.Tensor) -> torch.Tensor:
    """get_nonzero_rows of the released code, kept as it runs. The zero rows come first."""
    s = torch.abs(M.sum(dim=-1))
    _, idx = s.sort()
    n_nonzero = int((s > 0).sum())
    return M[idx[:n_nonzero]]


def _pairwise_dist_pow(X: torch.Tensor, Y: torch.Tensor, p: int) -> torch.Tensor:
    """Cost ||X_i - Y_j||_2^p for every pair of rows."""
    if p == 2:
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
        X_norm = (X * X).sum(dim=1, keepdim=True)                           # (N, 1)
        Y_norm = (Y * Y).sum(dim=1, keepdim=True).transpose(0, 1)           # (1, m)
        M = X_norm + Y_norm - 2.0 * X @ Y.transpose(0, 1)
        return M.clamp(min=0.0)
    return torch.cdist(X, Y, p=2) ** p


def _sinkhorn_loss(M: torch.Tensor, gamma: float, n_iters: int) -> torch.Tensor:
    """Entropic transport <P, M> - gamma E(P) between uniform marginals, in log space."""
    N, m = M.shape
    log_a = torch.full((N,), -np.log(N), device=M.device, dtype=M.dtype)
    log_b = torch.full((m,), -np.log(m), device=M.device, dtype=M.dtype)

    log_K = -M / gamma                                                       # (N, m)

    log_u = torch.zeros(N, device=M.device, dtype=M.dtype)
    log_v = torch.zeros(m, device=M.device, dtype=M.dtype)
    for _ in range(n_iters):
        # log v = log b - logsumexp over columns of (log_K + log u).
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)                  # (N, m)
    P = torch.exp(log_P)

    transport_cost = (P * M).sum()
    # The loss is <P, M> - gamma E(P), with E(P) = -sum P (log P - 1); the sum
    # below is -E(P), so it enters with a plus.
    neg_entropy = (P * (log_P - 1.0)).sum()
    return transport_cost + gamma * neg_entropy
