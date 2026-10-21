"""Scalable Gromov-Wasserstein learning partition, SGWL (Xu et al., NeurIPS 2019).

Recursive partitioning. Each part is split in cluster_num by a
Gromov-Wasserstein transport towards a target graph of isolated nodes with
self-loops, solved by proximal gradient with Sinkhorn steps, and a node goes
to the target of largest normalized mass. Neither the paper nor the released
code takes a reduction ratio, so stop_rule="count" (default) splits the
largest part until n_target parts exist; "size" is their rule.
Points the paper leaves open, settled from the released code: the target
marginal interpolates the sorted source one, which breaks the symmetry of a
uniform target; the Sinkhorn dual carries across outer steps, in exp form;
the node prior is squared; the cost (A + A^T) / 2 goes with beta = 0.15, as
in their recursive script; and the number of outer steps is the size of
each sub-graph.
cost_source="S", the GCN propagation as cost, is our addition.
Reference code: github.com/HongtengXu/s-gwl
Details in the method catalogue of the GraCo paper (SGWL).
"""

from __future__ import annotations

import heapq
import time
from typing import Optional

import torch

from graco.algorithms.common import resolve_target
from graco.core.conversions import csr_from_triplets, extract_triplets
from graco.core.partition import Partition, relabel_contiguous
from graco.core.propag import build_S


def coarsen_gw_sgwl(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    seed: int = 14,
    cost_source: str = "A_norm",
    beta: float = 0.15,
    outer_iteration: Optional[int] = None,
    inner_iteration: int = 5,
    node_prior: float = 1e-4,
    cluster_num: int = 2,
    stop_rule: str = "count",
    max_node_num: Optional[int] = None,
    node_distribution_b: float = 0.001,
    node_distribution_a: float = 0.0,
    device: Optional[torch.device] = None,
) -> tuple[Partition, dict]:
    """Split the graph recursively by Gromov-Wasserstein transport.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph, read through its edge_index and its edge_weight if present.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    seed : int, default 14
        Unused, since SGWL is deterministic. Kept for the common signature.
    cost_source : {"A_norm", "S"}, default "A_norm"
        Source cost: (A + A^T) / 2 (paper), or the GCN propagation.
    beta : float, default 0.15
        Weight of the proximal KL term; it has to match the scale of the cost.
    outer_iteration : int, optional
        Proximal steps per split, the size of the sub-graph by default, as in
        the released code.
    inner_iteration : int, default 5
        Sinkhorn steps per proximal step.
    node_prior : float, default 1e-4
        Weight of the node prior.
    cluster_num : int, default 2
        Number of parts of each split.
    stop_rule : {"count", "size"}, default "count"
        "count" splits the largest part until n_target parts exist. "size"
        makes a part a leaf once it holds at most max_node_num nodes, as the
        released code does; the count then depends on the graph.
    max_node_num : int, optional
        Threshold of the size rule, N // n_target by default.
    node_distribution_b : float, default 0.001
        Exponent b of the node measure (deg + a)^b.
    node_distribution_a : float, default 0.0
        Offset a of the same measure. Set it above 0 on a graph with isolated
        nodes, which otherwise get no mass.
    device : torch.device, optional
        Device of the computation, the CPU by default.

    Returns
    -------
    partition : Partition
    info : dict
        method : "sgwl".
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        seed : the seed passed.
        cost_source, beta, outer_iteration, inner_iteration, node_prior,
            cluster_num, stop_rule, node_distribution_b,
            node_distribution_a : the options used.
        max_node_num : the threshold used, None under the count rule.
        levels_reached : depth of the recursion.
        n_leaves : number of parts.
        total_time_s : wall time of the call.
        device : device the recursion ran on.
    """
    t0 = time.perf_counter()
    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    # A config file can give 2e-7 as a string, so cast every number.
    beta = float(beta)
    outer_iteration = None if outer_iteration is None else int(outer_iteration)
    inner_iteration = int(inner_iteration)
    node_prior = float(node_prior)
    cluster_num = int(cluster_num)
    max_node_num = None if max_node_num is None else int(max_node_num)
    node_distribution_b = float(node_distribution_b)
    node_distribution_a = float(node_distribution_a)

    if stop_rule not in {"count", "size"}:
        raise ValueError(f"stop_rule must be 'count' or 'size', got {stop_rule!r}")

    if cost_source not in {"A_norm", "S"}:
        raise ValueError(
            f"cost_source must be 'A_norm' or 'S', got {cost_source!r}"
        )

    A = _edge_index_to_csr(data.edge_index, N,
                           getattr(data, "edge_weight", None))
    if cost_source == "A_norm":
        C_s = _build_A_norm(A)
    else:
        C_s = build_S(data, mode="self_loop_adj", dtype=torch.float32)
    p_s = _degree_node_distribution(A, b=node_distribution_b,
                                    a=node_distribution_a)
    # CPU by default: most splits are small, where a GPU launch costs more
    # than the step.
    dev = device or torch.device("cpu")
    C_s = C_s.to(dev)
    p_s = p_s.to(dev)

    if max_node_num is None:
        effective_max_node_num = max(2, N // max(1, n_target))
    else:
        effective_max_node_num = max_node_num

    hyp = {
        "beta": beta,
        "outer_iteration": outer_iteration,
        "inner_iteration": inner_iteration,
        "node_prior": node_prior,
    }

    assignment = torch.full((N,), -1, dtype=torch.long, device=dev)
    stats = {"levels_reached": 0, "n_leaves": 0}
    if stop_rule == "count":
        _partition_by_count(
            C_s=C_s, p_s=p_s, assignment=assignment,
            n_target=n_target, cluster_num=cluster_num, hyp=hyp, stats=stats,
        )
    else:
        _recursive_partition(
            C_s=C_s, p_s=p_s,
            node_idx=torch.arange(N, dtype=torch.long, device=dev),
            assignment=assignment,
            cluster_num=cluster_num,
            depth=0,
            max_node_num=effective_max_node_num,
            hyp=hyp,
            stats=stats,
        )
    unassigned = torch.nonzero(assignment == -1, as_tuple=True)[0]
    if unassigned.numel() > 0:
        next_id = int(assignment.max().item()) + 1
        for i, u in enumerate(unassigned.tolist()):
            assignment[u] = next_id + i

    assignment_np = assignment.cpu().numpy()
    assignment_np, n_actual = relabel_contiguous(assignment_np)
    partition = Partition(assignment_np, n_actual)

    info = {
        "method": "sgwl",
        "r_requested": r_req,
        "r_actual": 1.0 - n_actual / N,
        "n_clusters": n_actual,
        "n_target": n_target,
        "seed": seed,
        "cost_source": cost_source,
        "beta": beta,
        "outer_iteration": outer_iteration,
        "inner_iteration": inner_iteration,
        "node_prior": node_prior,
        "cluster_num": cluster_num,
        "stop_rule": stop_rule,
        "max_node_num": effective_max_node_num if stop_rule == "size" else None,
        "node_distribution_b": node_distribution_b,
        "node_distribution_a": node_distribution_a,
        "levels_reached": stats["levels_reached"],
        "n_leaves": stats["n_leaves"],
        "total_time_s": time.perf_counter() - t0,
        # Read off the matrix the recursion consumed, not off the request, so a
        # run that quietly stayed on the host says so.
        "device": str(C_s.device),
    }
    return partition, info


def _edge_index_to_csr(
    edge_index: torch.Tensor, N: int, edge_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """Symmetric torch CSR adjacency, each edge at the larger of its two weights."""
    row = edge_index[0].cpu().long()
    col = edge_index[1].cpu().long()
    if edge_weight is None:
        val = torch.ones(row.numel(), dtype=torch.float32, device=row.device)
    else:
        val = edge_weight.cpu().to(torch.float32)

    all_row = torch.cat([row, col])
    all_col = torch.cat([col, row])
    all_val = torch.cat([val, val])
    key = all_row * (N + 1) + all_col
    order = torch.argsort(key)
    all_row = all_row[order]; all_col = all_col[order]; all_val = all_val[order]
    key = key[order]
    keep = torch.ones(key.numel(), dtype=torch.bool, device=key.device)
    if key.numel() > 1:
        keep[1:] = key[1:] != key[:-1]
    all_row = all_row[keep]; all_col = all_col[keep]; all_val = all_val[keep]
    return csr_from_triplets(all_row, all_col, all_val, N, N)


def _build_A_norm(A: torch.Tensor) -> torch.Tensor:
    """Source cost (A + A^T) / 2, as their recursive script passes it."""
    N = A.shape[0]
    row, col, val = extract_triplets(A)
    scaled = val / 2.0
    all_row = torch.cat([row, col])
    all_col = torch.cat([col, row])
    all_val = torch.cat([scaled, scaled])
    # Coalesce duplicate (row, col) entries by summing their values.
    key = all_row * N + all_col
    unique_key, inverse = torch.unique(key, return_inverse=True)
    coalesced_val = torch.zeros(
        unique_key.numel(), dtype=all_val.dtype, device=all_val.device,
    )
    coalesced_val.scatter_add_(0, inverse, all_val)
    new_row = unique_key // N
    new_col = unique_key % N
    return csr_from_triplets(new_row, new_col, coalesced_val, N, N)


def _degree_node_distribution(
    A: torch.Tensor, b: float = 0.001, a: float = 0.0,
) -> torch.Tensor:
    """Node measure (deg + a)^b, normalized, of shape (N, 1)."""
    deg = A.sum(dim=1, keepdim=True).to_dense().squeeze(1)
    p = torch.pow(deg + a, b)
    total = p.sum()
    if total.item() == 0.0:
        p = torch.ones_like(p) / p.numel()
    else:
        p = p / total
    return p.unsqueeze(1)                                   # (N, 1)


def _gw_proximal(
    C_s: torch.Tensor, C_t: torch.Tensor,
    p_s: torch.Tensor, p_t: torch.Tensor,
    hyp: dict,
) -> tuple[torch.Tensor, float]:
    """Transport and d_GW between C_s and C_t, by proximal gradient and Sinkhorn."""
    N_s = C_s.shape[0]
    N_t = C_t.shape[0]
    beta = hyp["beta"]

    T = p_s @ p_t.T                                         # (N_s, N_t)


    # L_const = (C_s^2 @ p_s) 1^T + 1 (p_t^T C_t^2)^T
    Cs_sq = _csr_elementwise_square(C_s)
    Ct_sq = _csr_elementwise_square(C_t)
    Cs_sq_ps = (Cs_sq @ p_s).to_dense()                     # (N_s, 1)
    Ct_sq_pt = (Ct_sq @ p_t).to_dense()                     # (N_t, 1)
    L_const = Cs_sq_ps.expand(N_s, N_t) + Ct_sq_pt.T.expand(N_s, N_t)

    prior = hyp["node_prior"]
    if prior != 0.0:
        L_const = L_const + prior * (p_s - p_t.T) ** 2

    # The dual `a` carries across outer iterations, as it does in their loop.
    a = torch.ones((N_s, 1), dtype=T.dtype, device=T.device) / N_s
    b = torch.ones((N_t, 1), dtype=T.dtype, device=T.device) / N_t

    # _split_subgraph builds a diagonal C_t, for which T C_t^T is a column
    # scaling; any other C_t takes the full product.
    ct_diag = _diagonal_or_none(C_t)

    def cross(T):
        if ct_diag is not None:
            return C_s @ (T * ct_diag)
        return C_s @ (T @ C_t.transpose(0, 1))

    for _outer in range(hyp["outer_iteration"]):
        L = L_const - 2.0 * cross(T)
        kernel = T * torch.exp(-L / max(beta, 1e-30))
        kernel_T = kernel.transpose(0, 1)
        for _inner in range(hyp["inner_iteration"]):
            b = p_t / (kernel_T @ a)
            a = p_s / (kernel @ b)
        T = (a @ b.T) * kernel

    L_final = L_const - 2.0 * cross(T)
    d_gw = float((L_final * T).sum().item())
    return T, d_gw


def _diagonal_or_none(C: torch.Tensor) -> torch.Tensor | None:
    """The diagonal of C as a (1, n) row when C is diagonal, else None."""
    if C.layout == torch.sparse_csr:
        row, col, val = extract_triplets(C)
        if not bool((row == col).all()):
            return None
        d = torch.zeros(C.shape[1], dtype=val.dtype, device=val.device)
        d[row] = val
        return d.unsqueeze(0)
    d = torch.diagonal(C)
    return d.unsqueeze(0) if bool((C == torch.diag(d)).all()) else None


def _csr_elementwise_square(C: torch.Tensor) -> torch.Tensor:
    """Return a new torch CSR with `val` squared elementwise."""
    row, col, val = extract_triplets(C)
    return csr_from_triplets(row, col, val * val, C.shape[0], C.shape[1])




def _estimate_target_distribution(
    p_s: torch.Tensor, dim_t: int,
) -> torch.Tensor:
    """Target marginal: p_s sorted in decreasing order, interpolated at dim_t points."""
    # Both grids index ps_sorted below, so they have to sit where it sits.
    x_t = torch.linspace(0.0, 1.0, dim_t, dtype=p_s.dtype, device=p_s.device)
    ps_sorted, _ = torch.sort(p_s[:, 0], descending=True)
    x_s = torch.linspace(0.0, 1.0, ps_sorted.numel(), dtype=p_s.dtype,
                         device=p_s.device)
    idx = torch.searchsorted(x_s, x_t).clamp(1, x_s.numel() - 1)
    x0 = x_s[idx - 1]; x1 = x_s[idx]
    y0 = ps_sorted[idx - 1]; y1 = ps_sorted[idx]
    slope = (y1 - y0) / (x1 - x0).clamp(min=1e-30)
    p_t = y0 + slope * (x_t - x0)
    p_t = p_t.unsqueeze(1)
    p_t = p_t / p_t.sum().clamp(min=1e-30)
    return p_t


def _sub_csr(C: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Rows and columns idx of a CSR matrix, idx[i] becoming local node i."""
    crow, col_all, val_all = C.crow_indices(), C.col_indices(), C.values()
    local_of = torch.full((C.shape[0],), -1, dtype=torch.long, device=C.device)
    local_of[idx] = torch.arange(idx.numel(), dtype=torch.long, device=C.device)

    # Gather only the rows idx names. The recursion splits the same graph over
    # and over, so reading every non-zero on each call costs the whole graph at
    # every node of the split tree.
    starts = crow[idx]
    counts = crow[idx + 1] - starts
    cum_excl = torch.cumsum(counts, 0) - counts
    pos = (torch.arange(int(counts.sum()), dtype=torch.long, device=C.device)
           + torch.repeat_interleave(starts - cum_excl, counts))

    col_local = local_of[col_all[pos]]
    keep = col_local >= 0
    row = torch.repeat_interleave(
        torch.arange(idx.numel(), dtype=torch.long, device=C.device), counts)
    return csr_from_triplets(
        row[keep], col_local[keep], val_all[pos[keep]], idx.numel(), idx.numel(),
    )


def _split_subgraph(
    C_s: torch.Tensor, p_s: torch.Tensor, node_idx: torch.Tensor,
    cluster_num: int, hyp: dict,
):
    """Children of one split of the sub-graph node_idx, or None when all its nodes land together."""
    N_sub = int(node_idx.numel())
    C_sub = _sub_csr(C_s, node_idx)
    p_sub = p_s[node_idx]
    p_sub = p_sub / p_sub.sum().clamp(min=1e-30)

    p_t = _estimate_target_distribution(p_sub, dim_t=cluster_num)
    ids = torch.arange(cluster_num, dtype=torch.long, device=C_s.device)
    C_t = csr_from_triplets(ids, ids, p_t[:, 0], cluster_num, cluster_num)

    # The released recursion sets this value to the sub-graph size.
    hyp_lvl = dict(hyp)
    if hyp["outer_iteration"] is None:
        hyp_lvl["outer_iteration"] = max(1, N_sub)

    T, _d_gw = _gw_proximal(C_sub, C_t, p_sub, p_t, hyp_lvl)

    labels = (T / p_t.T.clamp(min=1e-30)).argmax(dim=1)
    if torch.unique(labels).numel() < 2:
        return None
    return [node_idx[labels == k] for k in range(cluster_num)
            if bool((labels == k).any())]


def make_leaf(node_idx, assignment, stats):
    current_max = int(assignment.max().item())
    assignment[node_idx] = current_max + 1 if current_max >= 0 else 0
    stats["n_leaves"] += 1


def _recursive_partition(
    C_s: torch.Tensor, p_s: torch.Tensor,
    node_idx: torch.Tensor,
    assignment: torch.Tensor,
    *,
    cluster_num: int, depth: int, max_node_num: int,
    hyp: dict, stats: dict,
) -> None:
    """Size rule of the released code: a part of at most max_node_num nodes becomes a leaf."""
    stats["levels_reached"] = max(stats["levels_reached"], depth)
    N_sub = int(node_idx.numel())

    if N_sub <= max_node_num or N_sub < cluster_num:
        make_leaf(node_idx, assignment, stats)
        return

    children = _split_subgraph(C_s, p_s, node_idx, cluster_num, hyp)

    # No-progress guard: a degenerate split returns the same set, which the
    # released loop would queue again. We stop and take it as a leaf.
    if children is None:
        make_leaf(node_idx, assignment, stats)
        return

    for child in children:
        _recursive_partition(
            C_s=C_s, p_s=p_s, node_idx=child, assignment=assignment,
            cluster_num=cluster_num, depth=depth + 1,
            max_node_num=max_node_num, hyp=hyp, stats=stats,
        )


def _partition_by_count(
    C_s: torch.Tensor, p_s: torch.Tensor,
    assignment: torch.Tensor,
    *,
    n_target: int, cluster_num: int, hyp: dict, stats: dict,
) -> None:
    """Count rule: split the largest part first, until n_target parts exist."""
    N = int(assignment.numel())
    pending = [(-N, 0, 0, torch.arange(N, dtype=torch.long, device=C_s.device))]
    done = []
    tie = 1

    while pending and len(pending) + len(done) < n_target:
        _, _, depth, node_idx = heapq.heappop(pending)
        stats["levels_reached"] = max(stats["levels_reached"], depth)
        if int(node_idx.numel()) < cluster_num:
            done.append(node_idx)
            continue
        children = _split_subgraph(C_s, p_s, node_idx, cluster_num, hyp)
        if children is None:
            done.append(node_idx)
            continue
        for child in children:
            heapq.heappush(
                pending, (-int(child.numel()), tie, depth + 1, child))
            tie += 1

    for _, _, depth, node_idx in pending:
        stats["levels_reached"] = max(stats["levels_reached"], depth)
        done.append(node_idx)
    for node_idx in done:
        make_leaf(node_idx, assignment, stats)


def d_gw_proximal(
    C_s: torch.Tensor, C_t: torch.Tensor,
    p_s: torch.Tensor, p_t: torch.Tensor,
    *,
    beta: float = 0.15,
    outer_iteration: int = 300,
    inner_iteration: int = 5,
    node_prior: float = 0.0,
) -> float:
    """d_GW(C_s, C_t) by the SGWL proximal solver.

    beta has to match the scale of C_s; the default suits cost matrices of
    order one.
    """
    hyp = {
        "beta": beta,
        "outer_iteration": outer_iteration,
        "inner_iteration": inner_iteration,
        "node_prior": node_prior,
    }
    _T, d_gw = _gw_proximal(C_s, C_t, p_s, p_t, hyp)
    return d_gw
