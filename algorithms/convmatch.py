# Ported from amazon-science/convolution-matching (Apache-2.0).
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# The costs follow Dickens et al. 2024, written on torch sparse CSR where the
# reference uses DGL message passing. We kept most of their state layout:
# current_sum, influence, self_loop_weight, super_node_size and degree carry
# their names and their meaning, and their scaled_feat and inv_sqrt_degree
# became z and alpha, which every pair below shares. On top of the port we add
# pair="binary" | "normalized" and merge_candidates="knn+edges", neither
# upstream, and the exact cost combines the two terms of a common neighbour,
# as their sub-graph simulation does. The exact cost is computed for all
# candidate pairs at once, where the reference scores them one by one.
# See THIRD_PARTY_NOTICES.md.
"""Convolution matching, ConvMatch (Dickens et al., WWW Companion 2024).

Greedy pair merging. Candidate pairs come from a kNN on SGC features, each
pair is scored by how much its merge changes the output of one graph
convolution, and the cheapest disjoint pairs merge at each level.
method="approximate" is A-ConvMatch, the bound of the paper, and "exact"
computes the change itself. Our port adds the pair options, the "knn+edges"
candidates, and one pairs_per_level rule, max(10, N // 25), for all graphs.
It computes the exact cost for all candidate pairs at once, where the
reference code scores them one by one.
Reference code: github.com/amazon-science/convolution-matching
Details in the method catalogue of the GraCo paper (ConvMatch).
"""

from __future__ import annotations

import time
import warnings
from typing import Literal, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from graco.algorithms.common import resolve_target, sgc_propagate
from graco.core.conversions import csr_from_triplets, extract_triplets, sparse_add
from graco.core.conversions import (
    data_to_torch_csr_adjacency,
    torch_csr_to_scipy_csr,
)
from graco.core.laplacian import LaplacianKind
from graco.core.partition import (
    Partition,
    relabel_contiguous,
    sanity_check_partition_Q,
)



# Merges per level, which sets the number of levels and so the runtime. The
# rule sits in the range Dickens uses on the OGB graphs; pass pairs_per_level
# to use one of their values.
def pairs_per_level_rule(N: int) -> int:
    return max(10, N // 25)


def coarsen_convmatch(
    data,
    r: Optional[float] = None,
    n_clusters: Optional[int] = None,
    *,
    method: Literal["approximate", "exact"] = "approximate",
    pair: Literal["dickens", "binary", "normalized"] = "dickens",
    merge_candidates: Literal["knn", "knn+edges"] = "knn",
    laplacian_kind: LaplacianKind = "normalized_self_loop",
    pairs_per_level: Optional[int] = None,
    num_hops: int = 3,
    num_nearest_neighbors: int = 25,
    top_k_nn: int = 1,
    nearest_neighbors_keep_rate: float = 0.01,
    knn_embedding_space_dim: int = 15,
    reinit_pool_multiplier: float = 1.0,
    max_levels: int = 10000,
    seed: int = 14,
    device: Optional[torch.device] = None,
    max_memory_gb: float = 5.0,
    progress: bool = True,
    log_levels: bool = False,
    validate: bool = False,
) -> tuple[Partition, dict]:
    """Merge the pairs whose merge least changes one graph convolution.

    Give exactly one of r and n_clusters.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph with node features data.x.
    r : float in (0, 1), optional
        Target reduction ratio, n = ceil((1 - r) N) supernodes.
    n_clusters : int in [1, N], optional
        Target number of supernodes.
    method : {"approximate", "exact"}, default "approximate"
        "approximate" scores a pair by the bound of the paper (A-ConvMatch),
        "exact" by the change itself, slower by the degree of the pair.
    pair : {"dickens", "binary", "normalized"}, default "dickens"
        Reduction and lifting pair inside the cost. "dickens" (paper) takes
        the coarse operator of the normalized pair and moves the features
        with the binary pair. "binary" and "normalized" use one pair
        throughout.
    merge_candidates : {"knn", "knn+edges"}, default "knn"
        Candidate pairs: the kNN pairs (paper), or their union with the edges.
    laplacian_kind : str, default "normalized_self_loop"
        Recorded in info only.
    pairs_per_level : int, optional
        Merges per level, max(10, N // 25) by default. It sets the number of
        levels, and so the runtime.
    num_hops : int, default 3
        SGC propagation steps of the candidate embedding.
    num_nearest_neighbors : int, default 25
        Neighbours queried per node in the kNN.
    top_k_nn : int, default 1
        Nearest neighbours kept per node as candidates.
    nearest_neighbors_keep_rate : float, default 0.01
        Fraction of the closest kNN pairs kept over the whole graph.
    knn_embedding_space_dim : int, default 15
        PCA dimension of the kNN embedding.
    reinit_pool_multiplier : float, default 1.0
        The candidates are recomputed on the current graph when fewer than
        this many times pairs_per_level remain, 1.0 as in the released code.
    max_levels : int, default 10000
        Maximum number of levels.
    seed : int, default 14
        Seed of torch, numpy and the PCA.
    device : torch.device, optional
        Device of the computation, data.x.device by default.
    max_memory_gb : float, default 5.0
        Rough cap on the memory of the pair costs at each level, computed by
        blocks of pairs under it. The costs do not depend on the cap.
    progress : bool, default True
        Show a progress bar over the levels.
    log_levels : bool, default False
        Print one line per level with its time split between the candidate
        refresh, the cost, the pick and the rebuild.
    validate : bool, default False
        Check the partition against its lifting, in info["sanity"].

    Returns
    -------
    partition : Partition
    info : dict
        method : "convmatch".
        convmatch_variant, pair, merge_candidates : the options used.
        r_requested, r_actual : requested and obtained reduction ratio.
        n_clusters : number of supernodes returned.
        n_target : number of supernodes asked for.
        n_levels : number of levels run.
        laplacian_kind, pairs_per_level, num_hops, num_nearest_neighbors,
            top_k_nn, nearest_neighbors_keep_rate, knn_embedding_space_dim,
            reinit_pool_multiplier : the options used.
        n_candidate_pairs : candidate pairs left at the end.
        sgcn_time_s : wall time of the first candidate search.
        level_times_s : wall time of each level.
        levels : per-level counts and time split.
        total_time_s : wall time of the call.
        stalled : True if the descent ended more than 5% above n_target.
        device : device of the computation.
        seed : the seed used.
        sanity : the partition check, only when validate is True.
    """
    t_total = time.perf_counter()
    if method not in ("approximate", "exact"):
        raise ValueError(f"method must be 'approximate' or 'exact', got {method!r}")
    if pair not in ("dickens", "binary", "normalized"):
        raise ValueError(f"pair must be 'dickens', 'binary' or 'normalized', got {pair!r}")
    if merge_candidates not in ("knn", "knn+edges"):
        raise ValueError(
            f"merge_candidates must be 'knn' or 'knn+edges', got {merge_candidates!r}"
        )
    if data.x is None:
        raise ValueError("ConvMatch needs node features (data.x).")

    N = int(data.num_nodes)
    n_target, r_req = resolve_target(N, r, n_clusters)

    if pairs_per_level is None:
        pairs_per_level = pairs_per_level_rule(N)

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = device or data.x.device
    X = data.x.to(dtype=torch.float32, device=dev)
    A_super = data_to_torch_csr_adjacency(data, dtype=torch.float32).to(dev)

    # Candidate pairs on the original graph.
    t_sgcn = time.perf_counter()
    candidate_pairs = _sgcn_candidates_torch(
        X=X,
        A=A_super,
        num_hops=num_hops,
        num_nearest_neighbors=num_nearest_neighbors,
        top_k_nn=top_k_nn,
        keep_rate=nearest_neighbors_keep_rate,
        pca_dim=knn_embedding_space_dim,
        seed=seed,
    ).to(dev)
    if merge_candidates == "knn+edges":
        edge_pairs = _upper_triangular_pairs_torch(A_super)
        candidate_pairs = _union_pairs_torch(candidate_pairs, edge_pairs)
    sgcn_time_s = time.perf_counter() - t_sgcn
    if log_levels:
        print(f"[convmatch] N={N} target={n_target} pairs_per_level={pairs_per_level} "
              f"candidates={candidate_pairs.shape[0]} in {sgcn_time_s:.1f}s",
              flush=True)

    state, W_S = _init_state_torch(A_super, X, pair)
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    original_to_super = torch.arange(N, dtype=torch.long, device=dev)

    n_cur = N
    level_times: list[float] = []
    level_infos: list[dict] = []

    iterator = tqdm(range(1, max_levels + 1), disable=not progress, desc=f"ConvMatch[{method}]")
    for level in iterator:
        if n_cur <= n_target:
            break
        t_level = time.perf_counter()

        # Filter still-alive pairs.
        u_all, v_all = candidate_pairs[:, 0], candidate_pairs[:, 1]
        keep = alive[u_all] & alive[v_all]
        pairs_alive = candidate_pairs[keep]

        # Recompute the candidates on the current graph when the pool runs low.
        reinit_threshold = int(pairs_per_level * reinit_pool_multiplier)
        t_reinit = 0.0
        if pairs_alive.shape[0] < reinit_threshold and n_cur > n_target:
            t_r0 = time.perf_counter()
            new_pairs = _sgcn_candidates_torch(
                X=state["feat"], A=A_super,
                num_hops=num_hops,
                num_nearest_neighbors=num_nearest_neighbors,
                top_k_nn=top_k_nn,
                keep_rate=nearest_neighbors_keep_rate,
                pca_dim=knn_embedding_space_dim,
                seed=seed + level,   # vary seed per re-init for diversity
                alive=alive,
                self_loop_mass=(state["super_node_size"]
                                + state["self_loop_weight"]),
            ).to(dev)
            new_pairs = new_pairs[alive[new_pairs[:, 0]] & alive[new_pairs[:, 1]]]
            if merge_candidates == "knn+edges":
                edge_pairs = _upper_triangular_pairs_torch(A_super)
                new_pairs = _union_pairs_torch(new_pairs, edge_pairs)
            if new_pairs.shape[0] > 0:
                candidate_pairs = _union_pairs_torch(pairs_alive, new_pairs)
                pairs_alive = candidate_pairs
            t_reinit = time.perf_counter() - t_r0

        if pairs_alive.shape[0] == 0:
            break

        # A[u, v] for each alive pair (0 if not connected).
        t_cost0 = time.perf_counter()
        W = W_S if pair == "binary" else A_super
        edge_w = _lookup_edge_weights_csr(W, pairs_alive[:, 0], pairs_alive[:, 1])

        costs = costs_by_blocks(state, W, pairs_alive, edge_w, pair, method, max_memory_gb)

        t_cost = time.perf_counter() - t_cost0

        # Cheapest disjoint pairs.
        t_pick0 = time.perf_counter()
        k_level = min(pairs_per_level, n_cur - n_target)
        merges, merge_edge_w = _pick_nonoverlap(costs, pairs_alive, edge_w, k=k_level)
        t_pick = time.perf_counter() - t_pick0
        if merges.shape[0] == 0:
            break

        # Merge v into u.
        t_build0 = time.perf_counter()
        merge_edge_a = (merge_edge_w if pair != "binary"
                        else _lookup_edge_weights_csr(A_super, merges[:, 0], merges[:, 1]))
        _apply_merges_state(state, merges, merge_edge_a, merge_edge_w, pair)

        alive[merges[:, 1]] = False

        # Remap the node tracker and the candidates, so a supernode inherits
        # the candidates of both halves; without it clusters stop at two nodes.
        remap = torch.arange(N, dtype=torch.long, device=dev)
        remap[merges[:, 1]] = merges[:, 0]
        original_to_super = remap[original_to_super]
        candidate_pairs = remap[candidate_pairs]
        # (u, x) became (u, u) when x merged into u.
        mask = candidate_pairs[:, 0] != candidate_pairs[:, 1]
        candidate_pairs = candidate_pairs[mask]
        # (a, u) and (a, v) both became (a, u).
        candidate_pairs, _ = torch.sort(candidate_pairs, dim=1)
        candidate_pairs = torch.unique(candidate_pairs, dim=0)

        A_super = _rebuild_A_super(A_super, merges, N)
        if pair == "binary":
            W_S = _rebuild_A_super(W_S, merges, N)
        n_cur -= int(merges.shape[0])

        # Refresh current_sum, influence and h with one product.
        _refresh_state_matmul(state, W_S if pair == "binary" else A_super)

        t_build = time.perf_counter() - t_build0

        level_times.append(time.perf_counter() - t_level)
        level_infos.append({
            "n_merges": int(merges.shape[0]),
            "n_cur_after": int(n_cur),
            "n_candidates_alive": int(pairs_alive.shape[0]),
            "reinit_s": t_reinit,
            "cost_s": t_cost,
            "pick_s": t_pick,
            "rebuild_s": t_build,
        })
        if log_levels:
            print(f"[convmatch] level {level:>4}  merged {int(merges.shape[0]):>7}  "
                  f"left {n_cur:>9}  pool {int(pairs_alive.shape[0]):>9}  "
                  f"refresh {t_reinit:>7.1f}s  cost {t_cost:>7.1f}s  "
                  f"pick {t_pick:>7.1f}s  rebuild {t_build:>7.1f}s  "
                  f"total {level_times[-1]:>7.1f}s", flush=True)

    assignment_np, n_final = relabel_contiguous(original_to_super.cpu().numpy())
    partition = Partition(assignment_np, n_final)

    r_actual = 1.0 - n_final / N
    stalled = (n_final > int(np.ceil((1.0 - r_req) * N)) * 1.05)

    info: dict = {
        "method": "convmatch",
        "convmatch_variant": method,
        "pair": pair,
        "merge_candidates": merge_candidates,
        "r_requested": r_req,
        "r_actual": r_actual,
        "n_clusters": n_final,
        "n_target": n_target,
        "n_levels": len(level_times),
        "laplacian_kind": laplacian_kind,
        "pairs_per_level": pairs_per_level,
        "num_hops": num_hops,
        "num_nearest_neighbors": num_nearest_neighbors,
        "top_k_nn": top_k_nn,
        "nearest_neighbors_keep_rate": nearest_neighbors_keep_rate,
        "knn_embedding_space_dim": knn_embedding_space_dim,
        "reinit_pool_multiplier": reinit_pool_multiplier,
        "n_candidate_pairs": int(candidate_pairs.shape[0]),
        "sgcn_time_s": sgcn_time_s,
        "level_times_s": level_times,
        "levels": level_infos,
        "total_time_s": time.perf_counter() - t_total,
        "stalled": bool(stalled),
        "device": str(dev),
        "seed": seed,
    }

    if stalled:
        warnings.warn(
            f"ConvMatch stalled: r_requested={r_req:.3f}, r_actual={r_actual:.3f}, "
            f"n_clusters={n_final} (target ~{n_target})",
            stacklevel=2,
        )

    if validate:
        Q_sp = torch_csr_to_scipy_csr(partition.to_Q_combi(dtype=torch.float32))
        info["sanity"] = sanity_check_partition_Q(partition, Q_sp)

    return partition, info


def _init_state_torch(A: torch.Tensor, X: torch.Tensor, pair: str = "dickens"):
    """Per-node state of the greedy for the chosen pair, and W_S = S off its diagonal for "binary"."""
    N, F = X.shape
    dev = X.device
    degree = (A @ torch.ones(N, dtype=torch.float32, device=dev))
    super_node_size = torch.ones(N, dtype=torch.float32, device=dev)
    self_loop_weight = torch.zeros(N, dtype=torch.float32, device=dev)
    state = dict(
        feat=X.clone(),
        super_node_size=super_node_size,
        degree=degree,
        self_loop_weight=self_loop_weight,
    )
    W_S = None
    if pair == "binary":
        inv_sqrt = 1.0 / torch.sqrt(degree + 1.0)
        crow, col, vals = A.crow_indices(), A.col_indices(), A.values()
        row = torch.repeat_interleave(torch.arange(N, device=dev), crow[1:] - crow[:-1])
        W_S = torch.sparse_csr_tensor(crow, col, vals * inv_sqrt[row] * inv_sqrt[col],
                                      size=(N, N))
        state["alpha"] = 1.0 / super_node_size
        state["z"] = X.clone()
        state["w_diag"] = inv_sqrt * inv_sqrt
    else:
        state["alpha"] = 1.0 / torch.sqrt(degree + super_node_size)
        state["z"] = X * state["alpha"].unsqueeze(1)
        state["w_diag"] = super_node_size + self_loop_weight
    _refresh_state_matmul(state, W_S if pair == "binary" else A)
    return state, W_S


def _refresh_state_matmul(state: dict, W: torch.Tensor) -> None:
    """Recompute current_sum = W z, influence = W alpha and h after a rebuild of W."""
    z = state["z"]
    alpha = state["alpha"]
    state["current_sum"] = W @ z
    state["influence"] = (W @ alpha.unsqueeze(1)).squeeze(1)
    state["h"] = (state["current_sum"] + state["w_diag"].unsqueeze(1) * z) * alpha.unsqueeze(1)


def _compact_csr_alive(A: torch.Tensor, idx: torch.Tensor, N: int) -> torch.Tensor:
    """A restricted to the nodes idx, relabeled from 0."""
    row, col, val = extract_triplets(A)
    pos = torch.full((N,), -1, dtype=torch.long, device=A.device)
    pos[idx] = torch.arange(idx.numel(), dtype=torch.long, device=A.device)
    r, c = pos[row], pos[col]
    keep = (r >= 0) & (c >= 0)
    n = int(idx.numel())
    return csr_from_triplets(r[keep], c[keep], val[keep], n, n)


def _sgcn_candidates_torch(
    X: torch.Tensor,
    A: torch.Tensor,
    num_hops: int,
    num_nearest_neighbors: int,
    top_k_nn: int,
    keep_rate: float,
    pca_dim: int,
    seed: int,
    alive: Optional[torch.Tensor] = None,
    self_loop_mass: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Candidate pairs from a kNN on the PCA of the SGC features of the alive supernodes."""
    idx = None
    if alive is not None and bool((~alive).any()):
        idx = alive.nonzero(as_tuple=True)[0]
        A = _compact_csr_alive(A, idx, int(alive.numel()))
        X = X[idx]
        if self_loop_mass is not None:
            self_loop_mass = self_loop_mass[idx]

    N, F = X.shape
    if self_loop_mass is not None:
        # sgc_propagate adds I, so the diagonal has to hold the rest of the mass.
        extra = self_loop_mass - 1.0
        nz = extra > 0
        if bool(nz.any()):
            ids = torch.arange(N, dtype=torch.long, device=A.device)[nz]
            # A + diag(extra), MKL case in needs_coo
            A = sparse_add(A, csr_from_triplets(ids, ids,
                                                extra[nz].to(A.values().dtype), N, N))

    H = sgc_propagate(X, A, num_hops)

    # PCA on CPU, with sklearn.
    H_np = H.detach().cpu().numpy()
    H_c = H_np - H_np.mean(axis=0, keepdims=True)
    H_c = H_c / (H_c.std(axis=0, keepdims=True) + 1e-4)
    dim = min(pca_dim, H_c.shape[1], H_c.shape[0])
    if dim < H_c.shape[1]:
        H_c = PCA(n_components=dim, random_state=seed).fit_transform(H_c)

    # kNN in L1.
    k = min(num_nearest_neighbors, N)
    tree = cKDTree(H_c)
    # Every core, as upstream does; scipy defaults to one.
    dists, nbrs = tree.query(H_c, k=k, p=1, eps=0.01, workers=-1)
    dists = dists[:, 1:]
    nbrs = nbrs[:, 1:]

    # Top-k_nn pairs per node.
    top_k = min(top_k_nn, nbrs.shape[1])
    idx_i = np.repeat(np.arange(N), top_k)
    idx_j = nbrs[:, :top_k].reshape(-1)
    top_pairs = np.stack([idx_i, idx_j], axis=1)

    # Global fraction of closest kNN pairs.
    all_i = np.repeat(np.arange(N), nbrs.shape[1])
    all_j = nbrs.reshape(-1)
    all_d = dists.reshape(-1)
    n_keep = int(len(all_d) * keep_rate)
    if n_keep > 0:
        keep_idx = np.argpartition(all_d, n_keep - 1)[:n_keep]
        global_pairs = np.stack([all_i[keep_idx], all_j[keep_idx]], axis=1)
    else:
        global_pairs = np.empty((0, 2), dtype=np.int64)

    # Exact-match neighbors (radius 0.01, p=2).
    exact_lists = tree.query_ball_point(H_c, r=0.01, p=2, workers=-1)
    if any(len(lst) > 1 for lst in exact_lists):
        exact_pairs = np.array(
            [(i, j) for i, lst in enumerate(exact_lists) for j in lst if j != i],
            dtype=np.int64,
        ).reshape(-1, 2)
    else:
        exact_pairs = np.empty((0, 2), dtype=np.int64)

    all_pairs = np.concatenate([top_pairs, global_pairs, exact_pairs], axis=0)
    all_pairs = np.sort(all_pairs, axis=1)
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
    all_pairs = np.unique(all_pairs, axis=0)
    pairs = torch.from_numpy(all_pairs).to(torch.long)
    if idx is not None:
        pairs = idx.cpu()[pairs]
    return pairs


def _upper_triangular_pairs_torch(A: torch.Tensor) -> torch.Tensor:
    """Undirected edges (u < v) of a CSR adjacency, as an (n_pairs, 2) tensor."""
    row, col, _ = extract_triplets(A)
    mask = row < col
    return torch.stack([row[mask], col[mask]], dim=1).to(torch.long)


def _union_pairs_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Union of two pair lists, each pair sorted."""
    all_p = torch.cat([a, b], dim=0)
    all_p, _ = torch.sort(all_p, dim=1)
    return torch.unique(all_p, dim=0)


def _merged_output(state: dict, src: torch.Tensor, dst: torch.Tensor,
                   edge_w: torch.Tensor, pair: str):
    """Output h_w, features z_w and lift factors of the supernode merging src and dst."""
    size_u, size_v = state["super_node_size"][src], state["super_node_size"][dst]
    size_w = size_u + size_v
    feat_u, feat_v = state["feat"][src], state["feat"][dst]
    z_u, z_v = state["z"][src], state["z"][dst]
    feat_w = (size_u.unsqueeze(1) * feat_u + size_v.unsqueeze(1) * feat_v) / size_w.unsqueeze(1)
    csum_w = (state["current_sum"][src] + state["current_sum"][dst]
              - z_u * edge_w.unsqueeze(1) - z_v * edge_w.unsqueeze(1))
    beta_u = beta_v = None
    if pair == "binary":
        z_w = feat_w
        diag_w = state["w_diag"][src] + state["w_diag"][dst] + 2.0 * edge_w
        h_w = (csum_w + diag_w.unsqueeze(1) * z_w) / size_w.unsqueeze(1)
        return h_w, z_w, beta_u, beta_v
    deg_u, deg_v = state["degree"][src], state["degree"][dst]
    deg_w = deg_u + deg_v
    sqrt_deg_w = torch.sqrt(deg_w + size_w)
    if pair == "dickens":
        z_w = feat_w / sqrt_deg_w.unsqueeze(1)
    else:
        sig_u, sig_v = deg_u + size_u, deg_v + size_v
        z_w = (sig_u.unsqueeze(1) * z_u + sig_v.unsqueeze(1) * z_v) / (deg_w + size_w).unsqueeze(1)
        beta_u = torch.sqrt((deg_u + 1.0) / (deg_w + 1.0))
        beta_v = torch.sqrt((deg_v + 1.0) / (deg_w + 1.0))
    sl_w = state["self_loop_weight"][src] + state["self_loop_weight"][dst] + 2.0 * edge_w
    if pair == "normalized":
        alpha_w = torch.sqrt(deg_w + 1.0) / (deg_w + size_w)
        h_w = (csum_w + (size_w + sl_w).unsqueeze(1) * z_w) * alpha_w.unsqueeze(1)
    else:
        h_w = (csum_w + (size_w + sl_w).unsqueeze(1) * z_w) / sqrt_deg_w.unsqueeze(1)
    return h_w, z_w, beta_u, beta_v


def _own_rows_cost(state, src, dst, h_w, beta_u, beta_v):
    """Change of the outputs of src and dst themselves, lifted back to them."""
    h_u, h_v = state["h"][src], state["h"][dst]
    if beta_u is None:
        return (h_w - h_u).abs().sum(dim=1) + (h_w - h_v).abs().sum(dim=1)
    return ((beta_u.unsqueeze(1) * h_w - h_u).abs().sum(dim=1)
            + (beta_v.unsqueeze(1) * h_w - h_v).abs().sum(dim=1))


def _approximate_cost_batch(
    state: dict,
    pairs: torch.Tensor,
    edge_w: torch.Tensor,
    pair: str,
) -> torch.Tensor:
    """A-ConvMatch cost of a batch of pairs: exact own rows, neighbours bounded by the influences."""
    src, dst = pairs[:, 0], pairs[:, 1]
    h_w, z_w, beta_u, beta_v = _merged_output(state, src, dst, edge_w, pair)
    cost = _own_rows_cost(state, src, dst, h_w, beta_u, beta_v)
    scaled_change_u = (state["z"][src] - z_w).abs().sum(dim=1)
    scaled_change_v = (state["z"][dst] - z_w).abs().sum(dim=1)
    alpha = state["alpha"]
    src_adj_infl = state["influence"][src] - edge_w * alpha[dst]
    dst_adj_infl = state["influence"][dst] - edge_w * alpha[src]
    return cost + scaled_change_u * src_adj_infl + scaled_change_v * dst_adj_infl


def costs_by_blocks(state, W, pairs, edge_w, pair, method, max_memory_gb):
    """Cost of every pair, by blocks of pairs whose temporaries stay under about max_memory_gb.

    Each cost only reads its own pair and the state, which stays fixed during
    the level, so the blocks give the same costs as a single pass.
    """
    F = state["z"].shape[1]
    budget = max_memory_gb * 1e9 / 4    # float32 entries
    # per pair: about sixteen F-long float32 rows at the peak of the merged
    # output (gathered features, z, sums and their temporaries); the neighbour
    # terms of the exact cost hold about six copies of max_entries, so a tenth
    # of the budget leaves room for both
    block = max(1, int(budget // (16 * F)))
    max_entries = max(F, int(budget // 10))
    out = []
    for start in range(0, pairs.shape[0], block):
        p, w = pairs[start:start + block], edge_w[start:start + block]
        if method == "approximate":
            out.append(_approximate_cost_batch(state, p, w, pair))
        else:
            out.append(_exact_cost_batch(state, W, p, w, pair, max_entries=max_entries))
    return torch.cat(out)


def _exact_cost_batch(
    state: dict,
    W: torch.Tensor,
    pairs: torch.Tensor,
    edge_w: torch.Tensor,
    pair: str,
    max_entries: int = 50_000_000,
) -> torch.Tensor:
    """Exact ConvMatch cost of a batch of pairs, in chunks of at most max_entries floats."""
    src, dst = pairs[:, 0], pairs[:, 1]
    h_w, z_w, beta_u, beta_v = _merged_output(state, src, dst, edge_w, pair)
    cost = _own_rows_cost(state, src, dst, h_w, beta_u, beta_v)
    du = z_w - state["z"][src]
    dv = z_w - state["z"][dst]
    alpha = state["alpha"]

    crow, col, vals = W.crow_indices(), W.col_indices(), W.values()
    N = crow.shape[0] - 1
    F = du.shape[1]
    row_len = (crow[src + 1] - crow[src]) + (crow[dst + 1] - crow[dst])
    budget = max(1, max_entries // max(F, 1))
    bounds = [0]
    acc = 0
    for i, n_i in enumerate(row_len.tolist()):
        if acc + n_i > budget and i > bounds[-1]:
            bounds.append(i)
            acc = 0
        acc += n_i
    bounds.append(len(row_len))

    neigh = torch.zeros_like(cost)
    for a, b in zip(bounds[:-1], bounds[1:]):
        c_all, l_all, cu_all, cv_all = [], [], [], []
        for side, (x, y) in enumerate(((src[a:b], dst[a:b]), (dst[a:b], src[a:b]))):
            cnt = crow[x + 1] - crow[x]
            c = torch.repeat_interleave(torch.arange(a, b, device=cost.device), cnt)
            start = torch.repeat_interleave(crow[x], cnt)
            offset = torch.arange(int(cnt.sum()), device=cost.device) - torch.repeat_interleave(
                torch.cumsum(cnt, 0) - cnt, cnt)
            pos = start + offset
            l, w = col[pos], vals[pos]
            keep = (l != x[c - a]) & (l != y[c - a])
            c, l, w = c[keep], l[keep], w[keep]
            c_all.append(c)
            l_all.append(l)
            cu_all.append(w if side == 0 else torch.zeros_like(w))
            cv_all.append(torch.zeros_like(w) if side == 0 else w)
        c = torch.cat(c_all)
        if c.numel() == 0:
            continue
        key = c * N + torch.cat(l_all)
        uniq, inv = torch.unique(key, return_inverse=True)
        coef_u = torch.zeros(uniq.shape[0], dtype=du.dtype, device=du.device).scatter_add_(
            0, inv, torch.cat(cu_all))
        coef_v = torch.zeros_like(coef_u).scatter_add_(0, inv, torch.cat(cv_all))
        cc, ll = uniq // N, uniq % N
        delta = coef_u.unsqueeze(1) * du[cc] + coef_v.unsqueeze(1) * dv[cc]
        neigh.scatter_add_(0, cc, alpha[ll] * delta.abs().sum(dim=1))
    return cost + neigh


def _lookup_edge_weights_csr(A: torch.Tensor, us: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
    """A[u_i, v_i] for every query, 0 where there is no edge, by binary search on row * N + col."""
    N = A.shape[0]
    r, c, vals = extract_triplets(A)

    edge_hash = (r * N + c).contiguous()
    query_hash = (us * N + vs).contiguous()

    pos = torch.searchsorted(edge_hash, query_hash, right=False)
    result = torch.zeros(us.shape[0], dtype=vals.dtype, device=A.device)
    in_range = pos < edge_hash.shape[0]
    if in_range.any():
        pos_clamped = pos.clamp(max=edge_hash.shape[0] - 1)
        candidate = edge_hash[pos_clamped]
        hits = in_range & (candidate == query_hash)
        result[hits] = vals[pos[hits]]
    return result


def _pick_nonoverlap(
    costs: torch.Tensor,
    pairs: torch.Tensor,
    edge_w: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Up to k disjoint pairs, read cheapest first among the 10 k cheapest."""
    if pairs.shape[0] == 0 or k <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device), \
               torch.empty((0,), dtype=edge_w.dtype, device=pairs.device)
    n_pool = min(10 * k, costs.shape[0])
    top_vals, top_idx = torch.topk(costs, n_pool, largest=False, sorted=True)

    top_pairs = pairs[top_idx].cpu().numpy()
    top_w = edge_w[top_idx].cpu().numpy()

    seen: set[int] = set()
    kept: list[tuple[int, int]] = []
    kept_w: list[float] = []
    for i in range(top_pairs.shape[0]):
        u, v = int(top_pairs[i, 0]), int(top_pairs[i, 1])
        if u in seen or v in seen:
            continue
        kept.append((u, v))
        kept_w.append(float(top_w[i]))
        seen.add(u); seen.add(v)
        if len(kept) == k:
            break
    if not kept:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device), \
               torch.empty((0,), dtype=edge_w.dtype, device=pairs.device)
    merges = torch.tensor(kept, dtype=torch.long, device=pairs.device)
    merges_w = torch.tensor(kept_w, dtype=edge_w.dtype, device=pairs.device)
    return merges, merges_w


def _apply_merges_state(
    state: dict,
    merges: torch.Tensor,
    edge_a: torch.Tensor,
    edge_w: torch.Tensor,
    pair: str,
) -> None:
    """Write the merged state at u for every merge (u, v); v is dead and never read again."""
    u = merges[:, 0]
    v = merges[:, 1]

    size_u = state["super_node_size"][u]; size_v = state["super_node_size"][v]
    size_w = size_u + size_v
    feat_u = state["feat"][u]; feat_v = state["feat"][v]
    feat_w = (size_u.unsqueeze(1) * feat_u + size_v.unsqueeze(1) * feat_v) / size_w.unsqueeze(1)

    deg_u = state["degree"][u]; deg_v = state["degree"][v]
    deg_w = deg_u + deg_v
    sl_w = state["self_loop_weight"][u] + state["self_loop_weight"][v] + 2.0 * edge_a

    if pair == "binary":
        alpha_w = 1.0 / size_w
        z_w = feat_w
        w_diag_w = state["w_diag"][u] + state["w_diag"][v] + 2.0 * edge_w
    else:
        alpha_w = 1.0 / torch.sqrt(deg_w + size_w)
        if pair == "normalized":
            alpha_w = torch.sqrt(deg_w + 1.0) / (deg_w + size_w)
        if pair == "dickens":
            z_w = feat_w * alpha_w.unsqueeze(1)
        else:
            sig_u = (deg_u + size_u).unsqueeze(1); sig_v = (deg_v + size_v).unsqueeze(1)
            z_w = (sig_u * state["z"][u] + sig_v * state["z"][v]) / (deg_w + size_w).unsqueeze(1)
        w_diag_w = size_w + sl_w

    # Commit at u.
    state["super_node_size"][u] = size_w
    state["feat"][u] = feat_w
    state["degree"][u] = deg_w
    state["self_loop_weight"][u] = sl_w
    state["alpha"][u] = alpha_w
    state["z"][u] = z_w
    state["w_diag"][u] = w_diag_w
    # current_sum, influence, h are refreshed globally after the rebuild of W.


def _rebuild_A_super(A: torch.Tensor, merges: torch.Tensor, N: int) -> torch.Tensor:
    """A with every merged v mapped to its u, without diagonal."""
    row, col, vals = extract_triplets(A)
    row = row.clone()
    col = col.clone()
    vals = vals.clone()

    # remap[v_i] = u_i, else identity.
    remap = torch.arange(N, dtype=torch.long, device=A.device)
    remap[merges[:, 1]] = merges[:, 0]
    row = remap[row]
    col = remap[col]

    # Intra-cluster edges leave A; self_loop_weight keeps them.
    mask = row != col
    row = row[mask]; col = col[mask]; vals = vals[mask]

    # The remap creates duplicates, which only the COO coalesce sums.
    A_new = torch.sparse_coo_tensor(
        torch.stack([row, col]), vals, (N, N)
    ).coalesce().to_sparse_csr()
    return A_new
