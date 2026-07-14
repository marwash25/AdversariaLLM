"""Find a direction w in the dual cone of adjacent embedding differences."""
from math import sqrt, inf
import os
import time
import logging
from typing import Tuple, Any, Literal

import numpy as np
import torch
from scipy.optimize import linprog
from torch import Tensor
from fast_soft_sort.pytorch_ops import soft_sort
from transformers import PreTrainedModel


#TODO: remove if not used
def _find_min_gap_permutation(embedding_matrix: Tensor) -> Tuple[Tensor, float]:
    r"""Sort rows of embedding matrix based on their jth coordinate in non-decreasing order, 
    for j \in [d] with the largest minimal gap between adjacent rows, i.e.,  
    \max_{j \in [d]} \min_{i \in [k-1]} (E_{\sigma^j_{i+1}, j} - E_{\sigma^j_i, j}), 
    where \sigma^j is such that E_{\sigma_k, j} \geq \ldots \geq E_{\sigma_0, j}.
    Return reordered embedding matrix and the corresponding permutation.
    """
    k, d = embedding_matrix.shape
    max_min_gap = -float("inf")
    best_perm = torch.arange(k, device=embedding_matrix.device)
    best_j = None
    for j in range(d):
        perm = embedding_matrix[:, j].argsort(stable=True)
        gaps = embedding_matrix[perm, j].diff()
        min_gap = gaps.min().item()
        if min_gap > max_min_gap:
            max_min_gap = min_gap
            best_perm = perm
            best_j = j
    logging.info(f"Max min gap: {max_min_gap:.6g}, achieved at j = {best_j}")

    return best_perm, max_min_gap

def _projections_min_gap(E: Tensor, w: Tensor) -> Tuple[float, Tensor, Tensor]:
    proj = E @ w
    perm = proj.argsort(stable=True)
    sorted_proj = proj[perm]
    min_gap = sorted_proj.diff().min().item()
    return min_gap, perm, sorted_proj

def _embeddings_pca(embedding_matrix: Tensor) -> Tuple[Tensor, Tensor, float, Tensor]:
    """
    Find unit vector w that maximizes the sum of all squared pairwise gaps between embedding projections on w, i.e., 
    solve the PCA problem: 

            \max_{\| w \| \leq 1} w^\top M w = largest eigenvalue of M.

    where M  = 2k (\tilde{E}^T \tilde{E}) and \tilde{E} is the mean-centered matrix, 
    where each row is $E_i - \bar{E}$ with $\bar{E}$ the mean of the embedding vectors.

    Returns:
        w: eigenvector corresponding to the largest eigenvalue of M.
        perm: permutation sorting E @ w non-decreasingly.
        min_gap: min gap between adjacent embedding projections on w.
        sorted_embedding_projections: sorted projections of the embedding vectors on w.
    """
    if embedding_matrix.dtype != torch.float64:
        logging.warning("Converting E to float64 precision")
        embedding_matrix = embedding_matrix.double()

    k = embedding_matrix.shape[0]
    embeddings_mean = embedding_matrix.mean(dim=0, keepdim=True) # shape (1, d)
    # w = embeddings_mean[0].T / embeddings_mean.norm() # shape (d,)
    # min_gap, perm, sorted_proj = _projections_min_gap(embedding_matrix, w)
    # logging.info(f"Min gap achieved with embeddings mean unit vector w: {min_gap:.6g}") # 2.78673e-11 for Llama-3.2-1B-Instruct

    E_centered = embedding_matrix - embeddings_mean
    M = 2 * k * (E_centered.T @ E_centered) # shape (d, d)
    # compute full eigendecomposition (cheap relative to computing M: O(d^3) vs O(k d^2))
    eigenvalues, eigenvectors = torch.linalg.eigh(M) 
    logging.info(f"largest eigenvalue of embeddings covariance matrix = {eigenvalues[-1]:.6g}")
    w = eigenvectors[:, -1]
    w = w / w.norm()

    min_gap, perm, sorted_proj = _projections_min_gap(embedding_matrix, w)
    logging.info(f"Min gap achieved with PCA unit vector w: {min_gap:.6g}")

    return w, perm, min_gap, sorted_proj

def _randomly_permute_embeddings(embedding_matrix: Tensor, num_samples: int = 1) -> Tuple[Tensor, Tensor, float, Tensor]:
    """
    Sample max_retries random unit vectors w. Return one with largest minimum gap between adjacent embedding projections on w, 
    and the corresponding permutation that sorts the projections in non-decreasing order.
    """
    # we need to use float64 precision, otherwise couldn't find valid w even after 100 attempts 
    # for Llama-3.2-1B-Instruct, 1st conversation in adv_behaviors 
    if embedding_matrix.dtype != torch.float64:
        logging.warning("Converting E to float64 precision")
        embedding_matrix = embedding_matrix.double()

    k, d = embedding_matrix.shape
    best_min_gap = -inf

    for i in range(num_samples):
        w = torch.randn(d, dtype=torch.float64, device=embedding_matrix.device)
        w = w / w.norm()
        min_gap, perm, sorted_proj = _projections_min_gap(embedding_matrix, w)
        if min_gap > best_min_gap:
            best_min_gap = min_gap
            best_w = w
            best_perm = perm
            best_sorted_proj = sorted_proj
            logging.info(f"Found a random unit vector w with min gap {min_gap:.6g} at attempt {i+1}.")
            
    if best_min_gap <= 0.0:
        raise ValueError(
            f"Could not find a random unit vector w with distinct projections for all {k} rows "
            f"after {num_samples} attempts."
        )

    logging.info(f"Best min gap achieved with {num_samples} random samples of unit vector w: {best_min_gap:.6g}")
    return best_w, best_perm, best_min_gap, best_sorted_proj


def _valid_embeddings(
    model: PreTrainedModel,
    valid_token_ids: Tensor,
) -> Tensor:
    embedding_layer = model.get_input_embeddings()
    E = embedding_layer.weight[valid_token_ids].detach().float()
    if hasattr(embedding_layer, "embed_scale"):
        E = E * embedding_layer.embed_scale.float()
    return E


def _sorted_valid_projections(
    model: PreTrainedModel,
    valid_token_ids: Tensor,
    perm: Tensor,
    w: Tensor,
) -> Tensor:
    E = _valid_embeddings(model, valid_token_ids)
    # Match float64 precision used in _find_embeddings_dual_cone_w
    E = E.to(device=perm.device).double()
    projections = E @ w.to(device=perm.device, dtype=torch.float64)
    return projections[perm]


def _embeddings_pairwise_ext_dist(
    E: Tensor,
    mode: Literal["min", "max"],
    block_size: int = 2048,
) -> Tuple[float, int, int]:
    """Compute the min or max pairwise L2 distance over rows i < j.

    Computed in blocks of size block_size to avoid OOM error.
    """
    k, _ = E.shape
    if k < 2:
        return (inf if mode == "min" else -inf), -1, -1

    seeking_min = mode == "min"
    sentinel = inf if seeking_min else -inf
    ext_dist = torch.tensor(sentinel, device=E.device, dtype=E.dtype)
    ext_i, ext_j = -1, -1
    for i_start in range(0, k, block_size):
        i_end = min(i_start + block_size, k)
        Ei = E[i_start:i_end]
        for j_start in range(i_start, k, block_size):
            j_end = min(j_start + block_size, k)
            Ej = E[j_start:j_end]
            dists = torch.cdist(Ei, Ej, p=2)
            if i_start == j_start:
                mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
                dists = dists.masked_fill(~mask, sentinel)
            if seeking_min:
                block_ext, flat_idx = dists.min(), dists.argmin()
                is_better = block_ext < ext_dist
            else:
                block_ext, flat_idx = dists.max(), dists.argmax()
                is_better = block_ext > ext_dist
            if is_better:
                ext_dist = block_ext
                local_i = flat_idx // dists.shape[1]
                local_j = flat_idx % dists.shape[1]
                ext_i = i_start + local_i.item()
                ext_j = j_start + local_j.item()

    return ext_dist.item(), ext_i, ext_j


def _embeddings_min_dist(E: Tensor, block_size: int = 2048) -> float:
    """Compute min_{i < j} ||E_i - E_j||_2 over embedding rows and log it."""
    t0 = time.time()
    min_dist, min_i, min_j = _embeddings_pairwise_ext_dist(E, mode="min", block_size=block_size)
    elapsed = time.time() - t0
    if min_i < 0:
        logging.info("Embeddings min pairwise l2 distance = N/A (k < 2)")
        return min_dist

    logging.info(
        f"Embeddings min pairwise l2 distance = {min_dist:.6g} "
        f"at pair ({min_i}, {min_j}) "
        f"(computed in {elapsed:.2f}s)"
    )

    w = (E[min_i] - E[min_j]) / min_dist
    min_gap, perm, sorted_proj = _projections_min_gap(E, w)
    logging.info(f"Min gap achieved with min dist unit vector w: {min_gap:.6g}") # min_gap = 0 for Llama-3.2-1B-Instruct

    return min_dist


def _embeddings_max_dist(E: Tensor, block_size: int = 2048) -> float:
    """Compute max_{i < j} ||E_i - E_j||_2 over embedding rows and log it."""
    t0 = time.time()
    max_dist, max_i, max_j = _embeddings_pairwise_ext_dist(E, mode="max", block_size=block_size)
    elapsed = time.time() - t0
    if max_i < 0:
        logging.info("Embeddings max pairwise l2 distance = N/A (k < 2)")
        return max_dist

    logging.info(
        f"Embeddings max pairwise l2 distance = {max_dist:.6g} "
        f"at pair ({max_i}, {max_j}) "
        f"(computed in {elapsed:.2f}s)"
    )

    return max_dist


def _solve_dual_cone_lp(
    neg_U: Tensor,
) -> Tuple[Tensor, float, Any]:
    """Solve the LP problem:
       max_{t >= 0, w in [-1, 1]^d} t  subject to  U w >= t.
    """
    #TODO: maybe we should use float64 here too?
    n_ineq, d = neg_U.shape
    # Solve LP with linprog: min c^T x subject to A_ub x <= b_ub, x in bounds.
    # x = [w_0, ..., w_{d-1}, t], c = [0, ..., 0, -1], A_ub = [-U, 1], b_ub = 0,
    # bounds = [-1, 1]^d x [0, None].
    A_ub = np.empty((n_ineq, d + 1), dtype=np.float32)
    A_ub[:, :d] = neg_U
    A_ub[:, d] = 1.0

    c = np.zeros(d + 1, dtype=np.float32)
    c[-1] = -1.0
    b_ub = np.zeros(n_ineq, dtype=np.float32)
    bounds = [(-1.0, 1.0)] * d + [(0.0, None)]

    logging.info(
        f"Solving LP with {d + 1} variables and {n_ineq} constraints"
    )
    lp_result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"disp": True})  # set disp to False when done debugging
    if not lp_result.success:
        raise RuntimeError(f"LP failed: {lp_result.message}")

    t_opt = float(-lp_result.fun)
    
    w_opt = torch.tensor(lp_result.x[:-1], dtype=torch.float32)

    lambdas = -lp_result.ineqlin.marginals  # dual variables / Lagrange multipliers
    if not (lambdas >= 0.0).all():
        logging.warning("Lambdas are not non-negative.")
    if abs(lambdas.sum() - 1.0) > 1e-12:
        logging.warning(f"Lambdas do not sum to 1.")

    return w_opt, t_opt, lp_result


# TODO: refactor to have one common PGM solver used both here and in pgm_lovasz in dsm_optimizers.py
def _solve_dual_cone_pgm(
    E: Tensor,
    w_init: Tensor,
    norm: Literal["l2", "linf"] = "l2",
    normalize: bool = False,
    sort_epsilon: float = 1.0,
    min_epsilon: float = 1.0,
    sort_reg: Literal["l2", "kl"] | None = None,
    num_steps: int = 2000,
    log_every: int = 200,
) -> Tuple[Tensor, float, Tensor, Tensor]:
    """Run projected subgradient method (PGM) on the following problem:
       max_{\| w\| <= 1} \min_{i \in [k-1]} (B sort(E w))_i, 
    where B is the matrix with rows e_{i+1} - e_{i}, and sort operation
    is applied in non-decreasing order.
    
    Args:
        E: 2D tensor.
        w_init: 1D tensor, initial direction. 
        norm: norm used to constrain w, "l2" or "linf".
        normalize: if True, normalize gradients, otherwise use L = 
        sort_epsilon: if 0, use hard sort; if > 0, use soft sort (via fast-soft-sort).
        min_epsilon: if 0, use hard min; if > 0, use soft min via log-sum-exp,
        sort_reg: regularization method to use in soft sort; "l2" or "kl" (None for hard sort).
        num_steps: number of subgradient steps.
        log_every: log progress every this many steps (<= 0 disables).

    Returns:
        best_w: best direction found, normalized to unit norm (l2 or l-infinity)
        best_obj: min objective value achieved at best_w.
        best_perm: permutation sorting E @ best_w non-decreasingly.
        best_sorted_proj: sorted projections at best_w.
    """
    if norm not in ("l2", "linf"):
        raise ValueError(f"norm must be 'l2' or 'linf', got {norm!r}.")
    
    logging.info(
        f"Running PGM for {num_steps} iterations, with normalize={normalize}, "
        f"norm={norm}, sort_epsilon={sort_epsilon}, sort_reg={sort_reg}, min_epsilon={min_epsilon}"
    )

    if E.dtype != torch.float64:
        logging.warning("Converting E to float64 precision")
        E = E.double()

    def _soft_min(gaps: Tensor) -> Tensor:
        return -min_epsilon * torch.logsumexp(-gaps / min_epsilon, dim=0)

    hard_sort = sort_epsilon == 0
    hard_min = min_epsilon == 0

    def _obj_and_supergrad(E: Tensor, w: Tensor) -> Tuple[float, float, Tensor, Tensor, Tensor]:
        # evaluate objective and a supergradient at w
        # Match float64 precision used in _randomly_permute_embeddings
        # TODO: use _projections_min_gap instead of rewriting things here (can make the function output gaps or min_indices)
        if hard_sort and hard_min:
            proj = E @ w  
            perm = proj.argsort(stable=True)
            sorted_proj = proj[perm]
            gaps = sorted_proj.diff()  
            min_gap = gaps.min()
            min_indices = (gaps == min_gap).nonzero(as_tuple=True)[0]
            obj_value = min_gap.item()
            supergrad = (E[perm[min_indices + 1]] - E[perm[min_indices]]).mean(dim=0)
            soft_obj_value = obj_value
        else:
            with torch.enable_grad():
                w_var = w.detach().requires_grad_(True)
                proj = E @ w_var
                if hard_sort:
                    perm_var = proj.argsort(stable=True)
                    sorted_proj_var = proj[perm_var]
                    perm = perm_var.detach()
                    sorted_proj = sorted_proj_var.detach()
                else:
                    if sort_reg is None:
                        raise ValueError(f"sort_reg must be set when sort_epsilon > 0, got {sort_reg}")
                    sorted_proj_var = soft_sort(
                        proj.unsqueeze(0),
                        direction="ASCENDING",
                        regularization_strength=sort_epsilon,
                        regularization=sort_reg,
                    ).squeeze(0)
                    proj = proj.detach()
                    perm = proj.argsort(stable=True)
                    sorted_proj = proj[perm]

                gaps = sorted_proj_var.diff()
                obj = gaps.min() if hard_min else _soft_min(gaps)
                supergrad = torch.autograd.grad(obj, w_var)[0]

            soft_obj_value = obj.item()
            obj_value = sorted_proj.diff().min().item()

        return obj_value, soft_obj_value, supergrad, perm, sorted_proj

    w = w_init.to(device=E.device, dtype=torch.float64)

    best_w = w.clone()
    best_obj = -inf
    best_perm = None
    best_sorted_proj = None
    D = 2 if norm == "l2" else 2 * sqrt(E.shape[1]) # domain diameter
    if normalize:
        L = 1.0
    else:
        if hard_sort and hard_min:
            L = _embeddings_max_dist(E) # max_{i < j} ||E_j - E_i||_2
        else:
            raise ValueError(f"Not implemented yet")

    for iter in range(num_steps):
        obj_value, soft_obj_value, supergrad, perm, sorted_proj = _obj_and_supergrad(E, w)
        if obj_value > best_obj:
            best_obj, best_w, best_perm, best_sorted_proj = obj_value, w.clone(), perm, sorted_proj.clone()

        supergrad_norm = supergrad.norm()
        if log_every > 0 and (iter % log_every == 0 or iter == num_steps - 1):
            logging.info(
                f"PGM dual cone step {iter}: obj value = {obj_value:.6g}, soft obj value = {soft_obj_value:.6g}, best obj value = {best_obj:.6g}, "
                f"||supergrad|| = {supergrad_norm.item():.6g}"
            )
        if supergrad_norm < 1e-12:
            logging.info(f"PGM dual cone: supergradient norm < 1e-12 at step {iter}, stopping.")
            break


        eta = D / (L * sqrt(iter + 1))
        w = w + eta * ((supergrad / supergrad_norm) if normalize else supergrad)
        if norm == "l2":
            w /= w.norm()  
        else:
            w = torch.clamp(w, min=-1.0, max=1.0)

    logging.info(f"PGM finished after {iter + 1} steps with best obj value {best_obj:.6g}.")
    return best_w, best_obj, best_perm, best_sorted_proj


def _find_embeddings_dual_cone_w(
    model: PreTrainedModel,
    valid_token_ids: Tensor,
    init_w: Literal["random", "pca"] = "random",
    solver: Literal["lp", "pgm"] | None = None,
    solver_config: dict = {},
    seed: int = 0,
    save_file: str | None = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Find a vector w in R^d in the interior of the dual cone of differences of
    adjacent embedding vectors after permuting them, i.e., 
    
    w^\top(E_{\sigma_{i+1}} - E_{\sigma_i}) > 0 for all i in V, 
    where E is the model embedding matrix (restricted to valid_token_ids).

    If solver is None:
        Sample a random unit vector w such that the projections of the rows of E onto w are distinct, 
        then sort the rows in non-decreasing order of their projections onto w.

    If solver == "pgm":
        Find unit vector w and permutation sigma that maximize the min gap between adjacent embedding vectors 
        permuted by sigma, i.e.,

        \max_{\| w\| <= 1} \max_{\sigma} \min_{i \in [k-1]} w^\top(E_{\sigma_{i+1}} - E_{\sigma_i})
        = \max_{\| w\| <= 1} \min_{i \in [k-1]} (B ~\mathrm{sort}(E w))_i, 
        where B is the matrix with rows e_{i+1} - e_{i}.

        Use PGM initialized with the random w.

    If solver == "lp":
        Find w that maximizes the min gap between adjacent embedding vectors sorted based on random w.
        Let U be the matrix with rows {E_{\sigma_{i+1}} - E_{\sigma_i} : i in V}, where \sigma
        is the fixed permutation corresponding to the random w.
        Solve the LP problem:

        max_{t >= 0, w in [-1, 1]^d} t  subject to  U w >= t

    If save_file is set, cache results to that path.
    """

    embedding_matrix = _valid_embeddings(model, valid_token_ids)
    # float64 precision needed in _randomly_permute_embeddings and likely needed in _solve_dual_cone_pgm too (TODO: check)
    embedding_matrix = embedding_matrix.double()
    k = embedding_matrix.shape[0]
    # TODO:remove when done debugging
    # min_dist = _embeddings_min_dist(embedding_matrix) # 0.0166836 for Llama-3.2-1B-Instruct 
    # n_unique_rows = np.unique(embedding_matrix, axis=0).shape[0]
    # assert n_unique_rows == k, (f"Embedding matrix has {k - n_unique_rows} duplicate row(s).")

    if init_w == "random":
        torch.manual_seed(seed) # reset seed to ensure reproducibility of resulting w, perm 
        #TODO: run solver with different random initializations and pick the best final solution
        w, perm, min_gap, sorted_embedding_projections = _randomly_permute_embeddings(embedding_matrix)
    elif init_w == "pca":
        w, perm, min_gap, sorted_embedding_projections = _embeddings_pca(embedding_matrix)
    else:
        raise ValueError(f"Invalid init_w: {init_w}")

    if solver == "lp":
        # LP took > 3hrs to solve after permuting embeddings according to random w.
        # TODO: try initializing lp solver with random w. Also, try to solve problem with SVM instead of LP
        # linprog solver requires numpy inputs on CPU
        embedding_matrix = embedding_matrix[perm].to("cpu").numpy()
        neg_U = (embedding_matrix[:-1] - embedding_matrix[1:])
        del embedding_matrix
        w_opt, t_opt, lp_result = _solve_dual_cone_lp(neg_U)
        #TODO: update perm to the sorted order of projections onto w_opt (since this is optimal perm for fixed w_opt)
        sorted_embedding_projections = None # no need to compute here they will be computed in dsm 

    elif solver == "pgm":
        if solver_config["sort_epsilon"] > 0: # fast_soft_sort requires inputs to be on CPU (will convert to numpy internally)
            embedding_matrix = embedding_matrix.to("cpu")
            w = w.to("cpu")
        w_opt, t_opt, perm, sorted_embedding_projections = _solve_dual_cone_pgm(embedding_matrix, w, **solver_config)

    else:
        t_opt = min_gap
        w_opt = w

    assert t_opt >= 0.0, "t* should be non-negative."
    assert t_opt >= min_gap, "t* should be greater than or equal to the min gap achieved by random w."
    if t_opt == 0.0:
        raise ValueError("Did not find w in the interior of the dual cone, t* = 0.0.")
    logging.info(f"Found w in the interior of the dual cone with t* = {t_opt:.6g}.")

    perm = perm.to(model.device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(k, device=model.device)
    # normalize by t_opt. We can recover t_opt from 1/||w_opt||_\infty if solver=="lp" or 
    # 1/||w_opt||_2 otherwise
    w_opt = (w_opt / t_opt).to(model.device)
    if sorted_embedding_projections is not None:  
        sorted_embedding_projections = (sorted_embedding_projections / t_opt).to(model.device)

    if save_file is not None:
        os.makedirs(os.path.dirname(f"{save_file}"), exist_ok=True)
        torch.save(
            {"w_opt_scaled": w_opt, "t_opt": t_opt, "perm": perm, "inv_perm": inv_perm, "min_gap": min_gap, "lp_result": lp_result if solver == "lp" else None},
            save_file,
        ) # not storing premuted projections as it's cheaper to just recompute them

    return w_opt, perm, inv_perm, sorted_embedding_projections
