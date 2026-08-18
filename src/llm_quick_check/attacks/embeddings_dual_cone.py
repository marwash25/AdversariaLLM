"""Find a direction w in the dual cone of adjacent embedding differences."""
from math import sqrt, inf
from pathlib import Path
import time
import logging
from typing import Tuple, Any, Literal, Mapping, Optional
from tqdm import trange
import sys
import torch
from torch import Tensor
from fast_soft_sort.pytorch_ops import soft_sort
from transformers import PreTrainedModel


#TODO: remove if not used
def _max_min_gap_coordinate_permutation(embedding_matrix: Tensor) -> Tuple[Tensor, float]:
    r"""Sort rows of embedding matrix based on their jth coordinate in non-decreasing order, 
    for j \in [d] with the largest minimum gap between adjacent rows, i.e.,  
    \max_{j \in [d]} \min_{i \in [k-1]} (E_{\sigma^j_{i+1}, j} - E_{\sigma^j_i, j}), 
    where \sigma^j is such that E_{\sigma_k, j} \geq \ldots \geq E_{\sigma_0, j}.
    Return the corresponding permutation and minimum gap.
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


# TODO: refactor to have one common PGM solver used both here and in pgm_lovasz in dsm_optimizers.py
def _solve_dual_cone_pgm(
    E: Tensor,
    w_init: Tensor,
    norm: Literal["2", "inf"] = "2",
    normalize: bool = False,
    sort_epsilon: float = 1.0,
    min_epsilon: float = 1.0,
    sort_reg: Literal["l2", "kl"] | None = None,
    num_steps: int = 2000,
    log_every: int = 200,
) -> Tuple[Tensor, float, Tensor, Tensor]:
    """Solve the dual-cone problem by projected subgradient method (PGM):
       max_{\| w\| <= 1} \min_{i \in [k-1]} (B sort(E w))_i, 
    where B is the matrix with rows e_{i+1} - e_{i}, and sort operation
    is applied in non-decreasing order.
    
    Args:
        E: 2D tensor.
        w_init: 1D tensor, initial direction. 
        norm: norm used to constrain w, "l2" or "linf".
        normalize: if True, normalize gradients, otherwise use L (only supported for hard sort and hard min for now)
        sort_epsilon: if 0, use hard sort; if > 0, use soft sort (via fast-soft-sort).
        min_epsilon: if 0, use hard min; if > 0, use soft min via log-sum-exp,
        sort_reg: regularization method to use in soft sort; "l2" or "kl" (None for hard sort).
        num_steps: number of subgradient steps.
        log_every: log progress every this many steps (<= 0 disables).

    Returns:
        best_w: best direction found, normalized to unit norm (l2 or l-infinity)
        best_obj: min objective value achieved at best_w.
        best_perm: permutation sorting E best_w non-decreasingly.
        best_sorted_proj: sorted projections at best_w.
    """
    if norm not in ("2", "inf"):
        raise ValueError(f"norm must be '2' or 'inf', got {norm!r}.")
    
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
    w = w / w.norm(p=float(norm))

    best_w = w.clone()
    best_obj = -inf
    best_perm = None
    best_sorted_proj = None
    D = 2 if norm == "2" else 2 * sqrt(E.shape[1]) # domain diameter
    if normalize:
        L = 1.0
    else:
        if hard_sort and hard_min:
            L = _embeddings_max_dist(E) # max_{i < j} ||E_j - E_i||_2
        else:
            raise ValueError(f"Not implemented yet")

    for iter in (pbar := trange(num_steps + 1, file=sys.stdout)):
        obj_value, soft_obj_value, supergrad, perm, sorted_proj = _obj_and_supergrad(E, w)
        if obj_value > best_obj:
            best_obj, best_w, best_perm, best_sorted_proj = obj_value, w.clone(), perm, sorted_proj.clone()

        supergrad_norm = supergrad.norm()
        if log_every > 0 and (iter % log_every == 0 or iter == num_steps):
            pbar.set_postfix(
            {"obj value": obj_value, "soft obj value": soft_obj_value, "best obj value": best_obj, "||supergrad||": supergrad_norm.item()}
            )
        if supergrad_norm < 1e-12:
            pbar.write(f"PGM dual cone: supergradient norm < 1e-12 at step {iter}, stopping.")
            break

        if iter < num_steps: # no update in last iteration
            eta = D / (L * sqrt(iter + 1))
            w = w + eta * ((supergrad / supergrad_norm) if normalize else supergrad)
            if norm == "2":
                w /= w.norm()
            else:
                w = torch.clamp(w, min=-1.0, max=1.0)

    logging.info(f"PGM finished after {iter} steps with best obj value {best_obj:.6g}.")
    return best_w, best_obj, best_perm, best_sorted_proj


def _affine_min_norm_point(A: Tensor) -> Tensor:
    r"""Find the minimum-norm point in the affine hull of the
    columns of A (assumed to be affinely independent): 
    min_{x \in aff(A)} 0.5 ||x||^2_2 = min_{alpha : 1^T alpha = 1} 0.5 || A alpha ||^2_2.
    Return the barycentric coordinates alpha of the MNP.

    Implementation matches the one from the MNP algorithm in Francis Bach's 
    Matlab Submodular package (version 2.0), https://www.di.ens.fr/~fbach/submodular/. 
    """
    r = A.shape[1]
    c = A.norm()**2 / r
    # add a constant c to make M positive definite (augmented vectors [c; a_i] are linearly independent for any c > 0)
    # add a small ridge to avoid numerical issues
    M = A.T @ A + c + 1e-12 * torch.eye(r, dtype=A.dtype, device=A.device)

    try:
        # solve M v = 1 using Cholesky decomposition
        ones = torch.ones(r, 1, dtype=A.dtype, device=A.device)
        R = torch.linalg.cholesky(M)
        v = torch.cholesky_solve(ones, R).squeeze(1)
        # v = torch.linalg.solve(M, ones)
    except RuntimeError as e:
        logging.warning(f"Cholesky decomposition failed: {e}")
        return None
        
    return v / v.sum()

# TODO: when we want to use this for DCA inner problem, add option to restart from a point in conv(A)
def _min_norm_point(
    U: Tensor,
    w_init: Tensor | None = None,
    num_major_cycles: int = 1000,
    tol: float = 1e-6,
    log_every: int = 1,
) -> Tuple[Tensor, float, int, int]:
    r"""Solve the minimum-norm-point (MNP) problem by Wolfe's MNP algorithm:
    min_{x \in conv(u_i: u_i row of U)} 0.5 ||x||_2^2 = min_{lbd in simplex} 0.5 || U^T lbd ||_2^2,

    Equivalent to solving the dual problem: max_{w} min_{i} <u_i, w> - 0.5 ||w||_2^2 (w^* = x^*).

    Implementation based on Algorithm 1 in:
    @article{chakrabarty2014provable,
        title={Provable submodular minimization using Wolfe's algorithm},
        author={Chakrabarty, Deeparnab and Jain, Prateek and Kothari, Pravesh},
        journal={Advances in Neural Information Processing Systems},
        volume={27},
        year={2014}
    }

    Args:
        U: 2D tensor
        w_init: optional direction used to pick the initial vertex (argmin_i <u_i, w_init>). 
        num_major_cycles: maximum number of major cycles (atom insertions).
        tol: stop when the relative duality gap (||x||_2^2 - min_i <u_i, x>) / ||x||_2^2 <= tol.

    Returns:
        x: the minimum-norm point (d-vector), equal to U^T lbd*.
        n_active: number of atoms with positive weight at the solution.
        n_major: number of major cycles performed.
    """
    device, dtype = U.device, U.dtype

    def min_gaps(x: Tensor) -> Tuple[Tensor, Tensor]: # not using _projections_min_gap because we don't need to sort
        gaps = U @ x 
        min_gap, min_index = gaps.min(dim=0)
        return min_gap, min_index.item()

    init_index = 0 if w_init is None else min_gaps(w_init)[1] 
    active_indices = [init_index]
    n_active = 1
    A = U[active_indices].T # matrix with active atoms as columns
    lbd = torch.ones(1, dtype=dtype, device=device)
    x = A[:, 0].clone() 
    d = x.shape[0]

    for major_iter in (pbar := trange(num_major_cycles+1, file=sys.stdout)): # major cycle (last iter is just for logging)
        min_gap, min_index = min_gaps(x) # LMO: argmin_i <u_i, x> 
        x_norm_squared = torch.dot(x, x)

        duality_gap = x_norm_squared - min_gap
        if log_every > 0 and (major_iter % log_every == 0 or major_iter == num_major_cycles):
            relative_duality_gap = (0.0 if x_norm_squared.item() == 0.0 else (duality_gap / x_norm_squared).item())
            pbar.set_postfix(
                {"||x||_2": torch.sqrt(x_norm_squared).item(), "min gap": min_gap.item(), "relative duality gap": relative_duality_gap, "|active indices|": n_active}
            )

        if duality_gap <= tol * x_norm_squared: 
            logging.info(f"MNP converged after {major_iter + 1} major cycles with ||x||_2 = {torch.sqrt(x_norm_squared).item():.6g}, stopping.")
            break
        
        if min_index in active_indices:
        # min_index should not be in active_indices: If x = argmin_{z in aff(A)} ||z||_2 (holds up to numerical errors throughout the algorithm)
        # any point q in aff(A) satisfy q^Tx = ||x||_2^2 so termination condition above should be met but might due to numerical errors.
            logging.warning(f"MNP major cycle {major_iter}: new atom {min_index} is already in the active set, MNP should have terminated. "
                            f"Duality gap: {x_norm_squared - min_gap:.6g}, ||x||_2: {torch.sqrt(x_norm_squared).item():.6g}, # of active atoms: {n_active}, stopping.")
            break

        if n_active > d:
            logging.warning(f"MNP major cycle {major_iter}: # of active atoms {n_active} > d = {d}, affine minimizer is 0, and MNP should have either terminated or removed atoms from A in minor cycle. "
            f"Duality gap: {x_norm_squared - min_gap:.6g}, ||x||_2: {torch.sqrt(x_norm_squared).item():.6g}, stopping.")
            break

        if major_iter < num_major_cycles: # no update in last major cycle
            active_indices.append(min_index)
            n_active += 1
            A = torch.cat([A, U[min_index].unsqueeze(1)], dim=1)
            lbd = torch.cat([lbd, torch.zeros(1, dtype=dtype, device=device)])

            minor_iter = -1
            while True: # minor cycle (will run at most |active| times)
                assert minor_iter < 2*d, f"MNP minor cycle ran more than 2*d = {2*d} times. It should run at most |active| <= d+1 = {d+1} times."
                minor_iter += 1
                alpha = _affine_min_norm_point(A)
                if alpha is None:
                    logging.warning(f"MNP major cycle {major_iter}: Cholesky decomposition in affine minimizer failed, stopping.")
                    break 
                if (alpha > 1e-12).all(): # using 1e-12 instead of 0 to avoid numerical issues
                    lbd = alpha
                    x = A @ lbd
                    break
                # update x to the intersection of the boundary of conv(A) and the segment joining the affine solution y = A @ alpha and previous x. 
                # move toward y until an atom weight lbd_i hits zero (leaves conv(A))
                diff = alpha - lbd
                blocking = diff < 0 # not empty since lbd > 1e-12 and there exists alpha_i < 1e-12
                # theta = min(1, min_{alpha_i < lbd_i} lbd_i / (lbd_i - alpha_i))
                # which is equivalent to taking min over alpha_i < 0 if any, otherwise theta = 1.
                theta = min((-lbd[blocking] / diff[blocking]).min().item(), 1.0)
                lbd = lbd + theta * diff
                keep = lbd > 1e-12 
                active_indices = [active_indices[i] for i in range(n_active) if keep[i]]
                assert len(active_indices) < n_active, "At least one atom should be removed in each minor cycle."
                n_active = len(active_indices)
                A, lbd = A[:, keep], lbd[keep]
                lbd = lbd / lbd.sum()
                x = A @ lbd

            if log_every > 0 and (major_iter % log_every == 0 or major_iter == num_major_cycles - 1):
                pbar.set_postfix({"minor steps": minor_iter + 1})            

    return x, duality_gap.item(), n_active, major_iter + 1
    

def _solve_dual_cone_am(
    E: Tensor,
    w_init: Tensor,
    num_outer_steps: int = 100,
    num_inner_steps: int = 5000,
    outer_tol: float = 1e-6,
    inner_tol: float = 1e-6,
    log_every: int = 1,
) -> Tuple[Tensor, float, Tensor, Tensor]:
    r"""Solve the dual-cone problem by alternating maximization:
       max_{||w||_2 <= 1} max_{sigma} min_{i in [k-1]} w^T (E_{sigma_{i+1}} - E_{sigma_i}).

    In each outer step:
      1. fix w and update sigma to the non-decreasing order of E w (optimal sigma for current w);
      2. fix sigma and solve the concave maximization inner problem
             max_{||w||_2 <= 1} min_i <u_i, w>,   u_i = E_{sigma_{i+1}} - E_{sigma_i}
         by solving its dual min_{lbd in simplex} || U^T lbd ||_2, where U is the matrix with rows u_i,
        using MNP algorithm. Update w = U^T lbd^* / || U^T lbd^* ||_2.
    Objective should monotonically increase up to accuracy of inner problem. 

    Args:
        E: 2D tensor.
        w_init: 1D tensor, initial direction with non-zero minimum gap. 
        num_outer_steps: maximum number of sort/inner-solve alternations.
        num_inner_steps: maximum major cycles per inner min-norm-point solve.
        outer_tol: relative objective value change tolerance for the outer solve.
        inner_tol: relative duality-gap tolerance for the MNP problem.
        log_every: log progress every this many outer steps (<= 0 disables).

    Returns:
        best_w: best unit-norm direction found.
        best_obj: min objective value achieved at best_w.
        best_perm: permutation sorting E best_w non-decreasingly.
        best_sorted_proj: sorted projections at best_w.
    """
    if E.dtype != torch.float64:
        logging.warning("Converting E to float64 precision")
        E = E.double()

    logging.info(
        f"Running alternating maximization for up to {num_outer_steps} outer steps, "
        f"with num_inner_steps={num_inner_steps}, inner_tol={inner_tol}."
    )

    w = w_init.to(device=E.device, dtype=torch.float64)
    w = w / w.norm()

    best_w = w.clone() # should be last iterate if inner problem is solved exactly
    best_obj = -inf  
    best_perm = None
    best_sorted_proj = None
    prev_obj_value = -inf

    for iter in (pbar := trange(num_outer_steps, file=sys.stdout)):
        obj_value, perm, sorted_proj = _projections_min_gap(E, w)
        if iter == 0:
            assert obj_value > 0, "w_init should have non-zero minimum gap."

        E_sorted = E[perm]
        U = E_sorted[1:] - E_sorted[:-1]
        # w is used to pick the initial vertex, we can't restart from w itself since U changes in each outer step
        x_mnp, mnp_gap, n_active, n_MNP_steps = _min_norm_point(
            U, w_init=w, num_major_cycles=num_inner_steps, tol=inner_tol
        )
        x_mnp_norm = x_mnp.norm()

        # TODO: add a check that obj_value increased up to accuracy achieved for inner problem. 
        if obj_value > best_obj:
            best_obj, best_w, best_perm, best_sorted_proj = (
                obj_value, w.clone(), perm, sorted_proj.clone()
            )

        if log_every > 0 and (iter % log_every == 0 or iter == num_outer_steps - 1):
            pbar.set_postfix(
                {"obj value": obj_value, "best obj": best_obj, "MNP norm": x_mnp_norm.item(), "MNP duality gap": mnp_gap, "|MNP active indices|": n_active, "MNP steps": n_MNP_steps}
            ) 

        if iter > 0 and (obj_value - prev_obj_value) <= outer_tol * prev_obj_value:
            # use relative tolerance since obj_value can be very small (e.g., 1e-12 at w_init)
            logging.info(f"Alternating maximization converged after {iter + 1} outer steps with best obj value {best_obj:.6g}, stopping.")
            break
        prev_obj_value = obj_value

        if x_mnp_norm <= 1e-13:
            logging.warning(
                f"Alternating dual cone: min-norm point ~ 0 at outer step {iter} "
                f"(origin in convex hull of u_i's); stopping."
                # should not happen if init min_gap > 1e-12 and inner problem solve up inner_tol
            )
            break # alternatively we can restart from a random w

        w =  x_mnp / x_mnp_norm # we should have (U @ w).min() = x_mnp_norm - mnp_gap / x_mnp_norm

    return best_w, best_obj, best_perm, best_sorted_proj


def _find_embeddings_dual_cone_w(
    model: PreTrainedModel,
    valid_token_ids: Tensor,
    init_w: Literal["random", "pca"] = "random",
    solver: Literal["pgm", "am"] | None = None,
    solver_config: dict = {},
    seed: int = 0,
    save_file: str | Path | None = None,
    fingerprint: Optional[Mapping[str, Any]] = None,
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

        Use PGM initialized with init_w.
    
    If solver == "am":
        Solve the same problem as "pgm" (l2 norm only) by alternating maximization
        initialized with init_w.

    If save_file is set, cache results and fingerprint to that path.
    """

    embedding_matrix = _valid_embeddings(model, valid_token_ids)
    # float64 precision needed since min gap can be very small (e.g., 1e-12)
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

    if solver == "pgm":
        if solver_config["sort_epsilon"] > 0: # fast_soft_sort requires inputs to be on CPU (will convert to numpy internally)
            embedding_matrix = embedding_matrix.to("cpu")
            w = w.to("cpu")
        w_opt, t_opt, perm, sorted_embedding_projections = _solve_dual_cone_pgm(embedding_matrix, w, **solver_config)

    elif solver == "am":
        w_opt, t_opt, perm, sorted_embedding_projections = _solve_dual_cone_am(embedding_matrix, w, **solver_config)

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
    # normalize by t_opt. We can recover t_opt from 1/||w_opt||_\infty if solver=="pgm" and norm=="inf" or 
    # 1/||w_opt||_2 otherwise
    w_opt = (w_opt / t_opt).to(model.device)
    if sorted_embedding_projections is not None:  
        sorted_embedding_projections = (sorted_embedding_projections / t_opt).to(model.device)

    if save_file is not None:
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "w_opt_scaled": w_opt,
                "t_opt": t_opt,
                "perm": perm,
                "inv_perm": inv_perm,
                "min_gap": min_gap,
                "fingerprint": fingerprint,
            },
            save_path,
        ) # not storing permuted projections as it's cheaper to just recompute them

    return w_opt, perm, inv_perm, sorted_embedding_projections
