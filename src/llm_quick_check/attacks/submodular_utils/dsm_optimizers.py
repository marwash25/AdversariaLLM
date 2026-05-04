"""
Difference of submodular minimization optimizers
"""
import torch
from typing import Callable, List, Optional, Tuple, Literal
from torch import Tensor
from math import log2, inf, sqrt
from tqdm import trange
import sys
import logging
import time

from .setfn_reductions import SetFnReduction


# TODO: might be good to actually define a PGM class with step method to have standardized interface for different optimization methods
# for now let's implement it as a standalone function similar to Matlab code
# Note that this is will be mostly used for non-submodular functions. In DCA, we will use MNP as inner solver.
# TODO: if used for submodular functions, add ground set trimming and set L to upper bound sqrt(sum_i F_set(i)^2) if not provided
def pgm_lovasz(F_set_batch: SetFnReduction, x_init: Tensor, num_steps: int, L: float | str, gap_tol: Optional[float] = None):
    """Apply projected subgradient method (PGM) to the problem min_{X in [0,1]^n x b} f_L(X)
    where f_L is the Lovasz extension of a set function reduction F_set: 2^([n] x [b]) -> R
    of a discrete function F: V^n -> R.

    Args:
        F_set_batch: SetFnReduction instance.
        x_init: Initial solution in V^n. Tensor of type long and shape (n,) or (1, n).
        num_steps: Number of iterations.
        L: Positive float or string. Lipschitz constant of the Lovasz extension f_L. 
        If F_set is monotone, set to F_set(V), which holds even if F_set is not submodular.
        If F is submodular set to 3 max_S |F_set(S)| if known, otherwise set to 'singletons' to use sqrt(sum_i F_set(i)^2) bound.
        If F is neither, set to 'normalize' to normalize the subgradient or 'singletons' as heuristic.
        gap_tol: Stop when duality gap is less than gap_tol. 
        Use only if F_set is submodular, otherwise duality gap is not guaranteed to converge.

    Returns:
        discrete_obj_values: List of T floats, discrete objective values F(x^t) for each iteration t.
        T is number of iterations ran (includes initial iter, can be less than num_steps + 1 if converged before)
        continuous_obj_values: List of floats, continuous objective values f_L(X^t) for each iteration t.
        duality_gaps: List of floats, duality gaps for each iteration. Not true duality gaps if F_set is not submodular.
        discrete_sols: Tensor of shape (T, n) and type long, discrete solutions x^t in V^n for each iteration t.
        best_sol_idx: int, index of best discrete solution in discrete_sols.
        times: List of floats, times for each iteration.
        flops: List of ints, flops for each iteration. 
    """
    # TODO: add option to only store solutions that improve best objective. 
    # Keeping old doc string to reuse in this case:
    # discrete_obj_values: List of best discrete objective values min_{i <= t} F(x^i) for each iteration t.
    # continuous_obj_values: List of continuous objective values f_L(X^i*(b)) corresponding to best discrete solution 
    # x^i*(b) = argmin_{i <= t} F(x^i) for each iteration t.
    # discrete_sols: List of best discrete solution x^i*(b) for each iteration b.

    logging.info(f"Running PGM for {num_steps} iterations, L set to {L}, and gap tolerance to {gap_tol}")
    time_start = time.time() # include initialization time in iter 0 time

    if x_init.dim() == 1:
        x_init = x_init.unsqueeze(0)
    # map x_init to X in [0,1]^n x b
    X = F_set_batch.ints2binary(x_init)[0].to(dtype=torch.float)

    n, b = X.shape
    D = sqrt(n*b) # domain diameter
    # if gap_tol is not None:
    dual_avg = torch.zeros_like(X)

    flops_L = 0
    normalize = False
    if isinstance(L, str):
        if L == "singletons":
            # Set L to sqrt(sum_i F_set({i})^2) 
            L, flops_L = F_set_batch.singletons_L_bound()
            L = max(L, 1e-12) # L < 1e-12 shouldn't happen unless F = 0 but just in case
        elif L == "normalize":
            normalize = True
            L = 1.0
        else:
            raise ValueError("If L is a string, it must be either 'singletons' or 'normalize'.")

    assert L > 0, "Lipschitz constant L must be positive"

    discrete_obj_values = [0.0 for _ in range(num_steps+1)]
    continuous_obj_values = [0.0 for _ in range(num_steps+1)]
    duality_gaps = [0.0 for _ in range(num_steps+1)] # if gap_tol is not None else None
    discrete_sols = torch.empty((num_steps+1, n), dtype=torch.long, device=X.device)
    times = [0.0 for _ in range(num_steps+1)]
    flops = [0 for _ in range(num_steps+1)]
    best_discrete_obj = inf
    best_sol_idx = -1
    # best_continuous_obj = inf

    for iter in (pbar := trange(num_steps+1, file=sys.stdout)):
        subgradient, Fvalues, x_chain, flops_subgrad = F_set_batch.subgradient_lovasz_extension(X)
        F_round, x_round = F_set_batch.round_lovasz_extension(Fvalues=Fvalues, x_chain=x_chain)  
        cont_value = F_set_batch.lovasz_extension(X, subgradient)
        
        if F_round < best_discrete_obj: 
            best_discrete_obj = F_round
            best_sol_idx = iter
            # x_best = x_round
            # best_continuous_obj = cont_value

        discrete_obj_values[iter] = F_round # best_discrete_obj
        continuous_obj_values[iter] = cont_value # best_continuous_obj
        discrete_sols[iter] = x_round # x_best

        # if gap_tol is not None: # not used if gap_tol is None but we can still compute it since it's relatively cheap
        dual_avg = (dual_avg * iter + subgradient) / (iter + 1)
        # see Bach_learning_new Section 10.8 Proposition 10.4  #TODO: add proper reference here
        dual_value = torch.clamp(dual_avg, max=0).sum().item()
        duality_gap = best_discrete_obj - dual_value
        duality_gaps[iter] = duality_gap
        
        # PGM update is included in next iteration time
        times[iter] = time.time() - time_start
        
         # TODO: add flops for prefill to initial step flops as done in GCG if we do prefill
        flops[iter] = flops_subgrad # only subgradient involves function evaluations 
        if iter == 0: 
            flops[iter] += flops_L

        pbar.set_postfix({"Discrete obj value": discrete_obj_values[iter], "Continuous obj value": continuous_obj_values[iter], "Duality gap": duality_gaps[iter]})
        if gap_tol is not None and duality_gaps[iter] <= gap_tol:
                logging.info(f"Duality gap {duality_gap:.4f} <= tolerance {gap_tol:.4f} reached after {iter} iterations, stopping.")
                break
        
        time_start = time.time()
        if iter < num_steps: # no need to update in last iteration
            if normalize:
                subgradient_norm = torch.linalg.vector_norm(subgradient.float(), ord=2).item()
                if subgradient_norm < 1e-12:
                    logging.info(f"Subgradient norm {subgradient_norm:.4f} < 1e-12.")
                    #TODO: if F is submodular we should stop. Otherwise still stop?
                    if gap_tol is not None:
                        break
                subgradient /= max(subgradient_norm, 1e-12)
            
            eta = D / (L * sqrt(iter + 1))
            # TODO: add Polyak step (to use only in submodular case - again not sure it works for non-submodular)
            X = X - eta * subgradient 
            X = torch.clamp(X, min=0, max=1)

    continuous_obj_values = continuous_obj_values[:iter+1]
    discrete_obj_values = discrete_obj_values[:iter+1]
    duality_gaps = duality_gaps[:iter+1]
    discrete_sols = discrete_sols[:iter+1, :]
    times = times[:iter+1]
    flops = flops[:iter+1]
    
    return best_sol_idx, discrete_obj_values, continuous_obj_values, duality_gaps, discrete_sols, times, flops


# def dca_dsm(F_set_batch: SetFnReduction, x_init: Tensor, num_outer_steps: int, num_inner_steps: int, 
# inner_solver: Literal["pgm", "mnp"], outer_tol: Optional[float] = 1e-5, inner_gap_tol: Optional[float] = 1e-4, 
# tie_break: Literal["random"] = None, L_G: float | str = "singletons"):
#     """
#     Implement the difference of convex algorithm (DCA) variant from El Halabi et al. 2023 (Algorithm 2) 
#     for the difference of submodular minimization (DSM) problem min_{S} F_set(S):= G_set(S) - H_set(S), which
#     applies DCA to the equivalent continuous problem min_{X in [0,1]^n x b} f_L(X) := g_L(X) - h_L(X).
#     Here F_set, G_set, H_set: 2^([n] x [b]) -> R are the set function reductions of the discrete functions 
#     F, G, H: V^n -> R, and f_L, g_L, h_L are their Lovasz extensions.

#     @InProceedings{elhalabi2023dsm,
#       title={Difference of Submodular Minimization via DC Programming}, 
#       author={Marwa El Halabi and George Orfanides and Tim Hoheisel},
#       booktitle = {Proceedings of the 40th International Conference on Machine Learning},
#       year={2023},
#     }

#     Args:
#         F_set_batch: SetFnReduction instance. 
#         x_init: Initial solution in V^n. Tensor of type long and shape (n,) or (1, n).
    
#     Returns:
#     """
#     # Decided to implement DCA-Restart version for now since simpler and faster. 
#     # TODO: add DCA-LS version from our ContDSMin paper later since it can perform better in practice 
#     # when a good initialization is not provided.  
#     logging.info(f"Running DCA for {num_outer_steps} outer iterations and {num_inner_steps} inner iterations")
#     time_start = time.time() # include initialization time in iter 0 time

#     if x_init.dim() == 1:
#         x_init = x_init.unsqueeze(0)
#     # map x_init to X in [0,1]^n x b
#     X = F_set_batch.ints2binary(x_init)[0].to(dtype=torch.float)
#     n, b = X.shape

#     flops_L_G = 0
#     if isinstance(L_G, str):
#         if L_G == "singletons":
#             L_G, flops_L_G = G_set_batch.singletons_L_bound()
#             L_G = max(L_G, 1e-12) # L_G < 1e-12 shouldn't happen unless G = 0 but just in case
#         else:
#             raise ValueError("If L_G is a string, it must be 'singletons'.")
#     assert L_G > 0, "Lipschitz constant L_G must be positive"

#     for iter in (pbar := trange(num_outer_steps+1, file=sys.stdout)):
#         subgrad_G, Gvalues, x_chain, flops_subgrad = G_set_batch.subgradient_lovasz_extension(X) # subgrad_G is (n, b)

#         if inner_solver == "pgm":
#             L = L_G + torch.linalg.vector_norm(subgradient.float(), ord=2).item()
#             best_sol_idx, discrete_obj_values, continuous_obj_values, duality_gaps, discrete_sols, times, flops = \
#                 pgm_lovasz(G_set_batch, X, num_inner_steps, L, gap_tol=inner_gap_tol)
#         elif inner_solver == "mnp":
#             # TODO: implement MNP 
#             pass
#         else:
#             raise ValueError(f"Inner solver {inner_solver} not supported. Must be 'pgm' or 'mnp'.")
#         #pbar.set_postfix({"Discrete obj value": discrete_obj_values[iter], "Continuous obj value": continuous_obj_values[iter], "Duality gap": duality_gaps[iter]})

#     return
