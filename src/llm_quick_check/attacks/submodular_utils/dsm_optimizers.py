"""
Difference of submodular minimization optimizers
"""
import torch
from typing import Literal
from collections.abc import Callable
from copy import deepcopy
from torch import Tensor
from math import inf, sqrt
from tqdm import trange
import sys
import logging
import time

from .setfn_reductions import SetFnReduction
from .set_fn_instances import ModularSetFn
from .set_functions import LinearCombinationSetFn, SetFunction


def make_lattice_prefix_filter(
    filter_fn: Callable[[Tensor], list[int]] | None = None, filter_zero: bool = False
) -> Callable[[tuple[Tensor, Tensor], object | None], list[int]]:
    """Build a filter_prefixes callback using the cached lattice chain.

    filter_fn returns retained nonempty chain-row indices. The callback returns
    a list of ascending prefix indices, shifted by one, with zero prepended if allowed.
    """
    def filter_prefixes(order: tuple[Tensor, Tensor], auxiliary_input: object | None) -> list[int]:
        x_chain = auxiliary_input
        retain_idx = range(len(x_chain)) if filter_fn is None else filter_fn(x_chain)
        return ([] if filter_zero else [0]) + [i + 1 for i in sorted(retain_idx)]

    return filter_prefixes


# TODO: Both PGM and DCA essentially ignore filtering for the opt itself for now, except for storing filtered solutions at each iteration.
# This would change if we modify GCG loss to return larger values for unreachable solutions.

# TODO: might be good to actually define a PGM class with step method to have standardized interface for different optimization methods
# for now let's implement it as a standalone function similar to Matlab code
# Note that this is will be mostly used for non-submodular functions. In DCA, we will use MNP as inner solver.
# TODO: add ground set trimming for submodular F (add flag to enable/disable)
def pgm_lovasz_set(
    F_set: SetFunction,
    X_init: Tensor,
    num_steps: int,
    L: float | Literal["singletons", "normalize", "polyak"],
    tie_break: Literal["random"] | Tensor | None = None,
    gap_tol: float | None = None,
    filter_prefixes: Callable[[tuple[Tensor, Tensor], object | None], list[int]] | None = None,
):
    """Apply projected subgradient method (PGM) to the problem min_{X in [0,1]^n x b} f_L(X)
    where f_L is the Lovasz extension of a normalized set function F_set: 2^([n] x [b]) -> R.

    At each iteration, X is updated as X = proj_[0,1]^{n x b}(X - eta * s) with s a subgradient of f_L at X
    obtained by Edmonds' greedy algorithm, then rounded to a set S in [n] x [b]
    satisfying F_set(S) <= f_L(X).

    Args:
        F_set: SetFunction instance.
        X_init: Initial solution in [0,1]^(n x b), Tensor of shape (n, b).
        num_steps: Maximum number of optimization steps.
        L: Positive float or one of "singletons", "normalize", "polyak". Determines the step size eta.
        If a float, it is used as the Lipschitz constant of f_L, with eta = D / (L sqrt(t+1)) and D = sqrt(n b).
        If F_set is monotone, set to F_set(V), which holds even if F_set is not submodular.
        If F_set is submodular set to 3 max_S |F_set(S)| if known, or to 'singletons' to use sqrt(sum_i F_set(i)^2) bound,
        or to 'polyak' to use the Polyak step size eta = (f_L(X^t) - best dual value) / ||s||^2.
        If F_set is neither, set to 'normalize' to normalize the subgradient (L = ||s||) or 'singletons'/'polyak' as heuristics
        tie_break: Determines how ties in the sorting used by Edmonds' greedy algorithm are broken.
        If "random", a new random permutation is drawn at each iteration. If a Tensor of shape (n, b),
        it is used as a fixed tie-breaking order at every iteration. If None, ties are broken by the original order.
        gap_tol: Stop when duality gap is less than gap_tol.
        Use only if F_set is submodular, otherwise duality gap is not guaranteed to converge.
        filter_prefixes: Optional callback taking (order, auxiliary_input) and returning
        eligible prefix indices for filtered rounding.

    Returns:
        best_discrete_sol: Tuple (rows, cols) of long tensors, the unfiltered set with the lowest
        objective value over all iterations. None if no iteration was run.
        best_continuous_sol: Tensor of shape (n, b), the iterate X^t that produced best_discrete_sol.
        discrete_obj_values: List of T floats, discrete objective values F_set(S^t) for each iteration t.
        T is number of iterations ran (includes initial iter, can be less than num_steps + 1 if converged earlier).
        discrete_obj_values_filtered: same as discrete_obj_values but with filtered solutions if filtering is enabled.
        Entries are inf for iterations where no candidate solution was retained by the filter.
        continuous_obj_values: List of T floats, continuous objective values f_L(X^t) for each iteration t.
        duality_gaps: List of T floats, gap between the best discrete objective so far and the dual value of the
        averaged subgradient. True duality gaps only if F_set is submodular.
        discrete_sols_filtered: List of T sets as (rows, cols) tuples, filtered if filtering is enabled.
        Entries are None when no candidate solution was retained by the filter.
        times: List of T floats, wall clock time of each iteration (the update step is counted in the next iteration).
        flops: List of T ints, flops for each iteration.
        best_auxiliary_sol: Optional auxiliary output for best_discrete_sol.
        auxiliary_sols_filtered: Optional auxiliary outputs for discrete_sols_filtered, stacked along
        dimension 0 if all entries are tensors, otherwise a list of length T.
        This is a tensor of shape (T, n) containing cached lattice points x^t in V^n for SetFnReduction,
        or a list of None for functions without auxiliary output. Ignore entries with infinite filtered values.
    """
    # TODO: add option to only store solutions that improve best objective.
    # Keeping old doc string to reuse in this case:
    # discrete_obj_values: List of best discrete objective values min_{i <= t} F(x^i) for each iteration t.
    # continuous_obj_values: List of continuous objective values f_L(X^i*(b)) corresponding to best discrete solution
    # x^i*(b) = argmin_{i <= t} F(x^i) for each iteration t.
    # discrete_sols: List of best discrete solution x^i*(b) for each iteration b.

    logging.info(f"Running PGM for {num_steps} iterations, L set to {L}, and gap tolerance to {gap_tol}")
    time_start = time.time() # include initialization time in iter 0 time

    # Initialize X in [0,1]^(n x b)
    assert X_init.shape == (F_set.n, F_set.b), "X_init must have shape (n, b)"
    assert X_init.device == F_set.device, "X_init and F_set must use the same device"
    assert ((X_init >= 0) & (X_init <= 1)).all(), "X_init must have entries in [0, 1]"
    X = X_init

    n, b = X.shape
    D = sqrt(n*b) # domain diameter
    # if gap_tol is not None:
    dual_avg = torch.zeros_like(X)

    flops_L = 0
    normalize = False
    polyak = False
    if isinstance(L, str):
        if L == "singletons":
            # Set L to sqrt(sum_i F_set({i})^2)
            L, flops_L = F_set.singletons_L_bound()
            L = max(L, 1e-12) # L < 1e-12 shouldn't happen unless F = 0 but just in case
        elif L == "normalize":
            normalize = True
        elif L == "polyak":
            polyak = True
            max_dual_value = -inf
        else:
            raise ValueError("If L is a string, it must be 'singletons' or 'normalize' or 'polyak'.")
    else:
        assert L > 0, "Lipschitz constant L must be positive"

    if isinstance(tie_break, Tensor):
        assert tie_break.shape == (n, b), "Tie break tensor must be of shape (n, b)"
        tie_break = tie_break.to(device=X.device)

    discrete_obj_values = [0.0 for _ in range(num_steps+1)]
    discrete_obj_values_filtered = [0.0 for _ in range(num_steps+1)]
    continuous_obj_values = [0.0 for _ in range(num_steps+1)]
    duality_gaps = [0.0 for _ in range(num_steps+1)] # if gap_tol is not None else None
    discrete_sols_filtered = [] # needed when used as standalone solver
    auxiliary_sols_filtered = []
    best_auxiliary_sol = None
    times = [0.0 for _ in range(num_steps+1)]
    flops = [0 for _ in range(num_steps+1)]
    best_discrete_obj = inf
    best_continuous_sol = None # needed when used as inner solver for DCA
    best_discrete_sol = None # needed when used as inner solver for DCA
    # best_continuous_obj = inf

    for iter in (pbar := trange(num_steps+1, file=sys.stdout)):
        if isinstance(tie_break, Tensor):
            tie_breaker = tie_break
        elif tie_break == "random":
            tie_breaker = torch.randperm(n * b, device=X.device, dtype=torch.long).view(n, b)
        else:
            tie_breaker = None
        subgradient, order, Fvalues, flops_subgrad, auxiliary_data = F_set.subgradient_lovasz_extension(X, tie_breaker)
        (F_round, S_round, auxiliary_round), (F_round_filtered, S_round_filtered, auxiliary_round_filtered) = F_set.round_lovasz_extension(
            order=order, Fvalues=Fvalues, auxiliary_input=auxiliary_data,
            filter_prefixes=filter_prefixes,
        )
        cont_value = F_set.lovasz_extension(X, subgradient, Fvalues)

        if F_round < best_discrete_obj:
            best_discrete_obj = F_round
            best_discrete_sol = (S_round[0].clone(), S_round[1].clone())
            best_auxiliary_sol = auxiliary_round.clone() if isinstance(auxiliary_round, Tensor) else deepcopy(auxiliary_round)
            best_continuous_sol = X.clone()
            # best_continuous_obj = cont_value

        discrete_obj_values[iter] = F_round # best_discrete_obj
        discrete_obj_values_filtered[iter] = F_round_filtered
        continuous_obj_values[iter] = cont_value # best_continuous_obj
        discrete_sols_filtered.append(
            None if S_round_filtered is None else (S_round_filtered[0].clone(), S_round_filtered[1].clone())
        )
        auxiliary_sols_filtered.append(
            auxiliary_round_filtered.clone() if isinstance(auxiliary_round_filtered, Tensor) else deepcopy(auxiliary_round_filtered)
        )

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

        pbar.set_postfix({"Discrete obj value": discrete_obj_values[iter], "Discrete obj value filtered": discrete_obj_values_filtered[iter], "Continuous obj value": continuous_obj_values[iter], "Duality gap": duality_gaps[iter]})
        if gap_tol is not None and duality_gaps[iter] <= gap_tol:
            logging.info(f"Duality gap {duality_gap:.4f} <= tolerance {gap_tol:.4f} reached after {iter} iterations, stopping.")
            break

        time_start = time.time()
        if iter < num_steps: # no update in last iteration
            if polyak or normalize:
                subgradient_norm = subgradient.norm().item()
                if subgradient_norm < 1e-12:
                    logging.info(f"Subgradient norm {subgradient_norm:.4f} < 1e-12.")
                    #TODO: if F is submodular we should stop. Otherwise still stop?
                    if gap_tol is not None:
                        break
                if polyak:
                    max_dual_value = max(max_dual_value, dual_value)
                    eta = (continuous_obj_values[iter] - max_dual_value) / max(subgradient_norm, 1e-12)**2
                else:
                    eta = D / (max(subgradient_norm, 1e-12) * sqrt(iter + 1))
            else:
                eta = D / (L * sqrt(iter + 1))

            X = X - eta * subgradient
            X = torch.clamp(X, min=0, max=1)

    continuous_obj_values = continuous_obj_values[:iter+1]
    discrete_obj_values = discrete_obj_values[:iter+1]
    discrete_obj_values_filtered = discrete_obj_values_filtered[:iter+1]
    duality_gaps = duality_gaps[:iter+1]
    times = times[:iter+1]
    flops = flops[:iter+1]
    if all(isinstance(auxiliary, Tensor) for auxiliary in auxiliary_sols_filtered):
        auxiliary_sols_filtered = torch.stack(auxiliary_sols_filtered)

    return best_discrete_sol, best_continuous_sol, discrete_obj_values, discrete_obj_values_filtered, continuous_obj_values, duality_gaps, discrete_sols_filtered, times, flops, best_auxiliary_sol, auxiliary_sols_filtered


# TODO: add a generic version of DCA that accepts general set function 
def dca_dsm(
    F_set: SetFnReduction,
    G_set: SetFunction,
    H_set: SetFunction,
    x_init: Tensor,
    num_outer_steps: int,
    num_inner_steps: int,
    inner_solver: Literal["pgm", "mnp"],
    outer_tol: float = 1e-5,
    inner_gap_tol: float | None = 1e-5,
    tie_break: Literal["random"] | None = None,
    L_G: float | Literal["singletons"] = "singletons",
    filter_fn: Callable[[Tensor], list[int]] | None = None,
    filter_zero: bool = False,
):
    """
    Implement the difference of convex algorithm (DCA) variant from El Halabi et al. 2023 (Algorithm 2)
    for the difference of submodular minimization (DSM) problem min_{S} F_set(S):= G_set(S) - H_set(S), which
    applies DCA to the equivalent continuous problem min_{X in [0,1]^n x b} f_L(X) := g_L(X) - h_L(X).
    Here F_set, G_set, H_set: 2^([n] x [b]) -> R are set functions, and f_L, g_L, h_L are their
    Lovasz extensions. F_set is a reduction of a lattice function F: V^n -> R; G_set and H_set
    need not be lattice reductions.

    @InProceedings{elhalabi2023dsm,
      title={Difference of Submodular Minimization via DC Programming},
      author={Marwa El Halabi and George Orfanides and Tim Hoheisel},
      booktitle = {Proceedings of the 40th International Conference on Machine Learning},
      year={2023},
    }

    At each outer step, h_L is linearized at the current iterate X using a subgradient s of h_L, and the
    resulting convex upper bound f_upperbd(X) = g_L(X) - <s, X> on f_L is (approximately) minimized by the
    inner solver, warm started at X. At convergence, the algorithm restarts from the best neighbor of the
    current discrete solution, or stops if the current solution is already a local min.

    Args:
        F_set: SetFnReduction instance for F. 
        G_set: SetFunction instance for the submodular component G_set.
        H_set: SetFunction instance for the submodular component H_set.
        x_init: Initial solution in V^n. Tensor of type long and shape (n,) or (1, n).
        num_outer_steps: Maximum number of DCA (outer) iterations.
        num_inner_steps: Maximum number of iterations of the inner solver.
        inner_solver: "pgm" for pgm_lovasz_set, "mnp" (minimum norm point) is not implemented yet.
        outer_tol: Stop (or restart from the best neighbor) when f_L(X^t) - f_L(X^{t-1}) <= outer_tol.
        inner_gap_tol: Duality gap tolerance passed to the inner solver.
        tie_break: If "random", ties in the sorting used by Edmonds' greedy algorithm are broken by a random
        permutation drawn at each outer iteration. Otherwise ties are broken by the original order.
        L_G: Positive float or 'singletons'. Lipschitz constant of g_L, used to derive the Lipschitz constant
        L_G + ||s|| of the upper bound passed to PGM. 'singletons' uses the sqrt(sum_{i} G_set({i})^2) bound.
        filter_fn: Optional callback returning retained lattice-point indices.
        filter_zero: Whether to exclude the empty set from filtered rounding.

    Returns:
        discrete_obj_values: List of T floats, discrete objective values F(x^t) for each outer step t, where x^t
        is obtained by rounding the iterate X^t. T is the number of outer steps ran + 1 for initialization result
        (can be less than num_outer_steps + 1 if DCA converged to a local min earlier).
        discrete_obj_values_filtered: same as discrete_obj_values but with filtered solutions if filtering is enabled.
        Entries are inf for outer steps where no candidate solution was retained by the filter.
        continuous_obj_values: List of T floats, continuous objective values f_L(X^t) for each outer step t.
        discrete_sols_filtered: Tensor of shape (T, n) and type long, discrete solutions x^t in V^n for each outer
        step t. If filtering is enabled, these are the filtered solutions.
        times: List of T floats, wall clock time of each outer step (including its inner steps).
        flops: List of T ints, flops of each outer step (including its inner steps).
        inner_discrete_values: List of T lists of floats, discrete objective values of the inner solver
        for each outer step. Note these are values of the upper bound F_upperbd, not of F.
        inner_discrete_values_filtered: same as inner_discrete_values but with filtered solutions if filtering is enabled.
        inner_continuous_values: List of T lists of floats, continuous objective values of the inner solver
        for each outer step.
        inner_duality_gaps: List of T lists of floats, duality gaps of the inner solver for each outer step.
        inner_times: List of T lists of floats, times of the inner iterations of each outer step.
        inner_flops: List of T lists of ints, flops of the inner iterations of each outer step.
        All inner_*[0] are empty (initialization result, inner solver not run).
    """
    # Decided to implement DCA-Restart version for now since simpler and faster.
    # TODO: add DCA-LS version from our ContDSMin paper later since it can perform better in practice
    # when a good initialization is not provided? 
    # We would need to restrict the local search then to true set neighbors for that variant
    logging.info(f"Running DCA for {num_outer_steps} outer iterations and {num_inner_steps} inner iterations")
    time_start = time.time()

    if x_init.dim() == 1:
        x_init = x_init.unsqueeze(0)
    # map x_init to X in [0,1]^n x b
    X = F_set.map.ints2binary(x_init)[0].to(dtype=torch.long)
    n, b = X.shape
    filter_prefixes = make_lattice_prefix_filter(filter_fn, filter_zero) if filter_fn is not None else None

    flops_L_G = 0
    if isinstance(L_G, str):
        if L_G == "singletons":
            L_G, flops_L_G = G_set.singletons_L_bound()
        else:
            raise ValueError("If L_G is a string, it must be 'singletons'.")

    L_G = max(L_G, 1e-12) # L_G < 1e-12 shouldn't happen unless G = 0 but just in case
    assert L_G > 0, "Lipschitz constant L_G must be positive"


    discrete_obj_values = [0.0 for _ in range(num_outer_steps + 1)]
    inner_discrete_values: list[list[float]] = [[] for _ in range(num_outer_steps + 1)]
    discrete_obj_values_filtered = [0.0 for _ in range(num_outer_steps + 1)]
    inner_discrete_values_filtered: list[list[float]] = [[] for _ in range(num_outer_steps + 1)]
    continuous_obj_values = [0.0 for _ in range(num_outer_steps + 1)]
    inner_continuous_values: list[list[float]] = [[] for _ in range(num_outer_steps + 1)]
    inner_duality_gaps: list[list[float]] = [[] for _ in range(num_outer_steps + 1)]
    inner_times: list[list[float]] = [[] for _ in range(num_outer_steps + 1)]
    inner_flops: list[list[int]] = [[] for _ in range(num_outer_steps + 1)]
    discrete_sols_filtered = torch.empty((num_outer_steps + 1, n), dtype=torch.long, device=X.device)
    times = [0.0 for _ in range(num_outer_steps + 1)]
    flops = [0 for _ in range(num_outer_steps + 1)]

    # no need to compute initial obj value, it will be computed in first inner iteration
    initialization_time = time.time() - time_start

    for iter in (pbar := trange(num_outer_steps, file=sys.stdout)):
        result_idx = iter + 1
        tie_breaker = torch.randperm(n * b, device=X.device, dtype=torch.long).view(n, b) if tie_break == "random" else None
        subgrad_H, _, Hvalues, flops_subgrad_H, x_chain = H_set.subgradient_lovasz_extension(X, tie_breaker) # subgrad_H is (n, b)

        if inner_solver == "pgm":
            # minimize upper bound on F_set: F_set_upperbd(S) = G_set(S) - <subgrad_H, 1_S>
            F_set_upperbd = LinearCombinationSetFn([G_set, ModularSetFn(subgrad_H)],[1.0, -1.0])
            L_upperbd = L_G + torch.linalg.vector_norm(subgrad_H.float(), ord=2).item()

            # warm start pgm with current X as initial solution
            inner_best_set, inner_best_continuous_sol, inner_discrete_values[result_idx], inner_discrete_values_filtered[result_idx], \
            inner_continuous_values[result_idx], inner_duality_gaps[result_idx], inner_sets_filtered, inner_times[result_idx], inner_flops[result_idx], \
            _, inner_discrete_sols_filtered = \
                pgm_lovasz_set(
                    F_set_upperbd, X, num_inner_steps, L_upperbd, tie_break=tie_breaker,
                    gap_tol=inner_gap_tol, filter_prefixes=filter_prefixes,
                )

            prev_cont_value = inner_continuous_values[result_idx][0] # f_L_upperbd(X) = g_L(X) - <subgrad_H, X> = g_L(X) - h_L(X) = f_L(X)
            # X.float() is needed to avoid assertion being triggered due to difference between batched and single cross entropy loss evaluations,
            # since inner_continuous_values is set to Fvalues[nnz-1] in pgm while lovasz_extension(X) returns F_set(S) if X is of type long.
            assert abs(prev_cont_value - (continuous_obj_values[result_idx-1] if iter > 0 else  F_set.lovasz_extension(X.float())))  < 1e-12, \
            "prev_cont_value should match the continuous obj value of the previous outer step." # TODO: remove the check for iter==0 when done debugging

        elif inner_solver == "mnp":
            # TODO: implement MNP
            raise NotImplementedError("MNP is not implemented yet.")
        else:
            raise ValueError(f"Inner solver {inner_solver} not supported. Must be 'pgm' or 'mnp'.")

        # TODO: alternatively use binary X corresponding to inner_best_set.
        X = inner_best_continuous_sol

        # compute lovasz extension of F at X (for logging and checking convergence) and round (for logging and getting current discrete sol).
        # TODO: if we use integral solutions, no need to keep track of continuous obj values since they'll be equal to discrete ones.
        subgradient_F, order, Fvalues, flops_subgrad_F, x_chain = F_set.subgradient_lovasz_extension(X, tie_breaker) # use same tie breaker?
        (F_round, _, x_round), (F_round_filtered, _, x_round_filtered) = F_set.round_lovasz_extension(
            order=order, Fvalues=Fvalues, auxiliary_input=x_chain,
            filter_prefixes=filter_prefixes,
        )
        continuous_obj_values[result_idx] = F_set.lovasz_extension(X, subgradient_F, Fvalues)

        if continuous_obj_values[result_idx] > prev_cont_value + inner_duality_gaps[result_idx][-1]:
            logging.warning(
                f"Continuous obj value: {continuous_obj_values[result_idx]:.4f} is larger than previous one: "
                f"{prev_cont_value:.4f} + duality gap reached: {inner_duality_gaps[result_idx][-1]:.4f}. "
                "alpha used in decomposition of F should be increased to ensure it is a difference of DR-submodular functions."
            )

        if iter == 0:
            # discrete obj value of X before being updated matches the first inner discrete obj value if PGM uses same tie breaker
            discrete_obj_values[0] = inner_discrete_values[result_idx][0]
            discrete_obj_values_filtered[0] = inner_discrete_values_filtered[result_idx][0]
            continuous_obj_values[0] = prev_cont_value
            discrete_sols_filtered[0] = inner_discrete_sols_filtered[0]
            # include time/flops to evaluate objective in initialization time/flops
            # TODO: add flops for prefill to initial step flops as done in GCG if we do prefill later
            times[0] = initialization_time + inner_times[result_idx][0]
            flops[0] = flops_L_G + inner_flops[result_idx][0]

        # TODO: check if complement set is better, use that as current sol instead. See Prop G.8 in DSMin paper.
        discrete_obj_values[result_idx] = F_round
        discrete_obj_values_filtered[result_idx] = F_round_filtered
        discrete_sols_filtered[result_idx] = x_round_filtered
        # exclude initialization time/flops from first outer step time/flops
        times[result_idx] = time.time() - time_start - (times[0] if iter == 0 else 0.0)
        # flops_subgrad_H is 0 since current H doesn't involve a forward pass of the model, but keeping it to handle general case
        flops[result_idx] = sum(inner_flops[result_idx][1 if iter == 0 else 0:]) + flops_subgrad_H + flops_subgrad_F

        pbar.set_postfix({"Discrete obj value": discrete_obj_values[result_idx], "Discrete obj value filtered": discrete_obj_values_filtered[result_idx], \
        "Continuous obj value": continuous_obj_values[result_idx], "Inner duality gap reached": inner_duality_gaps[result_idx][-1]})

        if prev_cont_value - continuous_obj_values[result_idx] <= outer_tol:
            F_best_neighbor, best_neighbor, F_best_neighbor_filtered, best_neighbor_filtered, flops_local_search = F_set.get_best_neighbors(
                x_round, filter_fn
            )
            # include local search time and flops
            times[result_idx] = time.time() - time_start - (times[0] if iter == 0 else 0.0)
            flops[result_idx] += flops_local_search

            if F_best_neighbor < F_round:
                logging.info(f"DCA converged after {result_idx} outer steps but not to a local min, restarting from best neighbor "
                             f"with discrete obj value {F_best_neighbor:.4f} and discrete obj value filtered {F_best_neighbor_filtered:.4f}.")
                X = F_set.map.ints2binary(best_neighbor.unsqueeze(0))[0].to(dtype=torch.long)
                discrete_obj_values[result_idx] = F_best_neighbor
                # use current filtered discrete solution if better than best filtered neighbor
                discrete_obj_values_filtered[result_idx] = min(F_best_neighbor_filtered, F_round_filtered)
                discrete_sols_filtered[result_idx] = best_neighbor_filtered if F_best_neighbor_filtered < F_round_filtered else x_round_filtered
                # X is set to the binary matrix corresponding to R(best_neighbor), so f_L(X) = F_set(R(best_neighbor)) = F(M(R(best_neighbor))) = F_best_neighbor
                continuous_obj_values[result_idx] = F_best_neighbor 

            else:
                logging.info(f"DCA converged after {result_idx} outer steps to a local min, stopping.")
                break

        time_start = time.time()

    # truncate to actual number of outer iterations executed
    discrete_obj_values = discrete_obj_values[:result_idx + 1]
    inner_discrete_values = inner_discrete_values[:result_idx + 1]
    discrete_obj_values_filtered = discrete_obj_values_filtered[:result_idx + 1]
    inner_discrete_values_filtered = inner_discrete_values_filtered[:result_idx + 1]
    continuous_obj_values = continuous_obj_values[:result_idx + 1]
    inner_continuous_values = inner_continuous_values[:result_idx + 1]
    inner_duality_gaps = inner_duality_gaps[:result_idx + 1]
    discrete_sols_filtered = discrete_sols_filtered[:result_idx + 1, :]
    times = times[:result_idx + 1]
    inner_times = inner_times[:result_idx + 1]
    flops = flops[:result_idx + 1]
    inner_flops = inner_flops[:result_idx + 1]

    return (
        discrete_obj_values,
        discrete_obj_values_filtered,
        continuous_obj_values,
        discrete_sols_filtered,
        times,
        flops,
        inner_discrete_values,
        inner_discrete_values_filtered,
        inner_continuous_values,
        inner_duality_gaps,
        inner_times,
        inner_flops,
    )
