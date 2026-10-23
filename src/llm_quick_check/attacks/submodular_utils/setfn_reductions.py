"""
Classes for reductions from DR-submodular lattice functions to submodular set functions
and related utilities.
"""
from abc import ABC, abstractmethod
import torch
from typing import Any
from collections.abc import Callable, Mapping
from torch import Tensor
from math import log2, ceil, inf
import logging
from pathlib import Path
import time
from .lattice_functions import CallableLatticeFunction, LatticeFunction
from .set_functions import SetFunction, assert_valid_set

# TODO: Refactor all submodular_utils to work with general set functions on [n] x [b] and have SetFnReduction handle things
# specific to the reduction.

class SetToLatticeMap(ABC):
    """Base class for map M: 2^([n] x [b]) -> V^n and its right inverse R: V^n -> 2^([n] x [b]), 
    where V = {0, 1, ..., k - 1} and [M(S)]_i = sum_{j in [b], (i, j) in S} weights[j].

    M should be surjective, to have a right inverse R, i.e., a map which satisfies 
    M(R(x)) = x for all x in V^n; R(M(S)) need not equal S.
    Subclasses should implement get_weights and ints2binary.
    """
    #TODO: add a public wrapper for ints2binary to consolidate validation of inputs and output.

    def __init__(
        self,
        k: int, # TODO: k is not used in this class, we can let subclasses handle this
        n: int,
        device: torch.device,
    ):
        self.device = device
        self.k = k
        self.n = n
        self.weights = self.get_weights()
        assert self.weights.dim() == 1, "get_weights must return a 1D tensor of length b"
        assert self.weights.device == self.device, "get_weights must return a tensor on self.device"
        self.b = int(self.weights.shape[0])

    @abstractmethod
    def get_weights(self) -> Tensor:
        """Return weight vector of shape (b,) for the reduction."""

    def _assert_in_Vn(self, x: Tensor) -> None:
        assert x.dtype == torch.long and (x >= 0).all() and (x < self.k).all(), "x must have values in {0, ..., k - 1}"

    @abstractmethod
    def ints2binary(self, x: Tensor) -> Tensor:
        """Batched version of the right inverse R: V^n -> 2^([n] x [b]) with sets S in [n] x [b]
        represented by binary matrices X in {0,1}^n x b such that X[j, c] = 1 iff (j, c) in S.

        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.

        Returns:
            binary_matrices: Tensor of type bool and shape (batch_size, n, b).
            Each X[i] = binary_matrices[i] represents a subset S^i of [n] x [b] such that S^i = R(x[i]).
        """

    def ints2set(self, x: Tensor) -> tuple[list[Tensor], list[Tensor]]:  # TODO: not used anywhere yet, remove if not needed
        """Batched version of the right inverse R: V^n -> 2^([n] x [b]) with sets S in [n] x [b] represented
        by paired rows and cols indices, i.e., S = {(rows[j], cols[j]) for j in range(rows.shape[0])}.

        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            rows_list: List of batch_size 1D long tensors, containing row indices.
            cols_list: List of batch_size 1D long tensors, containing column indices.
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [b] such that S^i = R(x[i]).
            Empty tensors represent the empty set.
        """
        mask = self.ints2binary(x)

        batch_idx, rows, cols = mask.nonzero(as_tuple=True)
        rows_list: list[Tensor] = []
        cols_list: list[Tensor] = []
        for i in range(mask.size(0)):
            batch_mask = batch_idx == i
            rows_list.append(rows[batch_mask].long())
            cols_list.append(cols[batch_mask].long())
        return rows_list, cols_list

    def set2ints(self, rows_list: list[Tensor], cols_list: list[Tensor]) -> Tensor:  # used in F_set_batch
        """Batched version of map M: 2^([n] x [b]) -> V^n with sets S in [n] x [b] represented
        by paired rows and cols indices.

        Args:
            rows_list: List of 1D tensors of type long and length <= n x b
            cols_list: List of 1D tensors of type long and length <= n x b
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [b]. The
            (row, col) pairs within each set must be unique.
        Returns:
            x: Tensor of type long and shape (batch_size, n). Each row x[i] is an integer vector in V^n such that M(S^i) = x[i].
        """
        assert len(rows_list) == len(cols_list), "rows_list and cols_list must have the same length"
        # TODO: Replace per-set lists with a sparse COO representation (batch_idx, rows, cols) for efficiency (see version below).
        # For now will keep this simpler implementation.

        x = torch.zeros((len(rows_list), self.n), dtype=torch.long, device=self.device)
        for i, (rows, cols) in enumerate(zip(rows_list, cols_list)):
            assert_valid_set(rows, cols, n=self.n, b=self.b, device=self.device)
            if cols.numel() > 0:
                x[i].index_add_(0, rows, self.weights[cols])  # x[i, rows[j]] += weights[cols[j]] for all j
        return x

    # TODO: potentially move to these versions of ints2set and set2ints for efficiency.
    # Issue: zero vectors which correspond to empty sets are not included in the output!
    # This can be fixed by making batch_size not optional in set2ints_batched.
    # def ints2set_batched(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    #     """Same map as older ``ints2set``, but returns a single sparse COO layout instead of per-batch lists.

    #     Args:
    #         x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.

    #     Returns:
    #         batch_idx, rows, cols: 1D long tensors of equal length ``total`` such that for each t,
    #         (rows[t], cols[t]) lies in S^{batch_idx[t]} and M(S^i) = x[i] for all i. Batches with no
    #         elements do not appear (no rows for that ``i`` in ``batch_idx``).
    #     """
    #     assert x.dim() == 2 and x.shape[1] == self.n, "x must have shape (batch_size, n)"
    #     assert x.dtype == torch.long, "x must be of type long"
    #     assert x.device == self.device, "x must be on the same device as self.device"
    #     mask = self.ints2binary(x)
    #     batch_idx, rows, cols = mask.nonzero(as_tuple=True)
    #     return batch_idx.long(), rows.long(), cols.long()

    # def set2ints_batched(
    #     self,
    #     batch_idx: Tensor,
    #     rows: Tensor,
    #     cols: Tensor,
    #     batch_size: int | None = None,
    # ) -> Tensor:
    #     """Same map as ``set2ints``, but sets are given as one COO stream (``batch_idx``, ``rows``, ``cols``).

    #     For each index t, element (rows[t], cols[t]) belongs to S^{batch_idx[t]}. Multiple entries with the
    #     same ``batch_idx`` and ``rows`` (different ``cols``) accumulate like ``index_add_``.

    #     Args:
    #         batch_idx: 1D long tensor, batch index in [0, batch_size) for each set element.
    #         rows: 1D long tensor in [0, n).
    #         cols: 1D long tensor in [0, b).
    #         batch_size: Number of batches (rows of the output ``x``). If ``None``, inferred as
    #             ``int(batch_idx.max().item()) + 1`` when there is at least one element; if there are no
    #             elements, must be provided (use ``0`` for an empty output of shape ``(0, n)``).

    #     Returns:
    #         x: Tensor of shape (batch_size, n) with the same semantics as ``set2ints``.
    #     """
    #     assert batch_idx.dim() == 1 and rows.dim() == 1 and cols.dim() == 1
    #     assert batch_idx.shape == rows.shape == cols.shape, "batch_idx, rows, cols must have the same shape"
    #     assert batch_idx.device == rows.device == cols.device == self.device
    #     assert batch_idx.dtype == rows.dtype == cols.dtype == torch.long

    #     total = batch_idx.numel()
    #     if total == 0:
    #         if batch_size is None:
    #             return torch.zeros((0, self.n), dtype=torch.long, device=self.device)
    #         return torch.zeros((batch_size, self.n), dtype=torch.long, device=self.device)

    #     if batch_size is None:
    #         batch_size = int(batch_idx.max().item()) + 1
    #     assert batch_size > 0, "batch_size must be positive when there is at least one set element"
    #     assert (batch_idx >= 0).all() and (batch_idx < batch_size).all(), (
    #         "batch_idx must be in [0, batch_size)"
    #     )
    #     assert (rows >= 0).all() and (rows < self.n).all(), "rows must be in [0, n)"
    #     assert (cols >= 0).all() and (cols < self.b).all(), "cols must be in [0, b)"

    #     flat_idx = batch_idx * self.n + rows
    #     x_flat = torch.zeros(batch_size * self.n, dtype=torch.long, device=self.device)
    #     x_flat.scatter_add_(0, flat_idx, self.weights[cols])
    #     return x_flat.view(batch_size, self.n)

    def binary2ints(self, X: Tensor) -> Tensor: # TODO: not used anywhere yet, remove if not needed.
        """Batched version of map M: 2^([n] x [b]) -> V^n with sets S in [n] x [b] represented
        by binary matrices X in {0,1}^n x b:

        x[i, j] = sum_c X[i, j, c] * weights[c].

        Args:
            X: Tensor of shape (batch_size, n, b), bool or {0, 1}.

        Returns:
            x: Tensor of type long and shape (batch_size, n).
        """
        assert X.dim() == 3 and X.shape[1] == self.n and X.shape[2] == self.b, (
            f"X must have shape (batch_size, n, b) with n={self.n}, b={self.b}, got {tuple(X.shape)}"
        )
        assert X.device == self.device, "X must be on the same device as self.device"
        w = self.weights.view(1, 1, self.b)
        return (X.to(dtype=self.weights.dtype) * w).sum(dim=-1).to(torch.long)



class SetFnReduction(SetFunction):
    """Reduction from a lattice function F: V^n -> R where V = {0, 1,..., k - 1},
    to a set function F_set: 2^([n] x [b]) -> R, using a SetToLatticeMap for M.

    F is expected to be normalized,, i.e., F(0) = 0.
    """

    def __init__(
        self,
        lattice_fn: Callable[[Tensor], tuple[Tensor, int]] | LatticeFunction,
        reduction_map: SetToLatticeMap,
    ):
        super().__init__(n=reduction_map.n, b=reduction_map.b, device=reduction_map.device)
        self.map = reduction_map
        self.k = reduction_map.k
        assert self.k > 1, "k must be greater than 1"
        self.lattice_fn: LatticeFunction = (
            lattice_fn if isinstance(lattice_fn, LatticeFunction)
            else CallableLatticeFunction(self.k, self.n, lattice_fn)
        )
        assert self.lattice_fn.n == self.n, "lattice_fn.n must match reduction_map.n"
        assert self.lattice_fn.k == self.k, "lattice_fn.k must match reduction_map.k"
    def _eval_batch(self, rows_list: list[Tensor], cols_list: list[Tensor]) -> tuple[Tensor, int]:
        """Evaluate F_set on a non-empty batch of sets validated by eval_batch."""
        #TODO: set2ints also validates the sets, so this repeats validation. We can split map.set2ints
        # into a validating wrapper and a private implementation to avoid that.
        x = self.map.set2ints(rows_list, cols_list)
        return self.lattice_fn._eval_batch(x)

    def _eval_chain(self, rows: Tensor, cols: Tensor) -> tuple[Tensor, int, Tensor]:
        """Evaluate validated chain of sets via self.lattice_fn._eval_chain and return x_chain as auxiliary_data."""
        # this is more efficient than the default _eval_chain which would call SetFnReduction.eval_batch on S^i's 
        # and compute each x^i separately (O(n * b) vs O((n * b)^2))
        if rows.numel() == 0:
            x_chain = torch.empty((0, self.n), dtype=torch.long, device=self.device)
            return torch.empty(0, device=self.device), 0, x_chain
        return self.lattice_fn._eval_chain(rows, cols, self.map.weights)

    def _round_auxiliary_output(self, min_idx: int, auxiliary_input: Tensor) -> Tensor:
        """Return the cached lattice point corresponding to the selected prefix:
        x_min = M(S_min) in V^n

        auxiliary_input is x_chain. Return empty tensor of shape (n,) if no eligible prefix.
        """
        x_chain = auxiliary_input
        if min_idx < 0:
            return x_chain.new_empty(self.n)
        if min_idx == 0:
            return x_chain.new_zeros(self.n)
        return x_chain[min_idx - 1]

    def get_best_neighbors(
        self, x: Tensor, filter_fn: Callable[[Tensor], list[int]] | None = None
    ) -> tuple[float, Tensor, float, Tensor, int]:
        """Get the best neighbor of x in V^n for F, i.e., argmin_{i, j} F(x ± weight[j] e_i)

        Note this is different than F_set(S ± (i, j)) since some (i, j) that can't be added/removed 
        from S can be added/removed from x. Neighbors considered here are superset of S neighbors.
        The best neighbor found might not correspond to a neighbor of S. 

        Args:
            x: Tensor of shape (n,).
        Returns:
            F_best_neighbor: float.
            best_neighbor: Tensor of shape (n,).
            F_best_neighbor_filtered: float. Same as F_best_neighbor if filtering is disabled.
            best_neighbor_filtered: Tensor of shape (n,). Same as best_neighbor if filtering is disabled.
            flops: flop count, int.
        """
        assert x.device == self.device, "x must be on the same device as self.device"
        Fvalues, x_neighbors, flops = self.lattice_fn.eval_neighbors(x, self.map.weights)
        if x_neighbors.shape[0] == 0: # never happens with current reductions. Can happen if weights doesn't include 1
            return inf, x, inf, x, flops

        F_best_neighbor, best_idx = torch.min(Fvalues, dim=0)
        F_best_neighbor = F_best_neighbor.item()
        best_neighbor = x_neighbors[best_idx]

        if filter_fn is not None:
            # drop neighbors whose full prompt tokenization would be unreachable from any input string
            retain_idx = filter_fn(x_neighbors)
            if not retain_idx:
                logging.warning("No neighbors of current x are reachable. Setting F_best_neighbor_filtered=inf and best_neighbor_filtered to an empty tensor.")
                F_best_neighbor_filtered = inf
                best_neighbor_filtered = torch.empty((self.n,), dtype=torch.long, device=self.device)
            else:
                F_best_neighbor_filtered, best_idx_filtered = torch.min(Fvalues[retain_idx], dim=0)
                F_best_neighbor_filtered = F_best_neighbor_filtered.item()
                best_neighbor_filtered = x_neighbors[retain_idx][best_idx_filtered]
        else:
            F_best_neighbor_filtered = F_best_neighbor
            best_neighbor_filtered = best_neighbor

        return F_best_neighbor, best_neighbor, F_best_neighbor_filtered, best_neighbor_filtered, flops

    def hessian_upperbd_at_zero(self, normalized: bool = True, singleton_vals: Tensor | None = None, save_file: str | Path | None = None, fingerprint: Mapping[str, Any] | None = None) -> tuple[Tensor, int, float]:
        """Compute an approximate upper bound on the "Hessian" of F at 0:

        We want to compute:
        max_{x, a_1, a_2} ((F(x + a_1 e_i1 + a_2 e_i2) - F(x + a_2 e_i2)) - (F(x + a_1 e_i1) - F(x))) / (a_1 a_2)
        for all i1, i2 in [n], a_1, a_2 > 0 s.t x + a_1 e_i1 + a_2 e_i2 in V^n. This can be viewed as a discrete Hessian 
        bound because if F is differentiable, taking a_1, a_2 -> 0, gives ∇^2F(x)_{i1, i2}. 
        It's enough to consider a_1 = a_2 = 1 (max is reached there), but since we're only computing
        the bound at x=0, that's not enough. Bound at x=0 costs O(n^2 k^2) evaluations of F.

        We instead consider the maximum over only weights of the map a_j1 = weights[j1], a_j2 = weights[j2], i.e.,
        Q_{i1, i2} = max_{j1, j2 in [b]} ((F(a_j1 e_i1 + a_j2 e_i2) - F(a_j2 e_i2)) - (F(a_j1 e_i1) - F(0))) / (a_j1 a_j2)
                   = max_{j1, j2 in [b]} (F_set({v1, v2}) - F_set(v1) - F_set(v2)) / (a_j1 a_j2) where v1 = (i1, j1), v2 = (i2, j2),
        since F is normalized. This costs O(n^2 b^2) evaluations of F_set / F.

        If normalized is False, don't normalize by a_j1 a_j2.

        Returns:
            hessian_upperbd: symmetric (n, n) tensor Q
            flops: flop count for singleton and pair evaluations.
            time_taken: elapsed computation time in seconds.
        """
        t_start = time.time()
        nb = self.n * self.b
        if nb < 2: # possible if n = 1 and k = 2 (so b = 1), no distinct pairs v1, v2 in this case, so Q = 0
            return torch.zeros((self.n, self.n), device=self.device, dtype=torch.float32), 0, 0.0
        flops_singletons = 0
        if singleton_vals is None:
            singleton_vals, flops_singletons = self.eval_singletons()
        pair_vals, rows, cols, flops_pairs = self.eval_all_pairs()
        i1, i2 = rows[0], rows[1]
        j1, j2 = cols[0], cols[1]
        singleton_mat = singleton_vals.view(self.n, self.b)
        w = self.map.weights
        cross_vals = pair_vals - singleton_mat[i1, j1] - singleton_mat[i2, j2]
        normalized_cross_vals = cross_vals / (w[j1] * w[j2])

        hessian_upperbd_flat = torch.full((self.n * self.n,), -float("inf"), device=self.device, dtype=cross_vals.dtype)
        if normalized:
            hessian_upperbd_flat.scatter_reduce_(0, i1 * self.n + i2, normalized_cross_vals, reduce="amax", include_self=True)
        else:
            hessian_upperbd_flat.scatter_reduce_(0, i1 * self.n + i2, cross_vals, reduce="amax", include_self=True)

        hessian_upperbd = hessian_upperbd_flat.view(self.n, self.n)
        hessian_upperbd = torch.maximum(hessian_upperbd, hessian_upperbd.mT) # copy values of Q_{i1, i2} to Q_{i2, i1}

        hessian_max = hessian_upperbd_flat.max().item()
        logging.info(f"cross_vals_max: {cross_vals.max().item()}") # 0.51898 for Llama-3.2-1B-Instruct, 1st conversation in adv_behaviors
        logging.info(f"hessian_max: {hessian_max}") # becomes 0.07127 with normalized=True

        flops = flops_singletons + flops_pairs
        time_taken = time.time() - t_start
        if save_file is not None:
            save_path = Path(save_file)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "cross_vals": cross_vals.cpu(),
                    "normalized_cross_vals": normalized_cross_vals.cpu(),
                    "hessian_upperbd": hessian_upperbd.cpu(),
                    "normalized": normalized,
                    "n": self.n,
                    "b": self.b,
                    "flops": flops,
                    "time_taken": time_taken,
                    "fingerprint": fingerprint
                },
                save_path,
            )
        return hessian_upperbd, flops, time_taken


class EneReductionMap(SetToLatticeMap):
    """Ene-Nguyen's map M: 2^([n] x [b]) -> V^n for reduction from a DR-submodular lattice function F: V^n -> R,
    where V = {0, 1,..., k - 1}, to a submodular set function F_set: 2^([n] x [b]) -> R where F_set(S) = F(M(S)).

    M need not be injective. ints2binary implements a right inverse R by choosing one decomposition
    of each lattice point into a subset of the weights.

    M is defined in Lemma 1 of the paper:

    @article{ene2016reduction,
        title   = {A Reduction for Optimizing Lattice Submodular Functions with Diminishing Returns},
        author  = {Alina Ene and Huy L. Nguyen},
        year    = {2016},
        journal = {arXiv preprint arXiv: 1606.08362}}
    """
    def __init__(
        self,
        k: int,
        n: int,
        device: torch.device,
    ):
        assert k > 1, "k must be greater than 1"
        self.v_max = k - 1
        super().__init__(k, n, device)

    def get_weights(self) -> Tensor:
        """Multiset of b weights a_1, ..., a_b summing to v_max = k-1.

        Base weights: a_1 = 1, a_i = 2^{i-2} for 2 <= i <= m+1 (indices 0...m), with m = floor(log2(k-1)).
        Remainder weights: a_{m+1+j} = 2^{c_j} for 1 <= j <= p, where c_j in {0, ..., m-1} 
        is the j-th non-zero bit in the binary representation of v_max other than m. 
        Total # of weights is b = (m + 1) + p.
        """
        m = self.v_max.bit_length() - 1
        self.m = m
        idx_bits = torch.arange(m + 1, device=self.device, dtype=torch.long)
        self.v_max_bits = ((self.v_max >> idx_bits) & 1).bool()  # binary representation of v_max
        idx_bits = idx_bits[:-1]  # drop msb bit
        base_weights = 1 << idx_bits  # 2^0, ..., 2^{m-1}
        self.v_max_non_zero_bits = idx_bits[self.v_max_bits[:-1]]  # non-zero bits of v_max other than m
        self._remainder_col = {
            bit: m + 1 + idx for idx, bit in enumerate(self.v_max_non_zero_bits.tolist())
        }  # this is {c_j: m + 1 + j} in the lemma
        one = torch.tensor([1], dtype=torch.long, device=self.device)
        weights = torch.cat((one, base_weights, base_weights[self.v_max_non_zero_bits]))
        return weights

    # TODO: Maybe it's better to actually store the map from integers in V to sets, instead of recomputing it every time.
    def ints2binary(self, x: Tensor) -> Tensor:  # used to convert x_init in V^n to X in {0,1}^n x b in pgm and DCA and in ints2set
        """Implement R by choosing a decomposition of each entry in x into a subset of the weights a_i's:
        x[i,j] = sum_{c in [b]} X[i, j, c] * a_c, where X is a bool tensor of shape (batch_size, n, b).
        This ensures M(R(x)) = x, but does not recover arbitrary sets from their images under M.
        """
        assert x.dim() == 2 and x.shape[1] == self.n, "x must be (batch_size, n)"
        self._assert_in_Vn(x)
        assert x.device == self.device, "x must be on the same device as the reduction"

        m = self.m
        b = self.b
        v_max = self.v_max
        batch_size, n = x.shape

        is_zero = x == 0
        is_v_max = x == v_max
        non_trivial_idx = ~(is_zero | is_v_max)
        mask_x = torch.zeros(batch_size, n, b, dtype=torch.bool, device=self.device)
        mask_x[is_v_max] = True

        if m > 0:
            idx_bits = torch.arange(m + 1, device=self.device, dtype=torch.long)
            x_bits = ((x.unsqueeze(-1) >> idx_bits) & 1).bool()  # binary representation of x (batch_size, n, m + 1)
            diff_bits = self.v_max_bits.view(1, 1, m + 1) & ~x_bits  # bits that are 1 for v_max but 0 for x[b, i]'s
            largest_diff_bits = torch.where(
                diff_bits,
                idx_bits.view(1, 1, m + 1).expand(batch_size, n, -1),
                -1,  # if no diff bits are found, set to -1 (happens only for x[b, i] = v_max)
            ).max(dim=-1).values

            # r is the number that agrees with v_max on all bits larger than the largest_diff_bits and is 0 on smaller bits
            r = torch.zeros_like(x)
            r[non_trivial_idx] = (v_max >> largest_diff_bits[non_trivial_idx]) << largest_diff_bits[non_trivial_idx]

            # get decomposition mask for r
            mask_r = torch.zeros(batch_size, n, b, dtype=torch.bool, device= self.device)
            mask_r[:, :, : m + 1] = True  # all base weights are included since bit m is 1 in r
            # r has same non-zero bits as v_max for bits larger than largest_diff_bits, include corresponding columns in the decomposition
            mask_r[:, :, m + 1 :] = self.v_max_non_zero_bits.view(1, 1, -1) >= largest_diff_bits.unsqueeze(-1)

            diff_r_x = torch.where(non_trivial_idx, r - x, torch.zeros_like(x))
            diff_r_x_bits = ((diff_r_x.unsqueeze(-1) >> idx_bits[:-1]) & 1).bool()  # binary representation of r - x (batch_size, n, m)
            mask_x[non_trivial_idx] = mask_r[non_trivial_idx]
            mask_x[:, :, 1 : m + 1] &= ~diff_r_x_bits

        return mask_x


class EneSubmodularSetFnReduction(SetFnReduction):
    """Ene-Nguyen's set function reduction using EneReductionMap"""
    def __init__(
        self,
        lattice_fn: Callable[[Tensor], tuple[Tensor, int]] | LatticeFunction,
        k: int,
        n: int,
        device: torch.device,
    ):
        ene_map = EneReductionMap(k, n, device)
        super().__init__(lattice_fn, ene_map)



class BinaryRepresentationMap(SetToLatticeMap):
    """Binary representation map M: 2^([n] x [b]) -> V^n:
    x = M(S) is such that each x_i is the integer with binary representation X[i, :],
    where X is the binary matrix with 1 at indices in S, 0 elsewhere.
    Least significant bit is at column index 0 (bit index matches column index).

    BinaryRepresentationMap M can be used to reduce a DR-submodular lattice function F: V^n -> R, where V = {0, 1,..., k - 1},
    to a submodular set function F_set: 2^([n] x [b]) -> R where F_set(S) = F(M(S)), if k is a power of 2.
    """
    def __init__(
        self,
        k: int,
        n: int,
        device: torch.device,
    ):
        assert k > 1, "k must be greater than 1"
        super().__init__(k, n, device)

    def get_weights(self) -> Tensor:
        b = ceil(log2(self.k))
        weights = 1 << torch.arange(b, dtype=torch.long, device=self.device)  # more efficient than 2**torch.arange(b)
        return weights

    def ints2binary(self, x: Tensor) -> Tensor:
        """Map each entry in x to its binary representation"""
        assert x.dim() == 2 and x.shape[1] == self.n, "x must be (batch_size, n)"
        self._assert_in_Vn(x)
        assert x.device == self.device, "x must be on the same device as the reduction"

        idx_bits = torch.arange(self.b, device=self.device, dtype=torch.long)
        x_bits = ((x.unsqueeze(-1) >> idx_bits) & 1).bool()
        return x_bits


class BinarySubmodularSetFnReduction(SetFnReduction):
    """Set function reduction using BinaryRepresentationMap
    k should be a power of 2 for F_set to be submodular.

    TODO: If k is not a power of 2, k' = ceil(log2(k)) can be used and any integer >= k cut off.
    The resulting reduction would then only preserve DR-submodularity if F is non-decreasing (see overleaf notes)
    This can be an alternative to Ene-Nguyen's reduction if we decompose into non-decreasing DR-submodular functions.
    """

    def __init__(
        self,
        lattice_fn: Callable[[Tensor], tuple[Tensor, int]] | LatticeFunction,
        k: int,
        n: int,
        device: torch.device,
    ):
        binary_map = BinaryRepresentationMap(k, n, device)
        assert k == 2**binary_map.b, "k must be a power of 2"
        super().__init__(lattice_fn, binary_map)
