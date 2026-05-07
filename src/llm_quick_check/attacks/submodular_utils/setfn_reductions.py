"""
Classes for reductions from DR-submodular lattice functions to submodular set functions
and related utilities.
"""
from abc import ABC, abstractmethod
import torch
from typing import Callable, List, Optional, Tuple, Union
from torch import Tensor
from math import log2, ceil

from .lattice_functions import CallableLatticeFunction, LatticeFunction


# TODO: Refactor all submodular_utils to work with general set functions on [n] x [b] and have SetFnReduction handle things
# specific to the reduction.

def subgradient_lovasz_extension(
    lattice_fn: LatticeFunction,
    weights: Tensor,
    X: Tensor,
    tie_breaker: Optional[Tensor] = None,
):
    """Compute a subgradient of the Lovasz extension f_L of a submodular set function F_set: 2^([n] x [b]) -> R
    using Edmonds' greedy algorithm.

    F_set is given by F_set(S) = F(M(S)) where M: 2^([n] x [b]) -> V^n is [M(S)]_i = sum_{(i, j) in S} weights[j].
    F is assumed to be normalized, i.e., F(0) = 0.

    Args:
        lattice_fn: LatticeFunction instance for F.
        weights: Tensor of shape (b,)
        X: Tensor of shape (n, b) in [0,1]^n x b
        tie_breaker: Tensor of shape (n, b) used to break ties when sorting
            X.flatten(). If not provided, original order is used.
    Returns:
        subgradient: Tensor of shape (n, b)
        Fvalues: Tensor of shape (n x b,)
        x_chain: Tensor of shape (n x b, n)
    """
    # TODO: for now assume x is a 2D tensor, not sure if there's a reason to vectorize it
    # if X is already flattened, we can pass (n, b) to reshape it to (n, b) here
    assert X.dim() == 2, "X must be a 2D tensor"
    n, b = X.shape
    assert weights.dim() == 1 and weights.shape[0] == b, "weights must be a 1D tensor of shape (b,)"
    assert lattice_fn.n == n, "lattice_fn.n must match X.shape[0]"
    assert (
        lattice_fn.eval_batch(torch.zeros(1, n, dtype=torch.long, device=X.device))[0].item() == 0
    ), "F must be normalized"
    if tie_breaker is not None:
        assert tie_breaker.shape == X.shape, "tie_breaker must be the same shape as X"

    if tie_breaker is None:
        sorted_idx = torch.argsort(X.flatten(), descending=True, stable=True)
    else:
        sorted_idx = torch.argsort(tie_breaker.flatten(), descending=True, stable=True)
        sorted_idx = sorted_idx[torch.argsort(X.flatten()[sorted_idx], descending=True, stable=True)]

    rows, cols = torch.unravel_index(sorted_idx, X.shape)  # both are (n x b,)

    # evaluate F(x^i) for all x^i corresponding to S^i = {(rows[0], cols[0]), ..., (rows[i-1], cols[i-1])} for i in [n * b]
    # compute x^i's sequentially which is more efficient than calling SetFnReduction.set_fn on S^i's which will compute 
    # each x^i separately (O(n * b) vs O((n * b)^2)) 
    x_chain = torch.empty((rows.shape[0], n), dtype=torch.long, device=X.device)  # (m, n)
    x = torch.zeros(n, dtype=torch.long, device=X.device)
    for i in range(rows.shape[0]):
        x[rows[i]] += weights[cols[i]]
        x_chain[i] = x
    Fvalues, flops = lattice_fn.eval_chain(rows, cols, weights, x_chain)
    assert Fvalues.shape[0] == n * b, "lattice_fn must return one scalar per input row"

    # compute subgradient G[rows[i], cols[i]] = F_set(S^i) - F_set(S^{i-1}) = F(x^i) - F(x^{i-1}), assume F(0)= 0
    subgradient = torch.zeros_like(Fvalues)  # (n * b,)
    subgradient[sorted_idx] = torch.diff(Fvalues, prepend=torch.zeros(1, dtype=Fvalues.dtype, device=Fvalues.device))
    subgradient = subgradient.view_as(X)  # (n, b)

    return subgradient, Fvalues, x_chain, flops

class SetToLatticeMap(ABC):
    """Base class for map M: 2^([n] x [b]) -> V^n and its inverse M^{-1}: V^n -> 2^([n] x [b])
    where V = {0, 1, ..., k - 1} and [M(S)]_i = sum_{j in [b], (i, j) in S} weights[j].
    Subclasses should implement get_weights and ints2binary.
    """

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

    @abstractmethod
    def ints2binary(self, x: Tensor) -> Tensor:
        """Batched version of the inverse map M^{-1}: V^n -> 2^([n] x [b]) with sets S in [n] x [b]
        represented by binary matrices X in {0,1}^n x b such that X[j, c] = 1 iff (j, c) in S. 
        
        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.
        
        Returns:
            binary_matrices: Tensor of type bool and shape (batch_size, n, b). 
            Each X[i] = binary_matrices[i] represents a subset S^i of [n] x [b] such that M^{-1}(x[i]) = S^i.
        """

    def ints2set(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:  # TODO: not used anywhere yet, remove if not needed
        """Batched version of the inverse map M^{-1}: V^n -> 2^([n] x [b]) with sets S in [n] x [b] represented
        by paired rows and cols indices, i.e., S = {(rows[j], cols[j]) for j in range(rows.shape[0])}.

        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            rows_list: List of 1D tensors of type long and length <= n x b
            cols_list: List of 1D tensors of type long and length <= n x b
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [b] such that M^{-1}(x[i]) = S^i.
        """
        mask = self.ints2binary(x)

        batch_idx, rows, cols = mask.nonzero(as_tuple=True)
        rows_list: List[Tensor] = []
        cols_list: List[Tensor] = []
        for i in range(mask.size(0)):
            batch_mask = batch_idx == i
            rows_list.append(rows[batch_mask].long())
            cols_list.append(cols[batch_mask].long())
        return rows_list, cols_list

    def set2ints(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tensor:  # used in F_set_batch
        """Batched version of map M: 2^([n] x [b]) -> V^n with sets S in [n] x [b] represented
        by paired rows and cols indices.

        Args:
            rows_list: List of 1D tensors of type long and length <= n x b
            cols_list: List of 1D tensors of type long and length <= n x b
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [b].
        Returns:
            x: Tensor of type long and shape (batch_size, n). Each row x[i] is an integer vector in V^n such that M(S^i) = x[i].
        """
        assert len(rows_list) == len(cols_list), "rows_list and cols_list must have the same length"
        assert all(
            rows_list[i].device == self.device and cols_list[i].device == self.device
            for i in range(len(rows_list))
        ), "all rows_list and cols_list must be on the same device"
        # TODO: this doesn't check if (row, col) pairs are unique (so true set). Add this check,
        # or modify input to be sets of indices in [n x b] which can easily checked for uniqueness before
        # splitting into rows and cols. 
        # TODO: might be more efficient to take as input batch_idx, rows, cols instead, but for now will keep this
        # simpler implementation.

        x = torch.zeros((len(rows_list), self.n), dtype=torch.long, device=self.device)
        for i, (rows, cols) in enumerate(zip(rows_list, cols_list)):
            assert rows.shape[0] == cols.shape[0], "rows and cols must have the same length"
            if cols.numel() > 0:
                x[i].index_add_(0, rows, self.weights[cols])  # x[i, rows[j]] += weights[cols[j]] for all j
        return x

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



class SetFnReduction():
    """Reduction from a lattice function F: V^n -> R where V = {0, 1,..., k - 1},
    to a set function F_set: 2^([n] x [b]) -> R, using a SetToLatticeMap for M.

    F should be normalized, i.e., F(0) = 0.
    """

    def __init__(
        self,
        lattice_fn: Union[Callable[[Tensor], Tuple[Tensor, int]], LatticeFunction],
        reduction_map: SetToLatticeMap,
        filter_fn: Optional[Callable[[Tensor], Tensor]] = None,
        filter_zero: bool = False,
    ):
        self.map = reduction_map
        self.k = reduction_map.k
        self.n = reduction_map.n
        self.b = reduction_map.b
        assert self.k > 1, "k must be greater than 1"
        self.lattice_fn: LatticeFunction = (
            lattice_fn if isinstance(lattice_fn, LatticeFunction)
            else CallableLatticeFunction(self.n, lattice_fn)
        )
        assert self.lattice_fn.n == self.n, "lattice_fn.n must match reduction_map.n"
        self.filter_fn = filter_fn
        self.filter_zero = filter_zero
        self.device = reduction_map.device
 

    def __call__(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:
        return self.set_fn(rows_list, cols_list)

    def set_fn(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:  # used in singleton_L_bound
        """Batched version of F_set: 2^([n] x [b]) -> R: Compute F_set(S^i) for the set
        S^i = {(rows_list[i][j], cols_list[i][j]) for j in range(rows_list[i].shape[0])}.
        """
        x = self.map.set2ints(rows_list, cols_list)
        return self.lattice_fn(x)

    def get_best_neighbors(self, x: Tensor) -> Tuple[float, Tensor, float, Tensor, int]:
        """Get the best neighbor of x in V^n for F, i.e., argmin_{i, j} F(x ± weight[j] e_i)
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
        assert x.dim() == 1 and x.shape[0] == self.n, "x must have shape (n,)"
        assert x.dtype == torch.long, "x must be of type long"

        # get neighbors of x in V^n in the order:
        # 1) all x + weights[j] e_i in V^n for all i, j 
        # 2) all x - weights[j] e_i in V^n for all i, j

        # enumerate all (i, j) pairs in [n] x [b]
        i_idx = torch.arange(self.n, device=self.device, dtype=torch.long).repeat_interleave(self.b)  # (n*b,)
        j_idx = torch.arange(self.b, device=self.device, dtype=torch.long).repeat(self.n)  # (n*b,)

        weights = self.map.weights[j_idx]  # (n*b,)

        # check if the neighbor is in V^n
        add_valid = (x[i_idx] + weights) <= self.k - 1
        rmv_valid = (x[i_idx] - weights) >= 0

        num_add = add_valid.sum().item()
        num_rmv = rmv_valid.sum().item()
        num_neighbors = num_add + num_rmv

        x_neighbors = x.unsqueeze(0).expand(num_neighbors, self.n).clone()

        if num_add > 0:
            add_cols = i_idx[add_valid] 
            add_rows = torch.arange(num_add, device=self.device, dtype=torch.long)
            x_neighbors[add_rows, add_cols] += weights[add_valid]

        if num_rmv > 0:
            rmv_cols = i_idx[rmv_valid]  
            rmv_rows = torch.arange(num_rmv, device=self.device, dtype=torch.long) + num_add
            x_neighbors[rmv_rows, rmv_cols] -= weights[rmv_valid]

        Fvalues, flops = self.lattice_fn.eval_neighbors(x, self.map.weights, x_neighbors)
        F_best_neighbor, best_idx = torch.min(Fvalues)
        best_neighbor = x_neighbors[best_idx]

        if self.filter_fn is not None:
            # drop neighbors whose full prompt tokenization would be unreachable from any input string
            retain_idx = self.filter_fn(x_neighbors)
            F_best_neighbor_filtered, best_idx_filtered = torch.min(Fvalues[retain_idx])
            best_neighbor_filtered = x_neighbors[retain_idx][best_idx_filtered]
        else:
            F_best_neighbor_filtered = F_best_neighbor
            best_neighbor_filtered = best_neighbor

        return F_best_neighbor.item(), best_neighbor, F_best_neighbor_filtered.item(), best_neighbor_filtered, flops 

    def subgradient_lovasz_extension(self, X: Tensor, tie_breaker: Optional[Tensor] = None):
        return subgradient_lovasz_extension(self.lattice_fn, self.map.weights, X, tie_breaker)

    # TODO: the rest of these methods are not specific to set function reductions. Move them to a set function over [n] x [b] base class
    # or as separate functions?

    def singletons_L_bound(self) -> Tuple[float, int]:  # used in pgm and DCA
        """Compute sqrt(sum_i F_set({i})^2) where i ranges over [n] x [b].

           If F_set is submodular, this is a valid bound on the Lipschitz constant
           of its Lovasz extension f_L.
        """
        # evaluate all singletons in one batched call to F_set_batch.
        rows = torch.arange(self.n * self.b, device=self.device, dtype=torch.long) // self.b
        cols = torch.arange(self.n * self.b, device=self.device, dtype=torch.long) % self.b
        rows_list = [r.view(1) for r in rows]
        cols_list = [c.view(1) for c in cols]
        singleton_vals, flops_L = self.set_fn(rows_list, cols_list)
        L = torch.linalg.vector_norm(singleton_vals.float(), ord=2).item()
        return L, flops_L

    def lovasz_extension(self, X: Tensor, subgradient: Optional[Tensor] = None) -> float:
        """Evaluate the Lovasz extension f_L of F_set at X: f_L(X)"""
        if subgradient is None:
            subgradient = self.subgradient_lovasz_extension(X)[0]
        return (X * subgradient).sum().item()

    def round_lovasz_extension(
        self, X: Optional[Tensor] = None, Fvalues: Optional[Tensor] = None, x_chain: Optional[Tensor] = None
    ) -> Tuple[float, Tensor, float, Tensor]:
        """Round X in [0,1]^n x b to a subset S_min in [n] x [b] such that F_set(S_min) <= f_L(X)
        and map to corresponding x_min = M(S_min) in V^n
        If filtering is enabled, F_min_filtered, x_min_filtered correspond to the minimum over only 
        retained x^i's in the chain. Otherwise, they're the same as F_min, x_min. 
        #TODO: remove this when done testing to avoid cost of two min?
        """
        if Fvalues is None or x_chain is None:
            assert X is not None, "X must be provided if Fvalues and x_chain are not provided"
            _, Fvalues, x_chain, _ = self.subgradient_lovasz_extension(X)

        def round(Fvals: Tensor, sols: Tensor, filter_zero: bool) -> Tuple[float, Tensor]:
            F_min, min_idx = torch.min(Fvals, dim=0)
            if F_min >= 0 and not filter_zero:  # if filter_zero is True, don't round to zero
                return 0.0, torch.zeros_like(sols[0])
            return F_min.item(), sols[min_idx]

        F_min, x_min = round(Fvalues, x_chain, False)

        if self.filter_fn is not None:
            # drop x^i's in the chain whose full prompt tokenization would be unreachable from any input string
            retain_idx = self.filter_fn(x_chain)
            F_min_filtered, x_min_filtered = round(Fvalues[retain_idx], x_chain[retain_idx], self.filter_zero)
        else:    
            F_min_filtered, x_min_filtered = F_min, x_min

        return F_min, x_min, F_min_filtered, x_min_filtered


class EneReductionMap(SetToLatticeMap):
    """Ene-Nguyen's map M: 2^([n] x [b]) -> V^n for reduction from a DR-submodular lattice function F: V^n -> R,
    where V = {0, 1,..., k - 1}, to a submodular set function F_set: 2^([n] x [b]) -> R where F_set(S) = F(M(S)).

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

        Base weights: a_1 = 1, a_i = 2^{i-2} for 2 <= i <= m+1 (indices 0...m).
        Remainder weights: a_{m+1+j} = 2^{c_j} for 1 <= j <= p, where c_j is the j-th non-zero bit 
        in the binary representation of v_max other than m. Total # of weights is b = (m + 1) + p.
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
        """Decompose each entry in x into a sum of a subset of the weights a_i's: 
        x[i,j] = sum_{c in [b]} X[i, j, c] * a_c, where X is a bool tensor of shape (batch_size, n, b).
        """
        assert x.dim() == 2 and x.shape[1] == self.n, "x must be (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"
        assert (x >= 0).all() and (x < self.k).all(), "x must have values in {0,..,self.k - 1}"
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
        lattice_fn: Union[Callable[[Tensor], Tuple[Tensor, int]], LatticeFunction],
        k: int,
        n: int,
        device: torch.device,
        filter_fn: Optional[Callable[[Tensor], Tensor]] = None,
        filter_zero: bool = False,
    ):
        ene_map = EneReductionMap(k, n, device)
        super().__init__(lattice_fn, ene_map, filter_fn, filter_zero)



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
        assert x.dtype == torch.long, "x must be of type long"
        assert (x >= 0).all() and (x < self.k).all(), "x must have values in {0,..,self.k - 1}"
        assert x.device == self.device, "x must be on the same device as the reduction"

        idx_bits = torch.arange(self.b, device=self.device, dtype=torch.long)
        x_bits = ((x.unsqueeze(-1) >> idx_bits) & 1).bool()
        return x_bits


class BinarySubmodularSetFnReduction(SetFnReduction):
    """Set function reduction using BinaryRepresentationMap
    k should be a power of 2 for F_set to be submodular.
    """
    # If k is not a power of 2, we can use k' = ceil(log2(k)) and cut off any integer >= k.
    # The resulting reduction would then only preserve DR-submodularity if F is non-decreasing (see overleaf notes)
    # Keep this for now, might use it if we decompose into non-decreasing DR-submodular functions.
    def __init__(
        self,
        lattice_fn: Union[Callable[[Tensor], Tuple[Tensor, int]], LatticeFunction],
        k: int,
        n: int,
        device: torch.device,
        filter_fn: Optional[Callable[[Tensor], Tensor]] = None,
        filter_zero: bool = False,
    ):
        binary_map = BinaryRepresentationMap(k, n, device)
        assert k == 2**binary_map.b, "k must be a power of 2"
        super().__init__(lattice_fn, binary_map, filter_fn, filter_zero)


            