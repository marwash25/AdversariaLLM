"""
Submodular optimization utilities
"""
import torch
from typing import Callable, List, Optional, Tuple
from torch import Tensor
from math import log2



class EneSubmodularSetFnReduction:
    """Implement Ene's reduction from a DR-submodular function F: V^n -> R, where V = {0, 1,..., k - 1}, 
    to a submodular set function F_set: 2^([n] x [t]) -> R and its Lovasz extension subgradient computation.

    @article{ene2016reduction,
        title   = {A Reduction for Optimizing Lattice Submodular Functions with Diminishing Returns},
        author  = {Alina Ene and Huy L. Nguyen},
        year    = {2016},
        journal = {arXiv preprint arXiv: 1606.08362}
    """
    def __init__(self, F_batch: Callable[[Tensor], Tuple[Tensor, int]], k: int, n: int, device: torch.device):
        self.F_batch = F_batch
        self.device = device
        self.k = k
        self.n = n
        self.m = int(log2(self.k))

        


# The following reduction only works if k is a power of 2. If not, we can use k' = ceil(log2(k)) and cut off any integer >= k. 
# The resulting reduction would then only preserve DR-submodularity if F is non-decreasing (see overleaf notes)
class SubmodularSetFnReduction:
    """Implement binary representation reduction from a DR-submodular discrete function F: V^n -> R, 
    where V = {0, 1,..., k - 1} and k = 2^t, to a submodular set function F_set: 2^([n] x [t]) -> R 
    and its Lovasz extension subgradient computation.
    F_set(S) = F(M(S)), where X = J_S is the matrix with 1 at indices in S, 0
    elsewhere, and x = M(S) is the integer vector such that each x_i is the int
    with binary representation X[i, :].
    Least significant bit is at column index 0 (bit index matches column index).
    For simplicity, we represent S by separate rows and cols indices instead of
    a set of tupples, i.e., S = {(rows[i], cols[i]) for i in
    range(rows.shape(0))}. # TODO: modify this if needed
    """
    def __init__(self, F_batch: Callable[[Tensor], Tuple[Tensor, int]], k: int, n: int, device: torch.device):
        # F_batch is batched version of F. It takes a batch of inputs in V^n (Tensor of shape (batch_size, n)) 
        # and returns the values of F for each input (Tensor of shape (batch_size,)) and flop count (int).
        self.F_batch = F_batch
        self.device = device 
        self.k = k
        self.n = n
        self.t = int(log2(self.k))
        assert self.k == 2 ** self.t, "k must be a power of 2"
        self.powers = (1 << torch.arange(self.t, dtype=torch.long, device=self.device)) # more efficient than 2**torch.arange(t)
        
        self.F_set_batch = self._set_function_reduction()
        # TODO: normalize F(emptyset) = 0
        
    def __call__(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:
        return self.F_set_batch(rows_list, cols_list)

    def bitset2int(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tensor:
        """Implement batched version of map M: 2^([n] x [t]) -> V^n: Convert a list of
        subsets of [n] x [t] represented by rows and cols indices (List of 1DTensors of
        length <= n x t) to a batch of integer vectors in V^n (Tensor of shape
        (batch_size, n))."""
        #TODO: this doesn't check if (row, col) pairs are unique (so true set). Add this check, 
        # or modify input to be sets of indices in [n x t] which can easily check for uniqueness before 
        # splitting into rows and cols. For now we don't actualy use this function, so will decide depending on usage.
        #TODO: might be more efficient to take as input batch_idx, rows, cols instead, but for now will keep this 
        # simpler implementation, as I am not actually sure we'll use this for more than one set in the batch.
        # Same for int2bitset and F_set_batch.
        assert len(rows_list) == len(cols_list), "rows_list and cols_list must have the same length"
        assert all(rows_list[i].device == self.device and cols_list[i].device == self.device \
        for i in range(len(rows_list))), \
        "all rows_list and cols_list must be on the same device and have the same length"

        x = torch.zeros((len(rows_list), self.n), dtype=torch.long, device=self.device)
        for i, (rows, cols) in enumerate(zip(rows_list, cols_list)):
            assert rows.shape[0] == cols.shape[0], "rows and cols must have the same length"
            x[i].index_add_(0, rows, self.powers[cols]) # x[i, rows[j]] += powers[cols[j]] for all j
        return x

    def int2bitset(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Implement batched version of inverse map M^{-1}: V^n -> 2^([n] x [t]): 
        Convert a batch of integer vectors in V^n (Tensor of shape (batch_size, n))
        to a list of subsets of [n] x [t] represented by rows and cols indices (List of 1DTensors of
        length <= n x t))."""
        #TODO: adjust implementation if bitset2int is changed
        assert x.dim() == 2 and x.shape[1] == self.n, "x must be (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"
        idx = torch.arange(self.t, dtype=torch.long, device=x.device)
        bits = ((x.unsqueeze(-1) >> idx) & 1).bool() # (batch_size, n, t)
        batch_idx, rows, cols = bits.nonzero(as_tuple=True)
        rows_list = []
        cols_list = []
        for i in range(bits.size(0)):
            batch_mask = batch_idx == i
            rows_list.append(rows[batch_mask])
            cols_list.append(cols[batch_mask])
        return rows_list, cols_list

    def _set_function_reduction(self) -> Callable[[List[Tensor], List[Tensor]], Tuple[Tensor, int]]:
        #TODO: adjust implementation if bitset2int is changed
        def F_set_batch(rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:
            """Implement batched version of F_set: 2^([n] x [t]) -> R: Compute
            F_set(S^j) for the set S^j = {(rows_list[j][i], cols_list[j][i]) for i in
            range(rows_list[j].shape[0])}"""
            x = self.bitset2int(rows_list, cols_list)
            return self.F_batch(x)

        return F_set_batch

    def subgradient_lovasz_extension(self, X: Tensor, tie_breaker: Optional[Tensor] = None):
        """Compute a subgradient of the Lovasz extension of self.F_set using
        Edmonds' greedy algorithm.
        Args:
            X: Tensor of shape (n, t) in [0,1]^n x t
            tie_breaker: Tensor of shape (n, t) used to break ties when sorting
                X.flatten(). If not provided, original order is used.
        Returns:
            subgradient: Tensor of shape (n, t)
        """
        # TODO: for now assume x is a 2D tensor, not sure if there's a reason to vectorize it 
        if tie_breaker is None:
            sorted_idx = torch.argsort(X.flatten(), descending=True, stable=True)
        else: 
            sorted_idx = torch.argsort(tie_breaker.flatten(), descending=True, stable=True) 
            sorted_idx = sorted_idx[torch.argsort(X.flatten()[sorted_idx], descending=True, stable=True)]

        rows, cols = torch.unravel_index(sorted_idx, X.shape)
        # map sets S^i = {(rows[1], cols[1]), ..., (rows[i], cols[i])} to x^i in V^n and stack them in x_chain
        # more efficient than calling F_set on S^i's which would compute each x^i separately
        x = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        # no need to evaluate F(0) since F is normalized #TODO: normalize earlier
        x_chain = torch.empty((rows.shape[0], X.shape[0]), dtype=torch.long, device=X.device) # (n x t, n)
        for i in range(rows.shape[0]):
            x[rows[i]] += self.powers[cols[i]]
            x_chain[i] = x
        
        # compute F(x^i) for all x^i's
        Fvalues, _ = self.F_batch(x_chain) # TODO: add flop count handling here

        # compute subgradient g_i = F(x^i) - F(x^{i-1}), assume F(0) = 0
        subgradient = torch.zeros_like(Fvalues) # (n x t,)
        subgradient[sorted_idx] = torch.diff(Fvalues, prepend=torch.zeros_like(Fvalues[0]))
        subgradient = subgradient.view_as(X) # (n, t)

        return subgradient, Fvalues, sorted_idx 



 