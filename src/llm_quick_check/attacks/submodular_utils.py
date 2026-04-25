"""
Submodular optimization utilities
"""
import torch
from typing import Callable, List, Optional, Tuple
from torch import Tensor
from math import log2


#TODO: Maybe it's better to actually store the map from integers in V to sets, instead of recomputing it every time.
class EneSubmodularSetFnReduction:
    """Implement Ene-Nguyen's reduction from a DR-submodular function F: V^n -> R, where V = {0, 1,..., k - 1}, 
    to a submodular set function F_set: 2^([n] x [t]) -> R and its Lovasz extension subgradient computation.

    @article{ene2016reduction,
        title   = {A Reduction for Optimizing Lattice Submodular Functions with Diminishing Returns},
        author  = {Alina Ene and Huy L. Nguyen},
        year    = {2016},
        journal = {arXiv preprint arXiv: 1606.08362}

    F_set(S) = F(M(S)), where M: 2^([n] x [t]) -> V^n is the map described in Lemma 1 in the paper.
    S is represented by separate rows and cols indices instead of a set of tuples, i.e., S = {(rows[i], cols[i]) for i in
    range(rows.shape[0])}. # TODO: modify this if needed
    """
    def __init__(self, F_batch: Callable[[Tensor], Tuple[Tensor, int]], k: int, n: int, device: torch.device):
        self.F_batch = F_batch
        self.device = device
        self.k = k
        self.n = n
        self.v_max = self.k - 1
        self.weights = self.get_weights()

    def get_weights(self) -> Tensor:
        """Lemma 1 (Ene–Nguyen): multiset of t weights a_i summing to v_max = k-1.

        Base weights: a_1 = 1, a_i = 2^{i-2} for 2 <= i <= m+1 (indices 0...m).
        Remainder weights: a_{m+1+j} = 2^{c_j} for 1 <= j <= p, where c_j is the j-th non-zero bit in the binary representation of v_max other than m.
        """
        assert self.k > 1, "k must be greater than 1"
        
        m = self.v_max.bit_length() - 1
        self.m = m
        idx_bits = torch.arange(m + 1, device=self.device, dtype=torch.long)
        self.v_max_bits = ((self.v_max >> idx_bits) & 1).bool() # binary representation of v_max
        idx_bits = idx_bits[:-1] # drop msb bit
        base_weights = (1 << idx_bits) # 2^0, ..., 2^{m-1}
        self.v_max_non_zero_bits = idx_bits[self.v_max_bits[:-1]]# non-zero bits of v_max other than m
        p = len(self.v_max_non_zero_bits)
        self.t = (m + 1) + p
        self._remainder_col = {bit: m + 1 + idx for idx, bit in enumerate(self.v_max_non_zero_bits.tolist())} # this is {c_j: m + 1 + j} in the lemma
        one = torch.tensor([1], dtype=torch.long, device=self.device)
        weights = torch.cat((one, base_weights, base_weights[self.v_max_non_zero_bits]))
        return weights

    def _decomposition_mask(self, x: Tensor) -> Tensor:
        """Lemma 1 subset mask: mask[b, i, c] iff weight column c appears in the decomposition of x[b, i].

        x must be (batch_size, n), long, on self.device.
        """
        assert x.device == self.device, "x must be on the same device as the reduction"
        m = self.m
        t = self.t
        v_max = self.v_max
        batch_size, n = x.shape

        is_zero = x == 0
        is_v_max = x == v_max
        non_trivial_idx = ~(is_zero | is_v_max)
        mask_x = torch.zeros(batch_size, n, t, dtype=torch.bool, device= self.device)
        mask_x[is_v_max] = True

        if m > 0:
            idx_bits = torch.arange(m + 1, device=self.device, dtype=torch.long)
            x_bits = ((x.unsqueeze(-1) >> idx_bits) & 1).bool() # binary representation of x (batch_size, n, m + 1)
            diff_bits = self.v_max_bits.view(1, 1, m + 1) & ~x_bits # bits that are 1 for v_max but 0 for x[b, i]'s
            largest_diff_bits= torch.where(
                diff_bits,
                idx_bits.view(1, 1, m + 1).expand(batch_size, n, -1),
                -1, # if no diff bits are found, set to -1 (happens only for x[b, i] = v_max)
            ).max(dim=-1).values

            # r is the number that agrees with v_max on all bits larger than the largest_diff_bits and is 0 on smaller bits
            r = torch.zeros_like(x)
            r[non_trivial_idx] = (v_max >> largest_diff_bits[non_trivial_idx]) << largest_diff_bits[non_trivial_idx]

            # get decomposition mask for r
            mask_r = torch.zeros(batch_size, n, t, dtype=torch.bool, device= self.device)
            mask_r[:, :, : m + 1] = True # all base weights are included since bit m is 1 in r
            # r has same non-zero bits as v_max for bits larger than largest_diff_bits, include corresponding columns in the decomposition 
            mask_r [:, :, m + 1: ] = self.v_max_non_zero_bits.view(1, 1, -1) >= largest_diff_bits.unsqueeze(-1)

            diff_r_x = torch.where(non_trivial_idx, r - x, torch.zeros_like(x))
            diff_r_x_bits = ((diff_r_x.unsqueeze(-1) >> idx_bits[:-1]) & 1).bool() # binary representation of r - x (batch_size, n, m)
            mask_x[non_trivial_idx] = mask_r[non_trivial_idx] 
            mask_x[:, :, 1 : m + 1] &= ~diff_r_x_bits

        return mask_x

    def _column_indices_for_q(self, q: int, out_device: torch.device) -> Tensor:
        """Subset of column indices whose weights sum to q (0 <= q <= k-1)."""
        m = self._decomposition_mask(torch.tensor([[q]], dtype=torch.long, device=out_device))[0, 0]
        return m.nonzero(as_tuple=True)[0]

    def ints2set(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Batched version of the inverse map M^{-1}: V^n -> 2^([n] x [t])

        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            rows_list: List of 1D tensors of type long and length <= n x t
            cols_list: List of 1D tensors of type long and length <= n x t
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [t] such that M^{-1}(x[i]) = S^i.
        """ 
        assert x.dim() == 2 and x.shape[1] == self.n, "x must be (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"
        assert (x >= 0).all() and (x < self.k).all(), "x must have values in {0,..,self.k - 1}"

        batch_size = x.shape[0]
        mask = self._decomposition_mask(x)
        if not mask.any():
            empty = torch.empty(0, dtype=torch.long, device=x.device)
            return [empty.clone() for _ in range(batch_size)], [empty.clone() for _ in range(batch_size)]

        batch_idx, rows, cols = mask.nonzero(as_tuple=True)
        rows_list: List[Tensor] = []
        cols_list: List[Tensor] = []
        for b in range(batch_size):
            m_b = batch_idx == b
            rows_list.append(rows[m_b])
            cols_list.append(cols[m_b].long())
        return rows_list, cols_list

    def set2ints(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tensor:
        """Batched M: subsets of [n] x [t] -> V^n (same pattern as SubmodularSetFnReduction.bitset2int)."""
        assert len(rows_list) == len(cols_list), "rows_list and cols_list must have the same length"
        assert all(
            rows_list[i].device == self.device and cols_list[i].device == self.device
            for i in range(len(rows_list))
        ), "all rows_list and cols_list must be on the same device"
        x = torch.zeros((len(rows_list), self.n), dtype=torch.long, device=self.device)
        for i, (rows, cols) in enumerate(zip(rows_list, cols_list)):
            assert rows.shape[0] == cols.shape[0], "rows and cols must have the same length"
            if cols.numel() > 0:
                x[i].index_add_(0, rows, self.weights[cols])
        return x

    

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
    S is represented by separate rows and cols indices instead of a set of tuples, i.e., S = {(rows[i], cols[i]) for i in
    range(rows.shape[0])}. # TODO: modify this if needed
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

    def bitset2ints(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tensor:
        """Batched version of map M: 2^([n] x [t]) -> V^n.
        
        Args:
            rows_list: List of 1D tensors of type long and length <= n x t
            cols_list: List of 1D tensors of type long and length <= n x t
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [t].
        Returns:
            x: Tensor of type long and shape (batch_size, n). Each row x[i] is an integer vector in V^n such that M(S^i) = x[i].
        """
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

    def ints2bitset(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Batched version of the inverse map M^{-1}: V^n -> 2^([n] x [t])

        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            rows_list: List of 1D tensors of type long and length <= n x t
            cols_list: List of 1D tensors of type long and length <= n x t
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [t] such that M^{-1}(x[i]) = S^i.
        """
        #TODO: adjust implementation if bitset2int is changed
        assert x.dim() == 2 and x.shape[1] == self.n, "x must be (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"
        idx_bits = torch.arange(self.t, dtype=torch.long, device=x.device)
        x_bits = ((x.unsqueeze(-1) >> idx_bits) & 1).bool() # (batch_size, n, t)
        batch_idx, rows, cols = x_bits.nonzero(as_tuple=True)
        rows_list: List[Tensor] = []
        cols_list: List[Tensor] = []
        for i in range(x_bits.size(0)):
            batch_mask = batch_idx == i
            rows_list.append(rows[batch_mask])
            cols_list.append(cols[batch_mask])
        return rows_list, cols_list

    def _set_function_reduction(self) -> Callable[[List[Tensor], List[Tensor]], Tuple[Tensor, int]]:
        #TODO: adjust implementation if bitset2int is changed
        def F_set_batch(rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:
            """Batched version of F_set: 2^([n] x [t]) -> R: Compute F_set(S^i) for the set 
            S^i = {(rows_list[i][j], cols_list[i][j]) for j in range(rows_list[i].shape[0])}.
            """
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



 