"""
Submodular optimization utilities
"""
import torch
from typing import Callable, List, Optional, Tuple
from torch import Tensor
from math import log2, inf, sqrt
from tqdm import trange
import sys
import logging
import time


# TODO: If SubmodularSetFnReduction is used,create a base class with common functions of both reduction classes (essentially everything except how to map int to binary vector; 
# done in _decomposition_mask here and simple binary representation in SubmodularSetFnReduction.ints2bitset), with the subgradient computation in it too?

# TODO: Maybe it's better to actually store the map from integers in V to sets, instead of recomputing it every time.
class EneSubmodularSetFnReduction:
    """Implement Ene-Nguyen's reduction from a DR-submodular function F: V^n -> R, where V = {0, 1,..., k - 1}, 
    to a submodular set function F_set: 2^([n] x [b]) -> R.

    @article{ene2016reduction,
        title   = {A Reduction for Optimizing Lattice Submodular Functions with Diminishing Returns},
        author  = {Alina Ene and Huy L. Nguyen},
        year    = {2016},
        journal = {arXiv preprint arXiv: 1606.08362}

    F_set(S) = F(M(S)), where M: 2^([n] x [b]) -> V^n is the map described in Lemma 1 in the paper.
    S is represented by separate rows and cols indices, i.e., S = {(rows[i], cols[i]) for i in range(rows.shape[0])}. 
    """
    def __init__(self, F_batch: Callable[[Tensor], Tuple[Tensor, int]], k: int, n: int, device: torch.device):
        self.F_batch = F_batch
        self.device = device
        self.k = k
        self.n = n
        self.v_max = self.k - 1
        self.weights = self.get_weights()
        self.F_set_batch = self._set_function_reduction() # TODO: normalize F(emptyset) = 0

    def __call__(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:
        return self.F_set_batch(rows_list, cols_list)

    def get_weights(self) -> Tensor:
        """Multiset of b weights a_1, ..., a_b summing to v_max = k-1.

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
        self.b = (m + 1) + p
        self._remainder_col = {bit: m + 1 + idx for idx, bit in enumerate(self.v_max_non_zero_bits.tolist())} # this is {c_j: m + 1 + j} in the lemma
        one = torch.tensor([1], dtype=torch.long, device=self.device)
        weights = torch.cat((one, base_weights, base_weights[self.v_max_non_zero_bits]))
        return weights

    def _decomposition_mask(self, x: Tensor) -> Tensor:
        """ Decompose each entry in x into a sum of a subset of the weights a_i's

        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.
        
        Returns:
            mask: Tensor of type bool and shape (batch_size, n, b). 
            mask[i, j, c] = True iff weight a_c appears in the decomposition of x[i, j].
        """
        assert x.device == self.device, "x must be on the same device as the reduction"
        m = self.m
        b = self.b
        v_max = self.v_max
        batch_size, n = x.shape

        is_zero = x == 0
        is_v_max = x == v_max
        non_trivial_idx = ~(is_zero | is_v_max)
        mask_x = torch.zeros(batch_size, n, b, dtype=torch.bool, device= self.device)
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
            mask_r = torch.zeros(batch_size, n, b, dtype=torch.bool, device= self.device)
            mask_r[:, :, : m + 1] = True # all base weights are included since bit m is 1 in r
            # r has same non-zero bits as v_max for bits larger than largest_diff_bits, include corresponding columns in the decomposition 
            mask_r [:, :, m + 1: ] = self.v_max_non_zero_bits.view(1, 1, -1) >= largest_diff_bits.unsqueeze(-1)

            diff_r_x = torch.where(non_trivial_idx, r - x, torch.zeros_like(x))
            diff_r_x_bits = ((diff_r_x.unsqueeze(-1) >> idx_bits[:-1]) & 1).bool() # binary representation of r - x (batch_size, n, m)
            mask_x[non_trivial_idx] = mask_r[non_trivial_idx] 
            mask_x[:, :, 1 : m + 1] &= ~diff_r_x_bits

        return mask_x


    def ints2set(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Batched version of the inverse map M^{-1}: V^n -> 2^([n] x [b])

        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            rows_list: List of 1D tensors of type long and length <= n x b
            cols_list: List of 1D tensors of type long and length <= n x b
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [b] such that M^{-1}(x[i]) = S^i.
        """ 
        assert x.dim() == 2 and x.shape[1] == self.n, "x must be (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"
        assert (x >= 0).all() and (x < self.k).all(), "x must have values in {0,..,self.k - 1}"

        mask = self._decomposition_mask(x)

        batch_idx, rows, cols = mask.nonzero(as_tuple=True)
        rows_list: List[Tensor] = []
        cols_list: List[Tensor] = []
        for i in range(mask.size(0)):
            batch_mask = batch_idx == i
            rows_list.append(rows[batch_mask].long())
            cols_list.append(cols[batch_mask].long())
        return rows_list, cols_list

    def set2ints(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tensor:
        """Batched version of map M: 2^([n] x [b]) -> V^n. Same as SubmodularSetFnReduction.bitset2ints. 
        
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

        x = torch.zeros((len(rows_list), self.n), dtype=torch.long, device=self.device)
        for i, (rows, cols) in enumerate(zip(rows_list, cols_list)):
            assert rows.shape[0] == cols.shape[0], "rows and cols must have the same length"
            if cols.numel() > 0:
                x[i].index_add_(0, rows, self.weights[cols]) # x[i, rows[j]] += weights[cols[j]] for all j
        return x

    def _set_function_reduction(self) -> Callable[[List[Tensor], List[Tensor]], Tuple[Tensor, int]]:
        #TODO: adjust implementation if bitset2int is changed
        def F_set_batch(rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:
            """Batched version of F_set: 2^([n] x [b]) -> R: Compute F_set(S^i) for the set 
            S^i = {(rows_list[i][j], cols_list[i][j]) for j in range(rows_list[i].shape[0])}.
            """
            x = self.set2ints(rows_list, cols_list)
            return self.F_batch(x)

        return F_set_batch

    def subgradient_lovasz_extension(self, X: Tensor, tie_breaker: Optional[Tensor] = None):
        return subgradient_lovasz_extension(self.F_batch, self.weights, X, tie_breaker)

    def lovasz_extension(self, X: Tensor, subgradient: Optional[Tensor] = None) -> float:
        """Evaluate the Lovasz extension f_L of F_set at X: f_L(X)"""
        if subgradient is None:
            subgradient = self.subgradient_lovasz_extension(X)[0]
        return (X * subgradient).sum().item() 

    def round_lovasz_extension(self, X: Tensor, Fvalues: Optional[Tensor] = None, x_chain: Optional[Tensor] = None)-> Tuple[float, Tensor]:
        """Round X in [0,1]^n x b to a subset S_min in [n] x [b] such that F_set(S_min) <= f_L(X) 
        and map to corresponding x_min = M(S_min) in V^n"""
        if Fvalues is None or x_chain is None:
            _, Fvalues, x_chain, _ = self.subgradient_lovasz_extension(X)
        
        F_min, min_idx = torch.min(Fvalues, dim=0)
        if F_min >= 0:
            F_min = 0
            x_min = torch.zeros_like(x_chain[0])
        else:
            F_min = F_min.item()
            x_min = x_chain[min_idx]
        return F_min, x_min




# The following reduction only works if k is a power of 2. If not, we can use k' = ceil(log2(k)) and cut off any integer >= k. 
# The resulting reduction would then only preserve DR-submodularity if F is non-decreasing (see overleaf notes)
# Keep this for now, might use it if we decompose into non-decreasing DR-submodular functions.
class SubmodularSetFnReduction:
    """Implement binary representation reduction from a DR-submodular discrete function F: V^n -> R, 
    where V = {0, 1,..., k - 1} and k = 2^b, to a submodular set function F_set: 2^([n] x [b]) -> R.

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
        self.b = int(log2(self.k))
        assert self.k == 2 ** self.b, "k must be a power of 2"
        self.weights = 1 << torch.arange(self.b, dtype=torch.long, device=self.device) # more efficient than 2**torch.arange(b)
        self.F_set_batch = self._set_function_reduction() # TODO: normalize F(emptyset) = 0
        
    def __call__(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:
        return self.F_set_batch(rows_list, cols_list)

    def bitset2ints(self, rows_list: List[Tensor], cols_list: List[Tensor]) -> Tensor:
        """Batched version of map M: 2^([n] x [b]) -> V^n.
        
        Args:
            rows_list: List of 1D tensors of type long and length <= n x b
            cols_list: List of 1D tensors of type long and length <= n x b
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [b].
        Returns:
            x: Tensor of type long and shape (batch_size, n). Each row x[i] is an integer vector in V^n such that M(S^i) = x[i].
        """
        #TODO: this doesn'b check if (row, col) pairs are unique (so true set). Add this check, 
        # or modify input to be sets of indices in [n x b] which can easily check for uniqueness before 
        # splitting into rows and cols. For now we don'b actualy use this function, so will decide depending on usage.
        #TODO: might be more efficient to take as input batch_idx, rows, cols instead, but for now will keep this 
        # simpler implementation, as I am not actually sure we'll use this for more than one set in the batch.
        # Same for int2bitset and F_set_batch.
        assert len(rows_list) == len(cols_list), "rows_list and cols_list must have the same length"
        assert all(
            rows_list[i].device == self.device and cols_list[i].device == self.device
            for i in range(len(rows_list))
        ), "all rows_list and cols_list must be on the same device"

        x = torch.zeros((len(rows_list), self.n), dtype=torch.long, device=self.device)
        for i, (rows, cols) in enumerate(zip(rows_list, cols_list)):
            assert rows.shape[0] == cols.shape[0], "rows and cols must have the same length"
            if cols.numel() > 0:
                x[i].index_add_(0, rows, self.weights[cols]) # x[i, rows[j]] += weights[cols[j]] for all j
        return x

    def ints2bitset(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Batched version of the inverse map M^{-1}: V^n -> 2^([n] x [b])

        Args:
            x: Tensor of type long and shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            rows_list: List of 1D tensors of type long and length <= n x b
            cols_list: List of 1D tensors of type long and length <= n x b
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [b] such that M^{-1}(x[i]) = S^i.
        """
        #TODO: adjust implementation if bitset2ints is changed
        assert x.dim() == 2 and x.shape[1] == self.n, "x must be (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"
        assert (x >= 0).all() and (x < self.k).all(), "x must have values in {0,..,self.k - 1}"

        idx_bits = torch.arange(self.b, dtype=torch.long, device=x.device)
        x_bits = ((x.unsqueeze(-1) >> idx_bits) & 1).bool() # binary representation of x (batch_size, n, b)
        batch_idx, rows, cols = x_bits.nonzero(as_tuple=True)
        rows_list: List[Tensor] = []
        cols_list: List[Tensor] = []
        for i in range(x_bits.size(0)):
            batch_mask = batch_idx == i
            rows_list.append(rows[batch_mask].long())
            cols_list.append(cols[batch_mask].long())
        return rows_list, cols_list

    def _set_function_reduction(self) -> Callable[[List[Tensor], List[Tensor]], Tuple[Tensor, int]]:
        #TODO: adjust implementation if bitset2ints is changed
        def F_set_batch(rows_list: List[Tensor], cols_list: List[Tensor]) -> Tuple[Tensor, int]:
            """Batched version of F_set: 2^([n] x [b]) -> R: Compute F_set(S^i) for the set 
            S^i = {(rows_list[i][j], cols_list[i][j]) for j in range(rows_list[i].shape[0])}.
            """
            x = self.bitset2ints(rows_list, cols_list)
            return self.F_batch(x)
        return F_set_batch

    def subgradient_lovasz_extension(self, X: Tensor, tie_breaker: Optional[Tensor] = None):
        return subgradient_lovasz_extension(self.F_batch, self.weights, X, tie_breaker)

    def lovasz_extension(self, X: Tensor, subgradient: Optional[Tensor] = None) -> float:
        """Evaluate the Lovasz extension f_L of F_set at X: f_L(X)"""
        if subgradient is None:
            subgradient = self.subgradient_lovasz_extension(X)[0]
        return (X * subgradient).sum().item() 

    def round_lovasz_extension(self, X: Tensor, Fvalues: Optional[Tensor] = None, x_chain: Optional[Tensor] = None)-> Tuple[float, Tensor]:
        """Round X in [0,1]^n x b to a subset S_min in [n] x [b] such that F_set(S_min) <= f_L(X) 
        and map to corresponding x_min = M(S_min) in V^n"""
        if Fvalues is None or x_chain is None:
            _, Fvalues, x_chain, _ = self.subgradient_lovasz_extension(X)
        
        F_min, min_idx = torch.min(Fvalues, dim=0)
        if F_min >= 0:
            F_min = 0
            x_min = torch.zeros_like(x_chain[0])
        else:
            F_min = F_min.item()
            x_min = x_chain[min_idx]
        return F_min, x_min



def subgradient_lovasz_extension(F_batch: Callable[[Tensor], Tuple[Tensor, int]], weights: Tensor, X: Tensor, tie_breaker: Optional[Tensor] = None): 
    """Compute a subgradient of the Lovasz extension f_L of a submodular set function F_set: 2^([n] x [b]) -> R using Edmonds' greedy algorithm.

    F_set is given by F_set(S) = F(M(S)) where M: 2^([n] x [b]) -> V^n is [M(S)]_i = \sum_{(i, j) in S} weights[j].
    Weights can be for example powers of 2 for the binary representation map or a_i's from Ene-Nguyen's reduction.
    F is assumed to be normalized, i.e., F(0) = 0.

    Args:
        F_batch: Batched version of F. It takes a batch of inputs in V^n (Tensor of shape (batch_size, n)) 
        and returns the values of F for each input (Tensor of shape (batch_size,)) and flop count (int).
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
    assert F_batch(torch.zeros(1,n, dtype=torch.long, device=X.device))[0].item() == 0, "F must be normalized"
    if tie_breaker is not None:
        assert tie_breaker.shape == X.shape, "tie_breaker must be the same shape as X"
    
    if tie_breaker is None:
        sorted_idx = torch.argsort(X.flatten(), descending=True, stable=True)
    else: 
        sorted_idx = torch.argsort(tie_breaker.flatten(), descending=True, stable=True) 
        sorted_idx = sorted_idx[torch.argsort(X.flatten()[sorted_idx], descending=True, stable=True)]

    rows, cols = torch.unravel_index(sorted_idx, X.shape) # both are (n x b,)
    # map sets S^i = {(rows[0], cols[0]), ..., (rows[i], cols[i])} to x^i in V^n and stack them in x_chain
    # more efficient than calling F_set on S^i's which would compute each x^i separately
    x = torch.zeros(n, dtype=torch.long, device=X.device)
    # no need to evaluate F(0) since F is normalized 
    x_chain = torch.empty((rows.shape[0], n), dtype=torch.long, device=X.device) # (n x b, n)
    for i in range(rows.shape[0]):
        x[rows[i]] += weights[cols[i]]
        x_chain[i] = x
    
    # compute F(x^i) for all x^i's
    Fvalues, flops = F_batch(x_chain) 
    assert Fvalues.shape[0] == n * b, "F_batch must return one scalar per input row"

    # compute subgradient g_i = F(x^i) - F(x^{i-1}), assume F(0) = 0
    subgradient = torch.zeros_like(Fvalues)  # (n * b,)
    subgradient[sorted_idx] = torch.diff(Fvalues, prepend=torch.zeros(1, dtype=Fvalues.dtype, device=Fvalues.device))
    subgradient = subgradient.view_as(X) # (n, b)

    return subgradient, Fvalues, x_chain, flops



# TODO: might be good to actually define a PGM class with step method to have standardized interface for different optimization methods
# for now let's implement it as a standalone function similar to Matlab code
# Note that this is will be mostly used for non-submodular functions. In DCA, we will use MNP as inner solver.
# TODO: if used for submodular functions, add ground set trimming and set L to upper bound sqrt(sum_i F_set(i)^2) if not provided
# TODO: allow to pass SubmodularSetFnReduction object if we keep this
def pgm_lovasz(F_set_batch: EneSubmodularSetFnReduction, X_init: Tensor, num_steps: int, L: float | str, gap_tol: Optional[float] = None):
    """Run projected subgradient method for problem min_{X in [0,1]^n x b} f_L(X)
    where f_L is the Lovasz extension of a set function reduction F_set: 2^([n] x [b]) -> R
    of a discrete function F: V^n -> R.

    Args:
        F_set_batch: EneSubmodularSetFnReduction object. 
        X_init: Initial solution in [0,1]^n x b. Tensor of shape (n, b).
        num_steps: Number of iterations.
        L: Positive float or string. Lipschitz constant of the Lovasz extension f_L. 
        If F_set is monotone, set to F_set(V), which holds even if F_set is not submodular.
        If F is submodular set to 3 max_S |F_set(S)| if known, otherwise set to 'singletons' to use sqrt(sum_i F_set(i)^2) bound.
        If F is neither, set to 'normalize' to normalize the subgradient or 'singletons' as heuristic.
        gap_tol: Stop when duality gap is less than gap_tol. 
        Use only if F_set is submodular, otherwise duality gap is not guaranteed to converge.
    Returns:
        discrete_obj_values: List of discrete objective values F(x^t) for each iteration t.
        continuous_obj_values: List of continuous objective values f_L(X^t) for each iteration t.
        duality_gaps: List of duality gaps for each iteration. Not true duality gaps if F_set is not submodular.
        discrete_sols: List of discrete solutions x^t for each iteration t.
        times: List of times for each iteration.
        flops: List of flops for each iteration. 
    """
    # TODO: add option to only store solutions that improve best objective. 
    # Keeping old doc string to reuse in this case:
    # discrete_obj_values: List of best discrete objective values min_{i <= t} F(x^i) for each iteration t.
    # continuous_obj_values: List of continuous objective values f_L(X^i*(b)) corresponding to best discrete solution 
    # x^i*(b) = argmin_{i <= t} F(x^i) for each iteration t.
    # discrete_sols: List of best discrete solution x^i*(b) for each iteration b.

    logging.info(f"Running PGM for {num_steps} iterations")
    assert X_init.dim() == 2, "X_init must be a 2D tensor"
    time_start = time.time() # include initialization time in iter 0 time

    n, b = X_init.shape
    D = sqrt(n*b) # domain diameter
    X = X_init.clone()
    # if gap_tol is not None:
    dual_avg = torch.zeros_like(X)

    flops_L = 0
    normalize = False
    if isinstance(L, str):
        if L == "singletons":
            # Set L to sqrt(sum_i F_set({i})^2) where i ranges over [n] x [b].
            # We evaluate all singletons in one batched call to F_set_batch.
            rows = torch.arange(n * b, device=X.device, dtype=torch.long) // b
            cols = torch.arange(n * b, device=X.device, dtype=torch.long) % b
            rows_list = [r.view(1) for r in rows]
            cols_list = [c.view(1) for c in cols]
            singleton_vals, flops_L = F_set_batch(rows_list, cols_list)
            L = torch.linalg.vector_norm(singleton_vals.float(), ord=2).item()
        elif L == "normalize":
            normalize = True
            L = 1.0
        else:
            raise ValueError("If L is a string, it must be either 'singletons' or 'normalize'.")
    else:
        assert L > 0, "Lipschitz constant L must be positive"

    discrete_obj_values = [0.0 for _ in range(num_steps+1)]
    continuous_obj_values = [0.0 for _ in range(num_steps+1)]
    duality_gaps = [0.0 for _ in range(num_steps+1)] # if gap_tol is not None else None
    discrete_sols = [None for _ in range(num_steps+1)]
    times = [0.0 for _ in range(num_steps+1)]
    flops = [0 for _ in range(num_steps+1)]
    best_discrete_obj = inf
    # best_continuous_obj = inf

    for iter in (pbar := trange(num_steps+1, file=sys.stdout)):
        subgradient, Fvalues, x_chain, flops_subgrad = F_set_batch.subgradient_lovasz_extension(X)
        F_round, x_round = F_set_batch.round_lovasz_extension(X, Fvalues, x_chain)
        cont_value = F_set_batch.lovasz_extension(X, subgradient)
        
        if F_round < best_discrete_obj: 
            best_discrete_obj = F_round
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
    discrete_sols = discrete_sols[:iter+1]
    times = times[:iter+1]
    flops = flops[:iter+1]
    
    return discrete_obj_values, continuous_obj_values, duality_gaps, discrete_sols, times, flops