"""
Lattice function base classes and instances.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Union
from .setfn_reductions import SetToLatticeMap
import torch
from torch import Tensor
import logging

# TODO: for now we only use flops for forward passes. Create a class for GCG loss that tracks flops count 
# in its state, and remove flops everywhere else

class LatticeFunction(ABC):
    """Base class for lattice functions F: V^n -> R with batched evaluation and evaluation along a chain of inputs.
    Subclasses should implement eval_batch and may override eval_chain to provide a more
    efficient implementation for sequential evaluation (see SequentialLatticeFunction).
    """
    # add asserts needed for each method in this base class and have them call private version that can be overridden by subclasses, 
    # e.g., _eval_batch and _eval_chain, to avoid having to add asserts in each subclass.
    def __init__(self, k: int, n: int):
        self.n = n
        self.k = k

    @abstractmethod
    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        """Evaluate F on a batch of inputs in V^n 
        Args:
            x: Tensor of shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            Fvalues: Tensor of shape (batch_size,) with Fvalues[i] = F(x[i]).
            flops: flop count, int. 
        """

    def __call__(self, x: Tensor) -> Tuple[Tensor, int]:
        # TODO: make this work both for single input and batch of inputs
        return self.eval_batch(x)

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:
        """Evaluate F(x^i) for the chain of inputs x^i = x^{i-1} + weights[cols[i-1]] * e_{rows[i-1]}.

        Default: call eval_batch on x_chain. Override for more efficient evaluation.

        Args:
            rows, cols: 1D long tensors of length m <= n * b 
            weights: Tensor of shape (b,).
            x_chain: Tensor of shape (m, n) with x_chain[i] = x^{i+1}.

        Returns:
            Fvalues: Tensor of shape (m,) with Fvalues[i] = F(x^{i+1}) (F(0) not included).
            flops: flop count, int. 
        """
        assert rows.device == cols.device == weights.device == x_chain.device, "rows, cols, weights, and x_chain must be on the same device"
        m = rows.shape[0]
        device = rows.device
        assert x_chain.shape == (m, self.n), "x_chain must have shape (m, n)"

        if m==0:
            return torch.empty((0,), device=device), 0
                    
        # evaluate F(x^i) for all x^i's
        Fvalues, flops = self.eval_batch(x_chain)
        assert Fvalues.shape[0] == x_chain.shape[0], "eval_batch must return one scalar per chain step"
        return Fvalues, flops
    
    def eval_neighbors(self, x: Tensor, weights: Tensor, x_neighbors: Tensor) -> Tuple[Tensor, int]:
        """Evaluate F for all neighbors x ± weight[j] e_i of x in V^n 

        Default: call eval_batch on x_neighbors. Override for more efficient evaluation.

        Args:
            x: Tensor of shape (n,).
            weights: Tensor of shape (b,).
            x_neighbors: Tensor of shape (num_neighbors, n)
        Returns:
            Fvalues: Tensor of shape (num_neighbors,) with Fvalues[i] = F(x_neighbors[i]).
            flops: flop count, int.
        """
        assert x.device == weights.device == x_neighbors.device, "x, weights, and x_neighbors must be on the same device"
        assert x.dim() == 1 and x.shape[0] == self.n, "x must have shape (n,)"
        assert x.dtype == torch.long, "x must be of type long"
        assert x_neighbors.shape[1] == self.n, "x_neighbors must have shape (num_neighbors, n)"

        Fvalues, flops = self.eval_batch(x_neighbors)
        assert Fvalues.shape[0] == x_neighbors.shape[0], "eval_batch must return one scalar per neighbor"
        return Fvalues, flops



class SequentialLatticeFunction(LatticeFunction):
    """Base class for lattice functions F: V^n -> R with incremental add / remove along one coordinate from current state.

    Overrides eval_chain to use add and remove methods. Default add and remove methods are provided. 
    Override these methods and set_state for more efficient updates.
    """
    #TODO: refactor this class to have a state object that contains current_x and current_val which gets updated 
    # when set_state is called.

    def __init__(self, k: int, n: int):
        super().__init__(k, n)
        self.current_x: Optional[Tensor] = None
        self.current_val: Optional[Tensor] = None

    def set_state(self, x: Tensor, F_val: Tensor):
        """Set current_x to a copy of x and current_val to F_val.

        Used by eval_update, add_update, and remove_update. Subclasses with extra
        cached fields should also update them.
        """
        self.current_x = x.clone()
        self.current_val = F_val

    def eval_update(self, x: Tensor) -> Tuple[Tensor, int]:
        """Evaluate F(x) and update state

        Args:
            x: Tensor of shape (n,) or (1, n), integer vector in V^n.

        Returns:
            current_val: 0-dimensional tensor, F(x).
            flops: flop count from eval_batch, int.
        """
        if x.dim() == 2:
            assert x.shape[0] == 1, "if x is 2D it must have shape (1, n)"
            x = x.squeeze(0)
        assert x.dim() == 1 and x.shape[0] == self.n, "x must have shape (n,)"
        assert x.dtype == torch.long, "x must be of type long"

        vals, flops = self.eval_batch(x.unsqueeze(0))
        self.set_state(x, vals[0])
        assert self.current_val.device == self.current_x.device, "current_val must be on the same device as current_x"
        return self.current_val, flops

    # TODO: current add/remove methods don't check if new_x is in V^n. For now that's fine since they're only used in eval_neighbors.
    # Add these checks later
    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        """Evaluate F(current_x + weight * e_i). Don't update current state
        Default: call eval_batch. Override for more efficient update.
        
        Args:
            i: coordinate index in [0, n).
            weight: 0-dimensional tensor

        Returns:
            Fvalue: 0-dimensional tensor, F(current_x + weight * e_i).
            new_x: Tensor of shape (n,), current_x + weight * e_i.
            flops: flop count for this step, int.
        """
        assert weight.device == self.current_x.device, "weight must be on the same device as current_x"
        new_x = self.current_x.clone()
        new_x[i] += weight
        new_vals, flops = self.eval_batch(new_x.unsqueeze(0))
        return new_vals[0], new_x, flops

    def add_update(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        """Evaluate F(current_x + weight * e_i) and update state
        """
        new_val, new_x, flops = self.add(i, weight)
        self.set_state(new_x, new_val)
        return new_val, new_x, flops

    def remove(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        """Evaluate F(current_x - weight * e_i). Don't update current state
        Default: call eval_batch. Override for more efficient update.
        
        Args:
            i: coordinate index in [0, n).
            weight: 0-dimensional tensor

        Returns:
            Fvalue: 0-dimensional tensor, F(current_x - weight * e_i).
            new_x: Tensor of shape (n,), current_x - weight * e_i.
            flops: flop count for this step, int.
        """
        assert weight.device == self.current_x.device, "weight must be on the same device as current_x"
        new_x = self.current_x.clone()
        new_x[i] -= weight
        new_vals, flops = self.eval_batch(new_x.unsqueeze(0))
        return new_vals[0], new_x, flops

    def remove_update(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        """Evaluate F(current_x - weight * e_i) and update state
        """
        new_val, new_x, flops = self.remove(i, weight)
        self.set_state(new_x, new_val)
        return new_val, new_x, flops

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:
        assert rows.device == cols.device == weights.device == x_chain.device, "rows, cols, weights, and x_chain must be on the same device"
        device = rows.device
        m = rows.shape[0]
        assert x_chain.shape == (m, self.n), "x_chain must have shape (m, n)"

        if m==0:
            return torch.empty((0,), device=device), 0

        # set state to x^1  
        Fx1_val, flops = self.eval_update(x_chain[0])
        Fvalues = torch.empty((m,), dtype=Fx1_val.dtype, device=device)
        Fvalues[0] = Fx1_val
        for i in range(1, m):
            flops += self.add_update(rows[i], weights[cols[i]])[-1]
            assert torch.equal(x_chain[i], self.current_x), "x_chain[i] should match updated current_x"
            Fvalues[i] = self.current_val 
        return Fvalues, flops

    def eval_neighbors(self, x: Tensor, weights: Tensor, x_neighbors: Tensor) -> Tuple[Tensor, int]:
        """Expects x_neighbors to be neighbors of x in V^n in the order:
        1) all x + weights[j] e_i in V^n for all i, j 
        2) all x - weights[j] e_i in V^n for all i, j
        """
        assert x.device == weights.device == x_neighbors.device, "x, weights, and x_neighbors must be on the same device"
        assert x.dim() == 1 and x.shape[0] == self.n, "x must have shape (n,)"
        assert x.dtype == torch.long, "x must be of type long"
        assert weights.dim() == 1, "weights must be 1D"
        assert x_neighbors.dim() == 2 and x_neighbors.shape[1] == self.n, "x_neighbors must have shape (m, n)"

        # set state to x  
        # TODO: add option to provide F(x) so we don't need to recompute it. Can just call self.set_state(x, F_val) in this case.
        F_val, flops = self.eval_update(x)
        Fvalues = torch.empty((x_neighbors.shape[0],), dtype=F_val.dtype, device=x.device)
        b = weights.shape[0]
        idx = 0
        for i in range(self.n):
            for j in range(b):
                if x[i] + weights[j] <= self.k - 1:
                    new_val, new_x, flops_add = self.add(i, weights[j])
                    assert torch.equal(new_x, x_neighbors[idx]), "new_x should match x_neighbors[idx]"
                    Fvalues[idx] = new_val
                    flops += flops_add
                    idx += 1
        for i in range(self.n):
            for j in range(b):
                if x[i] - weights[j] >= 0:
                    new_val, new_x, flops_rmv = self.remove(i, weights[j])
                    assert torch.equal(new_x, x_neighbors[idx]), "new_x should match x_neighbors[idx]"
                    Fvalues[idx] = new_val
                    flops += flops_rmv
                    idx += 1
        assert idx == x_neighbors.shape[0], "idx should match x_neighbors.shape[0]"
        return Fvalues, flops



class CallableLatticeFunction(LatticeFunction):
    """Wrap a plain batched callable F_batch as a LatticeFunction."""

    __slots__ = ("_F_batch",)

    def __init__(self, k: int, n: int, F_batch: Callable[[Tensor], Tuple[Tensor, int]]):
        super().__init__(k, n)
        self._F_batch = F_batch

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        return self._F_batch(x)


class LinearCombinationLatticeFn(LatticeFunction):
    """Linear combination of lattice functions F_i: V^n -> R: F(x) = sum_{i=1} alpha_i F_i(x)

    Args:
        lattice_fn_list: list of lattice functions or callable functions
        alphas: list of floats
    """
    def __init__(self, lattice_fn_list: List[LatticeFunction], alphas: List[float]):
        assert len(lattice_fn_list) == len(alphas), "lattice_fn_list and alphas must have the same length"
        assert len(lattice_fn_list) > 0, "lattice_fn_list must be non-empty"
        n = lattice_fn_list[0].n
        k = lattice_fn_list[0].k
        assert all(lattice_fn.n == n for lattice_fn in lattice_fn_list), "all lattice functions must have the same n"
        assert all(lattice_fn.k == k for lattice_fn in lattice_fn_list), "all lattice functions must have the same k"
        super().__init__(k, n)
        self.lattice_fn_list = lattice_fn_list
        self.alphas = alphas


    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.dim() == 2 and x.shape[1] == self.n, "x must have shape (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"

        total_flops = 0
        Fvalues = None
        for alpha, F in zip(self.alphas, self.lattice_fn_list):
            vals, flops = F.eval_batch(x)
            total_flops += flops
            if Fvalues is None:
                Fvalues = alpha * vals
            else:
                Fvalues += alpha * vals

        return Fvalues, total_flops

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:

        total_flops = 0
        Fvalues = None
        for alpha, F in zip(self.alphas, self.lattice_fn_list):
            vals, flops = F.eval_chain(rows, cols, weights, x_chain)
            total_flops += flops
            if Fvalues is None:
                Fvalues = alpha * vals
            else:
                Fvalues += alpha * vals
        return Fvalues, total_flops


class QuadraticFn(SequentialLatticeFunction):
    """Quadratic lattice function F(x) = 0.5 * x^T Q x with symmetric Q.

    Pass a 2D tensor Q for a general quadratic, or a 0-dim tensor together with n 
    for an n-by-n constant matrix Q = c * 11^T, i.e. F(x) = 0.5 * c * (sum_i x_i)^2. 
    """

    def __init__(self, Q: Tensor, k: int, n: Optional[int] = None):
        if Q.dim() == 0:
            assert n is not None and n > 0, "n is required when Q is a scalar (constant matrix c * 11^T)"
            self._sum_x = None
        elif Q.dim() == 2 and Q.shape[0] == Q.shape[1]:
            assert n is None or n == Q.shape[0], "n must match Q.shape[0] when both are given"
            assert torch.equal(Q, Q.T), "Q must be symmetric" # switch to allclose if we want to allow small numerical errors
            n = Q.shape[0]
        else:
            raise ValueError("Q must be a square matrix or a scalar tensor (constant c) with n set")
        super().__init__(k, n)
        self.Q = Q

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.device == self.Q.device, "x and Q must be on the same device"
        if self.Q.dim() == 0: 
            sum_x = x.sum(dim=1)
            Fvalues = 0.5 * self.Q * sum_x**2
        else:
            xq = x @ self.Q  
            Fvalues = 0.5 * (x * xq).sum(dim=1)
        return Fvalues, 0

    def set_state(self, x: Tensor, F_val: Tensor, *, sum_x: Optional[Tensor] = None) -> None:
        """For scalar Q, pass sum_x to set _sum_x in O(1) when it is already known."""
        super().set_state(x, F_val)
        if self.Q.dim() == 0:
            self._sum_x = sum_x if sum_x is not None else self.current_x.sum()


    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        assert weight.device == self.Q.device == self.current_x.device, "weight, current_x and Q must be on the same device"
        new_x = self.current_x.clone()
        new_x[i] += weight
        if self.Q.dim() == 0:
            assert self._sum_x is not None
            new_sum = self._sum_x + weight
            new_val = 0.5 * self.Q * new_sum**2
        else:
            new_val = self.current_val + weight * (self.Q[i, :] * self.current_x).sum() + 0.5 * weight**2 * self.Q[i, i]
        return new_val, new_x, 0

    def remove(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        assert weight.device == self.Q.device == self.current_x.device, "weight, current_x and Q must be on the same device"
        new_x = self.current_x.clone()
        new_x[i] -= weight
        if self.Q.dim() == 0:
            assert self._sum_x is not None
            new_sum = self._sum_x - weight
            new_val = 0.5 * self.Q * new_sum**2
        else:
            new_val = self.current_val - weight * (self.Q[i, :] * self.current_x).sum() + 0.5 * weight**2 * self.Q[i, i]
        return new_val, new_x, 0

    # new_sum already computed in add/remove, so it's a bit inefficient to recompute it in add_update/remove_update, 
    # but want to keep return of add/remove consistent with base class. 
    # TODO: if we refactor SequentialLatticeFunction to maintain a state object this can be avoided
    def add_update(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        new_val, new_x, flops = self.add(i, weight)
        new_sum = self._sum_x + weight if self.Q.dim() == 0 else None 
        self.set_state(new_x, new_val, sum_x=new_sum)
        return new_val, new_x, flops

    def remove_update(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        new_val, new_x, flops = self.remove(i, weight)
        new_sum = self._sum_x - weight if self.Q.dim() == 0 else None 
        self.set_state(new_x, new_val, sum_x=new_sum)
        return new_val, new_x, flops


class ModularFn(SequentialLatticeFunction): # TODO: not used anywhere yet, remove if not needed
    """Modular lattice function F(x) = w^T x."""

    def __init__(self, w: Tensor, k: int):
        super().__init__(k, w.shape[0])
        self.w = w

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.device == self.w.device, "x and w must be on the same device"
        return (x * self.w).sum(dim=1), 0

    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        assert weight.device == self.w.device == self.current_x.device, "weight, current_x and w must be on the same device"
        new_x = self.current_x.clone()
        new_x[i] += weight
        new_val = self.current_val + weight * self.w[i]
        return new_val, new_x, 0

    def remove(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        assert weight.device == self.w.device == self.current_x.device, "weight, current_x and w must be on the same device"
        new_x = self.current_x.clone()
        new_x[i] -= weight
        new_val = self.current_val - weight * self.w[i]
        return new_val, new_x, 0


class LatticeFnWithModReduction(LatticeFunction):
    """Lattice function F: V^n -> R whose set function reduction F_set: 2^([n] x [b]) -> R is modular, i.e., 
    F_set(S) = \sum_{(i, j) in S} W[i, j] for some weight matrix W of shape (n, b).
    
    F_set is given by F_set(S) = F(M(S)) where M: 2^([n] x [b]) -> V^n is [M(S)]_i = sum_{j in [b], (i, j) in S} weights[j].
    Conversely, F is given by F(x) = F_set(M^{-1}(x)) where M^{-1}: V^n -> 2^([n] x [b]) is the inverse map of M.
    """
    # This doesn't have a simple closed form that doesn't require going through M^{-1}. 
    # This function is needed in DCA for H_lowerbd which is combined with G and their set function reduction is minimized by the inner solver
    # It's inefficient to go through this lattice function when we already have the form of the set function reduction.
    # Evaluating corresponding SetFnReduction.set_fn will map from sets to ints and back to sets in eval_batch. But we currently only use 
    # this method in singleton_L_bound which is not used for this function.
    # We override eval_chain to avoid unecessary map to ints and back.
    # TODO: refactor code to have set fn class and linear combination of set fns that can be both from reductions or not.

    def __init__(self, map: SetToLatticeMap, W: Tensor):
        assert W.shape[0] == map.n and W.shape[1] == map.b, "W must have shape (map.n, map.b)"
        super().__init__(map.k, map.n)
        self.W = W
        self.map = map

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.dim() == 2 and x.shape[1] == self.n, "x must have shape (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"
        assert x.device == self.W.device == self.map.device, "x, W, and map must be on the same device"
        X = self.map.ints2binary(x)
        return (X * self.W).sum(dim=(1, 2)), 0

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:
        """Evaluate F(x^i) for the chain of inputs x^i = x^{i-1} + weights[cols[i-1]] * e_{rows[i-1]}.
        by directly evaluating F_set(S^i) for the corresponding sets S^i = S^{i-1} + {(rows[i-1], cols[i-1])}.
        """
        assert rows.device == cols.device == self.W.device, "rows, cols, and self.W must be on the same device"
        device = rows.device
        m = rows.shape[0]

        if m==0:
            return torch.empty((0,), device=device), 0

        Fvalues = torch.zeros((m,), dtype=self.W.dtype, device=device)
        Fvalues[0] = self.W[rows[0], cols[0]]
        for i in range(1,m):
            Fvalues[i] = Fvalues[i-1] + self.W[rows[i], cols[i]]
           
        return Fvalues, 0


def make_zero_lattice_fn(k: int, n: int, dtype: torch.dtype = torch.float32) -> LatticeFunction:
    """Return a lattice function F(x)=0 for all x in V^n."""

    def zero_F_batch(x: Tensor) -> Tuple[Tensor, int]:
        assert x.dim() == 2 and x.shape[1] == n, "x must have shape (batch_size, n)"
        return torch.zeros((x.shape[0],), device=x.device, dtype=dtype), 0

    return CallableLatticeFunction(k, n, zero_F_batch)

def DR_submodular_decomposition(F_batch: LatticeFunction, alpha: float, device: torch.device) -> Tuple[LatticeFunction, LatticeFunction]:
    """Decompose a lattice function F: V^n -> R into the difference of two DR-submodular lattice functions G and H: 
    F = G - H, with G = F + H and H = - alpha * H' where H' = - 0.5 * x^T J x and J is the matrix of all ones. 
    F(x + a_ie_i) - F(x) - F(x + a_ie_i + a_je_j) + F(x + a_je_j) >= \alpha for all i, j in [n] and all a_i, a_j in [0,1].
    """
    if alpha >= 0: # shouldn't happen but useful to test if dca correctly reduces to its submin inner solver in this case
        logging.info("alpha >= 0 implies F is already DR-submodular, returning F as G and zero lattice function as H")
        H_batch = make_zero_lattice_fn(F_batch.k, F_batch.n)
        return F_batch, H_batch
    
    H_batch = QuadraticFn(torch.tensor(alpha, device=device), F_batch.k, F_batch.n)
    G_batch = LinearCombinationLatticeFn([F_batch, H_batch], [1.0, 1.0])
    return G_batch, H_batch
   
