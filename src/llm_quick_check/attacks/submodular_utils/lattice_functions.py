"""
Lattice function base classes and instances.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List, Union

import torch
from torch import Tensor

# TODO: for now we only use flops for forward passes. Create a class for GCG loss that tracks flops count 
# in its state, and remove flops everywhere else

class LatticeFunction(ABC):
    """Base class for lattice functions F: V^n -> R with batched evaluation and evaluation along a chain of inputs.
    Subclasses should implement eval_batch and may override eval_chain to provide a more
    efficient implementation for sequential evaluation (see SequentialLatticeFunction).
    """

    def __init__(self, n: int):
        self.n = n

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
        return self.eval_batch(x)

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        """Evaluate F(x^i) for the chain of inputs x^i = x^{i-1} + weights[cols[i]] * e_{rows[i]}.

        Default: materialize x_chain (x^i's stacked as rows) and call eval_batch. 

        Args:
            rows, cols: 1D long tensors of length m <= n * b 
            weights: Tensor of shape (b,).

        Returns:
            Fvalues: Tensor of shape (m,) with Fvalues[i] = F(x^{i+1}) (F(0) not included).
            x_chain: Tensor of shape (m, n) with x_chain[i] = x^{i+1}.
            flops: flop count, int. 
        """
        assert rows.device == cols.device == weights.device, "rows, cols, and weights must be on the same device"

        m = rows.shape[0]
        device = rows.device
        x_chain = torch.empty((m, self.n), dtype=torch.long, device=device)  # (m, n)
        if m==0:
            return torch.empty((0,), device=device), x_chain, 0
        
        x = torch.zeros(self.n, dtype=torch.long, device=device)
        for i in range(m):
            x[rows[i]] += weights[cols[i]]
            x_chain[i] = x

        # evaluate F(x^i) for all x^i's
        Fvalues, flops = self.eval_batch(x_chain)
        assert Fvalues.shape[0] == x_chain.shape[0], "eval_batch must return one scalar per chain step"
        return Fvalues, x_chain, flops
    
    # TODO: add a eval_neighbors method that evaluates F(x + weight[j] e_i) and F(x - weight[j] e_i) for all i in [n] and j in [b]
    # needed for local search in DCA again with batched and sequential evaluation


class SequentialLatticeFunction(LatticeFunction):
    """Base class for lattice functions F: V^n -> R with incremental add / remove along one coordinate from current state.

    Overrides eval_chain to use add and remove methods. Default add and remove methods are provided. 
    Override these methods for more efficient updates.
    """

    def __init__(self, n: int):
        super().__init__(n)
        self.current_x = None
        self.current_val = None

    def set_state(self, x: Tensor) -> Tuple[Tensor, int]:
        """Set current_x to a copy of x and current_val to F(x) using eval_batch.

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
        self.current_x = x.clone()
        self.current_val = vals[0]
        assert self.current_val.device == self.current_x.device, "current_val must be on the same device as current_x"
        return self.current_val, flops


    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, int]:
        """Increment current_x[i] by weight, and update current_val to F at the new x
        Default: call eval_batch. Override for more efficient update.
        
        Args:
            i: coordinate index in [0, n).
            weight: 0-dimensional tensor

        Returns:
            Fvalue: F(current_x + weight * e_i), float.
            Flops: flop count for this step, int.
        """
        assert weight.device == self.current_x.device, "weight must be on the same device as current_x"
        self.current_x[i] += weight
        vals, flops = self.eval_batch(self.current_x.unsqueeze(0))
        self.current_val = vals[0]
        return self.current_val, flops

    def remove(self, i: int, weight: Tensor) -> Tuple[Tensor, int]:
        """Decrement current_x[i] by weight, and update current_val to F at the new x
        Default: call eval_batch. Override for more efficient update.
        
        Args:
            i: coordinate index in [0, n).
            weight: 0-dimensional tensor

        Returns:
            Fvalue: F(current_x - weight * e_i), float.
            Flops: flop count for this step, int.
        """
        assert weight.device == self.current_x.device, "weight must be on the same device as current_x"
        self.current_x[i] -= weight
        vals, flops = self.eval_batch(self.current_x.unsqueeze(0))
        self.current_val = vals[0]
        return self.current_val, flops

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        assert rows.device == cols.device == weights.device, "rows, cols, and weights must be on the same device"
        device = rows.device
        m = rows.shape[0]
        x_chain = torch.empty((m, self.n), dtype=torch.long, device=device)
        if m==0:
            return torch.empty((0,), device=device), x_chain, 0

        # set state to x^1 
        x_chain[0] = torch.zeros(self.n, dtype=torch.long, device=device)
        x_chain[0][rows[0]] = weights[cols[0]] 
        Fx1_val, flops = self.set_state(x_chain[0])
        Fvalues = torch.empty((m,), dtype=Fx1_val.dtype, device=device)
        Fvalues[0] = Fx1_val
        for i in range(1, m):
            flops += self.add(rows[i], weights[cols[i]])[1]
            x_chain[i] = self.current_x
            Fvalues[i] = self.current_val
        return Fvalues, x_chain, flops


class CallableLatticeFunction(LatticeFunction):
    """Wrap a plain batched callable F_batch as a LatticeFunction."""

    __slots__ = ("_F_batch",)

    def __init__(self, n: int, F_batch: Callable[[Tensor], Tuple[Tensor, int]]):
        super().__init__(n)
        self._F_batch = F_batch

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        return self._F_batch(x)


class SetFnLinearCombination(LatticeFunction):
    """Linear combination of lattice functions F_i: V^n -> R: F(x) = sum_{i=1} alpha_i F_i(x)

    Args:
        n: int, dimension of the lattice
        F_batch_list: list of lattice functions or callable functions
        alphas: list of floats
    """
    def __init__(self, n: int, F_batch_list: List[Union[Callable[[Tensor], Tuple[Tensor, int]], LatticeFunction]], alphas: List[float]):
        super().__init__(n)
        assert len(F_batch_list) == len(alphas), "F_batch_list and alphas must have the same length"
        assert len(F_batch_list) > 0, "F_batch_list must be non-empty"
        self.F_batch_list = [F_batch if isinstance(F_batch, LatticeFunction)
            else CallableLatticeFunction(n, F_batch) for F_batch in F_batch_list]
        self.alphas = alphas
        self.seq_F_idx = [i for i, F in enumerate(self.F_batch_list) if isinstance(F, SequentialLatticeFunction)]
        self.nonseq_F_idx = [i for i, F in enumerate(self.F_batch_list) if not isinstance(F, SequentialLatticeFunction)]

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.dim() == 2 and x.shape[1] == self.n, "x must have shape (batch_size, n)"
        assert x.dtype == torch.long, "x must be of type long"

        total_flops = 0
        Fvalues = None
        for alpha, F in zip(self.alphas, self.F_batch_list):
            vals, flops = F.eval_batch(x)
            total_flops += flops
            if Fvalues is None:
                Fvalues = alpha * vals
            else:
                Fvalues += alpha * vals

        return Fvalues, total_flops

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        assert rows.device == cols.device == weights.device, "rows, cols, and weights must be on the same device"
        device = rows.device
        m = rows.shape[0]
        
        x_chain = torch.empty((m, self.n), dtype=torch.long, device=device)  # (m, n)
        if m == 0:
            return torch.empty((0,), device=device), x_chain, 0

        # build x_chain 
        x = torch.zeros(self.n, dtype=torch.long, device=device)
        for i in range(m):
            x[rows[i]] += weights[cols[i]]
            x_chain[i] = x

        # evaluate non-sequential Fns
        # we don't use their eval_chain to avoid building x_chain several times
        nonseq_flops = 0
        nonseq_Fvalues = None
        for F_idx in self.nonseq_F_idx:
            F, alpha = self.F_batch_list[F_idx], self.alphas[F_idx]
            vals, flops = F.eval_batch(x_chain)
            nonseq_flops += flops
            if nonseq_Fvalues is None:
                nonseq_Fvalues = alpha * vals
            else:
                nonseq_Fvalues += alpha * vals

        # evaluate sequential Fns
        # we can call eval_chain of each seq fn. This is slightly more efficient
        seq_flops = 0
        seq_Fvalues = None
        for F_idx in self.seq_F_idx:
            F, alpha = self.F_batch_list[F_idx], self.alphas[F_idx]
            # set state to x^1
            Fx1_val, flops = F.set_state(x_chain[0])
            seq_flops += flops
            if seq_Fvalues is None:
                seq_Fvalues = torch.zeros((m,), dtype=Fx1_val.dtype, device=device)
            seq_Fvalues[0] += alpha * Fx1_val
            
            for i in range(1, m):
                seq_flops += F.add(rows[i], weights[cols[i]])[1]
                seq_Fvalues[i] += alpha * F.current_val 

        if nonseq_Fvalues is None:
            Fvalues = seq_Fvalues
        elif seq_Fvalues is None:
            Fvalues = nonseq_Fvalues
        else:
            Fvalues = nonseq_Fvalues + seq_Fvalues
        total_flops = nonseq_flops + seq_flops

        return Fvalues, x_chain, total_flops


class QuadraticFn(SequentialLatticeFunction):
    """Quadratic lattice function F(x) = 0.5 * x^T Q x with symmetric Q.

    Pass a 2D tensor Q for a general quadratic, or a 0-dim tensor together with n 
    for an n-by-n constant matrix Q = c * 11^T, i.e. F(x) = 0.5 * c * (sum_i x_i)^2. 
    """

    def __init__(self, Q: Tensor, n: Optional[int] = None):
        if Q.dim() == 0:
            assert n is not None and n > 0, "n is required when Q is a scalar (constant matrix c * 11^T)"
            self._sum_x = None
        elif Q.dim() == 2 and Q.shape[0] == Q.shape[1]:
            assert n is None or n == Q.shape[0], "n must match Q.shape[0] when both are given"
            assert torch.equal(Q, Q.T), "Q must be symmetric" # switch to allclose if we want to allow small numerical errors
            n = Q.shape[0]
        else:
            raise ValueError("Q must be a square matrix or a scalar tensor (constant c) with n set")
        super().__init__(n)
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

    def set_state(self, x: Tensor) -> Tuple[Tensor, int]:
        val, flops = super().set_state(x)
        if self.Q.dim() == 0:
            self._sum_x = self.current_x.sum()
        return val, flops

    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, int]:  
        assert weight.device == self.Q.device == self.current_x.device, "weight, current_x and Q must be on the same device"
        if self.Q.dim() == 0:
            assert self._sum_x is not None
            self.current_x[i] += weight
            self._sum_x += weight
            self.current_val = 0.5 * self.Q * self._sum_x**2 
            # alternatively: self.current_val += self.Q * (weight * sum_x_old + 0.5 * weight^2) 
        else: 
            self.current_val += weight * (self.Q[i, :] * self.current_x).sum() + 0.5 * weight**2 * self.Q[i, i]
            self.current_x[i] += weight
        return self.current_val, 0

    def remove(self, i: int, weight: Tensor) -> Tuple[Tensor, int]:
        assert weight.device == self.Q.device == self.current_x.device, "weight, current_x and Q must be on the same device"
        if self.Q.dim() == 0:
            assert self._sum_x is not None
            self.current_x[i] -= weight
            self._sum_x -= weight
            self.current_val = 0.5 * self.Q * self._sum_x**2 
            # alternatively: self.current_val -= self.Q * (weight * sum_x_old + 0.5 * weight^2) 
        else: 
            self.current_val -= (weight * (self.Q[i, :] * self.current_x).sum() - 0.5 * weight**2 * self.Q[i, i])
            self.current_x[i] -= weight
        return self.current_val, 0


class ModularFn(SequentialLatticeFunction):
    """Modular lattice function F(x) = w^T x."""

    def __init__(self, w: Tensor):
        super().__init__(w.shape[0])
        self.w = w

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.device == self.w.device, "x and w must be on the same device"
        return (x * self.w).sum(dim=1), 0

    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, int]:
        assert weight.device == self.w.device == self.current_x.device, "weight, current_x and w must be on the same device"
        self.current_x[i] += weight
        self.current_val += weight * self.w[i]
        return self.current_val, 0

    def remove(self, i: int, weight: Tensor) -> Tuple[Tensor, int]:
        assert weight.device == self.w.device == self.current_x.device, "weight, current_x and w must be on the same device"
        self.current_x[i] -= weight
        self.current_val -= weight * self.w[i]
        return self.current_val, 0


