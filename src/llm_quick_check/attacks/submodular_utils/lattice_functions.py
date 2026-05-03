"""
Lattice function base class and subclasses.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch
from torch import Tensor


class LatticeFunction(ABC):
    """Base class for lattice functions F: V^n -> R with batched and sequential evaluation.
    Subclasses should implement eval_batch and may override eval_chain to provide a more 
    efficient implementation for sequential evaluation.
    """
    # TODO: create a subclass for sequential evaluation using add(x, weight, index) 
    # that specific functions can then implement similar to the LatticeFct class
    # in our Matlab code
    
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
            Fvalues: Tensor of shape (m,) with Fvalues[i] = F(x^{i+1}) (F(0) not included, F assumed normalized).
            x_chain: Tensor of shape (m, n) with x_chain[i] = x^{i+1}.
            flops: flop count, int. 
        """
        x = torch.zeros(self.n, dtype=torch.long, device=rows.device)
        x_chain = torch.empty((rows.shape[0], self.n), dtype=torch.long, device=rows.device)  # (m, n)
        for i in range(rows.shape[0]):
            x[rows[i]] += weights[cols[i]]
            x_chain[i] = x

        # evaluate F(x^i) for all x^i's
        Fvalues, flops = self.eval_batch(x_chain)
        assert Fvalues.shape[0] == x_chain.shape[0], "eval_batch must return one scalar per chain step"
        return Fvalues, x_chain, flops


class CallableLatticeFunction(LatticeFunction):
    """Wrap a plain batched callable F_batch as a LatticeFunction."""

    __slots__ = ("_F_batch",)

    def __init__(self, n: int, F_batch: Callable[[Tensor], Tuple[Tensor, int]]):
        super().__init__(n)
        self._F_batch = F_batch

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        return self._F_batch(x)

class Quadratic(LatticeFunction):
    """Quadratic lattice function F(x) = x^T Q x per row (batched). Q must be symmetric."""

    def __init__(self, Q: Tensor):
        assert Q.dim() == 2 and Q.shape[0] == Q.shape[1], "Q must be square"
        assert torch.allclose(Q, Q.mT), "Q must be symmetric"
        super().__init__(int(Q.shape[0]))
        self.Q = Q

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        Q = self.Q.to(device=x.device, dtype=torch.float32)
        xf = x.float()
        qx = xf @ Q
        v = (xf * qx).sum(dim=1)
        return v, 0
