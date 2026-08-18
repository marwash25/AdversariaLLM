"""
Lattice function base classes
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, List
import torch
from torch import Tensor

# TODO: for now we only use flops for forward passes. Create a class for cross-entropy loss that tracks flops count
# in its state, and remove flops everywhere else

class LatticeFunction(ABC):
    """Base class for lattice functions F: V^n -> R, with batched evaluation, evaluation along a chain of inputs,
    and evaluation of neighbors.

    Public methods eval_* validate inputs, then call _eval_*. Subclasses should implement _eval_batch and can override
    _eval_chain, _eval_neighbors to provide a more efficient implementation.
    """

    def __init__(self, k: int, n: int):
        self.n = n
        self.k = k

    def _assert_in_Vn(self, x: Tensor) -> None:
        assert x.dtype == torch.long and (x >= 0).all() and (x < self.k).all(), "x must have values in {0, ..., k - 1}"

    def eval_single(self, x: Tensor) -> Tuple[Tensor, int]:
        """Evaluate F on a single input in V^n.

        Args:
            x: Tensor of shape (n,).
        Returns:
            Fvalue: 0-dimensional tensor F(x).
            flops: flop count, int.
        """
        assert x.dim() == 1 and x.shape[0] == self.n, "x must have shape (n,)"
        self._assert_in_Vn(x)
        vals, flops = self._eval_batch(x.unsqueeze(0))
        return vals[0], flops

    def eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        """Evaluate F on a batch of inputs in V^n.

        Args:
            x: Tensor of shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            Fvalues: Tensor of shape (batch_size,) with Fvalues[i] = F(x[i]).
            flops: flop count, int.
        """
        assert x.dim() == 2 and x.shape[1] == self.n, "x must have shape (batch_size, n)"
        self._assert_in_Vn(x)
        return self._eval_batch(x)

    @abstractmethod
    def _eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        """Core batched evaluation; x has already been validated by eval_batch."""

    def __call__(self, x: Tensor) -> Tuple[Tensor, int]:
        if x.dim() == 1:
            return self.eval_single(x)
        return self.eval_batch(x)

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:
        """Evaluate F(x^i) for the chain of inputs x^i = x^{i-1} + weights[cols[i-1]] * e_{rows[i-1]}.

        Default: evaluate via _eval_batch on x_chain. Override _eval_chain for more efficient evaluation.

        Args:
            rows, cols: 1D long tensors of length m <= n * b
            weights: Tensor of shape (b,).
            x_chain: Tensor of shape (m, n) with x_chain[i] = x^{i+1}.

        Returns:
            Fvalues: Tensor of shape (m,) with Fvalues[i] = F(x^{i+1}) (F(0) not included).
            flops: flop count, int.
        """
        assert rows.device == cols.device == weights.device == x_chain.device, (
            "rows, cols, weights, and x_chain must be on the same device"
        )
        m = rows.shape[0]
        assert x_chain.shape == (m, self.n), "x_chain must have shape (m, n)"
        device = rows.device
        if m == 0:
            return torch.empty((0,), device=device), 0
        self._assert_in_Vn(x_chain)

        Fvalues, flops = self._eval_chain(rows, cols, weights, x_chain)
        assert Fvalues.shape[0] == m, "_eval_chain must return one scalar per chain step"
        return Fvalues, flops

    def _eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:
        """Chain evaluation after eval_chain checks; m > 0."""
        return self._eval_batch(x_chain)

    def _assert_eval_neighbors_inputs(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor
    ) -> None:
        assert x.device == weights.device == x_neighbors.device, (
            "x, weights, and x_neighbors must be on the same device"
        )
        assert x.dim() == 1 and x.shape[0] == self.n, "x must have shape (n,)"
        self._assert_in_Vn(x)
        assert x_neighbors.shape[1] == self.n, "x_neighbors must have shape (num_neighbors, n)"
        self._assert_in_Vn(x_neighbors)

    def eval_neighbors(self, x: Tensor, weights: Tensor, x_neighbors: Tensor) -> Tuple[Tensor, int]:
        """Evaluate F for all neighbors x ± weight[j] e_i of x in V^n.

        Default: evaluate via _eval_batch on x_neighbors. Override _eval_neighbors for efficiency.

        Args:
            x: Tensor of shape (n,).
            weights: Tensor of shape (b,).
            x_neighbors: Tensor of shape (num_neighbors, n)
        Returns:
            Fvalues: Tensor of shape (num_neighbors,) with Fvalues[i] = F(x_neighbors[i]).
            flops: flop count, int.
        """
        self._assert_eval_neighbors_inputs(x, weights, x_neighbors)
        Fvalues, flops = self._eval_neighbors(x, weights, x_neighbors)
        assert Fvalues.shape[0] == x_neighbors.shape[0], (
            "_eval_neighbors must return one scalar per neighbor"
        )
        return Fvalues, flops

    def _eval_neighbors(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor
    ) -> Tuple[Tensor, int]:
        """Neighbor evaluation after eval_neighbors checks."""
        return self._eval_batch(x_neighbors)



class SequentialLatticeFunction(LatticeFunction):
    """Base class for lattice functions F: V^n -> R with incremental add / remove along one coordinate from current state.

    Overrides _eval_chain to use add and remove methods. Default add and remove methods are provided.
    Override these methods and _set_state for more efficient updates.
    """

    def __init__(self, k: int, n: int):
        super().__init__(k, n)
        self.current_x: Tensor | None = None
        self.current_val: Tensor | None = None

    def _set_state(self, x: Tensor, F_val: Tensor):
        """Set current_x to a copy of x and current_val to F_val.

        Used by eval_update, add_update, and remove_update. Subclasses with extra
        cached fields should override this to update them.
        """
        self.current_x = x.clone()
        self.current_val = F_val

    def eval_update(self, x: Tensor) -> Tuple[Tensor, int]:
        """Evaluate F(x) and update state

        Args:
            x: Tensor of shape (n,) or (1, n), integer vector in V^n.

        Returns:
            current_val: 0-dimensional tensor, F(x).
            flops: flop count from eval_single, int.
        """
        if x.dim() == 2:
            assert x.shape[0] == 1, "if x is 2D it must have shape (1, n)"
            x = x.squeeze(0)

        val, flops = self.eval_single(x)
        self._set_state(x, val)
        assert self.current_val.device == self.current_x.device, "current_val must be on the same device as current_x"
        return self.current_val, flops

    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        """Evaluate F(current_x + weight * e_i). Don't update current state
        Default: call eval_single. Override for more efficient update.

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
        self._assert_in_Vn(new_x[i])
        new_val, flops = self.eval_single(new_x)
        return new_val, new_x, flops

    def add_update(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        """Evaluate F(current_x + weight * e_i) and update state
        """
        new_val, new_x, flops = self.add(i, weight)
        self._set_state(new_x, new_val)
        return new_val, new_x, flops

    def remove(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        return self.add(i, -weight)

    def remove_update(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        return self.add_update(i, -weight)

    def _assert_eval_neighbors_inputs(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor
    ) -> None:
        super()._assert_eval_neighbors_inputs(x, weights, x_neighbors)
        assert weights.dim() == 1, "weights must be 1D"
        assert x_neighbors.dim() == 2, "x_neighbors must be 2D"

    def _eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:
        device = rows.device
        m = rows.shape[0]

        # set state to x^1
        Fx1_val, flops = self.eval_update(x_chain[0])
        Fvalues = torch.empty((m,), dtype=Fx1_val.dtype, device=device)
        Fvalues[0] = Fx1_val
        for i in range(1, m):
            flops += self.add_update(rows[i], weights[cols[i]])[-1]
            assert torch.equal(x_chain[i], self.current_x), "x_chain[i] should match updated current_x"
            Fvalues[i] = self.current_val
        return Fvalues, flops

    def _eval_neighbors(self, x: Tensor, weights: Tensor, x_neighbors: Tensor) -> Tuple[Tensor, int]:
        """Incremental neighbor evaluation. Expects x_neighbors in this order: all x + weights[j] e_i in V^n,
        then all x - weights[j] e_i in V^n."""
        # set state to x
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

    def _eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        return self._F_batch(x)


class LinearCombinationLatticeFn(LatticeFunction):
    """Linear combination of lattice functions F_i: V^n -> R: F(x) = sum_{i=1} alpha_i F_i(x)

    Args:
        lattice_fn_list: list of LatticeFunction instances
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

    def _eval_linear_comb(
        self, eval_fn: Callable[[LatticeFunction], Tuple[Tensor, int]]
    ) -> Tuple[Tensor, int]:
        """Sum alpha_i * F_i via eval_fn(F_i), which must return (values, flops)."""
        total_flops = 0
        Fvalues = None
        for alpha, F in zip(self.alphas, self.lattice_fn_list):
            vals, flops = eval_fn(F)
            total_flops += flops
            if Fvalues is None:
                Fvalues = alpha * vals
            else:
                Fvalues += alpha * vals

        return Fvalues, total_flops

    def _eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        return self._eval_linear_comb(lambda F: F._eval_batch(x))

    def _eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:
        return self._eval_linear_comb(
            lambda F: F._eval_chain(rows, cols, weights, x_chain)
        )

    def _eval_neighbors(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor
    ) -> Tuple[Tensor, int]:
        return self._eval_linear_comb(
            lambda F: F._eval_neighbors(x, weights, x_neighbors)
        )

def make_zero_lattice_fn(k: int, n: int, dtype: torch.dtype = torch.float32) -> LatticeFunction:
    """Return a lattice function F(x)=0 for all x in V^n."""

    def zero_F_batch(x: Tensor) -> Tuple[Tensor, int]:
        return torch.zeros((x.shape[0],), device=x.device, dtype=dtype), 0

    return CallableLatticeFunction(k, n, zero_F_batch)



