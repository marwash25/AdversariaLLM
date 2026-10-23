"""
Lattice function base classes
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
import torch
from torch import Tensor

from .linear_combination import LinearCombinationMixin
from .set_functions import assert_valid_set

# TODO: for now we only use flops for forward passes. Create a class for cross-entropy loss that tracks flops count
# in its state, and remove flops everywhere else

class LatticeFunction(ABC):
    """Base class for lattice function F: V^n -> R, where V = {0, 1,..., k - 1}.
    
    Supports single and batched evaluation, evaluation along chains, and evaluation of neighbors.

    Public eval_* methods are wrappers for the private _eval_* methods, with inputs and outputs validated. 
    Subclasses should implement _eval_batch and can override _eval_chain and _eval_neighbors to provide a more efficient implementation.
    """

    def __init__(self, k: int, n: int):
        assert n > 0, "n must be positive"
        assert k > 1, "k must be greater than 1"

        self.n = n
        self.k = k

    def _assert_in_Vn(self, x: Tensor) -> None:
        assert x.dtype == torch.long and (x >= 0).all() and (x < self.k).all(), "x must have values in {0, ..., k - 1}"

    def eval_single(self, x: Tensor) -> tuple[Tensor, int]:
        """Evaluate F on a single input in V^n.

        Args:
            x: Tensor of shape (n,).
        Returns:
            Fvalue: 0-dimensional tensor F(x).
            flops: flop count, int.
        """
        assert x.dim() == 1 and x.shape[0] == self.n, "x must have shape (n,)"
        vals, flops = self.eval_batch(x.unsqueeze(0))
        return vals[0], flops

    def eval_batch(self, x: Tensor) -> tuple[Tensor, int]:
        """Evaluate F on a batch of inputs in V^n.

        Args:
            x: Tensor of shape (batch_size, n). Each row is an integer vector in V^n.
        Returns:
            Fvalues: Tensor of shape (batch_size,) with Fvalues[i] = F(x[i]).
            flops: flop count, int.
        """
        assert x.dim() == 2 and x.shape[1] == self.n, "x must have shape (batch_size, n)"
        self._assert_in_Vn(x)
        Fvalues, flops = self._eval_batch(x)
        assert Fvalues.shape == (x.shape[0],), "_eval_batch must return one scalar per input row"
        assert Fvalues.device == x.device, "_eval_batch must return values on the same device as x"
        return Fvalues, flops

    @abstractmethod
    def _eval_batch(self, x: Tensor) -> tuple[Tensor, int]:
        """Core batched evaluation; x has already been validated by eval_batch."""

    def __call__(self, x: Tensor) -> tuple[Tensor, int]:
        if x.dim() == 1:
            return self.eval_single(x)
        return self.eval_batch(x)

    def eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor | None = None
    ) -> tuple[Tensor, int, Tensor]:
        """Evaluate F(x^i) for the chain of inputs x^i = x^{i-1} + weights[cols[i-1]] * e_{rows[i-1]}
           for i in [m], m length of rows/cols (zero vector not included)

        Default: construct x_chain when absent and evaluate it via _eval_batch.
        Override _eval_chain for more efficient evaluation.

        Args:
            rows, cols: 1D long tensors of length m <= n * b, with unique paired indices.
            weights: Nonnegative long tensor of shape (b,).
            x_chain: Optional long tensor of shape (m, n) with x_chain[i] = x^{i+1}.
                A supplied chain is reused without modification. The public wrapper validates
                its shape, device, dtype, and domain without reconstructing the intended prefixes.

        Returns:
            Fvalues: Tensor of shape (m,) with Fvalues[i] = F(x^{i+1}) (F(0) not included).
            flops: flop count, int.
            x_chain: Evaluated chain of shape (m, n), or the supplied tensor itself.
        """
        assert weights.dim() == 1 and weights.dtype == torch.long and (weights >= 0).all(), "weights must be a 1D nonnegative long tensor"
        assert_valid_set(rows, cols, n=self.n, b=weights.numel(), device=weights.device)
        m = rows.numel()

        if x_chain is not None:
            assert x_chain.shape == (m, self.n), "x_chain must have shape (m, n)"
            assert x_chain.device == rows.device, "x_chain must be on the same device as rows"
            self._assert_in_Vn(x_chain)
        if m == 0:
            if x_chain is None:
                x_chain = torch.empty((0, self.n), dtype=torch.long, device=rows.device)
            return torch.empty((0,), device=rows.device), 0, x_chain

        Fvalues, flops, evaluated_chain = self._eval_chain(rows, cols, weights, x_chain)
        assert Fvalues.shape == (m,), "_eval_chain must return one scalar per chain step"
        assert Fvalues.device == rows.device, "_eval_chain must return values on the same device as rows"
        if x_chain is not None:
            assert evaluated_chain is x_chain, "_eval_chain must return the supplied x_chain tensor"
        return Fvalues, flops, evaluated_chain

    def _build_chain(self, rows: Tensor, cols: Tensor, weights: Tensor) -> Tensor:
        """Construct a nonempty chain from validated indices and weights, then check if in V^n."""
        x_chain = torch.empty((rows.numel(), self.n), dtype=torch.long, device=rows.device)
        x = torch.zeros(self.n, dtype=torch.long, device=rows.device)
        for i in range(rows.numel()):
            x[rows[i]] += weights[cols[i]]
            x_chain[i] = x
        self._assert_in_Vn(x_chain)
        return x_chain

    def _eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor | None = None
    ) -> tuple[Tensor, int, Tensor]:
        """Evaluate validated nonempty chain of inputs in one batch. Construct it when absent."""
        if x_chain is None:
            x_chain = self._build_chain(rows, cols, weights)
        Fvalues, flops = self._eval_batch(x_chain)
        return Fvalues, flops, x_chain

    def _assert_eval_neighbors_inputs(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor | None
    ) -> None:
        assert x.device == weights.device, "x and weights must be on the same device"
        assert x.dim() == 1 and x.shape[0] == self.n, "x must have shape (n,)"
        self._assert_in_Vn(x)
        assert weights.dim() == 1 and weights.dtype == torch.long and (weights >= 0).all(), "weights must be a 1D nonnegative long tensor"
        if x_neighbors is not None:
            assert x_neighbors.device == x.device, "x_neighbors must be on the same device as x"
            assert x_neighbors.dim() == 2 and x_neighbors.shape[1] == self.n, "x_neighbors must have shape (num_neighbors, n)"
            self._assert_in_Vn(x_neighbors)

    def eval_neighbors(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor | None = None
    ) -> tuple[Tensor, int, Tensor]:
        """Evaluate F for all neighbors x ± weight[j] e_i of x in V^n.

        Default: construct x_neighbors when absent and evaluate via _eval_batch.
        Override _eval_neighbors for efficiency.

        Args:
            x: Tensor of shape (n,).
            weights: Nonnegative long tensor of shape (b,).
            x_neighbors: Optional long tensor of shape (num_neighbors, n), ordered as
                all valid additions followed by all valid removals, each in flattened (i,j)
                order, including duplicates. A supplied tensor is reused without modification.
        Returns:
            Fvalues: Tensor of shape (num_neighbors,) with Fvalues[i] = F(x_neighbors[i]).
            flops: flop count, int.
            x_neighbors: Evaluated neighbors of shape (num_neighbors, n), or the supplied tensor itself.
        """
        self._assert_eval_neighbors_inputs(x, weights, x_neighbors)
        Fvalues, flops, evaluated_neighbors = self._eval_neighbors(x, weights, x_neighbors)
        assert Fvalues.shape == (evaluated_neighbors.shape[0],), (
            "_eval_neighbors must return one scalar per neighbor"
        )
        assert Fvalues.device == x.device, "_eval_neighbors must return values on the same device as x"
        if x_neighbors is not None:
            assert evaluated_neighbors is x_neighbors, "_eval_neighbors must return the supplied x_neighbors tensor"
        return Fvalues, flops, evaluated_neighbors

    def _build_neighbors(self, x: Tensor, weights: Tensor) -> Tensor:
        # get neighbors of x in V^n in the order:
        # 1) all x + weights[j] e_i in V^n for all i, j
        # 2) all x - weights[j] e_i in V^n for all i, j

        # enumerate all (i, j) pairs in [n] x [b]
        b = weights.numel()
        i_idx = torch.arange(self.n, device=x.device, dtype=torch.long).repeat_interleave(b)  # (n*b,)
        j_idx = torch.arange(b, device=x.device, dtype=torch.long).repeat(self.n)  # (n*b,)
        weights = weights[j_idx]  # (n*b,)

        # check if the neighbor is in V^n
        add_valid = (x[i_idx] + weights) <= self.k - 1
        rmv_valid = (x[i_idx] - weights) >= 0
        num_add = add_valid.sum().item()
        num_rmv = rmv_valid.sum().item()
        x_neighbors = x.unsqueeze(0).expand(num_add + num_rmv, self.n).clone()

        if num_add > 0:
            add_cols = i_idx[add_valid]
            add_rows = torch.arange(num_add, device=x.device, dtype=torch.long)
            x_neighbors[add_rows, add_cols] += weights[add_valid]

        if num_rmv > 0:
            rmv_cols = i_idx[rmv_valid]
            rmv_rows = torch.arange(num_rmv, device=x.device, dtype=torch.long) + num_add
            x_neighbors[rmv_rows, rmv_cols] -= weights[rmv_valid]

        return x_neighbors

    def _eval_neighbors(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor | None = None
    ) -> tuple[Tensor, int, Tensor]:
        """Neighbor evaluation after eval_neighbors checks."""
        if x_neighbors is None:
            x_neighbors = self._build_neighbors(x, weights)
        if x_neighbors.shape[0] == 0:
            return torch.empty(0, device=x.device), 0, x_neighbors
        Fvalues, flops = self._eval_batch(x_neighbors)
        return Fvalues, flops, x_neighbors



class SequentialLatticeFunction(LatticeFunction):
    """Base class for lattice functions F: V^n -> R with incremental add / remove along one coordinate from current state.

    Overrides _eval_chain to use add and remove methods. Default add and remove methods are provided.
    Override these methods and _set_state for more efficient updates.
    """
    #TODO: refactor this class to have a state object that contains current_x and current_val which gets updated
    # when _set_state is called.

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

    def eval_update(self, x: Tensor) -> tuple[Tensor, int]:
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

    def add(self, i: int, weight: Tensor) -> tuple[Tensor, Tensor, int]:
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

    def add_update(self, i: int, weight: Tensor) -> tuple[Tensor, Tensor, int]:
        """Evaluate F(current_x + weight * e_i) and update state
        """
        new_val, new_x, flops = self.add(i, weight)
        self._set_state(new_x, new_val)
        return new_val, new_x, flops

    def remove(self, i: int, weight: Tensor) -> tuple[Tensor, Tensor, int]:
        return self.add(i, -weight)

    def remove_update(self, i: int, weight: Tensor) -> tuple[Tensor, Tensor, int]:
        return self.add_update(i, -weight)

    def _eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor | None = None
    ) -> tuple[Tensor, int, Tensor]:
        # x_chain is not used for evaluation here, but when provided return the same tensor 
        # without allocating and filling another (m, n) matrix.
        device = rows.device
        m = rows.shape[0]

        # Set state to x^1
        build_chain = x_chain is None
        if build_chain:
            first_x = torch.zeros(self.n, dtype=torch.long, device=device)
            first_x[rows[0]] = weights[cols[0]]
            x_chain = torch.empty((m, self.n), dtype=torch.long, device=device)
            x_chain[0] = first_x
        else:
            first_x = x_chain[0]
        Fx1_val, flops = self.eval_update(first_x)
        Fvalues = torch.empty((m,), dtype=Fx1_val.dtype, device=device)
        Fvalues[0] = Fx1_val
        for i in range(1, m):
            flops += self.add_update(rows[i], weights[cols[i]])[-1]
            if build_chain:
                x_chain[i] = self.current_x
            else:
                assert torch.equal(x_chain[i], self.current_x), "x_chain[i] should match updated current_x"
            Fvalues[i] = self.current_val
        return Fvalues, flops, x_chain

    def _eval_neighbors(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor | None = None
    ) -> tuple[Tensor, int, Tensor]:
        """Incremental neighbor evaluation. Expects x_neighbors in this order: all x + weights[j] e_i in V^n,
        then all x - weights[j] e_i in V^n. Collect points from add/remove when x_neighbors is absent."""
        add_valid = (x[:, None] + weights) <= self.k - 1
        rmv_valid = (x[:, None] - weights) >= 0
        num_neighbors = (add_valid.sum() + rmv_valid.sum()).item()
        build_neighbors = x_neighbors is None
        if build_neighbors:
            x_neighbors = x.new_empty((num_neighbors, self.n))
        else:
            assert x_neighbors.shape[0] == num_neighbors, "x_neighbors must contain all valid additions and removals"
        if num_neighbors == 0:
            return torch.empty(0, device=x.device), 0, x_neighbors

        # set state to x
        # TODO: add option to provide F(x) so we don't need to recompute it. Can just call self._set_state(x, F_val) in this case.
        F_val, flops = self.eval_update(x)
        Fvalues = torch.empty((x_neighbors.shape[0],), dtype=F_val.dtype, device=x.device)
        b = weights.shape[0]
        idx = 0
        for i in range(self.n):
            for j in range(b):
                if add_valid[i, j]:
                    new_val, new_x, flops_add = self.add(i, weights[j])
                    if build_neighbors:
                        x_neighbors[idx] = new_x
                    else:
                        assert torch.equal(new_x, x_neighbors[idx]), "new_x should match x_neighbors[idx]"
                    Fvalues[idx] = new_val
                    flops += flops_add
                    idx += 1
        for i in range(self.n):
            for j in range(b):
                if rmv_valid[i, j]:
                    new_val, new_x, flops_rmv = self.remove(i, weights[j])
                    if build_neighbors:
                        x_neighbors[idx] = new_x
                    else:
                        assert torch.equal(new_x, x_neighbors[idx]), "new_x should match x_neighbors[idx]"
                    Fvalues[idx] = new_val
                    flops += flops_rmv
                    idx += 1
        assert idx == x_neighbors.shape[0], "idx should match x_neighbors.shape[0]"
        return Fvalues, flops, x_neighbors



class CallableLatticeFunction(LatticeFunction):
    """Wrap a plain batched callable F_batch as a LatticeFunction."""

    __slots__ = ("_F_batch",)

    def __init__(self, k: int, n: int, F_batch: Callable[[Tensor], tuple[Tensor, int]]):
        super().__init__(k, n)
        self._F_batch = F_batch

    def _eval_batch(self, x: Tensor) -> tuple[Tensor, int]:
        return self._F_batch(x)


class LinearCombinationLatticeFn(LinearCombinationMixin[LatticeFunction], LatticeFunction):
    """Linear combination of lattice functions F_i: V^n -> R: F(x) = sum_{i=1} alpha_i F_i(x)

    Args:
        lattice_fn_list: list of LatticeFunction instances
        alphas: list of floats
    """
    def __init__(self, lattice_fn_list: list[LatticeFunction], alphas: list[float]):
        self._init_linear_comb(lattice_fn_list, alphas)
        n = lattice_fn_list[0].n
        k = lattice_fn_list[0].k
        assert all(lattice_fn.n == n for lattice_fn in lattice_fn_list), "all lattice functions must have the same n"
        assert all(lattice_fn.k == k for lattice_fn in lattice_fn_list), "all lattice functions must have the same k"
        super().__init__(k, n)

    def _eval_batch(self, x: Tensor) -> tuple[Tensor, int]:
        return self._eval_linear_comb(lambda F: F._eval_batch(x))

    def _eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor | None = None
    ) -> tuple[Tensor, int, Tensor]:
        # The first component constructs the chain; subsequent components share it.
        return self._eval_linear_comb_with_auxiliary(
            lambda F, chain: F._eval_chain(rows, cols, weights, chain), x_chain
        )

    def _eval_neighbors(
        self, x: Tensor, weights: Tensor, x_neighbors: Tensor | None = None
    ) -> tuple[Tensor, int, Tensor]:
        # The first component constructs the neighbors; subsequent components share them.
        return self._eval_linear_comb_with_auxiliary(
            lambda F, neighbors: F._eval_neighbors(x, weights, neighbors), x_neighbors
        )

def make_zero_lattice_fn(k: int, n: int, dtype: torch.dtype = torch.float32) -> LatticeFunction:
    """Return a lattice function F(x)=0 for all x in V^n."""

    def zero_F_batch(x: Tensor) -> tuple[Tensor, int]:
        return torch.zeros((x.shape[0],), device=x.device, dtype=dtype), 0

    return CallableLatticeFunction(k, n, zero_F_batch)
