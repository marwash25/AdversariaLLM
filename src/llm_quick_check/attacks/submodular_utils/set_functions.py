"""Base class for set functions on [n] x [b]."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .linear_combination import LinearCombinationMixin


def assert_valid_set(rows: Tensor, cols: Tensor, *, n: int, b: int, device: torch.device) -> None:
    """Validate a subset of [n] x [b] represented by paired row and column indices.

    rows and cols must be 1D long tensors of equal length on device, with indices in
    [0, n-1] and [0, b-1], respectively. Pairs must be unique. 
    Empty tensors represent the empty set. 
    """
    assert rows.dim() == cols.dim() == 1, "rows and cols must be 1D tensors"
    assert rows.dtype == cols.dtype == torch.long, "rows and cols must be of type long"
    assert rows.shape == cols.shape, "rows and cols must have the same length"
    assert rows.device == cols.device == device, "rows and cols must be on the expected device"
    assert ((rows >= 0) & (rows < n)).all(), "rows must have values in {0, ..., n - 1}"
    assert ((cols >= 0) & (cols < b)).all(), "cols must have values in {0, ..., b - 1}"
    assert (rows * b + cols).unique().numel() == rows.numel(), "(row, col) pairs must be unique"


class SetFunction(ABC):
    """Base class for set function F: 2^([n] x [b]) -> R, with input sets represented by paired row and column indices.

    Public eval_* methods are wrappers for the private _eval_* methods, with inputs and outputs validated. 
    Subclasses should implement _eval_batch and can override _eval_chain to provide a more efficient implementation.

    F is expected to be normalized, i.e., F(empty set) = 0, for use by the Lovasz utilities.
    """
    # TODO: not sure we actually need these methods in this class

    def __init__(self, n: int, b: int, device: torch.device):
        assert n > 0 and b > 0, "n and b must be positive"
        self.n = n
        self.b = b
        self.device = torch.device(device)

    def eval_single(self, rows: Tensor, cols: Tensor) -> tuple[Tensor, int]:
        """Evaluate F(S) for S = {(rows[j], cols[j]) for j in range(len(rows))}.

        Args:
            rows, cols: 1D long tensors of equal length on self.device, listing distinct
                (row, col) pairs in S. Empty tensors represent the empty set.
        Returns:
            Fvalue: 0-dimensional tensor F(S) on self.device.
            flops: flop count, int.
        """
        Fvalues, flops = self.eval_batch([rows], [cols])
        return Fvalues[0], flops

    def eval_batch(self, rows_list: list[Tensor], cols_list: list[Tensor]) -> tuple[Tensor, int]:
        """Evaluate F on a batch of sets in [n] x [b].

        An empty batch returns an empty tensor and zero flops without calling _eval_batch.

        Args:
            rows_list: List of batch_size 1D long tensors on self.device, containing row indices.
            cols_list: List of batch_size 1D long tensors on self.device, containing column indices.
            Each rows_list[i], cols_list[i] pair represents a subset S^i of [n] x [b] 
            of distinct (row, col) pairs. Empty tensors represent the empty set.
        Returns:
            Fvalues: Tensor of shape (batch_size,) on self.device with Fvalues[i] = F(S^i).
            flops: flop count, int.
        """
        assert len(rows_list) == len(cols_list), "rows_list and cols_list must have the same length"
        for rows, cols in zip(rows_list, cols_list):
            assert_valid_set(rows, cols, n=self.n, b=self.b, device=self.device)
        if not rows_list:
            return torch.empty(0, device=self.device), 0

        Fvalues, flops = self._eval_batch(rows_list, cols_list)
        assert Fvalues.shape == (len(rows_list),), "_eval_batch must return one scalar per set"
        assert Fvalues.device == self.device, "_eval_batch must return values on self.device"
        return Fvalues, flops

    @abstractmethod
    def _eval_batch(self, rows_list: list[Tensor], cols_list: list[Tensor]) -> tuple[Tensor, int]:
        """Evaluate a nonempty batch of sets validated by eval_batch"""

    def __call__(
        self, rows: Tensor | list[Tensor], cols: Tensor | list[Tensor]
    ) -> tuple[Tensor, int]:
        if isinstance(rows, Tensor) and isinstance(cols, Tensor):
            return self.eval_single(rows, cols)
        assert isinstance(rows, list) and isinstance(cols, list), "rows and cols must both be tensors or both be lists"
        return self.eval_batch(rows, cols)

    def eval_chain(self, rows: Tensor, cols: Tensor) -> tuple[Tensor, int]:
        """Evaluate F(S^i) for the chain of inputs S^i = S^{i-1} U {(rows[i-1], cols[i-1])}
           for i in [m], m length of rows/cols (empty set not included)

        Default: evaluate all S^i's via _eval_batch. Override _eval_chain for more efficient evaluation.
        An empty chain returns an empty tensor and zero flops without calling _eval_chain.

        Args:
            rows, cols: 1D long tensors of length m <= n * b on self.device, listing distinct
                elements in chain order. 

        Returns:
            Fvalues: Tensor of shape (m,) on self.device with Fvalues[i] = F(S^{i+1}) (F(empty set) not included).
            flops: flop count, int.
        """
        assert_valid_set(rows, cols, n=self.n, b=self.b, device=self.device)
        m = rows.numel()
        if m == 0:
            return torch.empty(0, device=self.device), 0

        Fvalues, flops = self._eval_chain(rows, cols)
        assert Fvalues.shape == (m,), "_eval_chain must return one scalar per chain step"
        assert Fvalues.device == self.device, "_eval_chain must return values on self.device"
        return Fvalues, flops

    def _eval_chain(self, rows: Tensor, cols: Tensor) -> tuple[Tensor, int]:
        """Evaluate validated nonempty chain of sets in one batch."""
        rows_list = [rows[:t + 1] for t in range(rows.numel())]
        cols_list = [cols[:t + 1] for t in range(cols.numel())]
        return self._eval_batch(rows_list, cols_list)

    def eval_singletons(self) -> tuple[Tensor, int]:
        """Evaluate F({(i, j)}) for all (i, j) in [n] x [b] in one batched call."""
        flat_idx = torch.arange(self.n * self.b, device=self.device, dtype=torch.long)
        rows = flat_idx // self.b
        cols = flat_idx % self.b
        rows_list = [r.view(1) for r in rows]
        cols_list = [c.view(1) for c in cols]
        singleton_vals, flops = self.eval_batch(rows_list, cols_list)
        return singleton_vals, flops

    def singletons_L_bound(self, singleton_vals: Tensor | None = None) -> tuple[float, int]:  # used in pgm and DCA
        """Compute sqrt(sum_{(i, j) in [n] x [b]} F({(i, j)})^2).

           If F is submodular, this is a valid bound on the Lipschitz constant
           of its Lovasz extension f_L.
        """
        flops_L = 0
        if singleton_vals is None:
            singleton_vals, flops_L = self.eval_singletons()
        L = torch.linalg.vector_norm(singleton_vals, ord=2).item()
        return L, flops_L

    def eval_all_pairs(self) -> tuple[Tensor, Tensor, Tensor, int]:
        """Evaluate F({v1, v2}) for all v1 = (i1, j1), v2 = (i2, j2) in [n] x [b]
        with v_1 != v_2, in one batched call.

        Returns:
            pair_vals: Tensor of shape (num_pairs,) with num_pairs = n * b * (n * b - 1) / 2 and
                pair_vals[p] = F({v1, v2}) for the p-th (v1, v2) pair.
            rows: Tensor of shape (2, num_pairs) with rows[0, p] = i1, rows[1, p] = i2.
            cols: Tensor of shape (2, num_pairs) with cols[0, p] = j1, cols[1, p] = j2.
            flops: flop count, int.
        """
        nb = self.n * self.b
        # enumerate all pairs of flattened indices v1, v2 in [n * b] with v1 < v2
        flat_pair_idx = torch.triu_indices(nb, nb, offset=1, device=self.device)  # (2, num_pairs)
        idx_v1, idx_v2 = flat_pair_idx[0], flat_pair_idx[1]

        rows_v1, cols_v1 = idx_v1 // self.b, idx_v1 % self.b
        rows_v2, cols_v2 = idx_v2 // self.b, idx_v2 % self.b

        rows_list = list(torch.stack([rows_v1, rows_v2], dim=1).unbind(0))
        cols_list = list(torch.stack([cols_v1, cols_v2], dim=1).unbind(0))
        pair_vals, flops = self.eval_batch(rows_list, cols_list)
        rows = torch.stack([rows_v1, rows_v2], dim=0)
        cols = torch.stack([cols_v1, cols_v2], dim=0)
        return pair_vals, rows, cols, flops

    # TODO: if we refactor DCA to accept any set function, add eval_neighbors
    # and get_best_neighbors methods in this class, otherwise not needed


class LinearCombinationSetFn(LinearCombinationMixin[SetFunction], SetFunction):
    """Linear combination of set functions F_i: 2^([n] x [b]) -> R: F(S) = sum_{i=1} alpha_i F_i(S)

    SetFnReduction components must use the same reduction map (same map class and k)
    self.map holds the first such map, or None if no component has one.

    Args:
        set_fn_list: list of SetFunction instances
        alphas: list of floats
    """
    def __init__(self, set_fn_list: list[SetFunction], alphas: list[float]):
        self._init_linear_comb(set_fn_list, alphas)
        n = set_fn_list[0].n
        b = set_fn_list[0].b
        device = set_fn_list[0].device
        assert all(F.n == n for F in set_fn_list), "all set functions must have the same n"
        assert all(F.b == b for F in set_fn_list), "all set functions must have the same b"
        assert all(F.device == device for F in set_fn_list), "all set functions must be on the same device"
        super().__init__(n, b, device)

        maps = [getattr(F, "map", None) for F in self.functions]
        maps = [M for M in maps if M is not None]
        self.map = maps[0] if maps else None
        # With the dimension/device checks above, matching class and k is enough
        # for currently implemented maps. In general, we'd need to check if maps' weights match.
        assert all(
            type(M) is type(self.map) and M.k == self.map.k for M in maps
        ), "All set function reductions must use the same reduction map (same map class and k)"

    def _eval_batch(self, rows_list: list[Tensor], cols_list: list[Tensor]) -> tuple[Tensor, int]:
        return self._eval_linear_comb(lambda F: F._eval_batch(rows_list, cols_list))

    def _eval_chain(self, rows: Tensor, cols: Tensor) -> tuple[Tensor, int]:
        return self._eval_linear_comb(lambda F: F._eval_chain(rows, cols))
