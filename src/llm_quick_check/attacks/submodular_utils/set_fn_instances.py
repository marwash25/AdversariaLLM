"""Set function instances."""

from math import sqrt

import torch
from torch import Tensor

from .set_functions import LinearCombinationSetFn, SetFunction


class ModularSetFn(SetFunction):
    """Modular set function F(S) = sum_{(i, j) in S} W[i, j] for a weight matrix W of shape (n, b)."""

    def __init__(self, W: Tensor):
        assert W.dim() == 2, "W must have shape (n, b)"
        super().__init__(
            W.shape[0], W.shape[1], W.device, is_submodular=True,
            is_non_decreasing=bool((W >= 0).all()),
            is_non_increasing=bool((W <= 0).all()),
        )
        self.W = W

    def lovasz_L_bound(self) -> tuple[float, int]:
        """Return the cached exact Lipschitz constant of f_L = ||W||_2 (l2 vector norm)."""
        if self._lovasz_L_bound is None:
            self._lovasz_L_bound = torch.linalg.vector_norm(self.W).item()
        return self._lovasz_L_bound, 0

    def _eval_batch(self, S_batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, int]:
        Fvalues = torch.stack([self.W[rows, cols].sum() for rows, cols in S_batch])
        return Fvalues, 0

    def _eval_chain(self, order: tuple[Tensor, Tensor]) -> tuple[Tensor, int, None]:
        rows, cols = order
        return self.W[rows, cols].cumsum(0), 0, None

    def _eval_singletons(self) -> tuple[Tensor, int]:
        return self.W, 0

    def _eval_singletons_removal_marginals(self) -> tuple[Tensor, int]:
        return self.W, 0

class SquaredCardinalitySetFn(SetFunction):
    """Squared cardinality function F(S) = 0.5 * alpha * |S|^2."""

    def __init__(self, n: int, b: int, alpha: Tensor):
        super().__init__(
            n, b, alpha.device, is_submodular=bool(alpha <= 0),
            is_non_decreasing=bool(alpha >= 0), is_non_increasing=bool(alpha <= 0),
        )
        self.alpha = alpha

    def lovasz_L_bound(self) -> tuple[float, int]:
        """Return the cached exact bound |alpha| sqrt(m(4m^2 - 1)/12), where m = n*b.

        Gradients of f_L are permutations of g such that g_j = alpha * (j - 1/2) for j in [m]
        """
        if self._lovasz_L_bound is None:
            m = self.n * self.b
            self._lovasz_L_bound = abs(self.alpha.item()) * sqrt(m * (4 * m * m - 1) / 12)
        return self._lovasz_L_bound, 0

    def _eval_batch(self, S_batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, int]:
        cardinalities = self.alpha.new_tensor([rows.numel() for rows, _ in S_batch])
        return 0.5 * self.alpha * cardinalities.square(), 0

    def _eval_chain(self, order: tuple[Tensor, Tensor]) -> tuple[Tensor, int, None]:
        cardinalities = torch.arange(
            1, order[0].numel() + 1, dtype=self.alpha.dtype, device=self.device
        )
        return 0.5 * self.alpha * cardinalities.square(), 0, None

    def _eval_singletons(self) -> tuple[Tensor, int]:
        return (0.5 * self.alpha).expand(self.n, self.b), 0

    def _eval_singletons_removal_marginals(self) -> tuple[Tensor, int]:
        return (self.alpha * (self.n * self.b - 0.5)).expand(self.n, self.b), 0

def setfn_submodular_decomp(
    F: SetFunction,
    alpha: Tensor
) -> tuple[SetFunction, SetFunction]:
    """Decompose a set function F: 2^([n] x [b]) -> R into the difference of two submodular set functions G and H:
        
        F = G - H, with G = F + H and H(x) = 0.5 (alpha)_- |S|^2 

    where 
        F(S ∪ {v1}) - F(S) - F(S ∪ {v1,v2}) + F(S ∪ {v2}) >= alpha for all S, and distrinct v_1, v_2 not in S
    """
    assert alpha.dim() == 0, "alpha must be a scalar tensor"
    assert alpha.device == F.device, "alpha and F must be on the same device"
    alpha = torch.clamp(alpha, max=0)

    H = SquaredCardinalitySetFn(F.n, F.b, alpha)
    G = LinearCombinationSetFn([F, H], [1.0, 1.0], is_submodular=True)
    return G, H
