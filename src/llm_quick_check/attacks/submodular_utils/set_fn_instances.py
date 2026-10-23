"""Set function instances."""

import torch
from torch import Tensor

from .set_functions import SetFunction


class ModularSetFunction(SetFunction):
    """Modular set function F(S) = sum_{(i, j) in S} W[i, j] for a weight matrix W of shape (n, b)."""

    def __init__(self, W: Tensor):
        assert W.dim() == 2, "W must have shape (n, b)"
        super().__init__(W.shape[0], W.shape[1], W.device)
        self.W = W

    def _eval_batch(self, S_batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, int]:
        Fvalues = torch.stack([self.W[rows, cols].sum() for rows, cols in S_batch])
        return Fvalues, 0

    def _eval_chain(self, order: tuple[Tensor, Tensor]) -> tuple[Tensor, int, None]:
        rows, cols = order
        return self.W[rows, cols].cumsum(0), 0, None
