"""Shared evaluation for linear combinations of functions."""

from collections.abc import Callable
from typing import Generic, TypeVar

from torch import Tensor


Fn = TypeVar("Fn")


class LinearCombinationMixin(Generic[Fn]):
    """Linear combination of functions F_i: F = sum_{i=1} alpha_i F_i
    
    Store component functions and coefficients, and combine their evaluation results.
    Subclasses call _init_linear_comb in their constructor, validate domain compatibility,
    and provide typed evaluation methods that delegate to _eval_linear_comb.
    """

    def _init_linear_comb(self, functions: list[Fn], alphas: list[float]) -> None:
        assert len(functions) == len(alphas), "functions and alphas must have the same length"
        assert len(functions) > 0, "functions must be non-empty"
        self.functions = functions
        self.alphas = alphas

    def _eval_linear_comb(
        self, eval_fn: Callable[[Fn], tuple[Tensor, int]]
    ) -> tuple[Tensor, int]:
        """Sum alpha_i * F_i via eval_fn(F_i), which must return (values, flops)."""
        total_flops = 0
        Fvalues = 0
        for alpha, F in zip(self.alphas, self.functions):
            vals, flops = eval_fn(F)
            total_flops += flops
            Fvalues = Fvalues + alpha * vals
        return Fvalues, total_flops
