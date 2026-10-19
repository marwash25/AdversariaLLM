"""Regression tests for lattice evaluation through the shared linear-combination mixin."""

import pytest
import torch

from llm_quick_check.attacks.submodular_utils import LatticeFunction, LinearCombinationLatticeFn


class PolynomialLatticeFunction(LatticeFunction):
    def __init__(self, power=1, dtype=torch.float32, k=4, n=2):
        super().__init__(k, n)
        self.power = power
        self.dtype = dtype
        self.batch_calls = 0
        self.chain_calls = 0
        self.neighbor_calls = 0

    def _values(self, x):
        return x.to(self.dtype).pow(self.power).sum(dim=1)

    def _eval_batch(self, x):
        self.batch_calls += 1
        return self._values(x), 7 * x.shape[0]

    def _eval_chain(self, rows, cols, weights, x_chain=None):
        self.chain_calls += 1
        self.supplied_chain = x_chain
        if x_chain is None:
            x_chain = self._build_chain(rows, cols, weights)
        self.last_chain = x_chain
        return self._values(x_chain), 13, x_chain

    def _eval_neighbors(self, x, weights, x_neighbors):
        self.neighbor_calls += 1
        return self._values(x_neighbors), 19


@pytest.fixture
def combination():
    first = PolynomialLatticeFunction()
    second = PolynomialLatticeFunction(power=2, dtype=torch.float64)
    return LinearCombinationLatticeFn([first, second], [2.0, -0.5])


def test_batch_and_single_evaluation_promote_dtype_and_sum_flops(combination):
    x = torch.tensor([[0, 0], [1, 2], [3, 1]])
    values, flops = combination(x)
    torch.testing.assert_close(values, torch.tensor([0., 3.5, 3.], dtype=torch.float64))
    assert flops == 42
    value, flops = combination(x[1])
    assert value.shape == () and value.item() == 3.5
    assert flops == 14


@pytest.mark.parametrize("supply_chain", [False, True])
def test_chain_uses_specialized_component_evaluators(combination, supply_chain):
    rows = torch.tensor([1, 0, 1, 0])
    cols = torch.tensor([1, 0, 0, 1])
    weights = torch.tensor([1, 2])
    x_chain = torch.tensor([[0, 2], [1, 2], [1, 3], [3, 3]])
    values, flops, returned_chain = combination.eval_chain(
        rows, cols, weights, x_chain if supply_chain else None
    )
    torch.testing.assert_close(values, torch.tensor([2., 3.5, 3., 3.], dtype=torch.float64))
    torch.testing.assert_close(returned_chain, x_chain)
    if supply_chain:
        assert returned_chain is x_chain
    assert flops == 26
    assert all(F.chain_calls == 1 and F.batch_calls == 0 for F in combination.functions)
    assert all(F.last_chain is returned_chain for F in combination.functions)
    assert combination.functions[0].supplied_chain is (x_chain if supply_chain else None)
    assert combination.functions[1].supplied_chain is returned_chain


def test_nested_combinations_share_one_chain(combination):
    third = PolynomialLatticeFunction(power=3)
    outer = LinearCombinationLatticeFn([combination, third], [2.0, -1.0])
    values, flops, x_chain = outer.eval_chain(torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([1]))
    torch.testing.assert_close(values, torch.tensor([2., 4.], dtype=torch.float64))
    assert flops == 39
    assert all(F.last_chain is x_chain for F in [*combination.functions, third])


def test_neighbors_use_specialized_component_evaluators(combination):
    x = torch.tensor([1, 2])
    weights = torch.tensor([1, 2])
    neighbors = torch.tensor([[2, 2], [3, 2], [1, 3], [0, 2], [1, 1], [1, 0]])
    values, flops = combination.eval_neighbors(x, weights, neighbors)
    torch.testing.assert_close(values, torch.tensor([4., 3.5, 3., 2., 3., 1.5], dtype=torch.float64))
    assert flops == 38
    assert all(F.neighbor_calls == 1 and F.batch_calls == 0 for F in combination.functions)


def test_nested_combinations_preserve_neighbor_evaluation(combination):
    outer = LinearCombinationLatticeFn([combination, combination.functions[0]], [2.0, -1.0])
    values, flops = outer.eval_neighbors(
        torch.tensor([0, 0]), torch.tensor([1]), torch.tensor([[1, 0], [0, 1]])
    )
    torch.testing.assert_close(values, torch.tensor([2., 2.], dtype=torch.float64))
    assert flops == 57


@pytest.mark.parametrize("alphas, message", [([], "non-empty"), ([1.0], "same length")])
def test_rejects_empty_or_mismatched_component_lists(alphas, message):
    with pytest.raises(AssertionError, match=message):
        LinearCombinationLatticeFn([], alphas)


@pytest.mark.parametrize("kwargs, message", [({"n": 3}, "same n"), ({"k": 3}, "same k")])
def test_rejects_incompatible_lattice_domains(kwargs, message):
    with pytest.raises(AssertionError, match=message):
        LinearCombinationLatticeFn(
            [PolynomialLatticeFunction(), PolynomialLatticeFunction(**kwargs)], [1.0, -1.0]
        )
