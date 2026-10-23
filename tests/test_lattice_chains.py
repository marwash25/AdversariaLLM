"""Construction, reuse, and validation of lattice chains and neighbors."""

from unittest.mock import patch

import pytest
import torch

from llm_quick_check.attacks.submodular_utils.lattice_functions import (
    LatticeFunction,
    LinearCombinationLatticeFn,
    SequentialLatticeFunction,
)
from llm_quick_check.attacks.submodular_utils.lattice_fn_instances import QuadraticFn
from llm_quick_check.attacks.submodular_utils.setfn_reductions import EneReductionMap, SetFnReduction


class BatchQuadratic(LatticeFunction):
    def __init__(self):
        super().__init__(k=4, n=2)
        self.batch_calls = 0

    def _eval_batch(self, x):
        self.batch_calls += 1
        return x.to(torch.float64).square().sum(dim=1), 11 * x.shape[0]


class IncrementalQuadratic(SequentialLatticeFunction):
    def __init__(self):
        super().__init__(k=4, n=2)
        self.batch_inputs = []
        self.add_calls = 0

    def _eval_batch(self, x):
        self.batch_inputs.append(x.clone())
        return x.to(torch.float64).square().sum(dim=1), 11 * x.shape[0]

    def add(self, i, weight):
        self.add_calls += 1
        new_x = self.current_x.clone()
        new_x[i] += weight
        new_val = self.current_val + 2 * self.current_x[i] * weight + weight.square()
        return new_val, new_x, 3


@pytest.fixture
def chain():
    return (
        (torch.tensor([1, 0, 1, 0]), torch.tensor([1, 0, 0, 1])),
        torch.tensor([1, 2]),
        torch.tensor([[0, 2], [1, 2], [1, 3], [3, 3]]),
    )


@pytest.mark.parametrize("factory", [BatchQuadratic, IncrementalQuadratic])
@pytest.mark.parametrize("supply_chain", [False, True])
def test_chain_matches_direct_prefix_evaluation(factory, supply_chain, chain):
    order, weights, expected_chain = chain
    F = factory()
    supplied = expected_chain.clone() if supply_chain else None
    # A supplied chain must never be reconstructed. Sequential evaluation also
    # constructs missing points from its updates rather than the batch builder.
    if supply_chain or isinstance(F, SequentialLatticeFunction):
        with patch.object(F, "_build_chain", side_effect=AssertionError("unexpected chain construction")):
            values, flops, x_chain = F.eval_chain(order, weights, supplied)
    else:
        values, flops, x_chain = F.eval_chain(order, weights)
    torch.testing.assert_close(x_chain, expected_chain)
    torch.testing.assert_close(values, expected_chain.to(torch.float64).square().sum(dim=1))
    assert values.dtype == torch.float64
    if supply_chain:
        assert x_chain is supplied
        torch.testing.assert_close(supplied, expected_chain)
    if isinstance(F, IncrementalQuadratic):
        assert len(F.batch_inputs) == 1
        torch.testing.assert_close(F.batch_inputs[0], expected_chain[:1])
        assert F.add_calls == len(order[0]) - 1
        assert flops == 11 + 3 * (len(order[0]) - 1)
        torch.testing.assert_close(F.current_x, expected_chain[-1])
    else:
        assert F.batch_calls == 1
        assert flops == 11 * len(order[0])


@pytest.mark.parametrize("supply_chain", [False, True])
def test_real_sequential_quadratic_matches_batch(supply_chain, chain):
    order, weights, expected_chain = chain
    F = QuadraticFn(torch.tensor([[2., 1.], [1., -3.]], dtype=torch.float64), k=4)
    expected_values, _ = F.eval_batch(expected_chain)
    values, flops, x_chain = F.eval_chain(order, weights, expected_chain if supply_chain else None)
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(x_chain, expected_chain)
    assert flops == 0


@pytest.mark.parametrize("factory", [BatchQuadratic, IncrementalQuadratic])
@pytest.mark.parametrize("supply_chain", [False, True])
def test_empty_chain_does_not_evaluate(factory, supply_chain):
    F = factory()
    indices = torch.empty(0, dtype=torch.long)
    supplied = torch.empty((0, 2), dtype=torch.long) if supply_chain else None
    values, flops, x_chain = F.eval_chain((indices, indices), torch.tensor([1, 2]), supplied)
    assert values.shape == (0,)
    assert x_chain.shape == (0, 2) and x_chain.dtype == torch.long
    assert values.device == x_chain.device == indices.device
    assert flops == 0
    if supply_chain:
        assert x_chain is supplied
    if isinstance(F, IncrementalQuadratic):
        assert F.batch_inputs == [] and F.add_calls == 0 and F.current_x is None
    else:
        assert F.batch_calls == 0


@pytest.mark.parametrize("change, message", [
    (lambda o, w, x: ((o[0][:, None], o[1]), w, x), "1D tensors"),
    (lambda o, w, x: ((o[0], o[1][:-1]), w, x), "same length"),
    (lambda o, w, x: ((o[0].float(), o[1]), w, x), "type long"),
    (lambda o, w, x: ((o[0], o[1].float()), w, x), "type long"),
    (lambda o, w, x: (o, w.float(), x), "1D.*long"),
    (lambda o, w, x: (o, w[:, None], x), "1D.*long"),
    (lambda o, w, x: (o, -w, x), "strictly positive"),
    (lambda o, w, x: (o, w - 1, x), "strictly positive"),
    (lambda o, w, x: ((o[0] + 2, o[1]), w, x), "rows must"),
    (lambda o, w, x: ((o[0], o[1] + 2), w, x), "cols must"),
    (lambda o, w, x: ((o[0], torch.zeros_like(o[1])), w, x), "unique"),
    (lambda o, w, x: (o, w * 2, None), "values in"),
    (lambda o, w, x: (o, w, x[:, :1]), "shape"),
    (lambda o, w, x: (o, w, x.float()), "values in"),
    (lambda o, w, x: (o, w, x + 1), "values in"),
    (lambda o, w, x: ((o[0].to("meta"), o[1]), w, x), "expected device"),
    (lambda o, w, x: (o, w, x.to("meta")), "same device"),
])
def test_invalid_inputs_rejected_before_evaluation(change, message, chain):
    F = BatchQuadratic()
    with pytest.raises(AssertionError, match=message):
        F.eval_chain(*change(*chain))
    assert F.batch_calls == 0


@pytest.mark.parametrize("failure, message", [
    ("value_shape", "one scalar"),
    ("value_device", "same device"),
    ("chain_copy", "supplied x_chain"),
])
def test_invalid_evaluator_outputs_are_rejected(failure, message, chain):
    class InvalidResult(BatchQuadratic):
        def _eval_chain(self, order, weights, x_chain=None):
            values, flops, x_chain = super()._eval_chain(order, weights, x_chain)
            if failure == "value_shape":
                values = values[:, None]
            elif failure == "value_device":
                values = values.to("meta")
            elif failure == "chain_copy":
                x_chain = x_chain.clone()
            return values, flops, x_chain

    with pytest.raises(AssertionError, match=message):
        InvalidResult().eval_chain(*chain)


@pytest.mark.parametrize("factory", [BatchQuadratic, IncrementalQuadratic])
@pytest.mark.parametrize("supply_neighbors", [False, True])
@pytest.mark.parametrize("x, weights, expected", [
    ([1, 2], [1, 2], [[2, 2], [3, 2], [1, 3], [0, 2], [1, 1], [1, 0]]),
    ([1, 2], [2, 1, 2, 1], [[2, 2], [3, 2], [1, 3], [0, 2], [1, 1], [1, 0]]),
    ([0, 3], [1, 1, 4], [[1, 3], [0, 2]]),
])
def test_neighbors_match_direct_evaluation(factory, supply_neighbors, x, weights, expected):
    F = factory()
    x, weights, expected = torch.tensor(x), torch.tensor(weights), torch.tensor(expected)
    supplied = expected.clone() if supply_neighbors else None
    if supply_neighbors or isinstance(F, SequentialLatticeFunction):
        with patch.object(F, "_build_neighbors", side_effect=AssertionError("unexpected construction")):
            values, flops, neighbors = F.eval_neighbors(x, weights, supplied)
    else:
        values, flops, neighbors = F.eval_neighbors(x, weights)
    torch.testing.assert_close(neighbors, expected)
    torch.testing.assert_close(values, expected.to(torch.float64).square().sum(dim=1))
    if supply_neighbors:
        assert neighbors is supplied
    if isinstance(F, IncrementalQuadratic):
        assert len(F.batch_inputs) == 1 and F.add_calls == len(expected)
        torch.testing.assert_close(F.current_x, x)
        torch.testing.assert_close(F.current_val, x.to(torch.float64).square().sum())
        assert flops == 11 + 3 * len(expected)
    else:
        assert F.batch_calls == 1 and flops == 11 * len(expected)


@pytest.mark.parametrize("sequential_first", [False, True])
@pytest.mark.parametrize("weights", [[1, 2], [2, 1, 2, 1]])
def test_mixed_combination_shares_neighbors_from_first_component(sequential_first, weights):
    batch, sequential = BatchQuadratic(), IncrementalQuadratic()
    components = [sequential, batch] if sequential_first else [batch, sequential]
    F = LinearCombinationLatticeFn(components, [2., -1.])
    with patch.object(components[1], "_build_neighbors", side_effect=AssertionError("unexpected construction")):
        values, flops, neighbors = F.eval_neighbors(torch.tensor([1, 2]), torch.tensor(weights))
    torch.testing.assert_close(neighbors, torch.tensor([[2, 2], [3, 2], [1, 3], [0, 2], [1, 1], [1, 0]]))
    torch.testing.assert_close(values, neighbors.to(torch.float64).square().sum(dim=1))
    torch.testing.assert_close(sequential.current_x, torch.tensor([1, 2]))
    assert flops == 11 * len(neighbors) + 11 + 3 * len(neighbors)


@pytest.mark.parametrize("supply_neighbors", [False, True])
@pytest.mark.parametrize("weights", [[], [4], [4, 4]])
def test_empty_neighbors_do_not_evaluate(supply_neighbors, weights):
    batch, sequential = BatchQuadratic(), IncrementalQuadratic()
    combination = LinearCombinationLatticeFn([sequential, batch], [1., -1.])
    for F in (batch, sequential, combination):
        supplied = torch.empty((0, 2), dtype=torch.long) if supply_neighbors else None
        values, flops, neighbors = F.eval_neighbors(
            torch.tensor([1, 2]), torch.tensor(weights, dtype=torch.long), supplied
        )
        assert values.shape == (0,) and neighbors.shape == (0, 2) and flops == 0
        assert neighbors.dtype == torch.long and values.device == neighbors.device
        if supply_neighbors:
            assert neighbors is supplied
    assert batch.batch_calls == 0
    assert sequential.batch_inputs == [] and sequential.add_calls == 0
    assert sequential.current_x is None


@pytest.mark.parametrize("factory", [BatchQuadratic, IncrementalQuadratic])
@pytest.mark.parametrize("weights", [[0], [0, 1], [-1, 1]])
def test_neighbors_reject_nonpositive_weights_before_evaluation(factory, weights):
    F = factory()
    with pytest.raises(AssertionError, match="strictly positive"):
        F.eval_neighbors(torch.tensor([1, 2]), torch.tensor(weights))
    if isinstance(F, IncrementalQuadratic):
        assert F.batch_inputs == [] and F.add_calls == 0 and F.current_x is None
    else:
        assert F.batch_calls == 0


@pytest.mark.parametrize("weights", [[0], [0, 1], [-1, 1]])
def test_reduction_map_rejects_nonpositive_weights(weights):
    with patch.object(EneReductionMap, "get_weights", return_value=torch.tensor(weights)):
        with pytest.raises(AssertionError, match="strictly positive"):
            EneReductionMap(k=4, n=2, device=torch.device("cpu"))


def test_neighbor_selection_preserves_ties_and_filtering():
    F = BatchQuadratic()
    reduction = SetFnReduction(F, EneReductionMap(k=4, n=2, device=torch.device("cpu")))
    x = torch.tensor([1, 1])
    values, expected_flops, neighbors = F.eval_neighbors(x, reduction.map.weights)

    def retain_additions(candidates):
        torch.testing.assert_close(candidates, neighbors)
        return (candidates.sum(dim=1) > x.sum()).nonzero().flatten().tolist()

    best, point, filtered, filtered_point, flops = reduction.get_best_neighbors(x, retain_additions)
    assert best == 1. and filtered == 5. and flops == expected_flops
    torch.testing.assert_close(point, torch.tensor([0, 1]))
    torch.testing.assert_close(filtered_point, torch.tensor([2, 1]))
    assert reduction.get_best_neighbors(x, lambda _: [])[2] == float("inf")
    unfiltered = reduction.get_best_neighbors(x)
    assert unfiltered[0] == unfiltered[2] == values.min().item()
    torch.testing.assert_close(unfiltered[1], unfiltered[3])
