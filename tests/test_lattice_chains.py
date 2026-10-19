"""Construction, reuse, and validation of lattice chains."""

from unittest.mock import patch

import pytest
import torch

from llm_quick_check.attacks.submodular_utils.lattice_functions import (
    LatticeFunction,
    SequentialLatticeFunction,
)
from llm_quick_check.attacks.submodular_utils.lattice_fn_instances import QuadraticFn


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
        torch.tensor([1, 0, 1, 0]),
        torch.tensor([1, 0, 0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([[0, 2], [1, 2], [1, 3], [3, 3]]),
    )


@pytest.mark.parametrize("factory", [BatchQuadratic, IncrementalQuadratic])
@pytest.mark.parametrize("supply_chain", [False, True])
def test_chain_matches_direct_prefix_evaluation(factory, supply_chain, chain):
    rows, cols, weights, expected_chain = chain
    F = factory()
    supplied = expected_chain.clone() if supply_chain else None
    # A supplied chain must never be reconstructed. Sequential evaluation also
    # constructs missing points from its updates rather than the batch builder.
    if supply_chain or isinstance(F, SequentialLatticeFunction):
        with patch.object(F, "_build_chain", side_effect=AssertionError("unexpected chain construction")):
            values, flops, x_chain = F.eval_chain(rows, cols, weights, supplied)
    else:
        values, flops, x_chain = F.eval_chain(rows, cols, weights)
    torch.testing.assert_close(x_chain, expected_chain)
    torch.testing.assert_close(values, expected_chain.to(torch.float64).square().sum(dim=1))
    assert values.dtype == torch.float64
    if supply_chain:
        assert x_chain is supplied
        torch.testing.assert_close(supplied, expected_chain)
    if isinstance(F, IncrementalQuadratic):
        assert len(F.batch_inputs) == 1
        torch.testing.assert_close(F.batch_inputs[0], expected_chain[:1])
        assert F.add_calls == len(rows) - 1
        assert flops == 11 + 3 * (len(rows) - 1)
        torch.testing.assert_close(F.current_x, expected_chain[-1])
    else:
        assert F.batch_calls == 1
        assert flops == 11 * len(rows)


@pytest.mark.parametrize("supply_chain", [False, True])
def test_real_sequential_quadratic_matches_batch(supply_chain, chain):
    rows, cols, weights, expected_chain = chain
    F = QuadraticFn(torch.tensor([[2., 1.], [1., -3.]], dtype=torch.float64), k=4)
    expected_values, _ = F.eval_batch(expected_chain)
    values, flops, x_chain = F.eval_chain(rows, cols, weights, expected_chain if supply_chain else None)
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(x_chain, expected_chain)
    assert flops == 0


@pytest.mark.parametrize("factory", [BatchQuadratic, IncrementalQuadratic])
@pytest.mark.parametrize("supply_chain", [False, True])
def test_empty_chain_does_not_evaluate(factory, supply_chain):
    F = factory()
    indices = torch.empty(0, dtype=torch.long)
    supplied = torch.empty((0, 2), dtype=torch.long) if supply_chain else None
    values, flops, x_chain = F.eval_chain(indices, indices, torch.tensor([1, 2]), supplied)
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
    (lambda r, c, w, x: (r[:, None], c, w, x), "1D tensors"),
    (lambda r, c, w, x: (r, c[:-1], w, x), "same length"),
    (lambda r, c, w, x: (r.float(), c, w, x), "type long"),
    (lambda r, c, w, x: (r, c.float(), w, x), "type long"),
    (lambda r, c, w, x: (r, c, w.float(), x), "1D.*long"),
    (lambda r, c, w, x: (r, c, w[:, None], x), "1D.*long"),
    (lambda r, c, w, x: (r, c, -w, x), "nonnegative"),
    (lambda r, c, w, x: (r + 2, c, w, x), "rows must"),
    (lambda r, c, w, x: (r, c + 2, w, x), "cols must"),
    (lambda r, c, w, x: (r, torch.zeros_like(c), w, x), "unique"),
    (lambda r, c, w, x: (r, c, w * 2, None), "values in"),
    (lambda r, c, w, x: (r, c, w, x[:, :1]), "shape"),
    (lambda r, c, w, x: (r, c, w, x.float()), "values in"),
    (lambda r, c, w, x: (r, c, w, x + 1), "values in"),
    (lambda r, c, w, x: (r.to("meta"), c, w, x), "expected device"),
    (lambda r, c, w, x: (r, c, w, x.to("meta")), "same device"),
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
        def _eval_chain(self, rows, cols, weights, x_chain=None):
            values, flops, x_chain = super()._eval_chain(rows, cols, weights, x_chain)
            if failure == "value_shape":
                values = values[:, None]
            elif failure == "value_device":
                values = values.to("meta")
            elif failure == "chain_copy":
                x_chain = x_chain.clone()
            return values, flops, x_chain

    with pytest.raises(AssertionError, match=message):
        InvalidResult().eval_chain(*chain)
