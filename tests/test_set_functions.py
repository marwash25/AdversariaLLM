"""Tests for set evaluation and ordered chain prefixes, without a lattice representation."""

import pytest
import torch

from llm_quick_check.attacks.submodular_utils import LinearCombinationSetFn, SetFunction


class TableSetFunction(SetFunction):
    """An arbitrary normalized set function on a 2 x 2 ground set."""

    def __init__(self):
        super().__init__(n=2, b=2, device=torch.device("cpu"))
        # Index each subset by a bitmask of its flattened ground-set elements.
        self.table = torch.tensor(
            [0, 2, 5, 1, 7, 3, 8, 4, 6, 0, 9, -1, 5, 2, -3, 10], dtype=torch.float64
        )
        self.batch_calls = 0

    def _eval_batch(self, rows_list, cols_list):
        self.batch_calls += 1
        indices = [int((1 << (rows * self.b + cols)).sum()) for rows, cols in zip(rows_list, cols_list)]
        return self.table[indices], 7 * len(indices)


def test_batch_and_single_evaluation_preserve_set_identity():
    fn = TableSetFunction()
    empty = torch.empty(0, dtype=torch.long)
    rows = [empty, torch.tensor([0]), torch.tensor([0]), torch.tensor([0, 0])]
    cols = [empty, torch.tensor([0]), torch.tensor([1]), torch.tensor([1, 0])]

    values, flops = fn(rows, cols)
    torch.testing.assert_close(values, torch.tensor([0., 2., 5., 1.], dtype=torch.float64))
    assert flops == 28
    value, flops = fn(rows[-1], cols[-1])
    assert value.shape == () and value.item() == 1
    assert flops == 7
    assert fn.eval_single(empty, empty)[0].item() == 0


@pytest.mark.parametrize(
    "rows, cols, expected",
    [
        ([1, 0, 1, 0], [0, 0, 1, 1], [7., 3., 2., 10.]),
        ([0, 1, 0, 1], [1, 1, 0, 0], [5., 9., -1., 10.]),
        ([1, 0], [0, 0], [7., 3.]),
    ],
)
def test_default_chain_evaluates_ordered_prefixes_in_one_batch(rows, cols, expected):
    fn = TableSetFunction()
    values, flops = fn.eval_chain(torch.tensor(rows), torch.tensor(cols))
    torch.testing.assert_close(values, torch.tensor(expected, dtype=torch.float64))
    assert flops == 7 * len(expected)
    assert fn.batch_calls == 1


def test_empty_batch_and_chain_do_not_evaluate_function():
    fn = TableSetFunction()
    empty = torch.empty(0, dtype=torch.long)
    for values, flops in (fn.eval_batch([], []), fn.eval_chain(empty, empty)):
        assert values.shape == (0,) and values.device == fn.device
        assert flops == 0
    assert fn.batch_calls == 0


@pytest.mark.parametrize("method", ["eval_single", "eval_chain"])
@pytest.mark.parametrize(
    "rows, cols, message",
    [
        (torch.tensor([0, 0]), torch.tensor([1, 1]), "must be unique"),
        (torch.tensor([-1]), torch.tensor([0]), "rows must have values"),
        (torch.tensor([2]), torch.tensor([0]), "rows must have values"),
        (torch.tensor([0]), torch.tensor([-1]), "cols must have values"),
        (torch.tensor([0]), torch.tensor([2]), "cols must have values"),
        (torch.tensor([0, 1]), torch.tensor([0]), "same length"),
        (torch.tensor([[0]]), torch.tensor([[0]]), "1D tensors"),
        (torch.tensor([0.]), torch.tensor([0]), "type long"),
        (torch.tensor([0]), torch.tensor([0.]), "type long"),
        (torch.tensor([0]), torch.empty(1, dtype=torch.long, device="meta"), "expected device"),
    ],
)
def test_invalid_set_and_chain_inputs_are_rejected_before_evaluation(method, rows, cols, message):
    fn = TableSetFunction()
    with pytest.raises(AssertionError, match=message):
        getattr(fn, method)(rows, cols)
    assert fn.batch_calls == 0


def test_batch_rejects_mismatched_lists():
    fn = TableSetFunction()
    with pytest.raises(AssertionError, match="same length"):
        fn.eval_batch([torch.tensor([0])], [])
    assert fn.batch_calls == 0


def test_specialized_chain_evaluation_bypasses_batch_evaluation():
    class SpecializedSetFunction(TableSetFunction):
        def _eval_chain(self, rows, cols):
            # This test requests the two-element chain (0, 0), (0, 1).
            return torch.tensor([2., 1.], dtype=torch.float64), 11

    fn = SpecializedSetFunction()
    values, flops = fn.eval_chain(torch.tensor([0, 0]), torch.tensor([0, 1]))
    torch.testing.assert_close(values, torch.tensor([2., 1.], dtype=torch.float64))
    assert flops == 11 and fn.batch_calls == 0


def test_linear_combination_batch_and_single_evaluation():
    first = TableSetFunction()
    second = TableSetFunction()
    first.table = first.table.float()
    second.table = second.table.square()
    fn = LinearCombinationSetFn([first, second], [2.0, -0.5])
    empty = torch.empty(0, dtype=torch.long)
    rows = [empty, torch.tensor([0]), torch.tensor([0]), torch.tensor([0, 0])]
    cols = [empty, torch.tensor([0]), torch.tensor([1]), torch.tensor([1, 0])]

    values, flops = fn(rows, cols)
    torch.testing.assert_close(values, torch.tensor([0., 2., -2.5, 1.5], dtype=torch.float64))
    assert flops == 56
    value, flops = fn(rows[-1], cols[-1])
    assert value.shape == () and value.item() == 1.5
    assert flops == 14


def test_linear_combination_uses_each_components_chain_evaluator():
    class ChainOnlySetFunction(SetFunction):
        def __init__(self):
            super().__init__(2, 2, torch.device("cpu"))

        def _eval_batch(self, rows_list, cols_list):
            raise AssertionError("Chain evaluation must use the specialized evaluator")

        def _eval_chain(self, rows, cols):
            weights = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
            return weights[rows, cols].cumsum(0), 3

    first = TableSetFunction()
    fn = LinearCombinationSetFn([first, ChainOnlySetFunction()], [1.0, -2.0])
    values, flops = fn.eval_chain(torch.tensor([1, 0, 1, 0]), torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(values, torch.tensor([1., -5., -14., -10.], dtype=torch.float64))
    assert flops == 31 and first.batch_calls == 1


def test_nested_linear_combinations_preserve_values_and_flop_counts():
    first = TableSetFunction()
    second = TableSetFunction()
    second.table = second.table.square()
    inner = LinearCombinationSetFn([first, second], [2.0, -0.5])
    outer = LinearCombinationSetFn([inner, first], [3.0, -1.0])
    value, flops = outer(torch.tensor([0]), torch.tensor([1]))
    assert value.item() == -12.5
    assert flops == 21


def test_empty_linear_combination_inputs_do_not_evaluate_components():
    first = TableSetFunction()
    second = TableSetFunction()
    fn = LinearCombinationSetFn([first, second], [1.0, -1.0])
    empty = torch.empty(0, dtype=torch.long)
    for values, flops in (fn.eval_batch([], []), fn.eval_chain(empty, empty)):
        assert values.shape == (0,) and flops == 0
    assert first.batch_calls == second.batch_calls == 0


@pytest.mark.parametrize("alphas, message", [([], "non-empty"), ([1.0], "same length")])
def test_linear_combination_rejects_empty_or_mismatched_components(alphas, message):
    with pytest.raises(AssertionError, match=message):
        LinearCombinationSetFn([], alphas)


@pytest.mark.parametrize(
    "attribute, value, message",
    [("n", 3, "same n"), ("b", 3, "same b"), ("device", torch.device("meta"), "same device")],
)
def test_linear_combination_rejects_incompatible_ground_sets_or_devices(attribute, value, message):
    first = TableSetFunction()
    second = TableSetFunction()
    setattr(second, attribute, value)
    with pytest.raises(AssertionError, match=message):
        LinearCombinationSetFn([first, second], [1.0, -1.0])
