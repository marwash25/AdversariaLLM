"""Tests for set evaluation and ordered chain prefixes, without a lattice representation."""

import pytest
import torch

from llm_quick_check.attacks.submodular_utils import (
    EneReductionMap,
    LinearCombinationSetFn,
    ModularSetFn,
    SetFunction,
    setfn_submodular_decomp,
)
from llm_quick_check.attacks.submodular_utils.setfn_reductions import BinaryRepresentationMap


class TableSetFunction(SetFunction):
    """An arbitrary normalized set function on a 2 x 2 ground set."""

    def __init__(self):
        super().__init__(n=2, b=2, device=torch.device("cpu"))
        # Index each subset by a bitmask of its flattened ground-set elements.
        self.table = torch.tensor(
            [0, 2, 5, 1, 7, 3, 8, 4, 6, 0, 9, -1, 5, 2, -3, 10], dtype=torch.float64
        )
        self.batch_calls = 0

    def _eval_batch(self, S_batch):
        self.batch_calls += 1
        indices = [int((1 << (rows * self.b + cols)).sum()) for rows, cols in S_batch]
        return self.table[indices], 7 * len(indices)


def test_singleton_complement_marginals_order_and_flops():
    fn = TableSetFunction()
    values, flops = fn.eval_singleton_complement_marginals()
    torch.testing.assert_close(values, torch.tensor([13., 8., 11., 6.], dtype=torch.float64))
    assert flops == 7 * 5
    assert fn.batch_calls == 1


@pytest.mark.parametrize("weight", [-2., 0., 3.])
def test_single_element_complement_is_empty(weight):
    fn = ModularSetFn(torch.tensor([[weight]], dtype=torch.float64))
    values, flops = fn.eval_singleton_complement_marginals()
    torch.testing.assert_close(values, torch.tensor([weight], dtype=torch.float64))
    assert flops == 0


def test_batch_and_single_evaluation_preserve_set_identity():
    fn = TableSetFunction()
    empty = torch.empty(0, dtype=torch.long)
    rows = [empty, torch.tensor([0]), torch.tensor([0]), torch.tensor([0, 0])]
    cols = [empty, torch.tensor([0]), torch.tensor([1]), torch.tensor([1, 0])]

    values, flops = fn(list(zip(rows, cols)))
    torch.testing.assert_close(values, torch.tensor([0., 2., 5., 1.], dtype=torch.float64))
    assert flops == 28
    value, flops = fn((rows[-1], cols[-1]))
    assert value.shape == () and value.item() == 1
    assert flops == 7
    assert fn.eval_single((empty, empty))[0].item() == 0


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
    values, flops, auxiliary_data = fn.eval_chain((torch.tensor(rows), torch.tensor(cols)))
    torch.testing.assert_close(values, torch.tensor(expected, dtype=torch.float64))
    assert auxiliary_data is None
    assert flops == 7 * len(expected)
    assert fn.batch_calls == 1


def test_subgradient_order_is_reused_for_cached_rounding_and_filtering():
    fn = TableSetFunction()
    X = torch.tensor([[0.8, 0.4], [0.9, 0.2]], dtype=torch.float64)
    subgradient, order, values, flops, auxiliary = fn.subgradient_lovasz_extension(X)
    torch.testing.assert_close(order, (torch.tensor([1, 0, 0, 1]), torch.tensor([0, 0, 1, 1])))
    torch.testing.assert_close(values, torch.tensor([7., 3., 4., 10.], dtype=torch.float64))
    torch.testing.assert_close(subgradient, torch.tensor([[-4., 1.], [7., 6.]], dtype=torch.float64))
    assert flops == 28 and auxiliary is None

    calls = []

    def filter_prefixes(passed_order, auxiliary_input):
        assert passed_order is order and auxiliary_input is auxiliary
        calls.append(passed_order)
        return [2, 3]

    unfiltered, filtered = fn.round_lovasz_extension(
        order=order, Fvalues=values, auxiliary_input=auxiliary, filter_prefixes=filter_prefixes,
    )
    assert fn.batch_calls == 1 and len(calls) == 1
    assert unfiltered[0] == 0. and unfiltered[1][0].numel() == unfiltered[1][1].numel() == 0
    assert filtered[0] == 3. and filtered[2] is None
    torch.testing.assert_close(filtered[1], (torch.tensor([1, 0]), torch.tensor([0, 0])))
    assert all(selected.data_ptr() == full.data_ptr() for selected, full in zip(filtered[1], order))


@pytest.mark.parametrize("map_type", [EneReductionMap, BinaryRepresentationMap])
def test_tuple_map_conversions_preserve_batch_order_and_empty_sets(map_type):
    reduction_map = map_type(k=4, n=2, device=torch.device("cpu"))
    x = torch.tensor([[0, 0], [1, 3], [0, 0], [2, 1]])
    S_batch = reduction_map.ints2set(x)
    assert len(S_batch) == len(x) and all(isinstance(S, tuple) for S in S_batch)
    assert all(S_batch[i][0].numel() == S_batch[i][1].numel() == 0 for i in (0, 2))
    torch.testing.assert_close(reduction_map.set2ints(S_batch), x)
    assert reduction_map.ints2set(x[:0]) == []
    torch.testing.assert_close(reduction_map.set2ints([]), x[:0])


def test_empty_batch_and_chain_do_not_evaluate_function():
    fn = TableSetFunction()
    empty = torch.empty(0, dtype=torch.long)
    chain_values, chain_flops, auxiliary_data = fn.eval_chain((empty, empty))
    assert auxiliary_data is None
    for values, flops in (fn.eval_batch([]), (chain_values, chain_flops)):
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
        getattr(fn, method)((rows, cols))
    assert fn.batch_calls == 0


def test_batch_rejects_mismatched_paired_indices():
    fn = TableSetFunction()
    with pytest.raises(AssertionError, match="same length"):
        fn.eval_batch([(torch.tensor([0]), torch.empty(0, dtype=torch.long))])
    assert fn.batch_calls == 0


def test_specialized_chain_evaluation_bypasses_batch_evaluation():
    class SpecializedSetFunction(TableSetFunction):
        def _eval_chain(self, order):
            # This test requests the two-element chain (0, 0), (0, 1).
            return torch.tensor([2., 1.], dtype=torch.float64), 11, None

    fn = SpecializedSetFunction()
    values, flops, auxiliary_data = fn.eval_chain((torch.tensor([0, 0]), torch.tensor([0, 1])))
    torch.testing.assert_close(values, torch.tensor([2., 1.], dtype=torch.float64))
    assert auxiliary_data is None
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

    values, flops = fn(list(zip(rows, cols)))
    torch.testing.assert_close(values, torch.tensor([0., 2., -2.5, 1.5], dtype=torch.float64))
    assert flops == 56
    value, flops = fn((rows[-1], cols[-1]))
    assert value.shape == () and value.item() == 1.5
    assert flops == 14


def test_linear_combination_uses_each_components_chain_evaluator():
    class ChainOnlySetFunction(SetFunction):
        def __init__(self):
            super().__init__(2, 2, torch.device("cpu"))

        def _eval_batch(self, S_batch):
            raise AssertionError("Chain evaluation must use the specialized evaluator")

        def _eval_chain(self, order):
            rows, cols = order
            weights = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
            return weights[rows, cols].cumsum(0), 0 if rows.numel() == 0 else 3, None

    first = TableSetFunction()
    fn = LinearCombinationSetFn([first, ChainOnlySetFunction()], [1.0, -2.0])
    values, flops, auxiliary_data = fn.eval_chain((torch.tensor([1, 0, 1, 0]), torch.tensor([0, 0, 1, 1])))
    torch.testing.assert_close(values, torch.tensor([1., -5., -14., -10.], dtype=torch.float64))
    assert auxiliary_data is None
    assert flops == 31 and first.batch_calls == 1


def test_nested_linear_combinations_preserve_values_and_flop_counts():
    first = TableSetFunction()
    second = TableSetFunction()
    second.table = second.table.square()
    inner = LinearCombinationSetFn([first, second], [2.0, -0.5])
    outer = LinearCombinationSetFn([inner, first], [3.0, -1.0])
    value, flops = outer((torch.tensor([0]), torch.tensor([1])))
    assert value.item() == -12.5
    assert flops == 21


def test_empty_linear_combination_inputs_do_not_evaluate_components():
    first = TableSetFunction()
    second = TableSetFunction()
    fn = LinearCombinationSetFn([first, second], [1.0, -1.0])
    empty = torch.empty(0, dtype=torch.long)
    chain_values, chain_flops, auxiliary_data = fn.eval_chain((empty, empty))
    assert auxiliary_data is None
    for values, flops in (fn.eval_batch([]), (chain_values, chain_flops)):
        assert values.shape == (0,) and flops == 0
    assert first.batch_calls == second.batch_calls == 0


@pytest.mark.parametrize("alphas, message", [([], "non-empty"), ([1.0], "same length")])
def test_linear_combination_rejects_empty_or_mismatched_components(alphas, message):
    with pytest.raises(AssertionError, match=message):
        LinearCombinationSetFn([], alphas)


def test_modular_set_function_sums_weights_and_agrees_with_chain():
    W = torch.tensor([[1.0, 3.0], [5.0, -2.0]], dtype=torch.float64)
    fn = ModularSetFn(W)
    empty = torch.empty(0, dtype=torch.long)
    rows_list = [empty, torch.tensor([0]), torch.tensor([0]), torch.tensor([0, 1])]
    cols_list = [empty, torch.tensor([0]), torch.tensor([1]), torch.tensor([0, 1])]

    values, flops = fn(list(zip(rows_list, cols_list)))
    torch.testing.assert_close(values, torch.tensor([0.0, 1.0, 3.0, -1.0], dtype=torch.float64))
    assert values.dtype == W.dtype and values.device == W.device
    assert flops == 0
    assert fn.eval_single((empty, empty))[0].item() == 0

    chain_rows = torch.tensor([0, 1, 0])
    chain_cols = torch.tensor([0, 1, 1])
    chain_values, chain_flops, auxiliary_data = fn.eval_chain((chain_rows, chain_cols))
    expected = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(chain_values, expected)
    assert chain_values.dtype == W.dtype
    assert auxiliary_data is None and chain_flops == 0

    prefix_values, _ = fn(
        [(chain_rows[: t + 1], chain_cols[: t + 1]) for t in range(chain_rows.numel())],
    )
    torch.testing.assert_close(chain_values, prefix_values)

    empty_values, empty_flops, empty_aux = fn.eval_chain((empty, empty))
    assert empty_values.shape == (0,) and empty_values.dtype == W.dtype
    assert empty_flops == 0 and empty_aux is None


def test_modular_set_function_distinguishes_ene_colliding_sets():
    reduction_map = EneReductionMap(k=3, n=1, device=torch.device("cpu"))
    torch.testing.assert_close(reduction_map.weights, torch.tensor([1, 1]))
    W = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    fn = ModularSetFn(W)
    rows_s, cols_s = torch.tensor([0]), torch.tensor([0])
    rows_t, cols_t = torch.tensor([0]), torch.tensor([1])

    images = reduction_map.set2ints([(rows_s, cols_s), (rows_t, cols_t)])
    torch.testing.assert_close(images[0], images[1])
    assert images[0].tolist() == [1]

    values, flops = fn([(rows_s, cols_s), (rows_t, cols_t)])
    torch.testing.assert_close(values, torch.tensor([1.0, 3.0], dtype=torch.float64))
    assert flops == 0
    assert not hasattr(fn, "map")

    for rows, cols, expected in ((rows_s, cols_s, 1.0), (rows_t, cols_t, 3.0)):
        chain_values, _, auxiliary_data = fn.eval_chain((rows, cols))
        torch.testing.assert_close(chain_values, torch.tensor([expected], dtype=torch.float64))
        assert auxiliary_data is None


def test_setfn_submodular_decomp_values_and_slack():
    F = TableSetFunction()
    alpha = torch.tensor(-9.0, dtype=torch.float64)
    G, H = setfn_submodular_decomp(F, alpha)
    empty = torch.empty(0, dtype=torch.long)
    S_batch = [
        (empty, empty),
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([0, 1]), torch.tensor([0, 1])),
    ]

    Fvalues, _ = F(S_batch)
    Gvalues, _ = G(S_batch)
    Hvalues, _ = H(S_batch)
    torch.testing.assert_close(Hvalues, torch.tensor([0.0, -4.5, -18.0], dtype=torch.float64))
    torch.testing.assert_close(Gvalues - Hvalues, Fvalues)

    # H contributes -alpha to every pairwise submodularity slack.
    singleton_01 = (torch.tensor([0]), torch.tensor([1]))
    pair = (torch.tensor([0, 0]), torch.tensor([0, 1]))
    G_slack = Gvalues[1] + G.eval_single(singleton_01)[0] - Gvalues[0] - G.eval_single(pair)[0]
    H_slack = Hvalues[1] + H.eval_single(singleton_01)[0] - Hvalues[0] - H.eval_single(pair)[0]
    torch.testing.assert_close(G_slack, torch.tensor(15.0, dtype=torch.float64))
    torch.testing.assert_close(H_slack, -alpha)


def test_setfn_submodular_decomp_clamps_positive_alpha_to_zero():
    F = TableSetFunction()
    G, H = setfn_submodular_decomp(F, torch.tensor(2.0, dtype=torch.float64))
    S = (torch.tensor([0, 1]), torch.tensor([0, 1]))

    Fvalue, _ = F(S)
    Gvalue, _ = G(S)
    Hvalue, _ = H(S)
    torch.testing.assert_close(Gvalue, Fvalue)
    torch.testing.assert_close(Hvalue, torch.tensor(0.0, dtype=torch.float64))


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
