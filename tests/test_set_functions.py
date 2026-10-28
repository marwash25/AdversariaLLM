"""Tests for set evaluation and ordered chain prefixes, without a lattice representation."""

from unittest.mock import patch

import pytest
import torch

from llm_quick_check.attacks.submodular_utils import (
    EneReductionMap,
    LinearCombinationSetFn,
    ModularSetFn,
    SetFnReduction,
    SetFunction,
    setfn_submodular_decomp,
)
from llm_quick_check.attacks.submodular_utils.setfn_reductions import BinaryRepresentationMap
from llm_quick_check.attacks.submodular_utils.set_fn_instances import SquaredCardinalitySetFn


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


def test_singleton_removal_marginals_order_flops_and_cache():
    fn = TableSetFunction()
    values, flops = fn.get_singleton_removal_marginals()
    torch.testing.assert_close(values, torch.tensor([[13., 8.], [11., 6.]], dtype=torch.float64))
    assert flops == 7 * 5
    assert fn.batch_calls == 1
    cached, flops = fn.get_singleton_removal_marginals()
    assert cached is values and flops == 0
    assert fn.batch_calls == 1


def test_singletons_cache_is_reused_by_bound():
    fn = TableSetFunction()
    values, flops = fn.get_singleton_vals()
    torch.testing.assert_close(values, torch.tensor([[2., 5.], [7., 6.]], dtype=torch.float64))
    assert flops == 28 and fn.batch_calls == 1
    cached, flops = fn.get_singleton_vals()
    assert cached is values and flops == 0
    bound, flops = fn.lovasz_L_bound()
    expected = torch.tensor([[13., 8.], [11., 6.]], dtype=torch.float64).norm().item()
    assert bound == pytest.approx(expected) and flops == 35
    assert fn.lovasz_L_bound() == (bound, 0)
    assert fn.get_singleton_removal_marginals()[1] == 0
    assert fn.batch_calls == 2


@pytest.mark.parametrize(
    "values,properties,expected,calls,flops",
    [
        ([0., 1., 1., 2.], (True, True, False), 2 ** 0.5, 2, 21),
        ([0., 1., 1., 1.], (True, True, False), 1., 2, 21),
        ([0., -1., -1., -2.], (True, False, True), 2 ** 0.5, 2, 28),
        ([0., -0.5, -0.5, -2.], (True, False, True), 2., 2, 28),
        ([0., 1., 1., -10.], (True, False, False), 11 * 2 ** 0.5, 2, 35),
        ([0., 1., 1., -10.], (False, False, False), 11 * 2 ** 0.5, 2, 35),
        ([0., 0.5, 0.5, 2.], (False, True, False), 2., 1, 7),
        ([0., -1., -1., -1.], (False, False, True), 1., 1, 7),
    ],
)
def test_lovasz_bound_properties(values, properties, expected, calls, flops):
    fn = TableSetFunction()
    fn.n, fn.b = 1, 2
    fn.table = torch.tensor(values, dtype=torch.float64)
    fn.is_submodular, fn.is_non_decreasing, fn.is_non_increasing = properties
    bound, cost = fn.lovasz_L_bound()
    assert bound == pytest.approx(expected) and cost == flops
    assert fn.batch_calls == calls
    assert fn.max_abs_value == (abs(values[-1]) if any(properties[1:]) else None)
    assert fn.lovasz_L_bound() == (bound, 0)
    assert fn.batch_calls == calls
    if fn.is_submodular:
        for first in (1, 2):
            gradient = torch.tensor([values[first], values[3] - values[first]])
            assert gradient.norm().item() <= bound + 1e-6


@pytest.mark.parametrize("max_abs_value,expected", [(None, 4.), (1., 3.), (2., 4.)])
def test_lovasz_bound_constructor_global_bound_and_cache(max_abs_value, expected):
    class SaturatingFunction(SetFunction):
        def _eval_batch(self, S_batch):
            return torch.tensor([float(rows.numel() > 0) for rows, _ in S_batch]), 7 * len(S_batch)

    # Declare only submodularity to exercise the general endpoint/global bounds.
    fn = SaturatingFunction(
        1, 16, torch.device("cpu"), is_submodular=True, max_abs_value=max_abs_value,
    )
    assert fn.max_abs_value == max_abs_value
    assert fn.lovasz_L_bound() == (expected, 7 * 33)
    with patch.object(fn, "_eval_batch", side_effect=AssertionError("unexpected set evaluation")):
        assert fn.lovasz_L_bound() == (expected, 0)


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("submodular", [False, True])
def test_monotone_lovasz_bound_reuses_supplied_maximum(sign, submodular):
    fn = ModularSetFn(sign * torch.ones(1, 2))
    fn.is_submodular = submodular
    fn.max_abs_value = 3.  # Valid but looser than |F(V)| = 2.
    with patch.object(fn, "_eval_batch", side_effect=AssertionError("unexpected ground-set evaluation")):
        bound, flops = SetFunction.lovasz_L_bound(fn)
        assert bound == pytest.approx(2 ** 0.5 if submodular else 3.)
        assert flops == 0
        assert SetFunction.lovasz_L_bound(fn) == (bound, 0)
    assert fn.max_abs_value == 3.


def test_known_instance_properties():
    fn = ModularSetFn(torch.tensor([[1., -2.]]))
    assert fn.is_submodular and not fn.is_non_decreasing and not fn.is_non_increasing
    assert fn.lovasz_L_bound()[0] == pytest.approx(5 ** 0.5)
    fn = SquaredCardinalitySetFn(1, 2, torch.tensor(-1.))
    assert fn.is_submodular and fn.is_non_increasing and not fn.is_non_decreasing
    bound, flops = fn.lovasz_L_bound()
    assert bound == pytest.approx(2.5 ** 0.5) and flops == 0


@pytest.mark.parametrize("weights", [[[1., 2.]], [[-1., -2.]], [[2., -3.], [0., 4.]], [[0., 0.]]])
def test_modular_lovasz_bound_is_exact_and_cached(weights):
    W = torch.tensor(weights, dtype=torch.float64).T
    fn = ModularSetFn(W)
    expected = W.square().sum().sqrt().item()
    with patch.object(fn, "_eval_batch", side_effect=AssertionError("unexpected set evaluation")):
        bound, flops = fn.lovasz_L_bound()
    assert bound == pytest.approx(expected) and flops == 0
    with patch("torch.linalg.vector_norm", side_effect=AssertionError("unexpected norm computation")):
        assert fn.lovasz_L_bound() == (bound, 0)


def test_modular_singleton_arrays_need_no_set_evaluations():
    W = torch.tensor([[2., -3.], [0., 4.]], dtype=torch.float64).T
    fn = ModularSetFn(W)
    with patch.object(fn, "_eval_batch", side_effect=AssertionError("unexpected set evaluation")):
        for method in (fn.get_singleton_vals, fn.get_singleton_removal_marginals):
            values, flops = method()
            torch.testing.assert_close(values, W)
            assert flops == 0


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("alpha", [-2., 0., 3.])
def test_squared_cardinality_lovasz_bound_matches_chain_gradient(shape, alpha):
    fn = SquaredCardinalitySetFn(*shape, torch.tensor(alpha, dtype=torch.float64))
    X = torch.arange(fn.n * fn.b, dtype=torch.float64).reshape(shape)
    gradient, *_ = fn.subgradient_lovasz_extension(X)
    with patch.object(fn, "_eval_batch", side_effect=AssertionError("unexpected set evaluation")):
        bound, flops = fn.lovasz_L_bound()
    assert bound == pytest.approx(gradient.norm().item()) and flops == 0
    with patch("llm_quick_check.attacks.submodular_utils.set_fn_instances.sqrt", side_effect=AssertionError("unexpected recomputation")):
        assert fn.lovasz_L_bound() == (bound, 0)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3)])
@pytest.mark.parametrize("alpha", [-2., 0., 3.])
def test_squared_cardinality_singleton_formulas_match_set_evaluation(shape, alpha):
    fn = SquaredCardinalitySetFn(*shape, torch.tensor(alpha, dtype=torch.float64))
    expected_singletons, _ = SetFunction._eval_singletons(fn)
    expected_removals, _ = SetFunction._eval_singletons_removal_marginals(fn)
    with patch.object(fn, "_eval_batch", side_effect=AssertionError("unexpected set evaluation")):
        singletons, singleton_flops = fn.get_singleton_vals()
        removals, removal_flops = fn.get_singleton_removal_marginals()
    torch.testing.assert_close(singletons, expected_singletons)
    torch.testing.assert_close(removals, expected_removals)
    assert singleton_flops == removal_flops == 0


@pytest.mark.parametrize("map_type", [EneReductionMap, BinaryRepresentationMap])
@pytest.mark.parametrize("n,k", [(1, 2), (2, 8)])
def test_reduction_removal_marginals_avoid_mapping_complements(map_type, n, k):
    calls = []

    def objective(x):
        calls.append(x.clone())
        x = x.to(torch.float64)
        return x.square().sum(dim=1) + x.sum(dim=1).square(), 7 * len(x)

    reduction_map = map_type(k=k, n=n, device=torch.device("cpu"))
    fn = SetFnReduction(objective, reduction_map)
    expected, expected_flops = SetFunction._eval_singletons_removal_marginals(fn)
    expected_points = calls.pop()
    with patch.object(reduction_map, "set2ints", side_effect=AssertionError("unexpected map evaluation")):
        values, flops = fn.get_singleton_removal_marginals()
        cached, cached_flops = fn.get_singleton_removal_marginals()
    torch.testing.assert_close(values, expected)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], expected_points)
    assert flops == expected_flops == 7 * (fn.n * fn.b + 1)
    assert cached is values and cached_flops == 0


def test_new_dca_surrogates_reuse_component_caches_with_current_weights():
    F = TableSetFunction()
    _, H = setfn_submodular_decomp(F, torch.tensor(-2., dtype=torch.float64))
    # Repeated components should also reuse the same cache within a combination.
    G = LinearCombinationSetFn([F, F, H], [2., -1., 1.])
    G_singletons = torch.tensor([[1., 4.], [6., 5.]], dtype=torch.float64)
    G_removals = torch.tensor([[6., 1.], [4., -1.]], dtype=torch.float64)
    for iteration, W in enumerate((torch.ones(2, 2, dtype=torch.float64),
                                   torch.tensor([[2., -1.], [0., 3.]], dtype=torch.float64))):
        surrogate = LinearCombinationSetFn([G, ModularSetFn(W)], [1., -1.])
        singletons, singleton_flops = surrogate.get_singleton_vals()
        removals, removal_flops = surrogate.get_singleton_removal_marginals()
        torch.testing.assert_close(singletons, G_singletons - W)
        torch.testing.assert_close(removals, G_removals - W)
        assert singleton_flops == (28 if iteration == 0 else 0)
        assert removal_flops == (35 if iteration == 0 else 0)
        assert F.batch_calls == 2
        assert surrogate.get_singleton_vals()[1] == 0
        assert surrogate.get_singleton_removal_marginals()[1] == 0


@pytest.mark.parametrize("weight", [-2., 0., 3.])
def test_single_element_complement_is_empty(weight):
    fn = ModularSetFn(torch.tensor([[weight]], dtype=torch.float64))
    values, flops = fn.get_singleton_removal_marginals()
    torch.testing.assert_close(values, torch.tensor([[weight]], dtype=torch.float64))
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


@pytest.mark.parametrize(
    "alphas,expected",
    [
        ([1., 2., 0.], (True, False, False)),
        ([1., 0., 0.], (True, True, False)),
        ([0., 2., 0.], (True, False, True)),
        ([1., -2., 0.], (False, True, False)),
        ([-1., 2., 0.], (False, False, True)),
        ([1., 2., 1.], (False, False, False)),
        ([0., 0., 0.], (True, True, True)),
    ],
)
def test_linear_combination_infers_properties(alphas, expected):
    increasing = ModularSetFn(torch.ones(2, 2))
    decreasing = SquaredCardinalitySetFn(2, 2, torch.tensor(-1.))
    unknown = TableSetFunction()
    fn = LinearCombinationSetFn([increasing, decreasing, unknown], alphas)
    assert (fn.is_submodular, fn.is_non_decreasing, fn.is_non_increasing) == expected
    nested = LinearCombinationSetFn([fn, unknown], [2., 0.])
    assert (nested.is_submodular, nested.is_non_decreasing, nested.is_non_increasing) == expected


@pytest.mark.parametrize("property_name", ["is_submodular", "is_non_decreasing", "is_non_increasing"])
def test_linear_combination_preserves_explicit_properties(property_name):
    component = TableSetFunction()
    # Cancellation gives the zero function, which component-wise inference cannot establish.
    fn = LinearCombinationSetFn([component, component], [1., -1.], **{property_name: True})
    for name in ("is_submodular", "is_non_decreasing", "is_non_increasing"):
        assert getattr(fn, name) == (name == property_name)


def test_linear_combination_false_properties_still_allow_inference():
    component = ModularSetFn(torch.ones(2, 2))
    fn = LinearCombinationSetFn(
        [component], [1.], is_submodular=False, is_non_decreasing=False, is_non_increasing=False,
    )
    assert fn.is_submodular and fn.is_non_decreasing and not fn.is_non_increasing


@pytest.mark.parametrize(
    "component_bounds,alphas,explicit,expected",
    [
        ([10., 10.], [2., -3.], None, 50.),
        ([10., 10.], [2., -3.], 60., 60.),
        ([10., None], [2., -3.], None, None),
        ([10., None], [2., 0.], None, 20.),
        ([None, None], [0., 0.], None, 0.),
        ([None, None], [0., 0.], 0., 0.),
    ],
)
def test_linear_combination_max_abs_value(component_bounds, alphas, explicit, expected):
    components = [TableSetFunction(), TableSetFunction()]
    for component, bound in zip(components, component_bounds):
        component.max_abs_value = bound
    fn = LinearCombinationSetFn(components, alphas, max_abs_value=explicit)
    assert fn.max_abs_value == expected
    nested = LinearCombinationSetFn([fn], [-2.])
    assert nested.max_abs_value == (None if expected is None else 2 * expected)
    assert all(component.batch_calls == 0 for component in components)


def test_linear_combination_lovasz_bound_uses_component_sum_and_counts_flops():
    class SaturatingFunction(SetFunction):
        def _eval_batch(self, S_batch):
            return torch.tensor([float(rows.numel() > 0) for rows, _ in S_batch]), 7 * len(S_batch)

    rank = SaturatingFunction(1, 16, torch.device("cpu"), is_submodular=True, is_non_decreasing=True)
    modular = ModularSetFn(torch.ones(1, 16))
    fn = LinearCombinationSetFn([rank, modular], [1., -0.1], is_submodular=True)
    # Parent endpoint bound is 0.9 * sqrt(16) = 3.6; component sum is 1 + 0.1 * 4 = 1.4.
    bound, flops = fn.lovasz_L_bound()
    assert bound == pytest.approx(1.4)
    # Rank: 16 singletons, 17 full/complement sets, then one full-set evaluation for its bound.
    assert flops == 7 * 34
    with patch.object(rank, "lovasz_L_bound", side_effect=AssertionError("unexpected bound evaluation")):
        assert fn.lovasz_L_bound() == (bound, 0)
        nested = LinearCombinationSetFn([fn], [-2.])
        assert nested.lovasz_L_bound() == (2 * bound, 0)


def test_linear_combination_lovasz_bound_keeps_smaller_parent_and_skips_zero_weights():
    modular = ModularSetFn(torch.tensor([[2., -3.]]))
    unused = ModularSetFn(torch.ones(1, 2))
    fn = LinearCombinationSetFn([modular, modular, unused], [2., -1., 0.])
    with patch.object(unused, "lovasz_L_bound", side_effect=AssertionError("zero-weight bound evaluated")):
        bound, flops = fn.lovasz_L_bound()
    # Cancellation makes the parent bound sqrt(13), versus 3 * sqrt(13) from components.
    assert bound == pytest.approx(13 ** 0.5) and flops == 0
    assert fn.lovasz_L_bound() == (bound, 0)


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
