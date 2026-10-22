"""Focused checks for generic PGM, cached auxiliary points, and DCA integration."""

from unittest.mock import patch

import pytest
import torch

from llm_quick_check.attacks.submodular_utils import (
    EneReductionMap,
    LinearCombinationSetFn,
    ModularSetFunction,
    SetFnReduction,
    make_lattice_prefix_filter,
    dca_dsm,
    pgm_lovasz_set,
)


@pytest.mark.parametrize("reject_all", [False, True])
def test_general_pgm_projection_gap_and_filtered_history(reject_all):
    fn = ModularSetFunction(torch.tensor([[2., -1.]], dtype=torch.float64))
    X = torch.tensor([[0.9, 0.1]], dtype=torch.float64)
    result = pgm_lovasz_set(
        fn, X, num_steps=5, L=1., gap_tol=0.,
        filter_prefixes=(lambda rows, cols, auxiliary: []) if reject_all else None,
    )
    best, continuous, values, filtered, continuous_values, gaps, sets, times, flops, auxiliary, auxiliary_history = result
    torch.testing.assert_close(best, (torch.tensor([0]), torch.tensor([1])))
    torch.testing.assert_close(continuous, torch.tensor([[0., 1.]], dtype=torch.float64))
    assert values == [0., -1.]
    assert continuous_values == pytest.approx([1.7, -1.])
    assert gaps == [1., 0.]
    assert filtered == ([float("inf")] * 2 if reject_all else values)
    assert len(times) == 2 and flops == [0, 0]
    assert auxiliary is None and auxiliary_history == [None, None]
    if reject_all:
        assert sets == [None, None]
    else:
        assert sets[0][0].numel() == sets[0][1].numel() == 0
        torch.testing.assert_close(sets[1], best)
        assert sets[1][1].data_ptr() != best[1].data_ptr()
    torch.testing.assert_close(X, torch.tensor([[0.9, 0.1]], dtype=torch.float64))


@pytest.mark.parametrize("L", [3., "singletons", "normalize", "polyak"])
@pytest.mark.parametrize("gap_tol", [None, 1e-6])
def test_reduction_set_histories_align_with_stacked_lattice_points(L, gap_tol):
    reduction_map = EneReductionMap(k=4, n=2, device=torch.device("cpu"))
    fn = SetFnReduction(lambda x: (x[:, 0].float() - x[:, 1].float().square(), 7 * len(x)), reduction_map)
    X = torch.full((fn.n, fn.b), 0.5)
    settings = dict(
        num_steps=3, L=L, gap_tol=gap_tol, tie_break="random",
        filter_prefixes=make_lattice_prefix_filter(lambda chain: list(range(0, len(chain), 2)), True),
    )
    torch.manual_seed(12)
    result = pgm_lovasz_set(fn, X, **settings)
    best, _, values, filtered_values, _, _, sets, times, flops, best_point, points = result
    assert points.shape == (len(values), fn.n) and points.dtype == torch.long
    assert len(times) == len(values)
    assert flops == [7 * fn.n * fn.b * (1 + (L == "singletons"))] + [7 * fn.n * fn.b] * (len(values) - 1)
    assert fn.eval_single(*best)[0].item() == min(values)
    torch.testing.assert_close(reduction_map.set2ints([best[0]], [best[1]])[0], best_point)
    for selected, point, value in zip(sets, points, filtered_values):
        torch.testing.assert_close(reduction_map.set2ints([selected[0]], [selected[1]])[0], point)
        assert fn.eval_single(*selected)[0].item() == value


def test_iteration_zero_preserves_actual_set_and_copies_cached_point():
    reduction_map = EneReductionMap(k=3, n=1, device=torch.device("cpu"))
    calls = []

    def lattice_value(x):
        calls.append(x)
        return -2. * x[:, 0], 7 * len(x)

    fn = LinearCombinationSetFn(
        [SetFnReduction(lattice_value, reduction_map), ModularSetFunction(torch.tensor([[1., 3.]]))],
        [1., 1.],
    )
    result = pgm_lovasz_set(fn, torch.tensor([[1, 0]]), num_steps=0, L=1.)
    best, _, values, _, _, gaps, sets, _, flops, point, points = result
    torch.testing.assert_close(best, (torch.tensor([0]), torch.tensor([0])))
    assert values == [-1.] and gaps == [0.] and flops == [14]
    torch.testing.assert_close(point, torch.tensor([1]))
    torch.testing.assert_close(sets[0], best)
    torch.testing.assert_close(points[0], point)
    assert len(calls) == 1
    for retained in (*best, *sets[0], point, points[0]):
        assert retained.untyped_storage().nbytes() == retained.numel() * retained.element_size()
    # The other singleton maps to the same point but has a different value.
    assert fn.eval_single(torch.tensor([0]), torch.tensor([1]))[0].item() == 1.


def test_reduction_accepts_initialization_converted_by_caller():
    reduction_map = EneReductionMap(k=3, n=2, device=torch.device("cpu"))
    fn = SetFnReduction(lambda x: (-x.float().sum(dim=1), 7 * len(x)), reduction_map)
    X = reduction_map.ints2binary(torch.tensor([[1, 2]]))[0].long()
    with patch.object(reduction_map, "ints2binary", side_effect=AssertionError("unexpected inverse mapping")):
        result = pgm_lovasz_set(fn, X, num_steps=0, L=1.)
    assert len(result) == 11
    torch.testing.assert_close(result[1], X)
    torch.testing.assert_close(result[9], torch.tensor([2, 2]))
    assert result[2] == [-4.] and result[4] == [-3.] and result[8] == [28]
    assert result[10].shape == (1, 2) and result[10].dtype == torch.long


@pytest.mark.parametrize("prefixes", [None, [0], [1], []])
def test_pgm_preserves_mixed_set_values_and_cached_points(prefixes):
    reduction_map = EneReductionMap(k=3, n=1, device=torch.device("cpu"))
    # Choose the singleton that R would not recover from the shared lattice point.
    X = 1 - reduction_map.ints2binary(torch.tensor([[1]]))[0].long()
    weights = torch.where(X.bool(), 1., 3.)
    calls = []

    def lattice_value(x):
        calls.append(x)
        return -2. * x[:, 0], 7 * len(x)

    fn = LinearCombinationSetFn(
        [SetFnReduction(lattice_value, reduction_map), ModularSetFunction(weights)], [1., 1.],
    )
    with patch.object(reduction_map, "ints2binary", side_effect=AssertionError("unexpected inverse mapping")), \
         patch.object(reduction_map, "set2ints", side_effect=AssertionError("unexpected remapping")):
        result = pgm_lovasz_set(
            fn, X, num_steps=0, L=1.,
            filter_prefixes=None if prefixes is None else lambda rows, cols, auxiliary: prefixes,
        )
    assert len(calls) == 1 and len(result) == 11
    torch.testing.assert_close(result[0], X.nonzero(as_tuple=True))
    torch.testing.assert_close(result[9], torch.tensor([1]))
    torch.testing.assert_close(result[1], X)
    assert result[2] == [-1.] and result[4] == [-1.] and result[8] == [14]
    assert result[10].shape == (1, 1) and result[10].dtype == torch.long
    if prefixes == []:
        assert result[3] == [float("inf")]
        assert result[6] == [None]
    else:
        assert result[3] == ([0.] if prefixes == [0] else [-1.])
        torch.testing.assert_close(result[10], torch.tensor([[0 if prefixes == [0] else 1]]))


def test_reduction_accepts_one_element_continuous_warm_start():
    reduction_map = EneReductionMap(k=2, n=1, device=torch.device("cpu"))
    fn = SetFnReduction(lambda x: (-x[:, 0].float(), 0), reduction_map)
    X = torch.tensor([[0.25]], dtype=torch.float64)
    with patch.object(reduction_map, "ints2binary", side_effect=AssertionError("unexpected inverse mapping")):
        result = pgm_lovasz_set(fn, X, num_steps=0, L=1.)
    torch.testing.assert_close(result[1], X)
    assert result[4] == [-0.25]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("reject_all", [False, True])
def test_dca_uses_auxiliary_points_for_filtered_lattice_history(batched, reject_all):
    reduction_map = EneReductionMap(k=3, n=1, device=torch.device("cpu"))
    G = SetFnReduction(lambda x: (-3. * x[:, 0].square(), 0), reduction_map)
    H = SetFnReduction(lambda x: (-x[:, 0].float().square(), 0), reduction_map)
    F = SetFnReduction(lambda x: (-2. * x[:, 0].square(), 0), reduction_map)
    result = dca_dsm(
        F, G, H, torch.tensor([[1]]) if batched else torch.tensor([1]),
        num_outer_steps=2, num_inner_steps=2, inner_solver="pgm", L_G=5.,
        filter_fn=(lambda chain: []) if reject_all else lambda chain: (chain[:, 0] == 1).nonzero().flatten().tolist(),
        filter_zero=True,
    )
    values, filtered_values, continuous_values, points = result[:4]
    assert values == [-8., -8.] and continuous_values == [-2., -2.]
    assert points.shape == (2, 1) and points.dtype == torch.long
    assert result[6] == [[], [-8.]]
    if reject_all:
        assert filtered_values == [float("inf"), float("inf")]
        assert result[7] == [[], [float("inf")]]
    else:
        assert filtered_values == [-2., -2.]
        assert result[7] == [[], [-2.]]
        torch.testing.assert_close(points, torch.tensor([[1], [1]]))
