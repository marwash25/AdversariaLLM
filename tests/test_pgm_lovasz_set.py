"""Focused checks for generic PGM, cached auxiliary points, and DCA integration."""

from unittest.mock import patch

import pytest
import torch

from llm_quick_check.attacks.submodular_utils import (
    EneReductionMap,
    LinearCombinationSetFn,
    ModularSetFn,
    SetFnReduction,
    make_lattice_prefix_filter,
    dca_dsm,
    pgm_lovasz_set,
    trim_ground_set,
)


def test_trimming_preserves_zero_marginals_and_updates_input():
    fn = ModularSetFn(torch.tensor([[-2., 0.], [3., -1.]], dtype=torch.float64))
    X_init = torch.full((2, 2), 0.4, dtype=torch.float64)
    X, flops = trim_ground_set(fn, X_init)
    torch.testing.assert_close(X, torch.tensor([[1., 0.4], [0., 1.]], dtype=torch.float64))
    assert X is X_init
    assert flops == 0


@pytest.mark.parametrize("is_submodular", [False, True])
@pytest.mark.parametrize("L", [10., "L_bound"])
def test_pgm_trimming_initialization_updates_and_flops(is_submodular, L):
    # Submodular negative pair interaction: element 0 must be included,
    # element 1 excluded, and element 2 has singleton +1 but final marginal -1.
    def objective(x):
        x = x.to(torch.float64)
        return -x[:, 0] + 4 * x[:, 1] + x[:, 2] - 2 * x[:, 0] * x[:, 2], 7 * len(x)

    fn = SetFnReduction(objective, EneReductionMap(k=2, n=3, device=torch.device("cpu")))
    fn.is_submodular = is_submodular
    X_init = torch.tensor([[0.2], [0.8], [0.4]], dtype=torch.float64)
    iterates = []
    original_subgradient = fn.subgradient_lovasz_extension

    def record_iterate(X, tie_breaker):
        iterates.append(X.clone())
        return original_subgradient(X, tie_breaker)

    with patch.object(fn, "subgradient_lovasz_extension", side_effect=record_iterate), \
         patch.object(fn, "get_singleton_removal_marginals", wraps=fn.get_singleton_removal_marginals) as complements:
        result = pgm_lovasz_set(fn, X_init, num_steps=3, L=L)

    assert complements.call_count == int(is_submodular) + int(L == "L_bound")
    torch.testing.assert_close(X_init, torch.tensor([[0.2], [0.8], [0.4]], dtype=torch.float64))
    setup_flops = 49 if is_submodular or L == "L_bound" else 0
    assert result[8] == [21 + setup_flops, 21, 21, 21]
    if is_submodular:
        torch.testing.assert_close(iterates[0], torch.tensor([[1.], [0.], [0.4]], dtype=torch.float64))
        assert all(X[0, 0] == 1 and X[1, 0] == 0 for X in iterates)
    else:
        torch.testing.assert_close(iterates[0], X_init)
    repeated = pgm_lovasz_set(fn, X_init, num_steps=3, L=L)
    assert repeated[8] == [21, 21, 21, 21]
    assert repeated[2] == result[2]
    assert repeated[4] == pytest.approx(result[4])


@pytest.mark.parametrize("reject_all", [False, True])
@pytest.mark.parametrize("is_submodular", [False, True])
def test_general_pgm_projection_gap_and_filtered_history(reject_all, is_submodular):
    fn = ModularSetFn(torch.tensor([[2., -1.]], dtype=torch.float64))
    if not is_submodular:
        fn.is_submodular = False
    X = torch.tensor([[0.9, 0.1]], dtype=torch.float64)
    result = pgm_lovasz_set(
        fn, X, num_steps=5, L=1., gap_tol=0.,
        filter_prefixes=(lambda order, auxiliary: []) if reject_all else None,
    )
    best, continuous, values, filtered, continuous_values, gaps, sets, times, flops, auxiliary, auxiliary_history = result
    torch.testing.assert_close(best, (torch.tensor([0]), torch.tensor([1])))
    torch.testing.assert_close(continuous, torch.tensor([[0., 1.]], dtype=torch.float64))
    assert values == ([-1.] if is_submodular else [0., -1.])
    assert continuous_values == pytest.approx([-1.] if is_submodular else [1.7, -1.])
    assert gaps == ([0.] if is_submodular else [1., 0.])
    steps = len(values)
    assert filtered == ([float("inf")] * steps if reject_all else values)
    assert len(times) == steps and flops == [0] * steps
    assert auxiliary is None and auxiliary_history == [None] * steps
    if reject_all:
        assert sets == [None] * steps
    else:
        if not is_submodular:
            assert sets[0][0].numel() == sets[0][1].numel() == 0
        torch.testing.assert_close(sets[-1], best)
        assert sets[-1][1].data_ptr() != best[1].data_ptr()
    torch.testing.assert_close(X, torch.tensor([[0.9, 0.1]], dtype=torch.float64))


@pytest.mark.parametrize("L", [3., "L_bound", "normalize", "polyak"])
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
    chain_flops = 7 * fn.n * fn.b
    bound_flops = 7 * (2 * fn.n * fn.b + 1) if L == "L_bound" else 0
    assert flops == [chain_flops + bound_flops] + [chain_flops] * (len(values) - 1)
    assert fn.eval_single(best)[0].item() == min(values)
    torch.testing.assert_close(reduction_map.set2ints([best])[0], best_point)
    for selected, point, value in zip(sets, points, filtered_values):
        torch.testing.assert_close(reduction_map.set2ints([selected])[0], point)
        assert fn.eval_single(selected)[0].item() == value


def test_iteration_zero_preserves_actual_set_and_copies_cached_point():
    reduction_map = EneReductionMap(k=3, n=1, device=torch.device("cpu"))
    calls = []

    def lattice_value(x):
        calls.append(x)
        return -2. * x[:, 0], 7 * len(x)

    fn = LinearCombinationSetFn(
        [SetFnReduction(lattice_value, reduction_map), ModularSetFn(torch.tensor([[1., 3.]]))],
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
    assert fn.eval_single((torch.tensor([0]), torch.tensor([1])))[0].item() == 1.


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
        [SetFnReduction(lattice_value, reduction_map), ModularSetFn(weights)], [1., 1.],
    )
    with patch.object(reduction_map, "ints2binary", side_effect=AssertionError("unexpected inverse mapping")), \
         patch.object(reduction_map, "set2ints", side_effect=AssertionError("unexpected remapping")):
        result = pgm_lovasz_set(
            fn, X, num_steps=0, L=1.,
            filter_prefixes=None if prefixes is None else lambda order, auxiliary: prefixes,
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
