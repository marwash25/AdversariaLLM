"""Focused checks for generic PGM and preservation of the lattice solver's loop."""

from unittest.mock import patch

import pytest
import torch

from llm_quick_check.attacks.submodular_utils import (
    EneReductionMap,
    LinearCombinationSetFn,
    ModularSetFunction,
    SetFnReduction,
    make_lattice_prefix_filter,
    pgm_lovasz,
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
def test_reduction_matches_existing_pgm_histories(L, gap_tol):
    reduction_map = EneReductionMap(k=4, n=2, device=torch.device("cpu"))
    fn = SetFnReduction(lambda x: (x[:, 0].float() - x[:, 1].float().square(), 7 * len(x)), reduction_map)
    X = torch.full((fn.n, fn.b), 0.5)
    settings = dict(
        num_steps=3, L=L, gap_tol=gap_tol, tie_break="random",
        filter_prefixes=make_lattice_prefix_filter(lambda chain: list(range(0, len(chain), 2)), True),
    )
    results = []
    for solver in (pgm_lovasz, pgm_lovasz_set):
        torch.manual_seed(12)
        with patch("llm_quick_check.attacks.submodular_utils.dsm_optimizers.time.time", side_effect=range(100)):
            results.append(solver(fn, X, **settings))
    old, new = results
    torch.testing.assert_close(new[1], old[1])
    for i in (2, 3, 4, 5, 7, 8):
        assert new[i] == old[i]
    torch.testing.assert_close(new[9], old[0])
    torch.testing.assert_close(torch.stack(new[10]), old[6])
    for selected, point in zip(new[6], new[10]):
        torch.testing.assert_close(reduction_map.set2ints([selected[0]], [selected[1]])[0], point)


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
