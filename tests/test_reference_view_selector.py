"""Reference-view selection in ``model/reference_view_selector.py``.

The backbone moves one view to slot 0, runs global attention, and puts the
views back. If the "put back" ever stops being the exact inverse of the
"move", depth maps get attributed to the wrong input images -- output shapes,
dtypes and finiteness all stay correct, so only an explicit permutation test
catches it. That inverse is the centrepiece here; the strategies themselves
are pinned by construction rather than by recording what they currently
return.
"""

import typing
import pytest
import torch

from depth_anything_3.model.reference_view_selector import (
    RefViewStrategy,
    reorder_by_reference,
    restore_original_order,
    select_reference_view,
)

STRATEGIES = list(typing.get_args(RefViewStrategy))


def _tokens(b: int, s: int, n: int = 5, c: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, s, n, c, generator=g, dtype=torch.float64)


def _class_tokens(rows: list[list[float]], n: int = 3) -> torch.Tensor:
    """Build (1, S, n, C) where token 0 of each view is the given vector."""
    x = torch.zeros(1, len(rows), n, len(rows[0]), dtype=torch.float64)
    x[0, :, 0] = torch.tensor(rows, dtype=torch.float64)
    # Non-class tokens are never read by the selector; make them non-zero anyway.
    x[0, :, 1:] = 0.5
    return x


# ---------------------------------------------------------------------------
# strategy dispatch
# ---------------------------------------------------------------------------
def test_the_literal_lists_exactly_the_strategies_that_work():
    """``RefViewStrategy`` is the type the CLI help and the gradio dropdown
    are written from. If it drifts from the dispatcher, one of them lies."""
    x = _tokens(1, 4)
    for strategy in STRATEGIES:
        select_reference_view(x, strategy=strategy)
    assert set(STRATEGIES) == {"first", "middle", "saddle_balanced", "saddle_sim_range"}


def test_unknown_strategy_raises_and_names_the_alternatives():
    with pytest.raises(ValueError) as excinfo:
        select_reference_view(_tokens(1, 4), strategy="nope")
    for strategy in STRATEGIES:
        assert strategy in str(excinfo.value)


def test_a_single_view_short_circuits_before_the_strategy_is_read():
    """Documented behaviour, and the reason a bad ``--ref-view-strategy``
    survives single-image runs: the S <= 1 guard comes first."""
    assert torch.equal(
        select_reference_view(_tokens(2, 1), strategy="nope"), torch.zeros(2, dtype=torch.long)
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_selection_returns_one_valid_index_per_batch_item(strategy):
    b, s = 3, 5
    idx = select_reference_view(_tokens(b, s), strategy=strategy)
    assert idx.shape == (b,) and idx.dtype == torch.long
    assert torch.all((idx >= 0) & (idx < s))


def test_first_and_middle_are_purely_positional():
    x = _tokens(2, 5)
    assert torch.equal(select_reference_view(x, "first"), torch.zeros(2, dtype=torch.long))
    assert torch.equal(select_reference_view(x, "middle"), torch.full((2,), 2, dtype=torch.long))


def test_saddle_sim_range_picks_the_view_with_the_widest_similarity_spread():
    """Constructed by hand: view 0 is simultaneously near-parallel to view 3
    (cos 0.8) and anti-parallel to view 1 (cos -1), giving it a spread of 1.8
    -- strictly the largest of the four."""
    x = _class_tokens([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0.8, 0.6, 0]])
    assert select_reference_view(x, "saddle_sim_range").item() == 0


def test_saddle_sim_range_ignores_the_self_similarity():
    """Every view is perfectly similar to itself; if the diagonal were not
    removed, all spreads would be dominated by that 1.0 and the choice would
    be arbitrary. Two identical views make the omission observable."""
    x = _class_tokens([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    # Rows 0 and 1 see {1.0, 0.0}; row 2 sees {0.0, 0.0}. So 2 is never chosen.
    assert select_reference_view(x, "saddle_sim_range").item() in (0, 1)


@pytest.mark.parametrize("strategy", ["saddle_balanced", "saddle_sim_range"])
def test_feature_strategies_follow_the_view_not_the_slot(strategy):
    """Permuting the input views must move the chosen index with them: the
    metrics are per-view and symmetric, so the *same picture* has to win."""
    x = _tokens(1, 5, seed=3)
    perm = torch.tensor([3, 1, 4, 0, 2])
    chosen = select_reference_view(x, strategy).item()
    chosen_permuted = select_reference_view(x[:, perm], strategy).item()
    assert perm[chosen_permuted].item() == chosen


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_selection_is_deterministic(strategy):
    x = _tokens(2, 4, seed=11)
    assert torch.equal(select_reference_view(x, strategy), select_reference_view(x, strategy))


# ---------------------------------------------------------------------------
# reorder / restore
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s", [2, 3, 5, 8])
def test_reorder_moves_the_reference_first_and_keeps_the_rest_in_order(s):
    x = torch.arange(s, dtype=torch.float64).reshape(1, s, 1, 1)
    for ref in range(s):
        idx = torch.tensor([ref])
        expected = [ref] + [i for i in range(s) if i != ref]
        got = reorder_by_reference(x, idx).flatten().tolist()
        assert got == [float(i) for i in expected], (s, ref)


@pytest.mark.parametrize("s", [2, 3, 5, 8])
def test_restore_is_the_exact_inverse_of_reorder(s):
    x = _tokens(1, s, seed=5)
    for ref in range(s):
        idx = torch.tensor([ref])
        assert torch.equal(restore_original_order(reorder_by_reference(x, idx), idx), x)


def test_each_batch_item_uses_its_own_reference_index():
    b, s = 4, 6
    x = torch.arange(b * s, dtype=torch.float64).reshape(b, s, 1, 1)
    idx = torch.tensor([0, 5, 2, 3])
    reordered = reorder_by_reference(x, idx)
    torch.testing.assert_close(reordered[:, 0, 0, 0], x[torch.arange(b), idx, 0, 0])
    assert torch.equal(restore_original_order(reordered, idx), x)


@pytest.mark.parametrize("shape", [(1, 4, 7), (1, 4, 7, 9), (2, 4, 3, 5, 6)])
def test_reorder_works_for_any_trailing_shape(shape):
    """The backbone applies it to (B, S, N, C) tokens and the loop applies it
    again to ``local_x``; the helper must not assume a rank."""
    x = torch.randn(*shape, dtype=torch.float64)
    idx = torch.full((shape[0],), 2, dtype=torch.long)
    assert torch.equal(restore_original_order(reorder_by_reference(x, idx), idx), x)


def test_single_view_reorder_is_a_no_op():
    x = _tokens(2, 1)
    idx = torch.zeros(2, dtype=torch.long)
    assert reorder_by_reference(x, idx) is x
    assert restore_original_order(x, idx) is x
