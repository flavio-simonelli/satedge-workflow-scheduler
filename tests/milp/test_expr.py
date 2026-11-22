"""Tests of the linear expressions and rows of the MILP: they must be exact and linear."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import (
    LinExpr,
    Row,
    Sense,
    screen_empty_rows,
)


def test_variable_has_unit_coefficient() -> None:
    assert dict(LinExpr.var(3).terms) == {3: 1.0}


def test_scaled_variable_keeps_the_coefficient() -> None:
    assert dict(LinExpr.var(3, 2.5).terms) == {3: 2.5}


def test_zero_coefficients_are_dropped() -> None:
    expr = LinExpr.var(0) - LinExpr.var(0)
    assert dict(expr.terms) == {}
    assert expr.constant == 0.0


def test_addition_merges_coefficients_of_the_same_variable() -> None:
    expr = LinExpr.var(0) + LinExpr.var(0, 2.0) + LinExpr.var(1)
    assert dict(expr.terms) == {0: 3.0, 1: 1.0}


def test_subtraction_and_negation() -> None:
    expr = LinExpr.var(0) - LinExpr.var(1, 4.0)
    assert dict(expr.terms) == {0: 1.0, 1: -4.0}
    assert dict((-expr).terms) == {0: -1.0, 1: 4.0}


def test_numbers_are_constants_on_both_sides() -> None:
    expr = 3 + LinExpr.var(0) - 1
    assert dict(expr.terms) == {0: 1.0}
    assert expr.constant == 2.0
    reflected = 5 - LinExpr.var(0)
    assert dict(reflected.terms) == {0: -1.0}
    assert reflected.constant == 5.0


def test_scaling_multiplies_terms_and_constant() -> None:
    expr = 2 * (LinExpr.var(0) + 3)
    assert dict(expr.terms) == {0: 2.0}
    assert expr.constant == 6.0
    assert dict((LinExpr.var(0) * 0.5).terms) == {0: 0.5}


def test_products_of_variables_are_not_defined() -> None:
    with pytest.raises(TypeError):
        _ = LinExpr.var(0) * LinExpr.var(1)  # type: ignore[operator]


def test_operations_do_not_modify_their_operands() -> None:
    x = LinExpr.var(0)
    _ = x + LinExpr.var(1)
    _ = x * 3
    assert dict(x.terms) == {0: 1.0}


def test_in_place_addition_updates_the_left_operand() -> None:
    x = LinExpr.var(0)
    alias = x
    x += LinExpr.var(1)
    assert x is alias
    assert dict(x.terms) == {0: 1.0, 1: 1.0}


def test_terms_view_is_read_only() -> None:
    expr = LinExpr.var(0)
    with pytest.raises(TypeError):
        expr.terms[0] = 5.0  # type: ignore[index]


def test_total_of_nothing_is_zero() -> None:
    expr = LinExpr.total([])
    assert dict(expr.terms) == {}
    assert expr.constant == 0.0


def test_total_mixes_expressions_and_numbers() -> None:
    expr = LinExpr.total([LinExpr.var(0), 2.0, LinExpr.var(0, 3.0), LinExpr.var(1)])
    assert dict(expr.terms) == {0: 4.0, 1: 1.0}
    assert expr.constant == 2.0


def test_expressions_are_not_hashable() -> None:
    with pytest.raises(TypeError):
        hash(LinExpr.var(0))


def test_comparison_with_a_number_gives_a_row() -> None:
    row = LinExpr.var(0) + LinExpr.var(1) <= 1
    assert isinstance(row, Row)
    assert row.sense is Sense.LE
    assert row.rhs == 1.0
    assert dict(row.terms) == {0: 1.0, 1: 1.0}


def test_constants_on_the_left_move_to_the_right() -> None:
    row = LinExpr.var(0) + 2 >= 5
    assert row.sense is Sense.GE
    assert row.rhs == 3.0


def test_variables_on_both_sides_move_to_the_left() -> None:
    # x + 2y - 3 <= 1 - y  is  x + 3y <= 4
    row = LinExpr.var(0) + 2 * LinExpr.var(1) - 3 <= 1 - LinExpr.var(1)
    assert dict(row.terms) == {0: 1.0, 1: 3.0}
    assert row.rhs == 4.0


def test_equality_gives_an_equality_row() -> None:
    row = LinExpr.var(0) - LinExpr.var(0) + LinExpr.var(1) == 1
    assert row.sense is Sense.EQ
    assert dict(row.terms) == {1: 1.0}
    assert row.rhs == 1.0


def test_row_name_is_kept_and_can_be_changed() -> None:
    row = (LinExpr.var(0) <= 1).named("cap[0]")
    assert row.name == "cap[0]"
    assert Row(row.terms, row.sense, row.rhs).name == ""


@given(
    st.lists(
        st.tuples(st.integers(0, 4), st.floats(-10, 10, allow_nan=False)),
        max_size=12,
    ),
    st.floats(-10, 10, allow_nan=False),
)
def test_total_equals_repeated_addition(
    items: list[tuple[int, float]], constant: float
) -> None:
    parts = [LinExpr.var(v, c) for v, c in items] + [constant]
    by_hand = LinExpr()
    for part in parts:
        by_hand = by_hand + part
    total = LinExpr.total(parts)
    assert total.constant == pytest.approx(by_hand.constant)
    assert set(total.terms) == set(by_hand.terms)
    for var in total.terms:
        assert total.terms[var] == pytest.approx(by_hand.terms[var])


def test_a_row_without_variables_that_holds_is_dropped() -> None:
    kept = list(
        screen_empty_rows(
            [Row({}, Sense.LE, 1.0, "trivial"), Row({0: 1.0}, Sense.LE, 1.0)]
        )
    )
    assert [r.name for r in kept] == [""]


def test_a_row_without_variables_that_never_holds_is_a_model_error() -> None:
    with pytest.raises(ModelBuildError, match="never satisfied"):
        list(screen_empty_rows([Row({}, Sense.GE, 1.0, "impossible")]))


def test_equality_without_variables_needs_a_zero_right_hand_side() -> None:
    assert list(screen_empty_rows([Row({}, Sense.EQ, 0.0)])) == []
    with pytest.raises(ModelBuildError):
        list(screen_empty_rows([Row({}, Sense.EQ, 2.0, "eq")]))


def test_zero_terms_do_not_reach_a_row_built_from_rejection() -> None:
    rejection = LinExpr.number(0)
    row = LinExpr.var(0) >= 3 * (1 - rejection)
    assert dict(row.terms) == {0: 1.0}
    assert row.rhs == 3.0
