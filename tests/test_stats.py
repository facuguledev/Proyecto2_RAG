"""Cotas de ruido: test de signos, pareados y bootstrap."""
import math

import pytest

from stats import bootstrap_ci, bootstrap_diff_ci, count, paired, paired_groups, sign_test


def test_sign_test_matches_the_numbers_in_the_assignment():
    assert sign_test(5, 0) == pytest.approx(0.0625)  # "un 5-0 da alrededor de 0,06"
    assert sign_test(6, 0) == pytest.approx(0.03125)  # "un 6-0, alrededor de 0,03"
    assert sign_test(0, 6) == sign_test(6, 0)


def test_sign_test_without_discordant_pairs_is_one():
    assert sign_test(0, 0) == 1.0
    assert sign_test(3, 3) == 1.0


def test_paired_counts_only_ids_scored_in_both_arms():
    a = {"q1": True, "q2": False, "q3": False, "q4": True}
    b = {"q1": True, "q2": True, "q3": False}  # q4 fue un error en b: no cuenta
    result = paired(a, b)
    assert (result["n"], result["wins"], result["losses"], result["ties"]) == (3, 1, 0, 2)


def test_paired_groups_votes_once_per_group():
    # Guion d1: b gana 3 turnos. Guion d2: a gana 1. Por turno serían 3-1; por guion, 1-1.
    a = {"d1.t1": False, "d1.t2": False, "d1.t3": False, "d2.t1": True}
    b = {"d1.t1": True, "d1.t2": True, "d1.t3": True, "d2.t1": False}
    result = paired_groups(a, b, {i: i.split(".")[0] for i in a})
    assert (result["groups"], result["wins"], result["losses"]) == (2, 1, 1) and result["p_sign"] == 1.0


def test_bootstrap_of_a_constant_has_no_width():
    mean, lo, hi = bootstrap_ci({f"q{i}": 1.0 for i in range(10)}, n_boot=200)
    assert mean == lo == hi == 1.0


def test_bootstrap_by_group_resamples_whole_groups():
    # Dos guiones: uno con 4 aciertos, otro con 4 fallos. Remuestreando guiones, la media solo puede ser
    # 0, 0,5 o 1; remuestreando turnos saldría casi siempre cerca de 0,5.
    values = {f"a{i}": 1.0 for i in range(4)} | {f"b{i}": 0.0 for i in range(4)}
    groups = {i: i[0] for i in values}
    mean, lo, hi = bootstrap_ci(values, groups, n_boot=2000)
    assert mean == 0.5 and lo == 0.0 and hi == 1.0


def test_bootstrap_diff_is_paired():
    a = {f"q{i}": float(i % 2) for i in range(20)}
    diff, lo, hi = bootstrap_diff_ci(a, dict(a), n_boot=200)
    assert diff == lo == hi == 0.0


def test_bootstrap_of_nothing_is_nan():
    assert all(math.isnan(x) for x in bootstrap_ci({}))


def test_count_is_a_count():
    assert count({"a": True, "b": False, "c": True}) == "2/3"
