"""``tools/recompute_tables.py`` copia las reglas de puntaje para no depender de ``src/``. Estos tests verifican
que la copia diga lo mismo que el original, y, si los artefactos de la corrida están, que re-puntúe igual."""
import importlib.util
from pathlib import Path

import pytest

import evalset
import grounded
from evalset import Answer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recompute_tables", ROOT / "tools" / "recompute_tables.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

TEXTS = ["Sí, una mensualidad de renta.", "Si el contrato lo dice, 1.500 euros", "No obstante, 10,5 meses",
         "El 100% de la renta", "artículo 10.1 de la LAU", "Cinco años (5 años)", "1300,00 €", "",
         "No puedo responderlo con los pasajes disponibles.", "Sí: el interés legal.", "SÍ, claro"]
KEYS = [["una mensualidad"], ["1500 euros"], ["10.5 meses"], ["10"], ["artículo 10"], ["5 años"], ["1300"],
        ["interés legal"], ["si"]]


def test_normalize_and_contains_key_match_the_project():
    for text in TEXTS:
        assert rt.normalize(text) == evalset.normalize(text)
        for keys in KEYS:
            assert rt.contains_key(text, keys) == evalset.contains_key(text, keys), (text, keys)


def test_verdict_matches_the_project():
    for text in TEXTS:
        assert rt.verdict_of(text) == grounded.verdict_of(text)


@pytest.mark.parametrize("can_answer", [True, False])
@pytest.mark.parametrize("verdict", [None, "yes", "no"])
def test_correct_matches_grounded_score(can_answer, verdict):
    target = Answer(gold_sections=["BOE-A-1994-26003#a36@20130606"], answer_key=["una mensualidad"],
                    expected_verdict=verdict)
    for text in TEXTS:
        answer = grounded.GroundedAnswer(analysis="x", answer=text, can_answer=can_answer, citations=[])
        expected = grounded.score(answer, {}, target, [], grounded.Gold(set(), set()))["correct"]
        assert rt.correct(answer.model_dump(), target.model_dump()) == expected, text


def test_unanswerable_is_correct_iff_abstains():
    target = {"answerable": False, "answer_key": [], "expected_verdict": None}
    assert rt.correct({"answer": "x", "can_answer": False}, target) is True
    assert rt.correct({"answer": "x", "can_answer": True}, target) is False


def test_call_errors_and_answers_at_the_cap_are_their_own_category():
    target = {"answerable": True, "answer_key": ["a"], "expected_verdict": None}
    assert rt.correct(None, target) is None
    assert rt.correct({"answer": "a", "can_answer": True}, target, "CONTEXT_OVERFLOW") is None
    assert rt.correct({"answer": "a" * rt.ANSWER_MAX, "can_answer": True}, target) is None


def test_sign_test_and_group_votes():
    assert rt.sign_test(5, 0) == pytest.approx(0.0625)
    assert rt.sign_test(0, 0) == 1.0
    a = {"t1": False, "t2": False, "t3": True}
    b = {"t1": True, "t2": True, "t3": False}
    assert rt.paired(a, b)[:2] == (2, 1)
    assert rt.paired(a, b, {"t1": "g1", "t2": "g1", "t3": "g2"})[:2] == (1, 1)


RESULTS = ROOT / "artifacts" / "full" / "results"


@pytest.mark.skipif(not (RESULTS / "04_grounded_answers.jsonl").is_file(), reason="sin los artefactos de la corrida")
def test_rescoring_the_saved_rows_gives_the_saved_scores():
    items = {r["item_id"]: r for r in rt.read_jsonl(ROOT / "eval" / "single_turn.jsonl")}
    for row in rt.read_jsonl(RESULTS / "04_grounded_answers.jsonl"):
        assert rt.correct(row["answer"], items[row["item_id"]], row.get("error")) == row["score"].get("correct")


@pytest.mark.skipif(not (RESULTS / "04_grounded_answers.jsonl").is_file(), reason="sin los artefactos de la corrida")
def test_the_tool_runs_on_a_partial_delivery(tmp_path, monkeypatch, capsys):
    """Con el eval set y una sola etapa, el informe sale igual: las etapas sin archivo dicen qué falta."""
    for name in ("single_turn.jsonl", "dialogues.jsonl", "scenarios.jsonl", "FROZEN.json"):
        (tmp_path / "eval").mkdir(exist_ok=True)
        (tmp_path / "eval" / name).write_bytes((ROOT / "eval" / name).read_bytes())
    results = tmp_path / "artifacts" / "full" / "results"
    results.mkdir(parents=True)
    (results / "04_grounded_answers.jsonl").write_bytes((RESULTS / "04_grounded_answers.jsonl").read_bytes())
    monkeypatch.setattr("sys.argv", ["recompute_tables.py", "--root", str(tmp_path)])
    assert rt.main() == 0
    out = capsys.readouterr().out
    assert "| retrieval | 18/30 | 15/26 | 3/4 | 0 |" in out
    assert "sin `results/03_retrieval.jsonl`" in out


def test_an_edited_eval_set_is_reported(tmp_path, monkeypatch, capsys):
    for name in ("single_turn.jsonl", "dialogues.jsonl", "scenarios.jsonl", "FROZEN.json"):
        (tmp_path / "eval").mkdir(exist_ok=True)
        (tmp_path / "eval" / name).write_bytes((ROOT / "eval" / name).read_bytes())
    with open(tmp_path / "eval" / "dialogues.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    monkeypatch.setattr("sys.argv", ["recompute_tables.py", "--root", str(tmp_path)])
    assert rt.main() == 1
    assert "CAMBIARON" in capsys.readouterr().out
