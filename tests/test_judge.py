"""Etapa 7a: qué recibe el juez, qué se juzga y la auditoría. Sin servidor: la llamada se inyecta."""
import csv
import json
from datetime import datetime

import pytest

import llm
from evalset import Answer, Dialogue, EvalItem, EvalSet, Probe, Turn
from judge import (JudgeVerdict, Judgeable, agreement, audit_sample, audit_size, disagreements, judge_key,
                   judge_messages, judge_one, judged_row, judgeables, kind_of, latest_rows,
                   population_agreement, read_audit, refresh_audited, targets_stage4, targets_stage5,
                   write_audit)

ITEM = EvalItem(item_id="tp01", question="¿Puedo dejar el piso a los cuatro meses?", category="topical", topic="t",
                gold_sections=["BOE-A-1994-26003#a11@20261008"], answer_key=["seis meses", "6 meses"],
                expected_verdict="no")
OUT = EvalItem(item_id="oc01", question="¿Cuánto cuesta el registro en Andorra?", category="out_of_corpus",
               topic="t", answerable=False)
DIALOGUE = Dialogue(dialogue_id="dl01", turns=[
    Turn(gold_sections=["BOE-A-1994-26003#a36@20130606"], answer_key=["una mensualidad"], user="¿Fianza?",
         depends_on="standalone", oracle_rewrite="¿Cuánta fianza en un alquiler de vivienda?", assistant="Una."),
    Turn(gold_sections=["BOE-A-1994-26003#a36@20130606"], answer_key=["dos mensualidades"], user="¿Y un local?",
         depends_on="ellipsis", oracle_rewrite="¿Cuánta fianza al alquilar un local?", assistant="Dos.")])
EV = EvalSet(items=[ITEM, OUT], dialogues=[DIALOGUE], scenarios=[], file_hashes={})


def row(item_id: str, answer: str, can_answer: bool = True, correct: bool = True, arm: str = "retrieval",
        outcome: str = "answered") -> dict:
    return {"arm": arm, "item_id": item_id, "config_key": "k1",
            "answer": {"analysis": "SECRETO DEL GENERADOR", "answer": answer, "can_answer": can_answer,
                       "citations": []},
            "score": {"outcome": outcome, "correct": correct}}


def judgeable(answer: str = "No: hasta los seis meses no.", kind: str = "yes_no", can_answer: bool = True,
              contained: bool = True, item_id: str = "tp01") -> Judgeable:
    return Judgeable(stage="04", arm="retrieval", item_id=item_id, split="test", kind=kind, question="¿Puedo irme?",
                     answer_key=("seis meses",), expected_verdict="no" if kind == "yes_no" else None,
                     can_answer=can_answer, answer=answer, contained=contained)


# --- tipos y schema ------------------------------------------------------------------------------

def test_schema_reasons_first_in_both_orders():
    fields = list(JudgeVerdict.model_fields)
    assert fields == sorted(fields) == ["analysis", "verdict"]


def test_kind_of():
    assert kind_of(ITEM) == "yes_no"
    assert kind_of(OUT) == "unanswerable"
    assert kind_of(Answer(gold_sections=["x#a1@20200101"], answer_key=["k"])) == "answerable"
    probe = Probe(probe_id="p1", question="¿Y ahora?", asked_at=datetime(2026, 9, 3), ability="knowledge_update",
                  evidence_sessions=["s1"], answer_key=["310 euros"], expected_verdict="no")
    assert kind_of(probe, probe.ability) == "knowledge_update"


# --- qué recibe el juez --------------------------------------------------------------------------

def test_the_judge_never_sees_the_generators_analysis():
    found, _ = judgeables([row("tp01", "No, hacen falta seis meses.")], targets_stage4(EV), "04")
    text = json.dumps(judge_messages(found[0]), ensure_ascii=False)
    assert "SECRETO" not in text
    assert "<pregunta>" in text and "<clave>" in text and "<respuesta>" in text
    assert "seis meses | 6 meses" in text and "<veredicto_esperado>No</veredicto_esperado>" in text


def test_can_answer_goes_only_to_unanswerable_items():
    assert "<can_answer>" not in judge_messages(judgeable())[1]["content"]
    out = judge_messages(judgeable("No puedo responderlo.", kind="unanswerable", can_answer=False))
    assert "<can_answer>false</can_answer>" in out[1]["content"] and "<clave>" not in out[1]["content"]


def test_knowledge_update_rubric_accepts_the_old_value_too():
    j = Judgeable(stage="06", arm="store", item_id="p1", split="test", kind="knowledge_update", question="?",
                  answer_key=("310 euros",), expected_verdict="no", can_answer=True, answer="Antes 290, ahora 310.",
                  contained=True)
    content = judge_messages(j)[1]["content"]
    assert "mencionar la situación anterior no la hace incorrecta" in content and "veredicto" in content


def test_preferences_are_judged_literally():
    probe = Probe(probe_id="p6", question="¿Cuánto de fianza?", asked_at=datetime(2026, 9, 3), ability="preference",
                  evidence_sessions=["s1"], answer_key=["29/1994"])
    assert kind_of(probe, probe.ability) == "preference"
    assert "tal cual" in judge_messages(judgeable(kind="preference"))[1]["content"]


def test_an_abstention_on_an_answerable_item_is_incorrect_without_calling():
    def never(*args, **kw):
        raise AssertionError("no se llama al juez")
    verdict, call = judge_one(judgeable("No puedo, pero son seis meses.", can_answer=False), structured=never)
    assert verdict.verdict == "incorrect" and call["rule"] == "abstained"


# --- qué se juzga --------------------------------------------------------------------------------

def test_errors_are_not_judged():
    rows = [row("tp01", "No.", outcome="error", correct=None), {**row("oc01", "x"), "answer": None}]
    found, skipped = judgeables(rows, targets_stage4(EV), "04")
    assert found == [] and skipped["error"] == 2


def test_rows_that_are_not_answers_are_skipped_and_counted():
    retrieval_only = {"arm": "last_turn", "item_id": "dl01.t1", "metrics": {"hit@5": True}}
    found, skipped = judgeables([row("zz99", "x"), retrieval_only], targets_stage4(EV), "04")
    assert found == [] and skipped == {"unknown_id": 1, "no_answer": 1}


def test_stage5_targets_use_the_oracle_rewrite():
    targets = targets_stage5(EV)
    assert set(targets) == {"dl01.t1", "dl01.t2"}
    assert targets["dl01.t2"].question == "¿Cuánta fianza al alquilar un local?"


def test_judge_key_changes_with_the_answer():
    assert judge_key(judgeable("No, seis meses.")) != judge_key(judgeable("No, cinco meses."))
    assert judge_key(judgeable("No, seis meses.")) == judge_key(judgeable("No, seis meses."))


def write_results(tmp_path, rows, specs):
    path = tmp_path / "04_grounded_answers.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for key, eval_key in specs.items():
        (tmp_path / f"04_grounded_answers.config-{key}.json").write_text(
            json.dumps({"spec": {"eval": eval_key}}), encoding="utf-8")
    return path


def test_latest_rows_takes_the_last_config_and_the_last_row(tmp_path):
    old = {**row("tp01", "viejo"), "config_key": "k0"}
    first, last = {**row("tp01", "primera"), "config_key": "k1"}, {**row("tp01", "última"), "config_key": "k1"}
    path = write_results(tmp_path, [old, first, last], {"k0": "e", "k1": "e"})
    found = latest_rows(path, "e")
    assert [r["answer"]["answer"] for r in found] == ["última"]


def test_latest_rows_refuses_another_eval_set(tmp_path):
    path = write_results(tmp_path, [row("tp01", "x")], {"k1": "otro"})
    with pytest.raises(ValueError):
        latest_rows(path, "e")


# --- la llamada ----------------------------------------------------------------------------------

class FakeResult:
    def __init__(self, error=None):
        self.error = error

    def log(self):
        return {"error": self.error, "prompt_tokens": 100}


def test_judge_one_returns_the_verdict():
    fake = lambda messages, schema, **kw: (JudgeVerdict(analysis="dice seis meses", verdict="correct"), FakeResult())
    verdict, call = judge_one(judgeable(), structured=fake)
    assert verdict.verdict == "correct" and call["error"] is None


def test_judge_one_catches_a_context_overflow():
    def overflow(messages, schema, **kw):
        raise llm.ContextOverflow("prompt 9000 + num_predict 320 > num_ctx 8192")
    verdict, call = judge_one(judgeable(), structured=overflow)
    assert verdict is None and call["error"].startswith("CONTEXT_OVERFLOW (client)")


def test_judge_one_passes_explicit_num_ctx():
    seen = {}

    def spy(messages, schema, **kw):
        seen.update(kw)
        return JudgeVerdict(analysis="a", verdict="incorrect"), FakeResult()
    judge_one(judgeable(), structured=spy)
    assert seen["num_ctx"] == 8192 and 256 <= seen["num_predict"] <= 384


# --- lectura y auditoría -------------------------------------------------------------------------

def judged(n: int, n_disagree: int) -> list[dict]:
    rows = []
    for i in range(n):
        j = judgeable(f"respuesta {i}", item_id=f"q{i:03d}")
        verdict = JudgeVerdict(analysis="a", verdict="incorrect" if i < n_disagree else "correct")
        rows.append(judged_row(j, verdict, {"error": None}, "qwen3:8b"))
    return rows


def test_agreement_counts_and_direction():
    rows = judged(10, 3)  # contención: todas correctas; juez: 3 incorrectas
    result = agreement(rows, "contention", "judge")
    assert (result["n"], result["agree"], result["a_only"], result["b_only"]) == (10, 7, 3, 0)
    assert len(disagreements(rows)) == 3


def test_audit_size_is_30_or_20_percent():
    assert audit_size(100) == 30 and audit_size(400) == 80 and audit_size(12) == 12


def test_audit_sample_oversamples_disagreements_and_is_seeded():
    rows = judged(200, 10)
    sample = audit_sample(rows, seed=1)
    assert len(sample) == 40
    assert sum(r["stratum"] == "desacuerdo" for r in sample) == 10  # todos los desacuerdos (son menos de 20)
    assert [r["uid"] for r in sample] == [r["uid"] for r in audit_sample(rows, seed=1)]


def test_audit_roundtrip_is_blind_and_keeps_human_work(tmp_path):
    rows = judged(40, 5)
    sample = audit_sample(rows)
    assert write_audit(sample, tmp_path)
    with open(tmp_path / "audit.csv", encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    assert "judge_verdict" not in sheet[0] and "contained" not in sheet[0]  # ciega
    sheet[0]["human_verdict"], sheet[1]["human_verdict"] = "correct", "Incorrect"
    with open(tmp_path / "audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sheet)
    assert write_audit(audit_sample(rows, seed=9), tmp_path) is False  # no pisa lo completado
    done, pending = read_audit(tmp_path)
    assert len(done) == 2 and pending == len(sample) - 2
    assert {r["human_verdict"] for r in done} == {"correct", "incorrect"}
    assert all("judge_verdict" in r and "stratum" in r for r in done)
    assert population_agreement(done, "human", "judge") is not None


def test_a_sheet_saved_with_semicolons_keeps_its_verdicts(tmp_path):
    rows = judged(40, 5)
    write_audit(audit_sample(rows), tmp_path)
    with open(tmp_path / "audit.csv", encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    sheet[0]["human_verdict"] = "correct"
    with open(tmp_path / "audit.csv", "w", encoding="utf-8-sig", newline="") as f:  # como lo guarda un Excel en español
        writer = csv.DictWriter(f, fieldnames=list(sheet[0]), delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sheet)
    assert write_audit(audit_sample(rows, seed=9), tmp_path) is False
    done, _ = read_audit(tmp_path)
    assert len(done) == 1


def test_a_note_with_quotes_and_commas_does_not_shift_the_columns(tmp_path):
    write_audit(audit_sample(judged(30, 2)), tmp_path)
    with open(tmp_path / "audit.csv", encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    sheet[0]["human_verdict"], sheet[0]["human_note"] = "correct", 'dice "tres años más", pero transmite tres años'
    with open(tmp_path / "audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet[0]) + ["auditor"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({**row, "auditor": "auditor de prueba"} for row in sheet)
    done, pending = read_audit(tmp_path)
    assert len(done) == 1 and done[0]["auditor"] == "auditor de prueba" and pending == len(sheet) - 1


def test_the_sheet_does_not_reveal_the_condition(tmp_path):
    write_audit(audit_sample(judged(30, 2)), tmp_path)
    header = (tmp_path / "audit.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "arm" not in header and "stage" not in header


def test_refresh_audited_uses_the_current_judge():
    rows = judged(30, 0)
    audited = [{**rows[0], "human_verdict": "correct", "stratum": "acuerdo", "stratum_size": 30}]
    current = [{**rows[0], "judge_verdict": "incorrect"}]
    fresh, changed = refresh_audited(audited, current)
    assert changed == 1 and fresh[0]["judge_verdict"] == "incorrect"


def test_population_agreement_waits_for_every_stratum():
    audited = [{**r, "human_verdict": "correct", "stratum": "acuerdo", "stratum_size": 28} for r in judged(3, 0)]
    assert population_agreement(audited, "human", "judge", strata={"acuerdo", "desacuerdo"}) is None
    assert population_agreement(audited, "human", "judge", strata={"acuerdo"}) == 1.0


def test_read_audit_rejects_a_bad_verdict(tmp_path):
    write_audit(audit_sample(judged(30, 2)), tmp_path)
    with open(tmp_path / "audit.csv", encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    sheet[0]["human_verdict"] = "quizás"
    with open(tmp_path / "audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sheet)
    with pytest.raises(ValueError):
        read_audit(tmp_path)
