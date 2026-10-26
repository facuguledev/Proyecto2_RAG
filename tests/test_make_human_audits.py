"""``tools/make_human_audits.py``: la muestra de reescrituras es la de la Etapa 5, y una planilla con respuestas
no se pisa nunca."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_human_audits", ROOT / "tools" / "make_human_audits.py")
mha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mha)


def test_the_rewrite_sample_is_the_one_stage_5_read():
    dialogues = mha.read_jsonl(ROOT / "eval" / "dialogues.jsonl")
    chosen = mha.rewrite_sample(dialogues)
    assert len(chosen) == 20
    assert {turn_id for turn_id, _ in chosen} == set(mha.MODEL_REWRITE_LABELS)


def test_a_sheet_with_answers_is_never_overwritten(tmp_path):
    path = tmp_path / "s.csv"
    assert mha.write_sheet(path, ["id", "human_verdict"], [{"id": "1"}], "human_verdict").startswith("escrita")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("id;human_verdict\n1;correct\n")  # guardada desde un Excel en español, con ";"
    assert mha.write_sheet(path, ["id", "human_verdict"], [{"id": "1"}], "human_verdict").startswith("se conserva")
    assert "correct" in path.read_text(encoding="utf-8-sig")


def test_the_judge_sheet_hides_who_audited_and_what_they_said(tmp_path):
    art = tmp_path / "artifacts" / "full"
    (art / "judge").mkdir(parents=True)
    with open(art / "judge" / "audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mha.JUDGE_COLUMNS + ["auditor"])
        writer.writeheader()
        writer.writerow({"audit_id": "a001", "kind": "answerable", "question": "¿?", "answer_key": "x",
                         "expected_verdict": "", "can_answer": "True", "answer": "x", "human_verdict": "correct",
                         "human_note": "nota del modelo", "auditor": "un modelo"})
    (art / "judge" / "audit_key.jsonl").write_text(json.dumps({"audit_id": "a001"}) + "\n", encoding="utf-8")
    mha.judge_sheet(art)
    rows = mha.read_sheet(art / "judge_human" / "audit.csv")
    assert list(rows[0]) == mha.JUDGE_COLUMNS
    assert rows[0]["human_verdict"] == "" and rows[0]["human_note"] == ""
    assert (art / "judge_human" / "audit_key.jsonl").is_file()
