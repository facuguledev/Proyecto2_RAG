"""Escribe las planillas ciegas para que una persona rehaga las cuatro auditorías del proyecto.

    python tools/make_human_audits.py                 # desde la raíz del repo o del zip

Las auditorías de la entrega las completó un modelo (``decision_log.md``, sección 6). Estas planillas tienen
las mismas filas, en el mismo orden, sin lo que dijo el modelo:

| Planilla | Filas | Qué se completa |
|---|---|---|
| ``judge_human/audit.csv`` (+ ``audit_key.jsonl``, lo oculto) | los 82 juicios de la Etapa 7a | ``human_verdict``: correct / incorrect |
| ``memory/human_audit_records.csv`` | los 76 registros de la Etapa 6 | ``human_verdict``: bien / mal / nunca_se_dijo |
| ``results/05_summaries_human_audit.csv`` | las 25 actualizaciones del resumen (Etapa 5) | ``human_category``: correcta / inventado / relacion / faltante / corrupto |
| ``results/05_rewrites_human_read.csv`` | las 20 reescrituras leídas (Etapa 5) | ``human_category``: resuelve / agrega_ley / mezcla / arrastra / no_resuelve / pierde_fecha / otra |

Nunca pisa una planilla que ya tiene veredictos. Usa solo la biblioteca estándar: corre desde el zip. Las
categorías que puso el modelo en las reescrituras (la tabla de la notebook 05) quedan aparte, en
``results/05_rewrites_model_labels.json``, para compararlas después sin mostrárselas a quien audita.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
from pathlib import Path

JUDGE_COLUMNS = ["audit_id", "kind", "question", "answer_key", "expected_verdict", "can_answer", "answer",
                 "human_verdict", "human_note"]
MEMORY_COLUMNS = ["memory_id", "scenario", "source_session", "kind", "subject", "text", "valid_from", "valid_to",
                  "session_transcript", "human_verdict", "human_note"]
SUMMARY_COLUMNS = ["n", "dialogue_id", "before", "evicted", "after", "human_category", "human_note"]
REWRITE_COLUMNS = ["turn_id", "depends_on", "user", "rewrite", "oracle_rewrite", "human_category", "human_note"]

# La tabla "Lectura de las 20 reescrituras" de 05_conversation (las categorías del auditor-modelo).
MODEL_REWRITE_LABELS = {
    **dict.fromkeys(["dl01.t2", "dl01.t4", "dl01.t6", "dl03.t3", "dl03.t5", "dl04.t3", "dl05.t4", "dl06.t4",
                     "dl06.t5", "dl07.t2", "dl07.t5", "dl08.t2", "dl08.t4", "dl08.t5", "dl02.t5"], "resuelve"),
    "dl05.t2": "agrega_ley", "dl03.t2": "mezcla", "dl02.t3": "arrastra", "dl02.t2": "no_resuelve",
    "dl01.t3": "pierde_fecha",
}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_sheet(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    delimiter = max(",;\t", key=text.split("\n", 1)[0].count)
    return list(csv.DictReader(io.StringIO(text), dialect=csv.excel, delimiter=delimiter))


def has_answers(path: Path, field: str) -> bool:
    return path.is_file() and any((row.get(field) or "").strip() for row in read_sheet(path))


def write_sheet(path: Path, columns: list[str], rows: list[dict], field: str) -> str:
    """Escribe la planilla salvo que ya tenga respuestas. Con BOM: Excel la abre con los acentos bien."""
    if has_answers(path, field):
        return f"se conserva (ya tiene respuestas): {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return f"escrita, {len(rows)} filas: {path}"


def judge_sheet(art: Path) -> list[str]:
    source, target = art / "judge", art / "judge_human"
    if not (source / "audit.csv").is_file():
        return [f"sin {source / 'audit.csv'}: no hay planilla del juez"]
    rows = [{**r, "human_verdict": "", "human_note": ""} for r in read_sheet(source / "audit.csv")]
    out = [write_sheet(target / "audit.csv", JUDGE_COLUMNS, rows, "human_verdict")]
    shutil.copy2(source / "audit_key.jsonl", target / "audit_key.jsonl")  # lo oculto: etapa, brazo, juez, estrato
    return out


def memory_sheet(root: Path, art: Path) -> list[str]:
    """Con la transcripción de la sesión de la que salió cada registro: es contra eso que se audita."""
    source = art / "memory" / "audit_records.csv"
    if not source.is_file():
        return [f"sin {source}: no hay planilla de memoria"]
    transcripts = {}
    for scenario in read_jsonl(root / "eval" / "scenarios.jsonl"):
        for session in scenario["sessions"]:
            transcripts[session["session_id"]] = f"[{session['started_at']}] " + " | ".join(
                f"{'U' if m['role'] == 'user' else 'A'}: {m['content']}" for m in session["messages"])
    rows = [{**r, "session_transcript": transcripts.get(r["source_session"], ""), "human_verdict": "",
             "human_note": ""} for r in read_sheet(source)]
    return [write_sheet(art / "memory" / "human_audit_records.csv", MEMORY_COLUMNS, rows, "human_verdict")]


def summary_sheet(art: Path) -> list[str]:
    source = art / "results" / "05_summaries.jsonl"
    if not source.is_file():
        return [f"sin {source}: no hay resúmenes"]
    updates = [u for u in read_jsonl(source) if u.get("pass") == "answers" and u.get("arm") == "summary_h150"]
    rows = [{"n": n, "dialogue_id": u["dialogue_id"], "before": u["before"] or "(vacío)",
             "evicted": " | ".join(f"{'U' if m['role'] == 'user' else 'A'}: {m['content']}" for m in u["evicted"]),
             "after": u["after"] or "(falló)"} for n, u in enumerate(updates, start=1)]
    return [write_sheet(art / "results" / "05_summaries_human_audit.csv", SUMMARY_COLUMNS, rows, "human_category")]


def rewrite_sample(dialogues: list[dict], n: int = 20, seed: int = 0) -> list[tuple[str, str]]:
    """Las mismas 20 reescrituras que eligió la Etapa 5 (sección 4): con la misma semilla, de a una por
    etiqueta de dependencia y en ronda, entre los turnos no-primeros de los guiones de test."""
    pools: dict[str, list[tuple[str, str]]] = {}
    for d in dialogues:
        if d["split"] != "test":
            continue
        for number, turn in enumerate(d["turns"], start=1):
            if number > 1:
                pools.setdefault(turn["depends_on"], []).append((f"{d['dialogue_id']}.t{number}", turn["depends_on"]))
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    chosen: list[tuple[str, str]] = []
    while len(chosen) < n and any(pools.values()):
        for dependency in sorted(pools):
            if pools[dependency] and len(chosen) < n:
                chosen.append(pools[dependency].pop())
    return chosen


def rewrite_sheet(root: Path, art: Path) -> list[str]:
    source = art / "results" / "05_rewrites.jsonl"
    if not source.is_file():
        return [f"sin {source}: no hay reescrituras"]
    dialogues = read_jsonl(root / "eval" / "dialogues.jsonl")
    turns = {f"{d['dialogue_id']}.t{n}": t for d in dialogues for n, t in enumerate(d["turns"], start=1)}
    rewritten = {r["item_id"]: r["query"] for r in read_jsonl(source)}
    chosen = rewrite_sample(dialogues)
    if {turn_id for turn_id, _ in chosen} != set(MODEL_REWRITE_LABELS):
        return ["la muestra de reescrituras no es la de la Etapa 5: no se escribe la planilla"]
    rows = [{"turn_id": turn_id, "depends_on": dependency, "user": turns[turn_id]["user"],
             "rewrite": rewritten.get(turn_id) or "(falló)", "oracle_rewrite": turns[turn_id]["oracle_rewrite"]}
            for turn_id, dependency in chosen]
    labels = art / "results" / "05_rewrites_model_labels.json"
    labels.write_text(json.dumps(MODEL_REWRITE_LABELS, ensure_ascii=False, indent=1), encoding="utf-8")
    return [write_sheet(art / "results" / "05_rewrites_human_read.csv", REWRITE_COLUMNS, rows, "human_category")]


def write_all(root: Path, art: Path) -> list[str]:
    root, art = Path(root), Path(art)
    return judge_sheet(art) + memory_sheet(root, art) + summary_sheet(art) + rewrite_sheet(root, art)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--profile", default="full")
    args = ap.parse_args()
    root = Path(args.root)
    for line in write_all(root, root / "artifacts" / args.profile):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
