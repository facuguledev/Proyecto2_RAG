"""Prueba de humo de ``08_followups.py`` sin GPU, sin modelos y sin el corpus.

La notebook necesita el índice real (embeddings de bge-m3), el generador (qwen3:8b en Ollama) y el corpus,
y nada de eso está en el zip ni en CI. Esta prueba la corre de punta a punta contra **los artefactos
guardados de la corrida** y con dobles de todo lo demás:

- un corpus sintético con una sección por cada id que aparece en el eval set y en los resultados guardados
  (con la clave en el texto de las secciones gold);
- un índice falso (una unidad por id guardado, un ranking determinista por hash de la consulta);
- un generador falso (respuestas estructuradas válidas, que a veces aciertan y a veces se abstienen).

Con datos falsos, las verificaciones contra lo guardado fallan por construcción: se cuentan en lugar de
cortar (``followups.require``). Lo que prueba es la orquestación: que cada celda corre, que los archivos
nuevos se escriben con la forma esperada, que una segunda corrida retoma sin volver a llamar al modelo, y
que ``tools/recompute_tables.py`` lee lo nuevo. Los números que salen no significan nada.
"""
import hashlib
import json
import runpy
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "artifacts" / "full"
NEEDED = ["results/03_retrieval.jsonl", "results/04_grounded_answers.jsonl", "results/05_history.jsonl",
          "results/05_queries.jsonl", "results/05_summary_probe.jsonl", "results/05_summaries.jsonl",
          "results/05_rewrites.jsonl", "results/07_judge_granite3_3_8b.jsonl", "index/best_config.json",
          "judge/audit.csv", "judge/audit_key.jsonl", "memory/audit_records.csv"]
pytestmark = pytest.mark.skipif(not all((FULL / p).is_file() for p in NEEDED),
                                reason="sin los artefactos de la corrida (artifacts/full)")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def h(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def referenced_units(art: Path) -> dict[str, int]:
    """unit_id -> n_gen guardado (o 120), de cada unidad que aparece en una entrega guardada."""
    units: dict[str, int] = {}
    for name in ("03_retrieval", "04_grounded_answers", "05_queries", "06_memory"):
        path = art / "results" / f"{name}.jsonl"
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            for entry in row.get("delivered") or []:
                unit_id, n = (entry if isinstance(entry, list) else (entry, 120))
                units.setdefault(unit_id, n)
    for row in read_jsonl(art / "results" / "05_history.jsonl"):
        for unit_id in row.get("delivered") or []:
            units.setdefault(unit_id, 120)
    return units


def build_world(art: Path):
    """El corpus y las unidades sintéticas."""
    sys.path.insert(0, str(ROOT / "src"))
    import evalset
    from chunking import Unit
    from doc_schema import DocMetadata, Document, Section

    ev = evalset.load_eval(ROOT / "eval", require_frozen=False)
    keys: dict[str, set[str]] = {}
    for _, answer in evalset.corpus_questions(ev) + evalset.probes(ev):
        for sid in answer.gold_sections:
            keys.setdefault(sid, set()).update(answer.answer_key)
    saved = referenced_units(art)
    section_ids = set(keys) | {u.split("~", 1)[0] for u in saved}
    for row in read_jsonl(art / "results" / "05_queries.jsonl"):
        section_ids |= set(row.get("sections") or [])
    by_doc: dict[str, list[str]] = {}
    for sid in sorted(section_ids):
        by_doc.setdefault(sid.split("#", 1)[0], []).append(sid)
    docs, texts = [], {}
    for doc_id, sids in by_doc.items():
        text, sections = "", []
        for sid in sids:
            body = f"Texto de {sid}. " + " ".join(sorted(keys.get(sid, ())))
            start = len(text) + (2 if text else 0)
            text = f"{text}\n\n{body}" if text else body
            block = sid.split("#", 1)[1].split("@", 1)[0]
            day = sid.rsplit("@", 1)[1][:8]
            sections.append(Section(section_id=sid, title=f"Artículo {block}.", level=1, start=start,
                                    end=start + len(body), block_id=block, kind="article",
                                    valid_from=date(int(day[:4]), int(day[4:6]), int(day[6:8]))))
            texts[sid] = body
        docs.append(Document(doc_id=doc_id, title=f"Norma {doc_id}", text=text, sections=sections,
                             metadata=DocMetadata(status="current", region="Estatal")))

    def unit(unit_id: str, n: int) -> Unit:
        sid = unit_id.split("~", 1)[0]
        parent = unit_id if "~p" in unit_id else f"{sid}~p1"
        return Unit(unit_id=unit_id, section_id=sid, doc_id=sid.split("#", 1)[0], parent_id=parent, start=0,
                    end=len(texts[sid]), text=f"{texts[sid]} [{unit_id}]", prefix="",
                    header=f"[Norma · Artículo · vigente desde {sid.rsplit('@', 1)[1][:4]}]", n_gen=n)

    units = {uid: unit(uid, n) for uid, n in saved.items()}
    for sid in section_ids:  # cada sección tiene al menos un hijo y un padre
        units.setdefault(f"{sid}~c1", unit(f"{sid}~c1", 120))
        units.setdefault(f"{sid}~p1", unit(f"{sid}~p1", 300))
    return docs, units


@dataclass
class FakeTokenizer:
    def __call__(self, text, pair=None, add_special_tokens=True, return_offsets_mapping=False):
        spans, pos = [], 0
        for word in text.split():
            start = text.index(word, pos)
            spans.append((start, start + len(word)))
            pos = start + len(word)
        ids = list(range(len(spans) + (2 if add_special_tokens else 0)))
        if pair is not None:
            ids += list(range(len(pair.split()) + (2 if add_special_tokens else 0)))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


class FakeEmbedder:
    tokenizer = FakeTokenizer()
    max_seq_length = 512


def install_fakes(monkeypatch, art: Path, calls: list):
    import artifact_io
    import doc_schema
    import environment
    import evalset
    import grounded
    import index_loader
    import llm
    import retrieval
    from retrieval import Index, RetrievalConfig

    docs, units = build_world(art)
    frozen = json.loads((ROOT / "eval" / "FROZEN.json").read_text(encoding="utf-8"))
    for module in (doc_schema, evalset):
        monkeypatch.setattr(module, "load_documents", lambda path: docs)
        monkeypatch.setattr(module, "documents_hash", lambda d: frozen["corpus_sha"])
    monkeypatch.setattr(artifact_io, "check_manifest", lambda *a, **k: None)
    monkeypatch.setattr(environment, "artifacts_dir", lambda smoke: art)
    monkeypatch.setattr(environment, "summary", lambda: None)
    monkeypatch.setattr(environment, "gpu_report", lambda label="": None)

    best = json.loads((art / "index" / "best_config.json").read_text(encoding="utf-8"))
    children = sorted((u for u in units.values() if "~c" in u.unit_id), key=lambda u: u.unit_id)
    parents = {u.unit_id: u for u in units.values() if "~p" in u.unit_id}

    all_units = {u.unit_id: u for u in units.values()}
    index = Index(children=children, parents=parents, embedder=FakeEmbedder(), reranker=None)

    class FakeLoaded:
        def __init__(self):
            self.index, self.key, self.best = index, best["index"], best
            self.config = RetrievalConfig(**best["config"])
            self.encoder_spec = {"dtype": "float32"}
            self.units = all_units

        def release_gpu(self):
            pass

    def ranked_units(loaded, query, cfg=None, filters=None):
        cfg = cfg or loaded.config
        ranked = sorted(children, key=lambda u: h(query + u.unit_id))[:100]
        if cfg.date_filter:
            ranked = [u for u in ranked if h(u.unit_id) % 3]  # el filtro saca algunos
        delivered = retrieval.deliver(ranked, cfg.budget_tokens)
        result = retrieval.Retrieval([u.unit_id for u in ranked], delivered, {"dense": [u.unit_id for u in ranked]},
                                     [], {}, {"dense": 0.5}, {"total": 1.0})
        return ranked, result

    monkeypatch.setattr(index_loader, "load", lambda *a, **k: FakeLoaded())
    monkeypatch.setattr(index_loader, "ranked_units", ranked_units)

    def count(text: str) -> int:
        return max(1, len(text) // 4)

    monkeypatch.setattr(grounded, "token_counter", lambda model=None: count)
    monkeypatch.setattr(llm, "setup", lambda model, num_ctx: {"server": "fake", "digest": "fake", "key_order": None,
                                                              "thinking": None, "calibration": None})
    monkeypatch.setattr(llm, "use_cache", lambda path: None)
    monkeypatch.setattr(llm, "unload", lambda model: None)
    monkeypatch.setattr(llm, "placement", lambda model: "fake")
    monkeypatch.setattr(llm, "count_prompt_tokens", lambda messages, model, **k: sum(count(m["content"]) for m in messages))

    def structured(messages, schema, **kw):
        calls.append(schema.__name__)
        text = messages[-1]["content"]
        seed = h(text)
        if schema.__name__ in ("Summary", "SummaryV2"):
            previous = text.split("<resumen_anterior>\n", 1)[1].split("\n</resumen_anterior>", 1)[0]
            out = schema(summary=previous if (seed % 4 == 0 and previous != "(vacío)") else f"El usuario dijo {seed % 97}.")
        else:
            can = seed % 5 != 0
            out = schema(analysis="El pasaje responde.", can_answer=can, citations=[],
                         answer=("Sí, una mensualidad y cinco años." if seed % 2 else "No, tres años.") if can
                         else grounded.ABSTENTION)
        r = llm.ChatResult(text=out.model_dump_json(), done_reason="stop", prompt_tokens=count(text) + 4,
                           expected_prompt_tokens=count(text), output_tokens=50, ms=10.0, model="qwen3:8b")
        return out, r

    monkeypatch.setattr(llm, "structured", structured)
    return docs, units


def copy_artifacts(tmp_path: Path) -> Path:
    art = tmp_path / "artifacts" / "full"
    for rel in NEEDED + ["results/06_memory.jsonl", "results/07_judge_qwen3_8b.jsonl"]:
        source = FULL / rel
        if source.is_file():
            (art / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, art / rel)
    for spec in (FULL / "results").glob("*.config-*.json"):
        shutil.copy2(spec, art / "results" / spec.name)
    return art


def test_stage_8_runs_end_to_end_with_fakes(tmp_path, monkeypatch, capsys):
    import followups

    art = copy_artifacts(tmp_path)
    calls, checks = [], []
    install_fakes(monkeypatch, art, calls)
    monkeypatch.setattr(followups, "require", lambda condition, message: checks.append(bool(condition)))
    monkeypatch.setenv("SMOKE_TEST", "0")
    monkeypatch.chdir(ROOT)

    runpy.run_path(str(ROOT / "08_followups.py"), run_name="__main__")
    out = capsys.readouterr().out
    first_calls = len(calls)

    results = art / "results"
    metrics = read_jsonl(results / "03_retrieval_metrics.jsonl")
    assert {r["arm"] for r in metrics} == {"bm25", "dense", "+prefix", "+rrf", "+rerank", "+parent", "+date",
                                          "+region", "dense·A/A"}
    assert len(metrics) == 9 * 34  # 9 filas × 34 preguntas con respuesta
    answers = read_jsonl(results / "08_grounded_answers.jsonl")
    assert sorted({r["arm"] for r in answers}) == ["oracle_padded", "retrieval_date"] and len(answers) == 80
    position = read_jsonl(results / "08_position.jsonl")
    assert len(position) == 3 * 26
    assert {r["gold_index"] for r in position if r["arm"] == "pos_first"} == {0}
    assert all(r["gold_index"] == len(r["delivered"]) - 1 for r in position if r["arm"] == "pos_last")
    history = read_jsonl(results / "08_history.jsonl")
    assert len(history) == 49 and {r["arm"] for r in history} == {"summary_v2_h150"}
    assert {r["split"] for r in history} == {"dev", "test"}
    assert read_jsonl(results / "08_summary_probe.jsonl")
    assert any(u["pass"] == "answers" for u in read_jsonl(results / "08_summaries.jsonl"))
    assert (art / "judge_human" / "audit.csv").is_file() and (art / "memory" / "human_audit_records.csv").is_file()
    report = json.loads((art / "run_report.json").read_text(encoding="utf-8"))
    assert report["08_followups"]["status"] == "ok"
    assert len(checks) > 100  # todas las verificaciones contra lo guardado se evaluaron
    assert "Hasta que una persona complete" in out

    # Una segunda corrida retoma: no vuelve a generar respuestas (los resúmenes se rehacen con la caché, que acá
    # no existe; por eso solo se cuentan las respuestas).
    calls.clear()
    runpy.run_path(str(ROOT / "08_followups.py"), run_name="__main__")
    assert "GroundedAnswer" not in calls, f"la segunda corrida volvió a generar respuestas: {calls[:5]}"
    assert first_calls > 0

    # El recálculo desde el zip lee los archivos nuevos.
    sys.path.insert(0, str(ROOT / "tools"))
    import recompute_tables

    monkeypatch.setattr("sys.argv", ["recompute_tables.py", "--root", str(tmp_path)])
    shutil.copytree(ROOT / "eval", tmp_path / "eval")
    assert recompute_tables.main() == 0
    text = capsys.readouterr().out
    assert "clave desde gold" in text and "08_history.jsonl" in text and "retrieval_date" in text
