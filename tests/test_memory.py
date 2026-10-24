"""Etapa 6: el almacén, el camino de escritura y el de lectura. Sin servidor ni embedder: el modelo, el
embedder y el contador de tokens son falsos."""
from datetime import date, datetime

import numpy as np
import pytest

from evalset import Message, Probe, Scenario, Session
from memory import (MEMORY_BUDGET, NO_MEMORY_TEXT, Extraction, Fact, MemoryStore, WriteDecision, at_max_length,
                    audit_counts, audit_rows, distinct_resamples, evidence_recall, extraction_messages,
                    first_overflow, hedged, history_tokens, leaked, memory_filters, memory_regions, memory_slot,
                    minmax, recall, recency, record_line, replay, transcript, where_replay_stops_fitting,
                    write_audit, write_error_for, write_session)

FEB, MAY, SEP, OCT = (datetime(2026, 2, 16, 20), datetime(2026, 5, 11, 19), datetime(2026, 9, 3, 21),
                      datetime(2026, 10, 13, 18))


def words(text: str) -> int:
    return len(text.split())


VOCAB = ["edad", "renta", "ciudad", "citas", "hospital", "fianza", "soller", "heredado"]


def encode(texts: list[str]) -> np.ndarray:
    """Un embedder de juguete: una dimensión por palabra del vocabulario, normalizado."""
    rows = []
    for text in texts:
        v = np.array([float(w in text.lower()) for w in VOCAB]) + 1e-3
        rows.append(v / np.linalg.norm(v))
    return np.array(rows)


def fact(subject: str, text: str, kind="semantic", importance=5) -> Fact:
    return Fact(kind=kind, subject=subject, text=text, importance=importance)


def session(session_id: str, at: datetime, *pairs: tuple[str, str]) -> Session:
    messages = [Message(role=role, content=c) for u, a in pairs for role, c in (("user", u), ("assistant", a))]
    return Session(session_id=session_id, started_at=at, messages=messages)


S1 = session("s1", FEB, ("Tengo 35 años y vivo en Pamplona.", "Entendido."),
             ("¿Puedo pedir el bono?", "Sí, con 35 años como máximo."))
S2 = session("s2", MAY, ("Pago 290 euros de renta.", "Entra en el límite."))
S3 = session("s3", SEP, ("Cumplí 36 años.", "Felicidades."), ("¿Y la fianza?", "Una mensualidad."))


# --- El almacén ------------------------------------------------------------------------------------

def test_update_invalidates_instead_of_deleting():
    store = MemoryStore("sc")
    store.apply(WriteDecision(analysis="", decision="ADD"), fact("edad", "Tiene 35 años."), "s1", FEB, set())
    op = store.apply(WriteDecision(analysis="cumplió años", decision="UPDATE", target_id="m1",
                                   text="Tiene 36 años desde agosto de 2026."),
                     fact("edad", "Cumplió 36 años."), "s3", SEP, {"m1"})
    old, new = store.get("m1"), store.get("m2")
    assert op["op"] == "UPDATE" and op["closed"] == ["m1"] and len(store.records) == 2
    assert old.valid_to == SEP and old.superseded_by == "m2"
    assert new.text == "Tiene 36 años desde agosto de 2026." and new.valid_from == SEP


def test_a_probe_before_the_change_sees_the_old_value_as_current():
    store = MemoryStore("sc")
    store.apply(None, fact("edad", "Tiene 35 años."), "s1", FEB, set())
    store.apply(WriteDecision(analysis="", decision="UPDATE", target_id="m1"), fact("edad", "Tiene 36 años."),
                "s3", SEP, {"m1"})
    june = datetime(2026, 6, 15)
    assert [r.memory_id for r in store.current(june)] == ["m1"]
    assert [r.memory_id for r in store.current(OCT)] == ["m2"]
    assert "ya no vigente" not in record_line(store.get("m1"), june)
    assert "ya no vigente desde 2026-09-03" in record_line(store.get("m1"), OCT)


def test_a_target_that_was_not_shown_keeps_the_fact_and_closes_nothing_chosen():
    store = MemoryStore("sc")
    store.apply(None, fact("ciudad", "Vive en Pamplona."), "s1", FEB, set())
    update = store.apply(WriteDecision(analysis="", decision="UPDATE", target_id="m7"), fact("renta", "290"),
                         "s2", MAY, {"m1"})
    assert update["op"] == "ADD" and update["fallback"] == "bad_target" and update["closed"] == []
    invalidate = store.apply(WriteDecision(analysis="", decision="INVALIDATE", target_id="m1"),
                             fact("hospital", "Trabaja de noche."), "s3", SEP, set())  # m1 no se le mostró
    assert invalidate["op"] == "ADD" and invalidate["fallback"] == "bad_target"
    assert store.get("m1").valid_to is None


def test_a_decision_error_keeps_the_fact():
    store = MemoryStore("sc")
    op = store.apply(None, fact("renta", "Paga 290 euros."), "s2", MAY, set())
    assert op["op"] == "ADD" and op["fallback"] == "decision_error" and len(store.records) == 1


def test_noop_writes_nothing_and_invalidate_keeps_the_new_fact():
    store = MemoryStore("sc")
    store.apply(None, fact("deuda", "Debe las cuotas de la comunidad."), "s1", FEB, set())
    store.apply(WriteDecision(analysis="", decision="NOOP"), fact("deuda", "Debe cuotas."), "s2", MAY, {"m1"})
    assert len(store.records) == 1
    op = store.apply(WriteDecision(analysis="", decision="INVALIDATE", target_id="m1"),
                     fact("pago_comunidad", "Pagó todo lo que debía el 1 de septiembre de 2026."), "s3", SEP, {"m1"})
    assert op["new_id"] == "m2" and op["closed"] == ["m1"]
    assert store.get("m1").valid_to == SEP and store.get("m1").superseded_by == "m2"
    assert [r.memory_id for r in store.current(OCT)] == ["m2"]


def test_one_current_value_per_subject_except_episodes():
    store = MemoryStore("sc")
    store.apply(None, fact("renta", "Paga 290 euros."), "s2", MAY, set())
    op = store.apply(WriteDecision(analysis="", decision="ADD"), fact("renta", "Paga 310 euros."), "s3", SEP, set())
    assert op["closed"] == ["m1"] and [r.memory_id for r in store.current(OCT)] == ["m2"]
    store.apply(None, fact("consulta_fianza", "Preguntó por la fianza.", "episodic"), "s2", MAY, set())
    store.apply(None, fact("consulta_fianza", "Volvió a preguntar por la fianza.", "episodic"), "s3", SEP, set())
    assert sum(r.subject == "consulta_fianza" for r in store.current(OCT)) == 2


# --- El camino de escritura ------------------------------------------------------------------------

class FakeLLM:
    """Devuelve respuestas guionadas por schema y anota cada llamada."""

    def __init__(self, extractions: list, decisions: list):
        self.extractions, self.decisions, self.calls = list(extractions), list(decisions), []

    def __call__(self, messages, schema, num_predict):
        self.calls.append((schema.__name__, messages))
        answer = (self.extractions if schema is Extraction else self.decisions).pop(0)
        if answer is None:
            return None, {"error": "PARSE_FAILED: ...", "prompt_tokens": 10, "output_tokens": 0}
        return answer, {"prompt_tokens": 100, "output_tokens": 20}


def test_write_session_adds_without_a_call_while_the_store_is_empty():
    fake = FakeLLM([Extraction(analysis="dos datos nuevos de la usuaria", facts=[
        fact("edad", "Tiene 35 años."), fact("ciudad", "Vive en Pamplona.")])],
        [WriteDecision(analysis="es otro dato", decision="ADD")])
    store = MemoryStore("sc")
    cost = write_session(store, S1, structured=fake, encode=encode)
    assert [name for name, _ in fake.calls] == ["Extraction", "WriteDecision"]  # el segundo ya ve al primero
    assert cost["calls"] == 2 and cost["ops"] == {"ADD": 2} and store.ops[0]["no_call"] is True
    assert all(r.created_at == FEB == r.valid_from for r in store.records)


def test_extraction_sees_the_current_subjects():
    store = MemoryStore("sc")
    store.apply(None, fact("edad", "Tiene 35 años."), "s1", FEB, set())
    content = extraction_messages(S3, store.subjects(SEP))[-1]["content"]
    assert "Etiquetas (subject) que ya existen para este usuario: edad" in content
    assert "Etiquetas" not in extraction_messages(S1, [])[-1]["content"]


def test_write_session_shows_the_most_similar_records_to_the_decision():
    store = MemoryStore("sc")
    store.apply(None, fact("edad", "Tiene 35 años."), "s1", FEB, set())
    store.apply(None, fact("renta", "Paga 290 euros de renta."), "s2", MAY, set())
    fake = FakeLLM([Extraction(analysis="...", facts=[fact("edad", "Cumplió 36 años de edad.")])],
                   [WriteDecision(analysis="...", decision="UPDATE", target_id="m1", text="Tiene 36 años.")])
    write_session(store, S3, structured=fake, encode=encode, k_similar=1)
    shown = fake.calls[1][1][-1]["content"]
    assert "id m1" in shown and "id m2" not in shown
    assert store.get("m1").valid_to == SEP and store.get("m3").text == "Tiene 36 años."


def test_a_failed_extraction_writes_nothing_and_blocks_scoring_the_store():
    store = MemoryStore("sc")
    cost = write_session(store, S1, structured=FakeLLM([None], []), encode=encode)
    assert store.records == [] and cost["extraction_error"].startswith("PARSE_FAILED") and cost["errors"] == 1
    probe = Probe(probe_id="p", question="?", asked_at=OCT, ability="extraction", answer_key=["35"],
                  evidence_sessions=["s1", "s3"])
    assert write_error_for(probe, {"s1": cost["extraction_error"]}).startswith("WRITE_ERROR")
    assert write_error_for(probe, {"s2": "x"}) is None


def test_a_field_at_its_max_length_is_counted_as_cut():
    assert at_max_length(WriteDecision(analysis="x" * 400, decision="ADD")) == ["analysis"]
    assert at_max_length(fact("edad", "Tiene 35 años.")) == []


def test_leaked_counts_only_spans_that_are_in_the_analysis_and_not_in_the_transcript():
    analysis = "la usuaria probablemente trabaja en el hospital de Navarra y vive sola en el piso"
    store = MemoryStore("sc")
    store.apply(None, fact("edad", "Tiene 35 años y vive en Pamplona."), "s1", FEB, set())
    transcripts = {"s1": transcript(S1)}
    assert leaked(store.records, {"s1": [analysis]}, transcripts) == []
    store.records[0].text = "Dato: la usuaria probablemente trabaja en el hospital de Navarra y vive sola"
    assert leaked(store.records, {"s1": [analysis]}, transcripts) == ["m1"]
    # Un tramo que el análisis y el registro copian los dos de la transcripción no es una filtración.
    quoted = "Dijo: Usuario: Tengo 35 años y vivo en Pamplona. Asistente: Entendido."
    store.records[0].text = "Usuario: Tengo 35 años y vivo en Pamplona."
    assert leaked(store.records, {"s1": [quoted]}, {}) == ["m1"]  # sin la transcripción, contaría
    assert leaked(store.records, {"s1": [quoted]}, transcripts) == []


def test_leaked_catches_think_tags():
    store = MemoryStore("sc")
    store.apply(None, fact("edad", "<think>no sé</think> Tiene 35 años."), "s1", FEB, set())
    assert leaked(store.records, {}, {}) == ["m1"]


def test_hedged_is_by_whole_word():
    store = MemoryStore("sc")
    store.apply(None, fact("contrato", "Su nombre aparece en el contrato."), "s1", FEB, set())
    store.apply(None, fact("edad", "Probablemente tiene 35 años."), "s1", FEB, set())
    assert hedged(store.records) == ["m2"]


# --- La auditoría ----------------------------------------------------------------------------------

def test_the_audit_sheet_is_never_overwritten_once_it_has_verdicts(tmp_path):
    store = MemoryStore("sc")
    store.apply(None, fact("edad", "Tiene 35 años."), "s1", FEB, set())
    path = tmp_path / "audit_records.csv"
    assert write_audit(path, audit_rows({"sc": store})) == "written"
    counts, filled, total = audit_counts(path)
    assert counts is None and (filled, total) == (0, 1)
    path.write_text(path.read_text(encoding="utf-8").replace(",,\n", ",bien,\n"), encoding="utf-8")
    assert write_audit(path, audit_rows({"sc": store})).startswith("kept")
    assert audit_counts(path)[0] == {"bien": 1}
    store.records[0].text = "Tiene 36 años."  # otro almacén: la planilla vieja no se pisa
    assert write_audit(path, audit_rows({"sc": store})).startswith("stale")
    assert (tmp_path / "audit_records.new.csv").is_file() and audit_counts(path)[0] == {"bien": 1}


# --- El camino de lectura --------------------------------------------------------------------------

def test_minmax_flattens_only_an_exact_constant():
    assert minmax([3.0, 3.0]) == [0.0, 0.0]
    assert minmax([1.0, 1.0001]) == pytest.approx([0.0, 1.0])  # casi constante: se estira
    assert minmax([]) == []


def test_recency_decays_per_simulated_hour():
    store = MemoryStore("sc")
    store.apply(None, fact("edad", "Tiene 35 años."), "s1", FEB, set())
    assert recency(store.get("m1"), FEB) == 1.0
    assert recency(store.get("m1"), datetime(2026, 2, 17, 20)) == pytest.approx(0.995 ** 24)


def build_store() -> MemoryStore:
    store = MemoryStore("sc")
    store.apply(None, fact("edad", "Tiene 35 años.", importance=8), "s1", FEB, set())
    store.apply(None, fact("citas", "Quiere que le citen el número de la norma.", "procedural", 6), "s1", FEB, set())
    store.apply(None, fact("renta", "Paga 290 euros de renta.", importance=9), "s2", MAY, set())
    store.apply(WriteDecision(analysis="", decision="UPDATE", target_id="m1"),
                fact("edad", "Tiene 36 años.", importance=8), "s3", SEP, {"m1"})
    return store


def test_recall_only_sees_what_was_written_before_the_probe():
    store = build_store()
    chosen, info = recall(store, encode(["renta"])[0], datetime(2026, 6, 1), encode=encode, count=words)
    assert {r.memory_id for r in chosen} == {"m1", "m2", "m3"}
    assert set(info["ranges"]) == {"hours", "decay", "importance", "relevance"}


def test_recency_is_graded_not_a_step():
    # Con 0,995^h y meses de distancia, el min-max daría 1 a la última sesión y ~0 a todas las demás. En
    # escala logarítmica, la sesión del medio queda en el medio.
    store = MemoryStore("sc")
    for i, at in enumerate([FEB, MAY, SEP]):
        store.apply(None, fact(f"dato{i}", "algo"), f"s{i}", at, set())
    _, info = recall(store, encode(["algo"])[0], OCT, encode=encode, count=words, k=3)
    assert info["ranges"]["decay"][0] < 1e-5  # el decaimiento crudo de la más vieja: casi cero
    hours = [(OCT - at).total_seconds() / 3600 for at in (FEB, MAY, SEP)]
    middle = (hours[0] - hours[1]) / (hours[0] - hours[2])
    assert 0.2 < middle < 0.8


def test_recall_respects_k_and_budget_skipping_not_breaking():
    store = build_store()
    store.records[2].text = "x " * 200  # m3 no entra en un presupuesto chico
    chosen, info = recall(store, encode(["renta"])[0], OCT, encode=encode, count=words, k=10, budget=40)
    ids = [r.memory_id for r in chosen]
    assert "m3" not in ids and len(ids) >= 2 and info["tokens"] <= 40
    chosen, _ = recall(store, encode(["renta"])[0], OCT, encode=encode, count=words, k=1)
    assert len(chosen) == 1


def test_replay_keeps_the_newest_messages_in_order():
    text, sessions, used, complete = replay([S1, S2, S3], OCT, budget=10_000, count=words)
    assert sessions == ["s1", "s2", "s3"] and complete and text.index("35 años") < text.index("Cumplí 36")
    text, sessions, used, complete = replay([S1, S2, S3], OCT, budget=25, count=words)
    assert sessions[-1] == "s3" and "s1" not in sessions and used <= 25 and not complete
    assert not text.split(":\n", 1)[1].startswith("Asistente")  # nunca una respuesta sin su pregunta


def test_replay_and_store_share_the_time_boundary():
    _, sessions, _, _ = replay([S1, S2, S3], datetime(2026, 6, 15), budget=10_000, count=words)
    assert sessions == ["s1", "s2"]
    _, sessions, _, _ = replay([S1, S2, S3], MAY, budget=10_000, count=words)  # <=, como store.visible
    store = MemoryStore("sc")
    store.apply(None, fact("renta", "Paga 290."), "s2", MAY, set())
    assert sessions == ["s1", "s2"] and store.visible(MAY)
    text, sessions, used, complete = replay([S2], FEB, budget=10_000, count=words)
    assert text == NO_MEMORY_TEXT and sessions == [] and used == 0 and complete


PROBE = Probe(probe_id="p1", question="¿Cuántos años tengo?", asked_at=OCT, ability="knowledge_update",
              answer_key=["36"], evidence_sessions=["s1", "s3"])


def test_memory_slot_per_arm():
    scenario = Scenario(scenario_id="sc", persona="ficticia", sessions=[S1, S2, S3], probes=[
        PROBE, Probe(probe_id="p2", question="¿Mi hospital?", asked_at=OCT, ability="abstention", answerable=False),
        Probe(probe_id="p3", question="¿Tengo 35?", asked_at=datetime(2026, 6, 1), ability="knowledge_update",
              answer_key=["35"], expected_verdict="yes", evidence_sessions=["s1"]),
        Probe(probe_id="p4", question="¿Tengo 35?", asked_at=OCT, ability="knowledge_update", answer_key=["36"],
              expected_verdict="no", evidence_sessions=["s1", "s3"])])
    store = build_store()
    assert memory_slot("no_memory", scenario, None, PROBE, encode=encode, count=words)["text"] == NO_MEMORY_TEXT
    replayed = memory_slot("replay", scenario, None, PROBE, encode=encode, count=words)
    assert replayed["sessions"] == ["s1", "s2", "s3"] and replayed["complete"]
    small = memory_slot("replay_600", scenario, None, PROBE, encode=encode, count=words)
    assert small["tokens"] <= MEMORY_BUDGET
    stored = memory_slot("store", scenario, store, PROBE, encode=encode, count=words, query=encode(["edad"])[0])
    assert "Tiene 36 años." in stored["text"] and stored["memory_ids"]
    filtered = memory_slot("store_filter", scenario, store, PROBE, encode=encode, count=words,
                           query=encode(["edad"])[0])
    assert filtered["text"] == stored["text"]  # el mismo espacio: lo que cambia son los pasajes
    with pytest.raises(RuntimeError):
        memory_slot("store", scenario, None, PROBE, encode=encode, count=words)


# --- La memoria como filtro ------------------------------------------------------------------------

def filter_store() -> MemoryStore:
    store = MemoryStore("sc")
    store.apply(None, fact("ciudad", "Vive en Palma y tiene pisos en Mallorca."), "s1", FEB, set())
    store.apply(None, fact("contrato_soller", "Firmó el contrato del piso de Sóller el 1 de marzo de 2024."),
                "s2", MAY, set())
    store.apply(None, fact("contrato_heredado", "Firmó el contrato del piso heredado en septiembre de 2026."),
                "s3", SEP, set())
    store.apply(None, fact("consulta", "Se le explicó la ley de Andalucía.", "episodic"), "s3", SEP, set())
    return store


def probe(question: str, at: datetime = OCT) -> Probe:
    return Probe(probe_id="p", question=question, asked_at=at, ability="extraction", answer_key=["x"],
                 evidence_sessions=["s1"])


def test_memory_regions_reads_cities_and_community_names():
    store = filter_store()
    assert memory_regions(store.records[:1]) == ("Illes Balears",)
    assert memory_regions([store.records[3]]) == ("Andalucía",)  # por eso los episodios no entran al filtro
    no_city = MemoryStore("x")
    no_city.apply(None, fact("dueña", "La dueña vive en Tudela."), "s1", FEB, set())
    assert memory_regions(no_city.records) == ()


def test_memory_filters_take_the_contract_closest_to_the_question():
    store = filter_store()
    filters, info = memory_filters(store, probe("¿Cuándo vence el piso de Sóller?"), encode(["soller"])[0], encode)
    assert filters.period == (date(2024, 3, 1), date(2024, 4, 1)) and info["period_source"] == "memory:m2"
    assert filters.regions == ("Illes Balears",) and info["regions_source"] == "memory"  # no Andalucía
    filters, info = memory_filters(store, probe("¿Y el heredado?"), encode(["heredado"])[0], encode)
    assert filters.period == (date(2026, 9, 1), date(2026, 10, 1)) and info["period_source"] == "memory:m3"


def test_memory_filters_let_the_question_win():
    store = filter_store()
    filters, info = memory_filters(store, probe("¿Y si hubiera firmado en mayo de 2005 en Madrid?"),
                                   encode(["soller"])[0], encode)
    assert filters.period == (date(2005, 5, 1), date(2005, 6, 1)) and info["period_source"] == "question"
    assert filters.regions == ("Comunidad de Madrid",) and info["regions_source"] == "question"


def test_memory_filters_only_see_what_was_written_and_default_to_the_probe_day():
    store = filter_store()
    filters, info = memory_filters(store, probe("¿Algo?", datetime(2026, 3, 1, 10)), encode(["x"])[0], encode)
    assert info["period_source"] == "asked_at" and filters.period == (date(2026, 3, 1), date(2026, 3, 2))
    assert filters.regions == ("Illes Balears",)  # solo s1 estaba escrita


def test_evidence_recall_counts_sessions():
    assert evidence_recall({"s3", "s2"}, PROBE) == (1, 2)


def test_where_replay_stops_fitting_accumulates_across_scenarios():
    a = Scenario(scenario_id="a", persona="x", sessions=[S1, S2], probes=[
        Probe(probe_id="a1", question="?", asked_at=OCT, ability="abstention", answerable=False),
        Probe(probe_id="a2", question="?", asked_at=OCT, ability="knowledge_update", answer_key=["1"],
              expected_verdict="yes", evidence_sessions=["s1"]),
        Probe(probe_id="a3", question="?", asked_at=OCT, ability="knowledge_update", answer_key=["1"],
              expected_verdict="no", evidence_sessions=["s1"])])
    sizes = history_tokens(a, words)
    rows = where_replay_stops_fitting([a], words, {"budget": sizes[0], "big": 10_000})
    assert [r["accumulated"] for r in rows] == [sizes[0], sizes[0] + sizes[1]]
    assert [r["budget"] for r in rows] == [True, False]
    assert first_overflow(rows, "budget").startswith("deja de entrar en s2")
    assert first_overflow(rows, "big").startswith("entra entero")


def test_distinct_resamples():
    assert distinct_resamples(3) == 10 and distinct_resamples(1) == 1
