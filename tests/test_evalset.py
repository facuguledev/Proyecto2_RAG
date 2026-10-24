"""Eval set: contratos, validación contra el corpus, tamaños, congelado y preguntas derivadas."""
import json
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from evalset import (CATEGORIES, Dialogue, EvalItem, EvalSet, Message, Probe, Scenario, Session, Turn,
                     check_sizes, contains_key, derive_identifier_items, first_rank, freeze_eval,
                     identifier_candidates, load_eval, validate_eval)
from artifact_io import write_jsonl
from doc_schema import DocMetadata, Document, Section
from corpus import SNAPSHOT

# --- Un corpus de juguete ------------------------------------------------------------------------
# LAW: el artículo 1 tiene dos versiones (tres años hasta 2010, cinco después) y el artículo 2 tiene un
# gemelo en COPY, otra norma, que dice lo mismo con otro título.

LAW, COPY = "BOE-A-2000-1", "BOE-A-2000-2"
OLD_TERM, TERM = f"{LAW}#a1@20000101", f"{LAW}#a1@20100101"
DEPOSIT, COPY_DEPOSIT = f"{LAW}#a2@20000101", f"{COPY}#a1@20000101"


def make_doc(doc_id: str, sections: list[tuple], **metadata) -> Document:
    """``sections``: (section_id, valid_from, valid_to, texto, gemelos), en orden de aparición."""
    built, start = [], 0
    for section_id, valid_from, valid_to, text, twins in sections:
        block_id = section_id.split("#")[1].split("@")[0]
        built.append(Section(section_id=section_id, title=text.split("\n")[0], level=1, start=start,
                             end=start + len(text), block_id=block_id, kind="article",
                             valid_from=valid_from, valid_to=valid_to, twins=twins))
        start += len(text) + 2  # el "\n\n" entre secciones
    return Document(doc_id=doc_id, title=f"Norma {doc_id}", text="\n\n".join(s[3] for s in sections),
                    sections=built, metadata=DocMetadata(**metadata))


def make_corpus() -> list[Document]:
    law = make_doc(LAW, [
        (OLD_TERM, date(2000, 1, 1), date(2010, 1, 1), "Artículo 1. Plazo mínimo.\nEs de tres años.", []),
        (TERM, date(2010, 1, 1), None, "Artículo 1. Plazo mínimo.\nEs de cinco años.", []),
        (DEPOSIT, date(2000, 1, 1), None, "Artículo 2. Fianza.\nLa fianza es de dos mensualidades.",
         [COPY_DEPOSIT]),
    ], rango="Ley", numero_oficial="1/2000", region="Estatal")
    copy = make_doc(COPY, [
        (COPY_DEPOSIT, date(2000, 1, 1), None, "Artículo 1. Garantía del alquiler.\n"
         "La fianza es de dos mensualidades.", [DEPOSIT]),
    ], rango="Ley", numero_oficial="2/2000", region="Illes Balears")
    return [law, copy]


# --- Ítems, guiones y escenarios válidos; cada test cambia lo que necesita ------------------------

def item(**changes) -> EvalItem:
    fields = {"item_id": "q1", "question": "¿Cuánto me pueden pedir de fianza?", "category": "topical",
              "topic": "fianza", "gold_sections": [DEPOSIT, COPY_DEPOSIT], "answer_key": ["dos mensualidades"]}
    return EvalItem(**{**fields, **changes})


def out_of_corpus(**changes) -> EvalItem:
    return item(**{"category": "out_of_corpus", "answerable": False, "gold_sections": [],
                   "answer_key": [], **changes})


def turn(depends_on: str = "coreference", **changes) -> Turn:
    fields = {"user": "¿Y cuánto es la fianza?", "depends_on": depends_on,
              "oracle_rewrite": "¿Cuánto es la fianza de un alquiler de vivienda?",
              "assistant": "Dos mensualidades de renta.", "gold_sections": [DEPOSIT, COPY_DEPOSIT],
              "answer_key": ["dos mensualidades"]}
    return Turn(**{**fields, **changes})


def dialogue(dialogue_id: str = "g1", n_turns: int = 3, **changes) -> Dialogue:
    turns = [turn("standalone")] + [turn() for _ in range(n_turns - 1)]
    return Dialogue(**{"dialogue_id": dialogue_id, "turns": turns, **changes})


def session(session_id: str, day: int) -> Session:
    return Session(session_id=session_id, started_at=datetime(2026, 3, day, 10, 0), messages=[
        Message(role="user", content="Vivo en Palma y alquilo un piso."),
        Message(role="assistant", content="Entendido.")])


def probe(probe_id: str, ability: str = "extraction", **changes) -> Probe:
    fields = {"probe_id": probe_id, "question": "¿Dónde vivo?", "asked_at": datetime(2026, 3, 30),
              "ability": ability, "evidence_sessions": ["s1.a"], "answer_key": ["Palma"]}
    if ability == "abstention":
        fields.update(answerable=False, evidence_sessions=[], answer_key=[])
    return Probe(**{**fields, **changes})


def update_pair(scenario_id: str) -> list[Probe]:
    """El hecho que cambia: la misma pregunta antes y después de la sesión b, con veredictos opuestos."""
    before = probe(f"{scenario_id}.pu1", "knowledge_update", question="¿Vivo en Palma?", expected_verdict="yes",
                   evidence_sessions=[f"{scenario_id}.a"], asked_at=datetime(2026, 3, 10))
    after = probe(f"{scenario_id}.pu2", "knowledge_update", question="¿Vivo en Palma?", expected_verdict="no",
                  evidence_sessions=[f"{scenario_id}.a", f"{scenario_id}.b"])
    return [before, after]


def scenario(scenario_id: str = "s1", n_probes: int = 5, **changes) -> Scenario:
    probes = update_pair(scenario_id) + [probe(f"{scenario_id}.pn", "abstention")]
    probes += [probe(f"{scenario_id}.p{n}", evidence_sessions=[f"{scenario_id}.a"]) for n in range(n_probes - 3)]
    fields = {"scenario_id": scenario_id, "persona": "Inquilina ficticia en Palma",
              "sessions": [session(f"{scenario_id}.a", 1), session(f"{scenario_id}.b", 15)], "probes": probes}
    return Scenario(**{**fields, **changes})


def eval_set(items=(), dialogues=(), scenarios=()) -> EvalSet:
    return EvalSet(list(items), list(dialogues), list(scenarios), file_hashes={})


def failures_by_check(report) -> dict[str, list[str]]:
    return {check.name: check.failures for check in report.checks}


def failed_checks(report) -> list[str]:
    return [name for name, failures in failures_by_check(report).items() if failures]


# --- Modelos -------------------------------------------------------------------------------------

def test_an_unanswerable_question_carries_no_gold_key_or_verdict():
    with pytest.raises(ValidationError, match="sin respuesta"):
        out_of_corpus(gold_sections=[DEPOSIT])
    with pytest.raises(ValidationError, match="sin respuesta"):
        out_of_corpus(expected_verdict="no")


def test_an_answerable_question_needs_a_key_and_gold():
    with pytest.raises(ValidationError, match="al menos una clave"):
        item(answer_key=[])
    with pytest.raises(ValidationError, match="lleva gold"):
        item(gold_sections=[])


@pytest.mark.parametrize("key", ["", "  ", "-", "%"])
def test_a_key_without_letters_or_digits_is_an_error(key):
    # Sin esto, la clave "" está "contenida" en cualquier texto que termine en punto, y "-" en casi todos.
    with pytest.raises(ValidationError, match="sin letras ni números"):
        item(answer_key=["dos mensualidades", key])
    with pytest.raises(ValueError, match="sin letras ni números"):
        contains_key("La fianza es de dos mensualidades.", ["dos mensualidades", key])


def test_out_of_corpus_goes_with_unanswerable_in_both_directions():
    with pytest.raises(ValidationError, match="out_of_corpus lleva answerable=false"):
        item(category="out_of_corpus")
    with pytest.raises(ValidationError, match="answerable=false es solo para out_of_corpus"):
        out_of_corpus(category="topical")


def test_gold_docs_come_from_the_section_ids():
    assert item().gold_docs == [LAW, COPY]


def test_a_misspelled_field_is_an_error():
    with pytest.raises(ValidationError):
        item(answer_keys=["dos mensualidades"])


def test_only_the_first_turn_is_standalone():
    with pytest.raises(ValidationError, match="el primer turno"):
        Dialogue(dialogue_id="g1", turns=[turn("coreference")])
    with pytest.raises(ValidationError, match="solo el primer turno"):
        Dialogue(dialogue_id="g1", turns=[turn("standalone"), turn("standalone")])


def test_an_unanswerable_turn_is_labeled_unanswerable():
    with pytest.raises(ValidationError, match="answerable=false es solo para unanswerable"):
        turn("ellipsis", answerable=False, gold_sections=[], answer_key=[])
    assert not turn("unanswerable", answerable=False, gold_sections=[], answer_key=[]).answerable


def test_session_messages_go_in_user_assistant_pairs():
    with pytest.raises(ValidationError, match="de a pares"):
        Session(session_id="s", started_at=datetime(2026, 1, 1), messages=[Message(role="user", content="hola")])
    with pytest.raises(ValidationError, match="de a pares"):
        Session(session_id="s", started_at=datetime(2026, 1, 1), messages=[
            Message(role="assistant", content="hola"), Message(role="user", content="hola")])


def test_an_answerable_probe_names_its_evidence_and_abstention_is_unanswerable():
    with pytest.raises(ValidationError, match="en qué sesión"):
        probe("p", evidence_sessions=[])
    with pytest.raises(ValidationError, match="abstention lleva answerable=false"):
        probe("p", ability="abstention", answerable=True, answer_key=["x"], evidence_sessions=["s1.a"])
    with pytest.raises(ValidationError, match="nunca se dijo no tiene sesión"):
        probe("p", ability="abstention", evidence_sessions=["s1.a"])


def test_probe_evidence_must_be_an_earlier_session_of_the_scenario():
    late = probe("s1.p9", evidence_sessions=["s1.b"], asked_at=datetime(2026, 3, 10))  # s1.b empieza el 15
    with pytest.raises(ValidationError, match="no es una sesión anterior"):
        scenario(probes=scenario().probes + [late])
    unknown = probe("s1.p9", evidence_sessions=["s1.z"])
    with pytest.raises(ValidationError, match="no es una sesión anterior"):
        scenario(probes=scenario().probes + [unknown])


def test_sessions_go_in_time_order():
    with pytest.raises(ValidationError, match="después de la anterior"):
        scenario(sessions=[session("s1.a", 15), session("s1.b", 1)])


def test_a_scenario_has_a_fact_never_said():
    without_abstention = [p for p in scenario().probes if p.ability != "abstention"]
    with pytest.raises(ValidationError, match="nunca se dijo"):
        scenario(probes=without_abstention)


def test_the_fact_that_changes_is_a_pair_with_opposite_verdicts():
    before, _ = update_pair("s1")
    with pytest.raises(ValidationError, match="un hecho que cambia"):
        scenario(probes=[before, probe("s1.pn", "abstention")])  # solo el "antes": falta el veredicto opuesto
    with pytest.raises(ValidationError, match="un hecho que cambia"):
        scenario(probes=[probe("s1.p1"), probe("s1.pn", "abstention")])  # ninguna knowledge_update


def test_a_probe_without_as_of_asks_about_the_law_on_the_day_it_is_asked():
    # Una reforma entre la sonda (marzo de 2026) y la foto (octubre): la sonda pregunta por la versión de marzo.
    before, after = f"{LAW}#a1@20000101", f"{LAW}#a1@20260601"
    law = make_doc(LAW, [(before, date(2000, 1, 1), date(2026, 6, 1), "Artículo 1. Plazo.\nEs de cinco años.", []),
                         (after, date(2026, 6, 1), None, "Artículo 1. Plazo.\nEs de siete años.", [])])
    in_march = probe("s1.p9", gold_sections=[before], answer_key=["cinco años"])  # asked_at: 2026-03-30
    assert in_march.reference_day() == date(2026, 3, 30)
    assert validate_eval(eval_set(scenarios=[scenario(probes=scenario().probes + [in_march])]), [law]).ok
    too_new = probe("s1.p9", gold_sections=[after], answer_key=["siete años"])
    report = validate_eval(eval_set(scenarios=[scenario(probes=scenario().probes + [too_new])]), [law])
    assert failures_by_check(report)["gold_in_force"] == [f"s1.p9: {after} no rige el 2026-03-30"]


# --- Claves --------------------------------------------------------------------------------------

def test_contains_key_matches_whole_numbers_only():
    assert contains_key("más de 10 viviendas", ["10"])
    assert not contains_key("más de 100 viviendas", ["10"])
    assert not contains_key("un 10,5 por ciento", ["10"])


def test_contains_key_ignores_case_and_accents():
    assert contains_key("La fianza: DOS MENSUALIDADES.", ["dos mensualidades"])
    assert contains_key("Mas de diez años", ["más de diez años"])


def test_contains_key_normalizes_thousands_and_decimal_commas():
    assert contains_key("una multa de 1.500 euros", ["1500 euros"])
    assert contains_key("el 2,5 por ciento", ["2.5 por ciento"])


def test_contains_key_ignores_zero_cents_but_not_other_decimals():
    assert contains_key("La fianza es de 1.300,00 euros.", ["1.300"])
    assert not contains_key("Un interés del 2,05 por ciento.", ["2"])


def test_contains_key_is_true_if_any_key_matches():
    assert contains_key("cinco años", ["5 años", "cinco años"])
    assert not contains_key("cinco años", [])


# --- validate_eval -------------------------------------------------------------------------------

def test_a_correct_set_passes_every_check():
    report = validate_eval(eval_set([item(), out_of_corpus(item_id="q2")], [dialogue()], [scenario()]),
                           make_corpus())
    assert report.ok
    assert [check.name for check in report.checks] == [
        "unique_ids", "gold_exists", "gold_in_force", "key_in_gold", "twins_in_gold"]


def test_a_missing_gold_id_fails():  # la guarda de §7.4
    report = validate_eval(eval_set([item(gold_sections=[DEPOSIT, COPY_DEPOSIT, f"{LAW}#a99@20000101"])]),
                           make_corpus())
    assert failures_by_check(report)["gold_exists"] == [f"q1: {LAW}#a99@20000101 no está en el corpus"]


def test_an_undated_question_points_to_the_version_in_force_at_the_snapshot():
    current = item(gold_sections=[TERM], answer_key=["cinco años"])
    old = item(item_id="q2", gold_sections=[OLD_TERM], answer_key=["tres años"])
    report = validate_eval(eval_set([current, old]), make_corpus())
    assert failed_checks(report) == ["gold_in_force"]
    assert failures_by_check(report)["gold_in_force"] == [f"q2: {OLD_TERM} no rige el {SNAPSHOT}"]


def test_a_dated_question_points_to_the_version_in_force_that_day():
    in_2005 = item(category="metadata", gold_sections=[OLD_TERM], answer_key=["tres años"], as_of=date(2005, 3, 1))
    assert validate_eval(eval_set([in_2005]), make_corpus()).ok
    wrong = item(category="metadata", gold_sections=[TERM], answer_key=["cinco años"], as_of=date(2005, 3, 1))
    assert failed_checks(validate_eval(eval_set([wrong]), make_corpus())) == ["gold_in_force"]


def test_the_key_must_appear_in_the_gold_text():
    report = validate_eval(eval_set([item(answer_key=["tres mensualidades"])]), make_corpus())
    assert failed_checks(report) == ["key_in_gold"]


def test_every_gold_section_contains_a_key():
    # Como vale cualquiera, una sección sin la respuesta daría un acierto de recuperación sin entregarla.
    report = validate_eval(eval_set([item(gold_sections=[DEPOSIT, COPY_DEPOSIT, TERM])]), make_corpus())
    assert failures_by_check(report)["key_in_gold"] == [
        f"q1: {TERM} no contiene ninguna clave de ['dos mensualidades']"]


def test_a_twin_in_force_that_has_the_key_must_be_in_the_gold():
    report = validate_eval(eval_set([item(gold_sections=[DEPOSIT])]), make_corpus())
    assert failures_by_check(report)["twins_in_gold"] == [f"q1: falta el gemelo {COPY_DEPOSIT}"]


def test_a_twin_without_the_key_is_not_required():
    # "Artículo 2" está en la sección de LAW y no en su gemelo: el gemelo no da esta respuesta.
    assert validate_eval(eval_set([item(gold_sections=[DEPOSIT], answer_key=["artículo 2"])]), make_corpus()).ok


def test_repeated_ids_fail():
    report = validate_eval(eval_set([item(), item()]), make_corpus())
    assert failures_by_check(report)["unique_ids"] == ["q1: aparece 2 veces"]


def test_a_probe_key_may_come_from_the_conversation_but_its_gold_must_exist():
    with_gold = probe("s1.p9", gold_sections=[DEPOSIT, COPY_DEPOSIT], answer_key=["Palma"])
    assert validate_eval(eval_set(scenarios=[scenario(probes=scenario().probes + [with_gold])]), make_corpus()).ok
    bad_gold = probe("s1.p9", gold_sections=[f"{LAW}#a99@20000101"])
    report = validate_eval(eval_set(scenarios=[scenario(probes=scenario().probes + [bad_gold])]), make_corpus())
    assert failed_checks(report) == ["gold_exists"]


# --- check_sizes ---------------------------------------------------------------------------------

def question(category: str, item_id: str, split: str) -> EvalItem:
    if category == "out_of_corpus":
        return out_of_corpus(item_id=item_id, split=split)
    return item(item_id=item_id, category=category, split=split)


def smallest_full_set() -> tuple[list[EvalItem], list[Dialogue], list[Scenario]]:
    """Lo mínimo que pide §7.3: 30 preguntas test (3 fuera del corpus, 5 de identificador, 5 de
    metadatos), una dev por categoría, 8 guiones de 3 turnos y 3 escenarios de 2 sesiones y 5 sondas."""
    categories = ["out_of_corpus"] * 3 + ["identifier"] * 5 + ["metadata"] * 5 + ["topical"] * 17
    items = [question(category, f"t{n}", "test") for n, category in enumerate(categories)]
    items += [question(category, f"d{n}", "dev") for n, category in enumerate(CATEGORIES)]
    return items, [dialogue(f"g{n}") for n in range(8)], [scenario(f"s{n}") for n in range(3)]


def test_the_smallest_full_set_passes_check_sizes():
    assert check_sizes(eval_set(*smallest_full_set())).ok


def test_check_sizes_counts_the_test_split():
    items, dialogues, scenarios = smallest_full_set()
    # d0 (la dev de topical) pasa a test y t0 (una fuera del corpus de test) pasa a dev.
    moved = {"d0": "test", "t0": "dev"}
    items = [item.model_copy(update={"split": moved[item.item_id]}) if item.item_id in moved else item
             for item in items]
    report = check_sizes(eval_set(items, dialogues, scenarios))
    assert failures_by_check(report)["single_turn"] == [
        "2 de out_of_corpus en test, mínimo 3", "categorías sin preguntas dev: topical"]


def test_check_sizes_bounds_dialogues_and_scenarios():
    items, dialogues, scenarios = smallest_full_set()
    dialogues[0] = dialogue("g0", n_turns=7)
    scenarios = scenarios[:2]
    report = check_sizes(eval_set(items, dialogues, scenarios))
    assert failures_by_check(report)["dialogues"] == ["g0: 7 turnos, entre 3 y 6"]
    assert failures_by_check(report)["scenarios"] == ["2 escenarios en test, entre 3 y 6"]


# --- Congelar y cargar ---------------------------------------------------------------------------

def write_set(eval_dir, items, dialogues, scenarios) -> None:
    write_jsonl(eval_dir / "single_turn.jsonl", items)
    write_jsonl(eval_dir / "dialogues.jsonl", dialogues)
    write_jsonl(eval_dir / "scenarios.jsonl", scenarios)


@pytest.fixture
def frozen(tmp_path):
    """Un conjunto mínimo completo, congelado contra el corpus de juguete: (eval_dir, documents.jsonl)."""
    documents = tmp_path / "documents.jsonl"
    write_jsonl(documents, make_corpus())
    eval_dir = tmp_path / "eval"
    write_set(eval_dir, *smallest_full_set())
    freeze_eval(eval_dir, documents)
    return eval_dir, documents


def test_freeze_writes_the_hashes_and_load_accepts_them(frozen):
    eval_dir, documents = frozen
    written = json.loads((eval_dir / "FROZEN.json").read_text(encoding="utf-8"))
    assert set(written) == {"files", "corpus_sha", "key", "frozen_at"}
    ev = load_eval(eval_dir, documents)
    assert len(ev.items) == 34 and written["key"] == ev.key()


def test_loading_a_frozen_set_needs_its_corpus(frozen):
    eval_dir, _ = frozen
    with pytest.raises(ValueError, match="documents_path"):
        load_eval(eval_dir)


def test_a_file_edited_after_freezing_is_rejected(frozen):
    eval_dir, documents = frozen
    items, _, _ = smallest_full_set()
    items[0] = items[0].model_copy(update={"question": "¿Otra pregunta?"})
    write_jsonl(eval_dir / "single_turn.jsonl", items)
    with pytest.raises(ValueError, match="cambió después de congelar: single_turn.jsonl"):
        load_eval(eval_dir, documents)


def test_another_corpus_is_rejected(frozen, tmp_path):
    eval_dir, _ = frozen
    corpus = make_corpus()
    corpus[1].title = "Otra norma"
    other = tmp_path / "other.jsonl"
    write_jsonl(other, corpus)
    with pytest.raises(ValueError, match="el corpus no es el del congelado"):
        load_eval(eval_dir, other)


def test_freezing_again_without_changes_does_nothing(frozen):
    eval_dir, documents = frozen
    before = (eval_dir / "FROZEN.json").read_text(encoding="utf-8")
    freeze_eval(eval_dir, documents)
    assert (eval_dir / "FROZEN.json").read_text(encoding="utf-8") == before  # ni siquiera cambia la fecha


def test_freezing_an_edited_set_needs_overwrite(frozen):
    eval_dir, documents = frozen
    items, dialogues, scenarios = smallest_full_set()
    write_set(eval_dir, items[:-1] + [items[-1].model_copy(update={"notes": "editada"})], dialogues, scenarios)
    with pytest.raises(FileExistsError, match="decision_log.md"):
        freeze_eval(eval_dir, documents)
    freeze_eval(eval_dir, documents, overwrite=True)
    assert load_eval(eval_dir, documents).items[-1].notes == "editada"


def test_freeze_refuses_a_set_that_does_not_validate(tmp_path):
    documents = tmp_path / "documents.jsonl"
    write_jsonl(documents, make_corpus())
    items, dialogues, scenarios = smallest_full_set()
    items[0] = item(item_id="t0", gold_sections=[f"{LAW}#a99@20000101"])
    write_set(tmp_path, items, dialogues, scenarios)
    with pytest.raises(ValueError, match="gold_exists"):
        freeze_eval(tmp_path, documents)
    assert not (tmp_path / "FROZEN.json").exists()


def test_freezing_again_against_another_corpus_needs_overwrite(frozen, tmp_path):
    eval_dir, _ = frozen
    corpus = make_corpus()
    corpus[1].title = "Otra norma"
    other = tmp_path / "other.jsonl"
    write_jsonl(other, corpus)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        freeze_eval(eval_dir, other)


def test_loading_as_frozen_a_set_that_was_never_frozen_fails(tmp_path):
    documents = tmp_path / "documents.jsonl"
    write_jsonl(documents, make_corpus())
    write_set(tmp_path / "eval", [item()], [], [])
    with pytest.raises(FileNotFoundError, match="no está congelado"):
        load_eval(tmp_path / "eval", documents)


def test_loading_without_freezing_checks_nothing(tmp_path):
    write_set(tmp_path, [item(gold_sections=[DEPOSIT, COPY_DEPOSIT, f"{LAW}#a99@20000101"])], [], [])
    ev = load_eval(tmp_path, require_frozen=False)  # el smoke se carga así; el id lo atrapa validate_eval
    assert failed_checks(validate_eval(ev, make_corpus())) == ["gold_exists"]


def test_a_bad_row_names_its_file_and_row(tmp_path):
    write_set(tmp_path, [item(), item(item_id="q2")], [], [])
    with open(tmp_path / "single_turn.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write('{"item_id": "q3", "question": "¿?", "category": "trivia"}\n')
    with pytest.raises(ValueError, match="single_turn.jsonl, fila 3"):
        load_eval(tmp_path, require_frozen=False)


def test_the_key_changes_when_any_file_changes(tmp_path):
    write_set(tmp_path, [item()], [dialogue()], [scenario()])
    before = load_eval(tmp_path, require_frozen=False).key()
    write_jsonl(tmp_path / "dialogues.jsonl", [dialogue(notes="otra nota")])
    assert load_eval(tmp_path, require_frozen=False).key() != before


# --- Preguntas de identificador derivadas --------------------------------------------------------

def article(section_id: str, title: str) -> tuple:
    return (section_id, date(2000, 1, 1), None, f"Artículo {section_id.split('#a')[1].split('@')[0]}. {title}.\n"
            "Texto del artículo.", [])


def test_derived_questions_ask_for_the_article_by_its_number():
    items = derive_identifier_items(make_corpus(), n_dev=0, n_test=2)
    by_doc = {item.gold_docs[0]: item for item in items}
    regional = by_doc[COPY]
    assert regional.question == "¿Qué regula el artículo 1 de la Ley 2/2000 (Illes Balears)?"
    assert regional.answer_key == ["Garantía del alquiler"] and regional.category == "identifier"
    assert validate_eval(eval_set(items), make_corpus()).ok  # el gold rige, tiene la clave y sus gemelos


def test_a_title_shared_by_two_articles_is_not_derived():
    one = make_doc(LAW, [article(f"{LAW}#a1@20000101", "Fianza"), article(f"{LAW}#a2@20000101", "Obras")],
                   rango="Ley", numero_oficial="1/2000", region="Estatal")
    other = make_doc(COPY, [article(f"{COPY}#a7@20000101", "Fianza")],
                     rango="Ley", numero_oficial="2/2000", region="Estatal")
    assert [c.title for c in identifier_candidates([one, other])] == ["Obras"]


def test_a_title_named_by_another_block_is_not_derived():
    one = make_doc(LAW, [article(f"{LAW}#a1@20000101", "Entrada en vigor"),
                         article(f"{LAW}#a2@20000101", "Objeto del programa"),
                         article(f"{LAW}#a3@20000101", "Objeto del programa"),
                         article(f"{LAW}#a4@20000101", "Alquiler solidario")],
                   rango="Ley", numero_oficial="1/2000", region="Estatal")
    other = make_doc(COPY, [(f"{COPY}#df-2@20000101", date(2000, 1, 1), None,
                             "Disposición final segunda. Entrada en vigor.\nEl día siguiente.", [])],
                     rango="Ley", numero_oficial="2/2000", region="Estatal")
    assert [c.title for c in identifier_candidates([one, other])] == ["Alquiler solidario"]


def test_norms_without_an_official_number_and_long_titles_are_skipped():
    unnumbered = make_doc(LAW, [article(f"{LAW}#a1@20000101", "Obras")], rango="Resolución", region="Estatal")
    long_title = make_doc(COPY, [article(f"{COPY}#a1@20000101", "Régimen de las obras de mejora")],
                          rango="Ley", numero_oficial="2/2000", region="Estatal")
    assert identifier_candidates([unnumbered, long_title]) == []


def test_a_decree_takes_a_masculine_article():
    decree = make_doc(LAW, [article(f"{LAW}#a3@20000101", "Ayudas")],
                      rango="Real Decreto", numero_oficial="42/2022", region="Estatal")
    regional = make_doc(COPY, [article(f"{COPY}#a1@20000101", "Obras")],
                        rango="Ley", numero_oficial="2/2000", region="Galicia")
    questions = {item.question for item in derive_identifier_items([decree, regional], n_dev=0, n_test=2)}
    assert "¿Qué regula el artículo 3 del Real Decreto 42/2022?" in questions


def test_an_article_number_repeated_in_its_norm_is_not_derived():
    # Un reglamento anexo puede repetir la numeración: "el artículo 3" ya no nombra un solo artículo.
    law = make_doc(LAW, [(f"{LAW}#a3@20000101", date(2000, 1, 1), None, "Artículo 3. Ayudas.\nTexto.", []),
                         (f"{LAW}#r3@20000101", date(2000, 1, 1), None, "Artículo 3. Solicitudes.\nTexto.", []),
                         (f"{LAW}#a4@20000101", date(2000, 1, 1), None, "Artículo 4. Obras.\nTexto.", []),
                         (f"{LAW}#a4-2@20000101", date(2000, 1, 1), None, "Artículo 4.bis. Reformas.\nTexto.", [])],
                   rango="Ley", numero_oficial="1/2000", region="Estatal")
    assert [c.title for c in identifier_candidates([law])] == ["Obras", "Reformas"]  # "4.bis" no es "4"


def test_derivation_fails_when_there_are_not_enough_norms():
    with pytest.raises(ValueError, match="se pidieron 4"):  # hay una estatal y una autonómica: alcanzan para 2
        derive_identifier_items(make_corpus(), n_dev=0, n_test=4)


def test_derivation_is_deterministic_and_puts_dev_first():
    first = derive_identifier_items(make_corpus(), n_dev=1, n_test=1, seed=3)
    assert first == derive_identifier_items(make_corpus(), n_dev=1, n_test=1, seed=3)
    assert [item.split for item in first] == ["dev", "test"]


# --- Rankings ------------------------------------------------------------------------------------

def test_first_rank_is_the_position_of_the_first_wanted_id():
    assert first_rank(["a", "b", "c", "b"], ["c", "b"]) == 2
    assert first_rank(["a"], ["z"]) is None
