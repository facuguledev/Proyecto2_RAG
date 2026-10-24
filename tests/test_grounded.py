"""Etapa 4: pasaje, entrega, puntaje sin juez y lectura de citas. Sin servidor: lo que llama al LLM se prueba
en la notebook."""
from datetime import date

import pytest

from chunking import Unit
from evalset import Answer
from grounded import (ABSTENTION, Gold, GroundedAnswer, build_messages, cited_blocks, deliver, delivery,
                      passage_block, prompt_spec, score, system_prompt, verdict_of)

A36 = "BOE-A-1994-26003#a36@20130606"


def unit(unit_id: str, text: str, section_id: str = A36, doc_id: str = "BOE-A-1994-26003") -> Unit:
    return Unit(unit_id=f"{section_id}~{unit_id}", section_id=section_id, doc_id=doc_id,
                parent_id=f"{section_id}~p1", start=0, end=len(text), text=text, prefix="",
                header="[Ley 29/1994 · Artículo 36. Fianza. · vigente desde junio de 2013]")


def words(text: str) -> int:
    return len(text.split())


# --- pasaje y entrega ----------------------------------------------------------------------------

def test_the_passage_carries_the_citable_block_id_first():
    block = passage_block(unit("c1", "una mensualidad"))
    assert block.startswith("[BOE-A-1994-26003#a36 · Ley 29/1994 · Artículo 36.")
    assert block.endswith("\nuna mensualidad")


def test_deliver_skips_what_does_not_fit_and_keeps_going():
    big, small = unit("c1", "x " * 500), unit("c2", "fianza de una mensualidad")
    budget = words(passage_block(small)) + 5
    delivered = deliver([big, small], budget, words)
    assert [u.unit_id.split("~")[1] for u in delivered] == ["c2"]  # con un break, no se entregaba nada


def test_deliver_counts_the_id_too():
    u = unit("c1", "uno dos")
    exact = words(passage_block(u))
    assert deliver([u], exact, words) and not deliver([u], exact - 1, words)


def test_deliver_does_not_repeat_a_text():
    same = [unit("c1", "una mensualidad"), unit("c1", "una mensualidad", section_id="BOE-A-1994-26003#a36@19950101")]
    assert len(deliver(same, 10_000, words)) == 1


# --- prompt --------------------------------------------------------------------------------------

def test_closed_book_messages_have_no_context_block():
    messages = build_messages(system_prompt(date(2026, 10, 16), closed_book=True), "¿Cuánto es la fianza?", None)
    assert "<contexto>" not in messages[-1]["content"]


def test_history_goes_between_system_and_question():
    history = [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "hola, ¿qué querés saber?"}]
    messages = build_messages("sistema", "¿y la fianza?", [], history=history, memory="vive en Pamplona")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"].index("<memoria>") < messages[-1]["content"].index("<contexto>")


def test_schema_fields_sort_in_writing_order():
    fields = list(GroundedAnswer.model_fields)
    assert fields == sorted(fields) == ["analysis", "answer", "can_answer", "citations"]


def test_prompt_spec_changes_with_the_system_prompt():
    assert prompt_spec("a") == prompt_spec("a") and prompt_spec("a") != prompt_spec("b")


# --- veredicto y citas ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Sí, puede hacerlo.", "yes"), ("No: la ley exige seis meses.", "no"),
    ("Si el contrato es de 2016, sí.", "unclear"),  # condicional, no un sí
    ("El plazo es de cinco años.", "unclear"), ("", "unclear"),
    ("Sí, puede.", "yes"),  # "i" + tilde combinante (NFD)
    ("No obstante, sí puede hacerlo.", "unclear")])
def test_verdict_comes_from_the_first_raw_word(text, expected):
    assert verdict_of(text) == expected


def test_cited_blocks_accepts_brackets_and_versions():
    found = cited_blocks(["[BOE-A-1994-26003#a36 · Ley 29/1994]", "BOE-A-1994-26003#a9@20190306",
                          "BOE-A-1994-26003#a36", "BOE-A-1994-26003#a11."])
    assert found == ["BOE-A-1994-26003#a36", "BOE-A-1994-26003#a9", "BOE-A-1994-26003#a11"]


# --- score ---------------------------------------------------------------------------------------

GOLD = Answer(gold_sections=[A36], answer_key=["una mensualidad"])
YES_NO = Answer(gold_sections=["BOE-A-1994-26003#a11@20261008"], answer_key=["seis meses"], expected_verdict="no")
OUT = Answer(answerable=False)
DELIVERED = [unit("c1", "fianza en metálico de una mensualidad de renta")]
G = Gold({A36}, set())


def answer(text: str, can_answer: bool = True, citations=("BOE-A-1994-26003#a36",)) -> GroundedAnswer:
    return GroundedAnswer(analysis="...", answer=text, can_answer=can_answer, citations=list(citations))


def test_an_error_is_its_own_category():
    result = score(None, {"error": "PARSE_FAILED: ..."}, GOLD, DELIVERED, G)
    assert result["outcome"] == "error" and result["correct"] is None


def test_an_answer_cut_at_its_max_length_is_an_error():
    assert score(answer("x" * 1200), {}, GOLD, DELIVERED, G)["outcome"] == "error"


def test_correct_needs_the_key_in_answer_not_in_analysis():
    hidden = GroundedAnswer(analysis="es una mensualidad", answer="La que diga el contrato.", can_answer=True,
                            citations=[])
    assert score(hidden, {}, GOLD, DELIVERED, G)["correct"] is False
    assert score(answer("Una mensualidad de renta."), {}, GOLD, DELIVERED, G)["correct"] is True


def test_an_abstention_is_never_a_no():
    result = score(answer(ABSTENTION, can_answer=False, citations=()), {}, YES_NO, DELIVERED, Gold(set(), set()))
    assert result["outcome"] == "abstained" and result["correct"] is False and result["verdict_ok"] is False


def test_yes_no_needs_key_and_verdict():
    g = Gold(set(YES_NO.gold_sections), set())
    assert score(answer("No: hasta los seis meses no."), {}, YES_NO, [], g)["correct"] is True
    assert score(answer("Sí, pasados seis meses."), {}, YES_NO, [], g)["correct"] is False


def test_out_of_corpus_is_correct_only_when_abstaining():
    empty = Gold(set(), set())
    assert score(answer(ABSTENTION, can_answer=False, citations=()), {}, OUT, [], empty)["correct"] is True
    assert score(answer("Sí, 300 euros."), {}, OUT, [], empty)["correct"] is False


def test_citations_are_checked_against_what_was_delivered():
    result = score(answer("Una mensualidad.", citations=["BOE-A-1994-26003#a36", "BOE-A-2023-12203#a3"]),
                   {}, GOLD, DELIVERED, G)
    assert result["cited_not_delivered"] == ["BOE-A-2023-12203#a3"]
    assert result["cited_gold"] == ["BOE-A-1994-26003#a36"]


def test_cited_gold_is_checked_at_the_version_level():
    newer = [unit("c1", "otra redacción", section_id="BOE-A-1994-26003#a36@20261008")]
    assert score(answer("Una mensualidad."), {}, GOLD, newer, G)["cited_gold"] == []


def test_consistency_flags():
    flagged = score(answer("Una mensualidad.", can_answer=False), {}, GOLD, DELIVERED, G)
    assert flagged["flag_bool_says_abstain_text_answers"] is True
    clean = score(answer(ABSTENTION, can_answer=False, citations=()), {}, GOLD, DELIVERED, G)
    assert clean["flag_bool_says_abstain_text_answers"] is False


# --- delivery (las definiciones de la Etapa 3) ---------------------------------------------------

def test_a_document_hit_is_not_an_answer_delivered():
    other_article = unit("c1", "el arrendatario podrá desistir", section_id="BOE-A-1994-26003#a11@20261008")
    result = delivery(GOLD, [other_article], G)
    assert result["doc_hit"] is True
    assert result["section_delivered"] is False and result["gold_key_delivered"] is False


def test_a_passage_with_a_gold_text_counts_as_gold():
    copy = unit("c1", "una mensualidad", section_id="BOE-A-1994-26003#a36@19950101")  # otra versión, mismo texto
    assert delivery(GOLD, [copy], Gold({A36}, {"una mensualidad"}))["gold_key_delivered"] is True
    assert delivery(GOLD, [copy], Gold({A36}, {"una mensualidad"}))["gold_key_delivered_strict"] is False


def test_delivery_is_none_for_unanswerable():
    assert delivery(OUT, DELIVERED, Gold(set(), set()))["answer_delivered"] is None
