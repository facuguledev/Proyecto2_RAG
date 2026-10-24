"""Etapa 5: consultas, ventanas de historial, buffer de resumen y la conversación reproducida. Sin servidor
ni embedder: los contadores de tokens son palabras, y el LLM es una función de prueba."""
import re
from datetime import date

import pytest

import chunking
import conversation
from conversation import (ARMS, COMPARISONS, MESSAGE_MARGIN, REFERENCE, SUMMARY_MAX, HistoryArm, HistoryState,
                          Rewrite, Summary, SummaryBuffer, bound_query, conversation_system, full_conversation_query,
                          history_tokens, keep_last_tokens, passage_budget, query_limit, recall_probe, recall_system,
                          replay, rewrite, rewrite_messages, scripted_history, special_tokens, summarize, window,
                          window_losses, with_summary)
from evalset import Dialogue, Turn


def words(text: str) -> int:
    return len(text.split())


def word_offsets(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


class FakeTokenizer:
    """Palabras como tokens, con los especiales de XLM-RoBERTa: 2 en un texto, 4 en un par."""

    def __call__(self, a, b=None, add_special_tokens=True, return_offsets_mapping=False):
        n = len(a.split()) + (len(b.split()) if b else 0)
        out = {"input_ids": [0] * (n + ((4 if b else 2) if add_special_tokens else 0))}
        if return_offsets_mapping:
            out["offset_mapping"] = word_offsets(a)
        return out


def turn(user: str, depends_on: str = "standalone", answerable: bool = True, assistant: str = "respuesta",
         keys=("una mensualidad",)) -> Turn:
    gold = {"gold_sections": ["BOE-A-1994-26003#a36@20130606"], "answer_key": list(keys)} if answerable else {}
    return Turn(answerable=answerable, user=user, depends_on=depends_on, oracle_rewrite=f"reescrita: {user}",
                assistant=assistant, **gold)


DL1 = Dialogue(dialogue_id="dl01", turns=[turn("uno"), turn("dos", "coreference"), turn("tres", "ellipsis")])


def message(role: str, n_words: int) -> dict:
    return {"role": role, "content": " ".join(["w"] * n_words)}


def exchange(n_words: int) -> list[dict]:
    return [message("user", n_words), message("assistant", n_words)]


class FakeResult:
    def __init__(self, error=None):
        self.error = error

    def log(self) -> dict:
        return {"error": self.error, "output_tokens": 7}


# --- la conversación reproducida -----------------------------------------------------------------

def test_replay_keeps_ids_and_turn_numbers():
    turns = replay(DL1)
    assert [rt.turn_id for rt in turns] == ["dl01.t1", "dl01.t2", "dl01.t3"]
    assert [rt.number for rt in turns] == [1, 2, 3]


def test_the_history_is_the_scripted_one():
    history = scripted_history(replay(DL1)[:2])
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[1]["content"] == "respuesta"  # la guionada, no una generada


# --- Parte A -------------------------------------------------------------------------------------

def test_full_conversation_puts_the_newest_first_and_only_user_turns():
    history = scripted_history(replay(DL1)[:2])
    query, dropped = full_conversation_query(history, "tres", words, max_tokens=100)
    assert query.split(" \n") == ["tres", "dos", "uno"] and dropped == 0
    assert "respuesta" not in query


def test_full_conversation_drops_the_oldest_when_it_does_not_fit():
    history = [{"role": "user", "content": "viejo " * 10}, {"role": "assistant", "content": "x"},
               {"role": "user", "content": "nuevo"}, {"role": "assistant", "content": "y"}]
    query, dropped = full_conversation_query(history, "actual", words, max_tokens=3)
    assert query.split(" \n") == ["actual", "nuevo"] and dropped == 1


def test_full_conversation_keeps_the_current_question_even_if_it_alone_is_too_long():
    query, dropped = full_conversation_query([{"role": "user", "content": "otra"}], "x " * 50, words, max_tokens=10)
    assert query.startswith("x") and dropped == 1


# --- Parte A: la consulta acotada a la ventana del reranker o del embedder -----------------------

def test_special_tokens_are_measured_on_the_tokenizer():
    assert special_tokens(FakeTokenizer(), pair=False) == 2
    assert special_tokens(FakeTokenizer(), pair=True) == 4


def test_with_reranker_the_query_leaves_room_for_the_largest_child():
    assert query_limit(True, reranker_tokenizer=FakeTokenizer()) == 512 - chunking.CHILD_MAX - 4
    assert query_limit(False, embedder_tokenizer=FakeTokenizer()) == 510


def test_keep_last_tokens_keeps_the_end():
    assert keep_last_tokens("firmé en 2016 ¿cuánto me pedían?", 3, word_offsets) == ("¿cuánto me pedían?", True)
    assert keep_last_tokens("corta", 3, word_offsets) == ("corta", False)


def test_bound_query_drops_the_oldest_turns_of_the_full_conversation():
    history = [{"role": "user", "content": "uno dos tres"}, {"role": "assistant", "content": "r"},
               {"role": "user", "content": "cuatro"}, {"role": "assistant", "content": "r"}]
    query, cut = bound_query("full_conversation", "cinco", history, 2, word_offsets)
    assert query.split(" \n") == ["cinco", "cuatro"] and cut is True
    assert bound_query("full_conversation", "cinco", history, 10, word_offsets) == (
        "cinco \ncuatro \nuno dos tres", False)


def test_bound_query_cuts_a_long_single_query_and_flags_it():
    assert bound_query("rewrite", "a b c d", [], 2, word_offsets) == ("c d", True)
    assert bound_query("last_turn", "a b", [], 2, word_offsets) == ("a b", False)


def test_rewrite_sees_the_whole_dialogue_up_to_five_pairs():
    history = [m for n in range(6) for m in ({"role": "user", "content": f"u{n}"},
                                             {"role": "assistant", "content": f"a{n}"})]
    content = rewrite_messages(history, "¿y eso?")[-1]["content"]
    assert "Usuario: u0" not in content and "Usuario: u1" in content and "Asistente: a5" in content
    assert content.endswith("Última pregunta: ¿y eso?")


def test_the_first_turn_is_not_rewritten_and_makes_no_call():
    assert rewrite([], "¿Cuánto es la fianza?") == ("¿Cuánto es la fianza?", None)


def test_an_empty_rewrite_is_an_error_of_the_rewriter(monkeypatch):
    monkeypatch.setattr(conversation.llm, "structured", lambda *a, **k: (Rewrite(query="  "), FakeResult()))
    query, call = rewrite(scripted_history(replay(DL1)[:1]), "¿y eso?")
    assert query is None and call["error"].startswith("REWRITE_EMPTY")


def test_a_failed_rewrite_returns_none_with_the_error(monkeypatch):
    monkeypatch.setattr(conversation.llm, "structured", lambda *a, **k: (None, FakeResult("PARSE_FAILED: x")))
    assert rewrite(scripted_history(replay(DL1)[:1]), "¿y eso?") == (None, {"error": "PARSE_FAILED: x",
                                                                          "output_tokens": 7})


def test_rewrite_and_summary_schemas_have_no_reasoning_field():
    assert list(Rewrite.model_fields) == ["query"]
    assert list(Summary.model_fields) == ["summary"]


# --- Parte B: presupuestos y ventana -------------------------------------------------------------

def test_passages_get_what_the_history_did_not_use():
    assert passage_budget(120, total=1500) == 1380
    assert passage_budget(2000, total=1500) == 0


def test_window_keeps_the_newest_that_fit():
    history = [message("user", 10), message("assistant", 10), message("user", 3), message("assistant", 3)]
    kept, used = window(history, 2 * (3 + MESSAGE_MARGIN), words)
    assert kept == history[2:] and used == 2 * (3 + MESSAGE_MARGIN)


def test_window_never_starts_with_an_answer_without_its_question():
    history = [message("user", 10), message("assistant", 3), message("user", 3), message("assistant", 3)]
    kept, used = window(history, 3 * (3 + MESSAGE_MARGIN), words)
    assert [m["role"] for m in kept] == ["user", "assistant"]
    assert used == history_tokens(kept, words)


def test_window_counts_the_template_margin():
    history = [message("user", 5)]
    assert window(history, 5, words) == ([], 0)
    assert window(history, 5 + MESSAGE_MARGIN, words)[0] == history


def test_window_losses_measure_against_the_own_dialogue():
    dialogue = Dialogue(dialogue_id="d", turns=[turn("a " * 10), turn("b " * 10, "coreference"),
                                                turn("c " * 10, "ellipsis")])
    one_exchange = (10 + MESSAGE_MARGIN) + (1 + MESSAGE_MARGIN)  # "a a ... a" + "respuesta"
    rows = window_losses(replay(dialogue), cap=one_exchange + 10, count=words)
    assert [r["loses_turn_1"] for r in rows] == [False, False, True]  # el turno 3 ya no ve el turno 1
    assert rows[0]["history_tokens"] == 0 and rows[2]["lost_messages"] == 2
    assert not any(r["empty_by_cap"] for r in rows)


def test_a_small_cap_can_leave_a_turn_without_history():
    dialogue = Dialogue(dialogue_id="d", turns=[turn("a " * 30), turn("b", "coreference")])
    rows = window_losses(replay(dialogue), cap=20, count=words)  # el par mide 35 + 6: no entra ni el último
    assert [r["empty_by_cap"] for r in rows] == [False, True]  # el turno 1 no tenía historial: no cuenta


# --- Parte B: buffer de resumen ------------------------------------------------------------------

class FakeSummarizer:
    def __init__(self, fail: bool = False, text: str = "resumen"):
        self.calls, self.fail, self.text = [], fail, text

    def __call__(self, previous: str, evicted: list[dict]):
        self.calls.append((previous, evicted))
        if self.fail:
            return None, {"error": "PARSE_FAILED: prueba"}
        return f"{self.text} {len(self.calls)}", {"error": None}


def test_summary_buffer_does_nothing_below_the_trigger():
    summarizer = FakeSummarizer()
    buffer = SummaryBuffer(cap=100, count=words, summarizer=summarizer)
    assert buffer.add(exchange(10)) == [] and not summarizer.calls  # 30 tokens < 70


def test_summary_buffer_evicts_the_oldest_half_and_never_the_last_pair():
    summarizer = FakeSummarizer()
    buffer = SummaryBuffer(cap=100, count=words, summarizer=summarizer)
    for _ in range(3):
        buffer.add(exchange(10))  # 3 pares de 30 tokens: el tercero cruza 70
    assert len(summarizer.calls) == 1
    previous, evicted = summarizer.calls[0]
    assert previous == "" and len(evicted) == 4 and evicted[0]["role"] == "user"  # 2 de 3 pares
    summary, kept, used = buffer.view()
    assert summary == "resumen 1" and len(kept) == 2 and used <= 100
    assert buffer.updates[0]["after"] == "resumen 1" and buffer.updates[0]["at_maxlength"] is False


def test_with_two_pairs_only_the_older_goes():
    summarizer = FakeSummarizer(text="s " * 30)  # un resumen largo: aun así queda el último par
    buffer = SummaryBuffer(cap=100, count=words, summarizer=summarizer)
    buffer.add(exchange(15))
    buffer.add(exchange(15))  # 2 pares de 40 tokens cruzan 70
    assert len(summarizer.calls) == 1 and len(buffer.recent) == 2
    assert buffer.used() > 70  # sigue arriba del umbral, pero el último par no sale


def test_the_summary_takes_part_of_the_cap_and_the_view_fits():
    buffer = SummaryBuffer(cap=60, count=words, summarizer=FakeSummarizer(text="s " * 20))
    for _ in range(3):
        buffer.add(exchange(8))
    summary, kept, used = buffer.view()
    assert summary and used <= 60  # el resumen se cuenta, los recientes se recortan a lo que queda


def test_a_failed_summary_keeps_the_old_one_and_is_recorded():
    buffer = SummaryBuffer(cap=100, count=words, summarizer=FakeSummarizer(fail=True))
    for _ in range(3):
        buffer.add(exchange(10))
    assert buffer.summary == "" and buffer.updates[0]["error"].startswith("PARSE_FAILED")


def test_a_summary_at_the_cap_is_flagged():
    buffer = SummaryBuffer(cap=100, count=words, summarizer=lambda prev, ev: ("x" * SUMMARY_MAX, {"error": None}))
    for _ in range(3):
        buffer.add(exchange(10))
    assert buffer.updates[0]["at_maxlength"] is True


def test_an_empty_summary_is_an_error(monkeypatch):
    monkeypatch.setattr(conversation.llm, "structured", lambda *a, **k: (Summary(summary=" "), FakeResult()))
    new, call = summarize("", exchange(3))
    assert new is None and call["error"].startswith("SUMMARY_EMPTY")


def test_with_summary_without_summary_is_the_same_system():
    assert with_summary("sistema", "") == "sistema"
    assert "<resumen>" in with_summary("sistema", "el usuario alquila")


# --- Parte B: estados por brazo ------------------------------------------------------------------

def test_history_states_see_what_their_strategy_says():
    full = HistoryState(HistoryArm("full", "full", None), words)
    windowed = HistoryState(HistoryArm("w", "window", 20), words)
    for state in (full, windowed):
        for _ in range(3):
            state.add(exchange(5))
    assert len(full.view()[1]) == 6 and full.view()[2] == 60  # sin tope: entra todo lo guionado
    assert windowed.view()[2] <= 20 and len(windowed.view()[1]) == 2


def test_arms_need_what_their_strategy_uses():
    with pytest.raises(RuntimeError):
        HistoryState(HistoryArm("s", "summary", 150), words)  # sin summarizer
    with pytest.raises(RuntimeError):
        HistoryState(HistoryArm("w", "window", None), words)  # sin tope


def test_the_arms_are_the_caps_the_summary_and_the_reference():
    assert [a.name for a in ARMS] == ["window_h100", "window_h150", "window_h250", "summary_h150", "full_buffer"]
    assert conversation.PART_A_BUDGET == 1500 - 150  # los pasajes con el tope del experimento 2 lleno


def test_comparisons_move_one_variable_and_leave_the_reference_out():
    by_name = {arm.name: arm for arm in ARMS}
    for a, b in COMPARISONS:
        differs = [field for field in ("strategy", "cap") if getattr(by_name[a], field) != getattr(by_name[b], field)]
        assert len(differs) == 1, (a, b, differs)
    assert all(REFERENCE not in pair for pair in COMPARISONS)


# --- prompts y sonda del resumen -----------------------------------------------------------------

def test_conversation_prompt_is_stage_4_plus_one_rule():
    system = conversation_system(date(2026, 10, 16))
    assert "Respondé solo con lo que dicen los pasajes" in system and "conversación anterior" in system


def test_recall_prompt_allows_the_conversation_as_source():
    system = recall_system(date(2026, 10, 16))
    assert "Hoy es 2026-10-16" in system and "No vas a recibir pasajes" in system
    assert "Respondé solo con lo que dicen los pasajes" not in system


def test_recall_probe_is_derived_from_the_first_turn():
    question, target = recall_probe(DL1.turns[0])
    assert '"uno"' in question
    assert target.answerable and target.answer_key == ["una mensualidad"] and target.expected_verdict is None


def test_recall_probe_drops_keys_already_in_its_question():
    first = turn("¿Me pueden pedir una mensualidad de fianza o dos meses?", keys=("una mensualidad", "1 mes"))
    _, target = recall_probe(first)
    assert target.answer_key == ["1 mes"]
    _, none = recall_probe(turn("¿Es una mensualidad?", keys=("una mensualidad",)))
    assert none is None  # sin claves que no estén en la pregunta, el guion no tiene sonda
