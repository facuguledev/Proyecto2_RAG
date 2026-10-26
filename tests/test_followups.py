"""Etapa 8: la regla de la Etapa 3 repetida, qué versión llegó, las entregas con relleno, el resumen v2 y las
planillas de auditoría humana. Sin servidor ni modelos."""
import pytest

import conversation
import followups
from chunking import Unit
from conversation import SUMMARY_V2_ARM, HistoryState, SummaryBuffer

A9 = "BOE-A-1994-26003#a9"


def unit(section_id: str, text: str, n: int = 100, child: int = 1) -> Unit:
    return Unit(unit_id=f"{section_id}~c{child}", section_id=section_id, doc_id=section_id.split("#")[0],
                parent_id=f"{section_id}~p1", start=0, end=len(text), text=text, prefix="", header="[x]", n_gen=n)


def deliver(units, budget):
    """``retrieval.deliver`` sin el contador: en orden, lo que no entra se saltea, un texto repetido no entra."""
    out, seen, used = [], set(), 0
    for u in units:
        if u.text in seen or used + u.n_gen > budget:
            continue
        out.append(u)
        seen.add(u.text)
        used += u.n_gen
    return out


# --- La regla de la Etapa 3 ----------------------------------------------------------------------

PIECES = {"bm25": 1, "dense": 1, "+prefix": 2, "+date": 7}


def table(**rows):
    return {name: dict(enumerate(values)) for name, values in rows.items()}


def test_the_rule_prefers_fewer_pieces_when_the_top_row_is_not_distinguishable():
    primary = table(bm25=[0] * 10, dense=[0] * 10, **{"+prefix": [1] * 6 + [0] * 4, "+date": [1] * 8 + [0] * 2})
    assert followups.rule_choice(primary, PIECES, topical_ids=range(10)) == "+prefix"


def test_the_rule_keeps_the_top_row_when_it_is_distinguishable():
    primary = table(bm25=[0] * 10, dense=[0] * 10, **{"+prefix": [0] * 10, "+date": [1] * 8 + [0] * 2})
    assert followups.rule_choice(primary, PIECES, topical_ids=range(10)) == "+date"


def test_with_few_questions_the_rule_falls_back_to_a_baseline():
    """Lo que pasa en dev (8 preguntas con respuesta): casi nada se distingue, y gana la fila más simple."""
    primary = table(bm25=[1, 0, 0, 0], dense=[1, 1, 0, 0], **{"+prefix": [1, 1, 1, 0], "+date": [1, 1, 1, 1]})
    assert followups.rule_choice(primary, PIECES, topical_ids=range(4)) == "dense"


def test_rule_choice_counts_repeated_questions_in_a_resample():
    primary = table(bm25=[0, 0], dense=[0, 0], **{"+prefix": [1, 0], "+date": [1, 1]})
    # remuestreo con la pregunta 1 seis veces: +date gana 6-0 a +prefix (p = 0,03)
    assert followups.rule_choice(primary, PIECES, range(2), ids=[1] * 6) == "+date"
    # con las dos preguntas una vez, nada se distingue de +date (2-0, p = 0,5): gana la de menos piezas
    assert followups.rule_choice(primary, PIECES, range(2), ids=[0, 1]) == "bm25"


def test_rule_stability_is_seeded_and_counts_every_resample():
    primary = table(bm25=[0] * 10, dense=[0] * 10, **{"+prefix": [1] * 6 + [0] * 4, "+date": [1] * 8 + [0] * 2})
    a = followups.rule_stability(primary, PIECES, range(10), list(range(10)), n_boot=200, seed=1)
    b = followups.rule_stability(primary, PIECES, range(10), list(range(10)), n_boot=200, seed=1)
    assert a == b and sum(a.values()) == 200


# --- Qué versión llegó ---------------------------------------------------------------------------

def test_gold_versions_flags_the_other_versions_of_the_same_article():
    gold = f"{A9}@20190124"
    delivered = [unit(gold, "tres años"), unit(f"{A9}@20190306", "cinco años"), unit("OTRA#a1@20000101", "x")]
    out = followups.gold_versions(delivered, [gold], {gold})
    assert out == {"versions": [gold, f"{A9}@20190306"], "gold": True, "others": [f"{A9}@20190306"],
                   "only_gold": False}
    only = followups.gold_versions(delivered[:1], [gold], {gold})
    assert only["only_gold"] and only["others"] == []


def test_a_text_equivalent_version_counts_as_gold():
    gold, twin = f"{A9}@20190124", f"{A9}@20190125"
    out = followups.gold_versions([unit(twin, "tres años")], [gold], {gold, twin})
    assert out["gold"] and out["only_gold"]


def test_no_version_delivered():
    out = followups.gold_versions([unit("OTRA#a1@20000101", "x")], [f"{A9}@20190124"], {f"{A9}@20190124"})
    assert out == {"versions": [], "gold": False, "others": [], "only_gold": False}


# --- Entregas con relleno -------------------------------------------------------------------------

def test_distractors_drop_gold_sections_and_gold_texts():
    gold = f"{A9}@20190124"
    ranked = [unit(gold, "g"), unit("D#a1@1", "copia del gold"), unit("D#a2@1", "otro"), unit("D#a3@1", "más")]
    out = followups.distractors(ranked, {gold}, {"g", "copia del gold"})
    assert [u.section_id for u in out] == ["D#a2@1", "D#a3@1"]


def test_padded_oracle_puts_the_gold_first_and_fills_the_budget():
    gold = unit(f"{A9}@20190124", "g", n=300)
    ranked = [unit(f"D#a{i}@1", f"t{i}", n=400) for i in range(10)]
    out = followups.padded_oracle([gold], ranked, {gold.section_id}, {"g"}, 3000, deliver)
    assert out[0] is gold
    assert sum(u.n_gen for u in out) == 300 + 6 * 400  # lo que entra en 3.000
    assert all(u.section_id != gold.section_id for u in out[1:])


def test_positioned_keeps_the_same_passages_in_every_position():
    gold, others = unit("G#a1@1", "g"), [unit(f"D#a{i}@1", f"t{i}") for i in range(5)]
    sets = {p: followups.positioned(gold, others, p) for p in followups.POSITIONS}
    assert [u.section_id for u in sets["first"]][0] == "G#a1@1"
    assert [u.section_id for u in sets["middle"]][2] == "G#a1@1"
    assert [u.section_id for u in sets["last"]][-1] == "G#a1@1"
    assert all(sorted(u.unit_id for u in s) == sorted(u.unit_id for u in sets["first"]) for s in sets.values())
    with pytest.raises(ValueError):
        followups.positioned(gold, others, "end")


def test_position_set_leaves_room_for_the_gold():
    gold = unit("G#a1@1", "g", n=500)
    ranked = [unit(f"D#a{i}@1", f"t{i}", n=300) for i in range(30)]
    others = followups.position_set(gold, ranked, {"G#a1@1"}, {"g"}, 6000, deliver)
    assert sum(u.n_gen for u in others) + gold.n_gen <= 6000
    assert len(others) == 18
    with pytest.raises(ValueError):
        followups.position_set(unit("G#a1@1", "g", n=0), ranked, set(), set(), 6000, deliver)


def test_same_delivery_compares_ids_and_counts():
    units = [unit("D#a1@1", "x", n=10)]
    assert followups.same_delivery(units, [["D#a1@1~c1", 10]])
    assert not followups.same_delivery(units, [["D#a1@1~c1", 11]])


# --- El resumen v2 --------------------------------------------------------------------------------

def count(text: str) -> int:
    return len(text.split())


def pair(n: int) -> list[dict]:
    return [{"role": "user", "content": f"pregunta {n} " + "x " * 20},
            {"role": "assistant", "content": f"respuesta {n} " + "y " * 20}]


def test_the_v2_arm_uses_the_v2_limits_and_counts_copies():
    calls = []

    def summarizer(previous, evicted):
        calls.append((previous, evicted))
        return (previous or "resumen nuevo"), {"error": None}  # copia el anterior a partir de la segunda vez

    state = HistoryState(SUMMARY_V2_ARM, count, summarizer)
    assert state.buffer.max_chars == conversation.SUMMARY_V2_GRAMMAR_MAX
    for n in range(1, 6):
        state.add(pair(n))
    updates = state.buffer.updates
    assert len(updates) >= 2
    assert all("over_target" in u and "copied_previous" in u for u in updates)
    assert not updates[0]["copied_previous"] and updates[1]["copied_previous"]


def test_the_stage_5_summary_rows_keep_their_shape():
    """El brazo de la Etapa 5 no gana columnas nuevas: sus filas guardadas siguen siendo comparables."""
    buffer = SummaryBuffer(150, count, lambda prev, ev: ("r " * 300, {"error": None}))
    for n in range(1, 4):
        buffer.add(pair(n))
    assert set(buffer.updates[0]) == {"before", "evicted", "after", "error", "at_maxlength"}
    assert buffer.updates[0]["at_maxlength"]  # 600 caracteres pasan el tope de 220


def test_the_stage_5_arms_and_prompts_did_not_change():
    """La especificación de la Etapa 5 lleva vars() de cada brazo y los hashes del prompt y el schema del
    resumen: si cambiaran, la Etapa 5 no podría reanudarse ni reproducirse con sus filas guardadas."""
    assert [vars(a) for a in conversation.ARMS] == [
        {"name": "window_h100", "strategy": "window", "cap": 100},
        {"name": "window_h150", "strategy": "window", "cap": 150},
        {"name": "window_h250", "strategy": "window", "cap": 250},
        {"name": "summary_h150", "strategy": "summary", "cap": 150},
        {"name": "full_buffer", "strategy": "full", "cap": None}]
    assert conversation.Summary.model_json_schema()["properties"]["summary"]["maxLength"] == 220
    assert conversation.summary_messages("", [])[0]["content"] == conversation.SUMMARY_SYSTEM
    assert SUMMARY_V2_ARM.name not in {a.name for a in conversation.ARMS}


def test_summary_v2_needs_a_summarizer():
    with pytest.raises(RuntimeError):
        HistoryState(SUMMARY_V2_ARM, count, None)


# --- Planillas -----------------------------------------------------------------------------------

def test_blank_sheet_hides_the_previous_auditor():
    rows = [{"audit_id": "a1", "answer": "x", "human_verdict": "correct", "human_note": "ok", "auditor": "modelo"}]
    blank = followups.blank_sheet(rows)
    assert blank == [{"audit_id": "a1", "answer": "x", "human_verdict": "", "human_note": ""}]
    assert rows[0]["human_verdict"] == "correct"  # no toca la original


def test_sheet_agreement_counts_only_rows_filled_in_both():
    a = [{"id": "1", "human_verdict": "correct"}, {"id": "2", "human_verdict": "incorrect"},
         {"id": "3", "human_verdict": "correct"}]
    b = [{"id": "1", "human_verdict": "Correct "}, {"id": "2", "human_verdict": "correct"},
         {"id": "3", "human_verdict": ""}]
    out = followups.sheet_agreement(a, b, "id")
    assert out == {"n": 2, "agree": 1, "confusion": {("correct", "correct"): 1, ("incorrect", "correct"): 1},
                   "pending": 1}
