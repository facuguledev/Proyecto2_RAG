"""Métricas desde las filas crudas, equivalencia por texto y ruido."""
from datetime import date

import pytest

from chunking import Unit
from doc_schema import DocMetadata, Document, Section
from evalset import EvalItem, section_index
from evaluation import (bootstrap_indices, duplicated_chars, gold_texts, interval, item_metrics, paired,
                        sign_test, text_equivalents)

LAW = "BOE-A-2000-1"
OLD, NEW, BACK = f"{LAW}#a9@20000101", f"{LAW}#a9@20100101", f"{LAW}#a9@20150101"
OTHER = f"{LAW}#a2@20000101"


def corpus() -> list[Document]:
    """El art. 9 cambia en 2010 y en 2015 vuelve el texto del 2000 (una reforma derogada)."""
    texts = [(OLD, "Artículo 9. Plazo de tres años."), (NEW, "Artículo 9. Plazo de cinco años."),
             (BACK, "Artículo 9. Plazo de tres años."), (OTHER, "Artículo 2. Fianza de dos mensualidades.")]
    sections, start = [], 0
    for section_id, text in texts:
        block = section_id.split("#")[1].split("@")[0]
        sections.append(Section(section_id=section_id, title=text, level=1, start=start, end=start + len(text),
                                block_id=block, kind="article", valid_from=date(2000, 1, 1)))
        start += len(text) + 2
    return [Document(doc_id=LAW, title="Ley 1/2000", text="\n\n".join(t for _, t in texts), sections=sections,
                     metadata=DocMetadata())]


def question(**changes) -> EvalItem:
    fields = dict(item_id="q1", question="¿Qué plazo?", category="metadata", topic="plazo",
                  gold_sections=[BACK], answer_key=["tres años"])
    return EvalItem(**{**fields, **changes})


def test_an_identical_version_of_the_same_block_counts_as_the_gold():
    assert text_equivalents([BACK], section_index(corpus())) == {BACK, OLD}


def row(sections: list[str], delivered: list[str]) -> dict:
    return {"sections": sections, "delivered": [[u, 10] for u in delivered], "channels": {"dense": sections}}


def unit(unit_id: str, text: str) -> Unit:
    return Unit(unit_id, unit_id.split("~")[0], LAW, unit_id, 0, len(text), text, "", "[Ley]")


UNITS = {u.unit_id: u for u in [
    unit(f"{OLD}~c1", "Artículo 9. Plazo de tres años."), unit(f"{NEW}~c1", "Artículo 9. Plazo de cinco años."),
    unit(f"{BACK}~c1", "Artículo 9. Plazo de tres años."),
    unit(f"{OTHER}~c1", "Fianza; el plazo es de tres años según el art. 9."),
    unit(f"{NEW}~c2", "Disposición común: el plazo es de tres años."), unit(f"{BACK}~c2", "Disposición común: el plazo es de tres años."),
]}


def metrics(sections: list[str], delivered: list[str], gold: str = BACK) -> dict:
    accepted = text_equivalents([gold], section_index(corpus()))
    return item_metrics(row(sections, delivered), question(gold_sections=[gold]), accepted,
                        gold_texts(accepted, UNITS), UNITS)


def test_metrics_credit_the_text_equivalent_version_and_keep_the_strict_rank():
    result = metrics([NEW, OLD], [f"{OLD}~c1"])
    assert result["section_rank"] == 2 and result["section_rank_strict"] is None
    assert result["gold_key_delivered"] and result["gold_key_delivered_strict"] and result["section_delivered"]
    assert result["doc_rank"] == 1 and result["mrr10"] == 1.0 and result["channel_ranks"] == {"dense": 2}


def test_a_passage_identical_to_a_gold_passage_counts_as_delivered_from_the_gold():
    # La copia de NEW llegó primero, y la entrega salteó la de BACK por repetida: el texto llegó igual.
    result = metrics([NEW], [f"{NEW}~c2"])
    assert result["gold_key_delivered"] and not result["gold_key_delivered_strict"]


def test_a_key_delivered_from_another_section_is_not_the_primary_hit():
    result = metrics([OTHER], [f"{OTHER}~c1"])
    assert result["answer_delivered"] and not result["gold_key_delivered"] and not result["section_delivered"]


def test_mrr_counts_only_the_first_ten_documents():
    sections = [f"BOE-A-1999-{i}#a1@20000101" for i in range(10)] + [BACK]
    assert metrics(sections, [])["mrr10"] == 0.0


def test_duplicated_chars_measures_the_overlap_between_delivered_children():
    units = {"a": Unit("a", "s", LAW, "p", 0, 100, "", "", ""), "b": Unit("b", "s", LAW, "p", 80, 150, "", "", "")}
    assert duplicated_chars(["a", "b"], units) == 20


# --- Ruido -----------------------------------------------------------------------------------------

def test_sign_test_matches_the_assignment_numbers():
    assert sign_test(5, 0) == pytest.approx(0.0625)
    assert sign_test(6, 0) == pytest.approx(0.03125)
    assert sign_test(0, 0) == 1.0 and sign_test(3, 3) == 1.0


def test_paired_counts_wins_losses_and_ties():
    before = {"q1": True, "q2": False, "q3": False, "q4": True}
    after = {"q1": True, "q2": True, "q3": False, "q4": False}
    assert paired(before, after) == (1, 1, 2)
    with pytest.raises(ValueError):
        paired(before, {"q1": True})


def test_bootstrap_is_seeded_and_shared_across_rows():
    idx = bootstrap_indices(30, n_boot=2000, seed=0)
    assert (idx == bootstrap_indices(30, n_boot=2000, seed=0)).all()
    low, high = interval([1] * 25 + [0] * 5, idx)
    assert low < 25 / 30 < high and high - low < 0.35
