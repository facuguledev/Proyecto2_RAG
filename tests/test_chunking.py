"""Unidades del índice: qué entra, padres, hijos, prefijo y cabecera."""
import re
from datetime import date

import pytest

import chunking
from chunking import (build_units, full_prefix, header, indexed_sections, norm_label, pack, section_of,
                      validity, windows)
from doc_schema import DocMetadata, Document, Section


class WordTokenizer:
    """Un tokenizer de juguete: cada palabra es un token. Imita lo que usamos de ``tokenizers.Tokenizer``."""

    def encode_batch(self, texts, add_special_tokens=False):
        return [WordEncoding(text) for text in texts]


class WordEncoding:
    def __init__(self, text: str):
        self.offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        self.ids = list(range(len(self.offsets)))


TOK = WordTokenizer()
LAW = "BOE-A-2000-1"


def make_law(sections: list[tuple], **metadata) -> Document:
    """``sections``: (block_id, kind, valid_from, valid_to, texto), en orden."""
    built, start = [], 0
    for block_id, kind, valid_from, valid_to, text in sections:
        section_id = f"{LAW}#{block_id}@{valid_from:%Y%m%d}"
        built.append(Section(section_id=section_id, title=text.split("\n")[0], level=1, start=start,
                             end=start + len(text), block_id=block_id, kind=kind, valid_from=valid_from,
                             valid_to=valid_to, path=["TÍTULO I. Del contrato"]))
        start += len(text) + 2
    fields = {"rango": "Ley", "numero_oficial": "1/2000", "region": "Estatal", **metadata}
    return Document(doc_id=LAW, title="Ley 1/2000, de 7 de enero, de alquileres", sections=built,
                    text="\n\n".join(s[4] for s in sections), metadata=DocMetadata(**fields))


def words(n: int, word: str = "palabra") -> str:
    return " ".join([word] * n)


@pytest.fixture
def small_budgets(monkeypatch):
    """Presupuestos chicos, en palabras, para ver el corte con textos cortos."""
    monkeypatch.setattr(chunking, "CHILD_MAX", 40)
    monkeypatch.setattr(chunking, "CHILD_OVERLAP", 5)
    monkeypatch.setattr(chunking, "PARENT_MAX", 60)
    monkeypatch.setattr(chunking, "PREFIX_MAX", 15)


def long_article() -> Document:
    text = "Artículo 1. Plazo.\n" + "\n".join(words(12, f"p{i}") for i in range(10))  # 10 párrafos de 12
    return make_law([("a1", "article", date(2000, 1, 1), None, text)])


# --- Qué entra -----------------------------------------------------------------------------------

def test_headings_initial_notes_and_versions_that_never_ruled_are_left_out():
    doc = make_law([
        ("ti", "heading", date(2000, 1, 1), None, "TÍTULO I. Del contrato"),
        ("ni", "initial_note", date(2000, 1, 1), None, "Incluye las correcciones."),
        ("a1", "article", date(2000, 1, 1), date(2000, 1, 1), "Artículo 1. Nunca rigió."),
        ("a2", "article", date(2000, 1, 1), None, "Artículo 2. Rige."),
    ])
    assert [section.block_id for _, section in indexed_sections([doc])] == ["a2"]


# --- Prefijo y cabecera --------------------------------------------------------------------------

def test_validity_has_month_and_year_but_no_day():
    doc = make_law([("a1", "article", date(2013, 6, 6), date(2018, 12, 19), "Artículo 1. Plazo.")])
    text = validity(doc.sections[0])
    assert text == "vigente desde junio de 2013 hasta diciembre de 2018"
    assert "6" not in text.split() and "19" not in text.split()  # un día chocaría con "artículo 6"


def test_an_open_version_has_no_end_date():
    doc = make_law([("a1", "article", date(2013, 6, 6), None, "Artículo 1. Plazo.")])
    assert validity(doc.sections[0]) == "vigente desde junio de 2013"


def test_the_prefix_goes_from_most_to_least_important():
    doc = make_law([("a1", "article", date(2013, 6, 6), None, "Artículo 1. Plazo.")])
    assert full_prefix(doc, doc.sections[0]) == ("Ley 1/2000 | Artículo 1. Plazo. | vigente desde junio de 2013 | "
                                                 "TÍTULO I. Del contrato | Ley 1/2000, de 7 de enero, de alquileres")


def test_a_norm_without_number_is_named_by_its_title_up_to_the_year():
    doc = make_law([("a1", "article", date(2000, 1, 1), None, "Artículo 1.")], numero_oficial=None, rango="Orden")
    doc = doc.model_copy(update={"title": "Orden de 20 de mayo de 1969 por la que se aprueba algo"})
    assert norm_label(doc) == "Orden de 20 de mayo de 1969"
    assert "None" not in header(doc, doc.sections[0])


def test_a_regional_norm_carries_its_region():
    doc = make_law([("a1", "article", date(2000, 1, 1), None, "Artículo 1.")], region="Illes Balears")
    assert norm_label(doc) == "Ley 1/2000 (Illes Balears)"


def test_a_long_prefix_is_cut_from_the_end(small_budgets):
    doc = make_law([("a1", "article", date(2000, 1, 1), None, "Artículo 1. Plazo.\n" + words(5))])
    doc = doc.model_copy(update={"title": words(50, "título")})
    _, children = build_units([doc], TOK, TOK)
    assert len(children[0].prefix.split()) == 15
    assert children[0].prefix.startswith("Ley 1/2000 | Artículo 1. Plazo. | vigente desde enero de 2000")


# --- Corte ---------------------------------------------------------------------------------------

def test_pack_respects_the_budget_and_repeats_a_short_last_piece():
    assert pack([3, 3, 3, 3], budget=7) == [[0, 1], [2, 3]]  # 3 + 1 + 3 = 7
    assert pack([3, 3, 3, 3], budget=7, carry_max=3) == [[0, 1], [1, 2], [2, 3]]
    assert pack([10, 2], budget=7) == [[0], [1]]  # una pieza más grande queda sola


def test_windows_overlap_and_reach_the_end():
    offsets = [(i * 2, i * 2 + 1) for i in range(10)]  # 10 tokens de un carácter
    assert windows(offsets, size=4, overlap=1) == [(0, 7), (6, 13), (12, 19)]


def test_children_fit_the_reranker_budget_with_the_prefix(small_budgets):
    parents, children = build_units([long_article()], TOK, TOK)
    assert len(children) > 1 and all(child.n_rerank <= 40 for child in children)


def test_units_are_slices_of_the_document_and_children_stay_inside_their_parent(small_budgets):
    doc = long_article()
    parents, children = build_units([doc], TOK, TOK)
    by_id = {parent.unit_id: parent for parent in parents}
    assert len(parents) > 1  # 130 palabras no entran en un padre de 60
    for unit in parents + children:
        assert unit.text == doc.text[unit.start:unit.end]
        assert section_of(unit.unit_id) == unit.section_id == doc.sections[0].section_id
    for child in children:
        parent = by_id[child.parent_id]
        assert parent.start <= child.start and child.end <= parent.end


def test_consecutive_children_share_a_short_paragraph(small_budgets):
    text = "Artículo 1.\n" + "\n".join(["corto uno dos", words(20, "x"), "corto tres", words(20, "y")])
    doc = make_law([("a1", "article", date(2000, 1, 1), None, text)])
    _, children = build_units([doc], TOK, TOK)
    assert any(a.end > b.start for a, b in zip(children, children[1:]))  # hay solape


def test_a_giant_paragraph_is_split_into_overlapping_windows(small_budgets):
    doc = make_law([("a1", "article", date(2000, 1, 1), None, "Artículo 1.\n" + words(100, "w"))])
    _, children = build_units([doc], TOK, TOK)
    assert len(children) > 2 and all(child.n_rerank <= 40 for child in children)
    assert all(a.end > b.start for a, b in zip(children[1:], children[2:]))  # las ventanas se solapan
    assert children[-1].end == doc.sections[0].end  # nada del final se pierde


def test_the_cut_does_not_depend_on_the_prefix_text(small_budgets):
    # El presupuesto del hijo descuenta el prefijo siempre: la variante sin prefijo usa los mismos tramos.
    _, children = build_units([long_article()], TOK, TOK)
    spans = [(child.start, child.end) for child in children]
    assert spans == sorted(spans) and all(child.prefix for child in children)


def test_the_header_is_counted_in_the_generator_tokens(small_budgets):
    parents, _ = build_units([long_article()], TOK, TOK)
    for parent in parents:
        assert parent.n_gen == len(f"{parent.header}\n{parent.text}".split())


def test_children_are_numbered_within_their_section(small_budgets):
    doc = make_law([("a1", "article", date(2000, 1, 1), None, "Artículo 1.\n" + words(60, "x")),
                    ("a2", "article", date(2000, 1, 1), None, "Artículo 2.\n" + words(60, "y"))])
    _, children = build_units([doc], TOK, TOK)
    for section in doc.sections:
        ids = [c.unit_id for c in children if c.section_id == section.section_id]
        assert ids == [f"{section.section_id}~c{n}" for n in range(1, len(ids) + 1)]
