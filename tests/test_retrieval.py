"""Recuperación: guardas, canales, filtro, fusión, rerank, padres y entrega por presupuesto."""
from datetime import date

import numpy as np
import pytest

from chunking import Unit
from lexical import Bm25Index
from retrieval import (Filters, Index, RetrievalConfig, UnitMetadata, deliver, diff_rows, filter_mask,
                       lift_to_parents, parse_filters, parse_period, parse_regions, retrieve, rrf)


def unit(unit_id: str, text: str, n_gen: int = 10, parent_id: str | None = None) -> Unit:
    section_id = unit_id.split("~")[0]
    return Unit(unit_id, section_id, section_id.split("#")[0], parent_id or unit_id, 0, len(text), text,
                prefix=f"Ley 1/2000 | {text}", header="[Ley 1/2000]", n_rerank=5, n_gen=n_gen)


# Tres hijos: dos de un mismo padre (a1) y uno de otro (a2).
CHILDREN = [unit("L#a1@20000101~c1", "fianza de dos mensualidades", parent_id="L#a1@20000101~p1"),
            unit("L#a1@20000101~c2", "plazo de cinco años", parent_id="L#a1@20000101~p1"),
            unit("L#a2@20000101~c1", "obras de conservación", parent_id="L#a2@20000101~p1")]
PARENTS = {"L#a1@20000101~p1": unit("L#a1@20000101~p1", "fianza ... plazo", n_gen=30),
           "L#a2@20000101~p1": unit("L#a2@20000101~p1", "obras", n_gen=30)}


class KeywordEmbedder:
    """Un embedder de juguete: un eje por palabra clave."""

    AXES = ("fianza", "plazo", "obras")

    def encode(self, text, **kwargs):
        vector = np.array([float(word in text) for word in self.AXES])
        return vector / (np.linalg.norm(vector) or 1)


class LengthReranker:
    """Un reranker de juguete: prefiere el texto más corto."""

    def predict(self, pairs, **kwargs):
        return [-len(text) for _, text in pairs]


def full_index(**changes) -> Index:
    matrix = np.array([KeywordEmbedder().encode(child.text) for child in CHILDREN])
    pieces = dict(children=CHILDREN, parents=PARENTS, raw_matrix=matrix, prefixed_matrix=matrix,
                  bm25_raw=Bm25Index([c.unit_id for c in CHILDREN], [c.text for c in CHILDREN]),
                  bm25_prefixed=Bm25Index([c.unit_id for c in CHILDREN], [c.prefix for c in CHILDREN]),
                  embedder=KeywordEmbedder(), reranker=LengthReranker(),
                  metadata=UnitMetadata(valid_from=np.array([date(2000, 1, 1).toordinal()] * 3),
                                        valid_to=np.array([date(2010, 1, 1).toordinal(), date.max.toordinal(),
                                                           date.max.toordinal()]),
                                        region=np.array(["Estatal", "Estatal", "Illes Balears"], dtype=object)))
    return Index(**{**pieces, **changes})


# --- Guardas -------------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg, missing", [
    (RetrievalConfig(dense=True), "raw_matrix"),
    (RetrievalConfig(dense=True, prefix=True), "prefixed_matrix"),
    (RetrievalConfig(dense=True), "embedder"),
    (RetrievalConfig(dense=False, lexical=True), "bm25_raw"),
    (RetrievalConfig(dense=False, lexical=True, prefix=True), "bm25_prefixed"),
    (RetrievalConfig(rerank=True), "reranker"),
    (RetrievalConfig(parent=True), "parents"),
    (RetrievalConfig(date_filter=True), "metadata"),
    (RetrievalConfig(region_filter=True), "metadata"),
])
def test_a_flag_whose_piece_is_not_loaded_raises(cfg, missing):
    with pytest.raises(RuntimeError):
        retrieve("fianza", cfg, full_index(**{missing: None}))


def test_no_channel_raises():
    with pytest.raises(RuntimeError):
        retrieve("fianza", RetrievalConfig(dense=False, lexical=False), full_index())


def test_diff_rows_names_the_single_variable_and_flags_two():
    base = RetrievalConfig()
    assert diff_rows(base, RetrievalConfig(prefix=True)) == "one variable: prefix"
    assert "NOT a cumulative row" in diff_rows(base, RetrievalConfig(prefix=True, lexical=True))


# --- Piezas --------------------------------------------------------------------------------------

def test_rrf_sums_reciprocal_ranks_and_breaks_ties_by_id():
    assert rrf([["b", "a"], ["a", "b"]]) == ["a", "b"]  # empatados: gana el id menor
    assert rrf([["a", "b"], ["a", "c"]])[0] == "a"


def test_deliver_skips_what_does_not_fit_and_keeps_going():
    units = [unit("L#a1~c1", "uno", n_gen=50), unit("L#a2~c1", "grande", n_gen=200), unit("L#a3~c1", "dos", n_gen=40)]
    assert [u.unit_id for u in deliver(units, budget=100)] == ["L#a1~c1", "L#a3~c1"]  # sin break


def test_deliver_skips_a_text_already_delivered():
    units = [unit("L#a1@1~c1", "mismo texto"), unit("L#a1@2~c1", "mismo texto"), unit("L#a2~c1", "otro")]
    assert [u.unit_id for u in deliver(units, budget=100)] == ["L#a1@1~c1", "L#a2~c1"]


def test_lifting_to_parents_keeps_the_order_and_drops_repeats():
    ranked = ["L#a1@20000101~c2", "L#a2@20000101~c1", "L#a1@20000101~c1"]
    assert lift_to_parents(ranked, full_index()) == ["L#a1@20000101~p1", "L#a2@20000101~p1"]


def test_the_date_filter_uses_an_exclusive_end_and_rejects_empty_intervals():
    meta = UnitMetadata(valid_from=np.array([date(2000, 1, 1).toordinal(), date(2005, 3, 1).toordinal()]),
                        valid_to=np.array([date(2005, 3, 1).toordinal(), date(2005, 3, 1).toordinal()]),
                        region=np.array(["Estatal", "Estatal"], dtype=object))
    cfg = RetrievalConfig(date_filter=True)
    march = Filters(period=(date(2005, 3, 1), date(2005, 4, 1)))
    assert filter_mask(march, cfg, meta)[0].tolist() == [False, False]  # la 2.ª tiene intervalo vacío
    february = Filters(period=(date(2005, 2, 1), date(2005, 3, 1)))
    assert filter_mask(february, cfg, meta)[0].tolist() == [True, False]


def test_without_a_date_the_filter_keeps_what_rules_at_the_snapshot():
    allowed, info = filter_mask(Filters(), RetrievalConfig(date_filter=True), full_index().metadata)
    assert allowed.tolist() == [False, True, True] and info["period_source"] == "snapshot"


def test_the_region_filter_keeps_state_law():
    cfg = RetrievalConfig(region_filter=True)
    allowed, _ = filter_mask(Filters(regions=("Cataluña",)), cfg, full_index().metadata)
    assert allowed.tolist() == [True, True, False]


def test_a_filtered_child_never_comes_back_even_with_fewer_candidates():
    cfg = RetrievalConfig(dense=True, lexical=True, rerank=True, date_filter=True, region_filter=True)
    result = retrieve("fianza plazo obras", cfg, full_index(), Filters(regions=("Cataluña",)))
    assert "L#a1@20000101~c1" not in result.ranked  # no rige en la foto
    assert "L#a2@20000101~c1" not in result.ranked  # es balear
    assert all(u not in result.channels["dense"] for u in ("L#a1@20000101~c1", "L#a2@20000101~c1"))


def test_the_parent_row_has_the_same_section_ranking_as_the_child_row():
    child = retrieve("plazo fianza obras", RetrievalConfig(rerank=True), full_index()).to_row()
    parent = retrieve("plazo fianza obras", RetrievalConfig(rerank=True, parent=True), full_index()).to_row()
    assert child["sections"] == parent["sections"] and child["delivered"] != parent["delivered"]


# --- Parser --------------------------------------------------------------------------------------

@pytest.mark.parametrize("question, period", [
    ("Firmé el alquiler de mi casa en mayo de 2005 por un año.", (date(2005, 5, 1), date(2005, 6, 1))),
    ("Firmé un alquiler en 2017 y la renta se revisa.", (date(2017, 1, 1), date(2018, 1, 1))),
    ("Mi contrato de alquiler es de 2014. ¿Por cuánto se prorroga?", (date(2014, 1, 1), date(2015, 1, 1))),
    ("En diciembre de 2019, ¿qué regía?", (date(2019, 12, 1), date(2020, 1, 1))),
    ("Según la ley estatal por el derecho a la vivienda de 2023, ¿qué es un gran tenedor?", None),
    ("¿Qué regula el artículo 20 de la Ley 12/2023?", None),
    ("¿Qué tipo fijó el Plan de Vivienda 2005-2008?", None),
    ("Tengo 33 años y vivo de alquiler.", None),
    ("¿Qué dice la Ley de Arrendamientos Urbanos de 1994 sobre la fianza?", None),  # el año de la ley
    ("¿Qué decía la ley en marzo de 2023 sobre los desahucios?", (date(2023, 3, 1), date(2023, 4, 1))),
])
def test_parse_period(question, period):
    assert parse_period(question) == period


@pytest.mark.parametrize("question, regions", [
    ("En Navarra, ¿cuánto dura una habitación?", ("Comunidad Foral de Navarra",)),
    ("¿Qué regula el artículo 4 del Decreto-ley 3/2020 (Illes Balears)?", ("Illes Balears",)),
    ("En Cataluña, ¿cuánto tiempo vacío?", ("Cataluña",)),
    ("¿Puede la comunidad de vecinos prohibir los pisos turísticos?", ()),
    ("Soy propietario en Zaragoza.", ()),  # las ciudades no se mapean: un fallo declarado
    ("¿Qué dice la ley de vivienda de Castilla-La Mancha?", ("Castilla-La Mancha",)),
])
def test_parse_regions(question, regions):
    assert parse_regions(question) == regions


def test_parse_filters_reads_both_fields():
    assert parse_filters("En Navarra, en marzo de 2023") == Filters(
        period=(date(2023, 3, 1), date(2023, 4, 1)), regions=("Comunidad Foral de Navarra",))
