"""Canal léxico: términos, contaminación y BM25."""
import pytest

from lexical import Bm25Index, contamination, terms


# --- terms ---------------------------------------------------------------------------------------

def test_a_norm_number_is_a_single_term():
    found = terms("Según la Ley 29/1994, de Arrendamientos Urbanos")
    assert "29/1994" in found
    assert "29" not in found and "1994" not in found


def test_single_character_tokens_survive():
    assert "9" in terms("el artículo 9 de la ley")  # bm25s.tokenize lo descartaría


def test_spanish_stopwords_are_removed():
    assert terms("el plazo de la fianza") == terms("plazo fianza")


def test_stemming_joins_inflections_and_accents():
    assert terms("arrendamientos") == terms("arrendamiento")
    assert terms("información") == terms("informacion")
    assert terms("garantía") == terms("garantia")  # el stemmer solo, con la tilde, daba dos términos


def test_a_query_without_accents_drops_the_same_stopwords():
    assert terms("más plazo") == terms("mas plazo") == terms("plazo")


def test_a_decomposed_accent_does_not_split_the_word():
    assert terms("información") == terms("información")  # "o" + tilde combinante (NFD)


# --- contamination -------------------------------------------------------------------------------

GOLD = ("A la celebración del contrato será obligatoria la exigencia y prestación de fianza en "
        "metálico en cantidad equivalente a una mensualidad de renta en el arrendamiento de viviendas.")


def test_a_question_that_copies_the_passage_is_flagged():
    copied = "¿Es obligatoria la prestación de fianza en metálico en cantidad equivalente a una mensualidad?"
    result = contamination(copied, GOLD)
    assert result["flag"] and result["longest_shared_ngram"] >= 5


def test_a_paraphrased_question_is_not_flagged():
    paraphrased = "Voy a alquilar un piso: ¿cuánto dinero me pueden pedir por adelantado como garantía?"
    result = contamination(paraphrased, GOLD)
    assert not result["flag"] and result["longest_shared_ngram"] < 5


def test_overlap_is_the_share_of_question_terms_found_in_the_gold():
    result = contamination("fianza en metálico y garaje", GOLD)  # términos: fianz, metal, garaj
    assert result["overlap"] == round(2 / 3, 2)


def test_contamination_of_an_empty_question_is_zero():
    assert contamination("¿y?", GOLD) == {"overlap": 0.0, "longest_shared_ngram": 0, "flag": False}


# --- Bm25Index -----------------------------------------------------------------------------------

TEXTS = {
    "lau#a36": "Artículo 36. Fianza. A la celebración del contrato será obligatoria la fianza en metálico.",
    "lau#a9": "Artículo 9. Plazo mínimo. La duración del arrendamiento será libremente pactada por las partes.",
    "lph#a17": "Artículo 17. Los acuerdos de la Junta de propietarios se sujetarán a las siguientes normas.",
}


def make_index() -> Bm25Index:
    return Bm25Index(list(TEXTS), list(TEXTS.values()))


def test_search_ranks_the_matching_text_first():
    results = make_index().search("¿cuánto es la fianza de un alquiler?")
    assert results[0][0] == "lau#a36"


def test_search_leaves_out_texts_that_share_no_term_with_the_query():
    assert [doc_id for doc_id, _ in make_index().search("fianza")] == ["lau#a36"]
    assert make_index().search("zzz palabra inexistente") == []


def test_an_empty_query_finds_nothing():
    assert make_index().search("") == [] and make_index().search("de la el") == []  # solo stopwords


def test_ties_are_ordered_by_id_also_at_the_cut():
    index = Bm25Index(["b", "a"], ["fianza de un local", "fianza de una vivienda"])
    results = index.search("fianza")
    assert results[0][1] == results[1][1] and [doc_id for doc_id, _ in results] == ["a", "b"]
    assert [doc_id for doc_id, _ in index.search("fianza", k=1)] == ["a"]  # el corte respeta el mismo orden


def test_ids_and_texts_must_have_the_same_length():
    with pytest.raises(ValueError):
        Bm25Index(["a", "b"], ["un solo texto"])


def test_search_never_asks_for_more_results_than_texts():
    assert len(make_index().search("artículo", k=50)) == 3  # "artículo" está en los tres


def test_a_norm_number_finds_only_its_norm():
    index = Bm25Index(["a", "b"], ["La Ley 29/1994 de arrendamientos urbanos", "La Ley 12/2023 de vivienda"])
    assert [doc_id for doc_id, _ in index.search("Ley 29/1994")] == ["a", "b"]  # "ley" está en las dos
    assert [doc_id for doc_id, _ in index.search("29/1994")] == ["a"]


def test_allowed_filters_before_the_cut():
    index = Bm25Index(["a", "b", "c"], ["fianza fianza fianza", "fianza fianza", "fianza"])
    assert [doc_id for doc_id, _ in index.search("fianza", k=1)] == ["a"]
    # Filtrar después del top-1 dejaría la lista vacía; antes, devuelve el mejor de los permitidos.
    assert [doc_id for doc_id, _ in index.search("fianza", k=1, allowed=[False, True, True])] == ["b"]
