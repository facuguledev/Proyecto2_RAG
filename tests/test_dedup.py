"""Tests de src/dedup.py: shingles, MinHash y LSH, con conjuntos chicos y sintéticos."""
import random
import zlib
from itertools import combinations

import numpy as np
import pytest

import dedup

VOCABULARY = [f"palabra{i}" for i in range(500)]

# Tres conjuntos anidados (1 adentro de 2, y 2 adentro de 0). Jaccard exacto de cada par:
# (0, 1) = 0.80 · (1, 2) = 0.808 · (0, 2) = 0.99
BASE = set(range(1000))
NESTED_SETS = [BASE, BASE - set(range(200)), BASE - set(range(10))]


def random_text(rng: random.Random, n_words: int = 200) -> str:
    """Sopa de palabras: dos textos así no comparten ningún 5-grama."""
    return " ".join(rng.choices(VOCABULARY, k=n_words))


def indices(pairs: list[tuple[int, int, float]]) -> list[tuple[int, int]]:
    """Los (i, j) de una lista de (i, j, jaccard)."""
    return [(i, j) for i, j, _ in pairs]


# --- shingles ---

def test_shingles_of_text_with_fewer_than_k_words_is_empty():
    assert dedup.shingles("uno dos tres cuatro") == set()  # 4 palabras, k=5 por defecto
    assert dedup.shingles("") == set()


def test_shingles_are_the_crc32_of_each_word_kgram():
    # El hashing tiene que ser idéntico al de la auditoría de la Etapa 0.
    expected = {zlib.crc32(b"a b c d e"), zlib.crc32(b"b c d e f")}
    assert dedup.shingles("a b c d e f") == expected
    assert len(dedup.shingles("a b c d e f", k=3)) == 4


def test_shingles_ignore_case():
    upper = dedup.shingles("Ley de Vivienda y Alquiler Urbano")
    lower = dedup.shingles("ley de vivienda y alquiler urbano")
    assert upper  # si fuera vacío, la igualdad no probaría nada
    assert upper == lower


# --- jaccard y containment ---

def test_jaccard_of_identical_overlapping_disjoint_and_empty_sets():
    assert dedup.jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert dedup.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5  # 2 en común de 4 en total
    assert dedup.jaccard({1, 2}, {3, 4}) == 0.0
    assert dedup.jaccard(set(), set()) == 0.0


def test_containment_is_one_for_a_subset_even_though_jaccard_is_low():
    small, big = set(range(10)), set(range(100))
    assert dedup.containment(small, big) == 1.0
    assert dedup.containment(big, small) == 1.0  # no depende del orden
    assert dedup.jaccard(small, big) == pytest.approx(0.1)


def test_containment_with_an_empty_set_is_zero():
    assert dedup.containment(set(), {1, 2}) == 0.0
    assert dedup.containment({1, 2}, set()) == 0.0
    assert dedup.containment(set(), set()) == 0.0


# --- minhash_signatures ---

def test_minhash_is_deterministic_for_the_same_seed():
    sets = [{1, 2, 3}, {2, 3, 4}]
    first = dedup.minhash_signatures(sets, seed=7)
    assert np.array_equal(first, dedup.minhash_signatures(sets, seed=7))
    assert not np.array_equal(first, dedup.minhash_signatures(sets, seed=8))  # la semilla se usa de verdad


def test_permutation_coefficients_are_pinned():
    # Fijados: si cambian (otra versión de Python o NumPy, o un cambio de código), cambian los gemelos
    # y el hash del corpus. Este test lo dice antes que el manifiesto.
    a, b = dedup.permutation_coefficients(3, seed=0)
    assert [int(x) for x in a] == [1945170834, 917336428, 755071974]
    assert [int(x) for x in b] == [1732412521, 3512549702, 2181462142]


def test_minhash_estimates_the_exact_jaccard():
    rng = np.random.default_rng(1)
    pool = rng.integers(0, 2**32, size=3000).tolist()  # enteros del tamaño de un crc32
    a, b = set(pool[:2000]), set(pool[667:2667])  # 2000 cada uno, 1333 en común: Jaccard ≈ 0.5
    signatures = dedup.minhash_signatures([a, b])
    estimate = (signatures[0] == signatures[1]).mean()
    assert abs(estimate - dedup.jaccard(a, b)) < 0.1


def test_minhash_gives_an_empty_set_a_row_full_of_p():
    signatures = dedup.minhash_signatures([set(), {1, 2, 3}], n_perm=16)
    assert signatures.shape == (2, 16)
    assert signatures.dtype == np.uint64
    assert (signatures[0] == dedup.P).all()
    assert (signatures[1] < dedup.P).all()  # un conjunto con elementos nunca llega a P: los hashes son mod P


# --- lsh_candidate_pairs ---

def test_lsh_rejects_bands_times_rows_different_from_the_signature_length():
    signatures = dedup.minhash_signatures([{1, 2, 3}], n_perm=128)
    with pytest.raises(ValueError):
        dedup.lsh_candidate_pairs(signatures, bands=10, rows=4)  # 40 != 128


def test_lsh_pairs_rows_that_share_at_least_one_band():
    signatures = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 0, 0, 0, 0],  # igual a la 0 en la banda 0
        [9, 9, 9, 9, 5, 6, 7, 8],  # igual a la 0 en la banda 1
        [5, 6, 7, 8, 1, 2, 3, 4],  # mismos valores que la 0 pero en la otra banda: no cuenta
    ], dtype=np.uint64)
    assert dedup.lsh_candidate_pairs(signatures, bands=2, rows=4) == {(0, 1), (0, 2)}


def test_lsh_skips_buckets_with_max_bucket_members_or_more():
    signatures = np.ones((5, 4), dtype=np.uint64)  # 5 filas idénticas: un solo bucket de 5
    assert dedup.lsh_candidate_pairs(signatures, bands=1, rows=4, max_bucket=5) == set()
    assert len(dedup.lsh_candidate_pairs(signatures, bands=1, rows=4, max_bucket=6)) == 10  # C(5, 2)


# --- similar_pairs ---

def test_similar_pairs_finds_the_planted_near_copy_and_no_unrelated_pair():
    rng = random.Random(0)
    texts = [random_text(rng) for _ in range(20)]
    words = texts[7].split()
    words[50], words[120] = "modificada", "derogada"  # copia casi literal: 2 palabras cambiadas
    texts.append(" ".join(words))

    pairs = dedup.similar_pairs([dedup.shingles(t) for t in texts], threshold=0.7)

    assert indices(pairs) == [(7, 20)]
    # 196 shingles; cada palabra cambiada rompe 5, así que comparten 186 de una unión de 206
    assert pairs[0][2] == pytest.approx(186 / 206)


def test_similar_pairs_are_sorted_by_descending_jaccard_then_by_index():
    assert indices(dedup.similar_pairs(NESTED_SETS, threshold=0.5)) == [(0, 2), (1, 2), (0, 1)]
    copies = [set(BASE) for _ in range(5)]  # los 10 pares dan 1.0: desempata el índice
    assert indices(dedup.similar_pairs(copies, threshold=0.5)) == list(combinations(range(5), 2))


def test_similar_pairs_keep_jaccard_equal_to_or_above_the_threshold():
    at_threshold = dedup.similar_pairs(NESTED_SETS, threshold=0.8)
    above_threshold = dedup.similar_pairs(NESTED_SETS, threshold=0.9)
    assert len(at_threshold) == 3  # el par (0, 1) tiene Jaccard 0.8 justo: entra
    assert indices(above_threshold) == [(0, 2)]


def test_similar_pairs_derives_n_perm_from_bands_and_rows():
    # 20 bandas de 3 filas = 60 permutaciones, no las 128 de siempre: tiene que funcionar igual
    assert len(dedup.similar_pairs(NESTED_SETS, threshold=0.5, bands=20, rows=3)) == 3


def test_similar_pairs_skip_empty_sets():
    # Dos vacíos tienen la misma firma (todo P): son candidatos, pero no hay texto que comparar.
    assert dedup.similar_pairs([set(), set()], threshold=0.0) == []
