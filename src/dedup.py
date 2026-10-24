"""Casi-duplicados: detecta texto copiado entre documentos, no "temas parecidos".

Los temas parecidos los encuentran los embeddings. Acá: shingles de palabras + MinHash + LSH
proponen candidatos y el Jaccard exacto decide. Los shingles son los de la auditoría de la Etapa 0;
los coeficientes de MinHash salen de sha256, así el resultado no depende de la versión de NumPy.
"""
import hashlib
import re
import zlib
from itertools import combinations

import numpy as np

P = 4294967311  # primo justo arriba de 2**32, así que todo crc32 (< 2**32) queda por debajo de P


def shingles(text: str, k: int = 5) -> set[int]:
    """Hashes (crc32) de los k-gramas de palabras del texto. Con menos de k palabras, vacío."""
    words = re.findall(r"\w+", text.lower())
    return {zlib.crc32(" ".join(words[i:i + k]).encode()) for i in range(max(len(words) - k + 1, 0))}


def jaccard(a: set, b: set) -> float:
    """|a ∩ b| / |a ∪ b|. Dos conjuntos vacíos dan 0.0, no una división por cero."""
    common = len(a & b)  # |a ∪ b| = |a| + |b| - |a ∩ b|: sin armar la unión, que en normas largas pesa
    return common / max(len(a) + len(b) - common, 1)


def containment(a: set, b: set) -> float:
    """|a ∩ b| / min(|a|, |b|): qué fracción del más chico está en el otro.

    Jaccard subestima "A está (casi) adentro de B" cuando B es mucho más largo
    (un decreto-ley entero dentro de la ley que lo convalidó): la unión se infla
    con lo que B tiene de más. La contención sí lo detecta.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def permutation_coefficients(n_perm: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Los a, b de cada permutación h(x) = (a*x + b) % P, con 1 <= a < 2**31 y 0 <= b < 2**32.

    Salen de sha256 y no de numpy.random: un Generator de NumPy no garantiza el mismo stream entre
    versiones, y otra NumPy (la de Colab) cambiaría los gemelos y, con ellos, el hash del corpus.
    """
    def draw(name: str, i: int, high: int) -> int:
        digest = hashlib.sha256(f"{seed}:{name}:{i}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % high

    a = np.array([1 + draw("a", i, 2**31 - 1) for i in range(n_perm)], dtype=np.uint64)
    b = np.array([draw("b", i, 2**32) for i in range(n_perm)], dtype=np.uint64)
    return a, b


def minhash_signatures(sets: list[set[int]], n_perm: int = 128, seed: int = 0) -> np.ndarray:
    """Firma MinHash de cada conjunto: matriz (len(sets), n_perm) de uint64.

    Cada columna es una permutación h(x) = (a*x + b) % P y guarda el mínimo de h sobre el conjunto.
    La fracción de columnas en que dos firmas coinciden estima el Jaccard de los conjuntos.
    """
    a, b = permutation_coefficients(n_perm, seed)
    # Un conjunto vacío queda con P en todas las columnas: ningún hash real (mod P) llega a ese valor.
    signatures = np.full((len(sets), n_perm), P, dtype=np.uint64)
    for row, shingle_set in enumerate(sets):
        if shingle_set:
            x = np.fromiter(shingle_set, dtype=np.uint64)
            # a < 2**31 y x < 2**32, así que a*x + b < 2**64: la cuenta entra en uint64 sin desbordar.
            signatures[row] = ((a[:, None] * x[None, :] + b[:, None]) % P).min(axis=1)
    return signatures


def lsh_candidate_pairs(signatures: np.ndarray, bands: int = 32, rows: int = 4,
                        max_bucket: int = 200) -> set[tuple[int, int]]:
    """Pares (i, j), con i < j, de filas que coinciden en las `rows` columnas de al menos una banda.

    Cada firma se corta en `bands` bandas de `rows` columnas. Los buckets con max_bucket miembros
    o más se saltean: son texto repetido por todos lados (boilerplate) y generarían miles de
    pares que no dicen nada.
    """
    if bands * rows != signatures.shape[1]:
        raise ValueError(f"bands * rows = {bands * rows}, pero las firmas tienen {signatures.shape[1]} columnas")
    buckets: dict[tuple[int, bytes], list[int]] = {}
    for idx in range(len(signatures)):
        for band in range(bands):
            # El número de banda va en la clave: los mismos valores en bandas distintas no cuentan.
            key = (band, signatures[idx, band * rows:(band + 1) * rows].tobytes())
            buckets.setdefault(key, []).append(idx)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if 1 < len(members) < max_bucket:
            pairs.update(combinations(members, 2))
    return pairs


def similar_pairs(sets: list[set[int]], threshold: float, bands: int = 32, rows: int = 4,
                  max_bucket: int = 200, seed: int = 0) -> list[tuple[int, int, float]]:
    """Pares (i, j, jaccard) con Jaccard exacto >= threshold, de más a menos parecidos.

    MinHash + LSH solo proponen candidatos (es barato); lo que se compara con el umbral y se
    devuelve es el Jaccard exacto, no la estimación. Los conjuntos vacíos no forman pares.
    """
    signatures = minhash_signatures(sets, n_perm=bands * rows, seed=seed)
    pairs: list[tuple[int, int, float]] = []
    for i, j in lsh_candidate_pairs(signatures, bands, rows, max_bucket):
        if not sets[i] or not sets[j]:
            continue
        score = jaccard(sets[i], sets[j])
        if score >= threshold:
            pairs.append((i, j, score))
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    return pairs
