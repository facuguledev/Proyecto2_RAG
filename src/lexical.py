"""El canal léxico: qué es un término, el chequeo de contaminación y el índice BM25.

Una sola función, ``terms``, parte un texto en términos. La usan el índice BM25 y el chequeo de
contaminación del eval set: un chequeo que tokeniza distinto que BM25 mide un solapamiento distinto
del que infla a BM25 (assignment, Etapa 2).

No usamos ``bm25s.tokenize``: por defecto saca stopwords en inglés y descarta los tokens de un solo
carácter (el "9" de "artículo 9"), sin ningún aviso. A ``bm25s`` le pasamos nuestros términos.
"""
import re
import unicodedata

import bm25s
import Stemmer
from bm25s.tokenization import STOPWORDS_SPANISH

ACUTE = str.maketrans("áéíóú", "aeiou")  # la ñ y la ü se quedan: "año" no es "ano"
STOPWORDS = frozenset(word.translate(ACUTE) for word in STOPWORDS_SPANISH)  # las 313 de NLTK, sin tildes
_stemmer = Stemmer.Stemmer("spanish")

# Un número de norma ("29/1994") es un solo término. Partido en "29" y "1994" (dos números que
# aparecen en cientos de normas), una pregunta por la Ley 29/1994 no se distingue de una por la 12/2023.
TERM = re.compile(r"\d+/\d{4}|\w+")


def terms(text: str) -> list[str]:
    """Minúsculas y sin tildes, palabras y números de norma, sin stopwords y con stem en español.

    Las tildes se sacan antes del stemmer, que no las trata igual en todas las palabras ("garantía" daba
    "garant" y "garantia", "garanti"). Así una consulta escrita sin tildes encuentra lo mismo.
    """
    text = unicodedata.normalize("NFC", text).lower().translate(ACUTE)  # NFC: una tilde suelta no parte la palabra
    words = [word for word in TERM.findall(text) if word not in STOPWORDS]
    return _stemmer.stemWords(words)


# --- Contaminación -----------------------------------------------------------------------------

def ngrams(words: list[str], k: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def contamination(question: str, gold_text: str, n: int = 5) -> dict:
    """Cuánto copia una pregunta a su texto gold (el fragmento de la Etapa 2 del enunciado).

    - ``overlap``: qué fracción de los términos de la pregunta aparece en el gold;
    - ``longest_shared_ngram``: la secuencia de términos más larga que comparten, en el mismo orden;
    - ``flag``: esa secuencia tiene ``n`` términos o más, o sea una frase copiada y no un tema.
    """
    q, g = terms(question), terms(gold_text)
    gold_terms = set(g)
    overlap = sum(term in gold_terms for term in q) / max(len(q), 1)
    # De la más larga posible a la más corta: la primera que comparten es la más larga.
    longest = next((k for k in range(len(q), 0, -1) if ngrams(q, k) & ngrams(g, k)), 0)
    return {"overlap": round(overlap, 2), "longest_shared_ngram": longest, "flag": longest >= n}


# --- BM25 --------------------------------------------------------------------------------------

class Bm25Index:
    """BM25 sobre una lista de textos, cada uno con su id (en la Etapa 2, una sección entera)."""

    def __init__(self, ids: list[str], texts: list[str]):
        if len(ids) != len(texts):
            raise ValueError(f"{len(ids)} ids para {len(texts)} textos")
        self.ids = list(ids)
        self.bm25 = bm25s.BM25()
        self.bm25.index([terms(text) for text in texts], show_progress=False)

    def search(self, query: str, k: int = 10, allowed=None) -> list[tuple[str, float]]:
        """Los ``k`` mejores ``(id, puntaje)``, del más alto al más bajo; los empates, por id.

        Se puntúan todos los textos y se ordena acá: el top-k de bm25s deja los empates del corte en el
        orden interno de numpy, y entonces ``search(q, 1)`` no siempre sería el comienzo de
        ``search(q, 100)``. Quedan afuera los de puntaje 0: no comparten ningún término con la consulta, y
        contarlos como "recuperados" inflaría el hit@k de una consulta que no encontró nada.

        ``allowed`` (un booleano por texto, en el orden de ``ids``) filtra antes del top-k (Etapa 3). Filtrar
        después devolvería menos de ``k`` aunque hubiera más textos permitidos.
        """
        query_terms = terms(query)
        if not query_terms:
            return []  # vacía o de puras stopwords: no hay nada que buscar (y get_scores fallaría)
        scores = self.bm25.get_scores(query_terms)  # un puntaje por texto
        found = [i for i in range(len(self.ids)) if scores[i] > 0 and (allowed is None or allowed[i])]
        found.sort(key=lambda i: (-scores[i], self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in found[:k]]
