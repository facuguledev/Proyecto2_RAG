"""Métricas y ruido (Etapa 3; las Etapas 4-6 reusan el ruido).

Las métricas se calculan desde la fila cruda que guardó ``ResultsLog`` (``Retrieval.to_row``), nunca
dentro de ``retrieve``: arreglar una métrica no obliga a volver a correr la recuperación.

**Equivalencia por texto, en dos niveles.**
- *Sección:* 518 secciones son idénticas, byte a byte, a otra versión de su bloque (una reforma que se
  deroga y vuelve el texto anterior). Recuperar esa otra versión cuenta como acierto de sección.
- *Pasaje:* además, 6.064 hijos repiten el texto de un hijo de otra sección (el párrafo que una reforma
  no tocó). La entrega saltea la segunda copia de un texto, así que, si la copia vieja llega primero, la
  del gold no se entrega. Pero el generador recibió exactamente ese texto: cuenta como entregado desde el
  gold. Sin esto, las filas de hijos perdían aciertos que las de padres no pierden.

Al lado se reporta el acierto estricto (solo secciones gold o idénticas enteras).
"""
import math
from collections import defaultdict

import numpy as np

from chunking import Unit, passage_text, section_of
from evalset import Answer, Sections, block_of, contains_key, doc_of, first_rank, section_text
from retrieval import unique


def text_equivalents(gold: list[str], sections: Sections) -> set[str]:
    """El gold más las versiones del mismo bloque con texto idéntico a alguna sección gold."""
    by_block = defaultdict(list)
    for section_id in sections:
        by_block[block_of(section_id)].append(section_id)
    accepted = set(gold)
    for section_id in gold:
        text = section_text(sections, section_id)
        accepted |= {other for other in by_block[block_of(section_id)] if section_text(sections, other) == text}
    return accepted


def gold_texts(accepted: set[str], units: dict[str, Unit]) -> set[str]:
    """Los textos de todas las unidades (hijos y padres) de las secciones aceptadas."""
    return {unit.text for unit in units.values() if unit.section_id in accepted}


def from_gold(delivered: list[Unit], accepted: set[str], texts: set[str]) -> list[Unit]:
    """Los pasajes entregados que cuentan como del gold: de una sección aceptada o con el texto de una."""
    return [u for u in delivered if u.section_id in accepted or u.text in texts]


def delivered_metrics(delivered: list[Unit], answer: Answer, accepted: set[str], texts: set[str]) -> dict:
    """Lo que le llegó al generador, contra el gold. Lo usan esta etapa y las Etapas 4-6 (``grounded``)."""
    found = from_gold(delivered, accepted, texts)

    def with_key(units: list[Unit]) -> bool:
        return any(contains_key(passage_text(u), answer.answer_key) for u in units)

    return {
        # Primaria (pre-registrada): un pasaje del gold, con la clave, llegó al generador.
        "gold_key_delivered": with_key(found),
        "gold_key_delivered_strict": with_key([u for u in delivered if u.section_id in accepted]),
        "section_delivered": bool(found),
        "answer_delivered": with_key(delivered),
    }


def item_metrics(row: dict, answer: Answer, accepted: set[str], texts: set[str], units: dict[str, Unit]) -> dict:
    """Las métricas de una pregunta con respuesta en una fila.

    ``accepted``: las secciones gold y sus idénticas (``text_equivalents``); ``texts``: los textos de sus
    unidades (``gold_texts``); ``units``: unit_id -> Unit, para leer lo entregado.
    """
    sections = row["sections"]
    gold = set(answer.gold_sections)
    doc_rank = first_rank(unique([doc_of(s) for s in sections]), answer.gold_docs)
    delivered = [units[unit_id] for unit_id, _ in row["delivered"]]
    return {
        "doc_rank": doc_rank,
        "block_rank": first_rank(unique([block_of(s) for s in sections]), [block_of(s) for s in gold]),
        "section_rank": first_rank(sections, accepted),
        "section_rank_strict": first_rank(sections, gold),
        "mrr10": 1 / doc_rank if doc_rank and doc_rank <= 10 else 0.0,
        **delivered_metrics(delivered, answer, accepted, texts),
        "tokens": sum(n for _, n in row["delivered"]),
        "n_delivered": len(delivered),
        # El rango del gold en cada canal, antes de fusionar: es lo que se mira al abrir la caja.
        "channel_ranks": {name: first_rank(ranking, accepted) for name, ranking in row["channels"].items()},
    }


def hit(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def duplicated_chars(delivered: list[str], units: dict) -> int:
    """Caracteres entregados más de una vez por el solape entre hijos de un mismo documento."""
    total, spans = 0, defaultdict(list)
    for unit_id in delivered:
        unit = units[unit_id]
        total += unit.end - unit.start
        spans[unit.doc_id].append((unit.start, unit.end))
    covered = 0
    for intervals in spans.values():
        end = -1
        for a, b in sorted(intervals):
            covered += max(0, b - max(a, end))
            end = max(end, b)
    return total - covered


# --- Ruido -----------------------------------------------------------------------------------------

def paired(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, int]:
    """Victorias, derrotas y empates de ``b`` contra ``a``, pregunta por pregunta (mismas claves)."""
    if set(a) != set(b):
        raise ValueError("una comparación pareada necesita las mismas preguntas en las dos filas")
    wins = sum(b[i] and not a[i] for i in a)
    losses = sum(a[i] and not b[i] for i in a)
    return wins, losses, len(a) - wins - losses


def sign_test(wins: int, losses: int) -> float:
    """Test de signos exacto, bilateral, sobre las discordantes. Con un resultado binario es el McNemar exacto.

    Solo las preguntas que cambiaron llevan información: ``p = 2 · P(X <= min(wins, losses))`` con X ~
    Binomial(n, 1/2). Un 5-0 da 0,0625 y un 6-0, 0,03125.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_indices(n_items: int, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Los mismos remuestreos para todas las filas: así el intervalo de una diferencia es pareado."""
    return np.random.default_rng(seed).integers(0, n_items, size=(n_boot, n_items))


def interval(values, indices: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Intervalo bootstrap (percentiles) de la media de ``values`` sobre los remuestreos dados."""
    means = np.asarray(values, dtype=float)[indices].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
