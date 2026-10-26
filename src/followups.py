"""La Etapa 8 (seguimiento): lo que la revisión de la entrega encontró sin medir, medido.

Cada función es pura (sin modelos ni servidor) para que se pueda probar sin GPU; la notebook
``08_followups.py`` las usa con el índice y el generador reales.

- ``rule_choice`` repite la regla pre-registrada de la Etapa 3 (``03_retrieval.py``, sección 10), para
  aplicarla en ``dev`` y medir qué tan estable es en ``test``.
- ``gold_versions`` dice qué versiones del artículo gold le llegaron al generador: es el mecanismo de las
  fallas md02 y md05 de la Etapa 4 (dos versiones a la vista, aplicó la equivocada).
- ``distractors``, ``padded_oracle`` y ``positioned`` arman las entregas del oráculo con relleno y del
  experimento de posición: el pasaje gold más pasajes recuperados que no son del gold, hasta un presupuesto.
"""
import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping

from chunking import Unit
from evalset import block_of

PIECES = ("dense", "prefix", "lexical", "rerank", "parent", "date_filter", "region_filter")
POSITIONS = ("first", "middle", "last")


# --- Verificaciones contra lo guardado ------------------------------------------------------------

def require(condition: bool, message: str) -> None:
    """Corta la etapa si lo que se recalcula hoy no es lo que guardó una etapa anterior: compararía contra
    filas hechas con otra cosa. Una sola función, para que la prueba de humo con datos falsos
    (``tests/test_followups_dryrun.py``) pueda contar las verificaciones en lugar de cortarse en la primera."""
    if not condition:
        raise RuntimeError(message)


# --- La regla de la Etapa 3 ----------------------------------------------------------------------

def sign_test(wins: int, losses: int) -> float:
    """El mismo test de signos exacto que ``evaluation.sign_test`` (sin importar numpy)."""
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n)


def rule_choice(primary: Mapping[str, Mapping[str, bool]], n_pieces: Mapping[str, int],
                topical_ids: Iterable[str], ids: list[str] | None = None) -> str:
    """La mejor fila según la regla pre-registrada de la Etapa 3, sobre las preguntas ``ids``.

    1. La fila con más aciertos en la primaria (en un empate, la de menos piezas).
    2. Entre las que no se distinguen de ella (test de signos, p >= 0,05), la de menos piezas; con las mismas
       piezas, la de más aciertos en ``topical`` y después la de más aciertos.

    ``ids`` puede repetir preguntas (un remuestreo bootstrap): cada aparición cuenta.
    """
    names = list(primary)
    ids = list(ids if ids is not None else next(iter(primary.values())))
    topical = set(topical_ids)
    hits = {name: sum(bool(primary[name][i]) for i in ids) for name in names}
    top = max(names, key=lambda name: (hits[name], -n_pieces[name]))
    tied = []
    for name in names:
        wins = sum(bool(primary[name][i]) and not primary[top][i] for i in ids)
        losses = sum(bool(primary[top][i]) and not primary[name][i] for i in ids)
        if sign_test(wins, losses) >= 0.05:
            tied.append(name)
    topical_hits = {name: sum(bool(primary[name][i]) for i in ids if i in topical) for name in names}
    return min(tied, key=lambda name: (n_pieces[name], -topical_hits[name], -hits[name]))


def rule_stability(primary: Mapping[str, Mapping[str, bool]], n_pieces: Mapping[str, int],
                   topical_ids: Iterable[str], ids: list[str], n_boot: int = 2000, seed: int = 0) -> Counter:
    """Cuántas veces elige la regla cada fila al remuestrear las preguntas (bootstrap con semilla)."""
    rng = random.Random(seed)
    topical = list(topical_ids)
    return Counter(rule_choice(primary, n_pieces, topical, [rng.choice(ids) for _ in ids]) for _ in range(n_boot))


# --- Qué versión llegó ---------------------------------------------------------------------------

def gold_versions(delivered: Iterable[Unit], gold_sections: Iterable[str], accepted: set[str]) -> dict:
    """Las versiones de los artículos gold que le llegaron al generador.

    - ``versions``: las secciones entregadas cuyo bloque es el de una sección gold, sin repetir y en orden;
    - ``gold``: si llegó alguna aceptada (el gold o una versión con el mismo texto);
    - ``others``: versiones del mismo artículo que no son el gold (las que confunden a una pregunta con fecha);
    - ``only_gold``: llegó el gold y ninguna otra versión de su artículo.
    """
    blocks = {block_of(s) for s in gold_sections}
    versions = list(dict.fromkeys(u.section_id for u in delivered if block_of(u.section_id) in blocks))
    gold = [s for s in versions if s in accepted]
    others = [s for s in versions if s not in accepted]
    return {"versions": versions, "gold": bool(gold), "others": others, "only_gold": bool(gold) and not others}


# --- Entregas con relleno -------------------------------------------------------------------------

Deliver = Callable[[list[Unit], int], list[Unit]]  # grounded.deliver con su contador ya puesto


def distractors(ranked: Iterable[Unit], accepted: set[str], gold_texts: set[str]) -> list[Unit]:
    """Los pasajes recuperados que no son del gold: ni de una sección aceptada ni con el texto de una unidad
    gold (``evaluation.from_gold`` los contaría como gold). Conservan el orden del ranking: son los más
    parecidos a la pregunta, que es lo que hace difícil ignorarlos."""
    return [u for u in ranked if u.section_id not in accepted and u.text not in gold_texts]


def padded_oracle(gold_units: list[Unit], ranked: Iterable[Unit], accepted: set[str], gold_texts: set[str],
                  budget: int, deliver: Deliver) -> list[Unit]:
    """El oráculo de la Etapa 4 (los padres gold, primero los que tienen la clave) más distractores hasta el
    presupuesto: los mismos tokens que la fila de recuperación, con el gold garantizado. Lo único que lo
    separa del oráculo sin relleno son los distractores."""
    return deliver(list(gold_units) + distractors(ranked, accepted, gold_texts), budget)


def positioned(gold: Unit, others: list[Unit], position: str) -> list[Unit]:
    """El pasaje gold primero, en el medio o último entre ``others`` (que no cambian de orden entre sí)."""
    if position not in POSITIONS:
        raise ValueError(f"posición desconocida: {position!r}")
    at = {"first": 0, "middle": len(others) // 2, "last": len(others)}[position]
    return others[:at] + [gold] + others[at:]


def position_set(gold: Unit, ranked: Iterable[Unit], accepted: set[str], gold_texts: set[str], budget: int,
                 deliver: Deliver) -> list[Unit]:
    """Los distractores del experimento de posición: los que entran en ``budget`` menos el pasaje gold. Son
    los mismos para las tres posiciones, así que las tres entregas tienen exactamente los mismos tokens."""
    if gold.n_gen <= 0:
        raise ValueError("el pasaje gold tiene que venir contado (grounded.as_passages) antes de armar el set")
    return deliver(distractors(ranked, accepted, gold_texts), budget - gold.n_gen)


def same_delivery(a: Iterable[Unit], b: Iterable[list]) -> bool:
    """Si una entrega recalculada (unidades) es igual a una guardada (``[[unit_id, n_gen], ...]``)."""
    return [[u.unit_id, u.n_gen] for u in a] == [list(x) for x in b]


# --- Auditorías humanas ---------------------------------------------------------------------------

def blank_sheet(rows: list[dict], verdict_field: str = "human_verdict", note_field: str = "human_note",
                drop: tuple[str, ...] = ("auditor",)) -> list[dict]:
    """La misma planilla, con los veredictos y las notas vacías y sin la columna que dice quién la llenó:
    una persona la completa sin ver lo que dijo el auditor anterior."""
    out = []
    for row in rows:
        row = {k: v for k, v in row.items() if k not in drop}
        row[verdict_field], row[note_field] = "", ""
        out.append(row)
    return out


def sheet_agreement(a: list[dict], b: list[dict], key: str, verdict_field: str = "human_verdict") -> dict:
    """Acuerdo entre dos planillas completas de la misma muestra (fila por fila, por ``key``).

    Solo cuentan las filas con veredicto en las dos; ``confusion`` es (veredicto de a, veredicto de b)."""
    va = {r[key]: (r.get(verdict_field) or "").strip().lower() for r in a}
    vb = {r[key]: (r.get(verdict_field) or "").strip().lower() for r in b}
    both = [k for k in va if va[k] and vb.get(k)]
    confusion = Counter((va[k], vb[k]) for k in both)
    return {"n": len(both), "agree": sum(va[k] == vb[k] for k in both), "confusion": dict(confusion),
            "pending": sum(1 for k in va if not vb.get(k))}
