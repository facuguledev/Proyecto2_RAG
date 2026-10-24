"""La recuperación (Etapa 3): una función, un flag por pieza, y la entrega por presupuesto de tokens.

``retrieve(query, cfg, index)`` hace, en este orden:
1. el filtro de metadatos (una máscara sobre los hijos, antes de buscar);
2. el canal denso y/o el léxico (100 candidatos cada uno);
3. RRF, si hay dos canales;
4. el reranker sobre los primeros hijos;
5. la subida a padres;
6. la entrega hasta el presupuesto (tokens del generador).

Si un flag pide una pieza que no está cargada, lanza una excepción: una pieza que avisa y sigue produce
una fila idéntica a la anterior, y se leería "sin efecto" en una tabla que nunca corrió la pieza.

En todo el pipeline los empates se ordenan por id, para que una corrida se pueda repetir.
"""
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np

from chunking import MONTHS, Unit, searchable_text, section_of
from corpus import SNAPSHOT
from evalset import normalize
from lexical import Bm25Index


@dataclass(frozen=True)
class RetrievalConfig:
    dense: bool = True
    prefix: bool = False
    lexical: bool = False
    rerank: bool = False
    parent: bool = False
    date_filter: bool = False  # la fecha de la pregunta o, si no nombra ninguna, lo vigente en la foto
    region_filter: bool = False
    budget_tokens: int = 3000  # entregados al generador, contados en SU tokenizer
    n_candidates: int = 100  # por canal
    rerank_top: int = 50
    rrf_k: int = 60  # la constante del paper (Cormack et al., 2009): no se ajusta con 30 preguntas


def diff_rows(prev: RetrievalConfig, cur: RetrievalConfig) -> str:
    """Qué cambió entre dos filas. Una fila acumulativa cambia exactamente una variable (el fragmento del enunciado)."""
    changed = [k for k, v in asdict(cur).items() if asdict(prev)[k] != v]
    if not changed:
        return "identical configuration -- a duplicate row, not a new one"
    return (f"one variable: {changed[0]}" if len(changed) == 1
            else f"{len(changed)} variables differ ({', '.join(changed)}) -- NOT a cumulative row")


# --- El filtro: qué se lee de la pregunta ----------------------------------------------------------

@dataclass(frozen=True)
class Filters:
    """Lo que el filtro sabe de una pregunta. En la Etapa 6 lo llena la memoria en lugar del parser."""

    period: tuple[date, date] | None = None  # [desde, hasta): el mes o el año que nombra la pregunta
    regions: tuple[str, ...] = ()  # con el nombre de DocMetadata.region


# Nombre de la comunidad (como en los metadatos) -> cómo se la nombra, ya normalizado (sin tildes ni ñ).
# Nunca "comunidad" sola: "la comunidad de vecinos" no es una región.
REGIONS = {
    "Andalucía": ("andalucia", "andaluza", "andaluz"),
    "Aragón": ("aragon", "aragonesa"),
    "Principado de Asturias": ("asturias", "asturiana"),
    "Illes Balears": ("illes balears", "islas baleares", "baleares", "balear"),
    "Canarias": ("canarias", "canaria"),
    "Cantabria": ("cantabria", "cantabra"),
    "Castilla y León": ("castilla y leon",),
    "Castilla-La Mancha": ("castilla-la mancha", "castilla la mancha"),
    "Cataluña": ("cataluna", "catalunya", "catalana"),
    "Comunitat Valenciana": ("comunitat valenciana", "comunidad valenciana", "valenciana"),
    "Extremadura": ("extremadura", "extremena"),
    "Galicia": ("galicia", "gallega"),
    "Comunidad de Madrid": ("comunidad de madrid", "madrid", "madrilena"),
    "Región de Murcia": ("region de murcia", "murcia", "murciana"),
    "Comunidad Foral de Navarra": ("navarra",),
    "País Vasco": ("pais vasco", "euskadi", "vasca"),
    "La Rioja": ("la rioja", "riojana"),
    "Ciudad de Ceuta": ("ceuta",),
    "Ciudad de Melilla": ("melilla",),
}
REGION_PATTERNS = {region: re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, aliases)) + r")(?!\w)")
                   for region, aliases in REGIONS.items()}

# Un año que es parte de un nombre no es una fecha: "Ley 12/2023", "Plan de Vivienda 2005-2008".
NORM_NUMBER = re.compile(r"\d+/\d{4}|\d{4}\s*-\s*\d{4}")
# Tampoco el año de una ley: "la Ley de Arrendamientos Urbanos de 1994", "la ley estatal por el derecho a la
# vivienda de 2023". Se borra después de buscar el mes: "la ley en marzo de 2023" sí es una fecha.
LAW_YEAR = re.compile(r"\bley\b[^.?!,;]{0,60}?\bde (?:19|20)\d{2}\b")
MONTH_NUMBER = {name: number for number, name in enumerate(MONTHS, start=1)} | {"setiembre": 9}
MONTH = "(" + "|".join(MONTH_NUMBER) + ")"
YEAR = r"((?:19|20)\d{2})"
MONTH_YEAR = re.compile(rf"\b{MONTH} de {YEAR}\b")  # "en mayo de 2005"
YEAR_AFTER = re.compile(rf"\b(?:en|desde|hasta) {YEAR}\b")  # "firmé un alquiler en 2017"
# "mi contrato de alquiler es de 2014": "de 2014" solo cuenta después de contrato/alquiler/arrendamiento.
# Suelto, "de 2023" sería también el año de cualquier norma.
CONTRACT_YEAR = re.compile(rf"\b(?:contrato|alquiler|arrendamiento)s?\b[^.?!]{{0,40}}?\bde {YEAR}\b")


def parse_period(question: str) -> tuple[date, date] | None:
    """El mes o el año por el que pregunta, como período [desde, hasta). La pregunta no es más precisa."""
    text = NORM_NUMBER.sub(" ", normalize(question))
    if match := MONTH_YEAR.search(text):
        month, year = MONTH_NUMBER[match.group(1)], int(match.group(2))
        return date(year, month, 1), date(year + month // 12, month % 12 + 1, 1)
    text = LAW_YEAR.sub(" ", text)
    years = [m for pattern in (YEAR_AFTER, CONTRACT_YEAR) if (m := pattern.search(text))]
    if years:
        year = int(min(years, key=lambda m: m.start()).group(1))  # el que aparece primero
        return date(year, 1, 1), date(year + 1, 1, 1)
    return None


def parse_regions(question: str) -> tuple[str, ...]:
    text = normalize(question)
    return tuple(region for region, pattern in REGION_PATTERNS.items() if pattern.search(text))


def parse_filters(question: str) -> Filters:
    return Filters(period=parse_period(question), regions=parse_regions(question))


def period_of(filters: Filters) -> tuple[date, date]:
    """El período que aplica el filtro de fecha: el de la pregunta o, si no nombra ninguno, el día de la foto.

    Una pregunta sin fecha pregunta por la ley de hoy: es la regla con la que se etiquetó su gold.
    """
    return filters.period or (SNAPSHOT, SNAPSHOT + timedelta(days=1))


# --- El índice -------------------------------------------------------------------------------------

@dataclass
class UnitMetadata:
    """La vigencia y la región de cada hijo, como arreglos alineados con ``Index.children``."""

    valid_from: np.ndarray  # ordinal de la fecha
    valid_to: np.ndarray  # ordinal; una versión abierta tiene un valor más grande que cualquier fecha
    region: np.ndarray


def unit_metadata(children: list[Unit], docs) -> UnitMetadata:
    sections = {section.section_id: section for doc in docs for section in doc.sections}
    regions = {doc.doc_id: doc.metadata.region for doc in docs}
    return UnitMetadata(
        valid_from=np.array([sections[u.section_id].valid_from.toordinal() for u in children], dtype=np.int64),
        valid_to=np.array([(sections[u.section_id].valid_to or date.max).toordinal() for u in children],
                          dtype=np.int64),
        region=np.array([regions[u.doc_id] for u in children], dtype=object),
    )


@dataclass
class Index:
    """Las piezas cargadas. Una pieza que no está es ``None``, y ``retrieve`` falla si una fila la pide."""

    children: list[Unit]
    parents: dict[str, Unit] | None = None
    raw_matrix: np.ndarray | None = None  # un embedding normalizado por hijo, sin prefijo
    prefixed_matrix: np.ndarray | None = None  # con prefijo
    bm25_raw: Bm25Index | None = None
    bm25_prefixed: Bm25Index | None = None
    embedder: object | None = None  # SentenceTransformer: codifica la consulta
    reranker: object | None = None  # CrossEncoder
    metadata: UnitMetadata | None = None
    by_id: dict[str, Unit] = field(init=False)
    position: dict[str, int] = field(init=False)
    id_rank: np.ndarray = field(init=False)  # el orden alfabético de cada id: desempata los puntajes iguales

    def __post_init__(self):
        self.by_id = {unit.unit_id: unit for unit in self.children}
        self.position = {unit.unit_id: i for i, unit in enumerate(self.children)}
        self.id_rank = np.argsort(np.argsort(np.array([unit.unit_id for unit in self.children], dtype=object)))


def check_pieces(cfg: RetrievalConfig, index: Index) -> None:
    if not (cfg.dense or cfg.lexical):
        raise RuntimeError("no hay canal de búsqueda: dense=False y lexical=False")
    if cfg.dense and index.embedder is None:
        raise RuntimeError("dense=True pero no hay un embedder cargado para codificar la consulta")
    if cfg.dense and (index.prefixed_matrix if cfg.prefix else index.raw_matrix) is None:
        raise RuntimeError(f"dense=True, prefix={cfg.prefix} pero esa matriz de embeddings no está cargada")
    if cfg.lexical and (index.bm25_prefixed if cfg.prefix else index.bm25_raw) is None:
        raise RuntimeError(f"lexical=True, prefix={cfg.prefix} pero ese índice BM25 no está cargado")
    if cfg.rerank and index.reranker is None:
        raise RuntimeError("rerank=True pero no hay un reranker cargado")
    if cfg.parent and index.parents is None:
        raise RuntimeError("parent=True pero no está cargado el mapa hijo -> padre")
    if (cfg.date_filter or cfg.region_filter) and index.metadata is None:
        raise RuntimeError("hay un filtro de metadatos prendido pero no están cargados los metadatos de los hijos")


# --- Las piezas ------------------------------------------------------------------------------------

def filter_mask(filters: Filters, cfg: RetrievalConfig, meta: UnitMetadata) -> tuple[np.ndarray, dict]:
    """Qué hijos puede devolver la búsqueda, y qué hizo el filtro (para el registro)."""
    allowed = np.ones(len(meta.region), dtype=bool)
    info = {"period": None, "period_source": None, "regions": []}
    if cfg.date_filter:
        start, end = period_of(filters)
        info.update(period=[start.isoformat(), end.isoformat()],
                    period_source="question" if filters.period else "snapshot")
        # La vigencia se solapa con el período. El fin es exclusivo, y una versión con intervalo vacío
        # nunca rigió (vf < vt): sin esa condición pasaría si su fecha cae dentro del período.
        allowed &= ((meta.valid_from < end.toordinal()) & (start.toordinal() < meta.valid_to)
                    & (meta.valid_from < meta.valid_to))
    if cfg.region_filter and filters.regions:
        allowed &= np.isin(meta.region, list(filters.regions) + ["Estatal"])  # una ley estatal rige en todas
        info["regions"] = list(filters.regions)
    info["n_allowed"] = int(allowed.sum())
    return allowed, info


def dense_search(query: str, matrix: np.ndarray, index: Index, allowed, n: int) -> tuple[list[str], float | None]:
    q = np.asarray(index.embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    if not np.isfinite(q).all():
        raise RuntimeError(f"el embedding de la consulta no es finito: {query!r}")
    scores = matrix @ q
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf)
    order = np.lexsort((index.id_rank, -scores))  # por puntaje y, en un empate, por id
    order = [i for i in order[:n] if allowed is None or allowed[i]]  # lo filtrado nunca vuelve
    return [index.children[i].unit_id for i in order], (float(scores[order[0]]) if order else None)


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, unit_id in enumerate(ranking, start=1):
            scores[unit_id] = scores.get(unit_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))


def rerank(query: str, ranked: list[str], cfg: RetrievalConfig, index: Index) -> tuple[list[str], float | None]:
    """Reordena los primeros ``rerank_top`` hijos con el cross-encoder; el resto queda detrás, como venía."""
    top = ranked[:cfg.rerank_top]
    if not top:
        return ranked, None
    pairs = [(query, searchable_text(index.by_id[unit_id], cfg.prefix)) for unit_id in top]
    scores = index.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    reordered = sorted(zip(top, map(float, scores)), key=lambda pair: (-pair[1], pair[0]))
    return [unit_id for unit_id, _ in reordered] + ranked[cfg.rerank_top:], reordered[0][1]


def lift_to_parents(ranked: list[str], index: Index) -> list[str]:
    """Cada hijo sube a su padre, en el mismo orden; un padre que ya está no se repite."""
    parents, seen = [], set()
    for unit_id in ranked:
        parent_id = index.by_id[unit_id].parent_id
        if parent_id not in seen:
            seen.add(parent_id)
            parents.append(parent_id)
    return parents


def deliver(units: list[Unit], budget: int) -> list[Unit]:
    """Llena el presupuesto en el orden del ranking: lo que no entra se saltea y se sigue.

    Un ``break`` en la primera unidad que no entra descartaría también las más chicas que vienen
    después. Una unidad cuyo texto ya se entregó (otra versión con el mismo texto) no aporta nada.
    """
    delivered, seen_texts, used = [], set(), 0
    for unit in units:
        if unit.text in seen_texts or used + unit.n_gen > budget:
            continue
        delivered.append(unit)
        seen_texts.add(unit.text)
        used += unit.n_gen
    return delivered


# --- retrieve --------------------------------------------------------------------------------------

@dataclass
class Retrieval:
    """Lo que devuelve ``retrieve``: el ranking final, lo entregado y lo que hizo cada pieza."""

    ranked: list[str]  # ids de unidades (hijos, o padres con parent=True), del mejor al peor
    delivered: list[Unit]
    channels: dict[str, list[str]]  # el ranking de hijos de cada canal (dense, bm25, fused, reranked)
    reranked: list[str]  # los hijos que leyó el reranker (para el chequeo de su ventana)
    filters: dict
    max_scores: dict[str, float | None]
    latency_ms: dict[str, float]

    def to_row(self, n: int = 100) -> dict:
        """Lo crudo para ``ResultsLog``: las métricas se calculan después, desde acá."""
        return {
            "sections": unique([section_of(u) for u in self.ranked])[:n],
            "channels": {name: unique([section_of(u) for u in ranking])[:n] for name, ranking in self.channels.items()},
            "reranked": self.reranked,
            "delivered": [[unit.unit_id, unit.n_gen] for unit in self.delivered],
            "filters": self.filters,
            "max_scores": self.max_scores,
            "latency_ms": self.latency_ms,
        }


def unique(ids: list[str]) -> list[str]:
    """Sin repetidos, conservando el primer lugar de cada uno."""
    return list(dict.fromkeys(ids))


def retrieve(query: str, cfg: RetrievalConfig, index: Index, filters: Filters | None = None) -> Retrieval:
    """Recupera y entrega hasta ``cfg.budget_tokens``. ``filters=None`` los parsea de la pregunta."""
    check_pieces(cfg, index)
    latency, t0 = {}, time.perf_counter()

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        latency[name] = round((now - since) * 1000, 1)
        return now

    allowed, filter_info, t = None, {}, t0
    if cfg.date_filter or cfg.region_filter:
        allowed, filter_info = filter_mask(filters or parse_filters(query), cfg, index.metadata)
        t = lap("filter", t)

    channels, max_scores = {}, {}
    if cfg.dense:
        matrix = index.prefixed_matrix if cfg.prefix else index.raw_matrix
        channels["dense"], max_scores["dense"] = dense_search(query, matrix, index, allowed, cfg.n_candidates)
        t = lap("dense", t)
    if cfg.lexical:
        bm25 = index.bm25_prefixed if cfg.prefix else index.bm25_raw
        found = bm25.search(query, k=cfg.n_candidates, allowed=allowed)
        channels["bm25"], max_scores["bm25"] = [u for u, _ in found], (found[0][1] if found else None)
        t = lap("bm25", t)
    if cfg.dense and cfg.lexical:
        channels["fused"] = rrf([channels["dense"], channels["bm25"]], cfg.rrf_k)
    if cfg.dense and cfg.lexical:
        ranked = channels["fused"]
    elif cfg.dense:
        ranked = channels["dense"]
    else:
        ranked = channels["bm25"]

    reranked = []
    if cfg.rerank:
        reranked = ranked[:cfg.rerank_top]
        ranked, max_scores["rerank"] = rerank(query, ranked, cfg, index)
        channels["reranked"] = ranked
        t = lap("rerank", t)
    if allowed is not None:
        assert all(allowed[index.position[u]] for u in ranked), "volvió un hijo que el filtro había sacado"

    if cfg.parent:
        ranked = lift_to_parents(ranked, index)
        units = [index.parents[u] for u in ranked]
    else:
        units = [index.by_id[u] for u in ranked]
    delivered = deliver(units, cfg.budget_tokens)
    lap("deliver", t)
    latency["total"] = round((time.perf_counter() - t0) * 1000, 1)
    return Retrieval(ranked, delivered, channels, reranked, filter_info, max_scores, latency)
