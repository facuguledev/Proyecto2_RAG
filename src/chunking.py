"""Las unidades del índice (Etapa 3): qué secciones entran, padres, hijos, prefijo y cabecera.

- **Padre**: lo que se le puede entregar al generador cuando gana un hijo. Es la sección (un artículo en
  una versión) si entra en ``PARENT_MAX`` tokens del generador; si no, grupos de párrafos consecutivos.
- **Hijo**: lo que se indexa y se busca. Párrafos consecutivos de un padre, hasta ``CHILD_MAX`` tokens
  del reranker **con el prefijo incluido**. Un párrafo más largo se parte en ventanas con solape.

El corte es el mismo con y sin prefijo: el presupuesto del hijo reserva el lugar del prefijo aunque una
fila no lo use. Así la fila "+ prefijo" cambia el texto que se busca y no el corte (una variable).

Toda unidad es un tramo ``[start, end)`` de ``Document.text``: el texto sale de los offsets.
"""
import re
from dataclasses import dataclass, replace
from datetime import date
from typing import NamedTuple

from doc_schema import Document, Section
from evalset import norm_name

# Ningún gold es un encabezado ni una nota inicial: los encabezados ya viven en las migas de pan del
# prefijo, y las notas iniciales son "Incluye las correcciones... Ref. BOE-A-...".
INDEXED_KINDS = ("article", "preamble", "annex", "dispositive")
CHILD_MAX = 350  # tokens del reranker, prefijo incluido: con la consulta, el par entra en sus 512
CHILD_OVERLAP = 50  # tokens: el párrafo corto que se repite, o el solape entre ventanas
PARENT_MAX = 1200  # tokens del generador (§7.1: 1.000-1.500)
PREFIX_MAX = 120  # tokens del reranker
TEMPLATE = "prefix-v1 · header-v1"  # cambia si cambia el texto del prefijo o de la cabecera: va al manifiesto

MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre")
UNTIL_YEAR = re.compile(r"^.*?\b(?:1[89]|20)\d{2}\b")  # "Orden de 20 de mayo de 1969 por la que..." -> hasta el año


@dataclass(frozen=True)
class Unit:
    """Un hijo o un padre. ``parent_id`` es el propio id en un padre."""

    unit_id: str  # "<section_id>~c<n>" (hijo) o "<section_id>~p<n>" (padre)
    section_id: str
    doc_id: str
    parent_id: str
    start: int  # offsets en Document.text
    end: int
    text: str
    prefix: str  # el de su sección; solo lo usan las filas con prefix=True
    header: str  # la cabecera del pasaje entregado: igual en todas las filas
    n_rerank: int = 0  # tokens del reranker de prefijo + "\n" + texto
    n_gen: int = 0  # tokens del generador del pasaje entregado (cabecera + "\n" + texto)

    @property
    def is_child(self) -> bool:
        return self.unit_id != self.parent_id


def section_of(unit_id: str) -> str:
    """La sección de una unidad: 'BOE-A-1994-26003#a9@20190124~c2' -> 'BOE-A-1994-26003#a9@20190124'.

    Comparar el gold contra ids sin pelar daría un acierto 0 sin ningún error: por eso hay un solo helper.
    """
    return unit_id.split("~", 1)[0]


def searchable_text(unit: Unit, prefixed: bool) -> str:
    """El texto que leen los canales (denso, BM25, reranker)."""
    return f"{unit.prefix}\n{unit.text}" if prefixed else unit.text


def passage_text(unit: Unit) -> str:
    """El pasaje que se le entrega al generador: cabecera + texto crudo, en todas las filas."""
    return f"{unit.header}\n{unit.text}"


# --- Qué entra -----------------------------------------------------------------------------------

def never_ruled(section: Section) -> bool:
    """Una versión con intervalo vacío (same_day, expired_before_start, starts_at_norm_end) no rigió nunca."""
    return section.valid_to is not None and section.valid_from is not None and section.valid_from >= section.valid_to


def indexed_sections(docs: list[Document]) -> list[tuple[Document, Section]]:
    return [(doc, section) for doc in docs for section in doc.sections
            if section.kind in INDEXED_KINDS and not never_ruled(section)]


# --- Prefijo y cabecera --------------------------------------------------------------------------

def norm_label(doc: Document) -> str:
    """'Ley 29/1994', 'Ley 5/2018 (Illes Balears)', o el título hasta el año si no hay número oficial."""
    name = norm_name(doc)
    if name:
        return name
    match = UNTIL_YEAR.match(doc.title)
    return match.group() if match else doc.title


def month_year(day: date) -> str:
    return f"{MONTHS[day.month - 1]} de {day.year}"


def validity(section: Section) -> str:
    """Mes y año, sin el día: "hasta 20 de mayo" daría el término "20", que choca con "artículo 20"."""
    text = f"vigente desde {month_year(section.valid_from)}"
    return text + (f" hasta {month_year(section.valid_to)}" if section.valid_to else "")


def full_prefix(doc: Document, section: Section) -> str:
    """El prefijo sin recortar, de lo más a lo menos importante: el recorte se come el final.

    El título de una norma llega a 569 caracteres: primero, se comería la sección y la vigencia.
    """
    parts = [norm_label(doc), section.title, validity(section)]
    if section.path:
        parts.append(section.path[-1])
    parts.append(doc.title)
    return " | ".join(parts)


def header(doc: Document, section: Section) -> str:
    title = section.title if len(section.title) <= 150 else section.title[:150] + "…"
    return f"[{norm_label(doc)} · {title} · {validity(section)}]"


# --- Corte ---------------------------------------------------------------------------------------

def paragraphs(doc: Document, section: Section) -> list[tuple[int, int]]:
    """Los párrafos de una sección como tramos de Document.text (la Etapa 1 deja uno por línea)."""
    spans, start = [], section.start
    for line in doc.section_text(section).split("\n"):
        if line.strip():
            spans.append((start, start + len(line)))
        start += len(line) + 1
    return spans


def pack(sizes: list[int], budget: int, carry_max: int = 0) -> list[list[int]]:
    """Agrupa piezas consecutivas mientras entren en ``budget`` (un token más por cada unión).

    Con ``carry_max``, el grupo siguiente empieza repitiendo la última pieza del anterior si mide a lo
    sumo eso: el solape. Una pieza más grande que el presupuesto queda sola (el que llama lo verifica).
    """
    groups, current, used = [], [], 0
    for i, size in enumerate(sizes):
        if current and used + 1 + size > budget:
            groups.append(current)
            last = current[-1]
            if carry_max and len(current) > 1 and sizes[last] <= carry_max and sizes[last] + 1 + size <= budget:
                current, used = [last], sizes[last]
            else:
                current, used = [], 0
        used = used + 1 + size if current else size
        current.append(i)
    if current:
        groups.append(current)
    return groups


def windows(offsets: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Ventanas de ``size`` tokens que se solapan en ``overlap``, como tramos de caracteres."""
    spans, step = [], size - overlap
    for lo in range(0, len(offsets), step):
        hi = min(lo + size, len(offsets))
        spans.append((offsets[lo][0], offsets[hi - 1][1]))
        if hi == len(offsets):
            break  # la última ventana llegó al final del párrafo
    return spans


def encode(tokenizer, texts: list[str]):
    """``tokenizers.Tokenizer.encode_batch`` sin los tokens especiales (un test pasa uno de juguete)."""
    return tokenizer.encode_batch(texts, add_special_tokens=False) if texts else []


def count(tokenizer, texts: list[str]) -> list[int]:
    return [len(encoding.ids) for encoding in encode(tokenizer, texts)]


def cut_prefixes(prefixes: list[str], rerank_tok) -> list[str]:
    """Recorta cada prefijo a ``PREFIX_MAX`` tokens del reranker, por el final.

    El corte puede caer en medio de una palabra, y el texto cortado puede volver a tokenizarse con un token
    más (pasa en 4 de 22.014). No importa: el presupuesto del hijo descuenta la cuenta real del prefijo.
    """
    cut = []
    for text, encoding in zip(prefixes, encode(rerank_tok, prefixes)):
        cut.append(text if len(encoding.ids) <= PREFIX_MAX else text[:encoding.offsets[PREFIX_MAX - 1][1]])
    return cut


class Piece(NamedTuple):
    """Un tramo de Document.text que el empaquetado de hijos trata como indivisible."""

    start: int
    end: int
    size: int  # tokens del reranker


def child_pieces(spans: list[tuple[int, int]], texts: list[str], budget: int, rerank_tok) -> list[Piece]:
    """Los párrafos que entran en el presupuesto del hijo, y las ventanas de los que no."""
    pieces = []
    for (start, end), encoding in zip(spans, encode(rerank_tok, texts)):
        if len(encoding.ids) <= budget:
            pieces.append(Piece(start, end, len(encoding.ids)))
        else:
            pieces += [Piece(start + a, start + b, budget) for a, b in windows(encoding.offsets, budget, CHILD_OVERLAP)]
    return pieces


def build_units(docs: list[Document], rerank_tok, gen_tok) -> tuple[list[Unit], list[Unit]]:
    """Los padres y los hijos de todas las secciones indexadas, en el orden del corpus.

    ``rerank_tok`` y ``gen_tok`` son ``tokenizers.Tokenizer``: el del reranker (los hijos se dimensionan
    en sus tokens) y el del generador (los padres y los pasajes entregados).
    """
    sections = indexed_sections(docs)
    prefixes = cut_prefixes([full_prefix(doc, section) for doc, section in sections], rerank_tok)
    prefix_sizes = count(rerank_tok, prefixes)

    parents, children = [], []
    for (doc, section), prefix, prefix_size in zip(sections, prefixes, prefix_sizes):
        spans = paragraphs(doc, section)
        texts = [doc.text[a:b] for a, b in spans]
        common = {"section_id": section.section_id, "doc_id": doc.doc_id, "text": "", "prefix": prefix,
                  "header": header(doc, section)}
        body_budget = CHILD_MAX - prefix_size - 1  # el "\n" entre el prefijo y el texto
        before = len(children)  # los hijos se numeran dentro de su sección
        for p, group in enumerate(pack(count(gen_tok, texts), PARENT_MAX), start=1):
            parent_id = f"{section.section_id}~p{p}"
            parents.append(Unit(unit_id=parent_id, parent_id=parent_id, start=spans[group[0]][0],
                                end=spans[group[-1]][1], **common))
            pieces = child_pieces([spans[i] for i in group], [texts[i] for i in group], body_budget, rerank_tok)
            for piece_group in pack([piece.size for piece in pieces], body_budget, carry_max=CHILD_OVERLAP):
                first, last = pieces[piece_group[0]], pieces[piece_group[-1]]
                children.append(Unit(unit_id=f"{section.section_id}~c{len(children) - before + 1}", parent_id=parent_id,
                                     start=first.start, end=last.end, **common))

    by_doc = {doc.doc_id: doc for doc in docs}
    return with_counts(parents, by_doc, rerank_tok, gen_tok), with_counts(children, by_doc, rerank_tok, gen_tok)


def with_counts(units: list[Unit], by_doc: dict[str, Document], rerank_tok, gen_tok) -> list[Unit]:
    """Le pone a cada unidad su texto y sus cuentas exactas (las del empaquetado son aproximadas)."""
    texts = [by_doc[unit.doc_id].text[unit.start:unit.end] for unit in units]
    rerank = count(rerank_tok, [f"{unit.prefix}\n{text}" for unit, text in zip(units, texts)])
    gen = count(gen_tok, [f"{unit.header}\n{text}" for unit, text in zip(units, texts)])
    return [replace(unit, text=text, n_rerank=r, n_gen=g) for unit, text, r, g in zip(units, texts, rerank, gen)]
