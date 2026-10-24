"""De la foto cruda del BOE a doc_schema.Document (Etapa 1).

Un documento por norma y una sección por cada versión de cada bloque (artículo, preámbulo, título...).
Las decisiones están en decision_log.md (2026-10-17). En corto:
- el texto es el articulado: las notas de modificación y las editoriales van a `notes`, fuera del texto;
- el texto que cita una ley modificadora (<blockquote class="sangrado">, «...») sí es texto;
- cada versión rige en [valid_from, valid_to), y la vigente no es necesariamente la última;
- las tablas se linealizan ("Encabezado: valor; ...") y las imágenes dejan un marcador;
- la firma y la fórmula de promulgación (el pie y el encabezado repetidos del corpus) se descartan.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations, count
from typing import Callable, NamedTuple

from lxml import etree

import boe
import dedup
from cleaning import Cleaner
from doc_schema import Document, DocMetadata, Section

SNAPSHOT = date(2026, 10, 24)  # la foto fijada (decision_log.md, 2026-10-24; la del 2026-10-16 se perdió)
IMAGE = "[imagen omitida]"

# Tipo de bloque del BOE -> Section.kind. Un "encabezado" con algo más que títulos es un anexo.
KINDS = {"precepto": "article", "preambulo": "preamble", "encabezado": "heading",
         "nota_inicial": "initial_note", "parte_dispositiva": "dispositive"}
DEFAULT_TITLES = {"article": "Artículo", "preamble": "Preámbulo", "heading": "Título", "annex": "Anexo",
                  "initial_note": "Nota inicial", "dispositive": "Parte dispositiva"}
# Clase del <p> sin su sufijo ("capitulo_num" -> "capitulo") -> nivel de título.
HEADING_LEVELS = {"libro": 1, "titulo": 1, "anexo": 1, "capitulo": 2, "seccion": 3, "subseccion": 4}

# La fórmula de promulgación con que abren los preámbulos: el "encabezado repetido" de este corpus.
# Se buscan solo en los primeros párrafos del preámbulo (ver drop_promulgation).
PROMULGATION = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"^(juan carlos i|felipe vi|rey de españa)$",
    r"a todos los que la presente vieren",
    r"^sabed: que las cortes generales han aprobado",
    r"^(el|la) president[ea] (de|del) [^.:;]+$",  # "EL PRESIDENTE DE LA JUNTA DE EXTREMADURA", solo en su línea
    r"^(sea notorio|conózcase|se hace saber|hago saber)\b.*\b(ha|han) aprobado",
    r"^en nombre del rey .* promulgo",
    r"^que el parlamento de .* ha aprobado",
)]

# Normas modificadoras que la API no tiene consolidadas (respondió 404 el 2026-10-17): las versiones
# sin fecha de vigencia que introdujeron se fechan con su publicación. Cualquier otra que falte en la
# foto es un error, no una fecha más.
NOT_IN_API = {"BOE-A-2017-4971", "BOE-A-2021-2835"}

# Las disposiciones adicionales, transitorias, derogatorias y finales no cuelgan del último capítulo.
FINAL_PART = re.compile(r"disposici[oó]n(es)? (adicional|transitoria|derogatoria|final)", re.IGNORECASE)

# Departamento del BOE -> región. La misma comunidad figura con dos nombres (Valenciana).
REGIONS = {
    "Comunidad Autónoma de Andalucía": "Andalucía",
    "Comunidad Autónoma de Aragón": "Aragón",
    "Comunidad Autónoma de Canarias": "Canarias",
    "Comunidad Autónoma de Cantabria": "Cantabria",
    "Comunidad de Castilla y León": "Castilla y León",
    "Comunidad Autónoma de Cataluña": "Cataluña",
    "Comunidad de Madrid": "Comunidad de Madrid",
    "Comunidad Foral de Navarra": "Comunidad Foral de Navarra",
    "Comunitat Valenciana": "Comunitat Valenciana",
    "Comunidad Valenciana": "Comunitat Valenciana",
    "Comunidad Autónoma de Extremadura": "Extremadura",
    "Comunidad Autónoma de Galicia": "Galicia",
    "Comunidad Autónoma de las Illes Balears": "Illes Balears",
    "Comunidad Autónoma de La Rioja": "La Rioja",
    "Comunidad Autónoma del País Vasco": "País Vasco",
    "Comunidad Autónoma del Principado de Asturias": "Principado de Asturias",
    "Comunidad Autónoma de la Región de Murcia": "Región de Murcia",
}

# Estados que la fuente da mal, verificados a mano contra el texto de cada norma (decision_log.md).
CORRECTIONS = {
    "BOE-A-2005-21319": {  # Ley 11/2005 de las Illes Balears
        "status": "superseded", "end": date(2018, 1, 1), "superseded_by": "BOE-A-2018-806",
        "reason": "la fuente la da vigente, pero su nota inicial dice «Norma derogada, con efectos de 1 "
                  "de enero de 2018» (Ley balear 12/2017)"},
    "BOE-A-2012-14115": {  # Real Decreto-ley 27/2012
        "status": "superseded", "end": date(2013, 5, 15), "superseded_by": "BOE-A-2013-5073",
        "reason": "agotada sin fecha de derogación: su nota inicial dice que la sustituyó tácitamente "
                  "la Ley 1/2013, vigente desde el 15/05/2013"},
}

# Normas periódicas que se reemplazan sin derogarse: la fuente da vigentes a todas.
PERIODIC_SERIES = ["tipos de interés efectivos anuales"]  # 13 resoluciones anuales, de 2014 a 2026


# --- Lectura del XML ---------------------------------------------------------------------------

def parse_date(value: str | None) -> date | None:
    """'19950101' -> date(1995, 1, 1). La API escribe AAAAMMDD; vacío -> None."""
    return date(int(value[:4]), int(value[4:6]), int(value[6:8])) if value else None


def iso_date(value: str | None) -> str | None:
    """'19950101' -> '1995-01-01' (los metadatos extra se guardan como texto ISO)."""
    found = parse_date(value)
    return found.isoformat() if found else None


def tidy(text: str) -> str:
    """NFC y espacios colapsados, para lo que no es el texto (títulos, notas, miga de pan)."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def flat_text(el: etree._Element) -> str:
    """Todo el texto de un elemento, sin markup y con los espacios colapsados."""
    return tidy(_flat(el))


# Elementos que separan palabras. Unir el texto sin espacios convierte "personas<br/>que" en
# "personasque" (pasaba en los encabezados de las tablas).
WORD_BREAKS = {"p", "br", "td", "th", "tr", "li", "div"}


def _flat(el: etree._Element) -> str:
    parts = [el.text or ""]
    for child in el:
        gap = " " if child.tag in WORD_BREAKS else ""
        parts += [gap, _flat(child), gap, child.tail or ""]
    return "".join(parts)


@dataclass
class Extracted:
    """Lo que sale de una <version>: párrafos (clase del <p>, texto) y notas, ya separados."""
    paragraphs: list[tuple[str | None, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    removed: Counter = field(default_factory=Counter)  # regla -> caracteres eliminados (negativo = agregados)

    def count(self, rule: str, chars: int) -> None:
        self.removed[rule] += chars  # con chars=0 igual queda anotado que la regla tocó esta norma

    def note(self, text: str, rule: str) -> None:
        self.notes.append(text)
        self.count(rule, len(text))


def extract_version(version: etree._Element) -> Extracted:
    out = Extracted()
    for el in version:
        add_element(el, out)
    return out


def add_element(el: etree._Element, out: Extracted) -> None:
    if el.tag == "blockquote" and el.get("class") == "sangrado":
        # Texto citado por una ley modificadora («4. Son infracciones graves...»): es el contenido
        # de esa ley, no una nota. La auditoría de la Etapa 0 lo tiraba junto con las notas.
        for child in el:
            add_element(child, out)
    elif el.tag == "blockquote":
        out.note(flat_text(el), "notes_out")  # "Se modifica el apartado 2 por...", "Téngase en cuenta..."
    elif el.tag == "table":
        lines = table_lines(el)
        out.count("tables", len(flat_text(el)) - len("\n".join(lines)))
        out.paragraphs += [("tabla", line) for line in lines]
    elif el.get("class") == "cita_con_pleca" and flat_text(el).startswith("(*)"):
        out.note(flat_text(el), "editorial_inline")  # "(*) Téngase en cuenta que esta norma..."
    elif text := inline_text(el, out):
        out.paragraphs.append((el.get("class"), text))


def inline_text(el: etree._Element, out: Extracted) -> str:
    """El texto de un párrafo sin el markup: <a>, <strong>, <em> se van y sus palabras quedan."""
    # Los saltos de línea dentro de un <p> son la indentación del archivo XML, no del texto.
    return re.sub(r"\s*\n\s*", " ", _inline(el, out)).strip()


def _inline(el: etree._Element, out: Extracted) -> str:
    parts = [el.text or ""]
    for child in el:
        child_text = flat_text(child)
        if child.tag == "img":
            parts.append(f" {IMAGE} ")
            out.count("images", -len(IMAGE))
        elif child.tag == "sup" and child_text.startswith("(*)"):
            out.note(child_text, "editorial_inline")
        elif child.tag == "sup" and child_text in ("o", "a"):
            parts.append("º" if child_text == "o" else "ª")  # "5.<sup>o</sup>" -> "5.º", como el resto del corpus
            out.count("sup_ordinals", 0)
        elif child.tag == "br":
            parts.append(" ")
        else:
            parts.append(_inline(child, out))
        parts.append(child.tail or "")
    return "".join(parts)


# --- Tablas ------------------------------------------------------------------------------------

class Cell(NamedTuple):
    id: int  # una celda del XML repetida por colspan/rowspan conserva su id
    text: str
    header: bool


EMPTY = Cell(-1, "", False)  # una casilla que ninguna celda cubre


def table_grid(table: etree._Element) -> list[list[Cell]]:
    """Las celdas como grilla rectangular: una celda con colspan o rowspan ocupa cada casilla que cubre."""
    rows = table.xpath("./tr | ./thead/tr | ./tbody/tr | ./tfoot/tr")  # sin las filas de tablas anidadas
    slots: dict[tuple[int, int], Cell] = {}  # (fila, columna) -> celda
    ids = count()
    for r, tr in enumerate(rows):
        c = 0
        for td in (td for td in tr if td.tag in ("td", "th")):
            while (r, c) in slots:  # casilla ya ocupada por una celda de más arriba (rowspan)
                c += 1
            header = td.tag == "th" or any(p.get("class") == "cabeza_tabla" for p in td.iter("p"))
            cell = Cell(next(ids), flat_text(td), header)
            colspan = max(int(td.get("colspan") or 1), 1)
            for dr in range(max(int(td.get("rowspan") or 1), 1)):
                for dc in range(colspan):
                    slots[(r + dr, c + dc)] = cell
            c += colspan
    width = max((col for _, col in slots), default=-1) + 1
    return [[slots.get((r, c), EMPTY) for c in range(width)] for r in range(len(rows))]


def distinct(cells: list[Cell]) -> list[Cell]:
    """Cada celda del XML una sola vez, aunque ocupe varias casillas seguidas."""
    return [cell for k, cell in enumerate(cells) if k == 0 or cell.id != cells[k - 1].id]


def table_lines(table: etree._Element) -> list[str]:
    """Linealiza una tabla: cada fila como "Encabezado: valor; Encabezado: valor".

    Así una fila se entiende sola aunque el chunker corte la tabla, y BM25 ve los encabezados.
    Una tabla sin fila de encabezado (o de puro formato) queda "celda | celda".
    """
    grid = [row for row in table_grid(table) if any(cell.text for cell in row)]
    n_header = 0
    while n_header < len(grid) and all(cell.header for cell in grid[n_header] if cell.text):
        n_header += 1
    if n_header in (0, len(grid)):
        return [" | ".join(cell.text for cell in distinct(row) if cell.text) for row in grid]

    columns = zip(*grid[:n_header])  # la grilla es rectangular: cada columna de las filas de encabezado
    headers = [" / ".join(cell.text for cell in distinct(list(column)) if cell.text) for column in columns]
    lines = []
    for row in grid[n_header:]:
        pairs: list[list[str]] = []  # [encabezado, valor, valor...]
        previous = None
        for header, cell in zip(headers, row):
            if cell.text and cell.id != previous:
                if pairs and pairs[-1][0] == header:
                    pairs[-1].append(cell.text)  # un encabezado que abarca varias columnas (colspan)
                else:
                    pairs.append([header, cell.text])
            previous = cell.id
        lines.append("; ".join(f"{header}: {' '.join(values)}" if header else " ".join(values)
                               for header, *values in pairs))
    return lines


# --- Versiones y vigencia ----------------------------------------------------------------------

@dataclass
class DatedVersion:
    element: etree._Element
    valid_from: date
    valid_from_source: str
    valid_to: date | None = None  # exclusiva: el primer día en que ya no rige
    flags: list[str] = field(default_factory=list)

    def in_force(self, on: date) -> bool:
        return self.valid_from <= on and (self.valid_to is None or on < self.valid_to)


def undated_modifying_norms(xml: str) -> set[str]:
    """Las normas que introdujeron versiones sin fecha de vigencia: sus metadatos tienen que estar en la foto."""
    root = etree.fromstring(xml.encode("utf-8"))
    return {v.get("id_norma") for v in root.iter("version") if not v.get("fecha_vigencia")}


def modifying_norm_start(id_norma: str | None) -> date | None:
    """Desde cuándo rige la norma que introdujo una versión. Si sus metadatos no están en la foto, falla."""
    if id_norma in NOT_IN_API:
        return None
    return parse_date(boe.metadatos(id_norma, offline=True).get("fecha_vigencia"))


def version_start(version: etree._Element, start_of: Callable[[str | None], date | None]) -> tuple[date, str]:
    """Desde cuándo rige una versión, y de dónde salió esa fecha."""
    if found := parse_date(version.get("fecha_vigencia")):
        return found, "vigencia"
    # 18 versiones vienen sin fecha de vigencia. La de la norma que las introdujo es la mejor
    # aproximación: así el art. 10 de la LAU según el RDL 28/2026 rige desde el 15/11/2026, después
    # de la foto, y no desde que se publicó.
    if found := start_of(version.get("id_norma")):
        return found, "norma_modificadora"
    if found := parse_date(version.get("fecha_publicacion")):
        return found, "publicacion"
    raise ValueError(f"versión sin ninguna fecha (id_norma={version.get('id_norma')})")


def dated_versions(block: etree._Element, start_of: Callable[[str | None], date | None]) -> list[DatedVersion]:
    """Las versiones de un bloque en orden de vigencia, cada una con su intervalo [valid_from, valid_to)."""
    versions = [DatedVersion(v, *version_start(v, start_of)) for v in block.findall("version")]
    ordered = sorted(versions, key=lambda v: v.valid_from)  # sort estable: el mismo día conserva el orden del XML
    if any(a.valid_from > b.valid_from for a, b in zip(versions, versions[1:])):
        for v in ordered:
            v.flags.append("reordered")  # el XML no las traía en orden de vigencia
    for cur, nxt in zip(ordered, ordered[1:]):
        cur.valid_to = nxt.valid_from
        if cur.valid_to == cur.valid_from:
            cur.flags.append("same_day")  # otra versión del mismo día la reemplaza: nunca rige
    # fecha_caducidad es la fecha "a partir de la cual el bloque deja de mostrarse" (documentación de
    # la API): ninguna versión del bloque rige desde ese día.
    expiry = parse_date(block.get("fecha_caducidad"))
    if expiry:
        for v in ordered:
            if v.valid_from >= expiry:
                v.valid_to = v.valid_from
                v.flags.append("expired_before_start")  # el bloque caducó antes de que empezara: nunca rigió
            elif v.valid_to is None or v.valid_to > expiry:
                v.valid_to = expiry
    return ordered


def close_at(doc: Document, end: date) -> None:
    """Cuando la norma deja de regir en `end`, cierra ahí las versiones que seguían abiertas."""
    for s in doc.sections:
        if s.valid_to is not None:
            continue
        if s.valid_from < end:
            s.valid_to = end
        elif s.valid_from == end:
            # El "(Derogado)", o un RDL derogado el mismo día en que entraba en vigor: nunca rige.
            s.valid_to = end
            s.version_flags.append("starts_at_norm_end")
        else:
            # El BOE fecha algunas versiones después de la derogación (régimen transitorio): quedan abiertas.
            s.version_flags.append("after_norm_end")


# --- Títulos y miga de pan ---------------------------------------------------------------------

def heading_level(css_class: str | None) -> int | None:
    """Nivel de un párrafo de título ("titulo_num" -> 1, "capitulo_tit" -> 2); None si no es título."""
    return HEADING_LEVELS.get((css_class or "").split("_")[0])


def heading_of(paragraphs: list[tuple[str | None, str]]) -> tuple[int, str] | None:
    """[("titulo_num", "TÍTULO I"), ("titulo_tit", "Ámbito")] -> (1, "TÍTULO I. Ámbito"); None si no hay título."""
    found = [(heading_level(css), tidy(text)) for css, text in paragraphs if heading_level(css)]
    if not found:
        return None
    return min(level for level, _ in found), ". ".join(text.rstrip(".") for _, text in found)


def has_content(paragraphs: list[tuple[str | None, str]]) -> bool:
    """¿Trae algo más que títulos? Entonces el encabezado es un anexo, con contenido propio."""
    return any(heading_level(css) is None for css, _ in paragraphs)


@dataclass
class OpenHeading:
    """Un encabezado (título, capítulo...) que contiene a los bloques que lo siguen."""
    level: int
    titles: list[tuple[DatedVersion, str]]  # el título de cada versión del encabezado

    def title_on(self, day: date) -> str | None:
        """El título que tenía ese día, o None si ese día el encabezado no existía."""
        return next((title for dated, title in self.titles if dated.in_force(day)), None)


def open_heading(versions: list[tuple[DatedVersion, Extracted]]) -> OpenHeading | None:
    """Un bloque encabezado como nivel de la miga de pan; None si no trae títulos."""
    titled = [(dated, heading) for dated, content in versions if (heading := heading_of(content.paragraphs))]
    if not titled:
        return None
    return OpenHeading(level=titled[-1][1][0], titles=[(dated, title) for dated, (_, title) in titled])


def path_on(parents: list[OpenHeading], day: date) -> list[str]:
    """La miga de pan de una versión: los títulos que la contenían el día en que empezó a regir.

    (Los títulos también cambian: el título V de la LAU pasó de "Procesos arrendaticios" a "Régimen
    sancionador" en 2026, y el art. 38 de 1995 no estaba en ese régimen).
    """
    return [title for heading in parents if (title := heading.title_on(day))]


def section_title(kind: str, paragraphs: list[tuple[str | None, str]], text: str, block: etree._Element) -> str:
    if kind == "article" and paragraphs and paragraphs[0][0] == "articulo":
        return text.split("\n", 1)[0]  # "Artículo 9. Plazo mínimo."
    if kind in ("heading", "annex") and (heading := heading_of(paragraphs)):
        return heading[1]
    return tidy(block.get("titulo") or "") or DEFAULT_TITLES[kind]


def section_level(kind: str, paragraphs: list[tuple[str | None, str]], parents: list[OpenHeading]) -> int:
    if kind in ("heading", "annex") and (heading := heading_of(paragraphs)):
        return heading[0]
    return parents[-1].level + 1 if parents else 1


def drop_promulgation(paragraphs: list[tuple[str | None, str]], boe_id: str, cleaner: Cleaner) -> list:
    """Saca la fórmula de promulgación del comienzo del preámbulo ("JUAN CARLOS I / REY DE ESPAÑA / ...")."""
    n = 0
    while n < len(paragraphs) and any(p.search(tidy(paragraphs[n][1])) for p in PROMULGATION):
        n += 1
    if n:
        cleaner.record("promulgation", boe_id, sum(len(text) for _, text in paragraphs[:n]))
    return paragraphs[n:]


def unique_id(base: str, used: set[str]) -> str:
    """El id de la sección. Dos versiones del mismo bloque y del mismo día: "...@20190306.2"."""
    section_id, n = base, 1
    while section_id in used:
        n += 1
        section_id = f"{base}.{n}"
    used.add(section_id)
    return section_id


# --- Estado y metadatos ------------------------------------------------------------------------

def map_status(meta: dict) -> str:
    """Estado de la fuente -> DocMetadata.status (el mapeo está en corpus_card.md)."""
    statuses = {"N": "current", "S": "superseded"}
    if meta.get("vigencia_agotada") not in statuses:
        raise ValueError(f"vigencia_agotada inesperada: {meta.get('vigencia_agotada')!r}")
    return statuses[meta["vigencia_agotada"]]


def region(meta: dict) -> str:
    if meta["ambito"]["texto"] == "Estatal":
        return "Estatal"
    departamento = meta["departamento"]["texto"]
    if departamento not in REGIONS:
        raise ValueError(f"departamento sin región asignada: {departamento!r}")
    return REGIONS[departamento]


# La referencia que deroga una norma ENTERA no nombra artículos: ", por Ley ..." o
# ", con efectos desde el 1 de enero de 2016, por Ley ...". La parcial dice "los arts. 38 a 40, por ...".
WHOLE_NORM_DEROGATION = re.compile(r",?\s*(con efectos[^,]*,\s*)?por\b", re.IGNORECASE)


def derogated_by(analysis: dict) -> str | None:
    """La norma que derogó a esta entera, según las referencias "SE DEROGA" del análisis."""
    for ref in analysis["posteriores"]:
        if "DEROGA" in (ref["relacion"] or "") and WHOLE_NORM_DEROGATION.match((ref["texto"] or "").strip()):
            return ref["id_norma"]
    return None


def doc_metadata(boe_id: str, meta: dict, analysis: dict, notes: list[str]) -> DocMetadata:
    """Los metadatos de la norma, con el estado ya mapeado (y corregido, si hace falta)."""
    fix = CORRECTIONS.get(boe_id, {})
    status = fix.get("status") or map_status(meta)
    return DocMetadata(
        status=status,
        version=meta["fecha_actualizacion"],  # la versión del texto consolidado que tiene la foto
        source_url=meta["url_html_consolidada"],
        lang="es",
        source_status={"vigencia_agotada": meta.get("vigencia_agotada"),
                       "estatus_derogacion": meta.get("estatus_derogacion"),
                       "estatus_anulacion": meta.get("estatus_anulacion"),
                       "estado_consolidacion": meta["estado_consolidacion"]["texto"]},
        status_reason=fix.get("reason"),
        superseded_by=(fix.get("superseded_by") or derogated_by(analysis)) if status == "superseded" else None,
        rango=meta["rango"]["texto"],
        numero_oficial=meta.get("numero_oficial"),
        ambito=meta["ambito"]["texto"],
        departamento=meta["departamento"]["texto"],
        region=region(meta),
        materias=[m["texto"] for m in analysis["materias"]],
        fecha_disposicion=iso_date(meta.get("fecha_disposicion")),
        fecha_publicacion=iso_date(meta.get("fecha_publicacion")),
        fecha_vigencia=iso_date(meta.get("fecha_vigencia")),
        fecha_derogacion=iso_date(meta.get("fecha_derogacion")),
        # En las autonómicas, fecha_publicacion es la del BOE; la del boletín regional está en esta nota.
        nota_publicacion=next((tidy(n) for n in analysis["notas"] if n and n.startswith("Publicada en")), None),
        url_eli=meta.get("url_eli"),
        near_duplicates=[],
        notes=notes,
        snapshot=SNAPSHOT.isoformat(),
    )


# --- Una norma, el corpus ----------------------------------------------------------------------

def document_from_source(boe_id: str, meta: dict, analysis: dict, xml: str, cleaner: Cleaner,
                         start_of: Callable[[str | None], date | None] = modifying_norm_start) -> Document:
    """Una norma -> Document. Recibe lo ya leído de la foto, para poder probarlo con un XML de juguete."""
    pieces: list[tuple[dict, str]] = []  # (campos de la sección, texto normalizado), en orden
    doc_notes: list[str] = []
    open_headings: list[OpenHeading] = []  # los encabezados abiertos hasta el bloque actual
    used_ids: set[str] = set()

    for block in etree.fromstring(xml.encode("utf-8")).iter("bloque"):
        tipo, block_id = block.get("tipo"), block.get("id")
        if tipo == "firma":
            # "Madrid, 24 de noviembre de 1994. JUAN CARLOS R. ...": el pie repetido de este corpus.
            cleaner.record("signature", boe_id, len(flat_text(block)))
            continue
        if tipo not in KINDS:
            raise ValueError(f"{boe_id}#{block_id}: tipo de bloque desconocido {tipo!r}")

        versions = []
        for dated in dated_versions(block, start_of):
            content = extract_version(dated.element)
            for rule, chars in content.removed.items():
                cleaner.record(rule, boe_id, chars)
            if tipo == "preambulo":
                content.paragraphs = drop_promulgation(content.paragraphs, boe_id, cleaner)
            versions.append((dated, content))

        # La miga de pan: los encabezados que contienen a este bloque.
        if tipo == "precepto" and FINAL_PART.match(tidy(block.get("titulo") or "")):
            open_headings.clear()  # las disposiciones de la parte final no cuelgan del último capítulo
        heading = open_heading(versions) if tipo == "encabezado" else None
        while heading and open_headings and open_headings[-1].level >= heading.level:
            open_headings.pop()  # un capítulo nuevo cierra el anterior (y lo que tenga adentro)
        parents = list(open_headings)
        if heading:
            open_headings.append(heading)

        for dated, content in versions:
            text = cleaner.normalize("\n".join(paragraph for _, paragraph in content.paragraphs), boe_id)
            if not text:  # una versión con solo notas (p. ej. "Información relacionada"): van a la norma
                doc_notes += content.notes
                continue
            kind = "annex" if tipo == "encabezado" and has_content(content.paragraphs) else KINDS[tipo]
            fields = dict(
                section_id=unique_id(f"{boe_id}#{block_id}@{dated.valid_from:%Y%m%d}", used_ids),
                title=section_title(kind, content.paragraphs, text, block),
                level=section_level(kind, content.paragraphs, parents),
                block_id=block_id, kind=kind,
                valid_from=dated.valid_from, valid_to=dated.valid_to, valid_from_source=dated.valid_from_source,
                modified_by=dated.element.get("id_norma"), path=path_on(parents, dated.valid_from),
                notes=content.notes, version_flags=dated.flags)
            pieces.append((fields, text))

    doc = Document(doc_id=boe_id, title=tidy(meta["titulo"]), text="\n\n".join(text for _, text in pieces),
                   sections=with_offsets(pieces), metadata=doc_metadata(boe_id, meta, analysis, doc_notes))
    end = CORRECTIONS.get(boe_id, {}).get("end") or parse_date(meta.get("fecha_derogacion"))
    if end:
        close_at(doc, end)
    return doc


def with_offsets(pieces: list[tuple[dict, str]]) -> list[Section]:
    """Las secciones con sus offsets en el texto final, donde van unidas por "\\n\\n"."""
    sections, start = [], 0
    for fields, text in pieces:
        sections.append(Section(**fields, start=start, end=start + len(text)))
        start += len(text) + 2
    return sections


def build_document(boe_id: str, cleaner: Cleaner) -> Document:
    """El documento de una norma, leyendo SOLO la foto: lo que no está en la caché es un error."""
    return document_from_source(boe_id, boe.metadatos(boe_id, offline=True), boe.analisis(boe_id, offline=True),
                                boe.texto(boe_id, offline=True), cleaner)


def build_corpus(boe_ids: list[str], cleaner: Cleaner) -> list[Document]:
    docs = [build_document(boe_id, cleaner) for boe_id in boe_ids]
    apply_periodic_series(docs)
    annotate_near_duplicates(docs)  # después de las series: la relación depende del estado
    return docs


def apply_periodic_series(docs: list[Document]) -> None:
    """Cada resolución periódica deja sin efecto a la anterior: solo la última queda vigente."""
    for phrase in PERIODIC_SERIES:
        series = sorted((d for d in docs if phrase in d.title), key=lambda d: d.metadata.fecha_disposicion)
        for older, newer in zip(series, series[1:]):
            older.metadata.status = "superseded"
            older.metadata.superseded_by = newer.doc_id
            older.metadata.status_reason = f"norma periódica ({phrase}): la reemplaza {newer.doc_id}"
            newer_start = newer.metadata.fecha_vigencia or newer.metadata.fecha_publicacion
            close_at(older, date.fromisoformat(newer_start))


def annotate_near_duplicates(docs: list[Document], min_jaccard: float = 0.3, min_containment: float = 0.8,
                             twin_jaccard: float = 0.7) -> None:
    """Marca los casi-duplicados (la política está en corpus_card.md).

    - Norma <-> norma: todos los pares (177 normas son 15.576), con Jaccard y contención exactos.
      Si una de las dos está reemplazada, es una *versión* de la otra; si no, quedan *relacionadas*.
    - Sección <-> sección de otra norma (Jaccard >= 0,7, con MinHash + LSH): Section.twins.
    """
    shingles = [dedup.shingles(d.text) for d in docs]
    for i, j in combinations(range(len(docs)), 2):
        jaccard = dedup.jaccard(shingles[i], shingles[j])
        containment = dedup.containment(shingles[i], shingles[j])
        if jaccard >= min_jaccard or containment >= min_containment:
            a, b = docs[i], docs[j]
            relation = "version" if "superseded" in (a.metadata.status, b.metadata.status) else "related"
            for doc, other in ((a, b), (b, a)):
                doc.metadata.near_duplicates.append({"doc_id": other.doc_id, "jaccard": round(jaccard, 3),
                                                     "containment": round(containment, 3), "relation": relation})

    sections = [(d, s) for d in docs for s in d.sections if s.kind != "heading"]
    sets = [dedup.shingles(d.section_text(s)) for d, s in sections]
    big = [k for k, s in enumerate(sets) if len(s) >= 40]  # con menos, fórmulas cortas ("(Derogado)") empatan
    for i, j, _ in dedup.similar_pairs([sets[k] for k in big], twin_jaccard):
        (doc_a, sec_a), (doc_b, sec_b) = sections[big[i]], sections[big[j]]
        if doc_a.doc_id != doc_b.doc_id:
            sec_a.twins.append(sec_b.section_id)
            sec_b.twins.append(sec_a.section_id)

    # Ordenadas, para que el hash del corpus no dependa del orden en que vinieron las normas.
    for doc in docs:
        doc.metadata.near_duplicates.sort(key=lambda near: near["doc_id"])
        for s in doc.sections:
            s.twins.sort()


def naive_text(xml: str) -> str:
    """Lo que daría un loader ingenuo: el texto de cada elemento de cada versión, todo junto y en orden."""
    lines = []
    for block in etree.fromstring(xml.encode("utf-8")).iter("bloque"):
        for version in block.findall("version"):
            lines += [text for el in version if (text := flat_text(el))]
    return "\n".join(lines)
