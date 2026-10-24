"""Contratos de datos del proyecto: Document, Section y la validación de ``documents.jsonl``.

Las etapas siguientes (eval set, índice, prompts, memoria) leen estos contratos, así que acá se
falla fuerte: un campo mal escrito o un offset roto es un error, no algo que se guarda en silencio.
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from artifact_io import read_jsonl

# Cada lista de valores permitidos se escribe una sola vez; las tuplas son para recorrerlas.
Status = Literal["current", "superseded", "deprecated", "draft", "unknown"]
SectionKind = Literal["article", "preamble", "heading", "annex", "initial_note", "dispositive"]
DateSource = Literal["vigencia", "norma_modificadora", "publicacion"]
STATUSES, SECTION_KINDS, DATE_SOURCES = get_args(Status), get_args(SectionKind), get_args(DateSource)


# --- Modelos -------------------------------------------------------------------------------

class DocMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")  # el enunciado pide aceptar claves extra (p. ej. source_status)

    status: Status = "unknown"
    version: str | None = None
    source_url: str | None = None
    lang: str = "es"


class Section(BaseModel):
    """Un bloque del BOE (artículo, preámbulo, encabezado...) en una de sus versiones fechadas."""

    model_config = ConfigDict(extra="forbid")  # un campo mal escrito tiene que fallar, no guardarse en silencio

    section_id: str  # "<doc_id>#<block_id>@<YYYYMMDD>", p. ej. "BOE-A-1994-26003#a9@20190306"
    title: str
    level: int = Field(ge=0)
    start: int = Field(ge=0)  # offsets en Document.text (el texto ya normalizado)
    end: int
    block_id: str  # id del bloque en el BOE ("a9"); agrupa las versiones de un mismo bloque
    kind: SectionKind
    valid_from: date | None = None  # primer día en que la versión rige
    valid_to: date | None = None  # EXCLUSIVO: primer día en que ya no rige; None = sigue vigente
    valid_from_source: DateSource | None = None  # de dónde salió valid_from
    modified_by: str | None = None  # id de la norma que introdujo esta versión
    # pydantic copia los defaults mutables en cada instancia: las listas no se comparten.
    path: list[str] = []  # migas de pan de encabezados, p. ej. ["TÍTULO II. ...", "CAPÍTULO I. ..."]
    notes: list[str] = []  # notas de modificación o editoriales, fuera del texto
    twins: list[str] = []  # section_id de secciones casi idénticas en otros documentos
    version_flags: list[str] = []  # anomalías de la fuente, p. ej. "reordered", "same_day"

    def in_force(self, on: date) -> bool:
        """True si la versión rige en ``on``: valid_from <= on < valid_to (None = sin límite)."""
        if self.valid_from is not None and on < self.valid_from:
            return False
        if self.valid_to is not None and on >= self.valid_to:
            return False
        return True


class Document(BaseModel):
    """Una norma: su texto ya normalizado, las secciones que lo recortan y sus metadatos."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    text: str
    sections: list[Section]
    metadata: DocMetadata

    def section_text(self, section: Section) -> str:
        return self.text[section.start:section.end]


def documents_hash(docs: Iterable[Document]) -> str:
    """sha256 del corpus; no depende del orden en que vengan los documentos.

    Va en el manifiesto de cada índice: así un índice puede decir con qué corpus se construyó.
    """
    ordered = sorted(docs, key=lambda doc: doc.doc_id)
    digest = hashlib.sha256()
    for doc in ordered:
        row = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        digest.update((row + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_documents(path: Path) -> list[Document]:
    """Lee ``documents.jsonl``. Una línea inválida levanta ``ValidationError``: no se saltea."""
    return [Document.model_validate(row) for row in read_jsonl(path)]


# --- Validación de documents.jsonl ---------------------------------------------------------

@dataclass
class Check:
    name: str
    failures: list[str]


@dataclass
class ValidationReport:
    """Los checks de una validación. Lo usan ``validate_documents`` y ``evalset.validate_eval``."""

    name: str  # qué se validó, p. ej. la ruta de documents.jsonl
    summary: str  # qué había adentro, p. ej. "177 documentos, 8687 secciones"
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(not check.failures for check in self.checks)

    def show(self) -> "ValidationReport":
        """Imprime un renglón por check y devuelve ``self``, para encadenar con ``raise_if_failed``."""
        print(f"{self.name}: {self.summary}")
        for check in self.checks:
            if check.failures:
                first_three = "; ".join(check.failures[:3])
                print(f"  FALLA  {check.name} ({len(check.failures)}): {first_three}")
            else:
                print(f"  ok     {check.name}")
        return self

    def raise_if_failed(self) -> None:
        failed = [f"{check.name} ({len(check.failures)})" for check in self.checks if check.failures]
        if failed:
            raise ValueError(f"{self.name} no pasa la validación: " + ", ".join(failed))


# Caracteres que no se ven: guion blando, espacios de ancho cero, word joiner, BOM, y todo carácter
# de control menos el salto de línea (el tab y el \r también cuentan).
INVISIBLE_CHARS = re.compile("[\u00ad\u200b\u200c\u200d\u2060\ufeff\x00-\x09\x0b-\x1f\x7f]")
MARKUP = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(\s[^<>]*)?/?>")  # algo que parece una etiqueta XML/HTML
# Las notas editoriales del BOE citan su fuente así ("... Ref. BOE-A-2010-5"). Fuera de la nota
# inicial, en el texto de una sección, eso es una nota que se filtró al articulado.
LEAKED_NOTE = re.compile(r"Ref\. ?[A-Z]+-[A-Za-z]+-\d{4}-\d+")


# Cada check recibe los documentos ya parseados y devuelve un texto por fallo,
# que nombra el documento o la sección donde falló.

def check_unique_doc_id(docs: list[Document]) -> list[str]:
    counts = Counter(doc.doc_id for doc in docs)
    return [f"{doc_id}: aparece {n} veces" for doc_id, n in counts.items() if n > 1]


def check_unique_section_id(docs: list[Document]) -> list[str]:
    counts = Counter(section.section_id for doc in docs for section in doc.sections)
    return [f"{section_id}: aparece {n} veces" for section_id, n in counts.items() if n > 1]


def check_section_id_prefix(docs: list[Document]) -> list[str]:
    failures = []
    for doc in docs:
        for section in doc.sections:
            if not section.section_id.startswith(doc.doc_id + "#"):
                failures.append(f"{section.section_id}: no empieza con '{doc.doc_id}#'")
    return failures


def check_offsets(docs: list[Document]) -> list[str]:
    """0 <= start < end <= len(text), y las secciones en orden y sin pisarse.

    ``start >= 0`` ya lo exige el modelo (``Field(ge=0)``): un start negativo falla en ``parse``.
    """
    failures = []
    for doc in docs:
        previous_end = 0
        for section in doc.sections:
            if section.start >= section.end:
                failures.append(f"{section.section_id}: start {section.start} >= end {section.end}")
            if section.end > len(doc.text):
                failures.append(f"{section.section_id}: end {section.end} > len(text) {len(doc.text)}")
            if section.start < previous_end:
                failures.append(
                    f"{section.section_id}: start {section.start} < fin de la sección anterior {previous_end}")
            previous_end = section.end
    return failures


def check_empty_section(docs: list[Document]) -> list[str]:
    failures = []
    for doc in docs:
        for section in doc.sections:
            if doc.section_text(section).strip() == "":
                failures.append(f"{section.section_id}: el texto de la sección está vacío")
    return failures


def check_nfc(docs: list[Document]) -> list[str]:
    failures = []
    for doc in docs:
        if not unicodedata.is_normalized("NFC", doc.text):
            failures.append(f"{doc.doc_id}: text no está en NFC")
        if not unicodedata.is_normalized("NFC", doc.title):
            failures.append(f"{doc.doc_id}: title no está en NFC")
    return failures


def check_invisible_chars(docs: list[Document]) -> list[str]:
    failures = []
    for doc in docs:
        found = sorted(set(INVISIBLE_CHARS.findall(doc.text)))
        if found:
            codes = ", ".join(f"U+{ord(char):04X}" for char in found)
            failures.append(f"{doc.doc_id}: caracteres invisibles o de control ({codes})")
    return failures


def check_markup(docs: list[Document]) -> list[str]:
    failures = []
    for doc in docs:
        tag = MARKUP.search(doc.text)
        if tag:
            failures.append(f"{doc.doc_id}: etiqueta {tag.group()!r} en la posición {tag.start()}")
    return failures


def check_leaked_notes(docs: list[Document]) -> list[str]:
    failures = []
    for doc in docs:
        for section in doc.sections:
            found = LEAKED_NOTE.search(doc.section_text(section))
            if found and section.kind != "initial_note":
                failures.append(f"{section.section_id}: {found.group()!r} (una nota editorial en el texto)")
    return failures


def check_validity_interval(docs: list[Document]) -> list[str]:
    failures = []
    for doc in docs:
        for section in doc.sections:
            if (section.valid_from is not None and section.valid_to is not None
                    and section.valid_from > section.valid_to):
                failures.append(
                    f"{section.section_id}: valid_from {section.valid_from} > valid_to {section.valid_to}")
    return failures


def check_versions_do_not_overlap(docs: list[Document]) -> list[str]:
    """En cada fecha rige a lo sumo una versión de cada bloque: el gold "a una fecha" depende de eso."""
    failures = []
    for doc in docs:
        by_block: dict[str, list[Section]] = {}
        for section in doc.sections:
            by_block.setdefault(section.block_id, []).append(section)
        for versions in by_block.values():
            versions.sort(key=lambda s: s.valid_from or date.min)
            for earlier, later in zip(versions, versions[1:]):
                earlier_end = earlier.valid_to or date.max  # una versión abierta rige para siempre
                if later.valid_from is not None and earlier_end > later.valid_from:
                    failures.append(f"{earlier.section_id} (hasta {earlier.valid_to}) se pisa con "
                                    f"{later.section_id} (desde {later.valid_from})")
    return failures


# Los checks que corren sobre los documentos ya parseados, en este orden (``parse`` va aparte: es
# el que los arma). Para agregar uno: escribir ``check_xxx`` arriba y sumarlo acá.
DOCUMENT_CHECKS = {
    "unique_doc_id": check_unique_doc_id,
    "unique_section_id": check_unique_section_id,
    "section_id_prefix": check_section_id_prefix,
    "offsets": check_offsets,
    "empty_section": check_empty_section,
    "nfc": check_nfc,
    "invisible_chars": check_invisible_chars,
    "markup": check_markup,
    "leaked_notes": check_leaked_notes,
    "validity_interval": check_validity_interval,
    "versions_do_not_overlap": check_versions_do_not_overlap,
}


def _short_error(error: ValidationError) -> str:
    """El primer error de pydantic en una línea, p. ej. ``metadata.status: Input should be 'current', ...``."""
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"])
    return f"{field}: {first['msg']}" if field else first["msg"]


def _parse_lines(path: Path) -> tuple[list[Document], list[str]]:
    """Lee el archivo línea por línea: devuelve los Document que parsean y un fallo por cada línea que no."""
    docs = []
    failures = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                docs.append(Document.model_validate(json.loads(line)))
            except json.JSONDecodeError as error:
                failures.append(f"línea {number}: JSON inválido ({error})")
            except ValidationError as error:
                failures.append(f"línea {number}: {_short_error(error)}")
    return docs, failures


def validate_documents(path: Path) -> ValidationReport:
    """Corre todos los checks sobre ``documents.jsonl``. No levanta: junta los fallos en el reporte."""
    docs, parse_failures = _parse_lines(path)
    checks = [Check("parse", parse_failures)]
    for name, check in DOCUMENT_CHECKS.items():
        checks.append(Check(name, check(docs)))
    n_sections = sum(len(doc.sections) for doc in docs)  # de los documentos que parsearon
    return ValidationReport(str(path), f"{len(docs)} documentos, {n_sections} secciones", checks)
