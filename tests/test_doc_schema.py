"""Contratos de datos: modelos, hash del corpus y validación de documents.jsonl."""
import json
from datetime import date

import pytest
from pydantic import ValidationError

from artifact_io import write_jsonl
from doc_schema import (DATE_SOURCES, SECTION_KINDS, STATUSES, DocMetadata, Document, Section,
                        documents_hash, load_documents, validate_documents)

ARTICLES = ("Artículo 1. Objeto.\nLa ley regula el alquiler.",
            "Artículo 2. Plazo.\nEl plazo es de cinco años.")


# --- Armado de datos de prueba -------------------------------------------------------------

def make_section(**changes) -> Section:
    """Sección válida mínima; ``changes`` pisa los campos que cada test quiere cambiar."""
    fields = {"section_id": "BOE-A-2000-1#a1@20000101", "title": "Artículo 1", "level": 1,
              "start": 0, "end": 10, "block_id": "a1", "kind": "article"}
    return Section(**{**fields, **changes})


def make_doc(doc_id: str = "BOE-A-2000-1", texts: tuple[str, ...] = ARTICLES) -> Document:
    """Documento válido: un artículo por texto, separados por una línea en blanco, con offsets exactos."""
    sections = []
    start = 0
    for number, text in enumerate(texts, start=1):
        sections.append(make_section(
            section_id=f"{doc_id}#a{number}@20000101", title=f"Artículo {number}", block_id=f"a{number}",
            start=start, end=start + len(text), valid_from=date(2000, 1, 1)))
        start += len(text) + 2  # el "\n\n" que separa un artículo del siguiente
    return Document(doc_id=doc_id, title="Ley de prueba", text="\n\n".join(texts),
                    sections=sections, metadata=DocMetadata())


def to_json(doc: Document) -> str:
    return json.dumps(doc.model_dump(mode="json"), ensure_ascii=False)


def validate(tmp_path, *docs: Document):
    """Escribe los documentos en un documents.jsonl y devuelve el reporte de validate_documents."""
    path = tmp_path / "documents.jsonl"
    write_jsonl(path, docs)
    return validate_documents(path)


def failures_by_check(report) -> dict[str, list[str]]:
    return {check.name: check.failures for check in report.checks}


def failed_checks(report) -> list[str]:
    return [name for name, failures in failures_by_check(report).items() if failures]


# --- Modelos -------------------------------------------------------------------------------

def test_metadata_rejects_a_status_outside_the_allowed_values():
    with pytest.raises(ValidationError):
        DocMetadata(status="in force")  # el valor crudo de la fuente: hay que mapearlo a uno de STATUSES


def test_metadata_keeps_extra_keys():
    metadata = DocMetadata(status="current", source_status={"vigencia_agotada": "N"})
    assert metadata.model_dump()["source_status"] == {"vigencia_agotada": "N"}


def test_metadata_defaults():
    metadata = DocMetadata()
    assert metadata.status == "unknown" and metadata.lang == "es"
    assert metadata.version is None and metadata.source_url is None


def test_section_and_document_reject_unknown_fields():
    with pytest.raises(ValidationError):
        make_section(unknown_field=1)
    with pytest.raises(ValidationError):
        Document(**make_doc().model_dump(), unknown_field=1)


@pytest.mark.parametrize("field", ["level", "start"])
def test_section_rejects_negative_level_and_start(field):
    with pytest.raises(ValidationError):
        make_section(**{field: -1})


def test_constant_tuples_list_the_allowed_values():
    assert STATUSES == ("current", "superseded", "deprecated", "draft", "unknown")  # los que pide el enunciado
    assert "article" in SECTION_KINDS and "publicacion" in DATE_SOURCES


def test_in_force_includes_valid_from_and_excludes_valid_to():
    section = make_section(valid_from=date(2019, 3, 6), valid_to=date(2020, 1, 1))
    assert not section.in_force(date(2019, 3, 5))
    assert section.in_force(date(2019, 3, 6))  # primer día
    assert section.in_force(date(2019, 12, 31))  # último día
    assert not section.in_force(date(2020, 1, 1))  # valid_to es exclusivo


def test_in_force_is_open_ended_without_valid_to():
    section = make_section(valid_from=date(2019, 3, 6))
    assert section.in_force(date(2999, 1, 1))
    assert not section.in_force(date(2019, 3, 5))


def test_in_force_without_dates_is_always_true():
    assert make_section().in_force(date(1900, 1, 1))
    without_start = make_section(valid_to=date(2020, 1, 1))  # sin valid_from: sin límite inferior
    assert without_start.in_force(date(1900, 1, 1))


def test_section_text_returns_the_slice_of_the_document_text():
    doc = make_doc()
    assert doc.section_text(doc.sections[1]) == ARTICLES[1]


# --- Hash y carga --------------------------------------------------------------------------

def test_documents_hash_ignores_input_order():
    a, b = make_doc("BOE-A-2000-1"), make_doc("BOE-A-2000-2")
    assert documents_hash([a, b]) == documents_hash([b, a])


def test_documents_hash_changes_when_one_character_changes():
    a, b = make_doc("BOE-A-2000-1"), make_doc("BOE-A-2000-2")
    before = documents_hash([a, b])
    b.text = b.text.replace("cinco", "cinca")
    assert documents_hash([a, b]) != before


def test_documents_hash_ignores_the_key_order_inside_metadata():
    a, b = make_doc(), make_doc()
    a.metadata = DocMetadata(source_status={"vigencia_agotada": "N", "estatus_derogacion": "N"})
    b.metadata = DocMetadata(source_status={"estatus_derogacion": "N", "vigencia_agotada": "N"})
    assert documents_hash([a]) == documents_hash([b])


def test_load_documents_round_trips_what_write_jsonl_wrote(tmp_path):
    docs = [make_doc("BOE-A-2000-1"), make_doc("BOE-A-2000-2")]
    docs[0].sections[0].valid_to = date(2005, 6, 30)  # las fechas se escriben como texto y vuelven como date
    docs[1].metadata = DocMetadata(status="current", source_status={"vigencia_agotada": "N"})
    path = tmp_path / "documents.jsonl"
    write_jsonl(path, docs)
    assert load_documents(path) == docs


def test_load_documents_raises_on_an_invalid_line(tmp_path):
    path = tmp_path / "documents.jsonl"
    write_jsonl(path, [{"doc_id": "BOE-A-2000-1"}])  # faltan casi todos los campos
    with pytest.raises(ValidationError):
        load_documents(path)


# --- validate_documents: un test por check -------------------------------------------------

def test_a_valid_file_passes_every_check(tmp_path):
    report = validate(tmp_path, make_doc("BOE-A-2000-1"), make_doc("BOE-A-2000-2"))
    assert report.ok  # los "\n" del texto no cuentan como caracteres de control
    assert report.summary == "2 documentos, 4 secciones"
    assert [check.name for check in report.checks] == [
        "parse", "unique_doc_id", "unique_section_id", "section_id_prefix", "offsets",
        "empty_section", "nfc", "invisible_chars", "markup", "leaked_notes", "validity_interval",
        "versions_do_not_overlap"]


def test_lines_that_do_not_parse_fail_parse_and_the_rest_is_still_checked(tmp_path):
    bad_status = ('{"doc_id": "BOE-A-2000-2", "title": "x", "text": "t", "sections": [], '
                  '"metadata": {"status": "in force"}}')
    not_json = '{"doc_id": '
    duplicated = make_doc("BOE-A-2000-3")
    duplicated.sections[1].section_id = duplicated.sections[0].section_id
    # la línea 4, en blanco, se ignora
    lines = [to_json(make_doc("BOE-A-2000-1")), bad_status, not_json, "", to_json(duplicated)]
    path = tmp_path / "documents.jsonl"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

    report = validate_documents(path)

    parse_failures = failures_by_check(report)["parse"]
    assert len(parse_failures) == 2
    assert parse_failures[0].startswith("línea 2:") and "metadata.status" in parse_failures[0]
    assert parse_failures[1].startswith("línea 3:")
    assert failed_checks(report) == ["parse", "unique_section_id"]  # la línea 4 sí se revisó
    assert report.summary.startswith("2 documentos")  # cuenta solo las líneas que parsearon


def test_duplicate_doc_id_fails(tmp_path):
    report = validate(tmp_path, make_doc("BOE-A-2000-1"), make_doc("BOE-A-2000-1"))
    assert failures_by_check(report)["unique_doc_id"] == ["BOE-A-2000-1: aparece 2 veces"]


def test_duplicate_section_id_fails(tmp_path):
    doc = make_doc()
    doc.sections[1].section_id = doc.sections[0].section_id
    report = validate(tmp_path, doc)
    assert failed_checks(report) == ["unique_section_id"]
    assert failures_by_check(report)["unique_section_id"] == ["BOE-A-2000-1#a1@20000101: aparece 2 veces"]


def test_section_id_must_start_with_the_doc_id_and_the_hash(tmp_path):
    doc = make_doc()
    doc.sections[0].section_id = "BOE-A-1999-9#a1@20000101"  # de otro documento
    doc.sections[1].section_id = "BOE-A-2000-12#a2@20000101"  # el "#" importa: empieza con "BOE-A-2000-1"
    report = validate(tmp_path, doc)
    assert failed_checks(report) == ["section_id_prefix"]
    failures = failures_by_check(report)["section_id_prefix"]
    assert len(failures) == 2
    assert "BOE-A-1999-9#a1@20000101" in failures[0] and "BOE-A-2000-12#a2@20000101" in failures[1]


def test_end_beyond_the_text_fails_offsets(tmp_path):
    doc = make_doc()
    doc.sections[1].end = len(doc.text) + 10
    report = validate(tmp_path, doc)
    assert failed_checks(report) == ["offsets"]
    assert failures_by_check(report)["offsets"] == [
        f"BOE-A-2000-1#a2@20000101: end {len(doc.text) + 10} > len(text) {len(doc.text)}"]


def test_start_not_before_end_fails_offsets(tmp_path):
    doc = make_doc()
    doc.sections[0].end = doc.sections[0].start  # sección de largo cero
    report = validate(tmp_path, doc)
    assert "offsets" in failed_checks(report)
    assert "BOE-A-2000-1#a1@20000101" in failures_by_check(report)["offsets"][0]


def test_overlapping_sections_fail_offsets(tmp_path):
    doc = make_doc()
    doc.sections[1].start = doc.sections[0].end - 5  # pisa los últimos 5 caracteres del artículo 1
    report = validate(tmp_path, doc)
    assert failed_checks(report) == ["offsets"]
    assert "BOE-A-2000-1#a2@20000101" in failures_by_check(report)["offsets"][0]


def test_touching_sections_are_valid(tmp_path):
    doc = make_doc()
    doc.sections[0].end = doc.sections[1].start  # sin hueco: la segunda empieza donde termina la primera
    assert validate(tmp_path, doc).ok


def test_unsorted_sections_fail_offsets(tmp_path):
    doc = make_doc()
    doc.sections.reverse()
    report = validate(tmp_path, doc)
    assert failed_checks(report) == ["offsets"]


def test_whitespace_only_section_fails(tmp_path):
    report = validate(tmp_path, make_doc(texts=("Artículo 1. Objeto.", "   ")))
    assert failed_checks(report) == ["empty_section"]
    assert "BOE-A-2000-1#a2@20000101" in failures_by_check(report)["empty_section"][0]


def test_non_nfc_text_fails(tmp_path):
    report = validate(tmp_path, make_doc(texts=("Artículo 1. información", ARTICLES[1])))
    assert failed_checks(report) == ["nfc"]


def test_non_nfc_title_fails(tmp_path):
    doc = make_doc()
    doc.title = "Información"
    assert failed_checks(validate(tmp_path, doc)) == ["nfc"]


@pytest.mark.parametrize("char", ["\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\t", "\r", "\x00", "\x7f"],
                         ids=lambda char: f"U+{ord(char):04X}")
def test_invisible_chars_in_the_text_fail(tmp_path, char):
    report = validate(tmp_path, make_doc(texts=(f"Artículo 1. Obje{char}to.", ARTICLES[1])))
    assert failed_checks(report) == ["invisible_chars"]
    assert f"U+{ord(char):04X}" in failures_by_check(report)["invisible_chars"][0]


def test_markup_in_the_text_fails(tmp_path):
    report = validate(tmp_path, make_doc(texts=("Artículo 1. <p>hola</p>", ARTICLES[1])))
    assert failed_checks(report) == ["markup"]
    assert "<p>" in failures_by_check(report)["markup"][0]


def test_two_versions_of_a_block_in_force_on_the_same_day_fail(tmp_path):
    doc = make_doc()
    first, second = doc.sections
    second.block_id = first.block_id  # dos versiones del mismo bloque
    first.valid_from, first.valid_to = date(2000, 1, 1), date(2010, 1, 1)
    second.valid_from = date(2010, 1, 1)
    assert validate(tmp_path, doc).ok  # una termina el día en que empieza la otra: no se pisan
    first.valid_to = None  # abierta: rige para siempre y se pisa con la de 2010
    assert failed_checks(validate(tmp_path, doc)) == ["versions_do_not_overlap"]


def test_an_editorial_note_in_an_article_fails_but_not_in_the_initial_note(tmp_path):
    leaked = "Artículo 1. Objeto.\nSe modifica por la Ley 5/2010. Ref. BOE-A-2010-5"
    report = validate(tmp_path, make_doc(texts=(leaked, ARTICLES[1])))
    assert failed_checks(report) == ["leaked_notes"]
    doc = make_doc(texts=(leaked, ARTICLES[1]))
    doc.sections[0].kind = "initial_note"  # "Norma derogada ... Ref. BOE-A-..." es el texto de la nota inicial
    assert validate(tmp_path, doc).ok


def test_valid_from_after_valid_to_fails_but_the_same_day_does_not(tmp_path):
    doc = make_doc()
    doc.sections[0].valid_from = date(2005, 6, 30)
    doc.sections[0].valid_to = date(2005, 6, 30)  # mismo día: es una anomalía de la fuente, no un error
    assert validate(tmp_path, doc).ok
    doc.sections[0].valid_to = date(2005, 6, 29)
    report = validate(tmp_path, doc)
    assert failed_checks(report) == ["validity_interval"]
    assert "BOE-A-2000-1#a1@20000101" in failures_by_check(report)["validity_interval"][0]


# --- ValidationReport ----------------------------------------------------------------------

def test_report_is_not_ok_when_any_check_fails(tmp_path):
    doc = make_doc()
    doc.sections[1].end = len(doc.text) + 10
    assert not validate(tmp_path, doc).ok


def test_raise_if_failed_names_the_failed_checks_and_their_counts(tmp_path):
    doc = make_doc()
    doc.sections[1].end = len(doc.text) + 1
    report = validate(tmp_path, doc)
    with pytest.raises(ValueError, match=r"offsets \(1\)"):
        report.raise_if_failed()


def test_raise_if_failed_does_nothing_when_everything_passes(tmp_path):
    assert validate(tmp_path, make_doc()).raise_if_failed() is None


def test_show_prints_one_line_per_check_and_returns_the_report(tmp_path, capsys):
    broken = make_doc("BOE-A-2000-1")
    broken.sections[1].end = len(broken.text) + 10
    report = validate(tmp_path, broken, make_doc("BOE-A-2000-2"))

    assert report.show() is report  # devuelve self: validate_documents(p).show().raise_if_failed()

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("2 documentos, 4 secciones")
    assert "  ok     parse" in lines
    end = len(broken.text) + 10
    assert f"  FALLA  offsets (1): BOE-A-2000-1#a2@20000101: end {end} > len(text) {len(broken.text)}" in lines
    assert len(lines) == 1 + 12  # el encabezado y un renglón por check


def test_show_lists_at_most_three_failures_per_check(tmp_path, capsys):
    docs = [make_doc(f"BOE-A-2000-{n}") for n in range(1, 5)]
    for doc in docs:
        doc.sections[1].end = len(doc.text) + 10
    validate(tmp_path, *docs).show()
    (row,) = [row for row in capsys.readouterr().out.splitlines() if "offsets" in row]
    assert row.startswith("  FALLA  offsets (4): ")
    assert "BOE-A-2000-3#a2" in row and "BOE-A-2000-4#a2" not in row
