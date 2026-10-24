from datetime import date

import pytest
from lxml import etree

import corpus
from cleaning import Cleaner
from doc_schema import Document, DocMetadata, Section

DOC = "BOE-A-2000-1"
META = {
    "titulo": "Ley 1/2000, de 1 de enero, de prueba.", "fecha_actualizacion": "20261005T000000Z",
    "url_html_consolidada": f"https://www.boe.es/buscar/act.php?id={DOC}",
    "ambito": {"texto": "Estatal"}, "departamento": {"texto": "Jefatura del Estado"},
    "rango": {"texto": "Ley"}, "estado_consolidacion": {"texto": "Finalizado"},
    "vigencia_agotada": "N", "estatus_derogacion": "N", "estatus_anulacion": "N",
    "fecha_disposicion": "20000101", "fecha_publicacion": "20000101", "fecha_vigencia": "20000102",
}
ANALYSIS = {"materias": [{"codigo": "305", "texto": "Arrendamientos urbanos"}],
            "notas": [], "anteriores": [], "posteriores": []}

# Una norma de juguete con los casos raros de la foto real.
XML = """<response><data><texto>
<bloque id="pr" tipo="preambulo">
  <version id_norma="BOE-A-2000-1" fecha_publicacion="20000101" fecha_vigencia="20000102">
    <p class="subseccion">JUAN CARLOS I</p>
    <p class="subseccion">REY DE ESPAÑA</p>
    <p class="parrafo">A todos los que la presente vieren y entendieren.</p>
    <p class="parrafo">Esta ley regula el alquiler.</p>
  </version>
</bloque>
<bloque id="ti" tipo="encabezado" titulo="TÍTULO I">
  <version id_norma="BOE-A-2000-1" fecha_publicacion="20000101" fecha_vigencia="20000102">
    <p class="titulo_num">TÍTULO I</p>
    <p class="titulo_tit">Disposiciones generales</p>
  </version>
</bloque>
<bloque id="a1" tipo="precepto" titulo="Artículo 1">
  <version id_norma="BOE-A-2000-1" fecha_publicacion="20000101" fecha_vigencia="20000102">
    <p class="articulo">Artículo 1. Plazo.</p>
    <p class="parrafo">El plazo es de tres años.</p>
  </version>
  <version id_norma="BOE-A-2020-9" fecha_publicacion="20200301" fecha_vigencia="20200305">
    <p class="articulo">Artículo 1. Plazo.</p>
    <p class="parrafo">El plazo es de siete años.</p>
  </version>
  <version id_norma="BOE-A-2010-5" fecha_publicacion="20100101" fecha_vigencia="20100102">
    <p class="articulo">Artículo 1. Plazo.</p>
    <p class="parrafo">El plazo es de cinco años.</p>
    <blockquote><p class="nota_pie">Se modifica por la Ley 5/2010. Ref. BOE-A-2010-5</p></blockquote>
  </version>
  <version id_norma="BOE-A-2026-7" fecha_publicacion="20261007" fecha_vigencia="">
    <p class="articulo">Artículo 1. Plazo.</p>
    <p class="parrafo">El plazo es de diez años.</p>
  </version>
</bloque>
<bloque id="a2" tipo="precepto" titulo="Artículo 2">
  <version id_norma="BOE-A-2000-1" fecha_publicacion="20000101" fecha_vigencia="20000102">
    <p class="articulo">Artículo 2. Modificación de otra ley.</p>
    <p class="parrafo">El artículo 9 queda redactado así:</p>
    <blockquote class="sangrado"><p class="parrafo">«9. El plazo mínimo es de cinco años.»</p></blockquote>
    <table class="tabla"><tbody>
      <tr><td><p class="cabeza_tabla">Zona</p></td><td><p class="cabeza_tabla">Precio</p></td></tr>
      <tr><td><p class="cuerpo_tabla_izq">A</p></td><td><p class="cuerpo_tabla_der">1.000</p></td></tr>
    </tbody></table>
    <p class="imagen">
      <img src="/datos/imagenes/x.png"/>
    </p>
    <p class="parrafo">Una superficie de 90 m<sup>2</sup> según el artículo 5.<sup>o</sup></p>
    <p class="cita_con_pleca">(*) Téngase en cuenta que esta norma fue derogada.</p>
  </version>
</bloque>
<bloque id="fi" tipo="firma">
  <version id_norma="BOE-A-2000-1" fecha_publicacion="20000101" fecha_vigencia="20000102">
    <p class="firma_rey">JUAN CARLOS R.</p>
  </version>
</bloque>
</texto></data></response>"""


def start_of(id_norma):
    # Hace de los metadatos de la norma modificadora (en el build real salen de la foto).
    return {"BOE-A-2026-7": date(2026, 11, 15)}.get(id_norma)


def build(xml=XML, meta=META, cleaner=None):
    return corpus.document_from_source(DOC, meta, ANALYSIS, xml, cleaner or Cleaner(), start_of=start_of)


def section(doc, section_id):
    return next(s for s in doc.sections if s.section_id == section_id)


def test_one_section_per_block_version_in_date_order():
    doc = build()
    assert [s.section_id for s in doc.sections] == [
        f"{DOC}#pr@20000102", f"{DOC}#ti@20000102",
        f"{DOC}#a1@20000102", f"{DOC}#a1@20100102", f"{DOC}#a1@20200305", f"{DOC}#a1@20261115",
        f"{DOC}#a2@20000102",
    ]
    assert [s.kind for s in doc.sections][:3] == ["preamble", "heading", "article"]


def test_offsets_point_to_each_section_text():
    doc = build()
    assert doc.section_text(section(doc, f"{DOC}#a1@20100102")) == "Artículo 1. Plazo.\nEl plazo es de cinco años."
    assert doc.text.count("\n\n") == len(doc.sections) - 1


def test_validity_intervals_and_the_version_in_force_is_not_the_last_one():
    doc = build()
    v2010, v2020, v2026 = (section(doc, f"{DOC}#a1@{d}") for d in ("20100102", "20200305", "20261115"))
    assert v2010.valid_to == date(2020, 3, 5)
    assert v2020.valid_to == date(2026, 11, 15)
    assert v2020.in_force(corpus.SNAPSHOT) and not v2026.in_force(corpus.SNAPSHOT)
    assert "reordered" in v2010.version_flags  # el XML traía 2020 antes que 2010


def test_missing_date_comes_from_the_modifying_norm_then_from_publication():
    doc = build()
    v2026 = section(doc, f"{DOC}#a1@20261115")
    assert v2026.valid_from_source == "norma_modificadora"
    doc = corpus.document_from_source(DOC, META, ANALYSIS, XML, Cleaner(), start_of=lambda _: None)
    assert section(doc, f"{DOC}#a1@20261007").valid_from_source == "publicacion"


def test_notes_leave_the_text_and_quoted_text_stays():
    doc = build()
    v2010 = section(doc, f"{DOC}#a1@20100102")
    assert v2010.notes == ["Se modifica por la Ley 5/2010. Ref. BOE-A-2010-5"]
    assert "Se modifica por la Ley 5/2010" not in doc.text
    assert "«9. El plazo mínimo es de cinco años.»" in doc.text
    a2 = section(doc, f"{DOC}#a2@20000102")
    assert a2.notes == ["(*) Téngase en cuenta que esta norma fue derogada."]


def test_table_image_and_superscripts():
    text = build().text
    assert "Zona: A; Precio: 1.000" in text
    assert corpus.IMAGE in text
    assert "90 m2 según el artículo 5.º" in text


def test_signature_and_promulgation_formula_are_dropped_and_counted():
    cleaner = Cleaner()
    doc = build(cleaner=cleaner)
    assert "JUAN CARLOS" not in doc.text and "REY DE ESPAÑA" not in doc.text
    assert doc.section_text(section(doc, f"{DOC}#pr@20000102")) == "Esta ley regula el alquiler."
    assert cleaner.docs["signature"] == {DOC} and cleaner.docs["promulgation"] == {DOC}


def test_headings_give_path_and_level():
    doc = build()
    heading = section(doc, f"{DOC}#ti@20000102")
    article = section(doc, f"{DOC}#a1@20000102")
    assert (heading.title, heading.level) == ("TÍTULO I. Disposiciones generales", 1)
    assert article.path == ["TÍTULO I. Disposiciones generales"] and article.level == 2
    assert article.title == "Artículo 1. Plazo."


def test_same_day_versions_get_distinct_ids_and_the_first_never_rules():
    xml = """<texto><bloque id="a1" tipo="precepto">
      <version id_norma="X" fecha_publicacion="20000101" fecha_vigencia="20000102"><p>Uno.</p></version>
      <version id_norma="X" fecha_publicacion="20000101" fecha_vigencia="20000102"><p>Dos.</p></version>
    </bloque></texto>"""
    first, second = build(xml).sections
    assert (first.section_id, second.section_id) == (f"{DOC}#a1@20000102", f"{DOC}#a1@20000102.2")
    assert "same_day" in first.version_flags and not first.in_force(date(2000, 1, 2))


def test_undated_versions_name_the_norm_that_can_date_them():
    assert corpus.undated_modifying_norms(XML) == {"BOE-A-2026-7"}


def test_unknown_block_type_raises():
    with pytest.raises(ValueError, match="tipo de bloque"):
        build("<texto><bloque id='x' tipo='raro'><version fecha_vigencia='20000102'><p>a</p></version></bloque></texto>")


def test_table_with_colspan_rowspan_and_two_header_rows():
    table = etree.fromstring("""<table><tbody>
      <tr><td rowspan="2"><p class="cabeza_tabla">Ingresos</p></td><td colspan="2"><p class="cabeza_tabla">Años</p></td></tr>
      <tr><td><p class="cabeza_tabla">Año 1</p></td><td><p class="cabeza_tabla">Año 2</p></td></tr>
      <tr><td><p>Bajo</p></td><td><p>10</p></td><td><p>20</p></td></tr>
    </tbody></table>""")
    assert corpus.table_lines(table) == ["Ingresos: Bajo; Años / Año 1: 10; Años / Año 2: 20"]


def test_line_breaks_and_paragraphs_inside_a_cell_separate_words():
    cell = etree.fromstring("<td><p>Número de personas<br/>que convivan</p><p>en la vivienda</p></td>")
    assert corpus.flat_text(cell) == "Número de personas que convivan en la vivienda"
    assert corpus.flat_text(etree.fromstring("<p>100 m<sup>2</sup> de la <em>Ley</em></p>")) == "100 m2 de la Ley"


def test_a_header_over_two_columns_keeps_their_values_together():
    table = etree.fromstring("""<table>
      <tr><td colspan="2"><p class="cabeza_tabla">Disposición</p></td><td><p class="cabeza_tabla">Objeto</p></td></tr>
      <tr><td>D.</td><td>8-7-1922</td><td>Reglamento de Casas Baratas.</td></tr>
    </table>""")
    assert corpus.table_lines(table) == ["Disposición: D. 8-7-1922; Objeto: Reglamento de Casas Baratas."]


def test_table_without_header_keeps_repeated_values():
    table = etree.fromstring("<table><tr><td>D.</td><td>10</td><td>10</td></tr></table>")
    assert corpus.table_lines(table) == ["D. | 10 | 10"]


def test_status_mapping_and_regions():
    assert corpus.map_status({"vigencia_agotada": "S"}) == "superseded"
    assert corpus.map_status({"vigencia_agotada": "N"}) == "current"
    with pytest.raises(ValueError):
        corpus.map_status({})  # un valor inesperado es un error, no un "unknown" silencioso
    valencia = [corpus.region({"ambito": {"texto": "Autonómico"}, "departamento": {"texto": d}})
                for d in ("Comunitat Valenciana", "Comunidad Valenciana")]
    assert valencia == ["Comunitat Valenciana", "Comunitat Valenciana"]
    with pytest.raises(ValueError):
        corpus.region({"ambito": {"texto": "Autonómico"}, "departamento": {"texto": "Atlántida"}})


def test_derogation_closes_open_versions():
    xml = """<texto>
      <bloque id="a1" tipo="precepto"><version fecha_vigencia="20000102"><p>Uno.</p></version></bloque>
      <bloque id="a2" tipo="precepto">
        <version fecha_vigencia="20000102"><p>Dos.</p></version>
        <version fecha_vigencia="20150101"><p>(Derogado)</p></version>
      </bloque>
      <bloque id="a3" tipo="precepto"><version fecha_vigencia="20160101"><p>Tres, transitorio.</p></version></bloque>
    </texto>"""
    meta = META | {"vigencia_agotada": "S", "estatus_derogacion": "S", "fecha_derogacion": "20150101"}
    doc = build(xml, meta=meta)
    a1, a2_old, a2_derogated, a3 = doc.sections
    assert doc.metadata.status == "superseded"
    assert a1.valid_to == date(2015, 1, 1) and a2_old.valid_to == date(2015, 1, 1)
    assert a2_derogated.valid_to == date(2015, 1, 1)  # intervalo vacío: después del fin no rige nada
    assert not a2_derogated.in_force(date(2015, 1, 1)) and "starts_at_norm_end" in a2_derogated.version_flags
    assert a3.valid_to is None and "after_norm_end" in a3.version_flags


def test_block_that_expired_before_its_last_version_started():
    xml = """<texto><bloque id="a1" tipo="precepto" fecha_caducidad="20230915">
      <version fecha_vigencia="20200101"><p>Uno.</p></version>
      <version fecha_vigencia="20240101"><p>Nunca rigió.</p></version>
    </bloque></texto>"""
    old, never = build(xml).sections
    assert old.valid_to == date(2023, 9, 15)  # la caducidad del bloque corta también a la versión vieja
    assert never.valid_to == never.valid_from and "expired_before_start" in never.version_flags


def test_final_provisions_do_not_hang_from_the_last_chapter():
    xml = """<texto>
      <bloque id="ti" tipo="encabezado"><version fecha_vigencia="20000102">
        <p class="titulo_num">TÍTULO V</p><p class="titulo_tit">Régimen sancionador</p></version></bloque>
      <bloque id="a1" tipo="precepto" titulo="Artículo 1"><version fecha_vigencia="20000102"><p>Uno.</p></version></bloque>
      <bloque id="da" tipo="precepto" titulo="Disposición adicional primera">
        <version fecha_vigencia="20000102"><p>Adicional.</p></version></bloque>
    </texto>"""
    _, article, additional = build(xml).sections
    assert article.path == ["TÍTULO V. Régimen sancionador"]
    assert additional.path == [] and additional.level == 1


def test_an_old_version_keeps_the_heading_title_of_its_time():
    xml = """<texto>
      <bloque id="tv" tipo="encabezado">
        <version fecha_vigencia="19950101"><p class="titulo_num">TÍTULO V</p><p class="titulo_tit">Procesos</p></version>
        <version fecha_vigencia="20261001"><p class="titulo_num">TÍTULO V</p><p class="titulo_tit">Sanciones</p></version>
      </bloque>
      <bloque id="a38" tipo="precepto">
        <version fecha_vigencia="19950101"><p>Competencia.</p></version>
        <version fecha_vigencia="20261001"><p>Infracciones.</p></version>
      </bloque>
    </texto>"""
    sections = build(xml).sections
    assert [s.path for s in sections if s.block_id == "a38"] == [["TÍTULO V. Procesos"], ["TÍTULO V. Sanciones"]]


def test_a_missing_modifying_norm_is_an_error_unless_the_api_does_not_have_it(monkeypatch):
    def not_in_snapshot(boe_id, offline=False):
        raise FileNotFoundError(boe_id)

    monkeypatch.setattr(corpus.boe, "metadatos", not_in_snapshot)
    assert corpus.modifying_norm_start(next(iter(corpus.NOT_IN_API))) is None
    with pytest.raises(FileNotFoundError):
        corpus.modifying_norm_start("BOE-A-2099-1")  # nunca se re-fecha en silencio por la publicación


def test_whole_norm_derogations_with_or_without_effective_date():
    def analysis(texto):
        return {"posteriores": [{"id_norma": "X", "relacion": "SE DEROGA", "texto": texto}]}

    assert corpus.derogated_by(analysis(", por Ley 1/2000")) == "X"
    assert corpus.derogated_by(analysis(", con efectos desde el 1 de enero de 2016, por Ley 1/2000")) == "X"
    assert corpus.derogated_by(analysis("los arts. 38 a 40, por la Ley 1/2000")) is None  # parcial


def test_regional_promulgation_formulas_are_dropped_too():
    cleaner = Cleaner()
    paragraphs = [(None, "EL PRESIDENTE DEL PRINCIPADO DE ASTURIAS"),
                  (None, "Sea notorio que la Junta General del Principado de Asturias ha aprobado, y yo promulgo:"),
                  (None, "El Presidente de la Junta de Andalucía a todos los que la presente vieren, sabed:"),
                  (None, "PREÁMBULO"), (None, "El presidente de la comunidad aprobará el plan.")]
    assert corpus.drop_promulgation(paragraphs, DOC, cleaner) == paragraphs[3:]


def test_a_rowspan_in_the_body_repeats_the_value_in_each_row():
    table = etree.fromstring("""<table>
      <tr><th>Zona</th><th>Tipo</th></tr>
      <tr><td rowspan="2">A</td><td>1</td></tr>
      <tr><td>2</td></tr>
    </table>""")
    assert corpus.table_lines(table) == ["Zona: A; Tipo: 1", "Zona: A; Tipo: 2"]


def test_near_duplicates_are_symmetric_sorted_and_labeled():
    body = " ".join(f"palabra{i}" for i in range(60))  # 56 shingles distintos: alcanza el mínimo de 40
    docs = [_tiny_doc("R-2025", 2025), _tiny_doc("R-2024", 2024)]
    for doc in docs:
        doc.text = body
        doc.sections[0].end = len(body)
    docs[1].metadata.status = "superseded"
    corpus.annotate_near_duplicates(docs)
    near_2025, near_2024 = (doc.metadata.near_duplicates for doc in docs)
    assert near_2025 == [{"doc_id": "R-2024", "jaccard": 1.0, "containment": 1.0, "relation": "version"}]
    assert near_2024[0]["doc_id"] == "R-2025"
    assert docs[0].sections[0].twins == ["R-2024#a1@20240401"]


def _tiny_doc(doc_id: str, year: int) -> Document:
    text = f"Resolución {year}."
    return Document(
        doc_id=doc_id, title=f"Resolución de {year}, sobre tipos de interés efectivos anuales vigentes.",
        text=text, metadata=DocMetadata(status="current", fecha_disposicion=f"{year}-03-01", near_duplicates=[],
                                        fecha_vigencia=None, fecha_publicacion=f"{year}-04-01"),
        sections=[Section(section_id=f"{doc_id}#a1@{year}0401", title="Primero", level=1, start=0,
                          end=len(text), block_id="a1", kind="article", valid_from=date(year, 4, 1))])


def test_periodic_series_keeps_only_the_latest_current():
    docs = [_tiny_doc("R-2024", 2024), _tiny_doc("R-2026", 2026), _tiny_doc("R-2025", 2025)]
    corpus.apply_periodic_series(docs)
    by_id = {d.doc_id: d for d in docs}
    assert [by_id[i].metadata.status for i in ("R-2024", "R-2025", "R-2026")] == ["superseded", "superseded", "current"]
    assert by_id["R-2024"].metadata.superseded_by == "R-2025"
    assert by_id["R-2024"].sections[0].valid_to == date(2025, 4, 1)
