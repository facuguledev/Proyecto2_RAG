# %% [markdown]
r"""
# Etapa 1 · De la fuente a los documentos

Esta notebook convierte la foto del BOE del **2026-10-24** (177 normas de vivienda y alquiler, `data/subset_v1.json`) en `artifacts/<perfil>/documents.jsonl`. La unidad es **un documento por norma** y **una sección por cada versión de cada bloque** (artículo, preámbulo, título...), cada una con su intervalo de vigencia. Es el contrato `doc_schema.Document` que leen las Etapas 2, 3 y 6: el gold apunta a sus `section_id`, los chunks siguen sus secciones y los filtros leen sus metadatos.

La lógica vive en `src/`: `corpus.py` extrae, `cleaning.py` normaliza y cuenta, `dedup.py` busca casi-duplicados y `doc_schema.py` valida. Esta notebook la corre y muestra los números. Las decisiones y su porqué están en `decision_log.md` (Etapa 1).

| Paso del enunciado | Sección |
|---|---|
| 1 · Fijar la fuente | §1 |
| 2 · Extraer (notas, tablas, imágenes) | §2 |
| 3 · Normalizar | §3 |
| 4 · Ids estables | §4 |
| 5 · Metadatos validados contra el cuerpo | §5 |
| 6 · Casi-duplicados y su política | §6 |
| 7 · `documents.jsonl` y su manifiesto | §7 |
| 8 · Qué cambió la limpieza | §8 |

**Perfiles.** Con `SMOKE_TEST=1` en el entorno se arman solo los 10 documentos de lectura (§8) en `artifacts/smoke/`. Sus números no significan nada.
"""

# %%
import environment  # el del curso, primero: guarda de JAX, caché de HF dentro del repo, UTF-8

import difflib
import json
import os
import platform
import re
import subprocess
import sys
import unicodedata
import warnings
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version

ROOT = environment.ROOT
sys.path.insert(0, str(ROOT / "src"))  # nuestros módulos: boe, corpus, doc_schema, artifact_io, ...
if environment.on_colab():
    # La foto cruda del BOE vive en Drive, al lado de artifacts/ (se sube una vez desde local).
    os.environ.setdefault("P2_RAW", "/content/drive/MyDrive/Proyecto2_RAG/raw")

warnings.filterwarnings("ignore", message="IProgress not found")
import pandas as pd
from lxml import etree
from tokenizers import Tokenizer

try:
    from IPython.display import display  # en la notebook
except ImportError:
    display = print  # como script (tools/run_stages.py)

import artifact_io
import boe
import corpus
import doc_schema
from cleaning import Cleaner

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
ART = environment.artifacts_dir(SMOKE_TEST)
SUBSET = json.loads((ROOT / "data" / "subset_v1.json").read_text(encoding="utf-8"))
# Los 10 documentos de la lectura de punta a punta (auditoría de la Etapa 0, §10). También son el perfil smoke.
READING = ["BOE-A-1994-26003", "BOE-A-2023-12203", "BOE-A-2026-20823", "BOE-A-2018-17293",
           "BOE-A-1960-10906", "BOE-A-2020-4208", "BOE-A-2005-21319", "BOE-A-2024-7593",
           "BOE-A-2018-9774", "BOE-A-2015-10727"]
IDS = READING if SMOKE_TEST else [norm["id"] for norm in SUBSET["norms"]]

TOK = Tokenizer.from_pretrained("Qwen/Qwen3-8B")  # los tokens se cuentan con el tokenizer del generador


def ntok(texts: list[str]) -> list[int]:
    return [len(e.ids) for e in TOK.encode_batch(texts, add_special_tokens=False)] if texts else []


pd.set_option("display.max_colwidth", 110)
pd.set_option("display.width", 220)
environment.summary()
print("perfil:", ART.name, "| normas:", len(IDS))

# %% [markdown]
r"""
## 1 · La fuente fijada

La versión fijada es la foto de la API del 2026-10-24. Cada respuesta es un archivo en `boe_cache/`, con su fecha de descarga y su sha256 en `index.jsonl`; la `fecha_actualizacion` de cada norma está en `data/subset_v1.json` (en git). El build corre **offline**: si algo no está en la foto es un error, y nunca se baja en silencio una versión más nueva.

Hay 18 versiones que vienen sin `fecha_vigencia`. La mejor fecha disponible es la de la norma que las introdujo (§4), así que los metadatos de esas normas tienen que estar en la foto. Hay dos excepciones, declaradas en `corpus.NOT_IN_API`: la API no las tiene consolidadas (respondió 404 el 2026-10-17). Cualquier otra que falte es un error, porque si no esas versiones se re-fecharían en silencio con su publicación.
"""

# %%
needed = set().union(*(corpus.undated_modifying_norms(boe.texto(boe_id, offline=True)) for boe_id in IDS))
for norm in sorted(needed):
    if norm in corpus.NOT_IN_API:
        print("la API no la tiene consolidada:", norm, "-> sus versiones se fechan con fecha_publicacion")
    else:
        boe.metadatos(norm, offline=True)  # si no está en la foto, esto falla
        print("en la foto:", norm)

# %%
cache = boe.verify_cache()
print(f"archivos en la foto: {cache['files']:,} · faltan: {len(cache['missing'])} · "
      f"no coinciden con su sha256: {len(cache['mismatched'])}")
print(f"guardados con CRLF por Windows: {cache['crlf_files']:,} (por eso el sha256 se compara sobre el texto)")
assert not cache["missing"] and not cache["mismatched"], "la caché no es la foto del 2026-10-24"

drift = [norm["id"] for norm in SUBSET["norms"] if norm["id"] in IDS
         and boe.metadatos(norm["id"], offline=True)["fecha_actualizacion"] != norm["fecha_actualizacion"]]
print(f"normas cuya fecha_actualizacion no coincide con subset_v1.json: {len(drift)}")
assert not drift

# %% [markdown]
r"""
**Lectura.** La foto está completa: no falta ningún archivo, ninguno difiere de su sha256 y las normas siguen en la versión fijada. Ninguna celda de esta notebook va a la API: todo corre offline.
"""

# %% [markdown]
r"""
## 2 · Extracción

Cada norma es una lista de bloques tipados (`precepto`, `encabezado`, `preambulo`, `nota_inicial`, `parte_dispositiva`, `firma`), y cada bloque trae **todas** sus versiones fechadas. Qué se hace con cada cosa (`src/corpus.py`):

| Elemento | Qué es | Qué hacemos |
|---|---|---|
| `<p>` | párrafo | texto sin markup: `<a>`, `<strong>`, `<em>` se van y sus palabras quedan |
| `<blockquote class="sangrado">` | el texto que cita una ley modificadora («...») | **texto** |
| los demás `<blockquote>` | notas de modificación y editoriales | `Section.notes`, fuera del texto |
| `<p class="cita_con_pleca">`, `<sup>` "(\*) Téngase en cuenta" | nota editorial en línea | `notes` |
| `<table>` | tabla | linealizada: "Encabezado: valor; ..." por fila |
| `<img>` | imagen (una URL) | `[imagen omitida]` |
| `<sup>2</sup>`, `<sup>o</sup>` | exponente, ordinal | `m2`, `5.º` |
| bloque `firma` | el pie repetido del corpus | se descarta |
| la fórmula de promulgación ("REY DE ESPAÑA", "A todos los que la presente vieren...") | el encabezado repetido del corpus | se descarta |

Primero se arma el corpus entero. Las celdas que siguen miran qué hizo cada regla.
"""

# %%
cleaner = Cleaner()
DOCS = corpus.build_corpus(IDS, cleaner)
SECTIONS = [(doc, s) for doc in DOCS for s in doc.sections]
print(f"{len(DOCS)} documentos · {len(SECTIONS):,} secciones · {sum(len(d.text) for d in DOCS):,} caracteres")
display(pd.Series(Counter(s.kind for _, s in SECTIONS), name="secciones").to_frame())

# %% [markdown]
r"""
**El hallazgo de esta etapa: no todo `<blockquote>` es una nota.** `class="sangrado"` es el texto que cita una ley modificadora («4. Son infracciones graves...»). La auditoría de la Etapa 0 (`boe.parse_texto`) lo descartaba junto con las notas: no tiró ningún error y las leyes modificadoras quedaban sin su contenido.
"""

# %%
quotes: dict[str, list[str]] = {}
for boe_id in IDS:
    for version_el in etree.fromstring(boe.texto(boe_id, offline=True).encode()).iter("version"):
        for el in version_el.iter("blockquote"):
            css = el.get("class") or "(sin clase)"
            quotes.setdefault("noDesde<fecha>" if css.startswith("noDesde") else css, []).append(corpus.flat_text(el))
display(pd.DataFrame([
    {"clase": css, "blockquotes": len(texts), "tokens": sum(ntok(texts)), "ejemplo": texts[0][:95],
     "tratamiento": "texto" if css == "sangrado" else "notes"}
    for css, texts in sorted(quotes.items(), key=lambda item: -len(item[1]))]))

# %% [markdown]
r"""
**Lectura.** El texto citado (`sangrado`) es texto legal de verdad. Sin él, un "Artículo único. Se modifica la Ley X, que queda redactada así:" queda vacío. Las notas propiamente dichas salen del texto pero no se pierden: van a `Section.notes`, por si una etapa posterior las necesita (por ejemplo, "¿qué norma modificó este artículo?").
"""

# %%
# Tablas: una de cada tipo, antes (el texto plano de un loader ingenuo) y después (linealizada).
def table_kind(table) -> str:
    grid = corpus.table_grid(table)
    merged = any(td.get("colspan") or td.get("rowspan") for td in table.iter("td", "th"))
    if not grid or not all(cell.header for cell in grid[0] if cell.text):
        return "sin encabezado"
    return "con encabezado y celdas combinadas" if merged else "con encabezado"


examples, n_tables = {}, 0
for boe_id in IDS:
    for block in etree.fromstring(boe.texto(boe_id, offline=True).encode()).iter("bloque"):
        if block.get("tipo") == "firma":
            continue  # las tablas de la firma (maquetación) se van con la firma
        for table in block.iter("table"):
            n_tables += 1
            examples.setdefault(table_kind(table), (boe_id, table))
print(f"tablas fuera de las firmas: {n_tables}")
for kind, (boe_id, table) in examples.items():
    print(f"\n--- {kind} ({boe_id})\nANTES:   {corpus.flat_text(table)[:240]}")
    for line in corpus.table_lines(table)[:3]:
        print("DESPUÉS:", line[:240])

# %% [markdown]
r"""
**Lectura.** Linealizada, cada fila se entiende sola ("Cupo: 1000; Año 1: 0,00; ..."), aunque el chunker la separe del resto de la tabla. El caso difícil son las fórmulas maquetadas como tabla (el último ejemplo): ahí ni el texto plano ni la linealización recuperan la fracción. Son pocas y quedan como defecto conocido en la corpus card.
"""

# %%
n_images = sum(doc.text.count(corpus.IMAGE) for doc in DOCS)
print(f"imágenes reemplazadas por {corpus.IMAGE!r}: {n_images} en {sum(corpus.IMAGE in d.text for d in DOCS)} normas")
print("son URLs a /datos/imagenes/, no base64 (la ficha de la Etapa 0 decía base64: corregido)")

# %% [markdown]
r"""
## 3 · Normalización

Las reglas de `src/cleaning.py`, en este orden:
1. `nfc`
2. `invisibles`: guion blando, ancho cero, BOM y caracteres de control
3. `odd_spaces`: NBSP y otros, pasados a espacio común
4. `compat`: ² → 2 y ligaduras
5. `whitespace`: espacios repetidos dentro de la línea; una línea por párrafo

Se normaliza **antes** de calcular los offsets de las secciones.

No se usa NFKC entero porque también convierte el ordinal "1.º" en "1.o". La celda siguiente muestra todo lo que NFKC habría cambiado además de nuestras reglas.
"""

# %%
nfkc = Counter()
for doc in DOCS:
    for char, n in Counter(doc.text).items():
        if unicodedata.normalize("NFKC", char) != char:
            nfkc[char] += n
display(pd.DataFrame([{"carácter": char, "código": f"U+{ord(char):04X}",
                       "NFKC lo cambiaría por": unicodedata.normalize("NFKC", char), "apariciones": n}
                      for char, n in nfkc.most_common()]))

# %% [markdown]
r"""
**Lectura.** Además de los ordinales (º, ª), NFKC habría roto la prima de los subapartados: "a´) En un 10 por ciento" se convertiría en un espacio más un acento combinado. Lo que sí unificamos (² → 2) es lo que hacía que "m2" y "m²" fueran dos palabras distintas para BM25.
"""

# %%
# Guiones de corte de línea ("arren-\ndamiento"): el XML no trae saltos de línea dentro de un párrafo,
# así que esta regla no tiene nada que hacer. Lo más parecido son unos "palabra- palabra"; se
# reportan y no se tocan ("decreto- ley" es un error de tipeo, "existente- no" puede ser una raya).
print("guion seguido de salto de línea:", sum(doc.text.count("-\n") for doc in DOCS))
hyphen_space = re.compile(r"\b[a-záéíóúñ]{2,}- [a-záéíóúñ]{2,}\b")
print("palabra- palabra:", Counter(m for doc in DOCS for m in hyphen_space.findall(doc.text)).most_common())

# %% [markdown]
r"""
## 4 · Ids estables

- `doc_id` = el identificador oficial del BOE (`BOE-A-1994-26003`, `BOCL-h-2020-90108`). Sale de la fuente, no del orden de las filas ni del nombre de un archivo.
- `section_id` = `doc_id#bloque@AAAAMMDD`: el id de bloque del BOE (`a9`; el 9 bis es `a9-2`, y no se deriva del título) más el día desde el que rige esa versión.
  - Con la fecha en el id, una descarga nueva **agrega** ids y nunca cambia el contenido de uno existente. Ejemplo: los arts. 38-40 de la LAU estaban derogados y en octubre de 2026 se llenaron con otro régimen.
  - Si dos versiones del mismo bloque caen el mismo día, la segunda lleva `.2`.
- **Intervalos.** Cada versión rige en `[valid_from, valid_to)`. `valid_to` es el `valid_from` de la versión siguiente. Dos cosas cierran una versión antes:
  - la `fecha_caducidad` del bloque, que la documentación de la API define como la fecha "a partir de la cual el bloque deja de mostrarse";
  - el fin de la norma (su derogación).
- **La miga de pan** (`path`) de cada versión son los títulos que la contenían el día en que empezó a regir. Las disposiciones de la parte final (adicionales, transitorias, derogatorias, finales) no cuelgan del último capítulo.

El art. 9 de la LAU contra la línea de tiempo de la auditoría (Etapa 0, §6), y el art. 10, donde la versión vigente no es la última:
"""

# %%
lau = next(doc for doc in DOCS if doc.doc_id == "BOE-A-1994-26003")


def minimum_term(text: str) -> str:
    m = re.search(r"inferior a (\w+) años(, o inferior a (\w+) años si el arrendador fuese persona jurídica)?", text)
    return (m.group(1) + (f" / {m.group(3)} (persona jurídica)" if m.group(3) else "")) if m else "-"


display(pd.DataFrame([
    {"section_id": s.section_id, "valid_from": s.valid_from, "valid_to": s.valid_to, "la introdujo": s.modified_by,
     "plazo mínimo": minimum_term(lau.section_text(s)) if s.block_id == "a9" else "-",
     "vigente el 2026-10-24": s.in_force(corpus.SNAPSHOT), "marcas": ", ".join(s.version_flags)}
    for s in lau.sections if s.block_id in ("a9", "a10")]))

# %%
IN_FORCE = [(doc, s) for doc, s in SECTIONS if s.in_force(corpus.SNAPSHOT)]
by_block: dict[tuple[str, str], list] = {}
for doc, s in SECTIONS:
    by_block.setdefault((doc.doc_id, s.block_id), []).append(s)  # ya vienen en orden de vigencia
not_last = [key for key, versions in by_block.items() if any(v.in_force(corpus.SNAPSHOT) for v in versions[:-1])]

print(f"bloques: {len(by_block):,} · con más de una versión: {sum(len(v) > 1 for v in by_block.values()):,}")
print(f"secciones vigentes el {corpus.SNAPSHOT}: {len(IN_FORCE):,} de {len(SECTIONS):,}")
print(f"bloques donde la versión vigente NO es la última: {len(not_last)} -> {not_last}")
print("de dónde sale valid_from:", dict(Counter(s.valid_from_source for _, s in SECTIONS)))
print("anomalías de la fuente:", dict(Counter(flag for _, s in SECTIONS for flag in s.version_flags)))

# %% [markdown]
r"""
Qué significa cada marca:

- `reordered`: el XML no traía las versiones en orden de vigencia; se ordenan por fecha.
- `same_day`: dos versiones el mismo día; la primera nunca rige.
- `expired_before_start`: el bloque caducó antes de que empezara la versión, que entonces nunca rigió (una ley extremeña agregó artículos para el 2024-01-01 que caducaron en 2023).
- `starts_at_norm_end`: la versión empieza el mismo día en que termina la norma (el "(Derogado)", o el RDL 27/2026, derogado el día en que entraba en vigor); nunca rige.
- `after_norm_end`: el BOE fecha la versión después de la derogación de la norma (régimen transitorio); queda abierta.

Cada intervalo vacío tiene su marca: ninguna versión deja de regir sin una razón registrada.

**Lectura.**
- **El art. 9 coincide con la línea de tiempo de la auditoría.** El plazo mínimo fue cinco años, después tres (2013), cinco o siete (RDL 21/2018), tres otra vez cuando cayó ese RDL, cinco o siete (RDL 7/2019), y por último las tres versiones de octubre de 2026. Para una pregunta sobre un contrato de 2016, el gold es `a9@20130606`; para una pregunta sin fecha, `a9@20261008`.
- **En el art. 10 la versión vigente no es la última.** La del RDL 28/2026 rige recién desde el 15/11/2026. Quedarse con "la última versión", como hacía la auditoría, habría contestado con una norma que todavía no rige.
"""

# %% [markdown]
r"""
## 5 · Metadatos validados contra el cuerpo

`DocMetadata.status` solo acepta `current`, `superseded`, `deprecated`, `draft` o `unknown`. El mapeo, documentado en `corpus_card.md`:
- `vigencia_agotada = N` → `current`.
- `vigencia_agotada = S` → `superseded` (derogada, agotada por plazo o sustituida). El valor crudo queda en `source_status`.

Sobre ese mapeo van dos correcciones verificadas a mano (`corpus.CORRECTIONS`) y una regla para las normas periódicas (`corpus.PERIODIC_SERIES`).

La trampa es "metadatos que mienten": antes de que una etapa filtre por estado o por fecha, cada campo se contrasta con el texto. Primero el estado contra la nota inicial de cada norma:
"""

# %%
DEROGATED = re.compile(r"norma derogada|derogad[oa] por|vigencia agotada|tácitamente sustituid", re.I)
rows = []
for doc in DOCS:
    note = " ".join(doc.section_text(s) for s in doc.sections if s.kind == "initial_note")
    rows.append({"doc_id": doc.doc_id, "fuente: vigencia_agotada": doc.metadata.source_status["vigencia_agotada"],
                 "la nota inicial dice derogada": bool(DEROGATED.search(note)), "status": doc.metadata.status,
                 "razón de la corrección": doc.metadata.status_reason})
status = pd.DataFrame(rows).set_index("doc_id")
display(pd.crosstab(status["fuente: vigencia_agotada"], status["la nota inicial dice derogada"], margins=True))
disagree = status[(status["fuente: vigencia_agotada"] == "S") != status["la nota inicial dice derogada"]]
print(f"desacuerdo entre la fuente y el texto: {len(disagree)}/{len(status)}")
display(disagree)

# %%
series = [doc for doc in DOCS if corpus.PERIODIC_SERIES[0] in doc.title]
print(f"resoluciones anuales de tipos de interés: {len(series)}; la fuente da vigentes a "
      f"{sum(d.metadata.source_status['vigencia_agotada'] == 'N' for d in series)}")
display(pd.DataFrame([{"doc_id": d.doc_id, "fecha_disposicion": d.metadata.fecha_disposicion,
                       "status": d.metadata.status, "superseded_by": d.metadata.superseded_by} for d in series]))
display(pd.crosstab(status["fuente: vigencia_agotada"], status["status"], margins=True))
superseded = [d for d in DOCS if d.metadata.status == "superseded"]
print(f"reemplazadas con superseded_by: {sum(bool(d.metadata.superseded_by) for d in superseded)}/{len(superseded)} "
      f"(sin él, la fuente no nombra a la norma que la reemplazó: agotadas por plazo o derogadas por una ley general)")

# %% [markdown]
r"""
**Lectura.** Hay un solo desacuerdo entre la fuente y el texto: la Ley balear 11/2005, que figura vigente y cuyo propio texto dice "Norma derogada, con efectos de 1 de enero de 2018". Las 29 agotadas lo dicen también en su nota. El otro error de estado no se ve comparando con la nota: las 13 resoluciones anuales de tipos de interés figuran todas vigentes, y cada una reemplaza a la anterior. Ninguna corrección se hace "a ojo" en el código: las dos manuales están en `corpus.CORRECTIONS`, con su razón, y la serie periódica es una regla explícita.

Después, la fecha de vigencia contra el texto. Una norma que dice "entrará en vigor el día siguiente al de su publicación" debería tener `fecha_vigencia` = publicación + 1 día. En las autonómicas, la publicación que cuenta es la de su boletín regional, cuya fecha figura en `nota_publicacion`.
"""

# %%
MONTHS = {name: number for number, name in enumerate(
    "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split(), start=1)}


def spanish_date(text: str | None) -> date | None:
    """'Publicada en el DOG núm. 88, de 10 de mayo de 1988.' -> date(1988, 5, 10)."""
    m = re.search(r"(\d{1,2}) de ([a-z]+) de (\d{4})", (text or "").lower())
    return date(int(m[3]), MONTHS[m[2]], int(m[1])) if m and m[2] in MONTHS else None


def publication_group(doc) -> str:
    if doc.metadata.ambito == "Estatal":
        return "estatal"
    # Una autonómica con id BOE-A es la copia en el BOE de lo que salió antes en su boletín.
    return "autonómica, id BOE-A" if doc.doc_id.startswith("BOE-") else "autonómica, id de su boletín"


NEXT_DAY = re.compile(r"entrar[áa] en vigor (el|al) día siguiente (al|a la fecha) de su publicación", re.I)
rows = []
for doc in DOCS:
    entry = [doc.section_text(s) for s in doc.sections if "vigor" in s.title.lower()]
    if not doc.metadata.fecha_vigencia or not any(NEXT_DAY.search(text) for text in entry):
        continue
    vigencia = date.fromisoformat(doc.metadata.fecha_vigencia)
    published = date.fromisoformat(doc.metadata.fecha_publicacion)
    official = spanish_date(doc.metadata.nota_publicacion) or published  # la del boletín regional, si hay nota
    rows.append({"grupo": publication_group(doc), "con fecha_publicacion": published + timedelta(days=1) == vigencia,
                 "con la publicación oficial": official + timedelta(days=1) == vigencia})
dates = pd.DataFrame(rows)
display(dates.groupby("grupo").agg(**{"normas": ("grupo", "size"),
                                      "coincide, con fecha_publicacion": ("con fecha_publicacion", "sum"),
                                      "coincide, con la publicación oficial": ("con la publicación oficial", "sum")}))

# %% [markdown]
r"""
**Lectura.** `fecha_vigencia` coincide con el texto en todas las normas que se pueden chequear (las que entran en vigor "el día siguiente"), siempre que se use la publicación oficial. El campo engañoso es `fecha_publicacion`: en las autonómicas con id `BOE-A` es la fecha de la **copia en el BOE**, posterior a la del boletín regional, así que ninguna coincide. Las autonómicas con id de su propio boletín (`BOA-`, `BORM-`, ...) ya traen la fecha regional. Por eso las preguntas temporales (Etapa 2) y los filtros por fecha (Etapas 3 y 6) usan `valid_from`/`valid_to`, que salen de `fecha_vigencia`, y nunca `fecha_publicacion`.
"""

# %%
display(pd.Series(Counter(d.metadata.region for d in DOCS), name="normas").sort_values(ascending=False).to_frame())
print("departamentos que se unifican:", {k: v for k, v in corpus.REGIONS.items() if "Valencia" in k})

# %% [markdown]
r"""
## 6 · Casi-duplicados

Se miden en tres niveles, con `dedup.py`: shingles de 5 palabras; MinHash + LSH proponen candidatos y el Jaccard exacto decide.

- **Norma ↔ norma:** todos los pares, con Jaccard y contención. La contención ve un decreto-ley entero adentro de la ley que lo convalidó.
- **Sección ↔ sección de otra norma** (Jaccard ≥ 0,7): queda en `Section.twins`.
- **Versión ↔ versión del mismo bloque:** secciones distintas, unidas por `block_id`.
"""

# %%
title = {doc.doc_id: doc.title for doc in DOCS}
pairs = pd.DataFrame([
    {"a": doc.doc_id, "b": nd["doc_id"], "jaccard": nd["jaccard"], "contención": nd["containment"],
     "relación": nd["relation"], "título a": doc.title[:55], "título b": title[nd["doc_id"]][:55]}
    for doc in DOCS for nd in doc.metadata.near_duplicates if doc.doc_id < nd["doc_id"]],
    columns=["a", "b", "jaccard", "contención", "relación", "título a", "título b"])
print(f"pares de normas con Jaccard >= 0,3 o contención >= 0,8: {len(pairs)} · por relación: "
      f"{dict(Counter(pairs['relación']))} · copias (Jaccard >= 0,9): {int((pairs['jaccard'] >= 0.9).sum())}")
display(pairs.sort_values("contención", ascending=False).head(15))

# %%
with_twins = [(doc, s) for doc, s in SECTIONS if s.twins]
print(f"secciones con un gemelo casi literal en otra norma: {len(with_twins):,}/{len(SECTIONS):,}")
twin_pairs = Counter((doc.doc_id, twin.split("#")[0]) for doc, s in with_twins for twin in s.twins
                     if doc.doc_id < twin.split("#")[0])
for (a, b), n in twin_pairs.most_common(8):
    print(f"  {n:>4}  {title[a][:55]}  <->  {title[b][:55]}")
in_force_ids = {s.section_id for _, s in IN_FORCE}
both = sum(twin in in_force_ids for _, s in IN_FORCE for twin in s.twins) // 2
print(f"pares de gemelos vigentes los dos el {corpus.SNAPSHOT}: {both} (son los que la Etapa 2 etiqueta como conjunto)")

# %% [markdown]
r"""
**La política** (queda en los metadatos y en la corpus card):

- **A nivel norma no hay copias intercambiables** (ver el Jaccard máximo arriba). Los pares parecidos son:
  - **versiones**, cuando una reemplaza a la otra: un RDL y la ley que lo convalidó, un RDL derogado y el que lo re-decretó, una resolución anual y la del año siguiente;
  - normas **relacionadas**, que comparten texto.

  Se registran en `near_duplicates` con la relación, y la reemplazada lleva `superseded_by` cuando la fuente nombra la norma que la reemplazó. No se borra ni se fusiona ninguna.
- **Las versiones del mismo bloque dan respuestas distintas** ("tres años" en 2016, "cinco" hoy), así que son secciones distintas.
- **Compromiso para la Etapa 2:**
  - Una pregunta **sin fecha** tiene como gold la versión vigente al 2026-10-24, más sus `twins` vigentes, como conjunto: el mismo texto en otra norma es la misma respuesta.
  - Una pregunta **a una fecha** tiene como gold la versión vigente en esa fecha.
"""

# %% [markdown]
r"""
## 7 · `documents.jsonl` y su manifiesto
"""

# %%
corpus_sha = doc_schema.documents_hash(DOCS)
artifact_io.write_jsonl(ART / "documents.jsonl", DOCS)
doc_schema.validate_documents(ART / "documents.jsonl").show().raise_if_failed()

# El build es determinista: armarlo de nuevo da el mismo hash. Si no lo fuera, el eval set
# congelado contra este corpus no cerraría en Colab.
assert doc_schema.documents_hash(corpus.build_corpus(IDS, Cleaner())) == corpus_sha
print("corpus_sha:", corpus_sha)

# %%
def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


manifest = {
    "artifact": "documents.jsonl",
    "profile": ART.name,
    "snapshot": SUBSET["snapshot"],
    "source": SUBSET["source"],
    "subset_file_sha256": artifact_io.sha256_file(ROOT / "data" / "subset_v1.json"),
    "raw_cache": {"files": cache["files"], "index_sha256": cache["index_sha256"]},
    "n_documents": len(DOCS),
    "n_sections": len(SECTIONS),
    "n_sections_in_force_at_snapshot": len(IN_FORCE),
    "corpus_sha": corpus_sha,
    "documents_sha256": artifact_io.sha256_file(ART / "documents.jsonl"),
    "code_commit": git("rev-parse", "HEAD"),
    "code_dirty": bool(git("status", "--porcelain", "--", "src")),
    "cleaning": cleaner.summary(),
    "environment": {"env": "colab" if environment.on_colab() else "local",
                    "python": platform.python_version(), "platform": platform.platform()},
    "libraries": {lib: version(lib) for lib in ("lxml", "pydantic", "tokenizers", "numpy")},
    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
artifact_io.write_manifest(ART / "documents.manifest.json", manifest)
print(json.dumps({k: v for k, v in manifest.items() if k != "cleaning"}, ensure_ascii=False, indent=1))

# %% [markdown]
r"""
## 8 · Qué cambió la limpieza

Por regla: en cuántas normas cambió algo y cuántos caracteres sacó (negativo = agregó). Las reglas de extracción las cuenta `corpus.py`; las de normalización, `cleaning.py`.
"""

# %%
WHAT = {
    "notes_out": "notas de modificación y editoriales (<blockquote>) -> Section.notes",
    "editorial_inline": "notas editoriales en línea («(*) Téngase en cuenta...») -> notes",
    "signature": "bloque de firma",
    "promulgation": "fórmula de promulgación al comienzo del preámbulo",
    "tables": "tablas linealizadas (negativo: el encabezado se repite en cada fila)",
    "images": f"imágenes -> {corpus.IMAGE} (negativo: se agrega el marcador)",
    "sup_ordinals": "5.<sup>o</sup> -> 5.º (no cambia el largo)",
    "nfc": "acentos descompuestos -> compuestos",
    "invisibles": "guion blando, ancho cero, BOM, control",
    "odd_spaces": "NBSP y otros espacios -> espacio común (no cambia el largo)",
    "compat": "superíndices y ligaduras (² -> 2)",
    "whitespace": "espacios repetidos dentro de una línea",
}
rules = pd.DataFrame(cleaner.summary()).assign(qué=lambda t: t["rule"].map(WHAT))
display(rules.sort_values("chars_removed", ascending=False).reset_index(drop=True))

# %% [markdown]
r"""
**Lectura.** La regla que más cambia el texto, por lejos, es la que saca las notas de modificación. Las de normalización tocan muy poco: el XML del BOE es texto limpio (2 normas con acentos descompuestos, 5 con guiones blandos). Lo que haría fallar a un loader ingenuo no es la codificación sino la **estructura**:
- notas mezcladas con el articulado;
- versiones superpuestas;
- texto citado tratado como nota;
- la última versión tomada por la vigente.
"""

# %%
naive = {doc.doc_id: corpus.naive_text(boe.texto(doc.doc_id, offline=True)) for doc in DOCS}
naive_tokens = sum(ntok(list(naive.values())))
clean_tokens = sum(ntok([doc.text for doc in DOCS]))
in_force_tokens = sum(ntok([doc.section_text(s) for doc, s in IN_FORCE]))
print(f"tokens de Qwen3 · loader ingenuo (todas las versiones, con notas y firma): {naive_tokens:>11,}")
print(f"                · documents.jsonl (todas las versiones):                  {clean_tokens:>11,}")
print(f"                · solo lo vigente el {corpus.SNAPSHOT}:                      {in_force_tokens:>11,}")

# %% [markdown]
r"""
**Los números de la Etapa 0, corregidos.** Se armaron con `boe.parse_texto`, que (a) tiraba el texto citado con las notas y (b) tomaba como vigente la última versión de cada bloque. La lista de normas **no se re-selecciona**: está fijada en `data/subset_v1.json`. Sí se mide si la regla de densidad la habría cambiado.
"""

# %%
print(f"texto citado (<blockquote class='sangrado'>) que la Etapa 0 contaba como nota: "
      f"{sum(ntok(quotes.get('sangrado', []))):,} tokens, todas las versiones")

# Densidad temática con la extracción corregida: la última versión de cada artículo que ya regía en la foto.
HOUSING = re.compile(SUBSET["rule"]["density_regex"], re.I)
latest = {}
for doc, s in SECTIONS:
    if s.kind == "article" and s.valid_from <= corpus.SNAPSHOT:
        latest[(doc.doc_id, s.block_id)] = (doc.doc_id, doc.section_text(s))  # en orden de fecha: queda la última
total, housing = Counter(), Counter()
for (doc_id, text), n in zip(latest.values(), ntok([text for _, text in latest.values()])):
    total[doc_id] += n
    housing[doc_id] += n * bool(HOUSING.search(text))
below = sorted((round(housing[d] / total[d], 3), d) for d in total
               if housing[d] / total[d] < SUBSET["rule"]["density_min"])
print(f"normas que con la extracción corregida quedarían debajo del umbral de densidad "
      f"({SUBSET['rule']['density_min']}): {len(below)} {below}")

# %% [markdown]
r"""
**Lectura.** Lo vigente en la foto pesa menos que los 3,09 M de la Etapa 0, por tres razones:
- aquella cuenta tomaba la última versión de cada bloque, aunque todavía no rigiera;
- incluía las normas derogadas, que acá no tienen nada vigente;
- en sentido contrario, no contaba el texto citado.

Sigue sin entrar en ninguna ventana, que es lo que justifica construir recuperación. Ninguna norma cruza el umbral de densidad con la extracción corregida: el subconjunto fijado se habría elegido igual.
"""

# %% [markdown]
r"""
**Antes y después, sobre los 10 documentos de lectura.** Las líneas que la limpieza sacó (`-`) o cambió (`+`). Son los documentos crudos que se leyeron de punta a punta en la Etapa 0.
"""

# %%
for boe_id in READING:
    doc = next((d for d in DOCS if d.doc_id == boe_id), None)
    if doc is None:
        continue
    before = [line for line in naive[boe_id].splitlines() if line]
    after = [line for line in doc.text.splitlines() if line]
    diff = [line for line in difflib.unified_diff(before, after, lineterm="", n=0)
            if line[:1] in "+-" and line[:3] not in ("---", "+++")]
    removed, added = sum(line.startswith("-") for line in diff), sum(line.startswith("+") for line in diff)
    print(f"\n=== {boe_id} · {doc.title[:85]}")
    print(f"    líneas {len(before):,} -> {len(after):,} · quitadas {removed:,} · agregadas {added:,}")
    for line in diff[:8]:
        print("   ", line[:150])

# %% [markdown]
r"""
## 9 · Para la corpus card
"""

# %%
in_force_by_doc = Counter()
for (doc, s), n in zip(IN_FORCE, ntok([doc.section_text(s) for doc, s in IN_FORCE])):
    in_force_by_doc[doc.doc_id] += n
per_doc = pd.Series([in_force_by_doc[doc.doc_id] for doc in DOCS if doc.metadata.status == "current"])
PROMPT_BUDGET = 8192 - 768
card = {
    "documentos": len(DOCS),
    "secciones (bloque × versión)": len(SECTIONS),
    "secciones vigentes el 2026-10-24": len(IN_FORCE),
    "estado": dict(Counter(doc.metadata.status for doc in DOCS)),
    "tokens, todas las versiones": clean_tokens,
    "tokens vigentes el 2026-10-24": in_force_tokens,
    "tokens vigentes por norma vigente (p10 / mediana / p90 / máx.)":
        [int(per_doc.quantile(q)) for q in (0.1, 0.5, 0.9)] + [int(per_doc.max())],
    f"normas vigentes que entran solas en {PROMPT_BUDGET:,} tokens": f"{int((per_doc <= PROMPT_BUDGET).sum())}/{len(per_doc)}",
    "pares de normas casi-duplicadas": len(pairs),
    "secciones con gemelos en otra norma": len(with_twins),
    "corpus_sha": corpus_sha,
}
display(pd.Series(card, name="valor").to_frame())
display(pd.crosstab(pd.Series([d.metadata.ambito for d in DOCS], name="ámbito"),
                    pd.Series([d.metadata.status for d in DOCS], name="status"), margins=True))

# %% [markdown]
r"""
## Qué se lleva la Etapa 2

- **Los ids.** `doc_id#bloque@AAAAMMDD`.
  - El gold de una pregunta **sin fecha** es la versión vigente al 2026-10-24, más sus `twins` vigentes, como conjunto.
  - El de una pregunta **a una fecha** es la versión vigente en esa fecha (`Section.in_force`).
- **El hash del corpus** (`corpus_sha`, en el manifiesto). El congelado del eval set lo registra; una foto distinta lo rompe.
- **De dónde salen las preguntas especiales:**
  - las de identificador, de identificadores que aparecen en una sola norma;
  - las `out_of_corpus`, de los huecos conocidos (el procedimiento de desahucio de la LEC, la ley catalana 18/2007) y de lo que la foto no tiene (cualquier versión posterior al 2026-10-24).
"""
