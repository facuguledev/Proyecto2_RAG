# %% [markdown]
r"""
# Etapa 3 · Recuperación sobre el corpus del BOE

Es el protocolo del curso: la vara congelada, una variable por fila, la tabla leída contra el ruido. Ahora corre sobre un corpus que limpiamos nosotros. La pregunta es **qué piezas se ganan su costo en este corpus**.

Lo que dejó la Etapa 2 (BM25 sobre secciones enteras):
- en `topical`, el vocabulario no coincide ("caldera" contra "reparaciones"): es trabajo para el denso;
- en `metadata`, BM25 encuentra el artículo pero no la versión: es trabajo para el filtro por fecha;
- en `identifier`, el número de la norma no está en el texto de la sección: es trabajo para el prefijo.

Todo lo que sigue quedó escrito **antes** de la primera corrida completa, en `decision_log.md` (Etapa 3, pre-registrada):
- **Primaria:** "clave entregada desde el gold". Algún pasaje entregado es de una sección gold (o de una versión con el mismo texto) y contiene una clave.
- **Entrega:** 3.000 tokens del generador por pregunta, en todas las filas, con la misma cabecera por pasaje.
- **Filas:** `bm25` y `dense` (líneas de base); después, desde el denso, una pieza por fila: `+prefix`, `+rrf`, `+rerank`, `+parent`, `+date`, `+region`.
- **La mejor configuración:** la de más aciertos en la primaria. Si una con menos piezas no se distingue (test de signos, p ≥ 0,05), gana la de menos piezas.

La lógica vive en `src/`:
- `chunking.py`: qué entra al índice, los padres, los hijos, el prefijo y la cabecera;
- `retrieval.py`: `retrieve()`, un flag por pieza, el filtro y la entrega;
- `evaluation.py`: las métricas desde las filas crudas, y el ruido.

Son 0 llamadas al LLM.
"""

# %%
import environment  # el del curso, primero: guarda de JAX (bm25s lo importa), caché de HF, UTF-8

import os
import sys
import time
from dataclasses import asdict

ROOT = environment.ROOT
sys.path.insert(0, str(ROOT / "src"))  # nuestros módulos

import numpy as np
import pandas as pd

try:
    from IPython.display import display  # en la notebook
except ImportError:
    display = print  # como script (tools/run_stages.py)

import store
from console import StageRun

import artifact_io
import chunking
import doc_schema
import evalset
import evaluation
import retrieval
from chunking import Unit, passage_text, searchable_text
from lexical import Bm25Index
from retrieval import Index, RetrievalConfig

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
PROFILE = "smoke" if SMOKE_TEST else "full"
ART = environment.artifacts_dir(SMOKE_TEST)
INDEX_DIR = ART / "index"

GENERATOR_TOKENIZER, GENERATOR_REVISION = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
# Revisiones fijadas: el sha del repo en HF el 2026-10-23 (huggingface_hub.model_info).
EMBEDDER, EMBEDDER_REVISION = "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER, RERANKER_REVISION = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825"
MAX_SEQ = 512  # bge-m3 declara 8.192 pero trunca en silencio donde se lo fije: los hijos miden <= 350

DOCS = doc_schema.load_documents(ART / "documents.jsonl")
CORPUS_SHA = doc_schema.documents_hash(DOCS)
artifact_io.check_manifest(ART / "documents.manifest.json", {"corpus_sha": CORPUS_SHA})  # es el corpus de la Etapa 1
SECTIONS = evalset.section_index(DOCS)
if SMOKE_TEST:
    ev = evalset.load_eval(ROOT / "eval" / "smoke", require_frozen=False)  # el smoke nunca se congela
    evalset.validate_eval(ev, DOCS).show().raise_if_failed()  # sin congelar, load_eval no chequea los ids
else:
    ev = evalset.load_eval(ROOT / "eval", ART / "documents.jsonl")  # rechaza un conjunto editado o un corpus distinto
ITEMS = ev.items
ANSWERABLE = [item for item in ITEMS if item.answerable]
REPORTED = {"dev", "test"} if SMOKE_TEST else {"test"}  # el smoke es todo dev, y sus números no significan nada

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)
environment.summary()
run = StageRun.start("03_retrieval", ART, profile=PROFILE, config={"eval": ev.key(), "corpus_sha": CORPUS_SHA})
print(f"perfil {PROFILE} | corpus_sha {CORPUS_SHA[:12]}... | {len(DOCS)} normas, {len(SECTIONS):,} secciones | "
      f"eval {ev.key()}: {ev.summary()}")

# %% [markdown]
r"""
## 1 · El índice: qué entra, padres e hijos

- **Qué entra.** Los artículos, los preámbulos, los anexos y lo dispositivo, en todas sus versiones. Quedan afuera las versiones que nunca rigieron (intervalo vacío), los encabezados y las notas iniciales. Ningún gold es de esos tipos, y el chequeo de la sección 2 lo confirma.
- **Padre** = la sección (un artículo en una versión) si entra en 1.200 tokens del generador; si no, grupos de párrafos.
- **Hijo** = párrafos de un padre hasta 350 tokens del reranker, **con el prefijo incluido**. Lleva ~15% de solape, y un párrafo más largo se parte en ventanas. El corte es el mismo con y sin prefijo.
- **Prefijo:** `rango número (región) | título de la sección | vigencia en mes y año | último nivel de las migas | título de la norma`, recortado por el final a 120 tokens.

El índice se construye una vez y se reusa mientras no cambien el corpus, los parámetros o el código de `chunking.py` (`store.build_or_load`).
"""

# %%
from tokenizers import Tokenizer
from transformers import AutoTokenizer


def backend(repo: str, revision: str | None):
    """El ``tokenizers.Tokenizer`` de un modelo de HF, sin truncar ni rellenar: contamos, no recortamos."""
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision).backend_tokenizer
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


GEN_TOK = Tokenizer.from_pretrained(GENERATOR_TOKENIZER, revision=GENERATOR_REVISION)
RERANK_TOK = backend(RERANKER, RERANKER_REVISION)
EMBED_TOK = backend(EMBEDDER, EMBEDDER_REVISION)

INDEX_SPEC = {
    "corpus_sha": CORPUS_SHA,
    "indexed_kinds": list(chunking.INDEXED_KINDS),
    "child_max": chunking.CHILD_MAX, "child_overlap": chunking.CHILD_OVERLAP,
    "parent_max": chunking.PARENT_MAX, "prefix_max": chunking.PREFIX_MAX, "template": chunking.TEMPLATE,
    "tokenizers": {"child": [RERANKER, RERANKER_REVISION], "parent": [GENERATOR_TOKENIZER, GENERATOR_REVISION]},
    "code": store.sha256_text((ROOT / "src" / "chunking.py").read_text(encoding="utf-8")),
}


def build_index(path):
    parents, children = chunking.build_units(DOCS, RERANK_TOK, GEN_TOK)
    store.write_jsonl(INDEX_DIR / "parents.jsonl", [asdict(unit) for unit in parents])
    store.write_manifest(INDEX_DIR / "parents.jsonl", INDEX_SPEC)
    store.write_jsonl(path, [asdict(unit) for unit in children])


t0 = time.time()
store.build_or_load(INDEX_DIR / "chunks.jsonl", INDEX_SPEC, build_index)
store.check_manifest(INDEX_DIR / "parents.jsonl", INDEX_SPEC)  # los dos archivos salieron del mismo build
CHILDREN = [Unit(**row) for row in store.read_jsonl(INDEX_DIR / "chunks.jsonl")]
PARENTS = {row["unit_id"]: Unit(**row) for row in store.read_jsonl(INDEX_DIR / "parents.jsonl")}
UNITS = {unit.unit_id: unit for unit in CHILDREN} | PARENTS
assert not any("~" in section_id for section_id in SECTIONS), "un section_id con '~' rompería section_of"
print(f"índice en {time.time() - t0:.0f} s: {len(CHILDREN):,} hijos, {len(PARENTS):,} padres")

indexed = chunking.indexed_sections(DOCS)
left_out = pd.Series([section.kind if section.kind not in chunking.INDEXED_KINDS else "nunca rigió"
                      for doc in DOCS for section in doc.sections
                      if section.kind not in chunking.INDEXED_KINDS or chunking.never_ruled(section)])
print(f"secciones indexadas: {len(indexed):,} de {len(SECTIONS):,}; afuera, por motivo:")
display(left_out.value_counts().to_frame("secciones"))


def percentiles(values) -> dict:
    values = np.asarray(values)
    return {"min": int(values.min()), "p50": int(np.median(values)), "p90": int(np.percentile(values, 90)),
            "max": int(values.max())}


display(pd.DataFrame({
    "hijos · tokens del reranker (prefijo + texto)": percentiles([u.n_rerank for u in CHILDREN]),
    "hijos · tokens del prefijo": percentiles(chunking.count(RERANK_TOK, [u.prefix for u in CHILDREN])),
    "padres · tokens del generador (pasaje)": percentiles([u.n_gen for u in PARENTS.values()]),
    "hijos · tokens del generador (pasaje)": percentiles([u.n_gen for u in CHILDREN]),
}).T)
print("hijos por sección:", percentiles(pd.Series([u.section_id for u in CHILDREN]).value_counts()))
example = next((u for u in CHILDREN if u.section_id.startswith("BOE-A-1994-26003#a9@")), CHILDREN[0])  # LAU, art. 9
print("\nun hijo con su prefijo (lo que leen los canales con prefix=True):\n", searchable_text(example, True)[:700])
print("\nel mismo, como pasaje entregado:\n", passage_text(example)[:400])

# %% [markdown]
r"""
## 2 · Chequeos antes de medir

Si alguno falla, la etapa se corta: un índice que no puede entregar el gold daría una tabla sobre otra cosa.
- **El gold está en el índice (§4).** Cada sección gold (o una versión con el mismo texto) tiene un hijo **y** un padre cuyo pasaje contiene una clave. Una ventana podría partir la clave, y entonces ninguna fila podría entregarla.
- **Ventanas.** Ningún hijo con prefijo pasa los 512 tokens de bge-m3: los truncaría en silencio. Ningún padre pasa los 1.200 tokens del generador. La ventana del reranker se mide con los pares reales, en la sección 6.
"""

# %%
ACCEPTED = {item.item_id: evaluation.text_equivalents(item.gold_sections, SECTIONS) for item in ANSWERABLE}


def gold_in_index(items, accepted) -> list[str]:
    """Las preguntas cuyo gold no tiene un hijo y un padre con la clave en el índice."""
    missing = []
    for item in items:
        for kind, units in (("hijo", CHILDREN), ("padre", PARENTS.values())):
            if not any(u.section_id in accepted[item.item_id] and evalset.contains_key(passage_text(u), item.answer_key)
                       for u in units):
                missing.append(f"{item.item_id}: ningún {kind} de {sorted(accepted[item.item_id])} tiene la clave")
    return missing


missing = gold_in_index(ANSWERABLE, ACCEPTED)
if missing:
    raise RuntimeError("el gold no está en el índice:\n" + "\n".join(missing))
print(f"ok: las {len(ANSWERABLE)} preguntas con respuesta tienen un hijo y un padre con la clave")

# La guarda tiene que saltar con un id faltante (§7.4). Se arma en memoria, nunca en el archivo.
broken = ANSWERABLE[0].model_copy(update={"gold_sections": ["BOE-A-0000-0#a1@20000101"]})
fired = gold_in_index([broken], {broken.item_id: {"BOE-A-0000-0#a1@20000101"}})
assert fired, "la guarda del id gold faltante no saltó"
print("guarda con un id gold faltante ->", fired)

embed_lengths = [len(e.ids) for e in EMBED_TOK.encode_batch([searchable_text(u, True) for u in CHILDREN])]
parent_text_lengths = chunking.count(GEN_TOK, [u.text for u in PARENTS.values()])
print(f"hijos con prefijo de más de {MAX_SEQ} tokens de bge-m3: {sum(n > MAX_SEQ for n in embed_lengths)} "
      f"(máximo {max(embed_lengths)})")
print(f"padres de más de {chunking.PARENT_MAX} tokens del generador: "
      f"{sum(n > chunking.PARENT_MAX for n in parent_text_lengths)} (máximo {max(parent_text_lengths)})")
assert max(embed_lengths) <= MAX_SEQ and max(parent_text_lengths) <= chunking.PARENT_MAX
assert max(u.n_rerank for u in CHILDREN) <= chunking.CHILD_MAX, "un hijo pasa el presupuesto del reranker"

# %% [markdown]
r"""
## 3 · Embeddings

- `bge-m3` en fp16, sin prefijos de instrucción, `max_seq_length = 512`, normalizado.
- Hay dos pasadas: la de los hijos crudos y la de los hijos con prefijo.
- **Cada texto distinto se embebe una sola vez.** 518 secciones son idénticas a otra versión de su bloque: con un solo vector, su empate no lo decide el ruido de fp16.
- `store.cached_matrix` guarda la matriz en `ART/embeddings/`. Su clave es el encoder más los ids y los textos: un prefijo nuevo o un re-corte da una matriz nueva.
"""

# %%
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

DEVICE = environment.device()
# fp16 solo en Colab: la T4 tiene tensor cores. La GTX 1660 Super local no, y ahí fp16 midió 3,0 textos/s
# contra 14,2 en fp32 (decision_log.md, Etapa 3). El dtype va en la clave de la caché: no se mezclan.
FP16 = DEVICE == "cuda" and environment.on_colab()
embedder = SentenceTransformer(EMBEDDER, revision=EMBEDDER_REVISION, device=DEVICE)
if FP16:
    embedder.half()
embedder.max_seq_length = MAX_SEQ
ENCODER_SPEC = {"model": EMBEDDER, "revision": EMBEDDER_REVISION, "max_seq_length": MAX_SEQ,
                "dtype": "float16" if FP16 else "float32", "normalize": True, "instruction_prefix": None}


def encode(texts: list[str]) -> np.ndarray:
    return embedder.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)


def embed(name: str, texts: list[str]) -> np.ndarray:
    """Una fila por texto, calculando cada texto distinto una sola vez."""
    unique_texts = list(dict.fromkeys(texts))
    ids = [store.sha256_text(text)[:16] for text in unique_texts]
    matrix = store.cached_matrix(name, ENCODER_SPEC, ids, unique_texts, encode, art_dir=ART)
    row = {text: i for i, text in enumerate(unique_texts)}
    # fp16 en una GTX 16xx puede desbordar: un NaN desaparecería del ranking sin ningún error.
    norms = np.linalg.norm(matrix, axis=1)
    assert np.isfinite(matrix).all() and np.allclose(norms, 1, atol=1e-2), f"{name}: embeddings no finitos o sin normalizar"
    print(f"   {name}: {len(texts):,} hijos, {len(unique_texts):,} textos distintos")
    return np.asarray(matrix[[row[text] for text in texts]], dtype=np.float32)


t0 = time.time()
RAW_MATRIX = embed("children_raw", [searchable_text(u, False) for u in CHILDREN])
PREFIXED_MATRIX = embed("children_prefixed", [searchable_text(u, True) for u in CHILDREN])
embed_seconds = round(time.time() - t0)
print(f"embeddings en {embed_seconds} s en {DEVICE}; matriz {RAW_MATRIX.shape}")
environment.gpu_report("después de los embeddings")

# %% [markdown]
r"""
## 4 · Las piezas, y la guarda que salta

- **BM25:** `lexical.Bm25Index` con los mismos términos de la Etapa 2 (sin cambios: la contaminación sigue valiendo). Hay un índice sobre hijos crudos y otro sobre hijos con prefijo.
- **Reranker:** `mmarco-mMiniLMv2-L12-H384-v1`, `max_length = 512`, con logits crudos (sin sigmoide: el orden es el mismo, y el puntaje máximo de la sección 8 se lee mejor).
- **Metadatos:** la vigencia y la región de cada hijo, para el filtro.

Una fila que pide una pieza que no está cargada lanza una excepción: la celda lo muestra.
"""

# %%
t0 = time.time()
ids = [u.unit_id for u in CHILDREN]
bm25_raw = Bm25Index(ids, [searchable_text(u, False) for u in CHILDREN])
bm25_prefixed = Bm25Index(ids, [searchable_text(u, True) for u in CHILDREN])
print(f"BM25 (dos índices) en {time.time() - t0:.0f} s")
reranker = CrossEncoder(RERANKER, revision=RERANKER_REVISION, max_length=512, device=DEVICE,
                        activation_fn=torch.nn.Identity())
RERANKER_SPEC = {"model": RERANKER, "revision": RERANKER_REVISION, "max_length": 512, "scores": "logits"}

INDEX = Index(children=CHILDREN, parents=PARENTS, raw_matrix=RAW_MATRIX, prefixed_matrix=PREFIXED_MATRIX,
              bm25_raw=bm25_raw, bm25_prefixed=bm25_prefixed, embedder=embedder, reranker=reranker,
              metadata=retrieval.unit_metadata(CHILDREN, DOCS))
INDEX_KEY = store.sha256_json({"units": store.read_manifest(INDEX_DIR / "chunks.jsonl")["key"],
                               "encoder": ENCODER_SPEC, "reranker": RERANKER_SPEC})[:16]

try:
    retrieval.retrieve("fianza", RetrievalConfig(rerank=True), Index(children=CHILDREN, raw_matrix=RAW_MATRIX,
                                                                      embedder=embedder))
except RuntimeError as error:
    print("guarda: rerank=True sin reranker ->", error)
else:
    raise AssertionError("la guarda de la pieza faltante no saltó")

# %% [markdown]
r"""
## 5 · El filtro: qué lee de cada pregunta

El filtro se parsea **de la pregunta**, nunca de `as_of` (la etiqueta con la que se validó el gold).
- **Fecha:** el mes o el año que nombra la pregunta. Si no nombra ninguno, lo vigente en la foto (2026-10-24).
- **Región:** una comunidad nombrada en la pregunta. Quedan sus normas y las estatales.

La columna `período ∋ día del gold` sí lee `as_of`, pero es un diagnóstico, como leer el gold: dice si el período parseado contiene el día por el que se pregunta. `retrieve` no lo ve nunca.
"""

# %%
parsed = []
for item in ITEMS:
    filters = retrieval.parse_filters(item.question)
    period = retrieval.period_of(filters)  # la misma regla que aplica el filtro
    parsed.append({"id": item.item_id, "split": item.split, "categoría": item.category,
                   "período": f"{period[0]} → {period[1]}" if filters.period else "foto",
                   "regiones": ", ".join(filters.regions), "pregunta": item.question[:90],
                   "período ∋ día del gold": (period[0] <= item.reference_day() < period[1]) if item.answerable else None})
parsed = pd.DataFrame(parsed)
display(parsed)
print("dispara la fecha en", int((parsed["período"] != "foto").sum()), "preguntas y la región en",
      int((parsed["regiones"] != "").sum()), "| período sin el día del gold:",
      parsed.loc[parsed["período ∋ día del gold"] == False, "id"].tolist())  # noqa: E712

# %% [markdown]
r"""
## 6 · Las filas

Cada fila acumulativa agrega **una** pieza a la anterior (`diff_rows` lo verifica). Todas entregan hasta 3.000 tokens del generador.

Cada resultado se guarda crudo en `ART/results/03_retrieval.jsonl` (`store.ResultsLog`): los rankings deduplicados de cada canal, lo entregado, el filtro, los puntajes y la latencia. El `run_spec` lleva la configuración, la key del eval set, la key del índice y el sha de `retrieval.py`: si cambia cualquiera de esas cosas, la fila se recalcula, nunca se reusa.

**Chequeo A/A:** `dense` se corre dos veces y tiene que dar exactamente el mismo ranking.
"""

# %%
BASE = RetrievalConfig()
ROWS = {
    "bm25": RetrievalConfig(dense=False, lexical=True),
    "dense": BASE,
    "+prefix": RetrievalConfig(prefix=True),
    "+rrf": RetrievalConfig(prefix=True, lexical=True),
    "+rerank": RetrievalConfig(prefix=True, lexical=True, rerank=True),
    "+parent": RetrievalConfig(prefix=True, lexical=True, rerank=True, parent=True),
    "+date": RetrievalConfig(prefix=True, lexical=True, rerank=True, parent=True, date_filter=True),
    "+region": RetrievalConfig(prefix=True, lexical=True, rerank=True, parent=True, date_filter=True,
                               region_filter=True),
}
NAMES = list(ROWS)
for prev, cur in zip(NAMES[1:], NAMES[2:]):
    change = retrieval.diff_rows(ROWS[prev], ROWS[cur])
    assert change.startswith("one variable"), f"{cur}: {change}"
    print(f"{prev} -> {cur}: {change}")

# Todo el código del que depende una fila cruda: si cambia, la fila se recalcula, nunca se reusa.
CODE_SHA = store.sha256_json({name: store.sha256_text((ROOT / "src" / name).read_text(encoding="utf-8"))
                              for name in ("retrieval.py", "lexical.py", "evalset.py", "chunking.py")})[:16]
RESULTS = ART / "results" / "03_retrieval.jsonl"


def run_row(name: str, cfg: RetrievalConfig) -> dict[str, dict]:
    log = store.ResultsLog(RESULTS, run_spec={"config": asdict(cfg), "eval": ev.key(), "index": INDEX_KEY,
                                              "code": CODE_SHA})
    for item in ITEMS:
        if not log.done(name, item.item_id):
            result = retrieval.retrieve(item.question, cfg, INDEX)
            log.write({"arm": name, "item_id": item.item_id, **result.to_row()})
    return {row["item_id"]: row for row in log.rows(name)}


t0 = time.time()
RAW = {name: run_row(name, cfg) for name, cfg in ROWS.items()}
twin = run_row("dense·A/A", BASE)
assert all(twin[i]["sections"] == RAW["dense"][i]["sections"] and twin[i]["delivered"] == RAW["dense"][i]["delivered"]
           for i in twin), "A/A: dense no se repite"
print(f"{len(ROWS)} filas × {len(ITEMS)} preguntas en {time.time() - t0:.0f} s; A/A: rankings idénticos")

# La subida a padres conserva el orden: +parent cambia la entrega, nunca el ranking.
assert all(RAW["+parent"][i]["sections"] == RAW["+rerank"][i]["sections"] for i in RAW["+rerank"]), \
    "+parent cambió el ranking de secciones"

# La ventana del reranker, con los pares que leyó de verdad (consulta + hijo con prefijo).
question = {item.item_id: item.question for item in ITEMS}
pairs = [(question[i], searchable_text(UNITS[u], True)) for i, row in RAW["+rerank"].items() for u in row["reranked"]]
pair_lengths = [len(e.ids) for e in RERANK_TOK.encode_batch(pairs)]  # con los tokens especiales del par
over = sum(n > 512 for n in pair_lengths)
print(f"pares (consulta + hijo) leídos por el reranker: {len(pairs):,}; de más de 512 tokens: {over} "
      f"({over / len(pairs):.1%}); el más largo, {max(pair_lengths)}")
assert over == 0, "el reranker truncaría pares en silencio"

# %% [markdown]
r"""
## 7 · La tabla, contra el ruido

Sobre las preguntas **con respuesta de test** (las de fuera del corpus van en la sección 9).
- **Rankings deduplicados al nivel que se mide.** Documento (hit@k y MRR@10), bloque (el artículo en cualquier versión) y sección. Un acierto de sección cuenta el gold o una versión del mismo bloque con texto idéntico; `secc@10 estricto` cuenta solo el id gold.
- **Lo entregado:**
  - `clave desde gold` es la **primaria**;
  - `secc entregada`: algún pasaje de una sección gold;
  - `answer-delivered`: la clave en cualquier pasaje;
  - los tokens entregados y los caracteres repetidos por el solape entre hijos.
- **El ruido.** Cada fila contra la anterior, sobre la primaria: victorias / derrotas / empates, el test de signos exacto (= McNemar exacto), el bootstrap del 95% de la fila y el de la diferencia (los mismos remuestreos para todas las filas).
"""

# %%
GOLD_TEXTS = {item.item_id: evaluation.gold_texts(ACCEPTED[item.item_id], UNITS) for item in ANSWERABLE}
records = []
for name in ROWS:
    for item in ANSWERABLE:
        row = RAW[name][item.item_id]
        metrics = evaluation.item_metrics(row, item, ACCEPTED[item.item_id], GOLD_TEXTS[item.item_id], UNITS)
        delivered = [u for u, _ in row["delivered"]]
        records.append({"fila": name, "id": item.item_id, "split": item.split, "categoría": item.category, **metrics,
                        "dup_chars": evaluation.duplicated_chars(delivered, UNITS),
                        "chars": sum(UNITS[u].end - UNITS[u].start for u in delivered),
                        "ms": row["latency_ms"]["total"]})
M = pd.DataFrame(records)
TEST = M[M["split"].isin(REPORTED)]
TEST_IDS = sorted(TEST["id"].unique())
hit = evaluation.hit


def summary(frame: pd.DataFrame) -> dict:
    return {
        "n": len(frame),
        "doc@1": sum(hit(r, 1) for r in frame["doc_rank"]), "doc@5": sum(hit(r, 5) for r in frame["doc_rank"]),
        "doc@10": sum(hit(r, 10) for r in frame["doc_rank"]), "MRR@10": round(frame["mrr10"].mean(), 2),
        "bloque@10": sum(hit(r, 10) for r in frame["block_rank"]),
        "secc@1": sum(hit(r, 1) for r in frame["section_rank"]), "secc@5": sum(hit(r, 5) for r in frame["section_rank"]),
        "secc@10": sum(hit(r, 10) for r in frame["section_rank"]),
        "secc@10 estricto": sum(hit(r, 10) for r in frame["section_rank_strict"]),
        "clave desde gold": int(frame["gold_key_delivered"].sum()),
        "clave desde gold (estricto)": int(frame["gold_key_delivered_strict"].sum()),
        "secc entregada": int(frame["section_delivered"].sum()),
        "answer-delivered": int(frame["answer_delivered"].sum()),
        "tokens (media)": int(frame["tokens"].mean()), "tokens (mediana)": int(frame["tokens"].median()),
        "pasajes": round(frame["n_delivered"].mean(), 1),
        "solape %": round(100 * frame["dup_chars"].sum() / max(frame["chars"].sum(), 1), 1),
        "ms (mediana)": int(frame["ms"].median()),
    }


TABLE = pd.DataFrame({name: summary(TEST[TEST["fila"] == name]) for name in ROWS}).T
display(TABLE)

by_category = TEST.pivot_table(index="fila", columns="categoría", values="gold_key_delivered", aggfunc="sum",
                               sort=False).reindex(NAMES)
by_category["total"] = by_category.sum(axis=1)
print("primaria (clave desde gold) por categoría, en test:")
display(by_category)

means = TABLE["tokens (media)"]
if means.max() > 1.1 * means.min():
    print(f"ADVERTENCIA: los tokens entregados difieren en más de ±10% entre filas "
          f"({means.idxmin()}: {means.min()}, {means.idxmax()}: {means.max()}). Se esperaba en +parent "
          "(los padres son unidades grandes y el presupuesto se llena a saltos).")

PRIMARY = {name: TEST[TEST["fila"] == name].set_index("id")["gold_key_delivered"].reindex(TEST_IDS)
           for name in ROWS}
BOOT = evaluation.bootstrap_indices(len(TEST_IDS), n_boot=10_000, seed=0)
noise = []
for prev, cur in zip(NAMES, NAMES[1:]):
    wins, losses, ties = evaluation.paired(PRIMARY[prev].to_dict(), PRIMARY[cur].to_dict())
    low, high = evaluation.interval(PRIMARY[cur].astype(int), BOOT)
    d_low, d_high = evaluation.interval(PRIMARY[cur].astype(int) - PRIMARY[prev].astype(int), BOOT)
    label = f"{cur} vs {prev}" + (" (líneas de base: cambian dos flags)" if prev == "bm25" else "")
    noise.append({"comparación": label, "aciertos": f"{int(PRIMARY[prev].sum())} → {int(PRIMARY[cur].sum())}",
                  "victorias": wins, "derrotas": losses, "empates": ties,
                  "p (signos)": round(evaluation.sign_test(wins, losses), 3),
                  "IC95 de la fila": f"[{low:.0%}, {high:.0%}]", "IC95 de la diferencia": f"[{d_low:+.0%}, {d_high:+.0%}]"})
NOISE = pd.DataFrame(noise)
display(NOISE)

# %% [markdown]
r"""
## 8 · Abrir la caja en el movimiento más grande

El par de filas consecutivas con más preguntas que cambiaron de resultado. Para cada una se muestra el rango del gold en cada canal, antes de fusionar (denso, BM25, fusión y después del rerank), y las 3 primeras secciones de cada canal. La pregunta es **por qué** ganó (o perdió) la pieza, no solo si ganó.

RRF supone que los canales fallan de forma independiente: si los dos caen en los mismos pasajes equivocados, el consenso entierra el correcto.
"""

# %%
moves = {(prev, cur): sum(PRIMARY[prev][i] != PRIMARY[cur][i] for i in TEST_IDS) for prev, cur in zip(NAMES, NAMES[1:])}
BIGGEST = max(moves, key=moves.get)
print(f"movimiento más grande: {BIGGEST[0]} -> {BIGGEST[1]} ({moves[BIGGEST]} preguntas cambiaron)\n")


def title(section_id: str) -> str:
    doc, section = SECTIONS[section_id]
    return f"{chunking.norm_label(doc)} · {section.title[:60]} · {section.valid_from}"


for item in [i for i in ANSWERABLE if i.item_id in TEST_IDS and PRIMARY[BIGGEST[0]][i.item_id] != PRIMARY[BIGGEST[1]][i.item_id]]:
    print(f"== {item.item_id} [{item.category}] {item.question}")
    print(f"   gold: {item.gold_sections} | clave: {item.answer_key}")
    for name in BIGGEST:
        row = RAW[name][item.item_id]
        ranks = {ch: evalset.first_rank(ranking, ACCEPTED[item.item_id]) for ch, ranking in row["channels"].items()}
        print(f"   {name}: clave desde gold = {PRIMARY[name][item.item_id]} | rango del gold por canal: {ranks}")
        for channel, ranking in row["channels"].items():
            print(f"      {channel:9} " + " | ".join(title(s) for s in ranking[:3]))
    print()

# %% [markdown]
r"""
## 9 · Fuera del corpus: ¿puede la recuperación sola decir que la respuesta no está?

El puntaje máximo de cada pregunta, en la fila `+rerank`: el coseno del denso (con prefijo) y el logit del reranker. Se comparan las preguntas fuera del corpus (test + dev, que son pocas) con las que tienen respuesta. No se ajusta ningún umbral: es una descripción.

BM25 no entra: su puntaje crece con el largo de la consulta, y no se compara entre preguntas.
"""

# %%
ooc = []
for item in ITEMS:
    if item.answerable and item.split not in REPORTED:
        continue
    scores = RAW["+rerank"][item.item_id]["max_scores"]
    ooc.append({"id": item.item_id, "grupo": "con respuesta (test)" if item.answerable else "fuera del corpus",
                "denso máx": scores.get("dense"), "reranker máx": scores.get("rerank")})
OOC = pd.DataFrame(ooc)
display(OOC.groupby("grupo")[["denso máx", "reranker máx"]].describe().round(2).T)
display(OOC[OOC["grupo"] == "fuera del corpus"])

# %% [markdown]
r"""
## 10 · La mejor configuración (la regla pre-registrada)

1. La fila con más aciertos en la primaria, en test.
2. Si una fila con menos piezas no se distingue de ella (test de signos, p ≥ 0,05), gana la de menos piezas.
3. Si `topical` e `identifier` discrepan, manda `topical` (pre-registro de la Etapa 2). La tabla por categoría de la sección 7 lo muestra.

La elección se hace sobre test, y así se dice en el informe. Queda en `ART/index/best_config.json` para la Etapa 4.
"""

# %%
PIECES = ("dense", "prefix", "lexical", "rerank", "parent", "date_filter", "region_filter")
n_pieces = {name: sum(getattr(cfg, piece) for piece in PIECES) for name, cfg in ROWS.items()}
top = max(NAMES, key=lambda name: (PRIMARY[name].sum(), -n_pieces[name]))
tied = []
for name in NAMES:
    wins, losses, _ = evaluation.paired(PRIMARY[top].to_dict(), PRIMARY[name].to_dict())
    if evaluation.sign_test(wins, losses) >= 0.05:
        tied.append(name)
topical = by_category["topical"] if "topical" in by_category else pd.Series(0, index=NAMES)
# Entre las que no se distinguen: la de menos piezas; con las mismas piezas, la de más topical (pre-registro de la
# Etapa 2: si topical e identifier discrepan, manda topical) y después la de más aciertos. Las líneas de base
# también pueden ganar: si ninguna pieza paga, la mejor configuración es la más simple.
BEST = min(tied, key=lambda name: (n_pieces[name], -topical[name], -PRIMARY[name].sum()))
display(by_category.loc[tied])
print(f"más aciertos: {top} ({int(PRIMARY[top].sum())}/{len(TEST_IDS)}); no se distinguen de ella: {tied}")
print(f"mejor configuración por la regla: {BEST} ({int(PRIMARY[BEST].sum())}/{len(TEST_IDS)}, "
      f"{n_pieces[BEST]} piezas) -> {asdict(ROWS[BEST])}")
store.write_json(INDEX_DIR / "best_config.json", {
    "row": BEST, "config": asdict(ROWS[BEST]), "index": INDEX_KEY, "eval": ev.key(),
    "rule": "max gold_key_delivered on test; fewest pieces among rows with sign-test p >= 0.05 against it",
    "primary_on_test": {name: int(PRIMARY[name].sum()) for name in NAMES}, "n_test": len(TEST_IDS),
})

run.note("n_children", len(CHILDREN))
run.note("n_parents", len(PARENTS))
run.note("embed_seconds", embed_seconds)
run.note("index_key", INDEX_KEY)
run.note("primary_on_test", {name: int(PRIMARY[name].sum()) for name in NAMES})
run.note("best_row", BEST)
run.finish()

# %% [markdown]
r"""
## Para el informe

- ¿Qué piezas se ganaron su costo en este corpus, por encima de la cota de ruido, y cuáles sacamos? (tabla de ruido de la sección 7)
- Para el movimiento más grande, ¿qué mostró abrir la caja? ¿Ganó por la razón que da su paper? (sección 8)
- ¿Qué tan separados están el documento, la sección y la clave entregada, y qué predice eso para la Etapa 4?
- Con tokens igualados, ¿conviene entregar muchos hijos o pocos padres? (`+rerank` contra `+parent`)
- ¿Ayudó el filtro de metadatos, y en qué categoría? ¿Cuánto fue la regla "sin fecha, lo vigente" y cuánto la fecha parseada? (`+date`, y la sección 5)
- ¿Puede la recuperación sola decir que la respuesta no está? (sección 9)
"""
