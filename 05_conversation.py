# %% [markdown]
r"""
# Etapa 5 · La conversación: una ventana, dos fuentes

La API no tiene estado: todo lo que el modelo "recuerda" dentro de una sesión es lo que el código vuelve a poner en la ventana en cada turno. Con una conversación, dos cosas compiten por esa ventana: los pasajes del corpus y el historial. Cada turno tiene las mismas dos superficies de falla que el RAG de un solo turno:
- **con qué busca el recuperador** (Parte A);
- **qué ve el generador** (Parte B, el experimento central).

| Entra | Sale |
|---|---|
| los guiones congelados (`eval/dialogues.jsonl`), el índice y la mejor configuración de la Etapa 3 (`ART/index/best_config.json`) y el ensamblador de `grounded.py` | `results/05_rewrites.jsonl`, `05_queries.jsonl`, `05_history.jsonl`, `05_summary_probe.jsonl`, `05_summaries.jsonl` |

La lógica vive en `src/conversation.py`. El recuperador es el de la Etapa 3 (`index_loader`, `retrieval`), el ensamblador, la llamada y el puntaje son los de la Etapa 4 (`grounded.py`), y las cotas de ruido, las de `stats.py` y `evaluation.py`.

**Decidido antes de correr** (todo en `decision_log.md`):
- **Cada guion se reproduce solo.** El historial de un turno es lo guionado antes en su propio guion, y el turno 1 no tiene historial. Así lo que se recorta es información del propio guion, y no se mezclan datos de otras personas.
- **El historial es siempre el guionado**: la pregunta del usuario y la respuesta del asistente que se escribieron en la Etapa 2. Todos los brazos ven el mismo historial y solo se genera el turno actual. Si cada brazo se realimentara con sus respuestas, divergirían desde el turno 2.
- **Nunca entra un `analysis` al historial.** El resumen tampoco tiene campo de razonamiento.
- `qwen3:8b`, `think=False`, temperatura 0, `num_ctx = 8.192` y `num_predict = 768` en cada respuesta, igual que en la Etapa 4. La reescritura y el resumen (256) cambian solo `num_predict`, que no recarga el modelo.
- **El ruido se acota por guion, no por turno.** La p de cada comparación es la de un test de signos con un voto por guion (`stats.paired_groups`), y el bootstrap remuestrea guiones.
- Se reporta sobre los 8 guiones de `test` (42 turnos).

**Fases para la GPU** (8 GB: el generador no convive con el embedder y el reranker):
1. las reescrituras, con el LLM;
2. se descarga el LLM, se carga el índice y se recupera en lote para las cuatro consultas de cada turno;
3. el embedder y el reranker pasan a CPU;
4. la generación de la Parte B.

Como el historial es el guionado, ninguna consulta depende de lo que genera el modelo: todo se recupera antes de generar.
"""

# %%
import environment  # el del curso, primero: guarda de JAX, caché de HF, UTF-8

import os
import random
import sys
import time
from dataclasses import asdict

ROOT = environment.ROOT
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

try:
    from IPython.display import display  # en la notebook
except ImportError:
    display = print  # como script (tools/run_stages.py)

import llm
import store
from console import StageRun, clip, wrap

import artifact_io
import conversation
import doc_schema
import evalset
import evaluation
import grounded
import index_loader
import stats
from chunking import searchable_text
from conversation import ARMS, COMPARISONS, QUERY_TYPES, REFERENCE, HistoryState
from corpus import SNAPSHOT

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
ART = environment.artifacts_dir(SMOKE_TEST)
EVAL_DIR = ROOT / "eval" / "smoke" if SMOKE_TEST else ROOT / "eval"
DOCS_PATH = ART / "documents.jsonl"
DOCS = doc_schema.load_documents(DOCS_PATH)
CORPUS_SHA = doc_schema.documents_hash(DOCS)
artifact_io.check_manifest(ART / "documents.manifest.json", {"corpus_sha": CORPUS_SHA})
if SMOKE_TEST:  # el smoke nunca se congela: se valida contra su corpus
    ev = evalset.load_eval(EVAL_DIR, require_frozen=False)
    evalset.validate_eval(ev, DOCS).show().raise_if_failed()
else:
    ev = evalset.load_eval(EVAL_DIR, DOCS_PATH)
SECTIONS = evalset.section_index(DOCS)

MODEL, NUM_CTX, NUM_PREDICT = grounded.MODEL, grounded.NUM_CTX, grounded.NUM_PREDICT
TOTAL = conversation.TOTAL_BUDGET
REPORT_DIALOGUES = [d for d in ev.dialogues if SMOKE_TEST or d.split == "test"]

# Cada guion por separado: el historial de un turno es lo guionado antes, en su guion.
TURNS, HISTORY, SPLIT = {}, {}, {}
for dialogue in ev.dialogues:
    turns = conversation.replay(dialogue)
    for index, rt in enumerate(turns):
        TURNS[rt.turn_id] = rt
        HISTORY[rt.turn_id] = conversation.scripted_history(turns[:index])
        SPLIT[rt.turn_id] = dialogue.split
REPORT = [rt for d in REPORT_DIALOGUES for rt in conversation.replay(d)]
GROUPS = {turn_id: rt.dialogue_id for turn_id, rt in TURNS.items()}  # para remuestrear guiones

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)  # como script, pandas corta las tablas anchas
environment.summary()
print(f"corpus_sha {CORPUS_SHA[:12]}... | eval {ev.key()} | {ev.summary()} | "
      f"se reporta sobre {len(REPORT_DIALOGUES)} guiones ({len(REPORT)} turnos)")
run = StageRun.start("05_conversation", ART, profile=ART.name,
                     config={"model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "total_budget": TOTAL,
                             "arms": [a.name for a in ARMS], "eval": ev.key()})

# %% [markdown]
r"""
## 1 · Fase 1: las reescrituras

Para la Parte A, cada turno tiene cuatro consultas. Tres no llaman al LLM:

| Consulta | Qué es |
|---|---|
| `last_turn` | el último turno del usuario, tal cual |
| `full_conversation` | la pregunta actual y las anteriores del usuario, **la más nueva primero**, acotada a la ventana del modelo que la lee (sección 2) |
| `rewrite` | una reescritura del último turno hecha por el LLM a partir del guion hasta ese turno (a lo sumo 5 pares), con un schema de un solo campo (`{"query": str}`) |
| `oracle` | la reescritura oráculo de la Etapa 2 (`oracle_rewrite`) |

- **La conversación completa va al revés.** Los modelos truncan por la derecha. En orden cronológico se perdería justo la pregunta actual, que es lo único que dice qué se está preguntando. Solo entran turnos del usuario: las respuestas del asistente repetirían los términos de la pregunta anterior.
- **El reescritor ve el guion entero hasta el turno actual** (a lo sumo 5 pares, ~330 tokens). Los turnos `return` vuelven al turno 1: con solo los últimos pares no sabría a qué se vuelve.
- **El primer turno de cada guion no se reescribe.** Sin historial no hay nada que resolver, así que va tal cual y sin llamada.
- **Una reescritura vacía es un error del reescritor**: queda fuera de su columna y se cuenta aparte. Una que llega al tope de 300 caracteres está cortada (la gramática corta sin avisar): se usa, pero se cuenta como sospechosa.
"""

# %%
SERVER = llm.setup(MODEL, num_ctx=NUM_CTX)
llm.use_cache(ART / "llm_cache" / "qwen3_8b.jsonl")
run.note("server", {k: SERVER.get(k) for k in ("digest", "server", "key_order", "thinking", "calibration")})

REWRITE_SPEC = {"stage": "05A", "eval": ev.key(), "model": MODEL, "num_ctx": NUM_CTX,
                "num_predict": conversation.REWRITE_NUM_PREDICT, "history_turns": conversation.REWRITE_HISTORY_TURNS,
                "system": store.sha256_text(conversation.REWRITE_SYSTEM),
                "schema": store.sha256_json(conversation.Rewrite.model_json_schema())[:16]}
rewrites_log = store.ResultsLog(ART / "results" / "05_rewrites.jsonl", run_spec=REWRITE_SPEC)
llm.unload(MODEL)  # arranque en frío: ver decision_log.md, «Temperatura 0 no alcanza»
t0 = time.time()
for turn_id, rt in TURNS.items():
    if rewrites_log.done("rewrite", turn_id):
        continue
    query, call = conversation.rewrite(HISTORY[turn_id], rt.turn.user)
    rewrites_log.write({"arm": "rewrite", "item_id": turn_id, "dialogue_id": rt.dialogue_id, "turn": rt.number,
                        "query": query, "call": call, "error": (call or {}).get("error"),
                        "at_maxlength": query is not None and len(query) >= conversation.REWRITE_MAX})
REWRITE_ROWS = {r["item_id"]: r for r in rewrites_log.rows("rewrite")}
REWRITTEN = {turn_id: r["query"] for turn_id, r in REWRITE_ROWS.items()}
with_call = [r for r in REWRITE_ROWS.values() if r["call"] is not None]
print(f"{len(REWRITTEN)} turnos · {len(with_call)} reescrituras con llamada · "
      f"{sum(r['query'] is None for r in with_call)} fallidas · "
      f"{sum(r['at_maxlength'] for r in with_call)} en el tope de {conversation.REWRITE_MAX} caracteres · "
      f"{time.time() - t0:.0f} s")
run.note("rewrites", {"calls": len(with_call), "failed": sum(r["query"] is None for r in with_call),
                      "at_maxlength": sum(r["at_maxlength"] for r in with_call)})

# %% [markdown]
r"""
## 2 · Fase 2: el índice de la Etapa 3 y la recuperación en lote

Se descarga el LLM y se carga el índice con la mejor configuración de la Etapa 3. `index_loader.load` verifica que sea el mismo índice que se midió. Antes de recuperar nada, se verifica que cada sección gold de los guiones esté en el índice (§4).

**Cada consulta se acota a la ventana del modelo que la lee.** Así nada se trunca en silencio:
- **Con reranker** manda su ventana. El par (consulta + hijo) tiene que entrar en 512 tokens, y la Etapa 3 dimensionó los hijos en hasta 350 tokens del reranker contando con consultas cortas. Toda consulta se acota a 512 − 350 − los especiales del par, medidos en su tokenizer.
- **Sin reranker** manda el `max_seq_length` del embedder.

`full_conversation` pierde primero los turnos más viejos. Las otras tres se recortan conservando el final, donde suele estar la pregunta. Se cuenta cuántas consultas se recortaron por variante, y al final se verifica la ventana del reranker con los pares que leyó de verdad (paso 8 de la Etapa 3).

**El filtro de fecha y región**, si la configuración lo usa, se parsea de la consulta. Con `last_turn`, un turno como "¿y si fuera un local?" pierde la fecha que el usuario dijo en el turno 1, y el filtro cae en lo vigente hoy. Eso es parte de lo que mide la Parte A: la tabla muestra en cuántos turnos cada consulta trajo una fecha.

La entrega con la que se mide "le llegó la respuesta" (`gold_key_delivered`, la primaria de la Etapa 3) es la de la Parte B: 1.350 tokens, los pasajes con el tope del experimento 2 (150) lleno. Es fija para las cuatro consultas.
"""

# %%
llm.unload(MODEL)  # libera la VRAM para el embedder y el reranker
t0 = time.time()
LOADED = index_loader.load(ART, DOCS, ROOT, ev.key())
UNITS = LOADED.units
CFG = LOADED.config
print(f"índice {LOADED.key} · mejor fila de la Etapa 3: {LOADED.best['row']} · {asdict(CFG)} · "
      f"{time.time() - t0:.0f} s")
indexed = {unit.section_id for unit in LOADED.index.children}
missing = sorted({sid for rt in TURNS.values() for sid in rt.turn.gold_sections} - indexed)
assert not missing, f"secciones gold que no están en el índice: {missing}"
GOLD = {turn_id: grounded.Gold.of(rt.turn, SECTIONS, UNITS) for turn_id, rt in TURNS.items()}

reranker, embedder = LOADED.index.reranker, LOADED.index.embedder
LIMIT = conversation.query_limit(CFG.rerank, reranker.tokenizer if reranker else None, embedder.tokenizer,
                                 (reranker.max_seq_length if reranker else None) or 512, embedder.max_seq_length)
OFFSETS = conversation.hf_offsets(reranker.tokenizer if CFG.rerank else embedder.tokenizer)
print(f"consultas acotadas a {LIMIT} tokens del {'reranker' if CFG.rerank else 'embedder'}")

QUERIES: dict[str, dict[str, str | None]] = {qt: {} for qt in QUERY_TYPES}
TRUNCATED: dict[str, dict[str, bool]] = {qt: {} for qt in QUERY_TYPES}
for turn_id, rt in TURNS.items():
    raw = {"last_turn": rt.turn.user, "full_conversation": rt.turn.user, "rewrite": REWRITTEN.get(turn_id),
           "oracle": rt.turn.oracle_rewrite}
    for qt in QUERY_TYPES:
        if raw[qt] is None:  # la reescritura falló
            QUERIES[qt][turn_id] = None
            continue
        QUERIES[qt][turn_id], TRUNCATED[qt][turn_id] = conversation.bound_query(qt, raw[qt], HISTORY[turn_id],
                                                                                LIMIT, OFFSETS)
print("consultas recortadas, por variante (turnos de test):",
      {qt: f"{sum(TRUNCATED[qt].get(rt.turn_id, False) for rt in REPORT)}/{len(REPORT)}" for qt in QUERY_TYPES})

count = grounded.token_counter(MODEL)
RANKED: dict[str, dict[str, list]] = {qt: {} for qt in QUERY_TYPES}  # las unidades rankeadas, para la Parte B
QUERY_SPEC = {"stage": "05A", "eval": ev.key(), "index": LOADED.key, "config": asdict(CFG),
              "best_row": LOADED.best["row"], "queries": store.sha256_json(QUERIES), "query_limit": LIMIT,
              "part_a_budget": conversation.PART_A_BUDGET}
queries_log = store.ResultsLog(ART / "results" / "05_queries.jsonl", run_spec=QUERY_SPEC)
pairs = []  # (consulta, hijo) que leyó el reranker, para verificar su ventana
t0 = time.time()
for qt in QUERY_TYPES:
    for turn_id, rt in TURNS.items():
        query = QUERIES[qt][turn_id]
        if query is None:
            if not queries_log.done(qt, turn_id):
                queries_log.write({"arm": qt, "item_id": turn_id, "dialogue_id": rt.dialogue_id, "turn": rt.number,
                                   "depends_on": rt.turn.depends_on, "split": SPLIT[turn_id], "query": None,
                                   "metrics": None, "error": "REWRITE_FAILED"})
            continue
        # Se recupera siempre (es determinista y en lote): la Parte B necesita las unidades aunque la fila
        # ya esté escrita.
        units, result = index_loader.ranked_units(LOADED, query)
        RANKED[qt][turn_id] = units
        pairs += [(query, searchable_text(LOADED.index.by_id[u], CFG.prefix)) for u in result.reranked]
        if queries_log.done(qt, turn_id):
            continue
        delivered = grounded.deliver(units, conversation.PART_A_BUDGET, count)
        row = result.to_row() | {"delivered": [[u.unit_id, u.n_gen] for u in delivered]}
        metrics = (evaluation.item_metrics(row, rt.turn, GOLD[turn_id].accepted, GOLD[turn_id].texts, UNITS)
                   if rt.turn.answerable else None)
        queries_log.write({"arm": qt, "item_id": turn_id, "dialogue_id": rt.dialogue_id, "turn": rt.number,
                           "depends_on": rt.turn.depends_on, "split": SPLIT[turn_id], "query": query,
                           "truncated": TRUNCATED[qt][turn_id], **row, "metrics": metrics, "error": None})
print(f"{sum(len(r) for r in RANKED.values())} recuperaciones en {time.time() - t0:.0f} s")

if CFG.rerank:
    tok = reranker.tokenizer
    lengths = [len(ids) for ids in tok([q for q, _ in pairs], [c for _, c in pairs])["input_ids"]] if pairs else []
    print(f"pares (consulta + hijo) leídos por el reranker: {len(lengths):,}; el más largo, {max(lengths, default=0)} "
          f"tokens; de más de 512: {sum(n > 512 for n in lengths)}")
    assert not any(n > 512 for n in lengths), "el reranker truncaría pares en silencio"

# %% [markdown]
r"""
## 3 · Fase 3: la GPU queda para el generador

El embedder y el reranker pasan a CPU. Toda la recuperación de la etapa ya está hecha.
"""

# %%
LOADED.release_gpu()
environment.gpu_report("después de liberar el índice")

# %% [markdown]
r"""
## 4 · Las tablas de la Parte A

Sobre los turnos con respuesta de los guiones de test, con las métricas de la Etapa 3 (`evaluation.item_metrics`):
- **Rankings deduplicados al nivel que se mide.** Un acierto de sección cuenta el gold o una versión del mismo bloque con texto idéntico.
- hit@k de **documento** y de **sección** (k = 1, 5, 10), y MRR@10 de documento.
- **`clave desde gold`**: un pasaje del gold con la clave le llegó al generador, con los 1.350 tokens de la Parte B.
- hit@5 de documento y de sección **por etiqueta de dependencia** y **por número de turno**, que son las dos preguntas del informe.
- Si la configuración filtra por fecha: en cuántos turnos la consulta trajo una fecha.

Por etiqueta hay entre 3 y 10 turnos por celda: se leen como dirección, no como prueba.
"""

# %%
REPORT_IDS = {rt.turn_id for rt in REPORT}
A_ROWS = [r for r in queries_log.rows() if r["item_id"] in REPORT_IDS]
A = pd.DataFrame([{"arm": r["arm"], "item_id": r["item_id"], "dialogue_id": r["dialogue_id"], "turn": r["turn"],
                   "depends_on": r["depends_on"], "error": r["error"], **(r["metrics"] or {})}
                  for r in A_ROWS if TURNS[r["item_id"]].turn.answerable])
for column in ("doc_rank", "section_rank"):
    A[column] = pd.to_numeric(A[column])  # None -> NaN: "no apareció en el ranking"


def hits(frame: pd.DataFrame, column: str, k: int) -> int:
    return int(frame[column].apply(lambda rank: evaluation.hit(None if pd.isna(rank) else int(rank), k)).sum())


overall = []
for qt in QUERY_TYPES:
    frame = A[(A["arm"] == qt) & A["error"].isna()]
    overall.append({"consulta": qt, "n": len(frame), "errores": int((A["arm"] == qt).sum() - len(frame)),
                    **{f"doc@{k}": hits(frame, "doc_rank", k) for k in (1, 5, 10)},
                    **{f"sección@{k}": hits(frame, "section_rank", k) for k in (1, 5, 10)},
                    "MRR@10 doc": round(float(frame["mrr10"].mean()), 2) if len(frame) else None,
                    "clave desde gold": int(frame["gold_key_delivered"].sum()),
                    "recortadas": sum(TRUNCATED[qt].get(rt.turn_id, False) for rt in REPORT)})
display(pd.DataFrame(overall).set_index("consulta"))

if CFG.date_filter or CFG.region_filter:
    dated = {qt: sum(((r.get("filters") or {}).get("period_source") == "question") for r in A_ROWS if r["arm"] == qt)
             for qt in QUERY_TYPES}
    print(f"turnos en que la consulta trajo una fecha para el filtro (de {len(REPORT)}):", dated)


def by(column: str) -> pd.DataFrame:
    cells = {}
    for qt in QUERY_TYPES:
        frame = A[(A["arm"] == qt) & A["error"].isna()]
        for level, rank in (("doc", "doc_rank"), ("sección", "section_rank")):
            cells[(qt, f"{level}@5")] = {value: f"{hits(group, rank, 5)}/{len(group)}"
                                         for value, group in frame.groupby(column)}
    return pd.DataFrame(cells)


print("hit@5 por dependencia")
display(by("depends_on"))
print("hit@5 por número de turno")
display(by("turn"))

# %% [markdown]
r"""
### El ruido de la Parte A

En sección@5: cada consulta contra `last_turn`, y el oráculo contra la reescritura del modelo (Ye et al., Apéndice B.3: la reescritura humana no siempre es la mejor consulta).
- **La p es la del test de signos con un voto por guion** (`stats.paired_groups`): un guion gana si una consulta acierta más turnos que la otra. Con 8 guiones, lo mejor posible es 8-0 (p = 0,008).
- Las victorias y derrotas **por turno** se muestran como descripción. No llevan p: los turnos de un guion no son independientes.
- El intervalo de la diferencia es un bootstrap que remuestrea guiones.
"""


# %%
def section_hit_map(qt: str, k: int = 5) -> dict[str, bool]:
    frame = A[(A["arm"] == qt) & A["error"].isna()]
    return {row.item_id: evaluation.hit(None if pd.isna(row.section_rank) else int(row.section_rank), k)
            for row in frame.itertuples()}


def compare(a: str, b: str, ca: dict[str, bool], cb: dict[str, bool]) -> dict:
    turns, groups = stats.paired(ca, cb), stats.paired_groups(ca, cb, GROUPS)
    diff, lo, hi = stats.bootstrap_diff_ci(ca, cb, GROUPS)
    return {"comparación": f"{b} vs {a}",
            "por turno (descriptivo)": f"{turns['wins']}-{turns['losses']} ({turns['ties']} empates, n={turns['n']})",
            "por guion": f"{groups['wins']}-{groups['losses']} ({groups['ties']} empates)",
            "p (signos, por guion)": round(groups["p_sign"], 4),
            "diferencia": f"{diff:+.0%} ({lo:+.0%} a {hi:+.0%})"}


noise_a = [compare("last_turn", qt, section_hit_map("last_turn"), section_hit_map(qt)) for qt in QUERY_TYPES[1:]]
noise_a.append(compare("rewrite", "oracle", section_hit_map("rewrite"), section_hit_map("oracle")))
display(pd.DataFrame(noise_a).set_index("comparación"))

# %% [markdown]
r"""
**Lectura** (37 turnos con respuesta de 8 guiones de test; el ruido se cuenta por guion).
- **La clave entregada desde el gold:** `last_turn` 17, `rewrite` 16, `full_conversation` 13 y el oráculo 18. **Ninguna diferencia supera el ruido:** el mejor resultado por guion es oráculo contra `last_turn`, 4-0, con p = 0,125.
- **Las dos métricas no coinciden.** En sección@5 gana `rewrite` (16 contra 15 de `last_turn`); en clave entregada, `last_turn`. Con estos n, las dos son la misma cosa medida con ruido.
- **Por dependencia** (direcciones): `full_conversation` es la mejor a nivel de documento en las correferencias y las elipsis (9/10 y 5/5 a doc@5), pero a nivel de sección la ventaja casi desaparece (3/10 y 1/5), y en los cambios de tema es la peor (0/6 a sección@5): encuentra la norma de la que se viene hablando más que el artículo nuevo. `last_turn` falla justo donde se espera: el turno 2, la primera correferencia (2/8 a doc@5, contra 7/8 de `full_conversation`).
- **El oráculo no es mucho mejor que las del modelo** (18 contra 16-17): coincide con Ye et al. (Apéndice B.3), la reescritura humana no es necesariamente la mejor consulta.
- **Ninguna consulta se recortó** por la ventana del embedder: los guiones son cortos (la consulta más larga no llega a 510 tokens).
"""

# %% [markdown]
r"""
### Veinte reescrituras, para leer una por una (paso 3)

Se eligen 20 con una semilla fija, repartidas entre etiquetas de dependencia. Por cada una, anotar si:
- **responde la pregunta** en lugar de reescribirla;
- **agrega detalles** que no estaban en la conversación;
- **incluye razonamiento**.

Cualquiera de las tres dirige la recuperación como HyDE, por razones ajenas a reescribir. Los conteos y los ejemplos van a `decision_log.md` y al informe.
"""

# %%
rng = random.Random(0)
pools: dict[str, list] = {}
for rt in REPORT:
    if rt.number > 1:
        pools.setdefault(rt.turn.depends_on, []).append(rt)
for pool in pools.values():
    rng.shuffle(pool)
to_read = []
while len(to_read) < 20 and any(pools.values()):
    for dependency in sorted(pools):  # de a uno por etiqueta, en ronda
        if pools[dependency] and len(to_read) < 20:
            to_read.append(pools[dependency].pop())
for rt in to_read:
    flag = " · EN EL TOPE" if REWRITE_ROWS[rt.turn_id]["at_maxlength"] else ""
    print(f"--- {rt.turn_id} [{rt.turn.depends_on}]{flag}")
    print(wrap(f"usuario:   {rt.turn.user}"))
    print(wrap(f"reescrita: {REWRITTEN.get(rt.turn_id)}"))
    print(wrap(f"oráculo:   {rt.turn.oracle_rewrite}"))

# %% [markdown]
r"""
**Lectura de las 20 reescrituras.** El paso 3 pide buscar reescrituras que respondan la pregunta, agreguen detalles o incluyan pasos de razonamiento.

| Qué hizo | Cuántas | Cuáles |
|---|---|---|
| Resuelve la referencia, con datos del historial que sirven (la fecha, la ciudad, la edad) | 15 | dl01.t2, dl01.t4, dl01.t6, dl03.t3, dl03.t5, dl04.t3, dl05.t4, dl06.t4, dl06.t5, dl07.t2, dl07.t5, dl08.t2, dl08.t4, dl08.t5, dl02.t5 |
| **Agrega contenido de la ley que nadie dijo** (el efecto HyDE) | 1 | dl05.t2: "los doce meses previstos en el artículo 9 bis de la LAU" |
| Mezcla mal el historial | 1 | dl03.t2: "sin ascensor, propiedad de un tercero de 72 años" |
| Arrastra el tema anterior a un cambio de tema | 1 | dl02.t3: agrega "el contrato termina este año y el dueño no quiere renovarlo" a una pregunta sobre la herencia |
| No resuelve la referencia: copia el turno | 1 | dl02.t2: "¿Con cuánta antelación tengo que decírselo?" |
| Pierde el ancla temporal | 1 | dl01.t3: la fianza de un local "según la legislación española", sin 2016 |

Ninguna incluye pasos de razonamiento: el schema de un solo campo (`{"query"}`) no deja lugar. La que agrega contenido de la ley es la trampa del enunciado: dirige la búsqueda con una respuesta que el reescritor ya "sabe". Las cinco fallas están en correferencias, elipsis y cambios de tema, donde la reescritura más tiene que hacer. Explica por qué `rewrite` no le gana a `last_turn` en la Parte A.
"""

# %% [markdown]
r"""
## 5 · La consulta de la Parte B

La Parte B usa una sola consulta, la ganadora de la Parte A, fija en todos los brazos.
- **Se elige después de leer las tablas de arriba**, y solo entre las consultas que existen en producción: el oráculo no.
- **Se elige mirando `test`.** Es la única lectura que hay: los guiones de `dev` son 2. Que la Parte B use la consulta que mejor anduvo en esos mismos guiones favorece por igual a todos sus brazos, y eso se declara en el informe.
- **La regla, fijada antes de la corrida completa** (`decision_log.md`, Etapa 5): gana la consulta con más "clave entregada desde el gold" (la primaria de la Etapa 3, con la entrega de la Parte B). En un empate gana la más barata: `last_turn`, después `full_conversation`, después `rewrite` (una llamada al LLM por turno). La celda imprime también cuál ganó en sección@5, para que una diferencia entre las dos métricas sea explícita.
"""

# %%
DEPLOYABLE = ("last_turn", "full_conversation", "rewrite")  # en orden de costo: el empate lo gana la primera


def key_delivered_map(qt: str) -> dict[str, bool]:
    frame = A[(A["arm"] == qt) & A["error"].isna()]
    return {row.item_id: bool(row.gold_key_delivered) for row in frame.itertuples()}


delivered_by_query = {qt: sum(key_delivered_map(qt).values()) for qt in DEPLOYABLE}
QUERY_FOR_B = max(DEPLOYABLE, key=lambda qt: (delivered_by_query[qt], -DEPLOYABLE.index(qt)))
best = max(DEPLOYABLE, key=lambda qt: sum(section_hit_map(qt).values()))
assert QUERY_FOR_B in DEPLOYABLE, "el oráculo no existe en producción: no puede ser la consulta de la Parte B"
print(f"clave entregada desde el gold, por consulta: {delivered_by_query} -> QUERY_FOR_B = {QUERY_FOR_B} "
      f"(en sección@5 gana {best})")
run.note("query_for_b", {"chosen": QUERY_FOR_B, "gold_key_delivered": delivered_by_query, "best_section_at_5": best})

# %% [markdown]
r"""
**Lectura.** Por la regla fijada antes de la corrida, la Parte B usa **`last_turn`**: empata arriba en clave entregada y es la más barata. La consecuencia hay que leerla en la Parte B: los turnos que dependen del historial (correferencias y elipsis) buscan sin lo que el usuario dejó implícito, así que el historial solo puede ayudar a la generación, no a la recuperación.
"""

# %% [markdown]
r"""
## 6 · Parte B · qué conservar

**El experimento.** `num_ctx` queda fijo en 8.192. Dentro de él, un total de **1.500 tokens** para pasajes + historial, dimensionado según el historial y no según la ventana: el historial más largo dentro de un guion de test son ~330 tokens. El historial tiene un **tope**, y los pasajes reciben el total menos el historial que se usó (`grounded.deliver`). Así, en cada turno, los brazos ven el mismo total, y lo único que cambia es cuánto historial se conserva. El prompt de sistema va aparte.

| Experimento | Brazos | Qué se mueve |
|---|---|---|
| 1 · el tope | `window_h100`, `window_h150`, `window_h250` | el tope del historial, con una ventana por tokens (los mensajes más nuevos que entran) |
| 2 · la compresión | `window_h150`, `summary_h150` | la estrategia, con el tope fijo en 150, donde la ventana pierde el turno 1 en la mayoría de los turnos |
| referencia | `full_buffer` | todo el historial, sin tope; fuera de las comparaciones de una variable, con su total real |

- **Los topes del experimento 1 son un gradiente dentro del propio guion.** Un par usuario-asistente mide 64 tokens en promedio. Con 100, 150 y 250, la ventana pierde el turno 1 en unos 26, 19 y 7 de los 34 turnos no-primeros. Un tope de 300 o más casi no recortaría nada: sería otro "sin tope".
- **Con 100, a veces no entra ni el último par.** Un par largo mide hasta 114 tokens, y el turno queda sin historial: la tabla de régimen lo cuenta. Por eso el resumen no se prueba con 100: no entraría junto al último par.
- **Buffer de resumen, al estilo MemGPT.** Un resumen continuo más los turnos recientes, todo dentro del tope de 150.
  - **La actualización.** Cuando resumen + recientes cruzan el 70% del tope, la mitad más vieja de los recientes sale y se incorpora al resumen con una llamada (`Summary`, sin campo de razonamiento). El último par usuario-asistente nunca sale: es el que necesitan una correferencia o una elipsis, y sin él la comparación mezclaría "comprimir" con "perder el turno anterior".
  - **El tamaño.** El resumen tiene un tope de 220 caracteres (~64 tokens, ~77 con su envoltorio): la mitad del tope para el resumen y la otra mitad para el último par, que mide 64 tokens en promedio.
  - **Dónde va.** Al final del prompt de sistema, a la cabeza de la cola de mensajes.
- **El prompt** es el de la Etapa 4 más una regla: usar la conversación para entender a qué se refiere la pregunta y los datos del usuario, pero responder desde los pasajes. Sin ella, "respondé solo con lo que dicen los pasajes" le prohibiría al modelo usar la fecha que el usuario dijo dos turnos atrás.

**El régimen.** Antes de la primera respuesta, la tabla de abajo mide cada tope contra el guion actual:
- en cuántos turnos la ventana pierde el turno 1. La ventana conserva los mensajes más nuevos, así que perder cualquier mensaje del propio guion es perder el turno 1: las dos cuentas coinciden;
- en cuántos queda **sin historial por el tope**: había historial, pero no entró ni el último par.
"""

# %%
not_first = [rt for rt in REPORT if rt.number > 1]
regime = []
for arm in ARMS:
    if arm.strategy != "window":
        continue
    losses = [row for d in REPORT_DIALOGUES for row in conversation.window_losses(conversation.replay(d), arm.cap, count)]
    lost = [row for row in losses if row["loses_turn_1"]]
    regime.append({"brazo": arm.name, "tope": arm.cap, "pierde el turno 1": f"{len(lost)}/{len(not_first)}",
                   "sin historial por el tope": f"{sum(row['empty_by_cap'] for row in losses)}/{len(not_first)}",
                   "mensajes perdidos (media, turnos no-primeros)": round(
                       sum(row["lost_messages"] for row in losses if row["turn"] > 1) / max(len(not_first), 1), 1)})
display(pd.DataFrame(regime).set_index("brazo"))
longest = max(conversation.history_tokens(HISTORY[rt.turn_id], count) for rt in REPORT)
print(f"el historial más largo dentro de un guion: {longest} tokens · total pasajes + historial: {TOTAL}")
run.note("regime", {row["brazo"]: {"loses_turn_1": row["pierde el turno 1"], "empty": row["sin historial por el tope"]}
                    for row in regime} | {"longest_history": longest})

# %% [markdown]
r"""
### Los brazos

Orden de corrida fijo (el de `conversation.ARMS`), con el modelo descargado antes de cada brazo: así cada brazo se puede reproducir solo. Cada fila de `results/05_history.jsonl` guarda:
- lo que se entregó (historial y pasajes, con nuestra cuenta);
- el resumen vigente en ese turno, y las llamadas de resumen que hicieron falta para llegar a él (se le cuentan al turno que lo necesita);
- la respuesta y su puntaje;
- el registro de la llamada, con `prompt_eval_count`;
- `error` en el primer nivel, para que una corrida interrumpida reintente los errores transitorios.
"""

# %%
SYSTEM_B = conversation.conversation_system(SNAPSHOT)
B_SPEC = {"stage": "05B", "eval": ev.key(), "corpus_sha": CORPUS_SHA, "index": LOADED.key, "config": asdict(CFG),
          "best_row": LOADED.best["row"], "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
          "total_budget": TOTAL, "arms": [vars(a) for a in ARMS], "query": QUERY_FOR_B,
          "queries": store.sha256_json(QUERIES[QUERY_FOR_B]), "query_limit": LIMIT,
          "prompt": grounded.prompt_spec(SYSTEM_B), "system": store.sha256_text(SYSTEM_B),
          "summary_system": store.sha256_text(conversation.SUMMARY_SYSTEM),
          "summary_schema": store.sha256_json(conversation.Summary.model_json_schema())[:16],
          "summary_trigger": conversation.SUMMARY_TRIGGER, "summary_num_predict": conversation.SUMMARY_NUM_PREDICT,
          "message_margin": conversation.MESSAGE_MARGIN, "dialogues": [d.dialogue_id for d in REPORT_DIALOGUES]}
history_log = store.ResultsLog(ART / "results" / "05_history.jsonl", run_spec=B_SPEC)
SUMMARY_UPDATES: list[dict] = []


def first_turn_in_view(history: list[dict], rt) -> bool:
    first = TURNS[f"{rt.dialogue_id}.t1"].turn.user
    return any(m["role"] == "user" and m["content"] == first for m in history)


t0 = time.time()
for arm in ARMS:
    llm.unload(MODEL)
    for dialogue in REPORT_DIALOGUES:
        turns = conversation.replay(dialogue)
        state = HistoryState(arm, count, conversation.summarize)
        for index, rt in enumerate(turns):
            # El resumen se pone al día con todo lo guionado antes de este turno. En una re-corrida, sus
            # llamadas salen de la caché: el estado se reconstruye igual aunque el turno ya esté hecho.
            summary_calls = state.add(turns[index - 1].exchange) if index else []
            if history_log.done(arm.name, rt.turn_id):
                continue
            summary, history, history_used = state.view()
            units = RANKED[QUERY_FOR_B].get(rt.turn_id)
            fallback = units is None  # la reescritura de este turno falló: se busca con el último turno, y se marca
            delivered = grounded.deliver(RANKED["last_turn"][rt.turn_id] if fallback else units,
                                         conversation.passage_budget(history_used), count)
            passages_used = grounded.used_tokens(delivered)
            messages = grounded.build_messages(conversation.with_summary(SYSTEM_B, summary), rt.turn.user, delivered,
                                               history=history)
            answer, call = grounded.generate(messages)
            result = grounded.score(answer, call, rt.turn, delivered, GOLD[rt.turn_id])
            rewrite_call = REWRITE_ROWS[rt.turn_id]["call"] if QUERY_FOR_B == "rewrite" else None
            history_log.write({
                "arm": arm.name, "item_id": rt.turn_id, "dialogue_id": rt.dialogue_id, "turn": rt.number,
                "depends_on": rt.turn.depends_on, "query_type": QUERY_FOR_B, "query_fallback": fallback,
                "delivered": [u.unit_id for u in delivered], "passage_tokens": passages_used,
                "history_tokens": history_used, "total_tokens": history_used + passages_used,
                "history_messages": len(history), "history_dropped": len(state.messages) - len(history),
                "first_turn_in_view": rt.number == 1 or first_turn_in_view(history, rt),
                "summary": summary or None, "summary_calls": summary_calls, "rewrite_call": rewrite_call,
                "delivery": grounded.delivery(rt.turn, delivered, GOLD[rt.turn_id]),
                "answer": answer.model_dump() if answer else None,
                "score": result, "call": call, "error": call.get("error") or result.get("error")})
        if state.buffer is not None:
            SUMMARY_UPDATES += [u | {"pass": "answers", "arm": arm.name, "dialogue_id": dialogue.dialogue_id}
                                for u in state.buffer.updates]
    print(f"{arm.name}: listo ({time.time() - t0:.0f} s acumulados)")
print("llamadas:", {k: v for k, v in llm.STATS.items() if v})
print("dónde corre:", llm.placement(MODEL))

# %% [markdown]
r"""
### Las tablas de la Parte B

- **Acierto**: la misma vara que la Etapa 4 (`grounded.score`). Un turno con respuesta acierta si responde con una clave y, si es de sí/no, con el veredicto correcto. Un turno `unanswerable` acierta si y solo si se abstiene. Los errores de llamada se cuentan aparte.
- **Clave desde gold**: un pasaje del gold con la clave le llegó al generador. Con menos historial hay más pasajes: esta columna separa "llegó y no se usó" de "no llegó".
- **Prompt (servidor)**: `prompt_eval_count`, la cuenta que manda al comparar brazos. Al lado, el historial, los pasajes y el total con nuestra cuenta.
- **Turno 1 a la vista**: en cuántos turnos no-primeros la pregunta del turno 1 sigue literalmente en la ventana. En el resumen, lo que no está a la vista puede estar resumido.
- **Costo por turno**: llamadas (la respuesta, más la reescritura de la consulta y las actualizaciones del resumen que le tocaron), tokens de salida de todas ellas y segundos.
- **IC 95%**: bootstrap que remuestrea guiones.
"""


# %%
def b_rows(arm: str) -> list[dict]:
    return history_log.rows(arm)


def b_correct(arm: str) -> dict[str, bool]:
    return {r["item_id"]: r["score"]["correct"] for r in b_rows(arm) if r["score"]["outcome"] != "error"}


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values)) if values else None


def turn_cost(r: dict) -> dict:
    calls = [r["call"]] + r["summary_calls"] + ([r["rewrite_call"]] if r["rewrite_call"] else [])
    return {"calls": len(calls), "output": sum(c.get("output_tokens") or 0 for c in calls),
            "ms": sum(c.get("ms") or 0 for c in calls)}


table_b = []
for arm in ARMS:
    rs = b_rows(arm.name)
    scored = [r for r in rs if r["score"]["outcome"] != "error"]
    answerable = [r for r in scored if TURNS[r["item_id"]].turn.answerable]
    unanswerable = [r for r in scored if not TURNS[r["item_id"]].turn.answerable]
    later = [r for r in rs if r["turn"] > 1]
    delivered_rows = [r for r in rs if TURNS[r["item_id"]].turn.answerable]
    costs = [turn_cost(r) for r in rs]
    _, lo, hi = stats.bootstrap_ci(b_correct(arm.name), GROUPS)
    table_b.append({
        "brazo": arm.name + (" (referencia)" if arm.name == REFERENCE else ""),
        "acierto": f"{sum(r['score']['correct'] for r in scored)}/{len(scored)}", "IC 95%": f"{lo:.0%}-{hi:.0%}",
        "con respuesta": f"{sum(r['score']['correct'] for r in answerable)}/{len(answerable)}",
        "abstención correcta": f"{sum(r['score']['correct'] for r in unanswerable)}/{len(unanswerable)}",
        "errores": len(rs) - len(scored),
        "clave desde gold": f"{sum(bool(r['delivery']['gold_key_delivered']) for r in delivered_rows)}/"
                            f"{len(delivered_rows)}",
        "turno 1 a la vista": f"{sum(r['first_turn_in_view'] for r in later)}/{len(later)}",
        "prompt (servidor)": mean([r["call"]["prompt_tokens"] for r in rs if r["call"].get("prompt_tokens")]),
        "historial": mean([r["history_tokens"] for r in rs]), "pasajes": mean([r["passage_tokens"] for r in rs]),
        "total": mean([r["total_tokens"] for r in rs]),
        "llamadas/turno": round(sum(c["calls"] for c in costs) / max(len(rs), 1), 2),
        "tokens de salida/turno": mean([c["output"] for c in costs]),
        "s/turno": round(sum(c["ms"] for c in costs) / 1000 / max(len(rs), 1), 1),
        "consulta de respaldo": sum(r["query_fallback"] for r in rs),
    })
display(pd.DataFrame(table_b).set_index("brazo"))

# %% [markdown]
r"""
### ¿El total es el mismo en todos los brazos?

El total se iguala en nuestra cuenta: los pasajes reciben lo que el historial no usó. Pero la entrega no llena el presupuesto exacto, porque saltea pasajes que no entran, y la plantilla agrega tokens que nuestra cuenta no ve. Lo que manda es el servidor. Por turno se mira la diferencia entre el `prompt_eval_count` más alto y el más bajo de los brazos de las comparaciones. La referencia queda afuera y se reporta su total real.
"""

# %%
compared = sorted({name for pair in COMPARISONS for name in pair})
server = {name: {r["item_id"]: r["call"].get("prompt_tokens") for r in b_rows(name)} for name in compared + [REFERENCE]}
spreads = []
for rt in REPORT:
    values = [server[name].get(rt.turn_id) for name in compared]
    if all(v is not None for v in values):
        spreads.append(max(values) - min(values))
print(f"diferencia de prompt_eval_count entre brazos, por turno: máx {max(spreads, default=None)} · "
      f"media {mean(spreads)} tokens, sobre {len(spreads)} turnos")
reference = [v for v in server[REFERENCE].values() if v is not None]
print(f"{REFERENCE}: prompt_eval_count medio {mean(reference)}, máx {max(reference, default=None)} "
      f"(sin tope de historial: no está igualado por construcción)")
run.note("equal_total", {"max_spread": max(spreads, default=None), "mean_spread": mean(spreads)})

# %% [markdown]
r"""
### El ruido de la Parte B

Cada comparación mueve una sola cosa:
- **La p es la del test de signos por guion**, un voto por guion (`stats.paired_groups`).
- Las victorias y derrotas por turno se muestran como descripción.
- El intervalo es un bootstrap que remuestrea guiones (8), no turnos (42).

Después, el acierto por número de turno: el historial pesa más cuanto más avanzado está el guion.
"""

# %%
noise_b = [compare(a, b, b_correct(a), b_correct(b)) for a, b in COMPARISONS]
display(pd.DataFrame(noise_b).set_index("comparación"))

by_turn = pd.DataFrame({arm.name: {f"turno {n}": stats.count({k: v for k, v in b_correct(arm.name).items()
                                                              if TURNS[k].number == n})
                                   for n in sorted({rt.number for rt in REPORT})} for arm in ARMS})
display(by_turn)

# %% [markdown]
r"""
**Lectura** (42 turnos de test; total igualado en 1.500 tokens).
- **El tope restringe de verdad:** con 100, la ventana pierde el turno 1 en 26 de 34 turnos no-primeros; con 150, en 19; con 250, en 7.
- **El acierto casi no se mueve:** sobre los 41 turnos que puntuaron en todos los brazos (dl04.t4 dio error en tres; la tabla de arriba cuenta cada brazo sobre sus propios turnos), 18 (tope 100), 17 (150), 20 (250), 18 (resumen en 150) y 19 (buffer completo, referencia). **Ninguna comparación supera el ruido:** la mayor, 250 contra 150, es 3-0 por guion (p = 0,25).
- **Por qué no se mueve:** el turno 2 da 1/8 en todos los brazos. Con `last_turn`, la recuperación de esos turnos falla, y conservar el turno 1 en el historial no trae el pasaje que falta. A esta escala, la división pasajes/historial importa menos que con qué se busca.
- **El total quedó igualado:** la diferencia de `prompt_eval_count` entre brazos es de 46 tokens en promedio (máximo 140), con 1.867 tokens de prompt por turno.
- **Errores:** 3, los tres el mismo turno (dl04.t4): la respuesta llegó al tope de 1.200 caracteres y la gramática la cortó. Cuenta como error, no como respuesta.
- **Costo:** el resumen cuesta 1,6 llamadas por turno contra 1,0 de la ventana (las actualizaciones del resumen), y 247 tokens de salida contra 214.
"""

# %% [markdown]
r"""
## 7 · Poner a prueba el resumen (paso 9)

**La sonda.** Al final de cada guion, con el historial que ve cada brazo en ese punto, se pregunta: «Antes te pregunté: "<el primer turno>". ¿Qué me respondiste?». Se puntúa con la clave del primer turno.
- Sale mecánicamente de los guiones congelados: no se redacta ningún ítem nuevo.
- La sonda cita la pregunta del turno 1, así que **se sacan las claves que ya están en su texto**: esas se acertarían copiando la pregunta. Si no queda ninguna, el guion no tiene sonda, y se reporta cuántos quedan.
- Va sin pasajes, con reglas que permiten usar la conversación como fuente (`conversation.recall_system`). Es una pregunta lateral: no entra al historial.

**El control `no_history`**, la misma sonda sin historial, mide cuánto se responde sin recordar nada. El primer turno es una pregunta sobre la ley, y el modelo podría saber la respuesta de antes; a libro cerrado, la Etapa 0 dio 0/5.

**`turno 1 a la vista`** dice si la pregunta del turno 1 sigue literalmente en la ventana. Con guiones sueltos y tope 150, varía entre brazos. Si no está y la sonda acierta, la memoria vino del resumen.
"""

# %%
RECALL_SYSTEM = conversation.recall_system(SNAPSHOT)
PROBES = {d.dialogue_id: conversation.recall_probe(d.turns[0]) for d in REPORT_DIALOGUES}
PROBE_GOLD = {dialogue_id: grounded.Gold.of(target, SECTIONS, UNITS)
              for dialogue_id, (_, target) in PROBES.items() if target is not None}
excluded = sorted(dialogue_id for dialogue_id, (_, target) in PROBES.items() if target is None)
print(f"guiones con sonda: {len(PROBES) - len(excluded)}/{len(PROBES)}"
      + (f" · sin claves fuera de la pregunta: {', '.join(excluded)}" if excluded else ""))
PROBE_SPEC = B_SPEC | {"stage": "05B-probe", "recall_system": store.sha256_text(RECALL_SYSTEM),
                       "prompt": grounded.prompt_spec(RECALL_SYSTEM)}
probe_log = store.ResultsLog(ART / "results" / "05_summary_probe.jsonl", run_spec=PROBE_SPEC)
for arm in (*ARMS, None):  # None: el control sin historial
    name = arm.name if arm else "no_history"
    llm.unload(MODEL)
    for dialogue in REPORT_DIALOGUES:
        question, target = PROBES[dialogue.dialogue_id]
        turns = conversation.replay(dialogue)
        state = HistoryState(arm, count, conversation.summarize) if arm else None
        if state is not None:
            for rt in turns:
                state.add(rt.exchange)  # salvo la última, las llamadas de resumen salen de la caché de arriba
            SUMMARY_UPDATES += [u | {"pass": "probe", "arm": name, "dialogue_id": dialogue.dialogue_id}
                                for u in (state.buffer.updates if state.buffer is not None else [])]
        if target is None or probe_log.done(name, dialogue.dialogue_id):
            continue
        summary, history, history_used = state.view() if state is not None else ("", [], 0)
        answer, call = grounded.generate(grounded.build_messages(
            conversation.with_summary(RECALL_SYSTEM, summary), question, None, history=history))
        result = grounded.score(answer, call, target, None, PROBE_GOLD[dialogue.dialogue_id])
        probe_log.write({"arm": name, "item_id": dialogue.dialogue_id, "history_tokens": history_used,
                         "summary": summary or None, "first_turn_in_view": first_turn_in_view(history, turns[-1]),
                         "answer_key": target.answer_key, "answer": answer.model_dump() if answer else None,
                         "score": result, "call": call, "error": call.get("error") or result.get("error")})
store.write_jsonl(ART / "results" / "05_summaries.jsonl", SUMMARY_UPDATES)

probe_table = []
for name in [a.name for a in ARMS] + ["no_history"]:
    rs = probe_log.rows(name)
    scored = [r for r in rs if r["score"]["outcome"] != "error"]
    probe_table.append({"brazo": name, "recuerda": f"{sum(r['score']['correct'] for r in scored)}/{len(scored)}",
                        "se abstuvo": sum(r["score"]["outcome"] == "abstained" for r in scored),
                        "turno 1 a la vista": sum(r["first_turn_in_view"] for r in rs),
                        "errores": len(rs) - len(scored)})
display(pd.DataFrame(probe_table).set_index("brazo"))

# %% [markdown]
r"""
**Lectura** (7 guiones; dl02 queda afuera porque su clave del turno 1 está en la pregunta de la sonda).
- **Acá sí se ve el mecanismo:** con el mismo tope de 150, la ventana recuerda la respuesta del turno 1 en 1 de 7 guiones y el resumen en 3 de 7, sin tenerla a la vista en ninguno. Con 100, 0 de 7; con 250, 2 de 7; con el buffer completo, 7 de 7. Sin historial, 0 de 7 (el control: no se acierta por preentrenamiento).
- **Con 7 guiones, no es evidencia:** el resumen gana en 3 guiones (dl03, dl06, dl07) y la ventana en 1 (dl08), con p = 0,625. Es una dirección que coincide con lo que el resumen tiene que hacer.
- **El resumen está apretado:** 30 de sus 58 actualizaciones llegaron al tope de 220 caracteres, y la gramática las cortó. El tope se eligió para que resumen y último par entren en 150 tokens (decision_log); el costo es que el resumen pierde detalles, y eso es parte de lo que mide esta sonda.
- **Ningún resumen contiene razonamiento** (`analysis` o `<think>`): 0 de 58.
"""

# %% [markdown]
r"""
### La auditoría de los resúmenes

Abajo está cada actualización del resumen del brazo `summary_h150` en la corrida de respuestas: lo que salió de la ventana y el resumen que lo reemplazó. `05_summaries.jsonl` guarda todas, también las de la sonda. Se cuenta, por actualización, en las tres categorías de Wang et al. (Apéndice B.6):
- **hechos inventados**: algo que nadie dijo;
- **relaciones equivocadas**: datos ciertos mal atribuidos o mal conectados;
- **detalles faltantes**: un dato del usuario, o un número de la respuesta, que se perdió.

Con 220 caracteres, los detalles faltantes son esperables: es el costo de comprimir a la mitad de 150 tokens. Los conteos van a `decision_log.md` y al informe. La celda solo chequea lo mecánico:
- ningún resumen contiene un bloque `<think>` ni un campo `analysis`;
- cuántos llegaron al tope, que la gramática corta sin avisar: son sospechosos;
- cuántos fallaron.
"""

# %%
answers_pass = [u for u in SUMMARY_UPDATES if u["pass"] == "answers"]
for n, update in enumerate(answers_pass, start=1):
    flags = " · EN EL TOPE" if update["at_maxlength"] else ""
    print(f"\n=== {update['dialogue_id']} · actualización {n} · error: {update['error']}{flags}")
    for message in update["evicted"]:
        print(wrap(f"{'U' if message['role'] == 'user' else 'A'}: {clip(message['content'], 300)}"))
    print(wrap(f"RESUMEN: {update['after']}"))
leaks = sum("<think>" in (u["after"] or "") or "analysis" in (u["after"] or "") for u in SUMMARY_UPDATES)
print(f"\n{len(SUMMARY_UPDATES)} actualizaciones en total ({len(answers_pass)} en la corrida de respuestas) · "
      f"en el tope: {sum(u['at_maxlength'] for u in SUMMARY_UPDATES)} · "
      f"fallidas: {sum(bool(u['error']) for u in SUMMARY_UPDATES)} · con razonamiento filtrado: {leaks} · "
      f"filtraciones de </think> en todas las llamadas: {llm.STATS['think_leaks']}")
run.note("summary", {"updates": len(SUMMARY_UPDATES), "errors": sum(bool(u["error"]) for u in SUMMARY_UPDATES),
                     "at_maxlength": sum(u["at_maxlength"] for u in SUMMARY_UPDATES),
                     "reasoning_leaks": leaks, "think_leaks": llm.STATS["think_leaks"]})

# %% [markdown]
r"""
**La auditoría de los resúmenes**, sobre las 25 actualizaciones de la corrida de respuestas, en las tres categorías de Wang et al. (Apéndice B.6). La planilla ciega para que una persona la rehaga la escribe la Etapa 8 (`results/05_summaries_human_audit.csv`):

| Categoría | Cuántas | Ejemplo |
|---|---|---|
| Correctas | 6 | dl07: "El usuario, de 34 años, alquila un piso por 550 euros al mes. El Bono Alquiler Joven le daría 250 euros mensuales durante dos años." |
| Hechos inventados | 0 | — |
| Relaciones equivocadas | 1 | dl08: dice que el artículo 17 ter "no existe"; el asistente había dicho que no lo encontraba |
| Detalles faltantes | 16 | dl02: el resumen queda igual (y cortado en "Debe.") durante tres turnos, y se pierden los cuatro meses de preaviso, la subrogación del heredero y los dos meses para avisar |
| Texto corrupto | 2 | dl05: "salvo que la造成", "del volumen anual de la平台" (tokens en chino al llegar al tope) |

**Qué explica.** Los faltantes vienen de dos mecanismos:
- **El resumen se congela en el tope.** Al llegar a los 220 caracteres (30 de las 58 actualizaciones), el modelo vuelve a escribir el mismo texto cortado, y lo que se desaloja no entra. Pasa en dl02, dl03 y dl08 durante tres turnos seguidos.
- **A veces reemplaza en lugar de fusionar.** En dl01 la tercera actualización se queda solo con la fianza del local y pierde "septiembre de 2016", que es justo lo que necesita la pregunta de vuelta del turno 6.

Con un tope tan chico, el resumen pierde la mayor parte de lo que desaloja. Por eso no le gana a la ventana en la Parte B, y solo recupera el turno 1 en 3 de 7 guiones en la sonda. **Lo que se probó es este resumen, con este tope, no la estrategia en general.** La Etapa 8 prueba un resumidor v2 con el mismo tope de historial, que pide reescribir comprimiendo y no deja que la gramática corte el texto.
"""

# %% [markdown]
r"""
## Para el informe

- **¿Qué consulta ganó en qué tipo de dependencia, y cambia con el turno?** Ninguna supera el ruido. `full_conversation` encuentra mejor el documento en correferencias y elipsis y peor en los cambios de tema; `last_turn` falla en el turno 2 (sección 4).
- **¿Dónde quedó la mejor división pasajes/historial, y supera el ruido?** Ningún tope de historial se distingue de otro (sección 6). Con `last_turn` como consulta, el cuello de botella es la recuperación de los turnos dependientes, no el historial.
- **¿Qué hicieron mal los resúmenes?** En la auditoría, de 25: 0 hechos inventados, 1 relación equivocada, 16 con detalles faltantes y 2 con texto corrupto. Los faltantes vienen de un resumen que se congela en el tope de 220 caracteres (30 de 58 lo tocaron) o que reemplaza en lugar de fusionar.
- **¿Cuánto cuesta cada estrategia por turno?** Ventana: 1,0 llamadas y ~214 tokens de salida; resumen: 1,6 llamadas y ~247 (sección 6).
- **¿Dónde perdió la reescritura oráculo frente a la del modelo?** Por guion, oráculo contra `rewrite` es 4-1 (p = 0,375): no hay diferencia que defender.
"""

# %%
run.note("rows", {"queries": len(queries_log.rows()), "history": len(history_log.rows()),
                  "probe": len(probe_log.rows())})
run.finish()
