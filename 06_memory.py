# %% [markdown]
r"""
# Etapa 6 · Memoria entre sesiones

Entre sesiones no sobrevive nada que no se escriba. Acá la memoria es un corpus que el sistema escribe **mientras corre**, a partir de una fuente que se puede equivocar (su propia extracción), y que tiene que mantener al día cuando el usuario cambia de situación.

En el vocabulario de CoALA (Apéndice B.4):

| Memoria | Acá |
|---|---|
| de trabajo | la ventana que arma `grounded.build_messages` en cada sonda |
| episódica | las transcripciones de las sesiones (los brazos `replay`) y los registros `episodic` (qué preguntó, qué se le respondió) |
| semántica | los registros `semantic`: los datos del usuario y de su situación |
| procedimental | los registros `procedural`: cómo quiere que se le responda (citar el número de la norma, tutearlo, llamarlo "Pili") |

| Entra | Sale |
|---|---|
| los escenarios de la Etapa 2 (`eval/scenarios.jsonl`), el índice y la mejor configuración de la Etapa 3, el ensamblador de la Etapa 4 | `memory/` (los registros, el log de operaciones de cada escenario y la planilla de auditoría), `results/06_memory.jsonl` (una fila por sonda y brazo), `run_report.json` |

La lógica vive en `src/memory.py`. Esta notebook corre, muestra y lee.

**Cinco brazos, en este orden y con el modelo descargado al empezar cada fase:**
- `no_memory`: el piso. El espacio de memoria dice que no hay recuerdos;
- `replay`: la línea de base "si entra, pegalo". Se reproducen los mensajes de las sesiones anteriores a la sonda, de los más nuevos a los más viejos, hasta 3.000 tokens. Con estos escenarios entra todo el historial. En la evaluación propia de Mem0, esta línea de base **ganó en exactitud** (Apéndice B.4);
- `replay_600`: la misma reproducción con el presupuesto del almacén, 600 tokens. Es el régimen en el que el historial **no** entra, y deja una sola variable entre la reproducción y el almacén: qué se pone en los mismos 600 tokens;
- `store`: el almacén. Hasta 6 recuerdos en 600 tokens, elegidos por recencia + importancia + relevancia;
- `store_filter`: el mismo almacén y el mismo espacio de memoria, con los **pasajes filtrados por lo que dice la memoria** (la comunidad del usuario y la fecha de su contrato). Es el paso 4 de la consigna.

En los cuatro primeros, lo demás del prompt es igual: el mismo prompt de sistema, los mismos pasajes (la mejor configuración de la Etapa 3, con la pregunta de la sonda como consulta) y la misma pregunta. Se mueve una sola cosa, lo que va entre `<memoria>` y `</memoria>`. Entre `store` y `store_filter` se mueve otra sola cosa: el filtro de los pasajes.

**El tiempo es el del escenario.** Cada sesión y cada sonda tienen su fecha simulada. Una sonda hecha en junio solo ve lo que se escribió hasta junio, y ve como vigente lo que se invalidó en septiembre. La recencia se mide en horas simuladas, nunca con el reloj.
"""

# %%
import environment  # el del curso, primero: guarda de JAX, caché de HF, UTF-8

import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import date

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
import doc_schema
import evalset
import grounded
import index_loader
import memory
import retrieval
import stats
from evalset import block_of

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
DOC_REGION = {doc.doc_id: doc.metadata.region for doc in DOCS}

SCENARIOS = ev.scenarios  # en el orden del archivo: es el orden de corrida
PROBES = {probe.probe_id: (scenario, probe) for scenario in SCENARIOS for probe in scenario.probes}
REPORT = [probe for scenario in SCENARIOS if SMOKE_TEST or scenario.split == "test" for probe in scenario.probes]
REPORT_SCENARIOS = [s for s in SCENARIOS if SMOKE_TEST or s.split == "test"]
GROUP = {probe_id: scenario.scenario_id for probe_id, (scenario, _) in PROBES.items()}  # para el ruido

MODEL, NUM_CTX, NUM_PREDICT = grounded.MODEL, grounded.NUM_CTX, grounded.NUM_PREDICT
ARMS = memory.ARMS
MEMORY_DIR = ART / "memory"
count = grounded.token_counter(MODEL)

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)  # como script, pandas corta las tablas anchas
environment.summary()
print(f"corpus_sha {CORPUS_SHA[:12]}... | eval {ev.key()} | {len(SCENARIOS)} escenarios, {len(PROBES)} sondas | "
      f"se reporta sobre {len(REPORT)} sondas de {len(REPORT_SCENARIOS)} escenarios")

# %% [markdown]
r"""
## 1 · El recuperador de la Etapa 3

`index_loader.load` carga el índice y la mejor configuración que eligió la Etapa 3 (`best_config.json`), y **falla si la clave del índice no coincide**: la Etapa 3 habría elegido sobre otro índice. Antes se descarga el generador: en 8 GB no entran los dos juntos.

Apenas se carga, el embedder y el reranker pasan a CPU en fp32 (`release_gpu`; en una T4 entran junto con el generador y no se mueven). La escritura de la memoria usa el generador en la GPU y el embedder en CPU, para encontrar los recuerdos parecidos a un hecho nuevo. La recuperación de los pasajes se hace después, en un solo lote en GPU (sección 6), cuando los filtros de `store_filter` ya se pueden calcular.

`store_filter` usa la misma configuración con el filtro de fecha y el de región prendidos (`dataclasses.replace`).
"""

# %%
llm.unload(MODEL)  # libera la VRAM para el índice
LOADED = index_loader.load(ART, DOCS, ROOT, ev.key())
LOADED.release_gpu()
UNITS = LOADED.units
CONFIG, BUDGET = LOADED.config, LOADED.config.budget_tokens
FILTER_CONFIG = replace(CONFIG, date_filter=True, region_filter=True)
print(f"índice {LOADED.key}: {len(LOADED.index.children):,} hijos · mejor fila de la Etapa 3: {LOADED.best['row']}")
print(f"  configuración: {asdict(CONFIG)}")
print(f"  store_filter: {retrieval.diff_rows(CONFIG, FILTER_CONFIG)}")
GOLD = {pid: grounded.Gold.of(probe, SECTIONS, UNITS) for pid, (_, probe) in PROBES.items()}


def encode(texts: list[str]):
    """bge-m3 del índice, en CPU: la similitud de los recuerdos y la relevancia del camino de lectura."""
    return LOADED.index.embedder.encode(list(texts), batch_size=16, normalize_embeddings=True,
                                        convert_to_numpy=True)


run = StageRun.start("06_memory", ART, profile=ART.name,
                     config={"model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "passage_budget": BUDGET,
                             "memory_budget": memory.MEMORY_BUDGET, "memory_k": memory.MEMORY_K,
                             "replay_budgets": memory.REPLAY_BUDGETS, "arms": ARMS, "eval": ev.key(),
                             "index": LOADED.key, "retrieval": LOADED.best["row"]})

# %% [markdown]
r"""
## 2 · El régimen: ¿el historial entra en la ventana?

Cada afirmación del tipo "la memoria le gana (o pierde) al contexto completo" depende de esto. La cuenta, sin llamar al modelo:
- **la ventana para historial** es `num_ctx − num_predict` (7.424), menos el prompt sin memoria de la sonda más larga, menos el presupuesto de pasajes de la Etapa 3;
- **los presupuestos de reproducción**: 3.000 (`replay`) y 600 (`replay_600`, el mismo del almacén);
- **el historial de cada escenario**: sus sesiones reproducidas enteras.

Para cada presupuesto se dice dónde deja de entrar la reproducción:
- por escenario;
- con los escenarios reproducidos uno detrás de otro, como si fueran un solo usuario;
- sonda por sonda: cuántas reciben la reproducción recortada.
"""

# %%
longest = max((p for _, p in PROBES.values()), key=lambda p: count(p.question))
overhead = llm.count_prompt_tokens(grounded.build_messages(memory.probe_system(longest.asked_at.date()),
                                                           longest.question, [], memory=memory.NO_MEMORY_TEXT), MODEL)
WINDOW = grounded.PROMPT_LIMIT - overhead - BUDGET
print(f"prompt sin memoria ni pasajes (la sonda más larga): {overhead} · pasajes: {BUDGET} · "
      f"queda para el historial: {WINDOW} de {grounded.PROMPT_LIMIT}")
assert max(memory.REPLAY_BUDGETS.values()) <= WINDOW, "un presupuesto de replay no entra en la ventana"
assert memory.MEMORY_BUDGET <= WINDOW

LIMITS = {f"replay ({memory.REPLAY_BUDGET})": memory.REPLAY_BUDGET,
          f"replay_600 ({memory.MEMORY_BUDGET})": memory.MEMORY_BUDGET,
          f"ventana con pasajes ({WINDOW})": WINDOW,
          f"ventana sin pasajes ({grounded.PROMPT_LIMIT - overhead})": grounded.PROMPT_LIMIT - overhead}
regime = pd.DataFrame([{"escenario": s.scenario_id, "split": s.split,
                        "tokens por sesión": memory.history_tokens(s, count),
                        "historial": sum(memory.history_tokens(s, count)),
                        **{name: sum(memory.history_tokens(s, count)) <= limit for name, limit in LIMITS.items()}}
                       for s in SCENARIOS]).set_index("escenario")
display(regime)

chain_rows = memory.where_replay_stops_fitting(SCENARIOS, count, LIMITS)
display(pd.DataFrame(chain_rows).set_index("session"))
STOPS = {name: memory.first_overflow(chain_rows, name) for name in LIMITS}
for name, where in STOPS.items():
    print(f"escenarios seguidos · {name}: {where}")

truncated = {arm: sum(not memory.replay(scenario.sessions, probe.asked_at, budget, count)[3]
                      for scenario, probe in PROBES.values())
             for arm, budget in memory.REPLAY_BUDGETS.items()}
print(f"sondas con la reproducción recortada (de {len(PROBES)}): {truncated}")
run.note("regime", {"window_for_history": WINDOW, "overhead": overhead,
                    "history_tokens": regime["historial"].to_dict(), "chained": STOPS,
                    "probes_with_truncated_replay": truncated})

# %% [markdown]
r"""
**Lectura: el régimen.** **El historial entra en la ventana.** Cada escenario de test tiene entre 820 y 947 tokens de historial, y con los pasajes quedan 3.794 para él. Incluso los cuatro escenarios seguidos (3.629 tokens) entran. Por eso `replay` (3.000) es el "si entra, pegalo" real, y la única forma de ver el régimen en el que no entra es imponerlo: `replay_600` recorta 28 de las 30 sondas. Cada afirmación de abajo sobre memoria contra reproducción es en este régimen, con `num_ctx = 8.192`.
"""

# %% [markdown]
r"""
## 3 · El servidor
"""

# %%
SERVER = llm.setup(MODEL, num_ctx=NUM_CTX)
llm.use_cache(ART / "llm_cache" / "qwen3_8b.jsonl")
run.note("server", {k: SERVER.get(k) for k in ("digest", "server", "key_order", "thinking")})

# %% [markdown]
r"""
## 4 · El camino de escritura

Después de cada sesión, en orden de fecha:
1. **Extracción** con un schema: `analysis` primero, después una lista de hechos `{kind, subject, text, importance}`. El prompt recibe las etiquetas (`subject`) vigentes del usuario para que reuse la misma etiqueta para el mismo dato.
2. Por cada hecho, los **3 recuerdos vigentes más parecidos** (bge-m3 sobre `subject: text`).
3. **La decisión**, con otro schema (`analysis` < `decision` < `target_id` < `text`): ADD, UPDATE, NOOP o INVALIDATE.
4. **Se aplica con el tiempo simulado.** `created_at` y `valid_from` son el comienzo de la sesión.
   - UPDATE e INVALIDATE cierran el registro elegido (`valid_to`, `superseded_by`) y guardan el hecho nuevo, como Zep: el hecho que invalida también es un hecho. Invalidar, no borrar.
   - **Un único valor vigente por `subject`**: todo registro nuevo cierra a los vigentes con su misma etiqueta. Los episodios quedan afuera: "qué preguntó" no es el valor actual de nada.

Mientras no hay ningún recuerdo vigente, el hecho se agrega sin llamar al modelo: la única decisión posible es ADD. Un `target_id` que no estaba entre los recuerdos mostrados no se aplica a ciegas: se agrega el hecho sin cerrar nada, marcado en `fallback`.

**Nunca se guarda razonamiento.** De cada llamada se guardan solo los campos del hecho. Los `analysis` se juntan en memoria (no en disco), por sesión, para verificar más abajo que ninguno se coló en un registro. La gramática de Ollama corta un campo en su `max_length` sin error, así que se cuentan los campos que llegaron a su tope (`truncated`).
"""

# %%
STORE_SPEC = {"eval": ev.key(), "corpus_sha": CORPUS_SHA, "model": MODEL, "num_ctx": NUM_CTX,
              "extract_system": store.sha256_text(memory.EXTRACT_SYSTEM),
              "decide_system": store.sha256_text(memory.DECIDE_SYSTEM), "similar_k": memory.SIMILAR_K,
              "encoder": LOADED.encoder_spec}
llm.unload(MODEL)  # arranque en frío
think_before = llm.STATS["think_leaks"]
stores, write_costs, analyses = {}, [], {}
t0 = time.time()
for scenario in SCENARIOS:
    st = memory.MemoryStore(scenario.scenario_id)
    for session in sorted(scenario.sessions, key=lambda s: s.started_at):
        cost = memory.write_session(st, session, structured=memory.llm_structured, encode=encode)
        analyses[session.session_id] = cost.pop("analyses")
        write_costs.append(cost)
    st.save(MEMORY_DIR)
    store.write_manifest(MEMORY_DIR / f"{st.scope}.records.jsonl", STORE_SPEC, records=len(st.records))
    stores[scenario.scenario_id] = st
THINK_LEAKS = {"write": llm.STATS["think_leaks"] - think_before}
EXTRACTION_ERRORS = {c["session"]: c["extraction_error"] for c in write_costs if c["extraction_error"]}
print(f"escritura: {time.time() - t0:.0f} s · llamadas: {sum(c['calls'] for c in write_costs)} · "
      f"extracciones fallidas: {EXTRACTION_ERRORS or 0} · filtraciones de <think>: {THINK_LEAKS['write']}")
STORE_SHA = store.sha256_json({sid: [r.model_dump(mode="json") for r in st.records] for sid, st in stores.items()})
if not any(st.records for st in stores.values()):
    print("ATENCIÓN: ningún almacén tiene registros; los brazos store van a leer espacios vacíos")

costs = pd.DataFrame(write_costs).set_index("session")
display(costs.drop(columns=["at"]))
per_scenario = costs.groupby("scope")[["facts", "closed", "calls", "errors", "prompt_tokens", "output_tokens"]].sum()
per_scenario["registros"] = [len(stores[sid].records) for sid in per_scenario.index]
per_scenario["vigentes al final"] = [sum(r.valid_to is None for r in stores[sid].records) for sid in per_scenario.index]
display(per_scenario)
run.note("write", {"sessions": len(write_costs), "calls": int(costs["calls"].sum()),
                   "errors": int(costs["errors"].sum()), "extraction_errors": EXTRACTION_ERRORS,
                   "prompt_tokens": int(costs["prompt_tokens"].sum()),
                   "output_tokens": int(costs["output_tokens"].sum()),
                   "truncated_fields": dict(sum((Counter(t) for t in costs["truncated"]), Counter())),
                   "store_sha": STORE_SHA})

# %% [markdown]
r"""
## 5 · Qué guardó: la auditoría

Abajo, cada sesión está impresa entera y, debajo, los registros que salieron de ella. La planilla para anotar es `artifacts/<perfil>/memory/audit_records.csv`, con una fila por registro. Se completa la columna `human_verdict`:
- `bien`: dice lo que se dijo;
- `mal`: dice otra cosa (un monto o una fecha equivocados, una relación invertida);
- `nunca_se_dijo`: una suposición guardada como hecho.

Una nota opcional va en `human_note`. Cuando la planilla está completa, esta celda cuenta los veredictos.

Hay dos chequeos automáticos que no reemplazan esa lectura:
- registros con un bloque `<think>`, o con un tramo de 8 palabras que está en un `analysis` de su sesión y **no** está en la transcripción. Tiene que dar 0;
- registros escritos como hipótesis ("probablemente", "parece"…, por palabra completa). Son candidatos a `nunca_se_dijo`.
"""

# %%
TRANSCRIPTS = {s.session_id: memory.transcript(s) for scenario in SCENARIOS for s in scenario.sessions}
for sid, st in stores.items():
    scenario = next(s for s in SCENARIOS if s.scenario_id == sid)
    for session in sorted(scenario.sessions, key=lambda s: s.started_at):
        print(f"\n=== {session.session_id}")
        print(wrap(TRANSCRIPTS[session.session_id], indent="   | "))
        for r in (r for r in st.records if r.source_session == session.session_id):
            state = (f" · cerrado el {r.valid_to.date()}" + (f" por {r.superseded_by}" if r.superseded_by else "")
                     if r.valid_to else "")
            print(f"   {r.memory_id:>4} · {r.kind} · {r.subject} · importancia {r.importance}{state}")
            print(wrap(r.text, indent="          "))

LEAKS = {sid: memory.leaked(st.records, analyses, TRANSCRIPTS) for sid, st in stores.items()}
HEDGES = {sid: memory.hedged(st.records) for sid, st in stores.items()}
fallbacks = Counter(op["fallback"] for st in stores.values() for op in st.ops if op["fallback"])
print(f"\nregistros con análisis o <think>: {sum(map(len, LEAKS.values()))} {LEAKS}")
print(f"registros escritos como hipótesis: {sum(map(len, HEDGES.values()))} {HEDGES}")
print(f"decisiones que no se aplicaron como las pidió el modelo: {dict(fallbacks)}")

AUDIT = MEMORY_DIR / "audit_records.csv"
print(f"planilla de auditoría: {memory.write_audit(AUDIT, memory.audit_rows(stores))}")
verdicts, filled, total = memory.audit_counts(AUDIT)
sheet = pd.read_csv(AUDIT, encoding="utf-8-sig", dtype=str)
AUDITOR = ", ".join(sorted(set(sheet["auditor"].fillna("sin indicar")))) if "auditor" in sheet else "sin indicar"
print(f"veredictos de la auditoría ({AUDITOR}): {dict(verdicts)}" if verdicts
      else f"veredictos de la auditoría: {filled}/{total} completos")
run.note("audit", {"leaked": LEAKS, "hedged": HEDGES, "fallbacks": dict(fallbacks), "auditor": AUDITOR,
                   "verdicts": dict(verdicts) if verdicts else f"{filled}/{total}"})

# %% [markdown]
r"""
**Lectura: qué guardó.** La escritura hizo 89 llamadas, sin errores de extracción ni fugas de `<think>`, y dejó 76 registros (56 vigentes al final). La auditoría registro por registro contra la transcripción de su sesión da **60 bien, 10 mal y 6 que registran algo que nunca se dijo**. En los 3 escenarios de test, 50 / 3 / 3; en el de dev (sc04), 10 / 7 / 3. Los errores tienen cuatro formas:
- **Una etiqueta equivocada que cierra un dato verdadero.** En sc01, la ciudad de la usuaria (Pamplona) se guardó como `ciudad_arrendadora`; cuando dijo que la dueña vive en Tudela, Pamplona se cerró. La regla "un valor vigente por subject" funcionó, sobre un subject mal elegido. Lo mismo con Sevilla en sc04.
- **La fecha de la sesión tomada como dato.** "Firmó el contrato el 11 de mayo de 2026" (sc01) o "pagó los honorarios el 23 de febrero" (sc04): son las fechas en que se habló, no las de los hechos.
- **No-hechos que invalidan hechos.** "El usuario no especificó preferencia de citas" (sc02) y "de fechas" (sc04) se guardaron como valores nuevos del subject de la preferencia, y cerraron las preferencias reales que el usuario había pedido antes. Son las hipótesis guardadas como hechos de la consigna.
- **Relaciones y montos equivocados.** 650 euros guardado como cuota de la comunidad cuando es el alquiler que cobra (sc03); "la fianza se realiza mediante una inmobiliaria" y una explicación del asistente atribuida al usuario (sc04).

**Los dos chequeos automáticos no ven ninguno de esos errores.** `hedged` (lenguaje de hipótesis) marca 0. `leaked` (tramos del análisis que no están en la conversación) marca 7, pero leídos uno por uno, 5 dicen lo que se dijo: el extractor parafrasea la conversación igual en su análisis y en el registro. Solo 1 de los 7 está mal. Contar fugas de razonamiento no reemplaza leer los registros.
"""

# %% [markdown]
r"""
## 6 · La memoria como filtro, y los pasajes de todos los brazos

**El filtro de `store_filter`** (`memory.memory_filters`) es una regla fija, sin modelo. Lee los recuerdos `semantic` vigentes en el `asked_at` de la sonda:
- **la comunidad**: la que nombran los recuerdos, por su nombre (`retrieval.parse_regions`) o por una ciudad de un nomenclátor general (capitales de provincia, ciudades grandes, islas). El usuario dice "vivo en Zaragoza", no "vivo en Aragón";
- **el período**: el del recuerdo que fecha un contrato (contrato, alquiler, arrendamiento, firmó), leído con `retrieval.parse_period`. Si hay varios contratos, gana el más parecido a la pregunta: "el piso de Sóller" elige el contrato de Sóller, no el último;
- **lo que dice la pregunta manda.** Si nombra una fecha o una comunidad, es por esa que pregunta, y la memoria llena lo que la pregunta no dice;
- si nada fecha la sonda, el período es **el día de la sonda**, su "hoy" simulado. Con esa regla se etiquetó el gold (`Probe.reference_day`).

Los episodios no entran: repiten lo que dijo el asistente sobre la ley ("se le explicó la ley andaluza"), no dónde vive el usuario.

**Los pasajes.** Con los filtros calculados, se descarga el generador, el índice vuelve a la GPU y se recupera **todo en un solo lote**: la consulta de cada sonda sin filtro de memoria (para los primeros cuatro brazos) y con él (para `store_filter`). Los dos rankings salen del mismo dispositivo y la misma precisión: entre `store` y `store_filter` cambia solo el filtro. Después el índice vuelve a CPU y la GPU queda para el generador.
"""

# %%
probe_ids = list(PROBES)
QUERY = dict(zip(probe_ids, encode([PROBES[pid][1].question for pid in probe_ids])))  # CPU, como la memoria
FILTERS, FILTER_INFO = {}, {}
for pid, (scenario, probe) in PROBES.items():
    FILTERS[pid], FILTER_INFO[pid] = memory.memory_filters(stores[scenario.scenario_id], probe, QUERY[pid], encode)
display(pd.DataFrame([{"sonda": pid, "habilidad": PROBES[pid][1].ability, "período": " → ".join(i["period"]),
                       "de": i["period_source"], "comunidades": ", ".join(i["regions"]) or "—",
                       "de dónde": i["regions_source"] or "—"} for pid, i in FILTER_INFO.items()]).set_index("sonda"))

llm.unload(MODEL)
LOADED.to_gpu()
t0 = time.time()
RANKED, RETRIEVAL_INFO = {"base": {}, "filter": {}}, {"base": {}, "filter": {}}
for pid, (_, probe) in PROBES.items():
    for kind, cfg, filters in (("base", CONFIG, None), ("filter", FILTER_CONFIG, FILTERS[pid])):
        units, result = index_loader.ranked_units(LOADED, probe.question, cfg, filters)
        RANKED[kind][pid], RETRIEVAL_INFO[kind][pid] = units, result.filters
LOADED.release_gpu()
print(f"recuperación: {2 * len(PROBES)} consultas en {time.time() - t0:.0f} s")

DELIVERED = {kind: {pid: grounded.deliver(RANKED[kind][pid], BUDGET, count) for pid in PROBES}
             for kind in ("base", "filter")}


def passages_for(arm: str, pid: str):
    return DELIVERED["filter" if arm == "store_filter" else "base"][pid]


changed = [pid for pid in PROBES if [u.unit_id for u in DELIVERED["base"][pid]]
           != [u.unit_id for u in DELIVERED["filter"][pid]]]
print(f"sondas con pasajes distintos entre store y store_filter: {len(changed)} de {len(PROBES)}")
print(f"pasajes por sonda: {sum(len(d) for d in DELIVERED['base'].values()) / len(PROBES):.1f} en promedio; "
      f"tokens como máximo: {max(grounded.used_tokens(d) for k in DELIVERED for d in DELIVERED[k].values())}")
run.note("filters", {"changed_probes": changed, "info": FILTER_INFO})

# %% [markdown]
r"""
## 7 · El camino de lectura

Para cada sonda, en su `asked_at` simulado:
- **`store` y `store_filter`**: los recuerdos escritos hasta ese momento, puntuados con recencia + importancia + relevancia (coseno con la pregunta). Cada término se normaliza min-max y pesa 1, como en Generative Agents. Entran también los invalidados, marcados "ya no vigente desde…". El espacio se llena en orden de puntaje hasta 6 recuerdos o 600 tokens, y el que no entra se saltea;
- **`replay` y `replay_600`**: las sesiones de hasta ese momento, de la más nueva a la más vieja, hasta 3.000 o 600 tokens. La frontera es la misma que la del almacén (`<=`);
- **`no_memory`**: nada.

**La recencia va en escala logarítmica.** Con 0,995 por hora y meses entre sesiones, 0,995^h va de ~1 (la última sesión) a ~1e-10 (la primera). El min-max lo vuelve un escalón: 1 para la última sesión, ≈0 para todas las demás. El término que puntúa es log(0,995^h), y su min-max es el de −horas: conserva el orden y queda graduado. La tabla muestra los rangos crudos, en horas y como decaimiento, antes de normalizar. Se arma antes de llamar al modelo: la lectura no depende de la generación.

**Clave en el espacio de memoria** (una métrica sin juez, para las sondas con respuesta): ¿está la clave en lo que se puso entre `<memoria>` y `</memoria>`? Separa "no llegó" de "llegó y no se usó", igual que la clave entregada en la Etapa 4.
"""

# %%
SLOTS = {arm: {} for arm in ARMS}
for arm in ARMS:
    for pid, (scenario, probe) in PROBES.items():
        SLOTS[arm][pid] = memory.memory_slot(arm, scenario, stores[scenario.scenario_id], probe, encode=encode,
                                             count=count, query=QUERY[pid])
range_rows = [{"sonda": pid,
               "horas (mín-máx)": f"{r['hours'][0]:.0f}-{r['hours'][1]:.0f}",
               "0,995^h (mín-máx)": f"{r['decay'][0]:.2e}-{r['decay'][1]:.2e}",
               "importancia (mín-máx)": f"{r['importance'][0]:.0f}-{r['importance'][1]:.0f}",
               "relevancia (mín-máx)": f"{r['relevance'][0]:.3f}-{r['relevance'][1]:.3f}",
               "recuerdos": len(SLOTS["store"][pid]["memory_ids"]), "tokens": SLOTS["store"][pid]["tokens"]}
              for pid in PROBES if (r := SLOTS["store"][pid].get("ranges"))]
if range_rows:
    display(pd.DataFrame(range_rows).set_index("sonda"))
else:
    print("ninguna sonda tiene recuerdos visibles: no hay rangos que mostrar")

KEY_IN_SLOT = {arm: {pid: evalset.contains_key(SLOTS[arm][pid]["text"], PROBES[pid][1].answer_key)
                     for pid in PROBES if PROBES[pid][1].answerable} for arm in ARMS}
slot_table = pd.DataFrame({arm: {
    "tokens (media)": round(sum(SLOTS[arm][p.probe_id]["tokens"] for p in REPORT) / len(REPORT)),
    "tokens (máx)": max(SLOTS[arm][p.probe_id]["tokens"] for p in REPORT),
    "reproducción recortada": (sum(SLOTS[arm][p.probe_id]["complete"] is False for p in REPORT)
                               if arm in memory.REPLAY_BUDGETS else "—"),
    "clave en el espacio": stats.count({p.probe_id: KEY_IN_SLOT[arm][p.probe_id] for p in REPORT if p.answerable}),
} for arm in ARMS})
display(slot_table)

# %% [markdown]
r"""
## 8 · Las sondas, en los cinco brazos

Cada fila de `results/06_memory.jsonl` guarda:
- el brazo y la sonda;
- qué entró en el espacio de memoria (sesiones, ids, tokens, si la clave estaba);
- los pasajes (y, en `store_filter`, el filtro y de dónde salió), lo que le llegó del gold, la respuesta entera y el registro de la llamada;
- el error de la fila en el primer nivel, así un error transitorio se reintenta al retomar.

El `run_spec` incluye la clave del índice y la configuración de la Etapa 3, el hash de los almacenes, de los textos de cada espacio de memoria, de los filtros, de los pasajes entregados y de la plantilla del prompt (`grounded.prompt_spec`): si cualquiera cambia, las filas viejas no se mezclan.

**Un error de escritura no se puntúa como respuesta.** Si falló la extracción de una sesión con la evidencia de una sonda, el almacén nunca tuvo la oportunidad de guardarla. Las filas de `store` y `store_filter` de esa sonda son un error de escritura (`WRITE_ERROR`), no un acierto ni un fallo, y no se llama al modelo.

Cuando `store_filter` recibe los mismos pasajes que `store`, su prompt es idéntico y la caché de llamadas devuelve la misma respuesta: solo se paga una llamada nueva en las sondas en que el filtro cambió lo entregado.
"""

# %%
RUN_SPEC = {"stage": "06", "eval": ev.key(), "corpus_sha": CORPUS_SHA, "model": MODEL, "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT, "passage_budget": BUDGET, "memory_budget": memory.MEMORY_BUDGET,
            "memory_k": memory.MEMORY_K, "replay_budgets": memory.REPLAY_BUDGETS,
            "recency": f"log({memory.RECENCY_DECAY}^h) min-max", "arms": ARMS,
            "index": LOADED.key, "retrieval": {"row": LOADED.best["row"], "config": asdict(CONFIG)},
            "filter_config": asdict(FILTER_CONFIG),
            # las reglas, no un prompt armado: ese lleva la fecha simulada de cada sonda
            "system_rules": store.sha256_text(grounded.RULES_GROUNDED + memory.MEMORY_RULES),
            "prompt": grounded.prompt_spec(memory.probe_system(longest.asked_at.date())),
            "stores": STORE_SHA,
            "slots": store.sha256_json({arm: {pid: SLOTS[arm][pid]["text"] for pid in PROBES} for arm in ARMS}),
            "filters": store.sha256_json(FILTER_INFO),
            "delivered": store.sha256_json({kind: {pid: [[u.unit_id, u.text] for u in DELIVERED[kind][pid]]
                                                   for pid in PROBES} for kind in DELIVERED})}
log = store.ResultsLog(ART / "results" / "06_memory.jsonl", run_spec=RUN_SPEC)

llm.unload(MODEL)  # la lectura también arranca en frío
think_before = llm.STATS["think_leaks"]
t0 = time.time()
for arm in ARMS:
    for pid, (scenario, probe) in PROBES.items():
        if log.done(arm, pid):
            continue
        slot, delivered = SLOTS[arm][pid], passages_for(arm, pid)
        write_error = memory.write_error_for(probe, EXTRACTION_ERRORS) if arm in memory.STORE_ARMS else None
        if write_error:
            answer, call = None, {"error": write_error}
        else:
            messages = grounded.build_messages(memory.probe_system(probe.asked_at.date()), probe.question,
                                               delivered, memory=slot["text"])
            answer, call = grounded.generate(messages)
        score = grounded.score(answer, call, probe, delivered, GOLD[pid])
        hit, of = memory.evidence_recall(set(slot["sessions"]), probe)
        log.write({"arm": arm, "item_id": pid, "scenario_id": scenario.scenario_id, "split": scenario.split,
                   "ability": probe.ability, "asked_at": probe.asked_at.isoformat(),
                   "error": call.get("error") or score.get("error"),
                   "memory": {"tokens": slot["tokens"], "sessions": slot["sessions"],
                              "memory_ids": slot["memory_ids"], "complete": slot["complete"],
                              "key_in_slot": KEY_IN_SLOT[arm].get(pid)},
                   "evidence": {"hit": hit, "of": of},
                   "filters": FILTER_INFO[pid] if arm == "store_filter" else None,
                   "retrieval_filters": RETRIEVAL_INFO["filter" if arm == "store_filter" else "base"][pid],
                   "delivered": [[u.unit_id, u.n_gen] for u in delivered],
                   "delivered_tokens": grounded.used_tokens(delivered),
                   "delivery": grounded.delivery(probe, delivered, GOLD[pid]),
                   "answer": answer.model_dump() if answer else None, "score": score, "call": call})
    print(f"{arm}: listo ({time.time() - t0:.0f} s acumulados)")
THINK_LEAKS["read"] = llm.STATS["think_leaks"] - think_before
print("llamadas:", {k: v for k, v in llm.STATS.items() if v})
print(f"filtraciones de <think> por fase: {THINK_LEAKS}")
print("dónde corre:", llm.placement(MODEL))
run.note("think_leaks", THINK_LEAKS)

# %% [markdown]
r"""
## 9 · Resultados por habilidad

Como **conteos** (3/4, no 75%): con 22 sondas de test, y entre 3 y 6 por habilidad, un porcentaje sugeriría una precisión que no hay. Los errores (de llamada o de escritura) no entran en el denominador y se cuentan aparte.

- **Recall de evidencia**: de las sesiones donde está la evidencia (`evidence_sessions`), cuántas llegaron al prompt. En los almacenes cuenta la sesión de origen de cada recuerdo elegido; en las reproducciones, las sesiones con algún mensaje adentro. Sin juez.
- **Clave en el espacio**: en las sondas con respuesta, si la clave estaba en el espacio de memoria. Un acierto sin la clave en el espacio vino de los pasajes o de la pregunta.
- **Clave del gold entregada**: en las sondas con gold en el corpus, si llegó un pasaje del gold con la clave (la primaria de la Etapa 3). Es igual en los cuatro primeros brazos y puede cambiar en `store_filter`.
- **Prompt (servidor)**: `prompt_eval_count` medio por sonda. Es lo que paga cada brazo en lectura.
"""


# %%
def rows_of(arm: str, probes: list) -> list[dict]:
    ids = {p.probe_id for p in probes}
    return [r for r in log.rows(arm) if r["item_id"] in ids]


def correct_map(arm: str, probes: list) -> dict[str, bool]:
    return {r["item_id"]: r["score"]["correct"] for r in rows_of(arm, probes) if r["score"]["outcome"] != "error"}


by_ability = pd.DataFrame({arm: {ability: stats.count({k: v for k, v in correct_map(arm, REPORT).items()
                                                       if PROBES[k][1].ability == ability})
                                 for ability in evalset.ABILITIES} for arm in ARMS})
by_ability.loc["total"] = {arm: stats.count(correct_map(arm, REPORT)) for arm in ARMS}
display(by_ability)

summary = []
for arm in ARMS:
    rs = rows_of(arm, REPORT)
    scored = [r for r in rs if r["score"]["outcome"] != "error"]
    calls = [r["call"] for r in rs if r["call"].get("prompt_tokens") is not None]
    evidence = [r["evidence"] for r in rs if r["evidence"]["of"]]
    with_gold = [r for r in rs if PROBES[r["item_id"]][1].gold_sections]
    summary.append({
        "brazo": arm, "acierto": stats.count(correct_map(arm, REPORT)),
        "errores": len(rs) - len(scored),
        "de escritura": sum(str(r["error"] or "").startswith("WRITE_ERROR") for r in rs),
        "se abstuvo": sum(r["score"]["outcome"] == "abstained" for r in scored),
        "recall de evidencia": f"{sum(e['hit'] for e in evidence)}/{sum(e['of'] for e in evidence)}",
        "clave en el espacio": stats.count({r["item_id"]: r["memory"]["key_in_slot"] for r in rs
                                            if r["memory"]["key_in_slot"] is not None}),
        "clave del gold entregada": stats.count({r["item_id"]: bool(r["delivery"]["gold_key_delivered"])
                                                 for r in with_gold}),
        "memoria (tokens, media)": round(sum(r["memory"]["tokens"] for r in rs) / max(len(rs), 1)),
        "prompt (servidor, media)": round(sum(c["prompt_tokens"] for c in calls) / max(len(calls), 1)),
        "s/llamada": round(sum(c["ms"] for c in calls if not c.get("cached")) / 1000
                           / max(sum(not c.get("cached") for c in calls), 1), 1)})
summary = pd.DataFrame(summary).set_index("brazo")
display(summary)

# %% [markdown]
r"""
**Lectura** (22 sondas de 3 escenarios de test; conteos, no porcentajes).
- **Total:** sin memoria 8/22, `replay` 11/22, `replay_600` 10/22, `store` 10/22, `store_filter` 10/22.
- **Dónde gana la memoria a no tener memoria:** en las actualizaciones de conocimiento (2/6 → 4/6 con `store`, 5/6 con el filtro). Es la habilidad para la que existe el camino de escritura: invalidar el valor viejo y quedarse con el nuevo. `replay`, con todo el historial a la vista, se queda en 2/6: ve los dos valores y no siempre elige el último.
- **Dónde gana la reproducción a la memoria:** en la extracción (2/3 contra 1/3) y en lo multi-sesión (2/4 contra 1/4), donde hace falta el dato literal.
- **`store` encuentra las sesiones pero pierde el dato:** trae al espacio de memoria recuerdos de las sesiones con evidencia en 25 de 31 casos, pero la clave está en ese espacio en 5 de 19. La extracción resume, y lo que se resume se pierde: "no llegó" y "llegó y no se usó" se separan acá.
"""

# %% [markdown]
r"""
### El ruido, antes de afirmar nada

Tres lecturas por comparación:
- **por sonda**: victorias, derrotas y empates pareados, y el test de signos sobre las discordantes. Trata como independientes a las sondas de un mismo escenario, que comparten almacén, así que su p es optimista;
- **por escenario** (`stats.paired_groups`): un voto por escenario. Con 3 escenarios de test, el mejor resultado posible es 3-0, con p = 0,25. **Ninguna diferencia puede ser significativa a esta escala**, y eso es parte de la respuesta;
- **bootstrap remuestreando escenarios**. Con 3 escenarios hay solo 10 remuestras distintas: el intervalo es grueso y escalonado. Se muestra igual, con esa advertencia en la tabla.
"""

# %%
n_groups = len(REPORT_SCENARIOS)
noise = []
for a, b in [("no_memory", "replay"), ("no_memory", "store"), ("replay_600", "store"), ("replay", "store"),
             ("store", "store_filter")]:
    ca, cb = correct_map(a, REPORT), correct_map(b, REPORT)
    p = stats.paired(ca, cb)
    g = stats.paired_groups(ca, cb, GROUP)
    diff, lo, hi = stats.bootstrap_diff_ci(ca, cb, GROUP)
    noise.append({"comparación": f"{b} vs {a}", "sondas": p["n"],
                  "por sonda (gana b / gana a / empate)": f"{p['wins']} / {p['losses']} / {p['ties']}",
                  "p (signos, sonda)": round(p["p_sign"], 4),
                  "por escenario (gana b / gana a / empate)": f"{g['wins']} / {g['losses']} / {g['ties']}",
                  "p (signos, escenario)": round(g["p_sign"], 4),
                  "diferencia (bootstrap por escenario)": f"{diff:+.0%} ({lo:+.0%} a {hi:+.0%})",
                  "remuestras distintas": memory.distinct_resamples(n_groups)})
display(pd.DataFrame(noise).set_index("comparación"))

# %% [markdown]
r"""
**Lectura del ruido.** **Ninguna comparación supera el ruido.** Con 3 escenarios, el mejor resultado posible por escenario es 3-0 (p = 0,25), y el bootstrap tiene 10 remuestras distintas. Lo de arriba son direcciones, no hallazgos: la más clara, `replay` contra sin memoria, es 5-2 por sonda.
"""

# %% [markdown]
r"""
## 10 · ¿Ayudó el filtro por memoria?

`store` contra `store_filter`, sonda por sonda: mismo espacio de memoria, distintos pasajes. Para cada sonda en que el filtro cambió lo entregado:
- qué filtro se aplicó y de dónde salió;
- cuántos pasajes sacó, y cuántos de esos eran **de otra comunidad** o **de una versión que no regía en el período**;
- si cambió la clave del gold entregada y si cambió el acierto.

**El pasaje que permitiría decir "eso no está disponible para vos"** (paso 4 de la consigna). Un filtro por comunidad saca, por ejemplo, la ley de otra comunidad que regula lo que el usuario pregunta. Ese pasaje es justo el que le permitiría al modelo contestar "eso existe en tal comunidad, no en la tuya". No hay manera automática de saber cuál era ese pasaje. Se marcan como **candidatos** las sondas en las que la respuesta de `store` **citó** un bloque que el filtro sacó: el modelo se apoyaba en él. Debajo se imprimen esos pasajes para leerlos a mano.
"""


# %%
def overlaps(section_id: str, period: list[str]) -> bool:
    section = SECTIONS[section_id][1]
    start, end = date.fromisoformat(period[0]), date.fromisoformat(period[1])
    return (section.valid_from or date.min) < end and start < (section.valid_to or date.max)


store_rows = {r["item_id"]: r for r in log.rows("store")}
filter_rows = {r["item_id"]: r for r in log.rows("store_filter")}
effects, candidates = [], []
for pid in changed:
    probe, info = PROBES[pid][1], FILTER_INFO[pid]
    kept = {u.unit_id for u in DELIVERED["filter"][pid]}
    removed = [u for u in DELIVERED["base"][pid] if u.unit_id not in kept]
    allowed = set(info["regions"]) | {"Estatal"}
    other_region = [u for u in removed if info["regions"] and DOC_REGION[u.doc_id] not in allowed]
    out_of_period = [u for u in removed if not overlaps(u.section_id, info["period"])]
    rs, rf = store_rows.get(pid), filter_rows.get(pid)
    cited = set(rs["score"].get("cited") or []) if rs and rs["answer"] else set()
    cited_removed = sorted(cited & {block_of(u.section_id) for u in removed})
    effects.append({
        "sonda": pid, "split": PROBES[pid][0].split, "habilidad": probe.ability,
        "filtro": f"{' → '.join(info['period'])} ({info['period_source']}); {', '.join(info['regions']) or '—'}",
        "sacó": len(removed), "de otra comunidad": len(other_region), "fuera del período": len(out_of_period),
        "clave del gold (store → filtro)": f"{rs and rs['delivery']['gold_key_delivered']} → "
                                           f"{rf and rf['delivery']['gold_key_delivered']}",
        "acierto (store → filtro)": f"{rs and rs['score']['correct']} → {rf and rf['score']['correct']}",
        "citaba lo que sacó": ", ".join(cited_removed) or "—"})
    if cited_removed:
        candidates.append((pid, [u for u in removed if block_of(u.section_id) in cited_removed]))
if effects:
    display(pd.DataFrame(effects).set_index("sonda"))
else:
    print("el filtro por memoria no cambió los pasajes de ninguna sonda")

helped = []
for ability in evalset.ABILITIES:
    in_ability = {pid for pid in PROBES if PROBES[pid][1].ability == ability}
    p = stats.paired({k: v for k, v in correct_map("store", REPORT).items() if k in in_ability},
                     {k: v for k, v in correct_map("store_filter", REPORT).items() if k in in_ability})
    helped.append({"habilidad": ability, "sondas": p["n"], "pasajes distintos": len(in_ability & set(changed)),
                   "gana el filtro": p["wins"], "gana store": p["losses"], "empates": p["ties"]})
display(pd.DataFrame(helped).set_index("habilidad"))

for pid, units in candidates:
    print(f"\n--- candidato · {pid} · {PROBES[pid][1].question}")
    print(wrap(f"store respondió: {store_rows[pid]['answer']['answer']}", indent="     "))
    print(wrap(f"store_filter respondió: {(filter_rows[pid]['answer'] or {}).get('answer')}", indent="     "))
    for u in units:
        print(wrap(f"sacado: {grounded.passage_block(u)}", indent="     ", max_lines=6))
run.note("memory_filter", {"changed": len(changed), "candidates": [pid for pid, _ in candidates]})

# %% [markdown]
r"""
**Lectura.** El filtro armado desde la memoria (la comunidad y el período del contrato) cambió los pasajes entregados en 27 de las 30 sondas, y el resultado en 2: ayudó en una actualización de conocimiento (sc02-p2) y perjudicó una abstención. **1 a 1: ningún efecto que leer.** `store_filter` prende dos piezas de la Etapa 3 a la vez (fecha y región); como pieza de la memoria es una sola variable, pero no se puede decir cuál de las dos movió cada sonda. Los candidatos a "el filtro sacó el pasaje que permitía decir *eso no está disponible para vos*" están impresos arriba para leerlos a mano.
"""

# %% [markdown]
r"""
## 11 · Costo contra exactitud: la predicción de Mem0

En la evaluación propia de Mem0, el contexto completo sacó más puntaje que el almacén leyendo unas 15 veces más tokens (Apéndice B.4): la memoria compró costo, no exactitud, **donde el historial entraba**. Acá hay dos regímenes, los de la sección 2:
- `replay` contra `store`: el historial entra entero;
- `replay_600` contra `store`: el historial no entra, y los dos tienen el mismo espacio.

La tabla pone, por brazo y siempre sobre las sondas y los escenarios de **test** (el mismo denominador en todas las columnas):
- lo que se paga **por sonda**: los tokens del prompt según el servidor, en una llamada;
- lo que se paga **una vez por sesión** para escribir el almacén, repartido entre las sondas.

El almacén solo gana en costo si la escritura se amortiza en suficientes sondas. El filtro no agrega llamadas: es una regla.
"""

# %%
test_sessions = {s.session_id for scenario in REPORT_SCENARIOS for s in scenario.sessions}
test_costs = costs[costs.index.isin(test_sessions)]
write_calls = int(test_costs["calls"].sum())
write_tokens = int(test_costs["prompt_tokens"].sum() + test_costs["output_tokens"].sum())
mem0 = pd.DataFrame([{
    "brazo": arm, "acierto (test)": summary.loc[arm, "acierto"],
    "tokens de prompt por sonda": summary.loc[arm, "prompt (servidor, media)"],
    "llamadas de escritura": write_calls if arm in memory.STORE_ARMS else 0,
    "tokens de escritura por sonda": round(write_tokens / len(REPORT)) if arm in memory.STORE_ARMS else 0,
} for arm in ARMS]).set_index("brazo")
display(mem0)
print(f"régimen: el historial más largo de test es de "
      f"{regime.loc[[s.scenario_id for s in REPORT_SCENARIOS], 'historial'].max()} tokens; la ventana para "
      f"historial es de {WINDOW}. Con los escenarios seguidos: "
      + "; ".join(f"{name}: {where}" for name, where in STOPS.items()))

# %% [markdown]
r"""
**Lectura: la predicción de Mem0.** Donde la reproducción entra, el almacén **gana en costo y no en exactitud**: `store` pone 404 tokens de memoria por sonda contra 847 de `replay` (y 3.950 tokens de prompt contra 4.367), con 10/22 contra 11/22. Es la dirección de Mem0 (Apéndice B.4), sin que la diferencia de exactitud supere el ruido. El ahorro tiene un costo que la lectura no ve: 66 llamadas de escritura para los 3 escenarios de test, unos 2.100 tokens por sonda. Con el mismo presupuesto (600 tokens), `store` y `replay_600` empatan (10/22, 4-4 por sonda).
"""

# %% [markdown]
r"""
## 12 · Una actualización de punta a punta

El primer escenario con un par de `knowledge_update`:
- qué operaciones cambiaron el almacén;
- qué recuerdos había en el espacio de memoria en cada sonda del par;
- qué respondió cada brazo.

La sonda de "antes" se hace entre las dos sesiones, así que tiene que ver el valor viejo como vigente.
"""

# %%
demo = next(s for s in SCENARIOS if sum(p.ability == "knowledge_update" for p in s.probes) >= 2)
st = stores[demo.scenario_id]
print(f"=== {demo.scenario_id}: operaciones que cerraron algo")
for op in st.ops:
    if op["closed"] or op["fallback"]:
        print(f"  {op['at'][:10]} {op['session']} {op['op']:<10} target={op['target_id']} new={op['new_id']} "
              f"closed={op['closed']} fallback={op['fallback']} · {op['fact']['subject']}: "
              f"{clip(op['fact']['text'], 100)}")
for probe in [p for p in demo.probes if p.ability == "knowledge_update"]:
    print(f"\n--- {probe.probe_id} · {probe.asked_at:%Y-%m-%d} · veredicto esperado {probe.expected_verdict} · "
          f"clave {probe.answer_key}")
    print(wrap(f"P: {probe.question}"))
    print("  espacio de memoria (store):")
    print(wrap(SLOTS["store"][probe.probe_id]["text"], indent="     "))
    for arm in ARMS:
        r = next((r for r in log.rows(arm) if r["item_id"] == probe.probe_id), None)
        if r is None or r["answer"] is None:
            print(f"  {arm}: error {r and r['error']}")
            continue
        print(f"  {arm}: correct={r['score']['correct']} can_answer={r['answer']['can_answer']} "
              f"clave en el espacio={r['memory']['key_in_slot']}")
        print(wrap(r["answer"]["answer"], indent="     "))

# %% [markdown]
r"""
## Para el informe

- **¿En qué habilidades la memoria le ganó a no tener memoria, y en cuáles la reproducción le ganó a la memoria?** La memoria, en las actualizaciones de conocimiento; la reproducción, en la extracción y lo multi-sesión. Nada supera el ruido (sección 9).
- **Una actualización de punta a punta:** sección 12.
- **¿Qué guardó el extractor que no era cierto, o que no se dijo?** En la auditoría, de 76 registros: 10 mal y 6 que nunca se dijeron (en test, 3 y 3). Etiquetas equivocadas, fechas de la sesión tomadas como datos y no-hechos que invalidan preferencias reales (sección 5). Los chequeos automáticos no los detectan.
- **¿Alguna memoria mejoró la recuperación como filtro, o la perjudicó?** Una sonda de cada lado (sección 10).
- **¿En qué punto deja de entrar el historial?** Con estos escenarios, nunca en la ventana: los cuatro seguidos suman 3.629 tokens y hay 3.794. Con un presupuesto de 3.000 deja de entrar en el cuarto escenario; con 600, en la tercera sesión del primero (sección 2).
"""

# %%
run.note("rows", {arm: len(log.rows(arm)) for arm in ARMS})
run.finish()
