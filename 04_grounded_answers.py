# %% [markdown]
r"""
# Etapa 4 · Respuestas fundamentadas con un presupuesto en el que se puede confiar

La recuperación fija el techo y la generación decide cuánto de ese techo se alcanza. Esta notebook mide esa segunda superficie y construye la pieza que reusan las Etapas 5 y 6: **una ventana armada dentro de un presupuesto de tokens, con una cuenta verificada contra la del servidor en cada llamada.**

| Entra | Sale |
|---|---|
| `documents.jsonl`, el eval set congelado, el índice de la Etapa 3 y su mejor configuración (`index/best_config.json`) | `results/04_grounded_answers.jsonl` (una fila por pregunta y brazo), `llm_cache/`, `run_report.json` |

La lógica vive en `src/`:
- `grounded.py`: el pasaje, la entrega, la llamada y el puntaje sin juez. Lo reusan las Etapas 5 y 6;
- `index_loader.py`: carga el índice de la Etapa 3 y verifica que sea el mismo sobre el que se eligió la mejor configuración;
- `retrieval.py`, `chunking.py` y `evaluation.py`, de la Etapa 3: la recuperación, la entrega por presupuesto y la definición de "le llegó la respuesta";
- `stats.py`: las cotas de ruido.

**Decidido antes de correr** (todo en `decision_log.md`):
- `qwen3:8b`, `think=False`, temperatura 0, `num_ctx = 8.192` y `num_predict = 768` en cada llamada: el prompt entra en 7.424 tokens.
- **El pasaje es el de la Etapa 3**: su cabecera fija (norma, sección, vigencia) y su texto, con el id citable del bloque adelante. **La entrega también**: en orden de ranking, lo que no entra se saltea y se sigue. Cada pasaje se cuenta tal como lo ve el generador, id incluido, contra el presupuesto de la mejor configuración (3.000 tokens).
- **Tres brazos, en este orden y con el modelo descargado al empezar:** libro cerrado, recuperación y oráculo. La salida depende del estado de la caché de prompts del servidor (decision_log.md, «Temperatura 0 no alcanza»), así que el orden es parte de la configuración.
- **Se reporta sobre `test`** (30 preguntas). Las 10 de `dev` corren igual.
"""

# %%
import environment  # el del curso, primero: guarda de JAX, caché de HF, UTF-8

import os
import random
import sys
import time
from collections import Counter
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
import doc_schema
import evalset
import grounded
import index_loader
import stats
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
ITEMS = {item.item_id: item for item in ev.items}
REPORT = [item for item in ev.items if SMOKE_TEST or item.split == "test"]

MODEL, NUM_CTX, NUM_PREDICT = grounded.MODEL, grounded.NUM_CTX, grounded.NUM_PREDICT
ARMS = ["closed_book", "retrieval", "oracle"]  # el orden de corrida, fijo
SYSTEM = grounded.system_prompt(SNAPSHOT)
SYSTEM_CLOSED = grounded.system_prompt(SNAPSHOT, closed_book=True)

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)  # como script, pandas corta las tablas anchas
environment.summary()
print(f"corpus_sha {CORPUS_SHA[:12]}... | eval {ev.key()} | {ev.summary()} | se reporta sobre {len(REPORT)}")

# %% [markdown]
r"""
## 1 · El recuperador de la Etapa 3

`index_loader.load` carga los hijos, los padres, los embeddings (de la caché de la Etapa 3), BM25, el reranker y los metadatos del filtro, y lee la mejor configuración. Antes se descarga el generador: en 8 GB no entran los dos juntos. **Si la clave del índice no coincide con la de `best_config.json`, falla**: la Etapa 3 eligió sobre otro índice y hay que volver a correrla.

Se recupera para las 40 preguntas antes de generar, y después el embedder y el reranker pasan a CPU: la GPU queda para el generador.
"""

# %%
llm.unload(MODEL)  # libera la VRAM para el embedder y el reranker
t0 = time.time()
LOADED = index_loader.load(ART, DOCS, ROOT, ev.key())
UNITS = LOADED.units
CONFIG, BUDGET = LOADED.config, LOADED.config.budget_tokens
print(f"índice {LOADED.key}: {len(LOADED.index.children):,} hijos, {len(LOADED.index.parents):,} padres "
      f"({time.time() - t0:.0f} s)")
print(f"mejor configuración de la Etapa 3: {LOADED.best['row']} -> {asdict(CONFIG)}")
print(f"  clave entregada desde el gold en test, por fila: {LOADED.best['primary_on_test']} (de {LOADED.best['n_test']})")

GOLD = {item.item_id: grounded.Gold.of(item, SECTIONS, UNITS) for item in ev.items}
missing = [item.item_id for item in ev.items if item.answerable and not GOLD[item.item_id].texts]
assert not missing, f"preguntas cuyo gold no tiene ninguna unidad en el índice: {missing}"

t0 = time.time()
RANKED = {item.item_id: index_loader.ranked_units(LOADED, item.question)[0] for item in ev.items}
print(f"{len(RANKED)} recuperaciones en {time.time() - t0:.1f} s")
LOADED.release_gpu()
run = StageRun.start("04_grounded_answers", ART, profile=ART.name,
                     config={"model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "passage_budget": BUDGET,
                             "arms": ARMS, "eval": ev.key(), "index": LOADED.key, "retrieval": LOADED.best["row"]})

# %% [markdown]
r"""
## 2 · La entrega y el presupuesto

Cada brazo entrega:
- **libro cerrado**: nada, y reglas propias en el prompt ("respondé con lo que sepas"). Con el prompt fundamentado y sin pasajes, el modelo se abstendría siempre, y la fila no mediría lo que tiene que medir: cuánto del eval set responde el preentrenamiento (§7.4);
- **recuperación**: el ranking de la mejor configuración, entregado hasta el presupuesto;
- **oráculo**: los padres de las secciones gold (la Etapa 3 los corta en ≤ 1.200 tokens), primero los que contienen la clave, con el mismo presupuesto. En las preguntas sin respuesta no hay gold, así que reciben los mismos pasajes que la fila de recuperación: sin pasajes, abstenerse sería trivial.

**La cuenta.** Cada pasaje se cuenta con el tokenizer de `Qwen/Qwen3-8B`, con su id citable, su cabecera y su texto (`grounded.as_passages`). La Etapa 3 cuenta cabecera y texto; el id suma unos 13 tokens por pasaje, así que esta etapa entrega a veces un pasaje menos que la Etapa 3 con el mismo presupuesto. El prompt entero lo cuenta `llm.chat` con la plantilla de chat antes de mandar, y lo compara con `prompt_eval_count` después.

**"Le llegó la respuesta"** se mide con las definiciones de la Etapa 3 (`evaluation.delivered_metrics`): un pasaje es del gold si es de una sección gold, de una versión con el mismo texto, o si su texto es el de una unidad gold.
"""

# %%
count = grounded.token_counter(MODEL)
longest = max(ev.items, key=lambda item: count(item.question))
overhead = llm.count_prompt_tokens(grounded.build_messages(SYSTEM, longest.question, []), MODEL)
print(f"prompt sin pasajes (la pregunta más larga): {overhead} tokens · + presupuesto {BUDGET} = "
      f"{overhead + BUDGET} de {grounded.PROMPT_LIMIT} disponibles")
assert overhead + BUDGET <= grounded.PROMPT_LIMIT, "el presupuesto de pasajes no entra en num_ctx − num_predict"


def oracle_units(item: evalset.EvalItem) -> list:
    """Los padres de las secciones gold, en el orden del gold; dentro de cada una, primero los que tienen la clave."""
    parents = []
    for sid in item.gold_sections:
        mine = [u for u in LOADED.index.parents.values() if u.section_id == sid]
        parents += sorted(mine, key=lambda u: not evalset.contains_key(grounded.passage_block(u), item.answer_key))
    return parents


DELIVERED: dict[str, dict[str, list | None]] = {arm: {} for arm in ARMS}
for item in ev.items:
    retrieved = grounded.deliver(RANKED[item.item_id], BUDGET, count)
    DELIVERED["closed_book"][item.item_id] = None
    DELIVERED["retrieval"][item.item_id] = retrieved
    DELIVERED["oracle"][item.item_id] = (grounded.deliver(oracle_units(item), BUDGET, count) if item.answerable
                                         else retrieved)

rows = []
for arm in ["retrieval", "oracle"]:
    deliveries = [grounded.delivery(item, DELIVERED[arm][item.item_id], GOLD[item.item_id])
                  for item in REPORT if item.answerable]
    used = [grounded.used_tokens(DELIVERED[arm][item.item_id]) for item in REPORT]
    rows.append({"brazo": arm,
                 "pasajes (media)": round(sum(len(DELIVERED[arm][i.item_id]) for i in REPORT) / len(REPORT), 1),
                 "tokens (media)": round(sum(used) / len(used)), "tokens (máx)": max(used),
                 **{k: f"{sum(bool(d[k]) for d in deliveries)}/{len(deliveries)}"
                    for k in ("doc_hit", "section_delivered", "answer_delivered", "gold_key_delivered")}})
display(pd.DataFrame(rows).set_index("brazo"))
print(f"la Etapa 3 midió, para su fila {LOADED.best['row']}: "
      f"{LOADED.best['primary_on_test'][LOADED.best['row']]}/{LOADED.best['n_test']} con la clave entregada desde el gold")
oracle_misses = [item.item_id for item in ev.items if item.answerable and not grounded.delivery(
    item, DELIVERED["oracle"][item.item_id], GOLD[item.item_id])["gold_key_delivered"]]
print("preguntas en las que ni el oráculo entrega la clave:", oracle_misses or "ninguna")
run.note("oracle_misses", oracle_misses)

# %% [markdown]
r"""
**Lectura.**
- **La recuperación entrega lo mismo que midió la Etapa 3:** la clave desde el gold en 16 de 26 preguntas con respuesta de test, igual que la fila `+prefix`. El id citable (unos 13 tokens por pasaje) no costó ningún acierto: se entregan 9,9 pasajes y 2.968 tokens en promedio.
- **Documento, sección y respuesta están lejos:** llega algún pasaje del documento gold en 25 de 26, uno de la sección gold en 16 y la clave en cualquier pasaje en 19. Un acierto de documento predice mal la Etapa 4: en 9 preguntas llegó la norma correcta pero no el artículo.
- **El oráculo no está igualado en tokens:** recibe 2,4 pasajes y 1.086 tokens en promedio, porque el presupuesto es un tope, no un relleno. La brecha oráculo-recuperación mezcla "el pasaje correcto" con "menos distractores".
"""

# %% [markdown]
r"""
## 3 · El servidor

`llm.setup` levanta el servidor, verifica el digest del modelo, prueba que respete `think=False`, mira en qué orden escribe las claves del JSON y calibra nuestra cuenta contra `prompt_eval_count`. Después se activa la caché de llamadas y se descarga el modelo, para que la corrida arranque en frío.
"""

# %%
SERVER = llm.setup(MODEL, num_ctx=NUM_CTX)
llm.use_cache(ART / "llm_cache" / "qwen3_8b.jsonl")
run.note("server", {k: SERVER.get(k) for k in ("digest", "server", "key_order", "thinking", "calibration")})

# %% [markdown]
r"""
## 4 · Los tres brazos

Cada fila de `results/04_grounded_answers.jsonl` guarda:
- la configuración y el hash del eval set (en el `run_spec` de `store.ResultsLog`), con la clave del índice, la mejor configuración, la versión del servidor, el digest y el hash del schema y de la plantilla del prompt;
- los pasajes entregados y su cuenta de tokens;
- la respuesta entera, con su `analysis`;
- el registro de la llamada: la cuenta de la plantilla, `prompt_eval_count`, `done_reason`, el error si lo hubo y la latencia.

Una corrida interrumpida retoma donde quedó; un error transitorio (el servidor caído) se vuelve a intentar.
"""

# %%
RUN_SPEC = {"stage": "04", "eval": ev.key(), "corpus_sha": CORPUS_SHA, "model": MODEL, "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT, "passage_budget": BUDGET, "arms": ARMS,
            "index": LOADED.key, "retrieval": {"row": LOADED.best["row"], "config": asdict(CONFIG)},
            "server": SERVER["server"], "digest": SERVER["digest"],
            "prompt": grounded.prompt_spec(SYSTEM), "prompt_closed_book": grounded.prompt_spec(SYSTEM_CLOSED)}
log = store.ResultsLog(ART / "results" / "04_grounded_answers.jsonl", run_spec=RUN_SPEC)

llm.unload(MODEL)  # arranque en frío: ver decision_log.md, «Temperatura 0 no alcanza»
t0 = time.time()
for arm in ARMS:
    system = SYSTEM_CLOSED if arm == "closed_book" else SYSTEM
    for item in ev.items:
        if log.done(arm, item.item_id):
            continue
        delivered = DELIVERED[arm][item.item_id]
        answer, call = grounded.generate(grounded.build_messages(system, item.question, delivered))
        scored = grounded.score(answer, call, item, delivered, GOLD[item.item_id])
        # "error" en el primer nivel: store.ResultsLog lo lee ahí para reintentar al retomar un error
        # transitorio (el servidor caído), en lugar de dejarlo como resultado para siempre.
        log.write({"arm": arm, "item_id": item.item_id, "split": item.split, "category": item.category,
                   "error": scored.get("error"),
                   "delivered": [[u.unit_id, u.n_gen] for u in delivered] if delivered is not None else None,
                   "answer": answer.model_dump() if answer else None, "score": scored, "call": call})
    print(f"{arm}: listo ({time.time() - t0:.0f} s acumulados)")
print("llamadas:", {k: v for k, v in llm.STATS.items() if v})
print("dónde corre:", llm.placement(MODEL))

# %% [markdown]
r"""
## 5 · Las tres filas

- **Acierto**: respondió (`can_answer`), con una clave en `answer` y, si es de sí/no, con el veredicto correcto. En una pregunta fuera del corpus, acierta si y solo si se abstiene.
- **Errores**: parseo, truncamiento u overflow. Son su propia categoría: no entran en el denominador del acierto, y se cuentan aparte.
- **Se abstuvo teniéndola**: `can_answer = false` con un pasaje del gold que contiene la clave en su contexto.
- **Veredicto**: en las preguntas de sí/no, el veredicto solo, aparte de la clave.
- **Citó sin recibir**: citó un bloque que no se le entregó. **Citó gold**: citó un bloque del que se le entregó un pasaje del gold.
- **Prompt (servidor)**: `prompt_eval_count`. La última columna es la mayor diferencia entre nuestra cuenta con la plantilla y la del servidor.

El puntaje se recalcula al leer, a partir de la respuesta guardada: si se corrige la vara (`grounded.score`), las tablas la usan entera, sin mezclar filas puntuadas con la vieja.
"""


# %%
def rows_of(arm: str, items: list) -> list[dict]:
    ids = {item.item_id for item in items}
    rows = []
    for r in log.rows(arm):
        if r["item_id"] not in ids:
            continue
        item, delivered = ITEMS[r["item_id"]], DELIVERED[arm][r["item_id"]]
        answer = grounded.GroundedAnswer.model_validate(r["answer"]) if r["answer"] else None
        rows.append({**r, "score": grounded.score(answer, r["call"], item, delivered, GOLD[r["item_id"]]),
                     "delivery": grounded.delivery(item, delivered or [], GOLD[r["item_id"]])})
    return rows


def correct_map(arm: str, items: list) -> dict[str, bool]:
    return {r["item_id"]: r["score"]["correct"] for r in rows_of(arm, items) if r["score"]["outcome"] != "error"}


def arm_table(items: list) -> pd.DataFrame:
    table = []
    for arm in ARMS:
        rs = rows_of(arm, items)
        scored = [r for r in rs if r["score"]["outcome"] != "error"]
        answerable = [r for r in scored if ITEMS[r["item_id"]].answerable]
        unanswerable = [r for r in scored if not ITEMS[r["item_id"]].answerable]
        yes_no = [r for r in answerable if ITEMS[r["item_id"]].expected_verdict]
        calls = [r["call"] for r in rs if r["call"].get("prompt_tokens") is not None]
        fresh = [c for c in calls if not c.get("cached")]
        mean, lo, hi = stats.bootstrap_ci(correct_map(arm, items))
        table.append({
            "brazo": arm,
            "acierto": f"{sum(r['score']['correct'] for r in scored)}/{len(scored)}",
            "IC 95%": f"{lo:.0%}-{hi:.0%}",
            "con respuesta": f"{sum(r['score']['correct'] for r in answerable)}/{len(answerable)}",
            "abstención correcta": f"{sum(r['score']['correct'] for r in unanswerable)}/{len(unanswerable)}",
            "se abstuvo": sum(r["score"]["outcome"] == "abstained" for r in scored),
            "se abstuvo teniéndola": sum(r["score"]["outcome"] == "abstained" and bool(r["delivery"]["gold_key_delivered"])
                                         for r in answerable),
            "veredicto": f"{sum(bool(r['score']['verdict_ok']) for r in yes_no)}/{len(yes_no)}",
            "errores": len(rs) - len(scored),
            "citó sin recibir": sum(bool(r["score"]["cited_not_delivered"]) for r in scored if arm != "closed_book"),
            "citó gold": sum(bool(r["score"]["cited_gold"]) for r in answerable if arm != "closed_book"),
            "flags": sum(r["score"]["flag_bool_says_abstain_text_answers"] or r["score"]["flag_bool_says_answer_text_abstains"]
                         for r in scored),
            "prompt (servidor)": round(sum(c["prompt_tokens"] for c in calls) / max(len(calls), 1)),
            "s/llamada": round(sum(c["ms"] for c in fresh) / 1000 / max(len(fresh), 1), 1),
            "máx |plantilla − servidor|": max((abs(c["expected_prompt_tokens"] - c["prompt_tokens"]) for c in calls
                                               if c.get("expected_prompt_tokens") is not None), default=None),
        })
    return pd.DataFrame(table).set_index("brazo")


display(arm_table(REPORT))
by_category = pd.DataFrame({arm: {cat: stats.count({k: v for k, v in correct_map(arm, REPORT).items()
                                                    if ITEMS[k].category == cat})
                                  for cat in evalset.CATEGORIES} for arm in ARMS})
display(by_category)

# %% [markdown]
r"""
### El ruido, antes de afirmar nada

Por cada par de brazos: victorias, derrotas y empates pareados, el test de signos (= McNemar exacto) sobre las discordantes y un intervalo bootstrap de la diferencia. Con n = 30, una fila sola tiene unos ±12 puntos de ruido.
"""

# %%
noise = []
for a, b in [("closed_book", "retrieval"), ("retrieval", "oracle"), ("closed_book", "oracle")]:
    ca, cb = correct_map(a, REPORT), correct_map(b, REPORT)
    p = stats.paired(ca, cb)
    diff, lo, hi = stats.bootstrap_diff_ci(ca, cb)
    noise.append({"comparación (B vs A)": f"{b} vs {a}", "n": p["n"], "gana B": p["wins"], "gana A": p["losses"],
                  "empates": p["ties"], "p (signos)": round(p["p_sign"], 4),
                  "diferencia B − A": f"{diff:+.0%} ({lo:+.0%} a {hi:+.0%})"})
display(pd.DataFrame(noise).set_index("comparación (B vs A)"))

# %% [markdown]
r"""
**Lectura** (test, 30 preguntas; ±12 puntos de ruido por fila).
- **Libro cerrado: 5/30**, y 3 de las 26 con respuesta. Lejos de "la mitad" de §7.4: el eval set mide recuperación, no preentrenamiento. Se abstuvo en 7.
- **Recuperación contra libro cerrado: 14 victorias y 1 derrota** (p = 0,001; +43 puntos, IC +23 a +63). Es la única afirmación de esta tabla que no depende de cómo se lea el ruido.
- **Oráculo contra recuperación: 7 y 0** (p = 0,016; +23 puntos, IC +10 a +40). Hay margen de recuperación por ganar, con la salvedad de los tokens de la sección 2.
- **El oráculo no llega a 26/26: 22.** Con la respuesta en el contexto, el generador falla en 4.
- **Citas:** ningún brazo citó un bloque que no se le entregó (0 en 60 respuestas con pasajes), y ninguna respuesta se contradijo entre el booleano y el texto.
- **Sin respuesta (fuera del corpus):** 3 de 4 abstenciones correctas con pasajes, 2 de 4 a libro cerrado. Con 4 preguntas, no hay diferencia que leer.
- **Por categoría** (direcciones, no pruebas): la recuperación sube sobre todo `topical` (1/13 → 10/13); `identifier` queda en 2/6, y el oráculo lo lleva a 5/6. Ahí la falla es de recuperación.
"""

# %% [markdown]
r"""
## 6 · La tabla 2×2 de la fila de recuperación

Por pregunta con respuesta: **¿recuperó bien?** (llegó un pasaje del gold que contiene la clave: la primaria de la Etapa 3) × **¿respondió bien?** Separa las dos fallas que importan:
- *fiel al pasaje equivocado*: recuperó mal, respondió (`can_answer`) y **citó un pasaje entregado**. Si respondió mal sin citar nada de lo entregado, la respuesta salió de otro lado (el preentrenamiento), y va aparte;
- *lo tenía, respondió otra cosa*: recuperó bien y respondió mal.

Las fallas se leen acá sobre test para el informe. El prompt quedó fijo antes de la corrida: lo que se lea no vuelve a ajustarlo.
"""

# %%
cells = Counter()
examples: dict[str, list[dict]] = {}
for r in rows_of("retrieval", REPORT):
    item = ITEMS[r["item_id"]]
    if not item.answerable or r["score"]["outcome"] == "error":
        continue
    if r["delivery"]["gold_key_delivered"]:
        cell = "recuperó bien · respondió bien" if r["score"]["correct"] else "recuperó bien · respondió mal"
    elif r["score"]["correct"]:
        cell = "recuperó mal · respondió bien"
    elif r["score"]["outcome"] == "abstained":
        cell = "recuperó mal · se abstuvo"
    elif set(r["score"]["cited"]) - set(r["score"]["cited_not_delivered"]):
        cell = "recuperó mal · fiel al pasaje equivocado"
    else:
        cell = "recuperó mal · respondió sin apoyarse en lo entregado"
    cells[cell] += 1
    examples.setdefault(cell, []).append(r)
display(pd.Series(cells, name="preguntas").sort_index())

for cell in ["recuperó bien · respondió mal", "recuperó mal · fiel al pasaje equivocado",
             "recuperó mal · respondió sin apoyarse en lo entregado"]:
    for r in examples.get(cell, []):
        item = ITEMS[r["item_id"]]
        print(f"\n--- {cell} · {item.item_id} [{item.category}] can_answer={r['answer']['can_answer']}")
        print(wrap(f"P: {item.question}"))
        print(wrap(f"R: {r['answer']['answer']}"))
        print(wrap(f"análisis: {r['answer']['analysis']}"))
        print(f"   clave: {item.answer_key} · veredicto esperado: {item.expected_verdict} · "
              f"citó: {r['score']['cited']}")

# %% [markdown]
r"""
**Lectura.** De las 16 preguntas en las que llegó la clave desde el gold, **13 se respondieron bien**. Las 3 fallas son de dos tipos:
- **La versión equivocada (md02, md05).** El pasaje correcto estaba, junto con la versión siguiente del mismo artículo, y el modelo aplicó la siguiente a una fecha anterior a su vigencia: para un contrato de febrero de 2019 usó el texto "vigente desde marzo de 2019"; para marzo de 2023, el de junio. Leer intervalos de vigencia es una tarea de razonamiento temporal que el modelo hace mal, no un problema de recuperación.
- **Una paráfrasis (tp05).** La respuesta es correcta en sustancia ("le toca al casero arreglarla") pero no usa la palabra de la clave ("reparaciones"). Es un falso negativo de la contención: lo lee el juez de la Etapa 7.

Cuando la clave no llegó (10 preguntas): 3 abstenciones, 4 respuestas fieles a un pasaje equivocado, 1 sin apoyarse en lo entregado y 2 aciertos igual.
"""

# %% [markdown]
r"""
## 7 · La guarda, a propósito

Se arma un prompt demasiado grande: la misma pregunta con 11.000 tokens de pasajes, en el mismo `num_ctx` (8.192, así no se recarga el modelo).

1. Con la guarda puesta, `llm.chat` cuenta el prompt antes de mandar y lanza `ContextOverflow`. `grounded.generate` lo atrapa y lo registra como error de llamada.
2. Con `strict_ctx=False`, se manda igual. El servidor corta el medio del prompt (se quedan los primeros tokens y la cola) y el modelo responde igual.
"""

# %%
victim = next(item for item in REPORT if item.answerable and item.category == "topical")
huge = grounded.deliver(RANKED[victim.item_id], 11_000, count)
huge_messages = grounded.build_messages(SYSTEM, victim.question, huge)
_, guarded = grounded.generate(huge_messages)
print(f"con la guarda: {len(huge)} pasajes, {grounded.used_tokens(huge)} tokens -> {guarded['error']}")
assert str(guarded["error"]).startswith("CONTEXT_OVERFLOW"), "la demo tiene que disparar la guarda"

r = llm.chat(huge_messages, model=MODEL, num_ctx=NUM_CTX, num_predict=NUM_PREDICT, strict_ctx=False, cache=False,
             format=grounded.GroundedAnswer.model_json_schema())
print(f"\nsin la guarda: nuestra cuenta {r.expected_prompt_tokens} · el servidor evaluó {r.prompt_tokens} · "
      f"error: {r.error}")
try:
    truncated_answer = grounded.GroundedAnswer.model_validate_json(r.text)
    print(wrap(f"respondió (can_answer={truncated_answer.can_answer}): {truncated_answer.answer}"))
    print(f"clave {victim.answer_key} en answer: {evalset.contains_key(truncated_answer.answer, victim.answer_key)}")
except ValueError:
    print(wrap(f"respondió algo que no parsea: {clip(r.text, 700)}"))
run.note("guard_demo", {"item": victim.item_id, "passage_tokens": grounded.used_tokens(huge),
                        "expected": r.expected_prompt_tokens, "server": r.prompt_tokens, "error": r.error,
                        "client_guard": guarded["error"]})

# %% [markdown]
r"""
**Lectura.** Con la guarda, el prompt de 11.239 tokens no se manda. Sin ella, **el servidor evaluó 4.098 tokens**: no se quedó con 8.192, sino con la mitad de la ventana (los primeros tokens y la cola). El modelo igual respondió, con `can_answer = true`, una respuesta fluida y sin la clave: lo que se fue fueron las instrucciones y los pasajes mejor rankeados. Nada en la respuesta lo delata; lo detecta la comparación de nuestra cuenta con `prompt_eval_count` (`SILENT_TRUNCATION`).
"""

# %% [markdown]
r"""
## 8 · ¿Cuánto cambia la misma pregunta?

La caché de llamadas congela la primera salida, así que una re-corrida da idéntico por construcción y esconde la varianza (decision_log.md, «Temperatura 0 no alcanza»). Para medirla, se descarga el modelo y se vuelven a correr 10 preguntas de test de la fila de recuperación **en orden inverso y sin caché**. Son una muestra con semilla repartida entre las categorías. Cada respuesta que cambia es parte del piso de ruido.
"""

# %%
rng = random.Random(0)
by_cat: dict[str, list] = {}
for item in REPORT:
    by_cat.setdefault(item.category, []).append(item)
for group in by_cat.values():
    rng.shuffle(group)
sample = [group[i] for i in range(max(map(len, by_cat.values()))) for group in by_cat.values() if i < len(group)][:10]
llm.unload(MODEL)
changes = []
for item in reversed(sample):
    before = rows_of("retrieval", [item])[0]
    delivered = DELIVERED["retrieval"][item.item_id]
    answer, call = grounded.generate(grounded.build_messages(SYSTEM, item.question, delivered), cache=False)
    after = grounded.score(answer, call, item, delivered, GOLD[item.item_id])
    changes.append({"id": item.item_id,
                    "mismo texto": bool(answer) and before["answer"] is not None
                    and answer.answer == before["answer"]["answer"],
                    "mismo puntaje": after.get("correct") == before["score"].get("correct"),
                    "antes": before["score"].get("correct"), "después": after.get("correct")})
changes = pd.DataFrame(changes).set_index("id")
display(changes)
print(f"textos distintos: {(~changes['mismo texto']).sum()}/{len(changes)} · "
      f"puntajes distintos: {(~changes['mismo puntaje']).sum()}/{len(changes)}")
run.note("rerun", {"n": len(changes), "text_changed": int((~changes["mismo texto"]).sum()),
                   "score_changed": int((~changes["mismo puntaje"]).sum())})

# %% [markdown]
r"""
**Lectura.** Al volver a correr 10 preguntas en frío, en orden inverso y sin caché, **cambió el texto de 7 respuestas y el puntaje de ninguna**. La salida no es determinista con temperatura 0 (decision_log.md, «Temperatura 0 no alcanza»), pero a esta escala la variación es de redacción, no de veredicto. El piso de ruido de las tablas lo siguen fijando los ±12 puntos de n = 30.

## Para el informe

- **De las preguntas en las que llegó el pasaje con la respuesta, ¿cuántas se respondieron bien?** 13 de 16; las fallas, en la sección 6.
- **¿Qué tan grande es la brecha oráculo-recuperación y libro cerrado-recuperación?** +23 puntos (7-0) y +43 (14-1), las dos por encima del ruido. La primera con menos tokens en el oráculo.
- **¿Con qué frecuencia citó un pasaje que no se le dio, o se abstuvo teniendo la respuesta?** Nunca citó uno no entregado. Se abstuvo con la clave en el contexto 0 veces en recuperación y 1 en el oráculo.
- **Un prompt demasiado grande:** sección 7.
- **¿Importó la posición?** No se midió (recortado: decision_log.md, sección 3).
"""

# %%
run.note("rows", {arm: len(log.rows(arm)) for arm in ARMS})
run.finish()
