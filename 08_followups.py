# %% [markdown]
r"""
# Etapa 8 · Seguimiento: lo que la revisión de la entrega dejó sin medir

La revisión del trabajo entregado (2026-10-26, `decision_log.md`, sección 6) encontró siete puntos en los que el informe se apoyaba en algo que no se midió, o que se midió con un instrumento que no correspondía. Esta notebook mide los seis primeros (los dos primeros sin LLM) y prepara el séptimo, que tiene que hacerlo una persona:

| # | Lo que el informe afirmaba o dejaba abierto | Qué se mide acá | Sección |
|---|---|---|---|
| 1 | La primaria de la Etapa 3 no se podía recalcular desde el zip (necesita el texto de los pasajes) | Las métricas por ítem de las 9 filas, exportadas a `results/03_retrieval_metrics.jsonl` | 1 |
| 2 | La mejor configuración se eligió **sobre test** | La misma regla aplicada en `dev`, y su estabilidad en `test` con un bootstrap | 1 |
| 3 | "Recuperar la versión correcta (el filtro de fecha) no alcanza si el generador no razona sobre intervalos": la Etapa 4 nunca corrió con el filtro | Un brazo `retrieval_date` (la mejor fila + el filtro de fecha, una variable) y qué versiones del artículo le llegan al generador | 1 y 2 |
| 4 | El oráculo recibía 1.086 tokens contra 2.968 de la recuperación: su ventaja mezclaba "el pasaje correcto" con "menos distractores" | Un brazo `oracle_padded`: el gold más distractores hasta los mismos 3.000 tokens | 2 |
| 5 | La posición del pasaje no se midió | El gold primero, en el medio o último entre ~6.000 tokens de distractores | 3 |
| 6 | El resumen de la Etapa 5 se congelaba en su tope de 220 caracteres (30 de 58 actualizaciones): "no paga su costo" era sobre un resumidor roto | Un resumidor v2 con el mismo tope de historial | 4 |
| 7 | Las auditorías (juez, memoria, resúmenes, reescrituras) las hizo un modelo, no una persona | Las planillas ciegas para una persona, y el acuerdo persona-modelo cuando estén completas | 5 |

**Decidido antes de correr** (y registrado en `decision_log.md`, sección 6, antes de la corrida):
- **Nada de esto cambia la configuración ni el eval set.** La mejor fila sigue siendo `+prefix`, el eval set sigue congelado (`99676007c42e6f6e`) y las filas guardadas de las Etapas 3-7 no se tocan. Lo nuevo va a archivos nuevos (`results/08_*.jsonl`).
- **Cada brazo nuevo cambia una sola cosa** contra un brazo ya guardado, con el mismo prompt, el mismo `num_ctx = 8.192`, `num_predict = 768`, temperatura 0 y el modelo descargado al empezar cada brazo, en orden fijo.
- **Antes de generar, se verifica que el pipeline reproduzca lo guardado:** la recuperación, las entregas de la Etapa 4 y las de la ventana de la Etapa 5 tienen que dar exactamente las mismas unidades y la misma cuenta de tokens. Si no, la etapa se corta: compararía contra filas hechas con otra cosa.
- **Cómo se lee:** se reporta sobre `test`. Cada comparación, con victorias y derrotas pareadas y el test de signos (por guion en la Etapa 5). Una diferencia con p ≥ 0,05 se reporta como dirección, no como efecto, igual que en el resto del informe.
- **Presupuesto:** unas 40 + 40 llamadas (Etapa 4), 78 (posición) y ~120 (resumen v2, con dev y la sonda): ~280 llamadas, unos 10 minutos en la RTX 5070 a 1-2,5 s por respuesta.
"""

# %%
import environment  # el del curso, primero: guarda de JAX, caché de HF, UTF-8

import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace

ROOT = environment.ROOT
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

try:
    from IPython.display import display  # en la notebook
except ImportError:
    display = print  # como script (tools/run_stages.py)

import llm
import store
from console import StageRun, wrap

import artifact_io
import conversation
import doc_schema
import evalset
import evaluation
import followups
import grounded
import index_loader
import judge
import memory
import retrieval
import stats
from corpus import SNAPSHOT

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
ART = environment.artifacts_dir(SMOKE_TEST)
RESULTS = ART / "results"
EVAL_DIR = ROOT / "eval" / "smoke" if SMOKE_TEST else ROOT / "eval"
DOCS_PATH = ART / "documents.jsonl"
DOCS = doc_schema.load_documents(DOCS_PATH)
CORPUS_SHA = doc_schema.documents_hash(DOCS)
artifact_io.check_manifest(ART / "documents.manifest.json", {"corpus_sha": CORPUS_SHA})
if SMOKE_TEST:
    ev = evalset.load_eval(EVAL_DIR, require_frozen=False)
    evalset.validate_eval(ev, DOCS).show().raise_if_failed()
else:
    ev = evalset.load_eval(EVAL_DIR, DOCS_PATH)
SECTIONS = evalset.section_index(DOCS)
ITEMS = {item.item_id: item for item in ev.items}
REPORT = [item for item in ev.items if SMOKE_TEST or item.split == "test"]
REPORT_IDS = {item.item_id for item in REPORT}

MODEL, NUM_CTX, NUM_PREDICT = grounded.MODEL, grounded.NUM_CTX, grounded.NUM_PREDICT
SYSTEM = grounded.system_prompt(SNAPSHOT)
POSITION_BUDGET = 6000  # tokens de pasajes del experimento de posición (sección 3)

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
environment.summary()
print(f"corpus_sha {CORPUS_SHA[:12]}... | eval {ev.key()} | {ev.summary()} | se reporta sobre {len(REPORT)}")


def saved_rows(name: str, index_key: str | None = None, every_spec: bool = False) -> dict[tuple[str, str], dict]:
    """Las filas guardadas de una etapa anterior, una por (brazo, ítem): la última escrita.

    Solo las de una especificación con este eval set (y este índice, si la especificación lo nombra). Con
    ``every_spec=False``, de la especificación más reciente; la Etapa 3 tiene una por fila, y ahí van todas.
    """
    path = RESULTS / f"{name}.jsonl"
    specs = {}
    for spec_path in RESULTS.glob(f"{name}.config-*.json"):
        record = store.read_json(spec_path)
        spec = record["spec"]
        if spec.get("eval") == ev.key() and spec.get("index", index_key) == index_key:
            specs[spec_path.stem.split("config-", 1)[1]] = record["first_seen"]
    if not specs:
        raise FileNotFoundError(f"no hay filas de {name} con el eval set {ev.key()} y el índice {index_key}")
    keep = set(specs) if every_spec else {max(specs, key=specs.get)}
    rows = {}
    for row in store.read_jsonl(path):
        if row.get("config_key") in keep:
            rows[(str(row["arm"]), str(row["item_id"]))] = row
    return rows


def compare(label: str, a: dict[str, bool], b: dict[str, bool], groups: dict[str, str] | None = None) -> dict:
    """B contra A: por ítem (o un voto por grupo), con el test de signos y el bootstrap de la diferencia."""
    p = stats.paired(a, b)
    row = {"comparación (B vs A)": label, "n": p["n"], "A": f"{sum(a[i] for i in a if i in b)}/{p['n']}",
           "B": f"{sum(b[i] for i in b if i in a)}/{p['n']}", "gana B": p["wins"], "gana A": p["losses"],
           "p (signos)": round(p["p_sign"], 3)}
    if groups is not None:
        g = stats.paired_groups(a, b, groups)
        row |= {"guiones B-A": f"{g['wins']}-{g['losses']}", "p (por guion)": round(g["p_sign"], 3)}
        p_read = g["p_sign"]
    else:
        p_read = p["p_sign"]
    diff, lo, hi = stats.bootstrap_diff_ci(a, b, groups)
    row["diferencia B − A"] = f"{diff:+.0%} ({lo:+.0%} a {hi:+.0%})"
    if not p["n"] or p["wins"] == p["losses"]:
        row["lectura"] = "sin diferencia" if p["n"] else "—"
    else:
        row["lectura"] = "supera el ruido" if p_read < 0.05 else "dirección, dentro del ruido"
    return row


# %% [markdown]
r"""
## 1 · El índice, sin LLM: las métricas de la Etapa 3, la regla en dev y el filtro de fecha

`index_loader.load` carga el índice y verifica que su clave sea la de `best_config.json`. Toda la recuperación de esta notebook se hace acá, en lote, y después el embedder y el reranker pasan a CPU.

**1a · Las métricas por ítem de la Etapa 3.** Se recalculan desde las filas crudas guardadas (`results/03_retrieval.jsonl`) con `evaluation.item_metrics`, la misma función de la Etapa 3, y se escriben a `results/03_retrieval_metrics.jsonl`. Con eso, `tools/recompute_tables.py` recalcula la primaria desde el zip sin el corpus. La celda verifica que los aciertos por fila sean los de `best_config.json`.
"""

# %%
llm.unload(MODEL)
t0 = time.time()
LOADED = index_loader.load(ART, DOCS, ROOT, ev.key())
UNITS = LOADED.units
CFG, BUDGET = LOADED.config, LOADED.config.budget_tokens
print(f"índice {LOADED.key} · mejor fila {LOADED.best['row']} -> {asdict(CFG)} · {time.time() - t0:.0f} s")
run = StageRun.start("08_followups", ART, profile=ART.name,
                     config={"model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "eval": ev.key(),
                             "index": LOADED.key, "retrieval": LOADED.best["row"],
                             "position_budget": POSITION_BUDGET})

GOLD = {item.item_id: grounded.Gold.of(item, SECTIONS, UNITS) for item in ev.items}
ANSWERABLE = [item for item in ev.items if item.answerable]

RAW3 = saved_rows("03_retrieval", LOADED.key, every_spec=True)
SPEC3 = {path.stem.split("config-", 1)[1]: store.read_json(path)["spec"]
         for path in RESULTS.glob("03_retrieval.config-*.json")}
ROW_CONFIG = {arm: SPEC3[row["config_key"]]["config"] for (arm, _), row in RAW3.items()}
ROWS3 = list(LOADED.best["primary_on_test"])  # el orden de la Etapa 3: bm25, dense, +prefix, ...
metrics_rows = []
for (arm, item_id), row in sorted(RAW3.items()):
    item = ITEMS[item_id]
    if not item.answerable:
        continue
    metrics = evaluation.item_metrics(row, item, GOLD[item_id].accepted, GOLD[item_id].texts, UNITS)
    metrics_rows.append({"arm": arm, "item_id": item_id, "split": item.split, "category": item.category,
                         "config_key": row["config_key"], "metrics": metrics})
store.write_jsonl(RESULTS / "03_retrieval_metrics.jsonl", metrics_rows)
PRIMARY3 = {arm: {r["item_id"]: bool(r["metrics"]["gold_key_delivered"]) for r in metrics_rows if r["arm"] == arm}
            for arm in ROWS3}
TEST_ANS = sorted(i.item_id for i in ANSWERABLE if i.split == "test")
DEV_ANS = sorted(i.item_id for i in ANSWERABLE if i.split == "dev")
reproduced = {arm: sum(PRIMARY3[arm][i] for i in TEST_ANS) for arm in ROWS3}
print(f"{len(metrics_rows)} filas de métricas -> {RESULTS / '03_retrieval_metrics.jsonl'}")
print("primaria en test, recalculada:", reproduced)
followups.require(reproduced == LOADED.best["primary_on_test"],
                  f"las métricas recalculadas no son las de best_config.json ({LOADED.best['primary_on_test']}): "
                  "el índice o las filas de la Etapa 3 no son las que se midieron")
run.note("stage3_metrics", {"rows": len(metrics_rows), "primary_on_test": reproduced})

# %% [markdown]
r"""
**1b · La regla de la mejor fila, en dev y con remuestreo.** La Etapa 3 eligió con su regla pre-registrada sobre las 26 preguntas con respuesta de `test` (lo dice su sección 10). Acá se mide cuánto depende la elección de eso:
- **la misma regla en `dev`** (8 preguntas con respuesta). Con tan pocas, casi ninguna fila se distingue de otra, y la regla tiende a caer en la de menos piezas: se reporta, pero no cambia la configuración (decidido antes);
- **la estabilidad en `test`:** 2.000 remuestreos bootstrap de las 26 preguntas, con semilla, y en cuántos la regla vuelve a elegir `+prefix`. `followups.rule_choice` repite la regla de la Etapa 3, y la celda verifica que sobre test dé lo mismo que `best_config.json`.
"""

# %%
N_PIECES = {arm: sum(bool(ROW_CONFIG[arm][piece]) for piece in followups.PIECES) for arm in ROWS3}
TOPICAL = [i for i in TEST_ANS + DEV_ANS if ITEMS[i].category == "topical"]
choice_test = followups.rule_choice(PRIMARY3, N_PIECES, TOPICAL, TEST_ANS)
followups.require(choice_test == LOADED.best["row"],
                  f"la regla repetida eligió {choice_test}, la Etapa 3 {LOADED.best['row']}")
choice_dev = followups.rule_choice(PRIMARY3, N_PIECES, TOPICAL, DEV_ANS)
display(pd.DataFrame({"piezas": N_PIECES,
                      f"test ({len(TEST_ANS)})": {a: sum(PRIMARY3[a][i] for i in TEST_ANS) for a in ROWS3},
                      f"dev ({len(DEV_ANS)})": {a: sum(PRIMARY3[a][i] for i in DEV_ANS) for a in ROWS3}}))
stability = followups.rule_stability(PRIMARY3, N_PIECES, TOPICAL, TEST_ANS, n_boot=2000, seed=0)
print(f"la regla en test elige {choice_test}; en dev elegiría {choice_dev}")
print("en 2.000 remuestreos de test, elige:", {arm: f"{n / 2000:.0%}" for arm, n in stability.most_common()})
run.note("rule", {"test": choice_test, "dev": choice_dev, "bootstrap": dict(stability)})

# %% [markdown]
r"""
**1c · El filtro de fecha sobre la mejor fila.** `+prefix+date` es `+prefix` con `date_filter=True`: una variable. El filtro lee la fecha de la pregunta (o, si no nombra ninguna, lo vigente el día de la foto) y deja buscar solo entre las versiones que rigen en ese período. La fila `+date` de la Etapa 3 lo medía sobre `+parent` (con RRF, reranker y padres): esta es la comparación limpia contra la configuración que usan las Etapas 4-6.

Además de la primaria, para cada pregunta con respuesta se mira **qué versiones del artículo gold le llegan al generador** (`followups.gold_versions`) con la entrega de la Etapa 4. Es el mecanismo de md02 y md05: con dos versiones a la vista, el modelo aplicó la equivocada.
"""

# %%
CFG_DATE = replace(CFG, date_filter=True)
change = retrieval.diff_rows(CFG, CFG_DATE)
assert change == "one variable: date_filter", change
print(f"+prefix -> +prefix+date: {change}")
CODE_SHA = store.sha256_json({name: store.sha256_text((ROOT / "src" / name).read_text(encoding="utf-8"))
                              for name in ("retrieval.py", "lexical.py", "evalset.py", "chunking.py")})[:16]
date_log = store.ResultsLog(RESULTS / "08_retrieval_date.jsonl",
                            run_spec={"config": asdict(CFG_DATE), "eval": ev.key(), "index": LOADED.key,
                                      "code": CODE_SHA})
count = grounded.token_counter(MODEL)


def deliver_with(units: list, budget: int) -> list:
    return grounded.deliver(units, budget, count)


RANKED: dict[str, dict[str, list]] = {"+prefix": {}, "+prefix+date": {}}
for item in ev.items:
    units, result = index_loader.ranked_units(LOADED, item.question)
    RANKED["+prefix"][item.item_id] = units
    followups.require(followups.same_delivery(result.delivered, RAW3[("+prefix", item.item_id)]["delivered"]),
                      f"{item.item_id}: la recuperación +prefix de hoy no entrega lo que guardó la Etapa 3")
    units, result = index_loader.ranked_units(LOADED, item.question, CFG_DATE)
    RANKED["+prefix+date"][item.item_id] = units
    if not date_log.done("+prefix+date", item.item_id):
        date_log.write({"arm": "+prefix+date", "item_id": item.item_id, **result.to_row()})
print(f"recuperación de {len(ev.items)} preguntas con las dos configuraciones: +prefix reproduce la Etapa 3")

date_metrics = {r["item_id"]: evaluation.item_metrics(r, ITEMS[r["item_id"]], GOLD[r["item_id"]].accepted,
                                                      GOLD[r["item_id"]].texts, UNITS)
                for r in date_log.rows("+prefix+date") if ITEMS[r["item_id"]].answerable}
PRIMARY_DATE = {i: bool(m["gold_key_delivered"]) for i, m in date_metrics.items()}
display(pd.DataFrame([compare("+prefix+date vs +prefix (primaria de la Etapa 3)",
                              {i: PRIMARY3["+prefix"][i] for i in TEST_ANS}, {i: PRIMARY_DATE[i] for i in TEST_ANS})]
                     ).set_index("comparación (B vs A)"))
display(pd.DataFrame({cat: {"+prefix": stats.count({i: PRIMARY3["+prefix"][i] for i in TEST_ANS
                                                    if ITEMS[i].category == cat}),
                            "+prefix+date": stats.count({i: PRIMARY_DATE[i] for i in TEST_ANS
                                                         if ITEMS[i].category == cat})}
                      for cat in ("topical", "identifier", "metadata")}))

versions = []
for item in ANSWERABLE:
    if item.item_id not in REPORT_IDS:
        continue
    row = {"id": item.item_id, "categoría": item.category, "fecha": item.as_of}
    for name in RANKED:
        delivered = deliver_with(RANKED[name][item.item_id], BUDGET)
        found = followups.gold_versions(delivered, item.gold_sections, GOLD[item.item_id].accepted)
        row[f"{name}: llega el gold"] = found["gold"]
        row[f"{name}: otras versiones"] = len(found["others"])
    versions.append(row)
VERSIONS = pd.DataFrame(versions).set_index("id")
display(VERSIONS[VERSIONS["categoría"] == "metadata"])
summary = {name: {"gold y ninguna otra versión": int((VERSIONS[f"{name}: llega el gold"]
                                                     & (VERSIONS[f"{name}: otras versiones"] == 0)).sum()),
                  "gold junto con otra versión": int((VERSIONS[f"{name}: llega el gold"]
                                                     & (VERSIONS[f"{name}: otras versiones"] > 0)).sum()),
                  "otra versión sin el gold": int((~VERSIONS[f"{name}: llega el gold"]
                                                  & (VERSIONS[f"{name}: otras versiones"] > 0)).sum())}
           for name in RANKED}
display(pd.DataFrame(summary))
run.note("date_filter", {"primary": {"+prefix": sum(PRIMARY3["+prefix"][i] for i in TEST_ANS),
                                     "+prefix+date": sum(PRIMARY_DATE[i] for i in TEST_ANS)}, "versions": summary})

# %% [markdown]
r"""
**1d · Lo que necesitan las secciones 3 y 4, también en lote.**
- **Posición (sección 3):** para cada pregunta con respuesta de test, el padre gold con la clave (el primero del oráculo de la Etapa 4) y los distractores: los hijos que recuperó `+prefix`, sin los del gold, hasta 6.000 tokens menos el gold. Son los mismos para las tres posiciones.
- **El resumen v2 (sección 4):** la recuperación de la Parte B de la Etapa 5 (`last_turn`, acotada a la ventana del modelo que la lee). La celda verifica que dé los mismos rankings que guardó la Etapa 5.
"""


# %%
def oracle_units(item: evalset.EvalItem) -> list:
    """El oráculo de la Etapa 4: los padres gold, en el orden del gold, primero los que tienen la clave."""
    parents = []
    for sid in item.gold_sections:
        mine = [u for u in LOADED.index.parents.values() if u.section_id == sid]
        parents += sorted(mine, key=lambda u: not evalset.contains_key(grounded.passage_block(u), item.answer_key))
    return parents


POSITION_ITEMS = [item for item in REPORT if item.answerable]
POSITION_SETS = {}
for item in POSITION_ITEMS:
    gold = grounded.as_passages(oracle_units(item)[:1], count)[0]
    others = followups.position_set(gold, RANKED["+prefix"][item.item_id], GOLD[item.item_id].accepted,
                                    GOLD[item.item_id].texts, POSITION_BUDGET, deliver_with)
    POSITION_SETS[item.item_id] = (gold, others)
sizes = [len(others) + 1 for _, others in POSITION_SETS.values()]
tokens = [gold.n_gen + grounded.used_tokens(others) for gold, others in POSITION_SETS.values()]
longest = max(ev.items, key=lambda item: count(item.question))
overhead = llm.count_prompt_tokens(grounded.build_messages(SYSTEM, longest.question, []), MODEL)
print(f"posición: {len(POSITION_SETS)} preguntas · {min(sizes)}-{max(sizes)} pasajes · {min(tokens)}-{max(tokens)} "
      f"tokens · prompt sin pasajes {overhead} · límite {grounded.PROMPT_LIMIT}")
assert overhead + POSITION_BUDGET <= grounded.PROMPT_LIMIT, "el experimento de posición no entra en num_ctx − num_predict"

TURNS, HISTORY, SPLIT = {}, {}, {}
for dialogue in ev.dialogues:
    turns = conversation.replay(dialogue)
    for index, rt in enumerate(turns):
        TURNS[rt.turn_id] = rt
        HISTORY[rt.turn_id] = conversation.scripted_history(turns[:index])
        SPLIT[rt.turn_id] = dialogue.split
GROUPS = {turn_id: rt.dialogue_id for turn_id, rt in TURNS.items()}
TURN_GOLD = {turn_id: grounded.Gold.of(rt.turn, SECTIONS, UNITS) for turn_id, rt in TURNS.items()}
reranker, embedder = LOADED.index.reranker, LOADED.index.embedder
LIMIT = conversation.query_limit(CFG.rerank, reranker.tokenizer if reranker else None, embedder.tokenizer,
                                 (reranker.max_seq_length if reranker else None) or 512, embedder.max_seq_length)
OFFSETS = conversation.hf_offsets(reranker.tokenizer if CFG.rerank else embedder.tokenizer)
SAVED_QUERIES = saved_rows("05_queries", LOADED.key)
RANKED5 = {}
for turn_id, rt in TURNS.items():
    query, _ = conversation.bound_query("last_turn", rt.turn.user, HISTORY[turn_id], LIMIT, OFFSETS)
    units, result = index_loader.ranked_units(LOADED, query)
    RANKED5[turn_id] = units
    saved = SAVED_QUERIES.get(("last_turn", turn_id))
    followups.require(saved is not None and saved["sections"] == result.to_row()["sections"],
                      f"{turn_id}: la recuperación last_turn de hoy no es la que guardó la Etapa 5")
print(f"recuperación de {len(RANKED5)} turnos: reproduce la Parte A de la Etapa 5")

LOADED.release_gpu()
environment.gpu_report("después de liberar el índice")

# %% [markdown]
r"""
## 2 · Etapa 4 con dos brazos nuevos

| Brazo | Contra qué brazo guardado | La única diferencia |
|---|---|---|
| `retrieval_date` | `retrieval` | el filtro de fecha (`+prefix+date` en lugar de `+prefix`), con el mismo presupuesto de 3.000 tokens |
| `oracle_padded` | `oracle` | después de los padres gold, distractores (lo que recuperó `+prefix`, sin el gold) hasta 3.000 tokens |

`oracle_padded` contra `retrieval` también es una comparación de una variable con los tokens igualados: el gold garantizado y primero, contra lo que trae la recuperación. Las preguntas fuera del corpus reciben, como en la Etapa 4, los pasajes de la recuperación.

Antes de generar se verifica que las entregas de `retrieval` y `oracle` recalculadas hoy sean las que guardó la Etapa 4: mismas unidades y misma cuenta de tokens.
"""

# %%
SAVED4 = saved_rows("04_grounded_answers", LOADED.key)
DELIVERED = {"retrieval": {}, "oracle": {}, "retrieval_date": {}, "oracle_padded": {}}
for item in ev.items:
    retrieved = deliver_with(RANKED["+prefix"][item.item_id], BUDGET)
    oracle = deliver_with(oracle_units(item), BUDGET) if item.answerable else retrieved
    DELIVERED["retrieval"][item.item_id], DELIVERED["oracle"][item.item_id] = retrieved, oracle
    for arm in ("retrieval", "oracle"):
        followups.require(followups.same_delivery(DELIVERED[arm][item.item_id], SAVED4[(arm, item.item_id)]["delivered"]),
                          f"{arm}/{item.item_id}: la entrega de hoy no es la que guardó la Etapa 4")
    DELIVERED["retrieval_date"][item.item_id] = deliver_with(RANKED["+prefix+date"][item.item_id], BUDGET)
    DELIVERED["oracle_padded"][item.item_id] = (
        followups.padded_oracle(oracle_units(item), RANKED["+prefix"][item.item_id], GOLD[item.item_id].accepted,
                                GOLD[item.item_id].texts, BUDGET, deliver_with) if item.answerable else retrieved)
print("las entregas de retrieval y oracle reproducen la Etapa 4")
display(pd.DataFrame({arm: {"pasajes (media)": round(sum(len(DELIVERED[arm][i.item_id]) for i in REPORT) / len(REPORT), 1),
                            "tokens (media)": round(sum(grounded.used_tokens(DELIVERED[arm][i.item_id])
                                                        for i in REPORT) / len(REPORT))}
                      for arm in DELIVERED}).T)

SERVER = llm.setup(MODEL, num_ctx=NUM_CTX)
llm.use_cache(ART / "llm_cache" / "qwen3_8b.jsonl")
run.note("server", {k: SERVER.get(k) for k in ("digest", "server", "key_order", "thinking", "calibration")})

NEW_ARMS = ["retrieval_date", "oracle_padded"]
RUN_SPEC = {"stage": "08-04", "eval": ev.key(), "corpus_sha": CORPUS_SHA, "model": MODEL, "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT, "passage_budget": BUDGET, "arms": NEW_ARMS, "index": LOADED.key,
            "retrieval": {"row": LOADED.best["row"], "config": asdict(CFG)},
            "retrieval_date": asdict(CFG_DATE), "server": SERVER["server"], "digest": SERVER["digest"],
            "prompt": grounded.prompt_spec(SYSTEM)}
log4 = store.ResultsLog(RESULTS / "08_grounded_answers.jsonl", run_spec=RUN_SPEC)
t0 = time.time()
for arm in NEW_ARMS:
    llm.unload(MODEL)  # arranque en frío, como en la Etapa 4
    for item in ev.items:
        if log4.done(arm, item.item_id):
            continue
        delivered = DELIVERED[arm][item.item_id]
        answer, call = grounded.generate(grounded.build_messages(SYSTEM, item.question, delivered))
        scored = grounded.score(answer, call, item, delivered, GOLD[item.item_id])
        log4.write({"arm": arm, "item_id": item.item_id, "split": item.split, "category": item.category,
                    "error": scored.get("error"), "delivered": [[u.unit_id, u.n_gen] for u in delivered],
                    "answer": answer.model_dump() if answer else None, "score": scored, "call": call})
    print(f"{arm}: listo ({time.time() - t0:.0f} s acumulados)")


# %%
def rescore(row: dict, arm: str) -> dict:
    item, delivered = ITEMS[row["item_id"]], DELIVERED[arm][row["item_id"]]
    answer = grounded.GroundedAnswer.model_validate(row["answer"]) if row["answer"] else None
    return {**row, "score": grounded.score(answer, row["call"], item, delivered, GOLD[row["item_id"]]),
            "delivery": grounded.delivery(item, delivered or [], GOLD[row["item_id"]])}


ROWS4 = {arm: [rescore(r, arm) for (a, _), r in SAVED4.items() if a == arm] for arm in ("retrieval", "oracle")}
ROWS4 |= {arm: [rescore(r, arm) for r in log4.rows(arm)] for arm in NEW_ARMS}


def correct_map(arm: str) -> dict[str, bool]:
    return {r["item_id"]: bool(r["score"]["correct"]) for r in ROWS4[arm]
            if r["item_id"] in REPORT_IDS and r["score"]["outcome"] != "error"}


table4 = []
for arm in ("retrieval", "retrieval_date", "oracle", "oracle_padded"):
    rows = [r for r in ROWS4[arm] if r["item_id"] in REPORT_IDS]
    scored = [r for r in rows if r["score"]["outcome"] != "error"]
    answerable = [r for r in scored if ITEMS[r["item_id"]].answerable]
    table4.append({"brazo": arm + ("" if arm in NEW_ARMS else " (guardado)"),
                   "acierto": stats.count(correct_map(arm)),
                   "con respuesta": f"{sum(r['score']['correct'] for r in answerable)}/{len(answerable)}",
                   "metadata": stats.count({k: v for k, v in correct_map(arm).items()
                                            if ITEMS[k].category == "metadata"}),
                   "clave desde gold": f"{sum(bool(r['delivery']['gold_key_delivered']) for r in answerable)}/{len(answerable)}",
                   "errores": len(rows) - len(scored),
                   "citó sin recibir": sum(bool(r["score"]["cited_not_delivered"]) for r in scored),
                   "tokens (media)": round(sum(grounded.used_tokens(DELIVERED[arm][r["item_id"]]) for r in rows)
                                           / max(len(rows), 1))})
display(pd.DataFrame(table4).set_index("brazo"))
NOISE4 = pd.DataFrame([compare("retrieval_date vs retrieval", correct_map("retrieval"), correct_map("retrieval_date")),
                       compare("oracle_padded vs oracle", correct_map("oracle"), correct_map("oracle_padded")),
                       compare("oracle_padded vs retrieval", correct_map("retrieval"), correct_map("oracle_padded"))]
                      ).set_index("comparación (B vs A)")
display(NOISE4)
for item_id in ("md02", "md05"):
    if item_id in ITEMS:
        for arm in ("retrieval", "retrieval_date"):
            row = next(r for r in ROWS4[arm] if r["item_id"] == item_id)
            found = followups.gold_versions(DELIVERED[arm][item_id], ITEMS[item_id].gold_sections,
                                            GOLD[item_id].accepted)
            print(f"{item_id} · {arm}: correcta={row['score']['correct']} · versiones del artículo entregadas: "
                  f"{found['versions']}")
            print(wrap(f"   R: {row['answer']['answer'] if row['answer'] else row['error']}"))
run.note("stage4", {"table": table4, "noise": NOISE4.reset_index().to_dict("records")})

# %% [markdown]
r"""
## 3 · La posición del pasaje

Cada pregunta con respuesta de test recibe **el mismo conjunto de pasajes** (el padre gold y los distractores de la sección 1d, ~6.000 tokens y 15-25 pasajes) con el gold **primero, en el medio o último**. Lo único que cambia entre los tres brazos es el lugar del gold: tokens y pasajes son idénticos.

**Qué podría mostrar esta prueba, dicho antes:** 26 preguntas pareadas. Un efecto de posición se vería como preguntas que aciertan con el gold primero (o último) y fallan con el gold en el medio. Para que el test de signos dé p < 0,05 hacen falta al menos 6 discordantes, todas para el mismo lado. Con ~6.000 tokens en una ventana de 8.192, un nulo **no refuta** *Lost in the Middle*, que mide el efecto con 20-30 documentos y ventanas más largas; un efecto, en cambio, sería una advertencia directa para la entrega de 3.000 tokens de las Etapas 4-6, que pone el gold donde lo deja el ranking.
"""

# %%
POSITION_SPEC = {"stage": "08-position", "eval": ev.key(), "corpus_sha": CORPUS_SHA, "model": MODEL,
                 "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "budget": POSITION_BUDGET, "index": LOADED.key,
                 "positions": list(followups.POSITIONS), "server": SERVER["server"], "digest": SERVER["digest"],
                 "prompt": grounded.prompt_spec(SYSTEM)}
position_log = store.ResultsLog(RESULTS / "08_position.jsonl", run_spec=POSITION_SPEC)
for position in followups.POSITIONS:
    arm = f"pos_{position}"
    llm.unload(MODEL)
    for item in POSITION_ITEMS:
        if position_log.done(arm, item.item_id):
            continue
        gold, others = POSITION_SETS[item.item_id]
        delivered = followups.positioned(gold, others, position)
        answer, call = grounded.generate(grounded.build_messages(SYSTEM, item.question, delivered))
        scored = grounded.score(answer, call, item, delivered, GOLD[item.item_id])
        position_log.write({"arm": arm, "item_id": item.item_id, "split": item.split, "category": item.category,
                            "error": scored.get("error"), "gold_index": [u.unit_id for u in delivered].index(gold.unit_id),
                            "delivered": [[u.unit_id, u.n_gen] for u in delivered],
                            "answer": answer.model_dump() if answer else None, "score": scored, "call": call})
POSITION = {position: {r["item_id"]: bool(r["score"]["correct"]) for r in position_log.rows(f"pos_{position}")
                       if r["score"]["outcome"] != "error"} for position in followups.POSITIONS}
display(pd.DataFrame({p: {"acierto": stats.count(POSITION[p]),
                          "errores": sum(r["score"]["outcome"] == "error" for r in position_log.rows(f"pos_{p}")),
                          "prompt (servidor, media)": round(sum(r["call"].get("prompt_tokens") or 0
                                                               for r in position_log.rows(f"pos_{p}"))
                                                           / max(len(position_log.rows(f"pos_{p}")), 1))}
                      for p in followups.POSITIONS}).T)
NOISE_POS = pd.DataFrame([compare("medio vs primero", POSITION["first"], POSITION["middle"]),
                          compare("medio vs último", POSITION["last"], POSITION["middle"]),
                          compare("último vs primero", POSITION["first"], POSITION["last"])]
                         ).set_index("comparación (B vs A)")
display(NOISE_POS)
run.note("position", {"accuracy": {p: stats.count(v) for p, v in POSITION.items()},
                      "noise": NOISE_POS.reset_index().to_dict("records")})

# %% [markdown]
r"""
## 4 · El resumen v2

La auditoría de la Etapa 5 encontró dos mecanismos en el resumen: **se congela en el tope** (al llegar a 220 caracteres, el modelo vuelve a copiar el resumen anterior y la gramática lo corta antes de que entre lo nuevo; 30 de 58 actualizaciones) y **a veces reemplaza en lugar de fusionar**. La v2 (`conversation.summarize_v2`, decidida antes de correr, sin ajustar contra ningún resultado) cambia solo el resumidor:
- **la gramática deja de ser el límite:** 600 caracteres, que 35 palabras no alcanzan. Un resumen que pasa los 220 se cuenta (`over_target`), no se corta;
- **el prompt pide reescribir juntando lo anterior con lo nuevo, nunca copiar**, y dice qué sacar primero para que entre (lo respondido en turnos viejos) y qué conservar siempre (los datos del usuario).

Todo lo demás es la Etapa 5: el mismo tope de historial (150 tokens, adentro de los 1.500 de pasajes + historial), el mismo disparador (70% del tope), la misma consulta (`last_turn`), el mismo prompt y el historial guionado. Si el resumen v2 sale más largo, ocupa más del tope y los mensajes recientes se recortan: el costo queda adentro del mismo total.

**Primero corre sobre los 2 guiones de `dev`** (dl09 y dl10), como chequeo de que el congelamiento desaparece; no se ajusta nada con lo que salga. Después, los 8 de `test`, y la sonda del resumen (sección 7 de la Etapa 5) con la v2.

**Comparaciones** (por guion, sobre los turnos de test que puntuaron en todos los brazos):
- `summary_v2_h150` contra `summary_h150` (guardado): el resumidor, una variable;
- `summary_v2_h150` contra `window_h150` (guardado): la estrategia, con el tope fijo, como el experimento 2 de la Etapa 5.

Antes de generar se verifica que la ventana de tope 150 recalculada hoy entregue los mismos pasajes que guardó la Etapa 5.
"""

# %%
SYSTEM_B = conversation.conversation_system(SNAPSHOT)
SAVED5 = saved_rows("05_history", LOADED.key)
for dialogue in ev.dialogues:  # la ventana de tope 150, sin LLM, contra lo guardado
    if ("window_h150", f"{dialogue.dialogue_id}.t1") not in SAVED5:
        continue
    turns = conversation.replay(dialogue)
    state = conversation.HistoryState(conversation.HistoryArm("window_h150", "window", 150), count)
    for index, rt in enumerate(turns):
        if index:
            state.add(turns[index - 1].exchange)
        _, _, history_used = state.view()
        delivered = deliver_with(RANKED5[rt.turn_id], conversation.passage_budget(history_used))
        followups.require([u.unit_id for u in delivered] == SAVED5[("window_h150", rt.turn_id)]["delivered"],
                          f"{rt.turn_id}: la ventana de hoy no entrega lo que guardó la Etapa 5")
print("la ventana de tope 150 reproduce las entregas de la Etapa 5")


def first_turn_in_view(history: list[dict], rt) -> bool:
    first = TURNS[f"{rt.dialogue_id}.t1"].turn.user
    return any(m["role"] == "user" and m["content"] == first for m in history)


ARM_V2 = conversation.SUMMARY_V2_ARM
DIALOGUES_V2 = [d for d in ev.dialogues if d.split == "dev"] + [d for d in ev.dialogues if d.split != "dev"]
B_SPEC = {"stage": "08-05B", "eval": ev.key(), "corpus_sha": CORPUS_SHA, "index": LOADED.key, "config": asdict(CFG),
          "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "total_budget": conversation.TOTAL_BUDGET,
          "arm": vars(ARM_V2), "query": "last_turn", "query_limit": LIMIT, "prompt": grounded.prompt_spec(SYSTEM_B),
          "system": store.sha256_text(SYSTEM_B), "summary_system": store.sha256_text(conversation.SUMMARY_V2_SYSTEM),
          "summary_schema": store.sha256_json(conversation.SummaryV2.model_json_schema())[:16],
          "summary_trigger": conversation.SUMMARY_TRIGGER, "summary_num_predict": conversation.SUMMARY_NUM_PREDICT,
          "message_margin": conversation.MESSAGE_MARGIN, "dialogues": [d.dialogue_id for d in DIALOGUES_V2]}
history_log = store.ResultsLog(RESULTS / "08_history.jsonl", run_spec=B_SPEC)
UPDATES: list[dict] = []


def run_dialogue(dialogue) -> None:
    """Un guion con el resumen v2, como la Parte B de la Etapa 5: el historial guionado y solo el turno actual
    generado. En una re-corrida, las llamadas de resumen salen de la caché y el estado se reconstruye igual."""
    turns = conversation.replay(dialogue)
    state = conversation.HistoryState(ARM_V2, count, conversation.summarize_v2)
    for index, rt in enumerate(turns):
        summary_calls = state.add(turns[index - 1].exchange) if index else []
        if history_log.done(ARM_V2.name, rt.turn_id):
            continue
        summary, history, history_used = state.view()
        delivered = deliver_with(RANKED5[rt.turn_id], conversation.passage_budget(history_used))
        passages_used = grounded.used_tokens(delivered)
        messages = grounded.build_messages(conversation.with_summary(SYSTEM_B, summary), rt.turn.user, delivered,
                                           history=history)
        answer, call = grounded.generate(messages)
        result = grounded.score(answer, call, rt.turn, delivered, TURN_GOLD[rt.turn_id])
        history_log.write({
            "arm": ARM_V2.name, "item_id": rt.turn_id, "dialogue_id": rt.dialogue_id, "split": dialogue.split,
            "turn": rt.number, "depends_on": rt.turn.depends_on, "query_type": "last_turn",
            "delivered": [u.unit_id for u in delivered], "passage_tokens": passages_used,
            "history_tokens": history_used, "total_tokens": history_used + passages_used,
            "history_messages": len(history), "history_dropped": len(state.messages) - len(history),
            "first_turn_in_view": rt.number == 1 or first_turn_in_view(history, rt),
            "summary": summary or None, "summary_calls": summary_calls,
            "delivery": grounded.delivery(rt.turn, delivered, TURN_GOLD[rt.turn_id]),
            "answer": answer.model_dump() if answer else None,
            "score": result, "call": call, "error": call.get("error") or result.get("error")})
    UPDATES.extend(u | {"pass": "answers", "arm": ARM_V2.name, "dialogue_id": dialogue.dialogue_id,
                        "split": dialogue.split} for u in state.buffer.updates)


def freeze_line(updates: list[dict]) -> str:
    return (f"{len(updates)} actualizaciones · en el tope de {conversation.SUMMARY_V2_GRAMMAR_MAX}: "
            f"{sum(u['at_maxlength'] for u in updates)} · más largas que {conversation.SUMMARY_V2_TARGET}: "
            f"{sum(u['over_target'] for u in updates)} · copia del anterior: "
            f"{sum(u['copied_previous'] for u in updates)} · fallidas: {sum(bool(u['error']) for u in updates)}")


llm.unload(MODEL)
for split in ("dev", "test"):
    for dialogue in [d for d in DIALOGUES_V2 if (d.split == "dev") == (split == "dev")]:
        run_dialogue(dialogue)
    print(f"{split}: " + freeze_line([u for u in UPDATES if (u["split"] == "dev") == (split == "dev")]))

RECALL_SYSTEM = conversation.recall_system(SNAPSHOT)
PROBE_SPEC = B_SPEC | {"stage": "08-05B-probe", "recall_system": store.sha256_text(RECALL_SYSTEM),
                       "prompt": grounded.prompt_spec(RECALL_SYSTEM)}
probe_log = store.ResultsLog(RESULTS / "08_summary_probe.jsonl", run_spec=PROBE_SPEC)
llm.unload(MODEL)
for dialogue in DIALOGUES_V2:
    question, target = conversation.recall_probe(dialogue.turns[0])
    turns = conversation.replay(dialogue)
    state = conversation.HistoryState(ARM_V2, count, conversation.summarize_v2)
    for rt in turns:
        state.add(rt.exchange)  # salvo la última, las llamadas de resumen salen de la caché de arriba
    UPDATES += [u | {"pass": "probe", "arm": ARM_V2.name, "dialogue_id": dialogue.dialogue_id,
                     "split": dialogue.split} for u in state.buffer.updates]
    if target is None or probe_log.done(ARM_V2.name, dialogue.dialogue_id):
        continue
    summary, history, history_used = state.view()
    answer, call = grounded.generate(grounded.build_messages(
        conversation.with_summary(RECALL_SYSTEM, summary), question, None, history=history))
    result = grounded.score(answer, call, target, None, grounded.Gold.of(target, SECTIONS, UNITS))
    probe_log.write({"arm": ARM_V2.name, "item_id": dialogue.dialogue_id, "split": dialogue.split,
                     "history_tokens": history_used, "summary": summary or None,
                     "first_turn_in_view": first_turn_in_view(history, turns[-1]), "answer_key": target.answer_key,
                     "answer": answer.model_dump() if answer else None, "score": result, "call": call,
                     "error": call.get("error") or result.get("error")})
store.write_jsonl(RESULTS / "08_summaries.jsonl", UPDATES)


# %%
def b_correct(rows) -> dict[str, bool]:
    return {r["item_id"]: bool(r["score"]["correct"]) for r in rows
            if r["score"]["outcome"] != "error" and SPLIT[r["item_id"]] == "test"}


B = {"window_h150": b_correct(r for (a, _), r in SAVED5.items() if a == "window_h150"),
     "summary_h150": b_correct(r for (a, _), r in SAVED5.items() if a == "summary_h150"),
     ARM_V2.name: b_correct(history_log.rows(ARM_V2.name))}
COMMON = set.intersection(*(set(v) for v in B.values()))
B = {arm: {i: v for i, v in values.items() if i in COMMON} for arm, values in B.items()}
saved_updates = [u for u in store.read_jsonl(RESULTS / "05_summaries.jsonl")
                 if u.get("arm") == "summary_h150" and u.get("pass") == "answers"]
v2_updates = [u for u in UPDATES if u["pass"] == "answers" and u["split"] == "test"]
SAVED_PROBE = saved_rows("05_summary_probe", LOADED.key)
probe = {"window_h150": [r for (a, _), r in SAVED_PROBE.items() if a == "window_h150"],
         "summary_h150": [r for (a, _), r in SAVED_PROBE.items() if a == "summary_h150"],
         ARM_V2.name: [r for r in probe_log.rows(ARM_V2.name) if r.get("split") == "test"]}


def cost(rows) -> dict:
    rows = [r for r in rows if SPLIT[r["item_id"]] == "test"]
    calls = [[r["call"]] + r["summary_calls"] for r in rows]
    return {"llamadas/turno": round(sum(len(c) for c in calls) / max(len(rows), 1), 2),
            "tokens de salida/turno": round(sum(sum(x.get("output_tokens") or 0 for x in c) for c in calls)
                                           / max(len(rows), 1))}


table5 = []
for arm, rows in (("window_h150", [r for (a, _), r in SAVED5.items() if a == "window_h150"]),
                  ("summary_h150", [r for (a, _), r in SAVED5.items() if a == "summary_h150"]),
                  (ARM_V2.name, history_log.rows(ARM_V2.name))):
    updates = {"summary_h150": saved_updates, ARM_V2.name: v2_updates}.get(arm, [])
    table5.append({"brazo": arm + ("" if arm == ARM_V2.name else " (guardado)"),
                   f"acierto ({len(COMMON)} turnos comunes)": stats.count(B[arm]),
                   "sonda: recuerda el turno 1": stats.count({r["item_id"]: bool(r["score"]["correct"])
                                                              for r in probe[arm] if r["score"]["outcome"] != "error"}),
                   "actualizaciones": len(updates),
                   "en el tope de la gramática": sum(bool(u.get("at_maxlength")) for u in updates),
                   "copian el resumen anterior": sum(bool(u["after"]) and u["after"] == u["before"] for u in updates),
                   **cost(rows)})
display(pd.DataFrame(table5).set_index("brazo"))
NOISE5 = pd.DataFrame([compare(f"{ARM_V2.name} vs summary_h150", B["summary_h150"], B[ARM_V2.name], GROUPS),
                       compare(f"{ARM_V2.name} vs window_h150", B["window_h150"], B[ARM_V2.name], GROUPS)]
                      ).set_index("comparación (B vs A)")
display(NOISE5)
for update in v2_updates[:8]:
    print(wrap(f"{update['dialogue_id']} · {len(update['after'] or '')} caracteres: {update['after']}"))
run.note("summary_v2", {"table": table5, "noise": NOISE5.reset_index().to_dict("records"),
                        "updates": len(v2_updates)})

# %% [markdown]
r"""
## 5 · Las auditorías, por una persona

Las cuatro auditorías del proyecto las completó un modelo, no una persona: la planilla del juez (`judge/audit.csv`, 82 juicios), la de los registros de memoria (`memory/audit_records.csv`, 76), la de los resúmenes (25 actualizaciones) y la lectura de las 20 reescrituras. La consigna pide que esa vara la ponga una persona. Esta sección:
1. escribe **planillas ciegas**, con las mismas filas y sin lo que dijo el modelo (`tools/make_human_audits.py`; no pisa una planilla que ya tenga veredictos);
2. cuando una persona las completa, calcula su acuerdo con el auditor-modelo y vuelve a calibrar contra ella los tres instrumentos de la Etapa 7a (contención y los dos jueces), con el mismo reponderado por estrato.

Cómo completarlas: `AUDITORIA_HUMANA.md`, en la raíz de la entrega.
"""

# %%
sys.path.insert(0, str(ROOT / "tools"))
import make_human_audits  # noqa: E402

for line in make_human_audits.write_all(ROOT, ART):
    print(line)

HUMAN_DIR = ART / "judge_human"
human, pending = judge.read_audit(HUMAN_DIR)
model_sheet = {r["audit_id"]: r["human_verdict"].strip().lower() for r in judge._sheet_rows(ART / "judge" / "audit.csv")}
print(f"juez: {len(human)} juicios con veredicto de una persona · {pending} pendientes")
if human:
    granite = {r["uid"]: r for r in store.read_jsonl(RESULTS / "07_judge_granite3_3_8b.jsonl")}
    STRATA = {row["stratum"] for row in judge.read_jsonl(HUMAN_DIR / "audit_key.jsonl")}
    as_judge = {"juez qwen3:8b": human,
                "juez granite3.3:8b": [{**r, "judge_verdict": granite[r["uid"]]["judge_verdict"]} for r in human],
                "auditor-modelo": [{**r, "judge_verdict": model_sheet[r["audit_id"]]} for r in human]}
    calibration = []
    for name, rows in [("contención", human)] + list(as_judge.items()):
        source = "contention" if name == "contención" else "judge"
        result = judge.agreement(rows, "human", source)
        calibration.append({"instrumento contra la persona": name, "acuerdo (muestra)": f"{result['agree']}/{result['n']}",
                            "reponderado": judge.population_agreement(rows, "human", source, strata=STRATA),
                            "acepta lo que la persona rechaza": result["b_only"],
                            "rechaza lo que la persona acepta": result["a_only"]})
    display(pd.DataFrame(calibration).set_index("instrumento contra la persona"))
    run.note("human_judge_audit", {"done": len(human), "pending": pending, "table": calibration})
else:
    print("Hasta que una persona complete judge_human/audit.csv, la calibración de la Etapa 7a es contra el "
          "auditor-modelo, y así lo dice el informe.")

for label, path, key, model_path in [
        ("registros de memoria", ART / "memory" / "human_audit_records.csv", ("scenario", "memory_id"),
         ART / "memory" / "audit_records.csv")]:
    if not path.is_file():
        continue
    person = memory.read_audit(path)
    model = memory.read_audit(model_path)
    joined = lambda rows: [{**r, "key": "/".join(r[k] for k in key)} for r in rows]  # noqa: E731
    result = followups.sheet_agreement(joined(model), joined(person), "key")
    print(f"{label}: persona contra auditor-modelo, {result['agree']}/{result['n']} iguales · {result['pending']} "
          f"pendientes · matriz (modelo, persona): {result['confusion']}")
    run.note("human_memory_audit", result)
MODEL_LABELS = RESULTS / "05_rewrites_model_labels.json"
for label, path in [("resúmenes", RESULTS / "05_summaries_human_audit.csv"),
                    ("reescrituras", RESULTS / "05_rewrites_human_read.csv")]:
    if not path.is_file():
        continue
    rows = judge._sheet_rows(path)
    filled = [r for r in rows if (r.get("human_category") or "").strip()]
    print(f"{label}: {len(filled)}/{len(rows)} con categoría de una persona: "
          f"{dict(Counter(r['human_category'].strip().lower() for r in filled))}")
    if filled and label == "reescrituras" and MODEL_LABELS.is_file():
        model_labels = store.read_json(MODEL_LABELS)
        same = sum(r["human_category"].strip().lower() == model_labels.get(r["turn_id"]) for r in filled)
        print(f"   iguales a la categoría del auditor-modelo: {same}/{len(filled)}")

# %%
run.finish()
