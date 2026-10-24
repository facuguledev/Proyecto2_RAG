# %% [markdown]
r"""
# Etapa 7 · Una segunda lectura: el juez

La contención (`grounded.score`) puntúa buscando la clave por palabra completa. No ve una paráfrasis ("medio año" por "seis meses") y cuenta como acierto una respuesta que dice la clave y después la niega. Un juez LLM lee como una persona: la pregunta, la clave y la respuesta. Pero **un juez es un instrumento**: su puntaje promedio sin calibrar es un número sobre el juez, no sobre las respuestas. Por eso esta notebook lo compara con la contención y con una persona antes de leerlo.

| Entra | Sale |
|---|---|
| las respuestas guardadas de las Etapas 4, 5 y 6 (`results/*.jsonl`) y el eval set congelado | `results/07_judge_<modelo>.jsonl` (un veredicto por respuesta), `judge/audit.csv` (la planilla de la auditoría), `run_report.json` |

La lógica vive en `src/judge.py`.

**Qué recibe el juez.** La pregunta, la clave y la respuesta, en campos separados y etiquetados:
- **solo el campo `answer`**, nunca el `analysis` del generador (es su razonamiento, no lo que vio el usuario);
- **sin el contexto recuperado**: acá no se evalúa fidelidad al contexto, y con él el juez premiaría ser fiel al pasaje equivocado;
- en las preguntas sin respuesta, también `can_answer`;
- en la Etapa 5, la **reescritura oráculo** del turno en lugar de lo que escribió el usuario: "¿y si fuera un local?" no se puede juzgar sin el historial, y el juez no lo recibe.

**Una rúbrica por tipo de ítem** (`judge.RUBRICS`): con respuesta, sí/no (el veredicto cuenta, y abstenerse no es decir que no), sin respuesta (acierta solo si se abstiene) y actualización de conocimiento. Esta última sigue la regla de LongMemEval (Apéndice B.5): una respuesta con el valor nuevo cuenta aunque también mencione el viejo.

**Autopreferencia.** El juez por defecto es **`qwen3:8b`, el mismo modelo que generó las respuestas**. Un juez tiende a preferir lo que escribe su propio modelo, así que sus veredictos sobre estas respuestas no son neutrales. Para medirlo, un segundo juez de otra familia corre en una pasada aparte sobre las mismas respuestas guardadas (`SECOND_JUDGE`, abajo). La comparación que decide es contra la auditoría (sección 6).

**Decidido antes de correr:**
- veredicto estructurado (`analysis` primero, después `verdict`), temperatura 0, `num_ctx = 8.192` (el mismo del generador, así no se recarga el modelo) y `num_predict = 320`;
- los errores de llamada de las etapas anteriores **no se juzgan**: son su propia categoría donde ocurrieron. Los errores del propio juez se cuentan aparte;
- se juzga `test` (en el smoke, todo).
"""

# %%
import environment  # el del curso, primero: guarda de JAX, caché de HF, UTF-8

import os
import sys
import time
from collections import Counter

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

import doc_schema
import evalset
import judge
import stats

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
ART = environment.artifacts_dir(SMOKE_TEST)
EVAL_DIR = ROOT / "eval" / "smoke" if SMOKE_TEST else ROOT / "eval"
DOCS_PATH = ART / "documents.jsonl"
if SMOKE_TEST:  # el smoke nunca se congela: se valida contra su corpus
    ev = evalset.load_eval(EVAL_DIR, require_frozen=False)
    evalset.validate_eval(ev, doc_schema.load_documents(DOCS_PATH)).show().raise_if_failed()
else:
    ev = evalset.load_eval(EVAL_DIR, DOCS_PATH)
SPLITS = ("dev", "test") if SMOKE_TEST else ("test",)

# Un segundo juez de otra familia, para medir la autopreferencia: "tag=repo del tokenizer en HuggingFace".
# Sin el repo del tokenizer, llm.chat no puede contar el prompt y las guardas de contexto quedan apagadas.
# granite3.3:8b son 4,94 GB más de descarga; SECOND_JUDGE="" lo apaga.
SECOND_JUDGE = os.environ.get("SECOND_JUDGE", "granite3.3:8b=ibm-granite/granite-3.3-8b-instruct")
JUDGES = [judge.JUDGE_MODEL]
if SECOND_JUDGE:
    tag, repo = SECOND_JUDGE.split("=", 1)
    llm.HF_TOKENIZERS[tag] = repo
    JUDGES.append(tag)

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)  # como script, pandas corta las tablas anchas
environment.summary()
print(f"eval {ev.key()} | {ev.summary()} | jueces: {JUDGES} | splits: {SPLITS}")
run = StageRun.start("07_second_read", ART, profile=ART.name,
                     config={"judges": JUDGES, "num_ctx": judge.NUM_CTX, "num_predict": judge.NUM_PREDICT,
                             "rubric": judge.rubric_sha(), "splits": SPLITS, "eval": ev.key()})

# %% [markdown]
r"""
## 1 · Qué se juzga

De cada etapa se toman las filas de su corrida más reciente, y se verifica que se hayan puntuado contra este eval set (`judge.latest_rows`). Una fila queda afuera si:
- **es un error de llamada**;
- **no es una respuesta**, como las filas de recuperación de la Etapa 5A;
- **su id no es una pregunta del eval set**, como la sonda del resumen de la Etapa 5B.
"""

# %%
SOURCES = {"04": ART / "results" / "04_grounded_answers.jsonl",
           "05": ART / "results" / "05_history.jsonl",  # la Parte B: las respuestas de cada brazo de historial
           "06": ART / "results" / "06_memory.jsonl"}
missing = [path.name for path in SOURCES.values() if not path.is_file()]
# Una etapa sin resultados no es "no existe todavía": se juzgaría una parte y la tabla no lo diría.
assert SMOKE_TEST or not missing, f"faltan resultados de etapas anteriores: {missing}"
JUDGEABLE: list[judge.Judgeable] = []
coverage = []
for stage, path in SOURCES.items():
    if not path.is_file():
        coverage.append({"etapa": stage, "archivo": path.name, "estado": "falta"})
        continue
    rows = judge.latest_rows(path, ev.key())
    found, skipped = judge.judgeables(rows, judge.TARGETS[stage](ev), stage)
    kept = [j for j in found if j.split in SPLITS]
    JUDGEABLE += kept
    coverage.append({"etapa": stage, "archivo": path.name, "config": rows[0]["config_key"] if rows else None,
                     "filas": len(rows), "se juzgan": len(kept), "otro split": len(found) - len(kept),
                     **{f"afuera: {k}": v for k, v in skipped.items()}})
coverage = pd.DataFrame(coverage).set_index("etapa").fillna("")
display(coverage)
run.note("sources", coverage.reset_index().to_dict("records"))
display(pd.crosstab(pd.Series([f"{j.stage}/{j.arm}" for j in JUDGEABLE], name="etapa/brazo"),
                    pd.Series([j.kind for j in JUDGEABLE], name="tipo")))
assert JUDGEABLE, "no hay respuestas para juzgar: corré antes la Etapa 4"

# %% [markdown]
r"""
## 2 · Lo que lee el juez

El prompt de sistema, y el mensaje de una pregunta de sí/no tal como le llega.
"""

# %%
print(judge.SYSTEM)
example = next((j for j in JUDGEABLE if j.kind == "yes_no"), JUDGEABLE[0])
print("\n" + judge.judge_messages(example)[1]["content"])
calls = len(JUDGEABLE) * len(JUDGES)
print(f"\nllamadas: {len(JUDGEABLE)} respuestas × {len(JUDGES)} jueces = {calls}. A unos 3-5 s por veredicto "
      f"(prompt corto, salida de hasta {judge.NUM_PREDICT} tokens), del orden de {calls * 4 / 60:.0f} minutos.")

# %% [markdown]
r"""
## 3 · Los veredictos

Un juez por vez y con el modelo descargado al empezar: un solo modelo en la GPU, y el estado de la caché de prompts igual en cada corrida. Cada juez tiene su archivo de resultados; la caché de llamadas es una por modelo (la de `qwen3:8b` es la misma del generador). El id de cada veredicto incluye un hash de la respuesta juzgada y de su puntaje sin juez, así una etapa que se vuelve a correr se vuelve a juzgar, en lugar de reusar veredictos sobre respuestas viejas.

**Una abstención en una pregunta con respuesta no se le pregunta al juez:** es incorrecta por regla (`judge.ABSTAINED`). Si el juez la leyera, decidiría la abstención por el texto, y una respuesta que se abstiene pero menciona la clave le parecería correcta. Esas filas se cuentan aparte.
"""


# %%
def slug(model: str) -> str:
    return model.replace(":", "_").replace(".", "_").replace("/", "_")


def run_judge(model: str) -> list[dict]:
    for other in JUDGES:
        llm.unload(other)
    llm.setup(model, num_ctx=judge.NUM_CTX)
    llm.use_cache(ART / "llm_cache" / f"{slug(model)}.jsonl")
    llm.unload(model)  # arranque en frío
    log = store.ResultsLog(ART / "results" / f"07_judge_{slug(model)}.jsonl", run_spec={
        "stage": "07a", "eval": ev.key(), "judge": model, "num_ctx": judge.NUM_CTX,
        "num_predict": judge.NUM_PREDICT, "rubric": judge.rubric_sha()})
    t0 = time.time()
    for j in JUDGEABLE:
        arm, key = f"{j.stage}/{j.arm}", judge.judge_key(j)
        if log.done(arm, key):
            continue
        verdict, call = judge.judge_one(j, model=model)
        log.write({"arm": arm, "item_id": key, **judge.judged_row(j, verdict, call, model)})
    current = {(f"{j.stage}/{j.arm}", judge.judge_key(j)) for j in JUDGEABLE}
    rows = [r for r in log.rows() if (r["arm"], r["item_id"]) in current]
    print(f"{model}: {len(rows)} veredictos en {time.time() - t0:.0f} s · por regla (abstenciones): "
          f"{sum(r['by_rule'] for r in rows)} · errores del juez: {sum(r['judge_verdict'] is None for r in rows)} · "
          f"{llm.placement(model)}")
    return rows


VERDICTS = {model: run_judge(model) for model in JUDGES}
PRIMARY = VERDICTS[judge.JUDGE_MODEL]
run.note("verdicts", {model: len(rows) for model, rows in VERDICTS.items()})

# %% [markdown]
r"""
## 4 · El juez contra la contención

Por etapa y brazo, cuántas respuestas da por correctas cada instrumento y dónde discrepan:
- **solo el juez**: las da por correctas el juez y no la contención. Puede ser una paráfrasis que la contención no ve, o un juez indulgente;
- **solo contención**: al revés. Puede ser una clave mencionada y negada, o un juez severo.

El test de signos dice si un instrumento es sistemáticamente más indulgente que el otro. **No dice cuál tiene razón**: eso lo dice la auditoría (sección 6). Los desacuerdos uno por uno se leen recién después de la auditoría: la persona que la completa no tiene que ver antes qué dijo el juez.
"""


# %%
def agreement_table(rows: list[dict], key) -> pd.DataFrame:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["judge_verdict"] is not None:
            groups.setdefault(key(r), []).append(r)
    if not groups:  # un juez sin ningún veredicto válido: la tabla vacía, no un KeyError
        return pd.DataFrame(columns=["n", "contención", "juez", "acuerdo", "solo el juez", "solo contención",
                                     "p (signos)"])
    table = []
    for name, rs in sorted(groups.items()):
        a = judge.agreement(rs, "contention", "judge")
        table.append({"grupo": name, "n": a["n"],
                      "contención": f"{sum(r['contained'] for r in rs)}/{len(rs)}",
                      "juez": f"{sum(r['judge_verdict'] == 'correct' for r in rs)}/{len(rs)}",
                      "acuerdo": f"{a['agree']}/{a['n']}", "solo el juez": a["b_only"],
                      "solo contención": a["a_only"], "p (signos)": round(a["p_sign"], 4)})
    return pd.DataFrame(table).set_index("grupo")


display(agreement_table(PRIMARY, lambda r: f"{r['stage']}/{r['source_arm']}"))
display(agreement_table(PRIMARY, lambda r: r["kind"]))
total = judge.agreement([r for r in PRIMARY if r["judge_verdict"] is not None], "contention", "judge")
run.note("judge_vs_contention", total)

# %% [markdown]
r"""
**Lectura** (407 respuestas de test de las Etapas 4-6; 32 abstenciones resueltas por regla, sin llamar al juez).
- **`qwen3:8b` es más estricto que la contención, no más indulgente.** En las preguntas con respuesta da por correctas 58 de 216, contra 90 de la contención: 33 respuestas que la contención acepta y el juez rechaza, contra 1 al revés (p < 0,001). En las preferencias, 1 de 15 contra 7 de 15.
- **En las preguntas sin respuesta coinciden siempre** (52 de 52): ahí la regla es el booleano.
- **Por brazo, el juez deja el mismo orden que la contención:** libro cerrado 6/30, recuperación 13/30, oráculo 23/30 en la Etapa 4. En las Etapas 5 y 6, los brazos quedan tan juntos con el juez como con la contención.
- **Quién tiene razón** (el juez severo o la contención indulgente) no se decide acá: una respuesta que nombra la clave y la contradice, y una que no usa la palabra exacta, hay que leerlas. Eso es la sección 6: allí el juez resulta severo.
"""

# %% [markdown]
r"""
## 5 · La autopreferencia: un juez de otra familia

Si corrió el segundo juez, se compara con el primero sobre las mismas respuestas, y cada uno con la contención. Una señal de autopreferencia: que `qwen3:8b` sea más indulgente que el otro juez **justo donde discrepan**, y que la auditoría le dé la razón al otro. Que dos jueces coincidan no prueba que tengan razón: pueden equivocarse igual.
"""

# %%
if len(JUDGES) > 1:
    second = VERDICTS[JUDGES[1]]
    first_map = {r["uid"]: r["judge_verdict"] == "correct" for r in PRIMARY if r["judge_verdict"] is not None}
    second_map = {r["uid"]: r["judge_verdict"] == "correct" for r in second if r["judge_verdict"] is not None}
    p = stats.paired(second_map, first_map)
    print(f"{JUDGES[0]} vs {JUDGES[1]}, sobre {p['n']} respuestas: acuerdo {p['ties']}/{p['n']} · "
          f"solo {JUDGES[0]} da correcto: {p['wins']} · solo {JUDGES[1]}: {p['losses']} · p = {p['p_sign']:.4f}")
    display(agreement_table(second, lambda r: f"{r['stage']}/{r['source_arm']}"))
    run.note("judge_vs_judge", p)
else:
    print("Sin segundo juez (SECOND_JUDGE vacío): la autopreferencia queda nombrada pero sin medir.")

# %% [markdown]
r"""
**Lectura.** **Los dos jueces están lejos entre sí.** `granite3.3:8b` coincide con `qwen3:8b` en 305 de 407, y da por correctas 97 respuestas que `qwen3:8b` rechaza, contra 5 al revés (p < 0,001). Contra la contención, `granite3.3:8b` es indulgente: a libro cerrado acepta 14 de 30, contra 5 de la contención (9-0, p = 0,004).

La autopreferencia que se esperaba (un juez de la misma familia, indulgente con su generador) **no aparece**: el juez de la misma familia es el más estricto de los tres instrumentos. Pero dos jueces que no coinciden no dicen cuál tiene razón: es la advertencia de la consigna, un juez sin calibrar mide al juez. La calibración contra la auditoría está en la sección 6.
"""

# %% [markdown]
r"""
## 6 · La auditoría

**Por qué una auditoría aparte.** Comparar dos jueces entre sí mide si están de acuerdo, no si alguno tiene razón (§2.3).

La muestra tiene al menos 30 juicios o el 20%, lo que sea mayor. Se toma con semilla y estratificada: los desacuerdos juez-contención están sobrerrepresentados (hasta la mitad de la muestra), porque ahí es donde un instrumento se equivoca.

**La planilla es ciega.** `artifacts/<perfil>/judge/audit.csv` muestra la pregunta, el tipo, la clave y la respuesta. No muestra lo que dijeron el juez ni la contención, ni la etapa o el brazo (un "oracle" o un "closed_book" sugiere qué esperar). Todo eso queda aparte, en `audit_key.jsonl`, y se junta al leer.
"""

# %%
AUDIT_DIR = ART / "judge"
sample = judge.audit_sample(PRIMARY, seed=0)
written = judge.write_audit(sample, AUDIT_DIR)
print(("planilla nueva" if written else "la planilla ya tiene veredictos: se conserva la que hay") +
      f": {AUDIT_DIR / 'audit.csv'}")
audited, pending = judge.read_audit(AUDIT_DIR)
audited, changed = judge.refresh_audited(audited, PRIMARY)
STRATA = {row["stratum"] for row in judge.read_jsonl(AUDIT_DIR / "audit_key.jsonl")}
AUDITOR = ", ".join(sorted({row.get("auditor") or "sin indicar" for row in audited})) or "—"
print(f"auditor: {AUDITOR}")
print(f"{len(audited)} auditados · {pending} pendientes · estratos {sorted(STRATA)}"
      + (f" · {changed} veredictos del juez cambiaron desde que se armó la planilla" if changed else ""))
run.note("audit", {"done": len(audited), "pending": pending, "judge_changed": changed, "auditor": AUDITOR})
if pending:
    print(f"Faltan {pending} veredictos. Hasta completarla, la única comparación es juez vs contención "
          "(sección 4), que no dice quién tiene razón, y los desacuerdos no se muestran uno por uno.")

# %%
if audited:
    NAME = {"contention": "contención", "judge": f"juez {judge.JUDGE_MODEL}", "human": "auditor"}
    table = []
    for a, b in [("contention", "judge"), ("human", "judge"), ("human", "contention")]:
        result = judge.agreement(audited, a, b)
        table.append({"comparación (B vs A)": f"{NAME[b]} vs {NAME[a]}", "n": result["n"],
                      "acuerdo (muestra)": f"{result['agree']}/{result['n']}",
                      "ambos correcto": result["both_correct"], "ambos incorrecto": result["both_incorrect"],
                      "solo A da correcto": result["a_only"], "solo B da correcto": result["b_only"],
                      "p (signos)": round(result["p_sign"], 4),
                      "acuerdo reponderado": judge.population_agreement(audited, a, b, strata=STRATA)})
    display(pd.DataFrame(table).set_index("comparación (B vs A)"))
    print("El acuerdo de la muestra subestima al juez: la muestra sobrerrepresenta los desacuerdos. El reponderado "
          "pesa cada estrato por su tamaño en la población.")
    if len(JUDGES) > 1:
        second_by_uid = {r["uid"]: r for r in VERDICTS[JUDGES[1]]}
        joined = [{**r, "judge_verdict": second_by_uid[r["uid"]]["judge_verdict"]} for r in audited
                  if r["uid"] in second_by_uid]
        result = judge.agreement(joined, "human", "judge")
        print(f"{JUDGES[1]} vs el auditor (sobre la misma muestra): acuerdo {result['agree']}/{result['n']}, "
              f"solo el auditor {result['a_only']}, solo el juez {result['b_only']} · "
              f"reponderado {judge.population_agreement(joined, 'human', 'judge', strata=STRATA)}")

# %% [markdown]
r"""
**Lectura de la calibración** (82 juicios, auditados sin ver los veredictos).
- **La contención es el instrumento más cercano al auditor:** coincide en 73 de 82 (0,90 reponderado). Nunca acepta una respuesta que el auditor rechaza (0). Sus 9 desacuerdos son respuestas correctas con otras palabras ("al día en el pago" por "al corriente"). Es conservadora: sus números de las Etapas 4-6 son una cota inferior.
- **`qwen3:8b` como juez es demasiado severo:** coincide en 42 de 82 (0,75 reponderado), y rechaza 36 respuestas que el auditor acepta, contra 4 al revés. La autopreferencia que se esperaba no aparece. El juez de la misma familia es el que más castiga a su generador.
- **`granite3.3:8b` queda en el medio:** coincide en 69 de 82 (0,86 reponderado), con 8 respuestas que acepta y el auditor no, y 5 al revés.
- **Qué se puede leer, entonces.** Ningún juez mejora a la contención como vara de este eval set. Para las respuestas abiertas, el juez de otra familia es mejor que el de la misma, y los dos difieren entre sí más que cualquier brazo de cualquier etapa. Todo esto es contra un auditor que también es un modelo, aunque más grande y de otra familia.
"""

# %% [markdown]
r"""
### Los desacuerdos juez-contención, para leer

Cada uno con la pregunta, la clave, la respuesta y el análisis del juez. Se muestran recién con la auditoría completa, para no sesgarla.
"""

# %%
if pending:
    print(f"La auditoría tiene {pending} pendientes: los desacuerdos se muestran cuando esté completa.")
else:
    for r in judge.disagreements(PRIMARY):
        side = "solo el juez" if r["judge_verdict"] == "correct" else "solo contención"
        print(f"\n--- {side} · {r['uid']} [{r['kind']}] can_answer={r['can_answer']}")
        print(wrap(f"P: {r['question']}"))
        print(wrap(f"R: {r['answer']}"))
        print(f"   clave: {r['answer_key']} · veredicto esperado: {r['expected_verdict']}")
        print(wrap(f"juez: {r['judge_analysis']}"))

# %% [markdown]
r"""
## Para el informe

- **¿Dónde el juez no coincidió con la contención y con la auditoría, y en cuál se confía?** Con la contención, sobre todo en las preguntas con respuesta (33 contra 1) y en las preferencias. Con la auditoría, `qwen3:8b` rechaza 36 respuestas correctas. La contención es la más cercana al auditor (73/82, sin aceptar nunca una incorrecta): las tablas de las Etapas 4-6 se leen con ella, y son una cota inferior.
- **¿Cuánto importa la autopreferencia, y cómo se sabe?** Con dos jueces de familias distintas: el de la misma familia no es indulgente, sino el más estricto. El otro juez es mucho más indulgente que los dos. Que la diferencia entre jueces sea más grande que cualquier efecto de las etapas es en sí un resultado.
"""

# %% [markdown]
r"""
## 7 · 7b, el framework: recortada

El plan del grupo recorta desde el final, y lo primero que se recorta es 7b (decision_log.md, sección 1: 7b → 7a → ...). Reimplementar el loop de las Etapas 5-6 en LangGraph o LlamaIndex cuesta volver a correr todas las sondas de la Etapa 6 en una segunda implementación, a días de la entrega. Lo que 7b enseñaría (qué decide un framework por vos: `/v1` y su ventana de 4.096, `think` sin mandar, cuándo resume) ya está medido o controlado a mano en nuestro loop. El juez (7a) vale más por su costo: es una segunda lectura de **todas** las respuestas, y su auditoría es lo que pide la rúbrica.
"""

# %%
run.finish()
