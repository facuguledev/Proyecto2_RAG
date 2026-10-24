# %% [markdown]
r"""
# Etapa 2 · La vara de medir

Esta notebook muestra, valida y congela el conjunto de evaluación: lo que van a medir las Etapas 3 a 7. Son tres conjuntos, un archivo cada uno en `eval/`:

| Conjunto | Archivo | Contrato | Lo usa |
|---|---|---|---|
| Preguntas de un solo turno | `single_turn.jsonl` | `evalset.EvalItem` | Etapas 3-4 |
| Guiones de conversación | `dialogues.jsonl` | `evalset.Dialogue` | Etapa 5 |
| Escenarios de varias sesiones | `scenarios.jsonl` | `evalset.Scenario` | Etapa 6 |

El conjunto se **congela antes de que exista cualquier resultado** (`eval/FROZEN.json`). Desde ahí, `evalset.load_eval` rechaza un archivo editado o un corpus que cambió por debajo, y cualquier edición es una entrada de `decision_log.md`.

La lógica vive en `src/`:
- `evalset.py`: el contrato, la validación contra el corpus y el congelado. Es lo que el enunciado llama `schema.EvalItem`, `schema.freeze_eval`, etc. (el `schema.py` del curso no representa nuestro eval set: `decision_log.md`, «El repo del curso»);
- `lexical.py`: el canal léxico. El chequeo de contaminación y BM25 parten el texto con la misma función, `lexical.terms`.

Esta notebook corre, muestra y lee los resultados.
"""

# %%
import environment  # el del curso, primero: guarda de JAX (bm25s lo importa), caché de HF, UTF-8

import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = environment.ROOT
sys.path.insert(0, str(ROOT / "src"))  # nuestros módulos: evalset, lexical, doc_schema, artifact_io

import pandas as pd

try:
    from IPython.display import display  # en la notebook
except ImportError:
    display = print  # como script (tools/run_stages.py)

import artifact_io
import doc_schema
import evalset
import lexical

EVAL_DIR, SMOKE_DIR = ROOT / "eval", ROOT / "eval" / "smoke"
ART = environment.artifacts_dir(False)  # se congela contra el corpus completo, nunca contra el smoke
DOCS_PATH = ART / "documents.jsonl"
DOCS = doc_schema.load_documents(DOCS_PATH)
CORPUS_SHA = doc_schema.documents_hash(DOCS)
artifact_io.check_manifest(ART / "documents.manifest.json", {"corpus_sha": CORPUS_SHA})  # es el corpus de la Etapa 1
SECTIONS = evalset.section_index(DOCS)

pd.set_option("display.max_colwidth", 140)
pd.set_option("display.width", 220)
environment.summary()
print(f"corpus_sha {CORPUS_SHA[:12]}... | {len(DOCS)} normas, {len(SECTIONS):,} secciones")

# %% [markdown]
r"""
## 1 · Cómo se escribió

El método sale de la primera trampa de la consigna: una pregunta escrita con el pasaje abierto importa sus palabras raras, y BM25 las encuentra.

1. **Una lista de temas** (`eval/topics.md`): 30 temas. Salen de la corpus card, de los **títulos** de normas y artículos, y de lo que pregunta alguien que alquila, que tiene un piso o que vive en una comunidad.
2. **Por tema, la pregunta y la respuesta esperada**, en lenguaje de usuario (piso, casero, fianza), **antes** de buscar en el corpus.
3. **Después, el gold.** Se busca la sección y se verifica la respuesta contra su texto. La clave se copia del texto gold y la pregunta no. Si el corpus contradice lo esperado, cambia la clave, nunca la redacción.
4. **Contaminación y BM25** (secciones 5 y 6). Lo marcado se reescribe desde el tema.

Hay dos excepciones al paso 2:
- **Las de identificador se derivan,** no se escriben (sección 3).
- **Las 5 preguntas del chequeo a libro cerrado** de la Etapa 0 ya existían. Como vimos sus salidas, van a `dev`.

**Separación de roles (paso 5 de la consigna).** Los escenarios los escribió un autor que no construye la Etapa 6 y que no conoce su diseño. Se congelan antes de que exista una línea de código de memoria.

**Decidido antes de cualquier resultado:**
- **Una pregunta fuera del corpus** se responde bien si y solo si `can_answer = false`.
- **Pre-registro.** Si las columnas `topical` e `identifier` discrepan sobre una técnica, manda `topical`, porque es lo que pregunta un usuario real. `identifier` se lee como diagnóstico del canal léxico y del prefijo.
"""

# %% [markdown]
r"""
## 2 · Las preguntas de un solo turno

Cada una lleva:
- su categoría (`topical`, `identifier`, `metadata` u `out_of_corpus`);
- el gold, que es una lista de `section_id` donde vale cualquiera;
- una clave copiada del texto gold;
- el veredicto, si es de sí o no;
- `as_of`, si pregunta por una fecha.

`as_of` es una **etiqueta para validar el gold**. El pipeline no la lee, porque sería darle la respuesta: el filtro de la Etapa 3 tiene que sacar la fecha de la pregunta.
"""

# %%
ev = evalset.load_eval(EVAL_DIR, require_frozen=False)  # todavía sin congelar: se congela en la sección 10
items = pd.DataFrame([
    {"id": i.item_id, "split": i.split, "categoría": i.category, "pregunta": i.question,
     "clave": " | ".join(i.answer_key), "veredicto": i.expected_verdict, "as_of": i.as_of,
     "gold": ", ".join(i.gold_sections)}
    for i in ev.items])
display(pd.crosstab(items["categoría"], items["split"], margins=True))
test = items[items["split"] == "test"]
print("veredictos en test:", dict(Counter(test["veredicto"].dropna())),
      "| preguntas con fecha (as_of):", int(items["as_of"].notna().sum()))
display(items)

# %% [markdown]
r"""
**Lectura.**
- **Tamaños.** Son 30 preguntas de test con los mínimos de §7.3: 4 fuera del corpus, 6 de identificador y 7 de metadatos. Las 10 de dev incluyen las 5 del chequeo a libro cerrado.
- **Veredictos.** Hay 4 "sí" y 4 "no" en test: un brazo que dice que sí a todo acierta la mitad, no todas.
- **Fechas.** 9 preguntas llevan `as_of`, y son las que miden versiones. Por ejemplo, el contrato firmado en febrero de 2019 tenía 3 años de plazo mínimo y no 5.
- **Ruido.** Con n = 30, una fila de la Etapa 3 tiene unos ±12 puntos de ruido, y por categoría (13, 7 y 6 preguntas) bastante más. Las columnas por categoría se leen como una dirección, no como una prueba.
"""

# %% [markdown]
r"""
## 3 · Identificador: derivadas, no escritas

La consigna pide derivar las preguntas de identificador de "identificadores que aparecen en exactamente un documento". Acá el identificador es **un artículo pedido por su número**: "¿Qué regula el artículo 36 de la Ley 29/1994?".

Entra un artículo vigente si cumple tres cosas:
- su título no lo lleva ningún otro bloque del corpus (ni un artículo, ni una disposición, ni un capítulo), en ninguna versión;
- el título tiene a lo sumo 4 palabras;
- su norma tiene un número oficial único.

La clave es el título del artículo. Se toma una por norma, alternando estatales y autonómicas, con una semilla fija.

La primera versión de la regla solo pedía que el título fuera único entre artículos, y dejaba pasar títulos genéricos como "Entrada en vigor" u "Objeto y fines". Esos también aparecen en las disposiciones finales de casi todas las normas, y un modelo los adivina sin recuperar nada.

La celda deriva de nuevo y compara con el archivo: así se prueba que estas preguntas no se escribieron a mano.
"""

# %%
candidates = evalset.identifier_candidates(DOCS)
derived = evalset.derive_identifier_items(DOCS, n_dev=1, n_test=6)
assert derived == [i for i in ev.items if i.topic == evalset.DERIVED_TOPIC], "el archivo no tiene lo que deriva el código"
print(f"candidatos: {len(candidates)} artículos de {len({c.doc.doc_id for c in candidates})} normas")
display(pd.DataFrame([{"id": i.item_id, "split": i.split, "pregunta": i.question, "clave": i.answer_key[0],
                       "gold": ", ".join(i.gold_sections)} for i in derived]))

# %% [markdown]
r"""
## 4 · Validación contra el corpus

Las reglas se chequean en tres lugares.

**1. El modelo, al leer el archivo.** Son las reglas que no necesitan el corpus:
- una pregunta sin respuesta no lleva gold, clave ni veredicto;
- `out_of_corpus` va con `answerable=false`, y lo mismo `unanswerable` y `abstention`;
- solo el primer turno de un guion es `standalone`;
- la evidencia de una sonda es de una sesión anterior;
- cada escenario tiene un hecho que cambia y uno que nunca se dijo.

**2. `validate_eval`, contra el corpus:**
- `gold_exists`: cada id existe;
- `gold_in_force`: el gold rige el día por el que se pregunta. Es el compromiso de la Etapa 1 hecho chequeo: a una pregunta sin fecha le toca la versión vigente al 2026-10-24;
- `key_in_gold`: alguna clave está en el texto gold. Si no, *answer-delivered* (Etapa 3) no podría dar 1 ni con una recuperación perfecta;
- `twins_in_gold`: un gemelo vigente que contiene la clave va en el gold. Si no, recuperar la copia en otra norma contaría como un fallo;
- `unique_ids`.

**3. `check_sizes`, los tamaños de §7.3.** Va aparte porque el smoke se valida pero nunca se congela.
"""

# %%
evalset.validate_eval(ev, DOCS).show().raise_if_failed()
evalset.check_sizes(ev).show().raise_if_failed()

# %%
# ¿Qué fracción de las etiquetas gold son conjuntos? (pregunta para el informe)
answers = evalset.corpus_questions(ev) + evalset.probes(ev)
with_gold = [(name, a) for name, a in answers if a.gold_sections]
gold_sets = [(name, a) for name, a in with_gold if len(a.gold_sections) > 1]
print(f"gold como conjunto: {len(gold_sets)} de {len(with_gold)} preguntas con gold")
display(pd.DataFrame([{"pregunta": name, "gold": ", ".join(a.gold_sections)} for name, a in gold_sets]))

# %% [markdown]
r"""
**Lectura.** 7 de las 97 preguntas con gold (7 %) tienen un conjunto, y **ninguno por gemelos**: `twins_in_gold` no tuvo que agregar nada. Son respuestas que están en dos lugares:
- la ley estatal y la autonómica dicen lo mismo (la fianza en Aragón);
- o la respuesta necesita dos artículos (la infracción y su sanción; el registro de agentes y la obligación de inscribirse).

En las preguntas y los turnos, vale cualquiera de las secciones, y cada una contiene la clave. Una sección que solo **remite** a otra norma no entra: la ley foral navarra de habitaciones dice "se aplican las normas de vivienda" y la respuesta ("cinco años") está en la LAU. Si estuviera en el gold, recuperar la remisión contaría como acierto sin haber entregado la respuesta. En las sondas, el gold son las secciones en las que se apoya la respuesta, porque su clave puede salir de lo que dijo el usuario.
"""

# %% [markdown]
r"""
## 5 · Contaminación

Es el chequeo de la consigna, con `lexical.terms`: la misma partición en términos que usa BM25. Para cada pregunta con gold mide dos cosas contra cada sección gold, y se queda con la peor:
- `overlap`: qué fracción de sus términos aparece en el gold;
- `longest_shared_ngram`: la secuencia de términos más larga que comparten.

Se marca con 5 términos o más.

Se chequean:
- las preguntas de un turno;
- en los guiones, tanto el turno del usuario como la reescritura oráculo (en la Etapa 5 las dos son consultas);
- las sondas que tienen gold.
"""

# %%
def worst_contamination(text: str, answer) -> dict:
    """El chequeo contra cada sección gold; se queda con la que comparte el n-grama más largo."""
    checks = [lexical.contamination(text, evalset.section_text(SECTIONS, s)) for s in answer.gold_sections]
    return max(checks, key=lambda c: c["longest_shared_ngram"])


rows = []
for item in ev.items:
    if item.answerable:
        rows.append({"qué": "pregunta", "id": item.item_id, "grupo": item.category, "texto": item.question,
                     **worst_contamination(item.question, item)})
for dialogue in ev.dialogues:
    for n, turn in enumerate(dialogue.turns, start=1):
        if turn.answerable:
            for kind, text in (("turno", turn.user), ("reescritura", turn.oracle_rewrite)):
                rows.append({"qué": kind, "id": f"{dialogue.dialogue_id}.t{n}", "grupo": turn.depends_on,
                             "texto": text, **worst_contamination(text, turn)})
for scenario in ev.scenarios:
    for probe in scenario.probes:
        if probe.gold_sections:
            rows.append({"qué": "sonda", "id": probe.probe_id, "grupo": probe.ability, "texto": probe.question,
                         **worst_contamination(probe.question, probe)})
contamination = pd.DataFrame(rows)
display(contamination.groupby("qué").agg(n=("id", "size"), marcadas=("flag", "sum"),
                                         ngrama_max=("longest_shared_ngram", "max"),
                                         overlap_medio=("overlap", "mean")).round(2))
display(contamination.groupby(["qué", "longest_shared_ngram"]).size().unstack(fill_value=0))
print("marcadas (n-grama >= 5):")
display(contamination[contamination["flag"]])
print("las cinco con más solapamiento:")
display(contamination.sort_values(["longest_shared_ngram", "overlap"], ascending=False).head(5))

# %% [markdown]
r"""
**Lectura.**
- **Ninguno de los 141 textos chequeados** comparte 5 términos seguidos con su gold. El máximo es 4, en dos preguntas y dos reescrituras.
- **Las que más solapan nombran algo por su nombre oficial**: "tipo de interés efectivo anual", "Plan de Vivienda 2005-2008", "Bono Alquiler Joven". No hay otra forma de decirlo, y por eso no se reescribieron.
- **Un dato para la Etapa 5A.** Los turnos del usuario solapan bastante menos con el gold que sus reescrituras oráculo (0,31 contra 0,51 de media): la reescritura repone en palabras completas lo que el usuario dejó implícito. Si la reescritura oráculo le gana al último turno en BM25, parte de esa ventaja es este solapamiento, no la comprensión de la conversación.
"""

# %% [markdown]
r"""
## 6 · Solo BM25

La consigna pide correr ya una recuperación solo con BM25. Es un segundo detector de contaminación: una pregunta que copia al corpus queda primera.

Así se arma:
- **Qué se indexa:** cada sección entera, en todas sus versiones (8.687). Es la versión provisoria del canal léxico; la Etapa 3 indexa chunks.
- **Qué se mide:** dónde aparece el primer gold entre los 100 primeros, en tres niveles:
  - el **documento** (la norma);
  - el **bloque** (el artículo, en cualquiera de sus versiones);
  - la **sección** (la versión exacta).

  La distancia entre bloque y sección es el problema de las versiones, y no es un error de BM25: dos versiones pueden tener el mismo texto. Los empates se ordenan por id, así que gana la versión más vieja; el nivel bloque no tiene ese sesgo.
- **Secciones con la clave.** *Answer-delivered* (Etapa 3) cuenta la clave en cualquier pasaje entregado. Si la clave está en cientos de secciones ("cinco años"), esa métrica da 1 casi siempre, y para esa pregunta no dice nada: se lee con esta columna al lado.
"""

# %%
start = time.time()
ids = list(SECTIONS)
bm25 = lexical.Bm25Index(ids, [evalset.section_text(SECTIONS, i) for i in ids])
print(f"BM25 sobre {len(ids):,} secciones en {time.time() - start:.0f} s")


def gold_ranks(question: str, answer, k: int = 100) -> dict:
    ranked = [section_id for section_id, _ in bm25.search(question, k=k)]
    gold_blocks = [evalset.block_of(s) for s in answer.gold_sections]
    return {"rango_documento": evalset.first_rank([evalset.doc_of(s) for s in ranked], answer.gold_docs),
            "rango_bloque": evalset.first_rank([evalset.block_of(s) for s in ranked], gold_blocks),
            "rango_sección": evalset.first_rank(ranked, answer.gold_sections)}


NORMALIZED = {i: evalset.normalize(evalset.section_text(SECTIONS, i)) for i in ids}  # una sola vez


def sections_with_key(answer) -> int:
    """En cuántas secciones vigentes el día de la pregunta aparece alguna de sus claves."""
    patterns = [evalset.key_pattern(key) for key in answer.answer_key]
    day = answer.reference_day()
    return sum(any(p.search(NORMALIZED[i]) for p in patterns) for i, (_, s) in SECTIONS.items() if s.in_force(day))


ranks = pd.DataFrame([{"id": i.item_id, "split": i.split, "categoría": i.category, **gold_ranks(i.question, i),
                       "secciones_con_la_clave": sections_with_key(i)} for i in ev.items if i.answerable])


def hits(column: pd.Series, k: int) -> str:
    return f"{int((column <= k).sum())}/{len(column)}"  # un rango vacío (no está en el top 100) no suma


summary = []
for category, group in ranks[ranks["split"] == "test"].groupby("categoría"):
    for level in ("documento", "bloque", "sección"):
        column = group[f"rango_{level}"]
        summary.append({"categoría": category, "nivel": level,
                        **{f"hit@{k}": hits(column, k) for k in (1, 5, 10, 100)}})
display(pd.DataFrame(summary))
display(ranks)
answered_turns = [t for d in ev.dialogues for t in d.turns if t.answerable]
common = sum(sections_with_key(t) >= 50 for t in answered_turns) + int((ranks["secciones_con_la_clave"] >= 50).sum())
print(f"preguntas y turnos con una clave que está en 50 secciones o más: {common} de "
      f"{len(ranks) + len(answered_turns)}")

# %%
# Las preguntas fuera del corpus: el top 3 de BM25, para confirmar que ninguna sección las responde.
for item in ev.items:
    if not item.answerable:
        print(f"{item.item_id}: {item.question}")
        for section_id, score in bm25.search(item.question, k=3):
            doc, section = SECTIONS[section_id]
            print(f"    {score:5.1f}  {section.title[:60]}  |  {doc.title[:60]}")

# %% [markdown]
r"""
**Lectura.**
1. **BM25 no pone primera ninguna sección gold** (0 de 26 preguntas de test con respuesta). Si las preguntas copiaran el pasaje, BM25 las encontraría arriba. Es la evidencia de que el método funcionó.
2. **El vocabulario no coincide.** 6 de las 13 `topical` no tienen su sección gold ni en el top 100: el usuario dice "caldera" y "traspasar", la ley dice "reparaciones" y "cesión". Eso tiene que resolverlo el canal denso.
3. **Documento, bloque y sección quedan lejos:**
   - En `metadata`, el documento sale primero en 6 de 7 y el artículo correcto (en alguna versión) entra al top 10 en 6, pero la versión exacta solo en 4. La distancia entre bloque y sección son las versiones: BM25 no puede distinguir textos casi iguales, y ese es el lugar del filtro por fecha de la Etapa 3.
   - En `identifier`, ninguna sección entra al top 10, porque el número de la norma no está en el texto de la sección. El documento sí entra en 5 de 6. Eso tendría que arreglarlo el prefijo.
4. **Las claves comunes.** 29 de las 78 preguntas y turnos con respuesta tienen una clave que aparece en 50 secciones o más ("cinco años" en 188, "competentes en materia de vivienda" en 549). Para esas, *answer-delivered* (Etapa 3) va a dar 1 casi con cualquier pasaje: hay que leerla al lado de la sección recuperada, no sola.
5. **Las 6 preguntas fuera del corpus** traen vecinos plausibles y ninguna respuesta. La más peligrosa es oc03: BM25 trae las resoluciones de 2014 a 2016 con "nuevo tipo de interés efectivo anual", así que un sistema que no lea el año respondería con el tipo de otro año.
"""

# %% [markdown]
r"""
## 7 · Guiones

Cada turno lleva cuatro cosas:
- su **dependencia** del historial: `standalone` (solo el primero), `coreference`, `ellipsis`, `topic_shift`, `return` o `unanswerable`;
- el gold y la clave;
- la **respuesta guionada** del asistente: en la Etapa 5 todos los brazos ven esta respuesta como historial, no la suya;
- la **reescritura oráculo**: la pregunta reescrita para entenderse sin historial.

**Regla de redacción:** el turno 1 da un dato personal (la fecha del contrato, la región, si es un local), y el último turno vuelve a él. Esa es la prueba del resumen de la Etapa 5B, ya congelada.
"""

# %%
turns = pd.DataFrame([
    {"guion": d.dialogue_id, "split": d.split, "turno": n, "depende": t.depends_on, "usuario": t.user,
     "asistente": t.assistant, "reescritura": t.oracle_rewrite, "clave": " | ".join(t.answer_key),
     "veredicto": t.expected_verdict, "as_of": t.as_of, "gold": ", ".join(t.gold_sections)}
    for d in ev.dialogues for n, t in enumerate(d.turns, start=1)])
display(pd.crosstab(turns["depende"], turns["split"], margins=True))
display(pd.crosstab(turns["depende"], turns["turno"]))
last = turns.groupby("guion").tail(1)
print(f"guiones cuyo último turno vuelve al principio (return): {int((last['depende'] == 'return').sum())}/{len(last)}")

# %%
for dialogue_id, rows in turns.groupby("guion"):
    print(f"=== {dialogue_id} ({rows['split'].iloc[0]})")
    display(rows.drop(columns=["guion", "split"]).set_index("turno"))

# %% [markdown]
r"""
**Lectura.**
- **Etiquetas.** Hay 42 turnos de test en 8 guiones, y cada etiqueta de seguimiento aparece al menos 5 veces: 10 `coreference`, 5 `ellipsis`, 6 `topic_shift`, 5 `unanswerable` y 8 `return`.
- **`return` queda confundido con el turno tardío.** Los 10 guiones terminan volviendo al dato del turno 1, así que `return` se concentra en los turnos 4 a 6. En la Etapa 5A, una caída en el turno 6 puede venir de cualquiera de las dos cosas; para separarlas sirven los dos `return` del medio.
- **Los turnos sin respuesta** están en el medio (turnos 3 a 5) y son adyacentes: una ley de Castilla-La Mancha, un artículo 17 ter que no existe, el precio medio de una habitación, lo que tarda un juzgado.
- **Fechas.** Tres guiones (dl01, dl06 y dl10) preguntan por un contrato de una fecha pasada: todo su gold es la versión de esa fecha. Un recuperador sin filtro de fecha va a traer la vigente.
"""

# %% [markdown]
r"""
## 8 · Escenarios

Una persona ficticia, sesiones con **fechas simuladas** y sondas hechas después. Cada sonda tiene:
- la **habilidad** que evalúa (`extraction`, `multi_session`, `temporal`, `knowledge_update`, `abstention`, `preference`);
- las sesiones donde está la evidencia.

Cada escenario tiene un hecho que **cambia**, medido como par: la misma pregunta antes y después del cambio, con veredictos opuestos. También tiene un hecho que **nunca se dijo**.
"""

# %%
probes = pd.DataFrame([
    {"escenario": s.scenario_id, "split": s.split, "sonda": p.probe_id, "habilidad": p.ability,
     "pregunta": p.question, "asked_at": p.asked_at, "evidencia": ", ".join(p.evidence_sessions),
     "clave": " | ".join(p.answer_key), "veredicto": p.expected_verdict, "gold": ", ".join(p.gold_sections)}
    for s in ev.scenarios for p in s.probes])
display(pd.crosstab(probes["habilidad"], probes["split"], margins=True))

# %%
for scenario in ev.scenarios:
    print(f"=== {scenario.scenario_id} ({scenario.split}): {scenario.persona}")
    for session in scenario.sessions:
        print(f"--- {session.session_id}, {session.started_at:%Y-%m-%d %H:%M}")
        for message in session.messages:
            print(f"   {message.role:>9}: {message.content}")
    display(probes[probes["escenario"] == scenario.scenario_id].drop(columns=["escenario", "split"]).set_index("sonda"))

# %% [markdown]
r"""
**Lectura.**
- **Habilidades.** Hay 22 sondas de test en 3 escenarios, con las seis habilidades:
  - 6 de `knowledge_update` (tres pares de antes y después, con veredictos opuestos);
  - 4 de `multi_session`;
  - 3 de cada una de las otras: `extraction`, `temporal`, `abstention` y `preference`.
- **Preferencias.** Las tres de test se puntúan por contención ("29/1994", "artículo 10", "Pili"). Dos también exigen el contenido correcto: solo la de "Pili" mide la preferencia sola.
- **La memoria como filtro.** Tres escenarios preguntan por el depósito de la fianza, y cada región responde distinto:
  - Aragón: dos meses;
  - Baleares: 30 días, ante el IBAVI;
  - Andalucía: desde 2026 ya no se deposita.
- **Fechas.** Todas las sesiones son anteriores a la foto del corpus (2026-10-24), y las sondas se hacen entre mayo y octubre de 2026, con fechas simuladas.
- **Lo que no cubren:** una preferencia que cambia, un dato que dijo el asistente (y no el usuario), y un escenario donde la memoria tendría que olvidar algo a pedido.
"""

# %% [markdown]
r"""
## 9 · Smoke

`eval/smoke/` es un conjunto mínimo para probar que el código corre (§7.4). Tiene:
- 6 preguntas, todas de dev, una o dos por categoría;
- 2 guiones de 3 turnos: dl10 recortado, y uno propio del smoke (el otro guion dev usa la ley foral navarra, que no está en el perfil smoke);
- 1 escenario de 2 sesiones y 3 sondas (sc04 recortado).

Todo el gold está en las 10 normas del perfil smoke. Se carga sin congelar, sus números no significan nada y nunca toca el conjunto congelado.

**La guarda del id faltante (§7.4).** Un id gold que no existe tiene que saltar. Se arma en memoria y no en el archivo: en disco haría fallar a propósito cada corrida smoke de las etapas siguientes.
"""

# %%
smoke = evalset.load_eval(SMOKE_DIR, require_frozen=False)
smoke_docs = doc_schema.load_documents(environment.artifacts_dir(True) / "documents.jsonl")
evalset.validate_eval(smoke, smoke_docs).show().raise_if_failed()

first = smoke.items[0]
broken = first.model_copy(update={"gold_sections": first.gold_sections + ["BOE-A-1994-26003#a999@20000101"]})
report = evalset.validate_eval(evalset.EvalSet([broken], [], [], {}), smoke_docs).show()
assert [check.name for check in report.checks if check.failures] == ["gold_exists"]

# %% [markdown]
r"""
## 10 · Congelar

`freeze_eval` hace tres cosas, en orden:
1. valida contra el corpus completo (sección 4);
2. chequea los tamaños de §7.3;
3. escribe `eval/FROZEN.json`, con el sha256 de cada archivo y el `corpus_sha`.

Volver a correrlo sin cambios no hace nada. Si algo cambió, falla hasta que haya una entrada en `decision_log.md` y se congele de nuevo con `overwrite=True`.

**`FROZEN.json` se commitea antes de que exista cualquier resultado.**
"""

# %%
frozen = evalset.freeze_eval(EVAL_DIR, DOCS_PATH)
print(json.dumps(frozen, indent=1))
ev = evalset.load_eval(EVAL_DIR, DOCS_PATH)  # congelado: rechaza un archivo editado o un corpus distinto
print("key del conjunto (va en cada fila de resultados):", ev.key())

# %%
# La guarda en acción: una copia del conjunto con una pregunta editada no carga.
with tempfile.TemporaryDirectory() as tmp:
    copy = Path(tmp) / "eval"
    shutil.copytree(EVAL_DIR, copy, ignore=shutil.ignore_patterns("smoke", "topics.md"))
    rows = artifact_io.read_jsonl(copy / "single_turn.jsonl")
    rows[0]["question"] += " (editada)"
    artifact_io.write_jsonl(copy / "single_turn.jsonl", rows)
    try:
        evalset.load_eval(copy, DOCS_PATH)
        raise AssertionError("una copia editada no debería cargar")
    except ValueError as error:
        print("rechazada:", error)

# %% [markdown]
r"""
## 11 · Para el informe
"""

# %% [markdown]
r"""
Lo que esta etapa deja para las preguntas del informe:

- **¿Cómo se evitó que las preguntas copiaran al corpus?** Se escribieron desde temas, con la respuesta esperada antes de buscar; las claves se copiaron del gold y las preguntas no.
  - Ninguno de los 141 textos chequeados comparte 5 términos seguidos con su gold (el máximo es 4).
  - BM25 no pone primera ninguna sección gold.
  - Las de identificador se derivaron con código.
- **¿Qué fracción del gold son conjuntos?** El 7 % (7 de 97), todos porque la respuesta está en dos lugares: una ley estatal y una autonómica, o dos artículos. Ninguno por gemelos. Las secciones que solo remiten a otra norma no entran en el gold.
- **¿Qué habilidades de memoria se cubren?** Las seis, con 3 a 6 sondas de test cada una (sección 8). Afuera quedan las preferencias que cambian, los datos dichos por el asistente y el olvido a pedido.
- **Una pregunta real que el conjunto no puede medir:** "¿Me conviene más alquilar o comprar con la ayuda del Plan Estatal?". Pide un consejo y una cuenta con datos del usuario, no una regla que esté en un artículo, y la contención de una clave no puede puntuarla.
- **Si `topical` e `identifier` no coinciden sobre una técnica, ¿en cuál se confía?** En `topical`, decidido antes de cualquier resultado: es lo que pregunta un usuario real. `identifier` mide un caso de búsqueda literal que la sección 6 ya mostró que depende del prefijo.
"""
