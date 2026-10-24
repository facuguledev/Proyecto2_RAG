"""Un juez LLM (Etapa 7a): una segunda lectura de las respuestas guardadas en las Etapas 4-6.

La contención (``grounded.score``) puntúa por cadena: no ve una paráfrasis ("medio año" por "seis meses") y
cuenta una clave que la respuesta niega después. El juez lee lo mismo que un humano: la pregunta, la clave
y la respuesta, en campos separados y etiquetados. Es un instrumento, y se calibra contra una persona
antes de leerlo (assignment, Etapa 7a).

Decisiones de diseño:
- **Qué recibe.** Solo el campo ``answer``, nunca ``analysis`` (es razonamiento del generador, no lo que
  vio el usuario). Sin el contexto recuperado: acá no se evalúa fidelidad, y con el contexto el juez
  premiaría ser fiel al pasaje equivocado. En las preguntas sin respuesta recibe también ``can_answer``.
- **Una rúbrica por tipo de ítem** (``RUBRICS``): con respuesta, sí/no, sin respuesta y actualización de
  conocimiento. La de actualización sigue la regla de LongMemEval (Apéndice B.5): una respuesta con el
  valor nuevo cuenta aunque también mencione el viejo.
- **Los errores de llamada no se juzgan**: son su propia categoría en la etapa que los produjo.
- **El juez por defecto es ``qwen3:8b``, el mismo modelo que el generador.** La autopreferencia es real y
  se nombra; un segundo juez de otra familia corre en una pasada aparte sobre las mismas respuestas.

Formato de las filas que se juzgan (las escribe ``grounded.score`` en la Etapa 4): ``item_id``, ``arm``,
``answer`` = {analysis, answer, can_answer, citations} o None, y ``score`` = {outcome, correct, ...}.
"""
import csv
import hashlib
import io
import json
import math
import random
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field

import llm
from evalset import Answer, EvalSet
from stats import paired
from store import read_jsonl

JUDGE_MODEL = "qwen3:8b"
NUM_CTX = 8192  # el mismo del generador: cambiarlo entre etapas recargaría el modelo
NUM_PREDICT = 320  # un veredicto y dos oraciones de análisis
PROMPT_VERSION = 2  # la plantilla de judge_messages: si cambia, los veredictos viejos no se reusan

Kind = Literal["answerable", "yes_no", "unanswerable", "knowledge_update", "preference"]
Source = Literal["judge", "contention", "human"]


class JudgeVerdict(BaseModel):
    # El razonamiento primero en la declaración y en orden alfabético (analysis < verdict): el juez razona
    # antes de comprometerse en cualquier servidor (llm.probe_key_order).
    analysis: str = Field(max_length=400)
    verdict: Literal["correct", "incorrect"]


SYSTEM = """Sos un evaluador. Recibís una pregunta sobre legislación española de vivienda y alquiler, la clave de respuesta (lo que una respuesta correcta tiene que decir) y la respuesta de un asistente, cada una en su campo. Decidís si la respuesta es correcta según la regla del ítem.

- Juzgá solo contra la clave y la regla: no uses lo que sepas de la ley, y no premies una respuesta por ser larga o segura de sí misma.
- La clave puede venir en varias formas equivalentes, separadas por " | ". Alcanza con que la respuesta diga una, con esas u otras palabras que signifiquen lo mismo.
- Una respuesta que dice la clave pero también afirma algo que la contradice es incorrecta.
- En analysis, primero y en una o dos oraciones: qué dice la respuesta frente a la clave. Después verdict: "correct" o "incorrect"."""

RUBRICS: dict[str, str] = {
    "answerable": ("La pregunta tiene respuesta. La respuesta es correcta si transmite la clave. Si el asistente "
                   "dice que no puede responder, es incorrecta."),
    "yes_no": ("La pregunta se responde con sí o no. La respuesta es correcta si su veredicto (sí o no) coincide "
               "con el veredicto esperado y además transmite la clave. Si el asistente dice que no puede "
               "responder, es incorrecta: abstenerse no es decir que no."),
    "unanswerable": ("La pregunta NO tiene respuesta en las fuentes del asistente. La respuesta es correcta si y "
                     "solo si el asistente se abstiene: can_answer es false y la respuesta no afirma ningún dato "
                     "como si lo supiera. Cualquier respuesta concreta es incorrecta, aunque suene plausible."),
    # Los pares de actualización comparten la clave y difieren en el veredicto: la sonda del "antes" se
    # responde con la situación anterior. El juez no recibe el valor viejo ni la fecha, así que la regla de
    # LongMemEval ("el valor nuevo cuenta aunque se mencione el viejo") se aplica como: lo que importa es el
    # veredicto a la fecha de la pregunta, y mencionar la otra situación no lo hace incorrecto.
    "knowledge_update": ("La situación del usuario cambió con el tiempo. La respuesta es correcta si transmite la "
                         "clave y corresponde a la situación del usuario al momento de la pregunta; mencionar la "
                         "situación anterior no la hace incorrecta. Si el asistente dice que no puede responder, "
                         "es incorrecta."),
    # Una preferencia sobre cómo responder ("decime siempre el número de la norma", "llamame Pili") se puntúa
    # por contención literal (decision_log, Etapa 2): una paráfrasis no la cumple.
    "preference": ("La pregunta pone a prueba una preferencia que el usuario expresó antes sobre cómo quiere que "
                   "se le responda, y la clave es ese elemento literal (un número de norma, un artículo, un "
                   "nombre, una fecha). La respuesta es correcta si y solo si contiene la clave tal cual; una "
                   "paráfrasis o una descripción no cuentan."),
}
VERDICT_RULE = " Además, su veredicto (sí o no) tiene que coincidir con el veredicto esperado: abstenerse no es decir que no."


def rubric_sha() -> str:
    """Va en el run_spec: una rúbrica, una plantilla o un schema editados son otro juez, y sus veredictos no se
    mezclan con los viejos."""
    spec = [SYSTEM, RUBRICS, VERDICT_RULE, PROMPT_VERSION, JudgeVerdict.model_json_schema()]
    return hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def kind_of(target: Answer, ability: str | None = None) -> Kind:
    if not target.answerable:
        return "unanswerable"
    if ability in ("knowledge_update", "preference"):
        return ability
    return "yes_no" if target.expected_verdict else "answerable"


# --- Qué se juzga --------------------------------------------------------------------------------

class Target(NamedTuple):
    """Lo que el eval set dice de un id: la pregunta que ve el juez, su respuesta esperada y su tipo."""
    question: str
    answer: Answer
    kind: Kind
    split: str


@dataclass(frozen=True)
class Judgeable:
    stage: str  # "04", "05", "06"
    arm: str
    item_id: str
    split: str
    kind: Kind
    question: str
    answer_key: tuple[str, ...]
    expected_verdict: str | None
    can_answer: bool
    answer: str  # solo el campo answer: el analysis del generador nunca llega al juez
    contained: bool  # el acierto sin juez de la etapa (score.correct)
    source_config: str = ""  # el config_key de la fila juzgada (regla 12: cada fila sabe de dónde salió)

    @property
    def uid(self) -> str:
        return f"{self.stage}/{self.arm}/{self.item_id}"


def targets_stage4(ev: EvalSet) -> dict[str, Target]:
    return {item.item_id: Target(item.question, item, kind_of(item), item.split) for item in ev.items}


def targets_stage5(ev: EvalSet) -> dict[str, Target]:
    """Las filas de ``05_history.jsonl``: ``item_id`` = "{dialogue_id}.t{n}" (``conversation.replay``).

    El juez recibe la reescritura oráculo del turno, no lo que escribió el usuario: "¿y si fuera un
    local?" no se puede juzgar sin el historial, y el juez no lo recibe.
    """
    return {f"{dialogue.dialogue_id}.t{n}": Target(turn.oracle_rewrite, turn, kind_of(turn), dialogue.split)
            for dialogue in ev.dialogues for n, turn in enumerate(dialogue.turns, start=1)}


def targets_stage6(ev: EvalSet) -> dict[str, Target]:
    """Las filas de ``06_memory.jsonl``: ``item_id`` = ``probe_id``."""
    return {probe.probe_id: Target(probe.question, probe, kind_of(probe, probe.ability), scenario.split)
            for scenario in ev.scenarios for probe in scenario.probes}


TARGETS: dict[str, Callable[[EvalSet], dict[str, Target]]] = {
    "04": targets_stage4, "05": targets_stage5, "06": targets_stage6}


def latest_rows(path: Path, eval_key: str, config_key: str | None = None) -> list[dict]:
    """Las filas de un ``store.ResultsLog`` de otra etapa: la última de cada (brazo, ítem), de una configuración.

    Sin ``config_key`` se toma la de la última fila escrita (la corrida más reciente). Su ``run_spec``
    tiene que ser contra este eval set: si no, se juzgarían respuestas a otras preguntas sin ningún error.
    Una última línea cortada (una corrida interrumpida a mitad de escritura) se saltea, como en ``store``.
    """
    rows = read_jsonl(path)
    if not rows:
        return []
    config_key = config_key or rows[-1]["config_key"]
    spec_path = path.with_name(f"{path.stem}.config-{config_key}.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))["spec"]
    if spec.get("eval") != eval_key:
        raise ValueError(f"{path.name} (config {config_key}) se puntuó contra el eval set {spec.get('eval')}, "
                         f"no contra {eval_key}")
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("config_key") == config_key:
            latest[(str(row["arm"]), str(row["item_id"]))] = row
    return list(latest.values())


def judgeables(rows: list[dict], targets: dict[str, Target], stage: str) -> tuple[list[Judgeable], Counter]:
    """Las filas que se juzgan, y por qué se dejaron afuera las demás:

    - ``error``: un error de llamada (su propia categoría en la etapa que lo produjo);
    - ``no_answer``: una fila que no es una respuesta (p. ej. la recuperación de la Etapa 5A);
    - ``unknown_id``: un id que no es una pregunta del eval set (p. ej. la sonda del resumen de la 5B). La
      procedencia ya la chequeó ``latest_rows`` con el hash del eval set: no es un archivo equivocado.
    """
    found, skipped = [], Counter()
    for row in rows:
        if "answer" not in row or "score" not in row:
            skipped["no_answer"] += 1
            continue
        if row["item_id"] not in targets:
            skipped["unknown_id"] += 1
            continue
        if row["answer"] is None or row["score"].get("outcome") == "error":
            skipped["error"] += 1
            continue
        target = targets[row["item_id"]]
        found.append(Judgeable(
            stage=stage, arm=str(row["arm"]), item_id=row["item_id"], split=target.split, kind=target.kind,
            question=target.question, answer_key=tuple(target.answer.answer_key),
            expected_verdict=target.answer.expected_verdict, can_answer=bool(row["answer"]["can_answer"]),
            answer=row["answer"]["answer"], contained=bool(row["score"]["correct"]),
            source_config=str(row.get("config_key", ""))))
    return found, skipped


def judge_key(j: Judgeable) -> str:
    """El id de un juicio en el ResultsLog. Cambia si cambia la respuesta (una etapa re-corrida se re-juzga) o
    su puntaje sin juez (así la contención guardada nunca es vieja). Re-juzgar un mensaje idéntico no cuesta:
    lo devuelve la caché de llamadas."""
    payload = [j.answer, j.can_answer, j.contained, j.kind]
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[:8]
    return f"{j.stage}/{j.item_id}@{digest}"


# --- La llamada ----------------------------------------------------------------------------------

def judge_messages(j: Judgeable) -> list[dict]:
    rubric = RUBRICS[j.kind] + (VERDICT_RULE if j.kind in ("knowledge_update", "preference") and j.expected_verdict
                                else "")
    parts = [f"Regla de este ítem: {rubric}", f"<pregunta>\n{j.question}\n</pregunta>"]
    if j.kind != "unanswerable":
        parts.append(f"<clave>\n{' | '.join(j.answer_key)}\n</clave>")
    if j.expected_verdict:
        parts.append(f"<veredicto_esperado>{'Sí' if j.expected_verdict == 'yes' else 'No'}</veredicto_esperado>")
    if j.kind == "unanswerable":
        parts.append(f"<can_answer>{str(j.can_answer).lower()}</can_answer>")
    parts.append(f"<respuesta>\n{j.answer}\n</respuesta>")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


ABSTAINED = JudgeVerdict(analysis="can_answer=false en una pregunta con respuesta: incorrecta por regla, sin "
                                  "llamar al juez (la abstención es el booleano, no el texto).", verdict="incorrect")


def judge_one(j: Judgeable, *, model: str = JUDGE_MODEL, num_ctx: int = NUM_CTX, num_predict: int = NUM_PREDICT,
              cache: bool = True, structured: Callable = llm.structured) -> tuple[JudgeVerdict | None, dict]:
    """Un veredicto estructurado, temperatura 0. Un ``ContextOverflow`` es un error, como en la Etapa 4.

    Una abstención (``can_answer = false``) en un ítem con respuesta es incorrecta por regla y no se le
    pregunta al juez: si leyera el texto, decidiría la abstención por cadena, y un
    texto que menciona la clave mientras se abstiene le parecería correcto.
    """
    if j.kind != "unanswerable" and not j.can_answer:
        return ABSTAINED, {"error": None, "rule": "abstained", "model": model}
    try:
        verdict, r = structured(judge_messages(j), JudgeVerdict, model=model, num_ctx=num_ctx,
                                num_predict=num_predict, cache=cache)
    except llm.ContextOverflow as error:
        return None, {"error": f"CONTEXT_OVERFLOW (client): {error}", "model": model}
    return verdict, r.log()


def judged_row(j: Judgeable, verdict: JudgeVerdict | None, call: dict, model: str) -> dict:
    """La fila del ResultsLog del juez (sin ``arm``/``item_id``, que pone la notebook)."""
    return {**{k: v for k, v in asdict(j).items() if k not in ("arm", "item_id")},
            "source_arm": j.arm, "source_item": j.item_id, "uid": j.uid, "judge": model,
            "judge_verdict": verdict.verdict if verdict else None,
            "judge_analysis": verdict.analysis if verdict else None,
            "by_rule": call.get("rule") is not None,
            "error": call.get("error"), "call": call}


# --- Lectura -------------------------------------------------------------------------------------

def says_correct(row: dict, source: Source) -> bool | None:
    """El veredicto de una fuente como booleano; None si esa fuente no juzgó la fila."""
    if source == "judge":
        return None if row.get("judge_verdict") is None else row["judge_verdict"] == "correct"
    if source == "contention":
        return bool(row["contained"])
    verdict = (row.get("human_verdict") or "").strip().lower()
    return None if not verdict else verdict == "correct"


def agreement(rows: list[dict], a: Source, b: Source) -> dict:
    """Matriz de confusión (conteos) de ``a`` contra ``b`` y el test de signos sobre las discordantes.

    ``b_only`` son las filas que ``b`` da por correctas y ``a`` no: si el juez es ``b``, cuánto más
    indulgente es que ``a``.
    """
    pairs = {row["uid"]: (says_correct(row, a), says_correct(row, b)) for row in rows}
    pairs = {uid: (x, y) for uid, (x, y) in pairs.items() if x is not None and y is not None}
    matrix = Counter((x, y) for x, y in pairs.values())
    test = paired({uid: x for uid, (x, _) in pairs.items()}, {uid: y for uid, (_, y) in pairs.items()})
    return {"n": len(pairs), "agree": matrix[(True, True)] + matrix[(False, False)],
            "both_correct": matrix[(True, True)], "both_incorrect": matrix[(False, False)],
            "a_only": test["losses"], "b_only": test["wins"], "p_sign": test["p_sign"]}


def disagreements(rows: list[dict]) -> list[dict]:
    """Las filas juzgadas en las que el juez y la contención no coinciden: las que hay que leer."""
    return [row for row in rows if says_correct(row, "judge") is not None
            and says_correct(row, "judge") != says_correct(row, "contention")]


# --- Auditoría humana ----------------------------------------------------------------------------
# La planilla es ciega: la persona ve la pregunta, la clave y la respuesta, no el veredicto del juez ni el
# de la contención, ni la etapa o el brazo (un "closed_book" u "oracle" sugiere qué esperar). Todo eso queda
# en audit_key.jsonl y se junta al leer.

AUDIT_COLUMNS = ["audit_id", "kind", "question", "answer_key", "expected_verdict", "can_answer", "answer",
                 "human_verdict", "human_note"]
HIDDEN = ("uid", "stage", "source_arm", "source_item", "judge", "judge_verdict", "contained", "stratum",
          "stratum_size")


def audit_size(n_population: int, min_n: int = 30, fraction: float = 0.2) -> int:
    """Al menos 30 juicios o el 20%, lo que sea mayor (assignment, Etapa 7a), sin pasarse del total."""
    return min(n_population, max(min_n, math.ceil(fraction * n_population)))


def audit_sample(rows: list[dict], seed: int = 0, min_n: int = 30, fraction: float = 0.2) -> list[dict]:
    """Una muestra estratificada de juicios: los desacuerdos juez-contención, sobremuestreados.

    Hasta la mitad de la muestra son desacuerdos (todos, si son menos), y el resto acuerdos. Así la
    auditoría mira sobre todo donde los dos instrumentos discrepan, sin dejar de controlar los acuerdos
    (dos instrumentos pueden equivocarse igual). Cada fila lleva su estrato y el tamaño del estrato en la
    población, para reponderar (``population_agreement``).
    """
    judged = sorted((row for row in rows if says_correct(row, "judge") is not None), key=lambda r: r["uid"])
    n = audit_size(len(judged), min_n, fraction)
    disagree = [row for row in judged if says_correct(row, "judge") != says_correct(row, "contention")]
    agree = [row for row in judged if says_correct(row, "judge") == says_correct(row, "contention")]
    n_disagree = min(len(disagree), max(math.ceil(n / 2), n - len(agree)))
    rng = random.Random(seed)
    sample = ([{**row, "stratum": "desacuerdo", "stratum_size": len(disagree)}
               for row in rng.sample(disagree, n_disagree)] +
              [{**row, "stratum": "acuerdo", "stratum_size": len(agree)}
               for row in rng.sample(agree, n - n_disagree)])
    rng.shuffle(sample)  # mezclados: los estratos no se notan en el orden de la planilla
    return sample


def _sheet_rows(csv_path: Path) -> list[dict]:
    """Las filas de la planilla, con coma o con punto y coma: un Excel en español la guarda con ";".

    Del encabezado se deduce solo el separador; las comillas se leen como en cualquier CSV (``csv.excel``).
    Deducir el dialecto entero de una sola línea sin comillas leía mal un campo con comillas adentro, y
    desalineaba las columnas.
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    header = text.split("\n", 1)[0]
    delimiter = max(",;\t", key=header.count)
    return list(csv.DictReader(io.StringIO(text), dialect=csv.excel, delimiter=delimiter))


def _filled(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    return sum(bool((row.get("human_verdict") or "").strip()) for row in _sheet_rows(csv_path))


def write_audit(sample: list[dict], folder: Path) -> bool:
    """Escribe ``audit.csv`` (ciega, para completar a mano) y ``audit_key.jsonl`` (lo que no se muestra).

    No pisa una planilla que ya tiene veredictos humanos: devuelve False y deja todo como está. Una planilla
    sin completar se reemplaza, pero antes se copia a ``audit.csv.bak-<fecha>``: si la lectura de veredictos
    fallara, el trabajo de una persona no se pierde.
    """
    folder = Path(folder)
    csv_path = folder / "audit.csv"
    if _filled(csv_path):
        return False
    folder.mkdir(parents=True, exist_ok=True)
    if csv_path.is_file():
        shutil.copy2(csv_path, csv_path.with_name(f"audit.csv.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:  # con BOM: Excel lo abre bien
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for n, row in enumerate(sample, start=1):
            writer.writerow({
                "audit_id": f"a{n:03d}", "kind": row["kind"], "question": row["question"],
                "answer_key": " | ".join(row["answer_key"]), "expected_verdict": row["expected_verdict"] or "",
                "can_answer": row["can_answer"], "answer": row["answer"], "human_verdict": "", "human_note": ""})
    with open(folder / "audit_key.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for n, row in enumerate(sample, start=1):
            f.write(json.dumps({"audit_id": f"a{n:03d}", **{k: row[k] for k in HIDDEN}}, ensure_ascii=False) + "\n")
    return True


def read_audit(folder: Path) -> tuple[list[dict], int]:
    """Las filas auditadas (la planilla junto con lo oculto) y cuántas faltan completar.

    ``human_verdict`` es "correct", "incorrect" o vacío (pendiente); cualquier otra cosa es un error que
    dice en qué fila está.
    """
    folder = Path(folder)
    hidden = {}
    for line in (folder / "audit_key.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            hidden[row["audit_id"]] = row
    done, pending = [], 0
    for row in _sheet_rows(folder / "audit.csv"):
        verdict = (row.get("human_verdict") or "").strip().lower()
        if verdict not in ("", "correct", "incorrect"):
            raise ValueError(f"{row['audit_id']}: human_verdict es {row['human_verdict']!r}; "
                             "tiene que ser correct, incorrect o vacío")
        if row["audit_id"] not in hidden:
            raise KeyError(f"{row['audit_id']} no está en audit_key.jsonl: la planilla no es de esta muestra")
        if not verdict:
            pending += 1
            continue
        done.append({**row, **hidden[row["audit_id"]], "human_verdict": verdict})
    return done, pending


def refresh_audited(audited: list[dict], current: list[dict]) -> tuple[list[dict], int]:
    """Las filas auditadas con el veredicto ACTUAL del juez y la contención actual, y cuántas cambiaron desde
    que se armó la planilla. Si el juez se volvió a correr, "juez vs humano" compara contra el juez de ahora."""
    by_uid = {row["uid"]: row for row in current}
    fresh, changed = [], 0
    for row in audited:
        now = by_uid.get(row["uid"])
        if now is None:
            continue  # la respuesta auditada ya no está entre las juzgadas (su etapa se volvió a correr)
        changed += (now["judge_verdict"], now["contained"]) != (row["judge_verdict"], row["contained"])
        fresh.append({**row, "judge_verdict": now["judge_verdict"], "contained": now["contained"]})
    return fresh, changed


def population_agreement(audited: list[dict], a: Source, b: Source, strata: set[str] | None = None) -> float | None:
    """El acuerdo entre ``a`` y ``b`` estimado para toda la población, reponderando cada estrato por su tamaño.

    El acuerdo crudo de la muestra subestima al juez, porque la muestra sobrerrepresenta los desacuerdos.
    Con ``strata`` (los estratos de la muestra), devuelve None mientras alguno no tenga filas auditadas:
    descartarlo en silencio daría el acuerdo de un solo estrato como si fuera el de la población.
    """
    present = {row["stratum"] for row in audited}
    if strata is not None and not set(strata) <= present:
        return None
    total, weighted = 0, 0.0
    for stratum in present:
        rows = [row for row in audited if row["stratum"] == stratum]
        result = agreement(rows, a, b)
        if result["n"] == 0:
            continue
        size = rows[0]["stratum_size"]
        total += size
        weighted += size * result["agree"] / result["n"]
    return weighted / total if total else None
