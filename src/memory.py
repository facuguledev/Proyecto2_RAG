"""Memoria entre sesiones (Etapa 6): el almacén, su camino de escritura y su camino de lectura.

En el vocabulario de CoALA (Apéndice B.4): la ventana que arma ``grounded.build_messages`` es la memoria de
trabajo; las transcripciones de las sesiones, la episódica; los hechos extraídos, la semántica; las
preferencias sobre cómo responder, la procedimental. Acá viven las tres últimas como registros.

Tres decisiones que vienen del enunciado:
- **Invalidar, no borrar** (Zep/Graphiti): UPDATE e INVALIDATE cierran el registro viejo (``valid_to``) y
  guardan el hecho nuevo. "¿Qué había dicho antes?" sigue teniendo respuesta, y una sonda hecha *antes*
  del cambio ve el valor viejo como vigente.
- **El tiempo es el del escenario, nunca el reloj.** ``created_at`` y ``valid_from`` son el comienzo
  simulado de la sesión; la recencia se mide en horas simuladas.
- **Nunca se guarda razonamiento.** De la extracción y de la decisión se guardan solo los campos del hecho;
  ``analysis`` no llega a ningún registro ni al log de operaciones, y ``leaked`` lo verifica.

La memoria también alimenta la recuperación (paso 4): ``memory_filters`` arma el ``retrieval.Filters`` de la
Etapa 3 con la comunidad y la fecha del contrato que dicen los recuerdos (brazo ``store_filter``).

Todo lo que habla con el modelo o con el embedder se inyecta (``structured``, ``encode``, ``count``): los
tests corren sin servidor y la notebook pasa ``llm_structured`` y el bge-m3 del índice de la Etapa 3.
"""
import csv
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

import grounded
import llm
import retrieval
from artifact_io import write_jsonl
from evalset import Probe, Scenario, Session, normalize

MEMORY_BUDGET = 600  # tokens del generador para el espacio de memoria (§7.3: 300-1.000)
MEMORY_K = 6  # recuerdos como máximo por sonda (§7.3: 3-10)
REPLAY_BUDGET = 3000  # "si entra, pegalo": la reproducción con lugar para todo el historial
SIMILAR_K = 3  # recuerdos vigentes que ve la decisión de escritura
RECENCY_DECAY = 0.995  # por hora simulada (Generative Agents, §7.3); ver ``recall`` para la escala
EXTRACT_PREDICT = 768
DECIDE_PREDICT = 384
LINE_MARGIN = 2  # el salto de línea entre renglones, que la cuenta de texto pelado no ve

# El orden de corrida, fijo. replay_600 es la reproducción con el presupuesto del almacén: entre ella y
# store cambia una sola cosa (qué se pone en los mismos 600 tokens), y es el régimen en que el historial
# NO entra. replay (3.000) es la línea de base "si entra, pegalo". store_filter es store con los pasajes
# filtrados por lo que dice la memoria (``memory_filters``): el mismo espacio de memoria, otra recuperación.
ARMS = ["no_memory", "replay", "replay_600", "store", "store_filter"]
REPLAY_BUDGETS = {"replay": REPLAY_BUDGET, "replay_600": MEMORY_BUDGET}
STORE_ARMS = ("store", "store_filter")
NO_MEMORY_TEXT = "(no hay recuerdos de conversaciones anteriores con este usuario)"
VERDICTS = ("bien", "mal", "nunca_se_dijo")

Kind = Literal["semantic", "episodic", "procedural"]


# --- Schemas ---------------------------------------------------------------------------------------

class Fact(BaseModel):
    kind: Kind
    subject: str = Field(max_length=60)
    text: str = Field(max_length=300)
    importance: int = Field(ge=1, le=10)


class Extraction(BaseModel):
    # El razonamiento primero y también primero en orden alfabético (analysis < facts).
    analysis: str = Field(max_length=600)
    facts: list[Fact]


class WriteDecision(BaseModel):
    # "analysis" < "decision" < "target_id" < "text": con un servidor que escribe las claves en orden
    # alfabético, una decisión llamada "action" se escribiría antes que su justificación.
    analysis: str = Field(max_length=400)
    decision: Literal["ADD", "UPDATE", "NOOP", "INVALIDATE"]
    target_id: str | None = None
    text: str | None = None


class MemoryRecord(BaseModel):
    memory_id: str
    kind: Kind
    subject: str  # un único valor VIGENTE por subject (salvo los episodios: ver ``MemoryStore._close_subject``)
    text: str
    created_at: datetime  # cuándo lo escribió el sistema (tiempo simulado)
    valid_from: datetime  # desde cuándo es cierto (el comienzo simulado de la sesión)
    valid_to: datetime | None = None  # se setea al invalidar; el registro se queda
    superseded_by: str | None = None  # el registro que lo cerró
    source_session: str
    importance: int = Field(default=5, ge=1, le=10)


# --- Prompts ---------------------------------------------------------------------------------------

EXTRACT_SYSTEM = """Sos el módulo de memoria de un asistente sobre legislación española de vivienda y alquiler. Vas a leer una conversación con un usuario, con su fecha, y tenés que anotar lo que conviene recordar para conversaciones futuras.

Anotá hechos cortos, uno por idea, de tres tipos:
- semantic: datos del usuario y de su situación (edad, dónde vive, qué alquila o qué tiene, montos, fechas de sus contratos, quién es la otra parte).
- procedural: cómo quiere que le respondas (por ejemplo, que le cites el número de la norma, que lo tutees, cómo llamarlo).
- episodic: qué preguntó en esta conversación y qué se le respondió, en una oración.

Reglas:
- Solo lo que el usuario dijo o lo que se le respondió en esta conversación. Nada de suposiciones: si algo no se dijo, no lo anotes.
- Lo que dijo el asistente sobre la ley no es un dato del usuario: si vale la pena, va como episodic ("se le explicó que...").
- Escribí las fechas completas (día, mes y año) cuando se dijeron. Si el usuario dice "en julio", anotá el año que corresponde a la fecha de la conversación.
- subject: una etiqueta corta en minúsculas con guiones bajos que nombre el dato (por ejemplo: edad, renta_habitacion, ciudad_arrendadora, preferencia_citas). El mismo dato lleva siempre la misma etiqueta: si ya existe una etiqueta para ese dato, usala.
- importance: de 1 (trivial) a 10 (clave para futuras preguntas sobre su vivienda).
- En analysis, primero y en una o dos oraciones: qué datos nuevos aparecieron."""

DECIDE_SYSTEM = """Sos el módulo de memoria de un asistente. Tenés un hecho nuevo sobre el usuario y los recuerdos ya guardados que más se le parecen. Decidí qué hacer:
- ADD: el hecho es nuevo; ningún recuerdo habla de lo mismo.
- UPDATE: un recuerdo habla de lo mismo con un valor que ya no es cierto (cambió la renta, la edad, el contrato...). Poné su id en target_id y en text el recuerdo actualizado. El viejo no se borra: queda como historia.
- NOOP: el hecho ya está guardado con el mismo contenido.
- INVALIDATE: el hecho dice que un recuerdo ya no es cierto, sin ser un valor nuevo de lo mismo (por ejemplo, "ya pagó la deuda" frente a "debe las cuotas"). Poné su id en target_id. El hecho nuevo se guarda igual, y el viejo queda como historia.

Reglas:
- target_id solo puede ser uno de los ids que aparecen abajo.
- text solo con ADD, UPDATE o INVALIDATE, como una oración que se entienda sola, con fechas completas.
- En analysis, primero y en una o dos oraciones: por qué."""

MEMORY_RULES = f"""Memoria:
- Entre <memoria> y </memoria> va lo que este usuario te contó en conversaciones anteriores, con su fecha. Para esta conversación, la memoria es una fuente igual que los pasajes: la regla de responder solo con los pasajes incluye también la memoria.
- Para los datos del usuario (su situación, fechas, montos, lo que te pidió) usá la memoria; para lo que dice la ley, los pasajes. Una respuesta puede necesitar las dos cosas.
- Si un dato cambió, vale el más reciente. Lo marcado "ya no vigente" es historia: no lo uses como si fuera actual.
- Si un dato del usuario no está ni en la memoria ni en la pregunta, no lo inventes: poné can_answer en false y en answer escribí exactamente: "{grounded.ABSTENTION}"
- Si el usuario te pidió una forma de responder (por ejemplo, citar el número de la norma o llamarlo de cierta manera), respetala."""


def probe_system(today) -> str:
    """El mismo prompt en todos los brazos: solo cambia lo que va en el espacio de memoria.

    La regla de memoria hace falta porque el prompt fundamentado de la Etapa 4 dice "respondé solo con lo
    que dicen los pasajes": sin ella, una sonda de extracción ("¿en qué localidad vive la dueña?") se
    respondería con una abstención aunque el dato esté en la memoria. ``today`` es el "hoy" simulado de la
    sonda (``asked_at``), nunca su ``as_of``.
    """
    return f"{grounded.system_prompt(today)}\n\n{MEMORY_RULES}"


def message_line(role: str, content: str) -> str:
    return f"{'Usuario' if role == 'user' else 'Asistente'}: {content}"


def session_header(session: Session) -> str:
    return f"Conversación del {session.started_at.date().isoformat()}:"


def transcript(session: Session) -> str:
    return "\n".join([session_header(session), *(message_line(m.role, m.content) for m in session.messages)])


def extraction_messages(session: Session, subjects: list[str] | None = None) -> list[dict]:
    """La sesión para extraer. ``subjects`` son las etiquetas vigentes del usuario: sin verlas, el modelo
    inventa una etiqueta nueva para el mismo dato ("edad" y "edad_usuaria"), y el valor viejo no se cierra."""
    content = transcript(session)
    if subjects:
        content = f"Etiquetas (subject) que ya existen para este usuario: {', '.join(subjects)}\n\n{content}"
    return [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": content}]


def record_line(record: MemoryRecord, now: datetime | None = None) -> str:
    """Un recuerdo como renglón. Si ya no rige en ``now``, lo dice: es historia, no un dato actual."""
    tags = [record.valid_from.date().isoformat(), record.kind, record.subject]
    if record.valid_to is not None and (now is None or record.valid_to <= now):
        tags.append(f"ya no vigente desde {record.valid_to.date().isoformat()}")
    return f"[{' · '.join(tags)}] {record.text}"


def decision_messages(fact: Fact, similar: list[MemoryRecord], at: datetime) -> list[dict]:
    shown = "\n".join(f"- id {r.memory_id}: {record_line(r, at)}" for r in similar)
    content = (f"Fecha de la conversación: {at.date().isoformat()}\n\n"
               f"Hecho nuevo [{fact.kind} · {fact.subject}]: {fact.text}\n\n"
               f"Recuerdos guardados que más se le parecen:\n{shown}")
    return [{"role": "system", "content": DECIDE_SYSTEM}, {"role": "user", "content": content}]


# --- El almacén ------------------------------------------------------------------------------------

class MemoryStore:
    """Los registros de un usuario (un escenario) y el log de cada operación que los cambió.

    La frontera temporal es ``<=`` en todos lados (``visible``, ``current``, ``replay``): lo escrito en una
    sesión tiene ``created_at`` igual al comienzo de la sesión, y el camino de escritura necesita ver lo
    que acaba de escribir en esa misma sesión. Ninguna sonda del eval set comparte el instante de una
    sesión, así que para las sondas ``<=`` y ``<`` dan lo mismo; con ``<=`` en los dos brazos, el almacén
    y la reproducción ven exactamente las mismas sesiones.
    """

    def __init__(self, scope: str):
        self.scope = scope
        self.records: list[MemoryRecord] = []
        self.ops: list[dict] = []
        self.vectors: dict[str, np.ndarray] = {}  # memory_id -> embedding normalizado

    def get(self, memory_id: str | None) -> MemoryRecord | None:
        return next((r for r in self.records if r.memory_id == memory_id), None)

    def visible(self, at: datetime) -> list[MemoryRecord]:
        """Lo que el sistema ya había escrito en ``at``, vigente o no."""
        return [r for r in self.records if r.created_at <= at]

    def current(self, at: datetime) -> list[MemoryRecord]:
        """Lo escrito hasta ``at`` que todavía rige en ``at``."""
        return [r for r in self.visible(at) if r.valid_to is None or r.valid_to > at]

    def subjects(self, at: datetime) -> list[str]:
        return sorted({r.subject for r in self.current(at)})

    def _add(self, fact: Fact, text: str, session_id: str, at: datetime) -> MemoryRecord:
        record = MemoryRecord(memory_id=f"m{len(self.records) + 1}", kind=fact.kind, subject=fact.subject,
                              text=text, created_at=at, valid_from=at, source_session=session_id,
                              importance=fact.importance)
        self.records.append(record)
        return record

    def _close(self, record: MemoryRecord, at: datetime, by: str) -> None:
        record.valid_to, record.superseded_by = at, by

    def _close_subject(self, new: MemoryRecord, at: datetime) -> list[str]:
        """Un único valor vigente por subject: el registro nuevo cierra a los otros vigentes con su subject.

        Los episodios quedan afuera: "qué preguntó y qué se le respondió" no es el valor actual de nada, y dos
        consultas sobre la fianza en sesiones distintas no se reemplazan una a la otra.
        """
        if new.kind == "episodic":
            return []
        same = [r for r in self.current(at) if r.memory_id != new.memory_id and r.subject == new.subject
                and r.kind != "episodic"]
        for r in same:
            self._close(r, at, new.memory_id)
        return [r.memory_id for r in same]

    def apply(self, decision: WriteDecision | None, fact: Fact, session_id: str, at: datetime,
              shown_ids: set[str]) -> dict:
        """Aplica una decisión y la registra en ``ops``. Devuelve la operación.

        - ADD guarda el hecho; UPDATE e INVALIDATE cierran el registro elegido y guardan el hecho nuevo (como
          Zep: el hecho que invalida también es un hecho); NOOP no cambia nada.
        - Todo registro nuevo cierra a los vigentes con su mismo subject (``_close_subject``). ``closed``
          lista todo lo que cerró la operación.

        Dos casos que no son lo que pidió el modelo, y quedan marcados en ``fallback``:
        - sin decisión (error de llamada): se agrega el hecho. Se prefiere un duplicado visible en la
          auditoría a perder un dato en silencio;
        - un ``target_id`` que no estaba entre los recuerdos mostrados o que ya no rige: se agrega el hecho
          sin cerrar nada elegido por el modelo (no se cierra un registro al azar).
        """
        requested = decision.decision if decision else "ADD"
        op = {"scope": self.scope, "session": session_id, "at": at.isoformat(), "fact": fact.model_dump(),
              "op": requested, "target_id": decision.target_id if decision else None, "new_id": None,
              "closed": [], "fallback": None if decision else "decision_error"}
        target = self.get(decision.target_id) if decision else None
        target_ok = (target is not None and target.memory_id in shown_ids
                     and (target.valid_to is None or target.valid_to > at))
        if requested in ("UPDATE", "INVALIDATE") and not target_ok:
            op["fallback"], op["op"] = "bad_target", "ADD"
        if op["op"] != "NOOP":
            text = (decision.text if decision and decision.text else fact.text).strip()
            new = self._add(fact, text, session_id, at)
            op["new_id"] = new.memory_id
            if op["op"] in ("UPDATE", "INVALIDATE"):
                self._close(target, at, new.memory_id)
                op["closed"].append(target.memory_id)
            op["closed"] += self._close_subject(new, at)
        self.ops.append(op)
        return op

    def save(self, folder: Path) -> None:
        write_jsonl(Path(folder) / f"{self.scope}.records.jsonl", self.records)
        write_jsonl(Path(folder) / f"{self.scope}.ops.jsonl", self.ops)


# --- Camino de escritura ---------------------------------------------------------------------------

Structured = Callable[[list[dict], type, int], tuple[BaseModel | None, dict]]
Encode = Callable[[list[str]], np.ndarray]


def llm_structured(messages: list[dict], schema: type, num_predict: int, *, model: str = grounded.MODEL,
                   num_ctx: int = grounded.NUM_CTX) -> tuple[BaseModel | None, dict]:
    """``llm.structured`` con ``num_ctx`` fijo; un ``ContextOverflow`` es un error de llamada más."""
    try:
        answer, r = llm.structured(messages, schema, model=model, num_ctx=num_ctx, num_predict=num_predict)
    except llm.ContextOverflow as error:
        return None, {"error": f"CONTEXT_OVERFLOW (client): {error}", "model": model}
    if r.expected_prompt_tokens is None:  # sin tokenizer, las guardas de contexto quedaron apagadas
        raise RuntimeError(f"llm.chat no pudo contar el prompt de {model}: las guardas de contexto están apagadas")
    return answer, r.log()


def fact_line(fact: Fact) -> str:
    return f"{fact.subject}: {fact.text}"


def _vectors(store: MemoryStore, records: list[MemoryRecord], encode: Encode) -> np.ndarray:
    missing = [r for r in records if r.memory_id not in store.vectors]
    if missing:
        for r, v in zip(missing, encode([f"{r.subject}: {r.text}" for r in missing])):
            store.vectors[r.memory_id] = np.asarray(v, dtype=np.float32)
    return np.stack([store.vectors[r.memory_id] for r in records])


def most_similar(store: MemoryStore, query: np.ndarray, records: list[MemoryRecord], k: int,
                 encode: Encode) -> list[MemoryRecord]:
    if not records:
        return []
    scores = _vectors(store, records, encode) @ query
    return [records[i] for i in np.argsort(-scores, kind="stable")[:k]]


def at_max_length(model: BaseModel) -> list[str]:
    """Los campos de texto que llegaron a su ``max_length``. La gramática de Ollama corta en el tope sin
    error: un campo que llegó al tope está cortado."""
    found = []
    for name, info in type(model).model_fields.items():
        limit = next((m.max_length for m in info.metadata if getattr(m, "max_length", None)), None)
        value = getattr(model, name)
        if limit and isinstance(value, str) and len(value) >= limit:
            found.append(name)
    return found


def write_session(store: MemoryStore, session: Session, *, structured: Structured, encode: Encode,
                  k_similar: int = SIMILAR_K) -> dict:
    """Extrae los hechos de una sesión y decide, uno por uno, qué hacer con cada uno.

    Si todavía no hay ningún recuerdo vigente, el hecho se agrega sin llamar al modelo: la única decisión
    posible es ADD. Los hechos de una misma sesión se procesan en orden, así que el segundo ya ve al
    primero. Devuelve el costo de la sesión, los campos cortados por su tope (``truncated``) y los
    ``analysis`` de cada llamada, que no se guardan en ningún lado: sirven para que ``leaked`` verifique
    que no se colaron en un registro.
    """
    at = session.started_at
    extraction, call = structured(extraction_messages(session, store.subjects(at)), Extraction, EXTRACT_PREDICT)
    calls, analyses, truncated = [call], [], Counter()
    ops: list[dict] = []
    if extraction is not None:
        analyses.append(extraction.analysis)
        truncated.update(f"extraction.{f}" for f in at_max_length(extraction))
        for fact in extraction.facts:
            truncated.update(f"fact.{f}" for f in at_max_length(fact))
            current = store.current(at)
            if not current:
                op = store.apply(WriteDecision(analysis="", decision="ADD"), fact, session.session_id, at, set())
                op["no_call"] = True  # el mismo dict que quedó en store.ops
                ops.append(op)
                continue
            query = np.asarray(encode([fact_line(fact)])[0], dtype=np.float32)
            similar = most_similar(store, query, current, k_similar, encode)
            decision, call = structured(decision_messages(fact, similar, at), WriteDecision, DECIDE_PREDICT)
            calls.append(call)
            if decision is not None:
                analyses.append(decision.analysis)
                truncated.update(f"decision.{f}" for f in at_max_length(decision))
            ops.append(store.apply(decision, fact, session.session_id, at, {r.memory_id for r in similar}))
    return {"scope": store.scope, "session": session.session_id, "at": at.isoformat(),
            "facts": len(extraction.facts) if extraction is not None else 0,
            "extraction_error": calls[0].get("error") if extraction is None else None,
            "ops": dict(Counter(op["op"] for op in ops)),
            "closed": sum(len(op["closed"]) for op in ops),
            "fallbacks": dict(Counter(op["fallback"] for op in ops if op["fallback"])),
            "truncated": dict(truncated), **call_cost(calls), "analyses": analyses}


def call_cost(calls: list[dict]) -> dict:
    return {"calls": len(calls), "errors": sum(bool(c.get("error")) for c in calls),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
            "output_tokens": sum(c.get("output_tokens") or 0 for c in calls)}


def write_error_for(probe: Probe, extraction_errors: Mapping[str, str]) -> str | None:
    """Si alguna sesión con la evidencia de la sonda no se pudo extraer, el almacén no tuvo la oportunidad
    de guardarla: la respuesta del brazo store no se puntúa (es un error de escritura, no de lectura)."""
    failed = [sid for sid in probe.evidence_sessions if sid in extraction_errors]
    return f"WRITE_ERROR: extracción fallida en {', '.join(failed)}" if failed else None


# --- Chequeos de lo guardado -----------------------------------------------------------------------

THINK = re.compile(r"</?think>", re.IGNORECASE)
HEDGES = re.compile(r"(?<!\w)(probablemente|posiblemente|quizas?|parece|supongo|seguramente|podria ser|"
                    r"al parecer)(?!\w)")
SPAN_WORDS = 8


def spans(text: str, words: int = SPAN_WORDS) -> set[tuple[str, ...]]:
    w = normalize(text).split()
    return {tuple(w[i:i + words]) for i in range(len(w) - words + 1)}


def leaked(records: list[MemoryRecord], analyses: Mapping[str, list[str]],
           transcripts: Mapping[str, str]) -> list[str]:
    """Los registros con un bloque ``<think>`` o con un tramo de 8 palabras que está en un ``analysis`` de su
    sesión y NO está en la transcripción de esa sesión. Un tramo que el análisis y el registro copian los dos
    de lo que dijo el usuario no es una filtración del razonamiento."""
    found = []
    for r in records:
        from_analysis = set().union(*(spans(a) for a in analyses.get(r.source_session, []) if a))
        if THINK.search(r.text) or (spans(r.text) & from_analysis) - spans(transcripts.get(r.source_session, "")):
            found.append(r.memory_id)
    return found


def hedged(records: list[MemoryRecord]) -> list[str]:
    """Registros escritos como hipótesis ("probablemente..."), por palabra completa ("aparece" no es
    "parece"): candidatos a "nunca se dijo". Es un diagnóstico para la auditoría humana, no un veredicto."""
    return [r.memory_id for r in records if HEDGES.search(normalize(r.text))]


# --- La auditoría humana ---------------------------------------------------------------------------

AUDIT_FIELDS = ["memory_id", "scenario", "source_session", "kind", "subject", "text", "valid_from", "valid_to",
                "human_verdict", "human_note"]


def audit_rows(stores: Mapping[str, MemoryStore]) -> list[dict]:
    return [{"memory_id": r.memory_id, "scenario": sid, "source_session": r.source_session, "kind": r.kind,
             "subject": r.subject, "text": r.text, "valid_from": r.valid_from.isoformat(),
             "valid_to": r.valid_to.isoformat() if r.valid_to else "", "human_verdict": "", "human_note": ""}
            for sid, st in stores.items() for r in st.records]


def read_audit(path: Path) -> list[dict]:
    # utf-8-sig: una planilla guardada desde Excel trae BOM, y sin esto la primera columna no se llama memory_id
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_audit(path: Path, rows: list[dict]) -> str:
    """Escribe la planilla de auditoría, salvo que ya tenga veredictos: esos no se pisan nunca.

    Si la planilla con veredictos es de otro almacén (cambiaron los registros), la nueva va al lado, en
    ``<nombre>.new.csv``, y se avisa: los veredictos viejos no valen para los registros nuevos.
    """
    path = Path(path)
    if path.is_file():
        old = read_audit(path)
        if any(row.get("human_verdict", "").strip() for row in old):
            same = [(r["scenario"], r["memory_id"], r["text"]) for r in old] == \
                   [(r["scenario"], r["memory_id"], r["text"]) for r in rows]
            if same:
                return "kept: ya tiene veredictos, para estos mismos registros"
            path = path.with_suffix(".new.csv")
            status = f"stale: la planilla con veredictos es de otro almacén; la nueva va en {path.name}"
        else:
            status = "rewritten: no tenía veredictos"
    else:
        status = "written"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return status


def audit_counts(path: Path) -> tuple[Counter | None, int, int]:
    """(conteo de veredictos si la planilla está completa o None, filas con veredicto válido, filas)."""
    rows = read_audit(path)
    verdicts = [row.get("human_verdict", "").strip() for row in rows]
    valid = [v for v in verdicts if v in VERDICTS]
    return (Counter(valid) if rows and len(valid) == len(rows) else None), len(valid), len(rows)


# --- Camino de lectura -----------------------------------------------------------------------------

def minmax(xs: list[float], eps: float = 1e-9) -> list[float]:
    """A [0, 1]. Solo un término EXACTAMENTE constante va a 0: uno casi constante igual se estira, por eso
    ``recall`` devuelve el rango crudo de cada término antes de normalizar."""
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < eps:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def hours_since(record: MemoryRecord, now: datetime) -> float:
    return max((now - record.created_at).total_seconds() / 3600, 0.0)


def recency(record: MemoryRecord, now: datetime) -> float:
    """El decaimiento de Generative Agents, 0,995 por hora simulada. Solo para mostrar su rango: el término
    que puntúa es su logaritmo (ver ``recall``)."""
    return RECENCY_DECAY ** hours_since(record, now)


def recall(store: MemoryStore, query: np.ndarray, now: datetime, *, encode: Encode,
           count: Callable[[str], int], k: int = MEMORY_K, budget: int = MEMORY_BUDGET) -> tuple[list[MemoryRecord], dict]:
    """Los recuerdos para una sonda hecha en ``now``: recencia + importancia + relevancia, cada término
    normalizado min-max y con peso 1 (Generative Agents), sobre lo escrito hasta ``now``.

    **La recencia va en escala logarítmica.** Con 0,995 por hora y meses entre sesiones, 0,995^h va de ~1
    (la última sesión) a ~1e-10 (la primera): el min-max lo vuelve un escalón, 1 para la última sesión y
    ≈0 para todas las demás. El término es log(0,995^h) = h·log(0,995), y su min-max es el de −h: conserva
    el orden y queda graduado (una sesión de hace dos meses no pesa lo mismo que una de hace ocho).

    Entran también los invalidados, marcados "ya no vigente": compiten igual que los demás. El espacio se
    llena en orden de puntaje hasta ``k`` recuerdos y ``budget`` tokens; el que no entra se saltea.
    Devuelve (elegidos, info) con el rango crudo de cada término (``hours`` y ``decay`` para la recencia) y
    los puntajes.

    La recencia se mide desde que se escribió, no desde la última vez que se recuperó (como en el paper):
    las sondas son preguntas independientes, y contar los accesos haría que el orden de las sondas cambie
    las respuestas.
    """
    records = store.visible(now)
    if not records:
        return [], {"ranges": {}, "scores": {}, "tokens": 0}
    hours = [hours_since(r, now) for r in records]
    terms = {"recency": [-h for h in hours],
             "importance": [float(r.importance) for r in records],
             "relevance": [float(x) for x in _vectors(store, records, encode) @ query]}
    ranges = {"hours": (min(hours), max(hours)),
              "decay": (RECENCY_DECAY ** max(hours), RECENCY_DECAY ** min(hours)),
              "importance": (min(terms["importance"]), max(terms["importance"])),
              "relevance": (min(terms["relevance"]), max(terms["relevance"]))}
    normed = [minmax(xs) for xs in terms.values()]
    scores = [sum(parts) for parts in zip(*normed)]
    order = sorted(range(len(records)), key=lambda i: (-scores[i], records[i].memory_id))
    chosen, used = [], 0
    for i in order:
        if len(chosen) == k:
            break
        n = count(record_line(records[i], now)) + LINE_MARGIN
        if used + n > budget:
            continue
        chosen.append(records[i])
        used += n
    return chosen, {"ranges": ranges, "scores": {records[i].memory_id: round(scores[i], 3) for i in order},
                    "tokens": used}


def memory_text(records: list[MemoryRecord], now: datetime) -> str:
    """El espacio de memoria, en orden cronológico (lo puntuado decide qué entra, no en qué orden se lee)."""
    if not records:
        return NO_MEMORY_TEXT
    ordered = sorted(records, key=lambda r: (r.valid_from, r.memory_id))
    return "\n".join(f"- {record_line(r, now)}" for r in ordered)


def replay(sessions: list[Session], before: datetime, budget: int,
           count: Callable[[str], int]) -> tuple[str, list[str], int, bool]:
    """La línea de base "reproducí todo lo que entre": los mensajes de las sesiones de hasta ``before``
    (``<=``, la misma frontera que el almacén), del más nuevo al más viejo, hasta ``budget`` tokens.
    Devuelve (texto, sesiones con algún mensaje adentro, tokens, ¿entró todo?).

    Acá cortar en el primero que no entra es la idea: lo que se pierde son los mensajes más viejos. Si el
    corte deja una respuesta sin su pregunta, la respuesta también sale.
    """
    past = sorted((s for s in sessions if s.started_at <= before), key=lambda s: s.started_at)
    kept: dict[str, list[str]] = {}
    used, complete = 0, True
    for session in reversed(past):
        header = session_header(session)
        n_header = count(header) + LINE_MARGIN
        lines = []
        for message in reversed(session.messages):
            line = message_line(message.role, message.content)
            n = count(line) + LINE_MARGIN + (0 if lines else n_header)
            if used + n > budget:
                break
            lines.insert(0, line)
            used += n
        if lines and lines[0].startswith("Asistente:"):
            used -= count(lines.pop(0)) + LINE_MARGIN + (0 if lines else n_header)
        if lines:
            kept[session.session_id] = [header, *lines]
        if len(lines) < len(session.messages):
            complete = False
            break  # no entró entera: las sesiones más viejas tampoco entran
    order = [s.session_id for s in past if s.session_id in kept]
    text = "\n\n".join("\n".join(kept[sid]) for sid in order)
    return (text or NO_MEMORY_TEXT), order, used, complete


def memory_slot(arm: str, scenario: Scenario, store: MemoryStore | None, probe: Probe, *, encode: Encode,
                count: Callable[[str], int], query: np.ndarray | None = None) -> dict:
    """Lo que va en el espacio de memoria de una sonda, según el brazo. Lo demás del prompt es igual.

    Devuelve el texto, las sesiones de las que viene lo que entró (para el recall de evidencia), los tokens
    de nuestra cuenta, si la reproducción entró entera y, en el almacén, los ids elegidos y el rango crudo de
    cada término del puntaje.
    """
    if arm == "no_memory":
        return {"text": NO_MEMORY_TEXT, "sessions": [], "tokens": 0, "memory_ids": [], "complete": None}
    if arm in REPLAY_BUDGETS:
        text, sessions, used, complete = replay(scenario.sessions, probe.asked_at, REPLAY_BUDGETS[arm], count)
        return {"text": text, "sessions": sessions, "tokens": used, "memory_ids": [], "complete": complete}
    if arm in STORE_ARMS:  # store_filter lee lo mismo: lo que cambia son sus pasajes
        if store is None or query is None:
            raise RuntimeError(f"arm={arm!r} pero no hay almacén o embedding de la consulta")
        chosen, info = recall(store, query, probe.asked_at, encode=encode, count=count)
        return {"text": memory_text(chosen, probe.asked_at), "sessions": sorted({r.source_session for r in chosen}),
                "tokens": info["tokens"], "memory_ids": [r.memory_id for r in chosen], "complete": None,
                "ranges": info["ranges"]}
    raise ValueError(f"brazo desconocido: {arm}")


# --- La memoria como filtro de la recuperación (paso 4) --------------------------------------------
# Los hechos del usuario que se corresponden con metadatos del corpus se vuelven el ``retrieval.Filters`` de
# la Etapa 3: la comunidad donde vive (``region`` del documento) y la fecha "a la cual" pregunta (la de su
# contrato). Es una regla fija, sin modelo: el filtro tiene que poder leerse y repetirse.

# El usuario nombra su ciudad, no su comunidad ("vivo en Zaragoza"), y ``retrieval.REGIONS`` solo conoce las
# comunidades. Un nomenclátor general: las capitales de provincia, algunas ciudades grandes y las islas, ya
# normalizado. No sale del eval set (Sóller, Inca o Tudela no están). Sin "la palma" (la isla): la ciudad
# de Palma la taparía.
CITY_REGIONS = {
    **dict.fromkeys(("almeria", "cadiz", "cordoba", "granada", "huelva", "jaen", "malaga", "sevilla", "jerez",
                     "marbella"), "Andalucía"),
    **dict.fromkeys(("huesca", "teruel", "zaragoza"), "Aragón"),
    **dict.fromkeys(("oviedo", "gijon"), "Principado de Asturias"),
    **dict.fromkeys(("palma", "palma de mallorca", "mallorca", "menorca", "ibiza", "eivissa", "formentera"),
                    "Illes Balears"),
    **dict.fromkeys(("las palmas", "gran canaria", "tenerife", "santa cruz de tenerife", "lanzarote",
                     "fuerteventura"), "Canarias"),
    **dict.fromkeys(("santander",), "Cantabria"),
    **dict.fromkeys(("avila", "burgos", "leon", "palencia", "salamanca", "segovia", "soria", "valladolid",
                     "zamora"), "Castilla y León"),
    **dict.fromkeys(("albacete", "ciudad real", "cuenca", "guadalajara", "toledo"), "Castilla-La Mancha"),
    **dict.fromkeys(("barcelona", "girona", "gerona", "lleida", "lerida", "tarragona"), "Cataluña"),
    **dict.fromkeys(("valencia", "alicante", "alacant", "castellon", "elche"), "Comunitat Valenciana"),
    **dict.fromkeys(("badajoz", "caceres", "merida"), "Extremadura"),
    **dict.fromkeys(("a coruna", "la coruna", "lugo", "ourense", "orense", "pontevedra", "vigo",
                     "santiago de compostela"), "Galicia"),
    **dict.fromkeys(("cartagena",), "Región de Murcia"),
    **dict.fromkeys(("pamplona", "iruna"), "Comunidad Foral de Navarra"),
    **dict.fromkeys(("bilbao", "vitoria", "san sebastian", "donostia"), "País Vasco"),
    **dict.fromkeys(("logrono",), "La Rioja"),
}
CITY_PATTERNS = {city: re.compile(r"(?<!\w)" + re.escape(city) + r"(?!\w)") for city in CITY_REGIONS}
CONTRACT_WORDS = re.compile(r"(?<!\w)(contrat\w*|alquil\w*|arrend\w*|firm\w*)")


def memory_regions(records: list[MemoryRecord]) -> tuple[str, ...]:
    """Las comunidades que nombran los recuerdos: por su nombre (``retrieval.parse_regions``) o por una ciudad
    del nomenclátor. En el orden de ``retrieval.REGIONS``, para que el resultado no dependa de los registros."""
    text = " ".join(r.text for r in records)
    found = set(retrieval.parse_regions(text))
    normalized = normalize(text)
    found |= {region for city, region in CITY_REGIONS.items() if CITY_PATTERNS[city].search(normalized)}
    return tuple(region for region in retrieval.REGIONS if region in found)


def memory_period(store: MemoryStore, records: list[MemoryRecord], query: np.ndarray,
                  encode: Encode) -> tuple[tuple[date, date] | None, str | None]:
    """El período del contrato por el que pregunta la sonda, y el id del recuerdo del que sale.

    Candidatos: los recuerdos que hablan de un contrato (contrato, alquiler, arrendamiento, firmó) y que
    ``retrieval.parse_period`` sabe fechar. Si hay varios (dos contratos, dos pisos), gana el más parecido a la
    pregunta: "el piso de Sóller" elige el contrato de Sóller, no el último. Empates por fecha y por id.
    """
    candidates = [r for r in records if CONTRACT_WORDS.search(normalize(r.text)) and retrieval.parse_period(r.text)]
    if not candidates:
        return None, None
    sims = _vectors(store, candidates, encode) @ query
    best = max(range(len(candidates)),
               key=lambda i: (round(float(sims[i]), 6), candidates[i].valid_from, candidates[i].memory_id))
    return retrieval.parse_period(candidates[best].text), candidates[best].memory_id


def memory_filters(store: MemoryStore, probe: Probe, query: np.ndarray,
                   encode: Encode) -> tuple[retrieval.Filters, dict]:
    """El ``retrieval.Filters`` de una sonda, armado con la pregunta y la memoria vigente en ``asked_at``.

    - **Lo que dice la pregunta manda**: si nombra una fecha o una comunidad, es por esa que pregunta ("¿y en
      Madrid?"). La memoria llena lo que la pregunta no dice.
    - Solo los recuerdos ``semantic`` vigentes: son los datos del usuario. Los episodios repiten lo que dijo
      el asistente sobre la ley, y un recuerdo cerrado ya no describe su situación.
    - Si ni la pregunta ni la memoria fechan nada, el período es el día de la sonda: es su "hoy" simulado, y
      con esa regla se etiquetó el gold (``Probe.reference_day``). La Etapa 3 usa la foto, que es el "hoy"
      de sus preguntas.
    Devuelve el filtro y de dónde salió cada parte.
    """
    facts = [r for r in store.current(probe.asked_at) if r.kind == "semantic"]
    q_period, q_regions = retrieval.parse_period(probe.question), retrieval.parse_regions(probe.question)
    m_period, m_period_id = memory_period(store, facts, query, encode)
    m_regions = memory_regions(facts)
    if q_period:
        period, period_source = q_period, "question"
    elif m_period:
        period, period_source = m_period, f"memory:{m_period_id}"
    else:
        day = probe.asked_at.date()
        period, period_source = (day, day + timedelta(days=1)), "asked_at"
    if q_regions:
        regions, regions_source = q_regions, "question"
    elif m_regions:
        regions, regions_source = m_regions, "memory"
    else:
        regions, regions_source = (), None
    info = {"period": [period[0].isoformat(), period[1].isoformat()], "period_source": period_source,
            "regions": list(regions), "regions_source": regions_source}
    return retrieval.Filters(period=period, regions=tuple(regions)), info


def evidence_recall(used_sessions: set[str], probe: Probe) -> tuple[int, int]:
    """(sesiones con evidencia que llegaron al prompt, sesiones con evidencia). Sin juez: las sesiones con
    la evidencia están etiquetadas en el eval set."""
    evidence = set(probe.evidence_sessions)
    return len(evidence & used_sessions), len(evidence)


# --- El régimen: ¿entra el historial? -------------------------------------------------------------

def history_tokens(scenario: Scenario, count: Callable[[str], int]) -> list[int]:
    """Tokens de cada sesión reproducida (encabezado y mensajes, con su margen de renglón)."""
    return [count(session_header(s)) + LINE_MARGIN
            + sum(count(message_line(m.role, m.content)) + LINE_MARGIN for m in s.messages)
            for s in sorted(scenario.sessions, key=lambda s: s.started_at)]


def where_replay_stops_fitting(scenarios: list[Scenario], count: Callable[[str], int],
                               limits: dict[str, int]) -> list[dict]:
    """Los escenarios reproducidos uno detrás de otro, como si fueran un solo usuario: tokens acumulados por
    sesión y, para cada límite, si todavía entra. Sirve para decir dónde deja de entrar la reproducción."""
    rows, total = [], 0
    for scenario in scenarios:
        for session, size in zip(sorted(scenario.sessions, key=lambda s: s.started_at),
                                 history_tokens(scenario, count)):
            total += size
            rows.append({"session": session.session_id, "tokens": size, "accumulated": total,
                         **{name: total <= limit for name, limit in limits.items()}})
    return rows


def first_overflow(rows: list[dict], limit_name: str) -> str:
    """'entra entero (N tokens)' o 'deja de entrar en <sesión> (N tokens acumulados)'."""
    over = next((r for r in rows if not r[limit_name]), None)
    if over is None:
        return f"entra entero ({rows[-1]['accumulated'] if rows else 0} tokens)"
    return f"deja de entrar en {over['session']} ({over['accumulated']} tokens acumulados)"


def distinct_resamples(groups: int) -> int:
    """Remuestras distintas (multiconjuntos) de un bootstrap sobre ``groups`` grupos: con 3 escenarios, 10."""
    return math.comb(2 * groups - 1, groups) if groups else 0
