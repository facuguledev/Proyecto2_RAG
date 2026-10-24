"""Respuestas fundamentadas (Etapa 4): el pasaje, la entrega con presupuesto, la llamada al generador y el
puntaje sin juez. Lo reusan las Etapas 5 y 6, así todas arman la ventana y puntúan con la misma vara.

Los pasajes son las unidades de la Etapa 3 (``chunking.Unit``): su cabecera fija (norma, sección, vigencia)
más su texto, como los midió la Etapa 3, con el id citable del bloque adelante. La entrega es la de la
Etapa 3 (``retrieval.deliver``: en orden de ranking, lo que no entra se saltea y se sigue, una copia de un
texto ya entregado no se repite), con cada pasaje contado tal como lo ve el generador, id incluido.

Las tres decisiones que vienen del chequeo a libro cerrado de la Etapa 0 (decision_log.md, 2026-10-16):
- la instrucción de sí/no es global, pero el veredicto se lee solo en los ítems con ``expected_verdict``,
  de la primera palabra del texto crudo (sin sacar tildes: "sí" no es "si"), y solo si ``can_answer``;
- el prompt dice qué escribir en ``answer`` al abstenerse (si no, el campo se degenera en "No");
- la clave se busca por palabra completa (``evalset.contains_key``) y solo en ``answer``, nunca en
  ``analysis``.
"""
import re
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from datetime import date

from pydantic import BaseModel, Field

import evaluation
import llm
import retrieval
from chunking import Unit
from evalset import Answer, block_of, contains_key, normalize
from store import sha256_json

MODEL = "qwen3:8b"
NUM_CTX = 8192
NUM_PREDICT = 768
PROMPT_LIMIT = NUM_CTX - NUM_PREDICT  # 7.424: lo que puede ocupar el prompt, contado con la plantilla

ABSTENTION = "No puedo responderlo con los pasajes disponibles."
# La gramática de Ollama impone estos topes cortando el texto, sin error (decision_log, 2026-10-23): una
# respuesta que llega al tope está cortada y se trata como un error de llamada, igual que un num_predict.
ANALYSIS_MAX, ANSWER_MAX = 600, 1200


class GroundedAnswer(BaseModel):
    # Declarados en el orden en que el modelo los escribe, obligatorios, y con nombres que también quedan
    # en ese orden alfabéticamente (analysis < answer < can_answer < citations): razona antes de responder
    # en cualquier servidor. Ollama no le muestra el schema al modelo: las instrucciones van en el prompt.
    analysis: str = Field(max_length=ANALYSIS_MAX)
    answer: str = Field(max_length=ANSWER_MAX)
    can_answer: bool
    citations: list[str]


RULES_GROUNDED = f"""Vas a recibir pasajes de normas del BOE entre <contexto> y </contexto>. Cada pasaje empieza con una línea entre corchetes: su id, la norma, la sección y desde cuándo hasta cuándo rige ese texto.

Reglas:
- Respondé solo con lo que dicen los pasajes, no con lo que sepas de antes.
- Si la pregunta menciona una fecha, usá el texto que regía en esa fecha. Si no, el que rige hoy.
- Si los pasajes no alcanzan para responder, poné can_answer en false y en answer escribí exactamente: "{ABSTENTION}"
- Si la pregunta se responde con sí o no, empezá answer con "Sí" o con "No".
- En citations poné los ids (lo que va antes del primer "·" en la línea entre corchetes) de los pasajes en los que se apoya la respuesta. Solo ids de pasajes que recibiste.
- En analysis, primero y en una o dos oraciones: qué pasaje responde y por qué. Después answer, en pocas oraciones."""

RULES_CLOSED_BOOK = f"""No vas a recibir pasajes: respondé con lo que sepas.

Reglas:
- Si no sabés la respuesta con seguridad, poné can_answer en false y en answer escribí exactamente: "{ABSTENTION}"
- Si la pregunta se responde con sí o no, empezá answer con "Sí" o con "No".
- Dejá citations vacía.
- En analysis, primero y en una o dos oraciones: de dónde sale la respuesta. Después answer, en pocas oraciones."""


def system_head(today: date) -> str:
    """El encabezado común a todos los prompts de sistema de las Etapas 4-6.

    ``today`` es la fecha de la foto (o el "hoy" simulado de una sonda de la Etapa 6), nunca el ``as_of``
    de una pregunta: esa es una etiqueta del eval set, y dársela al modelo sería darle la respuesta.
    """
    return (f"Sos un asistente que responde preguntas sobre la legislación española de vivienda y alquiler. "
            f"Hoy es {today.isoformat()}. Respondé en español.")


def system_prompt(today: date, closed_book: bool = False) -> str:
    """El mismo encabezado para todos los brazos; solo cambian las reglas de libro cerrado."""
    return f"{system_head(today)}\n\n{RULES_CLOSED_BOOK if closed_book else RULES_GROUNDED}"


# --- Pasajes y entrega -----------------------------------------------------------------------------

def passage_block(unit: Unit) -> str:
    """La cabecera de la Etapa 3 con el id citable adelante: ``[doc_id#bloque · norma · sección · vigencia]``.

    El id es el del bloque, no el de la versión: un modelo no copia bien "@20130606", y la versión ya la dice
    la vigencia. Por eso las citas se verifican a nivel de bloque.
    """
    if not unit.header.startswith("["):
        raise ValueError(f"{unit.unit_id}: la cabecera no empieza con '[': {unit.header[:40]!r}")
    return f"[{block_of(unit.section_id)} · {unit.header[1:]}\n{unit.text}"


def token_counter(model: str = MODEL) -> Callable[[str], int]:
    def count(text: str) -> int:
        n = llm.count_tokens(text, model)
        if n is None:
            raise RuntimeError(f"no hay tokenizer para {model} en llm.HF_TOKENIZERS: no se puede presupuestar")
        return n
    return count


_COUNTS: dict[tuple[str, str], int] = {}


def as_passages(units: list[Unit], count: Callable[[str], int]) -> list[Unit]:
    """Las unidades con ``n_gen`` = los tokens del pasaje tal como lo ve el generador, id incluido.

    La Etapa 3 cuenta la cabecera y el texto; el id citable suma unos 13 tokens por pasaje. Contarlo acá
    hace exacto el presupuesto, y por lo tanto igual entre brazos que entregan cantidades distintas.
    """
    out = []
    for unit in units:
        key = (unit.unit_id, unit.text)
        if key not in _COUNTS:
            _COUNTS[key] = count(passage_block(unit))
        out.append(replace(unit, n_gen=_COUNTS[key]))
    return out


def deliver(units: list[Unit], budget: int, count: Callable[[str], int]) -> list[Unit]:
    """La entrega de la Etapa 3 (``retrieval.deliver``) sobre los pasajes contados con su id."""
    return retrieval.deliver(as_passages(units, count), budget)


def used_tokens(delivered: list[Unit] | None) -> int:
    return sum(unit.n_gen for unit in delivered or [])


def context_block(delivered: list[Unit]) -> str:
    body = "\n\n".join(passage_block(u) for u in delivered) if delivered else "(no se encontraron pasajes)"
    return f"<contexto>\n{body}\n</contexto>"


def build_messages(system: str, question: str, delivered: list[Unit] | None,
                   history: list[dict] | None = None, memory: str | None = None) -> list[dict]:
    """Sistema + historial + un mensaje de usuario con la memoria, los pasajes y la pregunta.

    ``delivered=None`` es libro cerrado: sin bloque de contexto. ``history`` son mensajes ya recortados
    a su presupuesto (Etapa 5); ``memory``, el texto del espacio de memoria (Etapa 6).
    """
    parts = []
    if memory:
        parts.append(f"<memoria>\n{memory}\n</memoria>")
    if delivered is not None:
        parts.append(context_block(delivered))
    parts.append(f"Pregunta: {question}")
    return [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": "\n\n".join(parts)}]


_SAMPLE = Unit(unit_id="D#a1@20000101~c1", section_id="D#a1@20000101", doc_id="D", parent_id="D#a1@20000101~p1",
               start=0, end=5, text="texto", prefix="", header="[Norma · Artículo 1. · vigente desde enero de 2000]")


def prompt_spec(system: str) -> dict:
    """Lo que determina una respuesta además de la pregunta y los pasajes: va en el ``run_spec`` de cada
    etapa. Un cambio en el schema o en la plantilla del mensaje (id, contexto, memoria) invalida las filas
    viejas en lugar de mezclarlas con las nuevas."""
    sample = build_messages(system, "¿?", [_SAMPLE], history=[{"role": "user", "content": "h"}], memory="m")
    return {"schema": sha256_json(GroundedAnswer.model_json_schema())[:16], "template": sha256_json(sample)[:16]}


# --- La llamada ----------------------------------------------------------------------------------

def generate(messages: list[dict], *, model: str = MODEL, num_ctx: int = NUM_CTX, num_predict: int = NUM_PREDICT,
             strict_ctx: bool = True, cache: bool = True) -> tuple[GroundedAnswer | None, dict]:
    """Una respuesta estructurada. Devuelve (respuesta o None, registro de la llamada).

    Un ``ContextOverflow`` (el prompt no entra, detectado antes de mandar) no llega como ``r.error`` sino
    como excepción: se atrapa y queda en la misma categoría que los demás errores de llamada.

    Si ``llm.chat`` no pudo contar el prompt (falta el tokenizer), sus dos guardas quedaron apagadas y solo
    avisó: acá eso es una excepción, porque un truncamiento ya no se detectaría.
    """
    try:
        answer, r = llm.structured(messages, GroundedAnswer, model=model, num_ctx=num_ctx,
                                   num_predict=num_predict, strict_ctx=strict_ctx, cache=cache)
    except llm.ContextOverflow as error:
        return None, {"error": f"CONTEXT_OVERFLOW (client): {error}", "model": model}
    if r.expected_prompt_tokens is None:
        raise RuntimeError(f"llm.chat no pudo contar el prompt de {model}: las guardas de contexto están apagadas")
    return answer, r.log()


# --- Puntaje sin juez ----------------------------------------------------------------------------

def verdict_of(answer: str) -> str:
    """'yes', 'no' o 'unclear', de la primera palabra del texto crudo. Sin sacar tildes: "Si el contrato..."
    es un condicional, no un sí. En NFC, para que una tilde combinante no parta "Sí" en dos, y "No obstante"
    es un conector, no un veredicto."""
    words = re.findall(r"\w+", unicodedata.normalize("NFC", answer).lower())
    if not words or words[:2] == ["no", "obstante"]:
        return "unclear"
    return {"sí": "yes", "no": "no"}.get(words[0], "unclear")


CITATION = re.compile(r"[A-Z]+-[A-Za-z]+-\d{4}-\d+#[\w.-]+")


def cited_blocks(citations: list[str]) -> list[str]:
    """Los bloques citados, sin repetir. Acepta el id con corchetes, con versión ("...#a9@20190306") o con un
    punto final de oración."""
    found = []
    for citation in citations:
        match = CITATION.search(citation)
        block = block_of(match.group().rstrip(".")) if match else citation.strip()
        if block and block not in found:
            found.append(block)
    return found


def is_abstention_text(answer: str) -> bool:
    return normalize(answer).startswith(normalize(ABSTENTION)[:25])  # "no puedo responderlo con lo"


class Gold:
    """El gold de una pregunta como lo mide la Etapa 3: las secciones aceptadas (el gold y sus versiones con
    texto idéntico, ``evaluation.text_equivalents``) y los textos de sus unidades (``evaluation.gold_texts``).
    Un pasaje con el texto de una unidad gold cuenta como del gold: el generador recibió ese texto."""

    def __init__(self, accepted: set[str], texts: set[str]):
        self.accepted, self.texts = accepted, texts

    @classmethod
    def of(cls, target: Answer, sections, units: dict[str, Unit]) -> "Gold":
        if not target.answerable:
            return cls(set(), set())
        accepted = evaluation.text_equivalents(target.gold_sections, sections)
        return cls(accepted, evaluation.gold_texts(accepted, units))


def delivery(target: Answer, delivered: list[Unit], gold: Gold) -> dict:
    """Qué le llegó al generador, contra el gold, con las definiciones de la Etapa 3. ``None`` en las
    preguntas sin respuesta.

    - ``doc_hit``: algún pasaje del documento gold (cualquier artículo de la norma);
    - ``section_delivered``: algún pasaje del gold;
    - ``answer_delivered``: la clave está en algún pasaje entregado, sea o no gold;
    - ``gold_key_delivered``: un pasaje del gold que además contiene la clave. Es la primaria de la Etapa 3
      y el "recuperó bien" de la tabla 2×2: un acierto de documento no es una respuesta entregada.
    """
    keys = ("doc_hit", "section_delivered", "answer_delivered", "gold_key_delivered", "gold_key_delivered_strict")
    if not target.answerable:
        return dict.fromkeys(keys)
    return {"doc_hit": any(u.doc_id in set(target.gold_docs) for u in delivered),
            **evaluation.delivered_metrics(delivered, target, gold.accepted, gold.texts)}


def score(answer: GroundedAnswer | None, call: dict, target: Answer, delivered: list[Unit] | None,
          gold: Gold) -> dict:
    """El puntaje de una respuesta. ``outcome`` es 'error', 'abstained' o 'answered'.

    Un error de llamada es su propia categoría: ``correct`` queda en None, nunca en False. Una respuesta que
    llegó al tope de ``answer`` está cortada (la gramática corta sin avisar), y también es un error.
    - Con respuesta: acierta si respondió (``can_answer``), contiene una clave y, si es de sí/no, el
      veredicto coincide.
    - Sin respuesta (fuera del corpus, no respondible): acierta si y solo si ``can_answer`` es false.

    ``cited_gold``: bloques citados de los que llegó un pasaje del gold. Es a nivel de versión: citar "#a9"
    cuando llegó la versión de 2026 y el gold es la de 2013 no es citar el gold.
    """
    if answer is None or call.get("error"):
        return {"outcome": "error", "correct": None, "error": call.get("error")}
    if len(answer.answer) >= ANSWER_MAX:
        return {"outcome": "error", "correct": None, "error": f"ANSWER_AT_MAXLENGTH: {ANSWER_MAX} caracteres"}
    key_hit = target.answerable and contains_key(answer.answer, target.answer_key)
    verdict = verdict_of(answer.answer) if target.expected_verdict else None
    verdict_ok = (answer.can_answer and verdict == target.expected_verdict) if target.expected_verdict else None
    if target.answerable:
        correct = answer.can_answer and key_hit and verdict_ok is not False
    else:
        correct = not answer.can_answer
    blocks = cited_blocks(answer.citations)
    delivered_blocks = {block_of(u.section_id) for u in delivered or []}
    gold_blocks = ({block_of(u.section_id) for u in evaluation.from_gold(delivered, gold.accepted, gold.texts)}
                   if delivered is not None else {block_of(s) for s in target.gold_sections})
    abstention_text = is_abstention_text(answer.answer)
    return {
        "outcome": "answered" if answer.can_answer else "abstained",
        "correct": correct,
        "key_hit": key_hit,
        "verdict": verdict,
        "verdict_ok": verdict_ok,
        "cited": blocks,
        "cited_not_delivered": [b for b in blocks if b not in delivered_blocks],
        "cited_gold": [b for b in blocks if b in gold_blocks],
        "analysis_at_maxlength": len(answer.analysis) >= ANALYSIS_MAX,  # el análisis cortado: la respuesta no
        # Diagnósticos de consistencia (no puntúan: el puntaje sale del booleano). El texto de abstención
        # con can_answer=true, o can_answer=false con una respuesta que no es la de abstención.
        "flag_bool_says_answer_text_abstains": answer.can_answer and abstention_text,
        "flag_bool_says_abstain_text_answers": not answer.can_answer and not abstention_text,
    }
