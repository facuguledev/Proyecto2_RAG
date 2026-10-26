"""La conversación (Etapa 5): con qué buscar (Parte A) y qué historial conservar (Parte B).

La API no tiene estado: lo que el modelo "recuerda" dentro de una sesión es lo que este código vuelve a
poner en la ventana en cada turno. Pasajes e historial compiten por el mismo presupuesto, y la ventana la
arma el ensamblador, nunca el servidor: pasado ``num_ctx``, Ollama descarta los mensajes más viejos sin
avisar (assignment, Etapa 5, primera trampa).

Reglas de toda la etapa:
- cada guion se reproduce **solo**: el historial de un turno es lo guionado antes en su propio guion (el
  turno 1 no tiene historial). Así lo que se recorta es información del propio guion, y no se mezclan
  datos de otras personas;
- el historial es siempre el **guionado** (la pregunta del usuario y la respuesta del asistente de la
  Etapa 2). Todos los brazos ven el mismo historial y solo se genera el turno actual: si cada brazo se
  realimentara con sus propias respuestas, divergirían desde el turno 2;
- en el historial nunca entra un ``analysis``, y el resumen tampoco tiene campo de razonamiento: lo que se
  escribe ahí vuelve a la ventana como si fuera la conversación;
- el total (pasajes + historial) es fijo, el historial tiene un **tope** y los pasajes reciben lo que el
  historial no usó. En cada turno los brazos ven el mismo total, y lo único que cambia es cuánto historial
  se conserva;
- el recuperador es el de la Etapa 3 (``index_loader``), con su mejor configuración. Como el historial es
  el guionado, ninguna consulta depende de lo que genera el modelo: todo se recupera antes de generar.
"""
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

import chunking
import grounded
import llm
from evalset import Answer, Dialogue, Turn, contains_key

TOTAL_BUDGET = 1500  # pasajes + historial, en tokens del generador
# Los topes del experimento 1. El historial más largo dentro de un guion de test son ~330 tokens y un par
# usuario-asistente mide 64 en promedio: con 100, 150 y 250 la ventana pierde el turno 1 en ~26, ~19 y ~7 de
# los 34 turnos no-primeros. Un tope de 300 o más casi no recorta nada del propio guion.
HISTORY_CAPS = (100, 150, 250)
COMPRESSION_CAP = 150  # el tope del experimento 2 (ventana contra resumen)
# La entrega con la que la Parte A mide "le llegó la respuesta": los pasajes con el tope del experimento 2
# lleno. Fija para las cuatro consultas: una variable.
PART_A_BUDGET = TOTAL_BUDGET - COMPRESSION_CAP
SUMMARY_TRIGGER = 0.7  # la línea de advertencia de MemGPT (§7.3), sobre el tope del historial
# La cuenta por mensaje es de texto pelado: no ve "<|im_start|>rol\n" ni "<|im_end|>\n" de la plantilla.
MESSAGE_MARGIN = 5
REWRITE_HISTORY_TURNS = 5  # el guion entero hasta el turno actual: los turnos "return" vuelven al turno 1
REWRITE_MAX = 300  # caracteres; la gramática de Ollama corta en el tope sin error
REWRITE_NUM_PREDICT = 256
# Con el tope en 150 tokens, la mitad para el resumen y la mitad para el último par usuario-asistente (un
# intercambio de test mide 64 tokens en promedio). 220 caracteres son ~64 tokens del generador (3,45
# caracteres por token en los guiones), ~77 con su envoltorio y el margen.
SUMMARY_MAX = 220
SUMMARY_NUM_PREDICT = 256
# El resumen v2 (Etapa 8). La auditoría de la Etapa 5 encontró que el de arriba se congela: al llegar a los 220
# caracteres, el modelo vuelve a copiar el resumen anterior y la gramática lo corta antes de que entre lo nuevo
# (30 de 58 actualizaciones). La v2 cambia solo el resumidor, con la misma estrategia y el mismo tope de
# historial: la gramática deja de ser el límite (600 caracteres, que el modelo no alcanza escribiendo 35
# palabras) y el prompt pide reescribir comprimiendo, sin copiar. Si el resumen sale largo, ocupa más del tope
# y los mensajes recientes se recortan (``SummaryBuffer.view``): el costo queda adentro del mismo tope.
SUMMARY_V2_GRAMMAR_MAX = 600
SUMMARY_V2_TARGET = SUMMARY_MAX  # el largo que se le pide (35 palabras); pasarlo se cuenta, no se corta
QUERY_TYPES = ("last_turn", "full_conversation", "rewrite", "oracle")


# --- La conversación reproducida -----------------------------------------------------------------

@dataclass(frozen=True)
class ReplayTurn:
    turn_id: str  # "dl01.t2", el mismo nombre que evalset.corpus_questions
    dialogue_id: str
    number: int  # desde 1, dentro de su guion
    turn: Turn

    @property
    def exchange(self) -> list[dict]:
        """Lo que entra al historial: la pregunta y la respuesta GUIONADA, nunca la generada."""
        return [{"role": "user", "content": self.turn.user}, {"role": "assistant", "content": self.turn.assistant}]


def replay(dialogue: Dialogue) -> list[ReplayTurn]:
    return [ReplayTurn(f"{dialogue.dialogue_id}.t{n}", dialogue.dialogue_id, n, turn)
            for n, turn in enumerate(dialogue.turns, start=1)]


def scripted_history(turns: list[ReplayTurn]) -> list[dict]:
    return [message for rt in turns for message in rt.exchange]


# --- Parte A · con qué buscar --------------------------------------------------------------------

Offsets = Callable[[str], list[tuple[int, int]]]  # texto -> (inicio, fin) de cada token, sin especiales


def hf_offsets(tokenizer) -> Offsets:
    """Los offsets de un tokenizer de HF (el del reranker o el del embedder), sin tokens especiales."""
    return lambda text: tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]


def special_tokens(tokenizer, pair: bool) -> int:
    """Cuántos tokens especiales agrega el tokenizer a un texto o a un par. Se mide: en XLM-RoBERTa son 2
    sueltos y 4 en un par (``<s> a </s></s> b </s>``)."""
    bare = len(tokenizer("a", add_special_tokens=False)["input_ids"])
    if pair:
        bare += len(tokenizer("b", add_special_tokens=False)["input_ids"])
        return len(tokenizer("a", "b")["input_ids"]) - bare
    return len(tokenizer("a")["input_ids"]) - bare


def query_limit(rerank: bool, reranker_tokenizer=None, embedder_tokenizer=None, reranker_max_length: int = 512,
                embedder_max_seq_length: int = 512) -> int:
    """El largo máximo de una consulta, en tokens sin especiales, para que nada la trunque en silencio.

    Con reranker manda su ventana: la Etapa 3 dimensionó los hijos en ``chunking.CHILD_MAX`` contando con
    consultas cortas, y el par (consulta + hijo) tiene que entrar en ``reranker_max_length``. Sin reranker,
    manda el ``max_seq_length`` del embedder.
    """
    if rerank:
        return reranker_max_length - chunking.CHILD_MAX - special_tokens(reranker_tokenizer, pair=True)
    return embedder_max_seq_length - special_tokens(embedder_tokenizer, pair=False)


def keep_last_tokens(text: str, limit: int, offsets: Offsets) -> tuple[str, bool]:
    """Los últimos ``limit`` tokens de ``text``. Devuelve (texto, si se recortó).

    Se conserva el final: en un turno del usuario la pregunta suele ir al final, después del contexto
    ("Firmé en 2016... ¿Cuánto me podían pedir?")."""
    spans = offsets(text)
    if len(spans) <= limit:
        return text, False
    return text[spans[-limit][0]:], True


def full_conversation_query(history: list[dict], question: str, count: Callable[[str], int],
                            max_tokens: int) -> tuple[str, int]:
    """La pregunta actual y las anteriores del usuario, **la más nueva primero**, hasta ``max_tokens``
    (tokens sin especiales: ``query_limit``). Devuelve (consulta, preguntas viejas que no entraron).

    Los modelos truncan por la derecha: en orden cronológico se perdería justo la pregunta actual. Así, lo
    que se pierde son las más viejas. Solo turnos del usuario (una elección: las respuestas del asistente
    repetirían los términos de la pregunta anterior).
    """
    pieces = [question] + [m["content"] for m in reversed(history) if m["role"] == "user"]
    kept = [pieces[0]]  # la actual siempre: es lo único que dice qué se pregunta ahora
    for piece in pieces[1:]:
        if count(" \n".join(kept + [piece])) > max_tokens:
            break  # las que siguen son más viejas: cortar acá es justamente la idea
        kept.append(piece)
    return " \n".join(kept), len(pieces) - len(kept)


def bound_query(query_type: str, query: str, history: list[dict], limit: int, offsets: Offsets) -> tuple[str, bool]:
    """Una consulta acotada a ``limit`` tokens. Devuelve (consulta, si se recortó algo).

    ``full_conversation`` se arma desde ``history`` y pierde primero los turnos más viejos; las otras tres se
    recortan conservando el final. Lo mismo lee cada canal (denso, BM25, reranker): una consulta, un texto.
    """
    if query_type == "full_conversation":
        joined, dropped = full_conversation_query(history, query, lambda t: len(offsets(t)), limit)
        joined, cut = keep_last_tokens(joined, limit, offsets)  # la pregunta actual sola ya no entraba
        return joined, dropped > 0 or cut
    return keep_last_tokens(query, limit, offsets)


class Rewrite(BaseModel):
    query: str = Field(max_length=REWRITE_MAX)  # un solo campo: nada con qué razonar, nada que responder


REWRITE_SYSTEM = """Reescribís la última pregunta de un usuario para que se entienda sola, sin la conversación. La vas a usar para buscar en legislación española de vivienda y alquiler.

Reglas:
- Una sola pregunta, en español, con los datos de la conversación que hagan falta para entenderla (de qué se habla, fechas, tipo de contrato, lugar).
- No la respondas, no agregues datos que no estén en la conversación y no expliques nada."""


def rewrite_messages(history: list[dict], question: str, n_turns: int = REWRITE_HISTORY_TURNS) -> list[dict]:
    """El prompt del reescritor: los últimos ``n_turns`` pares guionados y la pregunta actual."""
    recent = history[-2 * n_turns:]
    lines = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}" for m in recent)
    return [{"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"<conversación>\n{lines}\n</conversación>\n\nÚltima pregunta: {question}"}]


def rewrite(history: list[dict], question: str, *, model: str = grounded.MODEL,
            num_ctx: int = grounded.NUM_CTX) -> tuple[str | None, dict | None]:
    """(consulta reescrita o None si falló, registro de la llamada o None si no hubo).

    Sin historial no hay nada que resolver: el primer turno de cada guion va tal cual, sin llamada. Una
    reescritura vacía es un error del reescritor; una que llega a ``REWRITE_MAX`` está cortada (la gramática
    corta sin avisar) y se cuenta como sospechosa, pero se usa.
    """
    if not history:
        return question, None
    try:
        out, r = llm.structured(rewrite_messages(history, question), Rewrite, model=model, num_ctx=num_ctx,
                                num_predict=REWRITE_NUM_PREDICT)
    except llm.ContextOverflow as error:
        return None, {"error": f"CONTEXT_OVERFLOW (client): {error}", "model": model}
    call = r.log()
    if out is not None and not out.query.strip():
        return None, call | {"error": "REWRITE_EMPTY: la reescritura vino vacía"}
    return (out.query.strip() if out else None), call


# --- Parte B · qué conservar ---------------------------------------------------------------------

def passage_budget(history_used: int, total: int = TOTAL_BUDGET) -> int:
    """Los pasajes reciben lo que el historial no usó: el total es el mismo en todos los brazos."""
    return max(0, total - history_used)


def message_tokens(message: dict, count: Callable[[str], int]) -> int:
    return count(message["content"]) + MESSAGE_MARGIN


def history_tokens(history: list[dict], count: Callable[[str], int]) -> int:
    return sum(message_tokens(m, count) for m in history)


def window(history: list[dict], budget: int, count: Callable[[str], int]) -> tuple[list[dict], int]:
    """Los mensajes más nuevos que entran en ``budget``. Devuelve (conservados, tokens).

    A diferencia del ensamblador de pasajes, acá sí se corta en el primero que no entra: descartar los
    mensajes MÁS VIEJOS es justamente la idea. Y no se arranca con una respuesta sin su pregunta.
    """
    kept, used = [], 0
    for message in reversed(history):
        n = message_tokens(message, count)
        if used + n > budget:
            break
        kept.insert(0, message)
        used += n
    if kept and kept[0]["role"] == "assistant":
        used -= message_tokens(kept.pop(0), count)
    return kept, used


class Summary(BaseModel):
    # Sin campo de razonamiento: este texto vuelve a la ventana como si fuera la conversación, y un análisis
    # guardado acá serían hipótesis contadas como hechos.
    summary: str = Field(max_length=SUMMARY_MAX)


SUMMARY_SYSTEM = """Mantenés el resumen de una conversación entre un usuario y un asistente sobre legislación española de vivienda y alquiler. Recibís el resumen anterior (puede estar vacío) y los mensajes que salen de la ventana. Escribí el resumen nuevo, que reemplaza al anterior.

Reglas:
- Conservá primero los datos que dio el usuario (fechas, lugar, tipo de contrato, montos, quién es el propietario) y después lo que se le respondió, con sus números y plazos.
- No agregues nada que no esté en el resumen anterior o en los mensajes: ni suposiciones ni opiniones.
- A lo sumo 35 palabras, en tercera persona ("el usuario...")."""


def summary_messages(previous: str, evicted: list[dict], system: str = SUMMARY_SYSTEM) -> list[dict]:
    lines = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}" for m in evicted)
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"<resumen_anterior>\n{previous or '(vacío)'}\n</resumen_anterior>\n\n"
                                        f"<mensajes>\n{lines}\n</mensajes>"}]


Summarizer = Callable[[str, list[dict]], tuple[str | None, dict]]


def summarize(previous: str, evicted: list[dict], *, model: str = grounded.MODEL,
              num_ctx: int = grounded.NUM_CTX) -> tuple[str | None, dict]:
    """(resumen nuevo o None si la llamada falló, registro de la llamada). Uno vacío es un error."""
    try:
        out, r = llm.structured(summary_messages(previous, evicted), Summary, model=model, num_ctx=num_ctx,
                                num_predict=SUMMARY_NUM_PREDICT)
    except llm.ContextOverflow as error:
        return None, {"error": f"CONTEXT_OVERFLOW (client): {error}", "model": model}
    call = r.log()
    if out is not None and not out.summary.strip():
        return None, call | {"error": "SUMMARY_EMPTY: el resumen vino vacío"}
    return (out.summary.strip() if out else None), call


class SummaryV2(BaseModel):
    # Como ``Summary``: sin campo de razonamiento. El tope de la gramática es una red, no el largo pedido.
    summary: str = Field(max_length=SUMMARY_V2_GRAMMAR_MAX)


SUMMARY_V2_SYSTEM = """Mantenés el resumen de una conversación entre un usuario y un asistente sobre legislación española de vivienda y alquiler. Recibís el resumen anterior (puede estar vacío) y los mensajes que salen de la ventana. Escribí el resumen nuevo, que reemplaza al anterior.

Reglas:
- Reescribí el resumen entero, juntando el anterior con los mensajes nuevos. Nunca copies el resumen anterior tal cual: lo nuevo tiene que entrar.
- A lo sumo 35 palabras, en tercera persona ("el usuario..."). Para que entre, comprimí: primero sacá lo que ya se le respondió en turnos viejos y conservá siempre los datos que dio el usuario (fechas, lugar, tipo de contrato, montos, quién es el propietario).
- No agregues nada que no esté en el resumen anterior o en los mensajes: ni suposiciones ni opiniones."""


def summarize_v2(previous: str, evicted: list[dict], *, model: str = grounded.MODEL,
                 num_ctx: int = grounded.NUM_CTX) -> tuple[str | None, dict]:
    """Como ``summarize``, con el prompt y el schema de la v2."""
    try:
        out, r = llm.structured(summary_messages(previous, evicted, SUMMARY_V2_SYSTEM), SummaryV2, model=model,
                                num_ctx=num_ctx, num_predict=SUMMARY_NUM_PREDICT)
    except llm.ContextOverflow as error:
        return None, {"error": f"CONTEXT_OVERFLOW (client): {error}", "model": model}
    call = r.log()
    if out is not None and not out.summary.strip():
        return None, call | {"error": "SUMMARY_EMPTY: el resumen vino vacío"}
    return (out.summary.strip() if out else None), call


def summary_block(summary: str) -> str:
    return f"<resumen>\n{summary}\n</resumen>" if summary else ""


def with_summary(system: str, summary: str) -> str:
    """El resumen va al final del mensaje de sistema, a la cabeza de la cola de mensajes (MemGPT). Como
    mensaje de usuario se leería como un pedido del usuario."""
    return f"{system}\n\n{summary_block(summary)}" if summary else system


class SummaryBuffer:
    """Un resumen continuo más los mensajes recientes, al estilo MemGPT (Apéndice B.4).

    Cuando resumen + recientes cruzan ``trigger`` del tope, la mitad más vieja de los recientes (en pares
    usuario-asistente, redondeando para arriba) sale de la ventana y se incorpora al resumen con una
    llamada. El último par nunca sale: es el que necesitan una correferencia o una elipsis, y sin él la
    comparación con la ventana mezclaría "comprimir" con "perder el turno anterior". El resumen ocupa parte
    del tope. ``updates`` guarda cada actualización, para auditarla.
    """

    def __init__(self, cap: int, count: Callable[[str], int], summarizer: Summarizer,
                 trigger: float = SUMMARY_TRIGGER, max_chars: int = SUMMARY_MAX, target_chars: int | None = None):
        self.cap, self.count, self.summarizer, self.trigger = cap, count, summarizer, trigger
        self.max_chars = max_chars  # el tope de la gramática del resumidor: llegar ahí es un corte
        self.target_chars = target_chars  # el largo pedido, si es menor que el tope: pasarlo se cuenta aparte
        self.summary = ""
        self.recent: list[dict] = []
        self.updates: list[dict] = []

    def summary_tokens(self) -> int:
        return self.count(summary_block(self.summary)) + MESSAGE_MARGIN if self.summary else 0

    def used(self) -> int:
        return self.summary_tokens() + history_tokens(self.recent, self.count)

    def add(self, messages: list[dict]) -> list[dict]:
        """Agrega mensajes y resume si hace falta. Devuelve los registros de las llamadas que hizo."""
        self.recent.extend(messages)
        calls = []
        while self.used() > self.trigger * self.cap and len(self.recent) > 2:
            pairs = len(self.recent) // 2
            n_out = 2 * min(pairs - 1, (pairs + 1) // 2)  # la mitad más vieja, sin tocar el último par
            evicted, self.recent = self.recent[:n_out], self.recent[n_out:]
            new, call = self.summarizer(self.summary, evicted)
            calls.append(call)
            update = {"before": self.summary, "evicted": evicted, "after": new, "error": call.get("error"),
                      "at_maxlength": new is not None and len(new) >= self.max_chars}
            if self.target_chars is not None:  # solo la v2: las filas de la Etapa 5 quedan como estaban
                update["over_target"] = new is not None and len(new) > self.target_chars
                update["copied_previous"] = new is not None and bool(self.summary) and new == self.summary
            self.updates.append(update)
            if new is not None:
                self.summary = new
            # Si la llamada falló, el resumen anterior queda y lo desalojado se pierde: queda contado como
            # error en ``updates`` (devolverlo a la ventana no entraría, y reintentar daría lo mismo).
        return calls

    def view(self) -> tuple[str, list[dict], int]:
        """(resumen, recientes que entran, tokens). Los recientes se recortan a lo que deja el resumen: un
        par largo puede no entrar aunque el umbral no haya saltado."""
        kept, used = window(self.recent, self.cap - self.summary_tokens(), self.count)
        return self.summary, kept, used + self.summary_tokens()


@dataclass(frozen=True)
class HistoryArm:
    name: str
    strategy: str  # "window" | "summary" | "full"
    cap: int | None  # el tope del historial, en tokens del generador; None = sin tope


# En orden de corrida. Experimento 1: el tope, con la ventana. Experimento 2: la compresión con el tope en 150
# (la ventana 150 es la misma fila en los dos; con 100 el resumen no entraría junto al último par). El buffer
# completo es la fila de referencia, fuera de las comparaciones de una variable.
ARMS = (*(HistoryArm(f"window_h{cap}", "window", cap) for cap in HISTORY_CAPS),
        HistoryArm(f"summary_h{COMPRESSION_CAP}", "summary", COMPRESSION_CAP),
        HistoryArm("full_buffer", "full", None))
COMPARISONS = (("window_h100", "window_h150"), ("window_h150", "window_h250"), ("window_h100", "window_h250"),
               ("window_h150", "summary_h150"))
REFERENCE = "full_buffer"


# La v2 del resumen (Etapa 8): misma estrategia y mismo tope, otro resumidor. Va aparte de ``ARMS`` para que la
# especificación de la Etapa 5 (que lleva ``vars`` de cada brazo) no cambie.
SUMMARY_V2_ARM = HistoryArm(f"summary_v2_h{COMPRESSION_CAP}", "summary_v2", COMPRESSION_CAP)


class HistoryState:
    """El historial de un brazo a lo largo de UN guion: todo lo guionado y lo que el brazo ve.

    - ``window``: los mensajes más nuevos que entran en el tope;
    - ``summary``: ``SummaryBuffer`` con el tope;
    - ``summary_v2``: ``SummaryBuffer`` con el tope y el resumidor v2 (``summarize_v2``);
    - ``full``: todo, sin tope. Si el prompt no entra en ``num_ctx − num_predict``, ``grounded.generate`` lo
      registra como overflow (un error de llamada, no una respuesta).
    """

    def __init__(self, arm: HistoryArm, count: Callable[[str], int], summarizer: Summarizer | None = None):
        self.arm, self.count = arm, count
        self.messages: list[dict] = []
        self.buffer = None
        if arm.strategy in ("summary", "summary_v2"):
            if summarizer is None or arm.cap is None:
                raise RuntimeError(f"strategy={arm.strategy} necesita un summarizer y un tope")
            self.buffer = (SummaryBuffer(arm.cap, count, summarizer) if arm.strategy == "summary" else
                           SummaryBuffer(arm.cap, count, summarizer, max_chars=SUMMARY_V2_GRAMMAR_MAX,
                                         target_chars=SUMMARY_V2_TARGET))
        elif arm.strategy == "window" and arm.cap is None:
            raise RuntimeError("strategy=window necesita un tope")
        elif arm.strategy not in ("window", "full"):
            raise ValueError(f"estrategia desconocida: {arm.strategy}")

    def add(self, exchange: list[dict]) -> list[dict]:
        self.messages.extend(exchange)
        return self.buffer.add(exchange) if self.buffer is not None else []

    def view(self) -> tuple[str, list[dict], int]:
        """(resumen, mensajes que ve el generador, tokens de historial con el resumen)."""
        if self.buffer is not None:
            return self.buffer.view()
        if self.arm.strategy == "full":
            return "", list(self.messages), history_tokens(self.messages, self.count)
        kept, used = window(self.messages, self.arm.cap, self.count)
        return "", kept, used


def window_losses(turns: list[ReplayTurn], cap: int, count: Callable[[str], int]) -> list[dict]:
    """Por turno de un guion: el historial completo antes de responderlo y cuánto pierde la ventana con ``cap``.

    La ventana conserva un sufijo del historial, así que perder cualquier mensaje es perder el más viejo:
    el turno 1. ``empty_by_cap``: hay historial pero no entra ni el último par (con un tope chico, un par
    largo no cabe). Se mira ANTES de la primera respuesta: un tope que no recorta nada del propio guion hace
    que la tabla diga "el historial no importa"."""
    rows = []
    for index, rt in enumerate(turns):
        history = scripted_history(turns[:index])
        kept, _ = window(history, cap, count)
        rows.append({"turn_id": rt.turn_id, "turn": rt.number, "history_tokens": history_tokens(history, count),
                     "lost_messages": len(history) - len(kept), "loses_turn_1": len(kept) < len(history),
                     "empty_by_cap": bool(history) and not kept})
    return rows


# --- Prompts de la Parte B -----------------------------------------------------------------------

RULE_CONVERSATION = ("- Antes de la pregunta vas a ver la conversación anterior (o un resumen entre <resumen> y "
                     "</resumen>). Usala para entender a qué se refiere la pregunta y los datos del usuario (fechas, "
                     "tipo de contrato, lugar), pero la respuesta sale de los pasajes.")


def conversation_system(today: date) -> str:
    """El prompt de la Etapa 4 más una regla: sin ella, "respondé solo con lo que dicen los pasajes" le
    prohibiría al modelo usar la fecha que el usuario dijo dos turnos atrás."""
    return f"{grounded.system_prompt(today)}\n{RULE_CONVERSATION}"


RULES_RECALL = f"""Antes de la pregunta vas a ver la conversación anterior (o un resumen entre <resumen> y </resumen>). No vas a recibir pasajes.

Reglas:
- Respondé solo con lo que se dijo en la conversación (o en el resumen), no con lo que sepas de antes.
- Si no está, poné can_answer en false y en answer escribí exactamente: "{grounded.ABSTENTION}"
- Dejá citations vacía.
- En analysis, primero y en una o dos oraciones: dónde se dijo. Después answer, en pocas oraciones."""


def recall_system(today: date) -> str:
    return f"{grounded.system_head(today)}\n\n{RULES_RECALL}"


def recall_probe(first: Turn) -> tuple[str, Answer | None]:
    """La sonda del resumen (paso 9), derivada del guion congelado: se le pregunta qué respondió en el
    primer turno, y se puntúa con la clave de ese turno. No se redacta ningún ítem nuevo.

    La sonda cita la pregunta del turno 1, así que una clave que ya está en ese texto se acertaría copiando
    la pregunta: esas claves se sacan. Si no queda ninguna, el guion no tiene sonda (``None``).
    """
    question = f'Antes te pregunté: "{first.user}" ¿Qué me respondiste?'
    keys = [key for key in first.answer_key if not contains_key(question, [key])]
    if not keys:
        return question, None
    return question, Answer(gold_sections=first.gold_sections, answer_key=keys)
