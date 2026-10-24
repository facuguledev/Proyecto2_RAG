"""El conjunto de evaluación (Etapa 2): contratos, validación contra el corpus y congelado.

Acá vive lo que el enunciado llama ``schema.EvalItem``, ``schema.validate_eval``, ``schema.freeze_eval``
y ``schema.load_eval``: el ``schema.py`` del curso no representa nuestro eval set (decision_log.md,
2026-10-21), y el contrato de los documentos está en ``doc_schema.py``. Hay tres conjuntos, un archivo
cada uno en ``eval/``:

- ``single_turn.jsonl``: preguntas de un solo turno (``EvalItem``), para las Etapas 3-4;
- ``dialogues.jsonl``: guiones de conversación (``Dialogue``), para la Etapa 5;
- ``scenarios.jsonl``: escenarios de varias sesiones (``Scenario``), para la Etapa 6.

Cada regla se chequea en uno de tres lugares:
1. el modelo, al leer el archivo: lo que no necesita el corpus;
2. ``validate_eval``: contra el corpus (el gold existe, rige ese día y contiene la clave);
3. ``check_sizes``: los tamaños de §7.3, solo al congelar (el smoke se valida, pero es más chico).
"""
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple, get_args

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from artifact_io import read_jsonl, sha256_file, write_manifest
from corpus import SNAPSHOT
from doc_schema import Check, Document, Section, ValidationReport, documents_hash, load_documents

Category = Literal["topical", "identifier", "metadata", "out_of_corpus"]
Dependency = Literal["standalone", "coreference", "ellipsis", "topic_shift", "return", "unanswerable"]
Ability = Literal["extraction", "multi_session", "temporal", "knowledge_update", "abstention", "preference"]
Split = Literal["dev", "test"]
Verdict = Literal["yes", "no"]
CATEGORIES, DEPENDENCIES, ABILITIES = get_args(Category), get_args(Dependency), get_args(Ability)


# --- Modelos -------------------------------------------------------------------------------------

def doc_of(section_id: str) -> str:
    """El documento de una sección: 'BOE-A-1994-26003#a9@20190306' -> 'BOE-A-1994-26003'."""
    return section_id.split("#", 1)[0]


def block_of(section_id: str) -> str:
    """El bloque (el artículo, en cualquier versión): 'BOE-A-1994-26003#a9@20190306' -> 'BOE-A-1994-26003#a9'."""
    return section_id.split("@", 1)[0]


def require_label_iff_unanswerable(answerable: bool, has_label: bool, label: str) -> None:
    """La etiqueta de "sin respuesta" (out_of_corpus, unanswerable, abstention) va con answerable=false,
    y answerable=false va solo con ella: un "no" que el corpus respalda es una pregunta con respuesta."""
    if has_label and answerable:
        raise ValueError(f"{label} lleva answerable=false")
    if not has_label and not answerable:
        raise ValueError(f"answerable=false es solo para {label}")


class Answer(BaseModel):
    """Lo que se puntúa de una pregunta. Lo comparten las preguntas sueltas, los turnos y las sondas."""

    model_config = ConfigDict(extra="forbid")  # un campo mal escrito es un error, no un dato perdido

    answerable: bool = True
    # Dónde está la respuesta. En las preguntas sueltas y los turnos vale cualquiera de las secciones, y
    # cada una contiene una clave. En una sonda son las secciones en las que se apoya la respuesta (su
    # clave puede salir de lo que dijo el usuario).
    gold_sections: list[str] = []
    answer_key: list[str] = []  # cadenas cortas; alcanza con que la respuesta contenga una
    expected_verdict: Verdict | None = None  # solo en las preguntas de sí/no
    # El día por el que se pregunta (ver reference_day). Es una etiqueta para validar el gold: el pipeline
    # nunca la lee (sería darle la respuesta); la fecha la tiene que sacar de la pregunta o de la memoria.
    as_of: date | None = None

    @property
    def gold_docs(self) -> list[str]:
        return sorted({doc_of(section_id) for section_id in self.gold_sections})

    def reference_day(self) -> date:
        """El día cuya versión de la ley responde: as_of o, si no hay, la foto del corpus."""
        return self.as_of or SNAPSHOT

    @model_validator(mode="after")
    def check_answer(self):
        if not self.answerable and (self.gold_sections or self.answer_key or self.expected_verdict):
            raise ValueError("una pregunta sin respuesta no lleva gold, clave ni veredicto")
        if self.answerable and not self.answer_key:
            raise ValueError("una pregunta con respuesta lleva al menos una clave")
        if any(is_blank_key(key) for key in self.answer_key):
            raise ValueError("una clave sin letras ni números estaría contenida en casi cualquier respuesta")
        return self


class EvalItem(Answer):
    """Una pregunta de un solo turno (Etapas 3-4)."""

    item_id: str
    question: str
    category: Category
    topic: str  # el tema de eval/topics.md desde el que se escribió
    split: Split = "test"
    notes: str = ""  # dónde se verificó el gold, por qué una pregunta fuera del corpus es adyacente...

    @model_validator(mode="after")
    def check_category(self):
        require_label_iff_unanswerable(self.answerable, self.category == "out_of_corpus", "out_of_corpus")
        if self.answerable and not self.gold_sections:
            raise ValueError(f"{self.item_id}: una pregunta con respuesta lleva gold")
        return self


class Turn(Answer):
    """Un turno de un guion: la pregunta del usuario y la respuesta guionada del asistente."""

    user: str  # tal como la escribe el usuario, con sus referencias a lo anterior
    depends_on: Dependency
    oracle_rewrite: str  # la misma pregunta reescrita a mano para entenderse sin historial (Etapa 5A)
    assistant: str  # todos los brazos de la Etapa 5 ven esta respuesta como historial, no la suya

    @model_validator(mode="after")
    def check_dependency(self):
        require_label_iff_unanswerable(self.answerable, self.depends_on == "unanswerable", "unanswerable")
        if self.answerable and not self.gold_sections:
            raise ValueError("un turno con respuesta lleva gold")
        return self


class Dialogue(BaseModel):
    """Un guion de conversación (Etapa 5)."""

    model_config = ConfigDict(extra="forbid")

    dialogue_id: str
    turns: list[Turn]
    split: Split = "test"
    notes: str = ""

    @model_validator(mode="after")
    def check_turns(self):
        if not self.turns or self.turns[0].depends_on != "standalone":
            raise ValueError(f"{self.dialogue_id}: el primer turno es standalone")
        if any(turn.depends_on == "standalone" for turn in self.turns[1:]):
            raise ValueError(f"{self.dialogue_id}: solo el primer turno es standalone")
        return self


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class Session(BaseModel):
    """Una sesión de un escenario: lo que se dijo, con su fecha simulada."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    started_at: datetime  # el tiempo simulado del escenario, nunca el reloj real
    messages: list[Message]

    @model_validator(mode="after")
    def check_messages(self):
        roles = [message.role for message in self.messages]
        if not roles or roles != ["user", "assistant"] * (len(roles) // 2):
            raise ValueError(f"{self.session_id}: los mensajes van de a pares, primero el usuario")
        return self


class Probe(Answer):
    """Una pregunta de una sesión posterior que necesita algo dicho antes (Etapa 6)."""

    probe_id: str
    question: str
    asked_at: datetime  # tiempo simulado: la memoria solo puede saber lo que se dijo antes
    ability: Ability
    evidence_sessions: list[str] = []  # dónde se dijo lo que hace falta para responder

    def reference_day(self) -> date:
        """Sin as_of, una sonda pregunta por la ley del día en que se hace (su "hoy" simulado)."""
        return self.as_of or self.asked_at.date()

    @model_validator(mode="after")
    def check_ability(self):
        require_label_iff_unanswerable(self.answerable, self.ability == "abstention", "abstention")
        if self.answerable and not self.evidence_sessions:
            raise ValueError(f"{self.probe_id}: una sonda con respuesta dice en qué sesión está la evidencia")
        if not self.answerable and self.evidence_sessions:
            raise ValueError(f"{self.probe_id}: lo que nunca se dijo no tiene sesión de evidencia")
        return self


class Scenario(BaseModel):
    """Una persona ficticia, sus sesiones y las sondas que se le hacen después (Etapa 6)."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    persona: str  # ficticia: nada de datos personales reales (§6.1)
    sessions: list[Session]
    probes: list[Probe]
    split: Split = "test"
    notes: str = ""

    @model_validator(mode="after")
    def check_timeline(self):
        times = [session.started_at for session in self.sessions]
        if any(earlier >= later for earlier, later in zip(times, times[1:])):
            raise ValueError(f"{self.scenario_id}: cada sesión empieza después de la anterior")
        started = {session.session_id: session.started_at for session in self.sessions}
        for probe in self.probes:
            for session_id in probe.evidence_sessions:
                if session_id not in started or started[session_id] >= probe.asked_at:
                    raise ValueError(f"{probe.probe_id}: {session_id} no es una sesión anterior a la pregunta")
        if "abstention" not in {probe.ability for probe in self.probes}:
            raise ValueError(f"{self.scenario_id}: cada escenario tiene un hecho que nunca se dijo (abstention)")
        # Una actualización solo se ve en un par: la misma pregunta antes y después, con veredictos opuestos.
        # Sin veredicto, una memoria que nunca registró el cambio puntuaría igual que una que sí.
        update_verdicts = {p.expected_verdict for p in self.probes if p.ability == "knowledge_update"}
        if not {"yes", "no"} <= update_verdicts:
            raise ValueError(f"{self.scenario_id}: cada escenario tiene un hecho que cambia, medido con un par "
                             "de knowledge_update con veredictos sí y no")
        return self


# --- Leer el conjunto ----------------------------------------------------------------------------

EVAL_FILES = {"single_turn.jsonl": EvalItem, "dialogues.jsonl": Dialogue, "scenarios.jsonl": Scenario}


@dataclass
class EvalSet:
    items: list[EvalItem]
    dialogues: list[Dialogue]
    scenarios: list[Scenario]
    file_hashes: dict[str, str]  # archivo -> sha256 de su contenido

    def key(self) -> str:
        """El id de este conjunto: va en cada fila de resultados, y cambia si cambia cualquier archivo."""
        joined = "\n".join(f"{name} {sha}" for name, sha in sorted(self.file_hashes.items()))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def summary(self) -> str:
        n_turns = sum(len(dialogue.turns) for dialogue in self.dialogues)
        n_probes = sum(len(scenario.probes) for scenario in self.scenarios)
        return (f"{len(self.items)} preguntas, {len(self.dialogues)} guiones ({n_turns} turnos), "
                f"{len(self.scenarios)} escenarios ({n_probes} sondas)")


def read_rows(path: Path, model: type[BaseModel]) -> list:
    """Las filas de un archivo como ``model``. Una fila inválida es un error que dice cuál es."""
    rows = []
    for number, row in enumerate(read_jsonl(path), start=1):
        try:
            rows.append(model.model_validate(row))
        except ValidationError as error:
            raise ValueError(f"{path.name}, fila {number}: {error}") from error
    return rows


def load_eval(eval_dir: Path, documents_path: Path | None = None, require_frozen: bool = True) -> EvalSet:
    """Lee los tres archivos del conjunto.

    Congelado (lo normal), rechaza un archivo editado después de congelar y un corpus que no es el del
    congelado: los dos darían números sobre otra cosa, sin ningún error. ``require_frozen=False`` es
    para el smoke, que nunca se congela: no chequea hashes ni ids (los ids los chequea ``validate_eval``).
    """
    eval_dir = Path(eval_dir)
    rows = {name: read_rows(eval_dir / name, model) for name, model in EVAL_FILES.items()}
    ev = EvalSet(items=rows["single_turn.jsonl"], dialogues=rows["dialogues.jsonl"],
                 scenarios=rows["scenarios.jsonl"],
                 file_hashes={name: sha256_file(eval_dir / name) for name in EVAL_FILES})
    if require_frozen:
        if documents_path is None:
            raise ValueError("un conjunto congelado se carga contra su corpus: falta documents_path")
        check_frozen(ev, eval_dir / "FROZEN.json", documents_hash(load_documents(Path(documents_path))))
    return ev


def check_frozen(ev: EvalSet, frozen_path: Path, corpus_sha: str) -> None:
    if not frozen_path.exists():
        raise FileNotFoundError(f"{frozen_path} no existe: el conjunto no está congelado (freeze_eval)")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    edited = sorted(name for name, sha in ev.file_hashes.items() if frozen["files"].get(name) != sha)
    if edited:
        raise ValueError(f"cambió después de congelar: {', '.join(edited)}. Una edición del conjunto "
                         "congelado es una entrada de decision_log.md, y después se congela de nuevo")
    if frozen["corpus_sha"] != corpus_sha:
        raise ValueError(f"el corpus no es el del congelado: corpus_sha {corpus_sha[:12]}..., "
                         f"congelado contra {frozen['corpus_sha'][:12]}...")


# --- Claves de respuesta -------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Para comparar claves: minúsculas, sin acentos, sin separador de miles y con punto decimal.

    Se aplica una sola vez, sobre el texto crudo: dos pasadas convertirían "1,500" en "1.500" y después
    en "1500".
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)  # "1.500" -> "1500"
    text = re.sub(r"(?<=\d),0+\b", "", text)  # "1300,00" -> "1300": los céntimos en cero no cambian el número
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)  # "2,5" -> "2.5"
    return " ".join(text.split())


def is_blank_key(key: str) -> bool:
    """Una clave sin letras ni números ("", " ", "-") estaría contenida en casi cualquier texto."""
    return not re.search(r"\w", normalize(key))


def key_pattern(key: str) -> re.Pattern:
    """La clave como regex sobre texto ya normalizado: entera, ni pegada a otra palabra o dígito ni partida
    por un decimal. Así "10" no está en "100" ni en "10,5", y "artículo 10" no está en "artículo 10.1"."""
    if is_blank_key(key):
        raise ValueError(f"clave sin letras ni números: {key!r}")
    return re.compile(r"(?<!\w)(?<!\d\.)" + re.escape(normalize(key)) + r"(?!\w)(?!\.\d)")


def contains_key(text: str, keys: list[str]) -> bool:
    """¿Contiene ``text`` (crudo) alguna clave entera?"""
    patterns = [key_pattern(key) for key in keys]  # todas primero: una clave en blanco falla siempre
    text = normalize(text)
    return any(pattern.search(text) for pattern in patterns)


# --- Validación contra el corpus -----------------------------------------------------------------

Sections = dict[str, tuple[Document, Section]]  # section_id -> (su documento, la sección)


def section_index(docs: list[Document]) -> Sections:
    return {section.section_id: (doc, section) for doc in docs for section in doc.sections}


def section_text(sections: Sections, section_id: str) -> str:
    doc, section = sections[section_id]
    return doc.section_text(section)


def twins_with_key(sections: Sections, section_id: str, keys: list[str], day: date) -> list[str]:
    """Los gemelos de una sección (texto casi igual, en otra norma) que rigen ``day`` y contienen una clave."""
    twins = sections[section_id][1].twins if section_id in sections else []
    return [twin for twin in twins if twin in sections and sections[twin][1].in_force(day)
            and contains_key(section_text(sections, twin), keys)]


def corpus_questions(ev: EvalSet) -> list[tuple[str, Answer]]:
    """Las preguntas sueltas y los turnos, con un nombre para los mensajes. Se responden con el corpus, o
    con una abstención si son out_of_corpus o unanswerable."""
    named: list[tuple[str, Answer]] = [(item.item_id, item) for item in ev.items]
    for dialogue in ev.dialogues:
        named += [(f"{dialogue.dialogue_id}.t{n}", turn) for n, turn in enumerate(dialogue.turns, start=1)]
    return named


def probes(ev: EvalSet) -> list[tuple[str, Probe]]:
    return [(probe.probe_id, probe) for scenario in ev.scenarios for probe in scenario.probes]


def check_unique_ids(ev: EvalSet) -> list[str]:
    ids = [item.item_id for item in ev.items] + [dialogue.dialogue_id for dialogue in ev.dialogues]
    for scenario in ev.scenarios:
        ids += [scenario.scenario_id] + [session.session_id for session in scenario.sessions]
        ids += [probe.probe_id for probe in scenario.probes]
    return [f"{id_}: aparece {n} veces" for id_, n in Counter(ids).items() if n > 1]


def check_gold_exists(ev: EvalSet, sections: Sections) -> list[str]:
    """Cada id gold es una sección del corpus (§7.4: un id faltante tiene que saltar acá)."""
    return [f"{name}: {section_id} no está en el corpus" for name, answer in corpus_questions(ev) + probes(ev)
            for section_id in answer.gold_sections if section_id not in sections]


def check_gold_in_force(ev: EvalSet, sections: Sections) -> list[str]:
    """El gold rige el día por el que se pregunta: a una pregunta sin fecha le toca la versión vigente."""
    failures = []
    for name, answer in corpus_questions(ev) + probes(ev):
        day = answer.reference_day()
        for section_id in answer.gold_sections:
            if section_id in sections and not sections[section_id][1].in_force(day):
                failures.append(f"{name}: {section_id} no rige el {day}")
    return failures


def check_key_in_gold(ev: EvalSet, sections: Sections) -> list[str]:
    """Cada sección gold contiene alguna clave. Como vale cualquiera, una sección sin la respuesta (una que
    solo remite a otra norma) daría un acierto de recuperación sin entregar la respuesta.

    Las sondas quedan afuera: su clave puede salir de lo que dijo el usuario, no del corpus.
    """
    failures = []
    for name, answer in corpus_questions(ev):
        for section_id in answer.gold_sections:
            if section_id in sections and not contains_key(section_text(sections, section_id), answer.answer_key):
                failures.append(f"{name}: {section_id} no contiene ninguna clave de {answer.answer_key}")
    return failures


def check_twins_in_gold(ev: EvalSet, sections: Sections) -> list[str]:
    """Un gemelo que rige ese día y contiene la clave es la misma respuesta en otra norma: va en el gold.

    Si no está, recuperar la copia cuenta como un fallo (la trampa "dos documentos que son el mismo").
    """
    failures = []
    for name, answer in corpus_questions(ev) + probes(ev):
        day = answer.reference_day()
        missing = {twin for section_id in answer.gold_sections
                   for twin in twins_with_key(sections, section_id, answer.answer_key, day)}
        failures += [f"{name}: falta el gemelo {twin}" for twin in sorted(missing - set(answer.gold_sections))]
    return failures


def validate_eval(ev: EvalSet, docs: list[Document]) -> ValidationReport:
    """Chequea el conjunto contra el corpus. No levanta: junta los fallos en el reporte."""
    sections = section_index(docs)
    return ValidationReport("eval", ev.summary(), [
        Check("unique_ids", check_unique_ids(ev)),
        Check("gold_exists", check_gold_exists(ev, sections)),
        Check("gold_in_force", check_gold_in_force(ev, sections)),
        Check("key_in_gold", check_key_in_gold(ev, sections)),
        Check("twins_in_gold", check_twins_in_gold(ev, sections)),
    ])


# --- Tamaños (§7.3) ------------------------------------------------------------------------------
# Las cantidades se cuentan en test, que es lo que se reporta; los rangos por guion y por escenario
# valen para todos.

def single_turn_size_failures(items: list[EvalItem]) -> list[str]:
    test = [item for item in items if item.split == "test"]
    per_category = Counter(item.category for item in test)
    failures = [] if len(test) >= 30 else [f"{len(test)} preguntas en test, mínimo 30"]
    for category, minimum in {"out_of_corpus": 3, "identifier": 5, "metadata": 5}.items():
        if per_category[category] < minimum:
            failures.append(f"{per_category[category]} de {category} en test, mínimo {minimum}")
    without_dev = set(CATEGORIES) - {item.category for item in items if item.split == "dev"}
    if without_dev:
        failures.append(f"categorías sin preguntas dev: {', '.join(sorted(without_dev))}")
    return failures


def dialogue_size_failures(dialogues: list[Dialogue]) -> list[str]:
    n_test = sum(dialogue.split == "test" for dialogue in dialogues)
    failures = [] if 8 <= n_test <= 15 else [f"{n_test} guiones en test, entre 8 y 15"]
    failures += [f"{dialogue.dialogue_id}: {len(dialogue.turns)} turnos, entre 3 y 6"
                 for dialogue in dialogues if not 3 <= len(dialogue.turns) <= 6]
    return failures


def scenario_size_failures(scenarios: list[Scenario]) -> list[str]:
    n_test = sum(scenario.split == "test" for scenario in scenarios)
    failures = [] if 3 <= n_test <= 6 else [f"{n_test} escenarios en test, entre 3 y 6"]
    for scenario in scenarios:
        if not 2 <= len(scenario.sessions) <= 4:
            failures.append(f"{scenario.scenario_id}: {len(scenario.sessions)} sesiones, entre 2 y 4")
        if not 5 <= len(scenario.probes) <= 10:
            failures.append(f"{scenario.scenario_id}: {len(scenario.probes)} sondas, entre 5 y 10")
    return failures


def check_sizes(ev: EvalSet) -> ValidationReport:
    """Los tamaños de §7.3. Solo al congelar: el smoke se valida contra el corpus, pero es más chico."""
    return ValidationReport("eval", "tamaños de §7.3", [
        Check("single_turn", single_turn_size_failures(ev.items)),
        Check("dialogues", dialogue_size_failures(ev.dialogues)),
        Check("scenarios", scenario_size_failures(ev.scenarios)),
    ])


# --- Congelar ------------------------------------------------------------------------------------

def freeze_eval(eval_dir: Path, documents_path: Path, overwrite: bool = False) -> dict:
    """Valida el conjunto contra el corpus completo y escribe ``eval/FROZEN.json``.

    Se corre antes de que exista cualquier resultado, y el archivo se commitea. Volver a correrlo sin
    cambios no hace nada (la notebook se puede re-ejecutar). Si cambió el conjunto o el corpus, falla:
    una edición del conjunto congelado es una entrada de decision_log.md, y recién después se congela
    de nuevo con ``overwrite=True``.
    """
    eval_dir = Path(eval_dir)
    ev = load_eval(eval_dir, require_frozen=False)
    docs = load_documents(Path(documents_path))
    validate_eval(ev, docs).show().raise_if_failed()
    check_sizes(ev).show().raise_if_failed()

    frozen = {"files": ev.file_hashes, "corpus_sha": documents_hash(docs), "key": ev.key()}
    path = eval_dir / "FROZEN.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if all(previous.get(field) == value for field, value in frozen.items()):
            return previous  # ya estaba congelado así
        if not overwrite:
            raise FileExistsError(f"{path} ya existe y el conjunto o el corpus cambiaron: registrá la "
                                  "edición en decision_log.md y congelá de nuevo con overwrite=True")
    frozen["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_manifest(path, frozen)
    return frozen


# --- Preguntas de identificador derivadas --------------------------------------------------------

DERIVED_TOPIC = "derivada: un artículo pedido por su número"
ARTICLE_TITLE = re.compile(r"^Artículo (.+?)\.\s+(.+?)\.?$")  # "Artículo 36. Fianza." -> ("36", "Fianza")
# El número, con o sin título ("Artículo 3."), cortado igual que en ARTICLE_TITLE: "2.bis" no es "2".
ARTICLE_NUMBER = re.compile(r"^Artículo (.+?)\.(?:\s|$)")


class Candidate(NamedTuple):
    doc: Document
    section: Section
    number: str  # "36", "9 bis"
    title: str  # "Fianza"


def norm_name(doc: Document) -> str | None:
    """'Ley 29/1994', o 'Ley 5/2018 (Illes Balears)' si es autonómica; None sin número oficial."""
    meta = doc.metadata
    number = getattr(meta, "numero_oficial", None)
    if not number:
        return None
    name = f"{meta.rango} {number}"
    return name if meta.region == "Estatal" else f"{name} ({meta.region})"


def identifier_question(candidate: Candidate) -> str:
    rango = candidate.doc.metadata.rango
    of = "del" if rango.startswith(("Real Decreto", "Decreto")) else "de la"  # del Decreto, de la Ley
    return f"¿Qué regula el artículo {candidate.number} {of} {norm_name(candidate.doc)}?"


def blocks_naming(title: str, block_titles: set[tuple[str, str, str]]) -> int:
    """Cuántos bloques del corpus llevan ``title`` (palabras completas) en su título, en alguna versión."""
    title = normalize(title)
    pattern = re.compile(r"(?<!\w)" + re.escape(title) + r"(?!\w)")
    return len({(doc_id, block_id) for doc_id, block_id, text in block_titles
                if title in text and pattern.search(text)})  # el "in" descarta rápido; el regex confirma


def identifier_candidates(docs: list[Document]) -> list[Candidate]:
    """Los artículos vigentes que se pueden pedir por su número sin ambigüedad (ver abajo)."""
    names = Counter(norm_name(doc) for doc in docs)
    block_titles = {(doc.doc_id, section.block_id, normalize(section.title)) for doc in docs
                    for section in doc.sections}
    articles, numbers = [], Counter()
    for doc in docs:
        if not norm_name(doc) or names[norm_name(doc)] > 1:
            continue  # sin número oficial, o dos normas que se llaman igual
        for section in doc.sections:
            if section.kind != "article" or not section.in_force(SNAPSHOT):
                continue
            if number := ARTICLE_NUMBER.match(section.title):
                numbers[(doc.doc_id, number.group(1))] += 1
            if match := ARTICLE_TITLE.match(section.title):
                articles.append(Candidate(doc, section, *match.groups()))
    # Pedido por su número, el artículo tiene que ser uno solo en su norma (un reglamento anexo podría
    # repetir la numeración). Y su título tiene que nombrar una sola cosa en el corpus: ningún otro bloque
    # (artículo, disposición, capítulo) lo lleva. Así quedan afuera "Entrada en vigor" u "Objeto del programa".
    return [article for article in articles
            if numbers[(article.doc.doc_id, article.number)] == 1 and len(article.title.split()) <= 4
            and blocks_naming(article.title, block_titles) == 1]


def derive_identifier_items(docs: list[Document], n_dev: int, n_test: int, seed: int = 0) -> list[EvalItem]:
    """Las preguntas de identificador, derivadas del corpus en lugar de escritas (Etapa 2, paso 1).

    Entra un artículo vigente si ningún otro bloque del corpus lleva su título, ese título tiene a lo
    sumo 4 palabras y su norma tiene un número oficial único. La pregunta lo pide por su número ("¿Qué
    regula el artículo 36 de la Ley 29/1994?"), la clave es su título y el gold es la sección más sus
    gemelos con la clave (también los de otra norma: es el mismo texto, y twins_in_gold lo exige igual que
    en cualquier pregunta). Una por norma, alternando estatales y autonómicas. Es determinista: la
    notebook la vuelve a correr y compara con lo que hay en el archivo.
    """
    rng = random.Random(seed)
    by_norm: dict[str, list[Candidate]] = {}
    for candidate in identifier_candidates(docs):
        by_norm.setdefault(candidate.doc.doc_id, []).append(candidate)
    state = sorted(doc_id for doc_id, found in by_norm.items() if found[0].doc.metadata.region == "Estatal")
    regional = sorted(set(by_norm) - set(state))
    rng.shuffle(state)
    rng.shuffle(regional)
    picked = [doc_id for pair in zip(state, regional) for doc_id in pair][:n_dev + n_test]
    if len(picked) < n_dev + n_test:  # zip corta en la lista más corta: no devolver menos sin avisar
        raise ValueError(f"alcanzan para {len(picked)} preguntas alternando estatales y autonómicas; "
                         f"se pidieron {n_dev + n_test}")

    sections = section_index(docs)
    items = []
    for n, doc_id in enumerate(picked):
        candidate = rng.choice(by_norm[doc_id])
        gold = [candidate.section.section_id]
        gold += twins_with_key(sections, candidate.section.section_id, [candidate.title], SNAPSHOT)
        items.append(EvalItem(
            item_id=f"id{n + 1:02d}", question=identifier_question(candidate), category="identifier",
            topic=DERIVED_TOPIC, gold_sections=gold, answer_key=[candidate.title],
            split="dev" if n < n_dev else "test", notes=f"derive_identifier_items(seed={seed})"))
    return items


# --- Rankings ------------------------------------------------------------------------------------

def first_rank(ranked: list[str], wanted: list[str]) -> int | None:
    """La posición (desde 1) del primer id de ``wanted`` en ``ranked``; None si no aparece."""
    wanted_ids = set(wanted)
    return next((rank for rank, id_ in enumerate(ranked, start=1) if id_ in wanted_ids), None)
