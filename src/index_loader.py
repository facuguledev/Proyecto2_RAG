"""El índice de la Etapa 3 y su mejor configuración, cargados para las Etapas 4-6.

La Etapa 3 arma el índice dentro de ``03_retrieval.py`` y deja su elección en ``ART/index/best_config.json``
(con la clave del índice sobre el que se eligió). Este módulo lo carga con las mismas especificaciones y
**verifica esa clave**: si la Etapa 3 cambió el corte, el encoder o el reranker y no se volvió a correr,
las Etapas 4-6 fallan en lugar de responder con un índice distinto del que se midió.

Las constantes repiten las de ``03_retrieval.py``; la verificación de la clave es lo que impide que se
separen sin que nadie lo note.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

import chunking
import retrieval
from chunking import Unit, searchable_text
from doc_schema import Document, documents_hash
from lexical import Bm25Index
from retrieval import Index, RetrievalConfig

GENERATOR_TOKENIZER, GENERATOR_REVISION = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
EMBEDDER, EMBEDDER_REVISION = "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER, RERANKER_REVISION = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825"
MAX_SEQ = 512
GPU_NEEDED = 3.5e9  # bytes: bge-m3 en fp32 (~2,3 GB), el reranker (~0,5 GB) y margen para un lote


def unload_ollama_models() -> list[str]:
    """Descarga todos los modelos que el servidor de Ollama tiene en memoria; devuelve cuáles eran."""
    import requests

    import llm

    try:
        loaded = [m.get("name") or m.get("model") for m in
                  requests.get(f"{llm.HOST}/api/ps", timeout=10).json().get("models", [])]
    except (requests.exceptions.RequestException, ValueError):
        return []
    for name in loaded:
        llm.unload(name)
    return loaded


def wait_for_vram(needed: float = GPU_NEEDED, timeout_s: float = 60) -> None:
    """Espera a que la GPU tenga ``needed`` bytes libres antes de cargar modelos en ella.

    ``llm.unload`` le pide a Ollama que descargue el generador y vuelve enseguida, antes de que la VRAM
    se libere: subir el embedder justo después da un CUDA out of memory (le pasó a la Etapa 6 el
    2026-10-24). Si no se libera a tiempo, falla con un mensaje que dice qué la ocupa.
    """
    import time

    import torch

    if not torch.cuda.is_available():
        return
    deadline, unloaded = time.time() + timeout_s, False
    while True:
        free, total = torch.cuda.mem_get_info()
        if free >= min(needed, 0.9 * total):
            return
        if not unloaded:  # cualquier modelo que Ollama tenga cargado, no solo el generador (p. ej. el juez de la 7a)
            unload_ollama_models()
            unloaded = True
        if time.time() > deadline:
            raise RuntimeError(f"la GPU tiene {free / 1e9:.1f} GB libres y hacen falta {needed / 1e9:.1f}: ¿el "
                               "generador sigue cargado? (llm.unload, o nvidia-smi para ver quién la usa)")
        time.sleep(0.5)


def index_spec(docs: list[Document], root: Path) -> dict:
    """El ``INDEX_SPEC`` de la Etapa 3: lo que determina los hijos y los padres."""
    import store

    return {
        "corpus_sha": documents_hash(docs),
        "indexed_kinds": list(chunking.INDEXED_KINDS),
        "child_max": chunking.CHILD_MAX, "child_overlap": chunking.CHILD_OVERLAP,
        "parent_max": chunking.PARENT_MAX, "prefix_max": chunking.PREFIX_MAX, "template": chunking.TEMPLATE,
        "tokenizers": {"child": [RERANKER, RERANKER_REVISION], "parent": [GENERATOR_TOKENIZER, GENERATOR_REVISION]},
        "code": store.sha256_text((root / "src" / "chunking.py").read_text(encoding="utf-8")),
    }


@dataclass
class Loaded:
    index: Index
    key: str  # la clave del índice, la misma que calcula la Etapa 3
    config: RetrievalConfig  # la mejor configuración de la Etapa 3
    best: dict  # best_config.json entero (la fila, la regla, los aciertos por fila)
    encoder_spec: dict

    @property
    def units(self) -> dict[str, Unit]:
        return {unit.unit_id: unit for unit in self.index.children} | (self.index.parents or {})

    def _models(self) -> list:
        return [m for m in (self.index.embedder, self.index.reranker) if m is not None]

    def release_gpu(self) -> None:
        """Pasa el embedder y el reranker a CPU, en fp32, si la GPU no alcanza para ellos y el generador.

        Con 8 GB no conviven con ``qwen3:8b`` (~6,2 GB): las etapas recuperan en lote antes de generar, y las
        consultas que quedan van en CPU. En fp32, porque fp16 en CPU es lento y le faltan operaciones. En
        una T4 (15 GB) entran todos juntos y no se mueve nada: la recuperación queda en el mismo dispositivo
        y la misma precisión que en la Etapa 3.
        """
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory >= 12e9:
            return
        for model in self._models():
            model.to("cpu").float()
        torch.cuda.empty_cache()

    def to_gpu(self) -> None:
        """El inverso de ``release_gpu``: vuelven a la GPU con la precisión del encoder (fp16 si lo es)."""
        import torch

        if not torch.cuda.is_available():
            return
        wait_for_vram()
        for model in self._models():
            model.to("cuda")
        if self.encoder_spec["dtype"] == "float16" and self.index.embedder is not None:
            self.index.embedder.half()


def load(art: Path, docs: list[Document], root: Path, eval_key: str, device: str | None = None) -> Loaded:
    """El índice de la Etapa 3 con todas sus piezas, y su mejor configuración.

    Si los hijos, los padres o los embeddings no están en ``art`` se construyen (es lo mismo que haría la
    Etapa 3), pero ``best_config.json`` tiene que existir: la mejor configuración la elige la Etapa 3.
    """
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    import environment
    import store

    index_dir = Path(art) / "index"
    best_path = index_dir / "best_config.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"{best_path} no existe: corré antes la Etapa 3 (03_retrieval.py) en este perfil")
    best = store.read_json(best_path)
    if best["eval"] != eval_key:
        raise ValueError(f"la Etapa 3 eligió su configuración contra el eval set {best['eval']}, no {eval_key}")

    def backend(repo: str, revision: str):
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision).backend_tokenizer
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer

    spec = index_spec(docs, root)

    def build(path):
        parents, children = chunking.build_units(
            docs, backend(RERANKER, RERANKER_REVISION),
            Tokenizer.from_pretrained(GENERATOR_TOKENIZER, revision=GENERATOR_REVISION))
        store.write_jsonl(index_dir / "parents.jsonl", [asdict(unit) for unit in parents])
        store.write_manifest(index_dir / "parents.jsonl", spec)
        store.write_jsonl(path, [asdict(unit) for unit in children])

    store.build_or_load(index_dir / "chunks.jsonl", spec, build)
    store.check_manifest(index_dir / "parents.jsonl", spec)
    children = [Unit(**row) for row in store.read_jsonl(index_dir / "chunks.jsonl")]
    parents = {row["unit_id"]: Unit(**row) for row in store.read_jsonl(index_dir / "parents.jsonl")}

    device = device or environment.device()
    if device == "cuda":
        wait_for_vram()
    fp16 = device == "cuda" and environment.on_colab()  # la misma regla que la Etapa 3 (decision_log, 2026-10-23)
    embedder = SentenceTransformer(EMBEDDER, revision=EMBEDDER_REVISION, device=device)
    if fp16:
        embedder.half()
    embedder.max_seq_length = MAX_SEQ
    encoder_spec = {"model": EMBEDDER, "revision": EMBEDDER_REVISION, "max_seq_length": MAX_SEQ,
                    "dtype": "float16" if fp16 else "float32", "normalize": True, "instruction_prefix": None}

    def embed(name: str, texts: list[str]) -> np.ndarray:
        unique_texts = list(dict.fromkeys(texts))
        ids = [store.sha256_text(text)[:16] for text in unique_texts]
        matrix = store.cached_matrix(
            name, encoder_spec, ids, unique_texts,
            lambda batch: embedder.encode(batch, batch_size=32, normalize_embeddings=True, convert_to_numpy=True),
            art_dir=art)
        assert np.isfinite(matrix).all(), f"{name}: embeddings no finitos"
        row = {text: i for i, text in enumerate(unique_texts)}
        return np.asarray(matrix[[row[text] for text in texts]], dtype=np.float32)

    raw = embed("children_raw", [searchable_text(u, False) for u in children])
    prefixed = embed("children_prefixed", [searchable_text(u, True) for u in children])
    ids = [u.unit_id for u in children]
    reranker = CrossEncoder(RERANKER, revision=RERANKER_REVISION, max_length=512, device=device,
                            activation_fn=torch.nn.Identity())
    reranker_spec = {"model": RERANKER, "revision": RERANKER_REVISION, "max_length": 512, "scores": "logits"}
    index = Index(children=children, parents=parents, raw_matrix=raw, prefixed_matrix=prefixed,
                  bm25_raw=Bm25Index(ids, [searchable_text(u, False) for u in children]),
                  bm25_prefixed=Bm25Index(ids, [searchable_text(u, True) for u in children]),
                  embedder=embedder, reranker=reranker, metadata=retrieval.unit_metadata(children, docs))
    key = store.sha256_json({"units": store.read_manifest(index_dir / "chunks.jsonl")["key"],
                             "encoder": encoder_spec, "reranker": reranker_spec})[:16]
    if key != best["index"]:
        raise ValueError(f"el índice cargado ({key}) no es el que evaluó la Etapa 3 ({best['index']}): cambió el "
                         "corte, el encoder o el reranker. Volvé a correr 03_retrieval.py en este perfil")
    return Loaded(index=index, key=key, config=RetrievalConfig(**best["config"]), best=best,
                  encoder_spec=encoder_spec)


def ranked_units(loaded: Loaded, query: str, cfg: RetrievalConfig | None = None,
                 filters: retrieval.Filters | None = None) -> tuple[list[Unit], retrieval.Retrieval]:
    """Las unidades rankeadas (hijos o padres, según la configuración) y la recuperación entera.

    Las Etapas 5 y 6 entregan con presupuestos que cambian por turno: recuperan una vez y llaman a
    ``retrieval.deliver(units, budget)`` con cada presupuesto.
    """
    cfg = cfg or loaded.config
    result = retrieval.retrieve(query, cfg, loaded.index, filters)
    pool = loaded.index.parents if cfg.parent else {u.unit_id: u for u in loaded.index.children}
    return [pool[unit_id] for unit_id in result.ranked], result


def with_budget(cfg: RetrievalConfig, budget: int) -> RetrievalConfig:
    return replace(cfg, budget_tokens=budget)
