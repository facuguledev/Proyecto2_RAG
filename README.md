# Proyecto 2 · RAG con memoria sobre legislación española de vivienda

**IA Generativa Avanzada y Sistemas Multi-Agente** — Licenciatura en Tecnología Digital, Universidad Torcuato Di Tella.

Un asistente de recuperación aumentada (RAG) **con memoria** sobre 177 normas consolidadas del BOE sobre vivienda y alquiler, con todas las versiones fechadas de cada artículo. Mide, con modelos abiertos (`qwen3:8b` en Ollama, `bge-m3`, un reranker multilingüe), cómo repartir una ventana de contexto finita entre lo que dice el corpus y lo que se dijo en la conversación. Es una entrega del *Retrieval & Memory Lab* del curso: el enunciado está en [assignment.md](assignment.md).

*Basado en datos de la Agencia Estatal Boletín Oficial del Estado. Los textos consolidados tienen carácter meramente informativo: solo la publicación en el diario oficial tiene validez jurídica.*

## Por dónde empezar

| Para | Leé |
|---|---|
| los resultados y su lectura | [informe/informe.md](informe/informe.md) |
| cada decisión, con su razón y la alternativa descartada | [decision_log.md](decision_log.md) |
| el corpus: fuente, versión fijada, licencia, defectos conocidos | [corpus_card.md](corpus_card.md) |
| el conjunto de evaluación (congelado) | [eval/](eval/), con `FROZEN.json` |
| completar la auditoría humana pendiente | [AUDITORIA_HUMANA.md](AUDITORIA_HUMANA.md) |

## Las etapas

Cada etapa es un `.py` con marcadores `# %%`; su notebook se genera con `python tools/py_to_notebook.py`. En git van los notebooks limpios; los ejecutados, con sus salidas, van en el zip de la entrega.

| Etapa | Archivo | Qué hace |
|---|---|---|
| 0 | `00_corpus_audit`, `00_closed_book` | auditoría del corpus y chequeo a libro cerrado |
| 1 | `01_documents` | de la API del BOE a `documents.jsonl`: una sección por versión de cada artículo, con su vigencia |
| 2 | `02_eval_set` | 40 preguntas, 10 guiones y 4 escenarios de memoria; contaminación; congelado |
| 3 | `03_retrieval` | ocho filas acumulativas (BM25, denso, prefijo, RRF, reranker, padres, filtros) con presupuesto igualado |
| 4 | `04_grounded_answers` | libro cerrado, recuperación y oráculo; la tabla 2×2; la guarda de contexto |
| 5 | `05_conversation` | con qué buscar (cuatro consultas) y qué historial conservar (ventanas y resumen) |
| 6 | `06_memory` | memoria entre sesiones: reproducción contra almacén con invalidación |
| 7 | `07_second_read` | dos jueces LLM calibrados contra una auditoría |
| 8 | `08_followups` | seguimiento de la revisión de la entrega: filtro de fecha en la generación, oráculo con tokens igualados, posición del pasaje, resumidor v2, auditoría humana |

La lógica vive en `src/` (un módulo por tema: `corpus`, `cleaning`, `chunking`, `retrieval`, `grounded`, `conversation`, `memory`, `judge`, `followups`, ...); la plomería del curso (`environment.py`, `llm.py`, `store.py`, `schema.py`, `console.py`) en la raíz.

## Correrlo

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-group.txt
python verify_setup.py
python -m pytest -q                                     # sin GPU ni modelos
python tools/run_stages.py --from 01 --to 08            # necesita Ollama con qwen3:8b y una GPU
```

- **La fuente cruda no está en git** (`data/raw/boe_cache/`, assignment §6.1). La Etapa 1 la lee de esa caché, fijada por sha256 en su índice; `data/subset_v1.json` dice qué normas, con su fecha de actualización. La API del BOE cambia seguido: una descarga nueva no reproduce la foto del 2026-10-24 (`corpus_card.md`).
- **En Colab**, la primera celda de cada notebook clona este repo e instala las dependencias. Todas las corridas de la entrega se hicieron en una sola máquina local (RTX 5070 Laptop, 8 GB, Ollama 0.34.4); en Colab no se probó.

## Recalcular las tablas sin el corpus

```bash
python tools/recompute_tables.py            # desde la raíz del repo o del zip
python tools/make_human_audits.py           # las planillas ciegas para la auditoría humana
```

`recompute_tables.py` usa solo la biblioteca estándar y lee `eval/` y `artifacts/full/`: re-puntúa cada respuesta guardada, verifica los hashes del eval set congelado y rehace las tablas de las Etapas 3-8.

## Créditos y licencia

La plomería, el enunciado y la Etapa 0 resuelta (`notebooks/00_corpus_audit(python).ipynb`, sobre python-docs-es) son del *Retrieval & Memory Lab* del curso (https://github.com/solidgoldmagickarp/projecto_rag), MIT License, Copyright (c) 2026 Francisco Traversaro: ver [LICENSE](LICENSE). Los datos son del BOE, reutilizados bajo la Ley 37/2007 con la atribución de arriba.
