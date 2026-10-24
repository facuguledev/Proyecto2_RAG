"""Cliente mínimo de la API de legislación consolidada del BOE, con caché en disco.

Documentación oficial: https://www.boe.es/datosabiertos/documentos/APIconsolidada.pdf

Cada respuesta se guarda en ``<raw_dir>/boe_cache/`` (fuera de git; ver ``raw_dir``) y
se registra en ``index.jsonl`` con la fecha de descarga y su sha256: esa es la versión fijada del
corpus. Una segunda corrida lee de la caché y nunca vuelve a la API salvo ``refresh=True``; con
``offline=True``, lo que no está en la caché es un error (así un build solo puede leer la foto).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests
from lxml import etree

BASE = "https://www.boe.es/datosabiertos/api"
ROOT = Path(__file__).resolve().parents[1]


def raw_dir() -> Path:
    """Carpeta de la foto cruda, fuera de git: ``data/raw/``, o ``P2_RAW`` si está definida (en Colab, Drive)."""
    return Path(os.environ.get("P2_RAW", ROOT / "data" / "raw"))


CACHE = raw_dir() / "boe_cache"
_session = requests.Session()


def get(path: str, accept: str = "application/json", params: dict | None = None,
        refresh: bool = False, cache: Path = CACHE, offline: bool = False) -> str:
    key = hashlib.sha256(f"{path}|{accept}|{json.dumps(params, sort_keys=True)}".encode()).hexdigest()[:24]
    f = cache / f"{key}.{'json' if 'json' in accept else 'xml'}"
    if f.exists() and not refresh:
        return f.read_text(encoding="utf-8")
    if offline:
        raise FileNotFoundError(f"no está en la foto: {path}")
    cache.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        r = _session.get(f"{BASE}/{path}", headers={"Accept": accept}, params=params, timeout=120)
        if r.status_code == 200:
            break
        if r.status_code in (400, 404):
            raise FileNotFoundError(f"{r.status_code} {path}")
        time.sleep(2 ** attempt)
    r.raise_for_status()
    # Bytes tal como llegaron: write_text en Windows pasa LF a CRLF y el archivo deja de coincidir
    # con su sha256 (le pasó a la foto del 2026-10-16; ver verify_cache).
    f.write_bytes(r.content)
    with (cache / "index.jsonl").open("a", encoding="utf-8", newline="\n") as idx:
        idx.write(json.dumps({"key": key, "path": path, "params": params, "accept": accept,
                              "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "sha256": hashlib.sha256(r.content).hexdigest()}, ensure_ascii=False) + "\n")
    time.sleep(0.3)  # API pública: sin ráfagas
    return f.read_text(encoding="utf-8")  # lo mismo que devolverá cada lectura posterior de la caché


def verify_cache(cache: Path = CACHE) -> dict:
    """Compara cada archivo de la caché con el sha256 registrado al descargarlo.

    El hash se calcula sobre el texto decodificado, no sobre los bytes: los archivos de la foto del
    2026-10-16 se guardaron con CRLF en Windows (el contenido es el mismo; los bytes, no).
    """
    rows = [json.loads(line) for line in (cache / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    missing, mismatched, crlf = [], [], 0
    for row in rows:
        f = cache / f"{row['key']}.{'json' if 'json' in row['accept'] else 'xml'}"
        if not f.exists():
            missing.append(row["path"])
            continue
        crlf += b"\r\n" in f.read_bytes()
        if hashlib.sha256(f.read_text(encoding="utf-8").encode("utf-8")).hexdigest() != row["sha256"]:
            mismatched.append(row["path"])
    return {"files": len(rows), "missing": missing, "mismatched": mismatched, "crlf_files": crlf,
            "index_sha256": hashlib.sha256((cache / "index.jsonl").read_bytes()).hexdigest()}


def search(query: str, limit: int = -1, **params) -> list[dict]:
    """Listado filtrado. ``query`` usa la sintaxis campo:valor de la API (materia@codigo, titulo, ...)."""
    q = {"query": {"query_string": {"query": query}}}
    data = json.loads(get("legislacion-consolidada", params={"query": json.dumps(q), "limit": limit, **params}))["data"]
    return data if isinstance(data, list) else []  # sin resultados, la API devuelve {} en vez de []


def materias() -> dict[str, str]:
    return json.loads(get("datos-auxiliares/materias"))["data"]


def metadatos(boe_id: str, offline: bool = False) -> dict:
    d = json.loads(get(f"legislacion-consolidada/id/{boe_id}/metadatos", offline=offline))["data"]
    return d[0] if isinstance(d, list) else d


def analisis(boe_id: str, offline: bool = False) -> dict:
    # En XML, no en JSON: la conversión a JSON de la API conserva solo la ÚLTIMA <materia>
    # (la LAU tiene 17 en XML y 1 en JSON), sin ningún error.
    xml = get(f"legislacion-consolidada/id/{boe_id}/analisis", accept="application/xml", offline=offline)
    root = etree.fromstring(xml.encode())

    def refs(tag: str) -> list[dict]:
        out = []
        for r in root.iter(tag):
            rel = r.find("relacion")
            out.append({"id_norma": r.findtext("id_norma"), "relacion": r.findtext("relacion"),
                        "relacion_codigo": rel.get("codigo") if rel is not None else None,
                        "texto": r.findtext("texto")})
        return out

    return {"materias": [{"codigo": m.get("codigo"), "texto": m.text} for m in root.iter("materia")],
            "notas": [n.text for n in root.iter("nota")],
            "anteriores": refs("anterior"), "posteriores": refs("posterior")}


def texto(boe_id: str, offline: bool = False) -> str:
    """Texto consolidado completo, con todas las versiones de cada bloque, en una sola llamada."""
    return get(f"legislacion-consolidada/id/{boe_id}/texto", accept="application/xml", offline=offline)


def parse_texto(xml: str) -> list[dict]:
    """Bloques del texto consolidado, cada uno con sus versiones fechadas (lo usa la auditoría de la Etapa 0).

    Separa el articulado (``paras``) de las notas de modificación (``<blockquote>``, en ``notes``):
    un loader ingenuo las mezcla con el texto, y concatenar todas las versiones de un bloque
    repite el artículo con reglas contradictorias (la LAU pasa de 39.550 a 107.051 tokens).

    Límite conocido: trata TODO ``<blockquote>`` como nota, y ``class="sangrado"`` es el texto citado
    por una ley modificadora. La Etapa 1 extrae con ``corpus.py``, que lo conserva como texto.
    """
    root = etree.fromstring(xml.encode("utf-8"))
    blocks = []
    for b in root.iter("bloque"):
        versions = []
        for v in b.findall("version"):
            paras, notes, n_table, n_img = [], [], 0, 0
            for el in v:
                if el.tag == "blockquote":
                    notes.append(" ".join("".join(el.itertext()).split()))
                    continue
                n_table += el.tag == "table"
                n_img += len(el.findall(".//img")) + (el.tag == "img")
                txt = " ".join("".join(el.itertext()).split())
                if txt:
                    paras.append((el.get("class") or el.tag, txt))
            versions.append({"id_norma": v.get("id_norma"), "fecha_publicacion": v.get("fecha_publicacion"),
                             "fecha_vigencia": v.get("fecha_vigencia"), "paras": paras, "notes": notes,
                             "n_table": n_table, "n_img": n_img})
        blocks.append({"id": b.get("id"), "tipo": b.get("tipo"), "titulo": b.get("titulo"),
                       "fecha_caducidad": b.get("fecha_caducidad"), "versions": versions})
    return blocks
