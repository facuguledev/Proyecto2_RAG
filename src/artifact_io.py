"""Lectura y escritura de artefactos: JSONL, hashes de archivos y manifiestos."""
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel


def _open_for_writing(path: Path) -> TextIO:
    """Abre ``path`` para escribir texto UTF-8, creando las carpetas que falten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" fuerza fin de línea LF. En Windows el modo texto convierte \n en \r\n, y
    # entonces el mismo archivo da otro hash que en Colab (ya nos pasó con la caché cruda).
    return open(path, "w", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict | BaseModel]) -> int:
    """Escribe una fila JSON por línea (los modelos de pydantic, como JSON) y devuelve cuántas escribió."""
    count = 0
    with _open_for_writing(path) as f:
        for row in rows:
            data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: Path) -> str:
    """Hash del contenido del archivo; ``file_digest`` lo lee de a bloques, sin cargarlo entero."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_manifest(path: Path, manifest: dict) -> None:
    with _open_for_writing(path) as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")


def check_manifest(path: Path, expected: dict) -> dict:
    """Carga el manifiesto y exige que coincida con ``expected`` (puede traer claves de más).

    Una etapa que carga un artefacto tiene que negarse si lo armaron desde otro corpus:
    se corta con un error, nunca se avisa y se sigue.
    """
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    mismatches = []
    for key, value in expected.items():
        if key not in manifest or manifest[key] != value:
            mismatches.append(f"{key}: manifest={manifest.get(key)} esperado={value}")
    if mismatches:
        raise ValueError(f"{path} no coincide con lo esperado:\n" + "\n".join(mismatches))
    return manifest
