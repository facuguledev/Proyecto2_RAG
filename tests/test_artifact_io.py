"""Archivos de artefactos: JSONL, hashes y manifiestos."""
import hashlib
from datetime import date

import pytest
from pydantic import BaseModel

from artifact_io import check_manifest, read_jsonl, sha256_file, write_jsonl, write_manifest


class Row(BaseModel):
    day: date
    name: str


def test_write_jsonl_writes_lf_endings_and_keeps_key_order(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"b": 1, "a": 2}, {"b": 3, "a": 4}])
    # Se lee en bytes: en modo texto Windows convertiría \r\n en \n y el test no vería nada.
    assert path.read_bytes() == b'{"b": 1, "a": 2}\n{"b": 3, "a": 4}\n'


def test_write_jsonl_creates_parent_dirs_and_counts_rows(tmp_path):
    path = tmp_path / "artifacts" / "full" / "rows.jsonl"
    rows = ({"n": n} for n in range(3))  # un generador: la cuenta no puede salir de len()
    assert write_jsonl(path, rows) == 3
    assert path.exists()


def test_jsonl_round_trip_keeps_accents_and_dumps_models_as_json(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"title": "Artículo 1. Información"}, Row(day=date(2019, 3, 6), name="ñandú")])
    assert "Artículo 1. Información" in path.read_text(encoding="utf-8")  # sin escapes í
    assert read_jsonl(path) == [{"title": "Artículo 1. Información"}, {"day": "2019-03-06", "name": "ñandú"}]


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a": 1}\n\n{"a": 2}\n\n')  # a mano: un editor puede dejar líneas en blanco
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_sha256_file_matches_hashlib(tmp_path):
    data = "Artículo 1\n".encode("utf-8") * 100_000  # ~1,2 MB: varios bloques de lectura
    path = tmp_path / "big.txt"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_write_manifest_is_indented_json_with_lf_and_a_final_newline(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"corpus": "BOE", "n_docs": 2, "nota": "ñandú"})
    assert path.read_bytes().decode("utf-8") == '{\n "corpus": "BOE",\n "n_docs": 2,\n "nota": "ñandú"\n}\n'


def test_check_manifest_returns_the_manifest_when_it_matches(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"corpus_hash": "abc", "n_docs": 2, "model": "bge-m3"})
    manifest = check_manifest(path, {"corpus_hash": "abc", "n_docs": 2})
    # devuelve el manifiesto entero, no solo las claves esperadas
    assert manifest == {"corpus_hash": "abc", "n_docs": 2, "model": "bge-m3"}


def test_check_manifest_raises_listing_each_mismatch(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"corpus_hash": "abc", "n_docs": 2})
    with pytest.raises(ValueError) as error:
        check_manifest(path, {"corpus_hash": "xyz", "n_docs": 2, "model": "bge-m3"})
    message = str(error.value)
    assert "corpus_hash: manifest=abc esperado=xyz" in message
    assert "model: manifest=None esperado=bge-m3" in message  # una clave que falta también es un desacuerdo
    assert "n_docs" not in message  # lo que coincide no se lista
