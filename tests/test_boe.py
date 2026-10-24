from pathlib import Path

import boe


def test_raw_dir_is_data_raw_by_default(monkeypatch):
    monkeypatch.delenv("P2_RAW", raising=False)
    assert boe.raw_dir() == Path(boe.__file__).resolve().parents[1] / "data" / "raw"


def test_raw_dir_follows_p2_raw(monkeypatch, tmp_path):
    monkeypatch.setenv("P2_RAW", str(tmp_path))  # en Colab apunta a Drive
    assert boe.raw_dir() == tmp_path
