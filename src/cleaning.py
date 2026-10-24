"""Normalización del texto, con un contador por regla (Etapa 1, pasos 3 y 8).

Cada regla tiene nombre. El Cleaner cuenta, por regla, cuántas normas cambió y cuántos caracteres
eliminó (negativo = agregó): así "qué cambió la limpieza" es una tabla y no una impresión. Las reglas
de extracción (notas, firma, tablas, ...) viven en corpus.py y se anotan acá con `record`.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

# Guion blando (U+00AD), ancho cero, unión de palabras, BOM y caracteres de control.
INVISIBLE = re.compile("[­​‌‍⁠﻿\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# NBSP, espacios tipográficos (U+2000-U+200A: EN, EM, fino...) y otros que no son el espacio común.
ODD_SPACES = re.compile("[  -   　\t\r]")
# Compatibilidad dirigida: superíndices y ligaduras. NFKC haría esto, pero también cambia "1.º" por
# "1.o", y el corpus tiene 2.910 ordinales.
COMPAT = str.maketrans({"²": "2", "³": "3", "¹": "1", "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})


def nfc(text: str) -> str:
    # Un acento descompuesto ("o" + U+0301) es otra cadena para BM25 y para la contención.
    return unicodedata.normalize("NFC", text)


def invisibles(text: str) -> str:
    return INVISIBLE.sub("", text)


def odd_spaces(text: str) -> str:
    return ODD_SPACES.sub(" ", text)


def compat(text: str) -> str:
    return text.translate(COMPAT)


def whitespace(text: str) -> str:
    # Colapsa los espacios DENTRO de cada línea y conserva una línea por párrafo: el corte en
    # secciones y chunks usa esa estructura.
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


# En orden: NFC primero, para que las reglas siguientes vean caracteres ya compuestos.
NORMALIZATION_RULES = [
    ("nfc", nfc),
    ("invisibles", invisibles),
    ("odd_spaces", odd_spaces),
    ("compat", compat),
    ("whitespace", whitespace),
]


class Cleaner:
    def __init__(self) -> None:
        self.docs: defaultdict[str, set[str]] = defaultdict(set)  # regla -> normas que cambió
        self.chars: Counter[str] = Counter()  # regla -> caracteres eliminados (negativo = agregados)

    def record(self, rule: str, doc_id: str, chars_removed: int) -> None:
        self.docs[rule].add(doc_id)
        self.chars[rule] += chars_removed

    def apply(self, rule: str, fn, text: str, doc_id: str) -> str:
        out = fn(text)
        if out != text:
            self.record(rule, doc_id, len(text) - len(out))
        return out

    def normalize(self, text: str, doc_id: str) -> str:
        for rule, fn in NORMALIZATION_RULES:
            text = self.apply(rule, fn, text, doc_id)
        return text

    def summary(self) -> list[dict]:
        """Una fila por regla, incluidas las de normalización que no cambiaron nada."""
        rules = list(dict.fromkeys([*self.docs, *(rule for rule, _ in NORMALIZATION_RULES)]))
        return [{"rule": r, "documents": len(self.docs.get(r, ())), "chars_removed": self.chars.get(r, 0)}
                for r in rules]
