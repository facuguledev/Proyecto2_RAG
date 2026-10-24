from cleaning import NORMALIZATION_RULES, Cleaner


def test_nfc_composes_decomposed_accents():
    c = Cleaner()
    assert c.normalize("información", "D1") == "información"
    assert c.docs["nfc"] == {"D1"}
    assert c.chars["nfc"] == 1


def test_soft_hyphen_and_zero_width_are_removed():
    c = Cleaner()
    assert c.normalize("arren­damiento​", "D1") == "arrendamiento"
    assert c.chars["invisibles"] == 2


def test_nbsp_and_typographic_spaces_become_a_plain_space():
    c = Cleaner()
    assert c.normalize("Artículo 2 y 3", "D1") == "Artículo 2 y 3"
    assert c.docs["odd_spaces"] == {"D1"} and not c.docs["whitespace"]


def test_superscripts_become_digits_but_ordinals_survive():
    c = Cleaner()
    assert c.normalize("100 m² del artículo 1.º y la 2.ª", "D1") == "100 m2 del artículo 1.º y la 2.ª"


def test_whitespace_collapses_inside_lines_and_keeps_paragraphs():
    c = Cleaner()
    assert c.normalize("  Uno   dos \n\n\n  tres  ", "D1") == "Uno dos\ntres"


def test_counts_documents_not_calls():
    c = Cleaner()
    c.normalize("a b", "D1")
    c.normalize("c d", "D1")
    c.normalize("e f", "D2")
    assert c.docs["odd_spaces"] == {"D1", "D2"}


def test_unchanged_text_records_nothing():
    c = Cleaner()
    c.normalize("Texto limpio.", "D1")
    assert not c.docs


def test_record_counts_extraction_rules():
    c = Cleaner()
    c.record("notes_out", "D1", 120)
    c.record("notes_out", "D2", 30)
    c.record("images", "D1", -16)
    assert c.chars["notes_out"] == 150
    assert c.chars["images"] == -16


def test_summary_lists_every_normalization_rule_even_without_hits():
    c = Cleaner()
    c.record("signature", "D1", 50)
    rows = {r["rule"]: r for r in c.summary()}
    assert rows["signature"] == {"rule": "signature", "documents": 1, "chars_removed": 50}
    for rule, _ in NORMALIZATION_RULES:
        assert rows[rule]["documents"] == 0
