"""Nuestros módulos (src/) y los del curso (raíz) conviven en un mismo proceso."""
import artifact_io
import doc_schema
import evalset
import llm  # del curso: importa `store`; si un store nuestro lo tapara, fallaría acá
import schema as course_schema
import store


def test_llm_uses_the_course_store():
    assert hasattr(store, "ResultsLog") and llm.append_jsonl is store.append_jsonl


def test_our_modules_are_ours():
    assert hasattr(doc_schema.Section, "in_force") and hasattr(artifact_io, "check_manifest")


def test_vocabularies_match_the_course():
    # Las tablas del curso agrupan por estas etiquetas: se escriben igual ("out_of_corpus", no "out-of-corpus").
    assert evalset.CATEGORIES == course_schema.CATEGORIES
    assert evalset.DEPENDENCIES == course_schema.DEPENDENCIES
    assert evalset.ABILITIES == course_schema.ABILITIES
