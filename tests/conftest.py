import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Los módulos de src/ se importan sueltos (``import doc_schema``), no como paquete; los del curso
# (``environment``, ``llm``, ``store``) viven en la raíz. Ningún nombre se repite entre las dos carpetas.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
