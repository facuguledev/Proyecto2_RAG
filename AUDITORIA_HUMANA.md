# Auditoría humana · cómo completarla

Las cuatro auditorías de la entrega las completó un modelo de lenguaje, no una persona (`decision_log.md`, sección 6). La consigna pide auditar a mano (Etapa 7a: "auditá a mano al menos 30 juicios o el 20%, lo que sea mayor"). Estas planillas tienen las mismas filas, sin lo que dijo el modelo. Se completan en Excel, LibreOffice o Google Sheets y se guardan como CSV con el mismo nombre (coma o punto y coma, los dos se leen).

**Reglas para las cuatro:**
- No mirar antes la planilla del modelo ni las notebooks con sus lecturas: la auditoría tiene que ser ciega.
- Se juzga contra el texto (la clave, la transcripción, la conversación), no contra lo que uno sepa de la ley.
- Si una fila es dudosa, se decide igual y se explica en `human_note`.
- Al terminar: correr `08_followups` (sección 5) o `python tools/recompute_tables.py`, que calculan el acuerdo con el auditor-modelo y vuelven a calibrar los jueces.

**Orden sugerido**, de lo que más pesa en el informe a lo que menos: el juez (obligatorio por la consigna), la memoria, los resúmenes y las reescrituras. Las cuatro juntas son unas 3 horas.

## 1 · El juez · `artifacts/full/judge_human/audit.csv` (82 filas, ~1,5 h)

Cada fila es una respuesta de las Etapas 4-6: la pregunta, la clave (`answer_key`, formas equivalentes separadas por `|`), el veredicto esperado si es de sí/no, `can_answer` y la respuesta. En `human_verdict` va **`correct`** o **`incorrect`**, según el tipo (`kind`):

| `kind` | Es correcta si... |
|---|---|
| `answerable` | transmite la clave, con esas u otras palabras que signifiquen lo mismo. Si se abstiene, es incorrecta. |
| `yes_no` | el veredicto (la primera palabra, "Sí" o "No") coincide con `expected_verdict` **y** transmite la clave. Abstenerse no es decir que no. |
| `unanswerable` | se abstiene (`can_answer` false) y no afirma ningún dato como si lo supiera. |
| `knowledge_update` | transmite la clave y corresponde a la situación del usuario a la fecha de la pregunta; mencionar la situación anterior no la hace incorrecta. |
| `preference` | contiene la clave tal cual (un número de norma, un nombre); una paráfrasis no cuenta. |

En todos los tipos: una respuesta que dice la clave pero también afirma algo que la contradice es incorrecta. `audit_key.jsonl`, al lado, guarda lo que no se muestra (etapa, brazo, veredictos del juez): no hace falta abrirlo.

## 2 · La memoria · `artifacts/full/memory/human_audit_records.csv` (76 filas, ~1 h)

Cada fila es un registro que guardó el extractor de la Etapa 6, con la transcripción de la sesión de la que salió (`session_transcript`). En `human_verdict`:
- **`bien`**: el registro dice algo que el usuario dijo en esa sesión, atribuido a quien corresponde;
- **`mal`**: dice algo que se dijo, pero mal (otra persona, otra fecha, un dato cambiado, la fecha de la sesión tomada como dato);
- **`nunca_se_dijo`**: registra algo que nadie dijo, incluidos los "no-hechos" ("el usuario no especificó...").

## 3 · Los resúmenes · `artifacts/full/results/05_summaries_human_audit.csv` (25 filas, ~20 min)

Cada fila es una actualización del resumen de la Etapa 5: el resumen anterior (`before`), lo que salió de la ventana (`evicted`) y el resumen nuevo (`after`). En `human_category`, una de las categorías de Wang et al.:
- **`correcta`**: conserva lo que importaba de `before` y de `evicted`;
- **`inventado`**: agrega algo que no está en ninguno de los dos;
- **`relacion`**: datos ciertos, mal atribuidos o mal conectados;
- **`faltante`**: perdió un dato del usuario o un número de la respuesta;
- **`corrupto`**: texto cortado o ilegible.

Si tiene más de una falla, la más grave (inventado > relación > faltante > corrupto) y las otras en la nota.

## 4 · Las reescrituras · `artifacts/full/results/05_rewrites_human_read.csv` (20 filas, ~15 min)

Cada fila es un turno de un guion: lo que preguntó el usuario (`user`), la reescritura del modelo (`rewrite`) y la oráculo, escrita a mano en la Etapa 2 (`oracle_rewrite`). En `human_category`:
- **`resuelve`**: se entiende sola y trae del historial lo que hace falta (la fecha, la ciudad, de qué se habla);
- **`agrega_ley`**: agrega contenido de la ley que nadie dijo (el efecto HyDE);
- **`mezcla`**: mezcla mal datos del historial;
- **`arrastra`**: arrastra el tema anterior a un cambio de tema;
- **`no_resuelve`**: copia el turno sin resolver la referencia;
- **`pierde_fecha`**: pierde la fecha que decide la respuesta;
- **`otra`**: otra falla (explicarla en la nota).
