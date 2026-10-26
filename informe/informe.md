# Retrieval & Memory Lab · Informe

**Proyecto 2 · IA Generativa Avanzada · UTDT.** Asistente de RAG con memoria sobre legislación española de vivienda y alquiler (BOE, legislación consolidada).

*Basado en datos de la Agencia Estatal Boletín Oficial del Estado. Los textos consolidados tienen carácter meramente informativo: solo la publicación en el diario oficial tiene validez jurídica.*

**Cómo leerlo.** Cada sección resume una etapa y remite a su notebook (`NN_*.ipynb`, con salidas en el zip), donde está cada tabla con su lectura al lado. Las decisiones, con su razón y la alternativa descartada, están en `decision_log.md`. Salvo que se diga otra cosa, los números son sobre `test`, y cada comparación se lee contra su cota de ruido: con n = 30, una fila tiene unos ±12 puntos.

**El código** está en el repo público https://github.com/facurie/Proyecto2_RAG: la primera celda de cada notebook lo clona e instala las dependencias en Colab.

**Recalcular las tablas sin el corpus.** `python tools/recompute_tables.py`, desde la raíz del zip, recalcula las tablas de las Etapas 3-7 a partir de `eval/` y `artifacts/full/` solamente, con la biblioteca estándar: re-puntúa cada respuesta guardada con las mismas reglas y verifica los hashes del eval set congelado. Reproduce las tablas principales (aciertos por brazo, comparaciones con su test de signos y la calibración del juez); la primaria de la Etapa 3 necesita además `results/03_retrieval_metrics.jsonl`, que escribe la Etapa 8.

**Revisión posterior a la entrega (2026-10-26).** Una revisión del trabajo encontró siete puntos en los que una afirmación se apoyaba en algo que no se midió, o que se midió con un instrumento que no correspondía. Los más importantes: la auditoría del juez y la de la memoria las hizo un modelo y no una persona (secciones 8 y 10), y la configuración elegida entrega versiones de artículos que no rigen en la fecha de la pregunta (secciones 4 y 5). Las correcciones están en el texto, cada una en su sección, y resumidas en la sección 11. Ninguna tabla de las Etapas 3-7 cambió de valor; la de la Etapa 5B ahora compara todos los brazos sobre los mismos turnos.

---

## 1 · Decisiones de diseño

Las que más pesan en los resultados (el porqué completo, en `decision_log.md`):

- **Corpus vivo, foto fijada.** El BOE consolidado guarda cada versión de cada artículo con su vigencia. Eso da las mejores preguntas temporales y también el riesgo principal: la fuente cambia mientras se trabaja. La foto se fijó por hash (`corpus_sha`), y el eval set se congeló contra ella.
- **El gold es a nivel de sección (versión), no de documento.** Una pregunta sin fecha tiene como gold la versión vigente el día de la foto; una con fecha, la de ese día.
- **Una sola vara en todas las etapas.** La entrega por presupuesto de tokens (`retrieval.deliver`), la definición de "le llegó la respuesta" (`evaluation.delivered_metrics`: un pasaje del gold que contiene la clave) y el puntaje sin juez (`grounded.score`) se escriben una vez y los usan las Etapas 3-6.
- **Errores como categoría propia.** Una llamada que falla (overflow, parseo, truncamiento, o una respuesta cortada por la gramática en su tope) nunca cuenta como abstención ni como respuesta.
- **La abstención es el booleano del schema**, y el veredicto de sí/no se lee de la primera palabra de `answer`, solo en las preguntas de sí/no y solo si `can_answer`.
- **Presupuestos verificados contra el servidor.** `num_ctx = 8.192` y `num_predict = 768` en cada llamada; nuestra cuenta con la plantilla se compara con `prompt_eval_count` (diferencia constante de 4 tokens).
- **El ruido se acota por la unidad independiente:** por pregunta en las Etapas 3-4, por guion en la 5 y por escenario en la 6 (test de signos y bootstrap por grupo).
- **Las corridas no se paralelizan.** Una GPU, un `num_ctx` por experimento, y la salida depende del estado de la caché de prompts de Ollama: cada brazo arranca con el modelo descargado y en un orden fijo.

## 2 · El corpus (Etapas 0 y 1)

La ficha completa está al final de esta sección. En corto:

- **177 normas de vivienda y alquiler** (estatales y autonómicas), elegidas por una regla de densidad temática sobre 10 materias del BOE. Son 8.686 secciones (versiones de artículos), 5.766 vigentes el día de la foto y 4,78 M tokens del generador (2,89 M vigentes). No entra en la ventana: 47 de las 135 normas vigentes entran solas en 7.424 tokens.
- **La foto es la del 2026-10-24.** La original, del 2026-10-16, se perdió entre máquinas, y no se podía volver a bajar: en ocho días se habían actualizado 6 de las 177 normas, entre ellas la Ley 12/2023 y el Plan Estatal de Vivienda. Se rehízo la Etapa 0 con la misma regla (salen las mismas 177 normas) y se volvió a validar el eval set: falló un solo gold, porque el art. 10 de la LPH tiene una versión nueva vigente desde el 2026-10-23, con la misma respuesta. Se corrigió y se congeló de nuevo, antes de cualquier resultado (`decision_log.md`, sección 5).
- **Datos sucios que no tiran error:** la conversión a JSON de la API pierde todas las materias menos la última; el texto citado de las leyes modificadoras parecía una nota editorial; la versión vigente de un artículo no siempre es la última; 434 secciones tienen un gemelo casi literal en otra norma.

### Preguntas de la Etapa 0

- **¿Qué propiedades no cumple el corpus, cuánto cuesta la octava y en qué etapa se paga cada una?** De las siete, fallan tres:
  - **el texto no está limpio** para un loader ingenuo (versiones superpuestas, notas de modificación dentro del texto, tablas, el JSON que pierde materias). Lo paga la Etapa 1;
  - **los metadatos mienten a veces** (una ley balear figura vigente y su texto dice derogada; las resoluciones anuales reemplazadas figuran vigentes; `fecha_publicacion` es la de la copia en el BOE). Lo paga la Etapa 1 con el mapeo corregido, y lo heredaría cualquier filtro por fecha o región (Etapas 3 y 6);
  - **cambia mucho:** 18% de los preceptos tiene más de una versión, y la LAU cambió tres veces en octubre de 2026. Se paga con la foto fijada (Etapa 1) y se aprovecha en las preguntas "a una fecha" y en las actualizaciones de memoria (Etapas 2 y 6).
  
  Las otras cuatro se cumplen: la respuesta está (con huecos conocidos, la LEC y la ley catalana, que dan las preguntas fuera del corpus), hay estructura para cortar (bloques tipados, mediana de 230 tokens por precepto), los ids son estables y se puede armar un eval set. **La octava, la similitud, es alta:** 7% de los artículos tiene un gemelo casi literal en otra norma, y las versiones consecutivas de un mismo artículo tienen una mediana de Jaccard de 0,82. La pagan la Etapa 1 (versiones, no copias), la 2 (gold con fecha o como conjunto) y la 3 (el prefijo con la vigencia).
- **¿Entraría el subconjunto en la ventana? Y si entrara, ¿para qué construir recuperación?** No entra: son 2,89 M tokens vigentes (4,78 M con todas las versiones) contra 7.424 de prompt. 47 de las 135 normas vigentes entran solas, y para esas "si entra, pegalo" sería posible. Aun así, la recuperación decide qué versión llega: una pregunta "a una fecha" necesita la versión de ese día, y la Etapa 4 muestra que, con dos versiones del mismo artículo a la vista, el modelo aplica la equivocada (md02, md05). La fila "norma entera" no se corrió.
- **¿Cuánto del corpus pudo haber leído el generador, y cómo se distingue recuperación de recuerdo?** El 11% de los preceptos nació después de mayo de 2025 (el corte de Qwen3), y 18 normas se publicaron después: esas no las pudo leer. Lo anterior, aunque lo haya leído, no lo recuerda: a libro cerrado acierta 1 de 5 en el chequeo de la Etapa 0 y 3 de 26 en la fila de libro cerrado de la Etapa 4, con alucinaciones seguras. Esa fila es la que separa las dos cosas: lo que se acierta sin pasajes es recuerdo, y la diferencia con la fila de recuperación (+43 puntos) es lo que aporta recuperar.
- **¿Qué hace que dos documentos se confundan, y conviene conservarlo?** Las versiones de un mismo artículo (casi el mismo texto, distinta respuesta), los decretos-ley derogados y re-decretados (RDL 21/2018 ↔ 7/2019, 26 ↔ 29/2026), el decreto-ley y la ley que lo convalidó, y las 13 resoluciones anuales con la misma plantilla. Se conserva, porque es lo que el corpus pone a prueba (preguntas a una fecha, filtro de fecha, actualizaciones de memoria): no se borra ni se fusiona nada, y el gold apunta a la versión exacta o al conjunto de gemelos vigentes.

### Preguntas de la Etapa 1

- **¿Qué cambió la limpieza, en números, y qué regla cambió más?** De 5,14 M tokens (lo que daría un loader ingenuo) a 4,78 M en `documents.jsonl`, y 2,89 M vigentes. La regla que más cambió es la que saca las notas de modificación a `Section.notes`: 848.500 caracteres en 116 normas. Le siguen la firma (33.159 caracteres en 172 normas) y la fórmula de promulgación (17.580 en 77); las tablas linealizadas agregan 89.413. A nivel carácter el XML es limpio: NFC toca 2 normas y los caracteres invisibles, 5. La tabla por regla está en la ficha.
- **Un defecto que un loader ingenuo habría dejado pasar sin error, y cómo se encontró.** La conversión a JSON del análisis de la API conserva solo la última materia: 436 de 438 normas pierden materias, y la LAU queda con 1 de 17. No hay error ni advertencia; se ve al comparar las dos respuestas de la API (en XML la LAU tiene sus 17 materias), y por eso el análisis se lee en XML. Otro: el texto que cita una ley modificadora viene en un `<blockquote>`, igual que las notas, y descartarlo con ellas dejaba vacío el "Artículo único. Se modifica la Ley X, que queda redactada así:".
- **¿Cómo se definieron documento y sección, y qué se rompe si una etapa posterior no está de acuerdo?** Documento = una norma; sección = un bloque del BOE en una de sus versiones (`doc_id#bloque@AAAAMMDD`), con vigencia `[valid_from, valid_to)`. Si una etapa no lo respeta, un gold sin fecha (`doc_id#bloque`) es ambiguo entre versiones, y un índice que ignore `valid_to` le entrega al generador una versión que ya no rige o que todavía no rige.
- **¿Sobrevivirían los `doc_id` a una descarga nueva?** Sí. El `doc_id` es el identificador oficial del BOE, y el `section_id` lleva el día desde el que rige la versión: una versión nueva agrega un id en lugar de cambiarle el contenido a uno existente. Con `doc_id#bloque` solo, los arts. 38-40 de la LAU, derogados desde 2001, habrían cambiado de contenido en octubre de 2026. Lo probó la foto nueva: con 6 normas actualizadas, el eval set validó entero salvo un gold cuyo id seguía existiendo pero había dejado de ser la versión vigente.
- **¿Cómo se trataron los casi-duplicados, y a qué compromete eso en el eval set?** No se borra ni se fusiona nada. De los 90 pares de normas parecidas, ninguno es una copia: 87 son versiones (una reemplaza a la otra, y la reemplazada lleva `superseded_by`) y 3 son normas relacionadas que rigen las dos. 434 secciones tienen un gemelo casi literal en otra norma, y 65 de esos pares rigen los dos. El eval set queda comprometido a dos cosas: el gold de una pregunta sin fecha es la versión vigente más sus gemelos vigentes, como conjunto, y el de una pregunta a una fecha es la versión de ese día.

### La ficha del corpus (corpus card)

Es la misma que `corpus_card.md`. Los números son los de la foto del 2026-10-24 y salen de `01_documents.ipynb`.

#### Fuente y versión fijada

- **Fuente:** API de datos abiertos del BOE, colección de legislación consolidada: `https://www.boe.es/datosabiertos/api/legislacion-consolidada`. Documentación: `https://www.boe.es/datosabiertos/documentos/APIconsolidada.pdf` (versión del 2 de septiembre de 2025).
- **Qué trae cada norma:** metadatos (rango, ámbito, departamento, fechas, vigencia, estado de consolidación, ELI), análisis (materias, notas, referencias a normas anteriores y posteriores) y el texto consolidado con **todas las versiones fechadas** de cada bloque (`fecha_vigencia` y la norma que introdujo cada versión).
- **Versión fijada:** la descarga del **2026-10-24** (la del 2026-10-16 se perdió y no se podía volver a bajar: 6 normas habían cambiado). Las respuestas crudas están en `data/raw/boe_cache/` (fuera de git), con fecha de descarga y sha256 en `index.jsonl`. La lista de normas con la `fecha_actualizacion` de cada una está en `data/subset_v1.json` (en git). La fuente cambia seguido (la LAU tuvo versiones nuevas el 1, el 2 y el 8 de octubre de 2026); `?from=20261024` lista lo actualizado desde la foto.
- **Verificación de la foto** (Etapa 1):
  - Son 1.763 archivos. Ninguno falta ni difiere de su sha256 (comparado sobre el texto: la foto del 2026-10-16 se había guardado con CRLF en Windows).
  - Las 177 normas siguen en la `fecha_actualizacion` fijada.
  - El build lee solo la caché (offline).
- **Hash del corpus:** `corpus_sha = b01a2c0acc3508873861365d25b37d28ea0a8dacf7f868834ddb5207e28b526d`, en `artifacts/full/documents.manifest.json`. Un build en otra máquina (Colab) tiene que dar el mismo.
- **Idioma:** español jurídico peninsular; las normas autonómicas están en su versión castellana publicada en el BOE.

#### Licencia y atribución

- **Marco:** reutilización de la información del sector público, Ley 37/2007. Se permite el uso comercial y no comercial. Condiciones (aviso legal del BOE): no desnaturalizar el sentido de la información, no sugerir carácter oficial ni patrocinio del BOE, y conservar los metadatos y las fechas de actualización.
- **Atribución en obras derivadas:** "Basado en datos de la Agencia Estatal Boletín Oficial del Estado".
- **Aviso obligatorio:** los textos consolidados tienen **carácter meramente informativo**; solo la publicación en el diario oficial tiene validez jurídica. Hay que indicarlo en cada texto reutilizado: va en el encabezado del informe.
- Los datos crudos no se commitean (assignment §6.1).

#### Regla del subconjunto

1. **Universo:** normas consolidadas de 10 materias de vivienda del vocabulario controlado del BOE: Arrendamientos (303), Arrendamientos urbanos (305), Desahucios (2476), Propiedad Horizontal (5774), Viviendas (7162), Viviendas de Protección Oficial (7163), Viviendas de Renta Limitada (7164), Viviendas sociales (7166), Impuesto sobre Viviendas Desocupadas (4118), Cooperativas de viviendas (1720). Son **438 normas y 16,1 M tokens**.
2. **Densidad temática ≥ 0,45:** al menos 45% de los tokens del articulado vigente está en artículos que mencionan vivienda, alquiler, desahucio o propiedad horizontal (regex en `data/subset_v1.json`). Saca las normas ómnibus (presupuestos, medidas fiscales, LEC, Impuesto sobre Sociedades).
3. **`estado_consolidacion = Finalizado`:** saca 15 normas autonómicas cuyo texto el BOE marca como desactualizado.

**Resultado: 177 normas, 3,09 M tokens** del generador (tokenizer de `Qwen/Qwen3-8B`, última versión de cada bloque; la cuenta de la Etapa 0). Con la extracción de la Etapa 1:
- **4,78 M tokens con todas las versiones;**
- **2,89 M de texto vigente el 2026-10-24.**

La diferencia con los 3,09 M tiene tres causas: la última versión no siempre rige, las normas derogadas no tienen nada vigente, y el texto citado sí se cuenta. Con la extracción corregida ninguna norma cae debajo del umbral de densidad.

| | Vigentes | Agotadas | Total |
|---|---|---|---|
| Estatales | 81 | 11 | 92 |
| Autonómicas | 67 | 18 | 85 |
| **Total** | 148 | 29 | 177 |

- **Rangos:** 73 leyes, 22 decretos-ley, 22 reales decretos, 16 reales decretos-ley, 16 órdenes, 14 resoluciones y otros.
- **Tokens por norma (Etapa 0):** p10 2.196 · mediana 10.469 · p90 45.113 · máx. 121.280; 76 de 177 entran solas en un prompt de 7.424 tokens.
- **Tokens vigentes por norma vigente (Etapa 1):** p10 1.467 · mediana 12.641 · p90 51.862 · máx. 123.867. **47 de las 135** entran solas en 7.424 tokens: para esas preguntas, "si entra, pegalo" es posible.
- **Preceptos:** 5.468, con una mediana de 230 tokens.
- **Control de recall:** de las 13 normas estatales con "alquiler" en el título, 12 están en la unión de materias; la que falta (RD 933/2021, registro de hospedajes) está bien afuera.

#### Documento y sección

- **Documento = una norma.** `doc_id` es el identificador oficial del BOE (`BOE-A-1994-26003`, `BOCL-h-2020-90108`, ...), estable entre descargas.
- **Sección = un bloque del BOE en una de sus versiones.** El bloque puede ser un artículo, preámbulo, título, anexo, nota inicial o parte dispositiva; las firmas se descartan.
  - `section_id = doc_id#bloque@AAAAMMDD` (`BOE-A-1994-26003#a9@20130606`), con el id de bloque del BOE y el día desde el que rige esa versión.
  - El id de bloque **no** se deriva del título: el art. 9 bis es `a9-2`.
  - Si dos versiones del mismo bloque caen el mismo día, la segunda lleva `.2`.
- **Vigencia.** Cada sección rige en `[valid_from, valid_to)`. `Section.in_force(fecha)` dice si rige en esa fecha; la versión vigente no siempre es la última (el art. 10 de la LAU tiene una versión que rige recién desde el 15/11/2026). `block_id` agrupa las versiones de un mismo bloque. `path` es la miga de pan: los títulos y capítulos que contenían a esa versión el día en que empezó a regir. Las disposiciones adicionales, transitorias y finales no cuelgan de ninguno.
- **Texto.** `Document.text` es la concatenación de las secciones, separadas por una línea en blanco: un párrafo por línea, ya normalizado. Las notas de modificación y las editoriales quedan en `Section.notes`, fuera del texto.
- **Tamaño.** 177 documentos y 8.686 secciones, de las que 5.766 rigen el 2026-10-24. Hay 6.735 bloques, y 1.032 tienen más de una versión.

**Qué se rompe si una etapa posterior no está de acuerdo:**
- Un gold escrito como `doc_id#bloque`, sin fecha, es ambiguo entre versiones.
- Un índice que no respete `valid_to` le entrega al generador una versión que ya no rige, o que todavía no rige.

#### Mapeo de estado

`DocMetadata.status` solo acepta `current`, `superseded`, `deprecated`, `draft` o `unknown`. Los campos crudos (`vigencia_agotada`, `estatus_derogacion`, `estatus_anulacion`, `estado_consolidacion`) quedan en `source_status`, y la razón de cada corrección en `status_reason`.

| Fuente | `status` | Normas |
|---|---|---|
| `vigencia_agotada = N` | `current`, salvo las correcciones de abajo | 148 → 135 |
| `vigencia_agotada = S` (derogada, agotada por plazo o sustituida tácitamente) | `superseded` | 29 |
| **Correcciones verificadas contra el texto** (`corpus.CORRECTIONS`, `corpus.PERIODIC_SERIES`) | | |
| Ley balear 11/2005 (`BOE-A-2005-21319`): figura vigente, pero su nota inicial dice "Norma derogada, con efectos de 1 de enero de 2018" | `superseded`, cerrada el 2018-01-01 | 1 |
| Resoluciones anuales de tipos de interés de préstamos convenidos (13, de 2014 a 2026): todas figuran vigentes | `current` solo la de 2026; las otras 12, `superseded`, con `superseded_by` a la siguiente | 12 |
| RDL 27/2012 (`BOE-A-2012-14115`): agotada sin fecha de derogación | `superseded`, cerrada el 2013-05-15 (entrada en vigor de la Ley 1/2013, que lo sustituyó) | — |

**Resultado: 135 `current`, 42 `superseded`.** 37 de las 42 llevan `superseded_by`; las otras 5 no nombran a la norma que las reemplazó. Cuando una norma deja de regir, sus versiones se cierran en esa fecha: después de su fin no rige nada, salvo 3 versiones transitorias que el BOE fecha después de la derogación (`after_norm_end`).

**Tasas de desacuerdo con el cuerpo:**
- **Estado contra la nota inicial:** 1/177 (la ley balear).
- **`fecha_vigencia` contra "entrará en vigor el día siguiente al de su publicación":** 57/57, tomando la publicación oficial.
- **`fecha_publicacion`:** falla en 17 de las 22 autonómicas chequeables, las de id `BOE-A`, porque es la fecha de la copia en el BOE. Por eso los filtros por fecha usan `valid_from`/`valid_to`.

#### Defectos conocidos

Lo que un loader ingenuo dejaría pasar sin ningún error:

1. **El JSON del análisis pierde materias.** La conversión de la API conserva solo la última `<materia>`: 436 de 438 normas pierden materias (la LAU queda con 1 de 17). Se lee en XML.
2. **Versiones superpuestas.** El texto trae todas las versiones de cada bloque. Concatenarlas lleva el subconjunto de 3,09 M a 4,46 M tokens y mezcla reglas contradictorias (el art. 9 de la LAU dice "cinco años", "tres años" y otra vez "cinco años").
3. **Notas de modificación dentro del texto, y un `<blockquote>` que no es nota.** Las notas ("Se modifica el apartado 3 por...", "Téngase en cuenta...") son 2.685 `<blockquote>`, ~0,36 M tokens: van a `Section.notes`. Pero los 748 `<blockquote class="sangrado">` (~0,31 M tokens, 38 normas) son el texto que cita una ley modificadora. La auditoría de la Etapa 0 los descartaba con las notas, y "Artículo único. Se modifica la Ley X así:" quedaba vacío. En la Etapa 1 son texto.
4. **Metadatos que mienten o significan otra cosa.**
   - `vigencia_agotada` coincide con `estatus_derogacion` en 176 de 177 normas. La que no coincide está agotada sin derogación (sustitución tácita).
   - Contra el texto: 1 norma figura vigente y su texto dice derogada.
   - Las normas periódicas reemplazadas figuran vigentes.
   - En la unión (no en el subconjunto), la nota del RDL 2/2026 dice "Norma *deroga* por..." (sic): una búsqueda de "derogada" no la ve.
5. **`fecha_publicacion` no es la fecha oficial en las autonómicas con id `BOE-A`.** Es la de la copia en el BOE, posterior a la del boletín regional: en 64 normas autonómicas es posterior incluso a la entrada en vigor. La fecha regional está en una nota del análisis ("Publicada en el DOG núm. 88, de 10 de mayo de 1988").
   - Contra la cláusula "entrará en vigor el día siguiente al de su publicación", `fecha_publicacion` falla en 17/17 de esas normas, y la fecha regional acierta en las 17.
   - Usar `fecha_vigencia` (`valid_from`).
6. **La misma región con dos nombres** de departamento ("Comunitat Valenciana" ×5, "Comunidad Valenciana" ×1).
7. **Un mismo id de bloque puede cambiar por completo de contenido.** Los arts. 38-40 de la LAU estaban derogados desde 2001 y en octubre de 2026 se llenaron con un régimen sancionador para plataformas de alquiler de corta duración.
8. **Formato.**
   - Tablas e imágenes:
     - 23% de las normas tienen tablas: 166 fuera de las firmas, linealizadas. Las fórmulas maquetadas como tabla (Kt = 0,35 · Ht/Ho + ...) quedan ilegibles igual.
     - 162 imágenes (URLs a PNG, no base64) en 29 normas, reemplazadas por `[imagen omitida]`. Lo que dicen no está en el texto.
   - Preceptos gigantes: 3% supera 1.500 tokens, hasta 25 K en los "Artículo único" de leyes modificadoras.
   - Caracteres:
     - "m2" y "m²" conviven entre versiones. NFKC los unifica, pero también cambia "1.º" por "1.o" (2.911 ordinales) y rompe la prima de los subapartados "a´)". Se unifican solo superíndices y ligaduras.
     - El texto de una celda con `<br/>` pegaba palabras ("personasque"); corregido.
9. **La versión vigente no es la última.** Pasa en 3 bloques: el art. 10 de la LAU tiene una versión del RDL 28/2026 que rige recién desde el 15/11/2026. Además:
   - 18 versiones vienen sin `fecha_vigencia` y se fechan con la norma que las introdujo o, si no, con su publicación.
   - 11 bloques (54 versiones) traen las versiones fuera de orden.
   - 6 versiones se reemplazan el mismo día.
   - 23 versiones caducaron antes de empezar a regir.
   - 38 versiones empiezan el mismo día en que termina su norma.
   - 3 versiones están fechadas después de la derogación de su norma (régimen transitorio).
10. **La caché cruda no coincide en bytes con su sha256.** Se guardó en Windows con CRLF; el contenido sí coincide. La verificación compara el texto decodificado.

**Huecos de cobertura:** la ley catalana 18/2007 del derecho a la vivienda y otras 14 autonómicas (desactualizadas en el BOE, excluidas); el procedimiento de desahucio de la LEC (excluida por densidad, 0,13).

#### Limpieza (Etapa 1)

Por regla: en cuántas normas cambió algo y cuántos caracteres sacó (negativo = agregó). El tratamiento de cada elemento y su porqué están en `decision_log.md` (Etapa 1).

| Regla | Qué hace | Normas | Caracteres |
|---|---|---|---|
| `notes_out` | notas de modificación y editoriales (`<blockquote>`) → `Section.notes` | 116 | 848.500 |
| `signature` | descarta el bloque de firma | 172 | 33.159 |
| `promulgation` | descarta la fórmula de promulgación del preámbulo | 77 | 17.580 |
| `editorial_inline` | notas "(\*) Téngase en cuenta" en línea → `notes` | 5 | 1.491 |
| `whitespace` | espacios repetidos dentro de una línea | 5 | 41 |
| `invisibles` | guion blando, ancho cero, BOM, control | 5 | 33 |
| `nfc` | acentos descompuestos → compuestos | 2 | 17 |
| `odd_spaces` | NBSP y espacios tipográficos → espacio común | 92 | 0 |
| `sup_ordinals` | `5.<sup>o</sup>` → `5.º` | 3 | 0 |
| `compat` | superíndices y ligaduras (² → 2) | 13 | −1 |
| `images` | imagen → `[imagen omitida]` | 29 | −2.592 |
| `tables` | tablas linealizadas ("Encabezado: valor; ...") | 41 | −89.413 |

**En tokens del generador:**

| Versión del texto | Tokens |
|---|---|
| Lo que daría un loader ingenuo (todas las versiones, con notas y firmas) | 5,14 M |
| `documents.jsonl` (todas las versiones) | 4,78 M |
| Texto vigente el 2026-10-24 | 2,89 M |

Lo que más cambia es estructural (notas, versiones, texto citado), no de codificación: a nivel carácter, el XML del BOE es limpio.

#### Casi-duplicados: política

- **Norma ↔ norma** (todos los pares; Jaccard ≥ 0,3 o contención ≥ 0,8): 90 pares, ninguno es copia (Jaccard ≥ 0,9).
  - **87 son versiones:** una de las dos está reemplazada (RDL ↔ ley de convalidación, RDL derogado ↔ re-decretado, resolución anual ↔ la siguiente).
  - **3 son normas relacionadas** que comparten texto y rigen las dos.
  - Quedan en `metadata.near_duplicates`, y la reemplazada lleva `superseded_by`. No se borra ni se fusiona ninguna.
- **Sección ↔ sección de otra norma** (Jaccard ≥ 0,7): 434 secciones tienen un gemelo en `Section.twins`. 65 de esos pares rigen los dos el 2026-10-24.
- **Versión ↔ versión del mismo bloque:** son secciones distintas, porque dan respuestas distintas.
- **Para la Etapa 2:**
  - El gold de una pregunta sin fecha es la versión vigente, más sus gemelos vigentes como conjunto.
  - El de una pregunta a una fecha es la versión vigente en esa fecha.

#### Qué pone a prueba bien este corpus

- **Versiones con respuestas distintas.**
  - 18% de los preceptos tiene más de una versión (8% con 3 o más).
  - Las versiones consecutivas son casi idénticas en texto (Jaccard mediana 0,82) y distintas en la respuesta.
  - Preguntas "a una fecha" (un contrato firmado en 2016 tenía 3 años de plazo mínimo, no 5) y actualizaciones de memoria.
- **Vecinos confundibles.**
  - 7% de los artículos tiene un gemelo casi literal en otra norma. En la Etapa 1, contando versiones, son 434 secciones, y 65 pares rigen los dos a la vez.
  - Pares de RDL derogado ↔ RDL re-decretado (21/2018 ↔ 7/2019; 26 ↔ 29/2026; 27 ↔ 28/2026), decreto-ley ↔ ley que lo tramitó, y 13 resoluciones anuales con la misma plantilla.
- **Identificadores literales:** números de ley y de artículo, para la categoría `identifier` y para el canal léxico.
- **Recuperación frente a recuerdo.** 11% de los preceptos nació después de mayo de 2025 (corte de Qwen3) y 18 normas se publicaron después; entre ellas el Plan Estatal de Vivienda 2026-2030, la Ley 5/2025 de Vivienda de Andalucía y los RDL de octubre de 2026. El generador no pudo leerlas.
- **Región:** 85 normas autonómicas de 16 comunidades, para el filtro de metadatos y la memoria del usuario ("vivo en Baleares").
- **Abstención adyacente:** lo excluido (LEC, ley catalana) y lo derogado dan preguntas `out_of_corpus` plausibles.

#### Chequeo a libro cerrado

`00_closed_book.ipynb`: 5 preguntas sin recuperación, una por categoría más una posterior al corte. Corrida local del 2026-10-16 (modelo 63% en GPU): **0/5**, con 2 abstenciones y 3 alucinaciones seguras. Repetida el 2026-10-24 con el modelo 100% en GPU (Ollama 0.34.4): **1/5**. El resultado se mantiene con dos variantes de formato (schema sin la instrucción de sí/no, y texto libre).

`qwen3:8b` no conoce ni los datos más difundidos de este corpus (prórroga de la LAU, fianza del art. 36, umbral de gran tenedor). Un eval set sobre este corpus mide recuperación, no recuerdo.

## 3 · El conjunto de evaluación

- **40 preguntas de un turno** (30 de test: 13 `topical`, 6 `identifier`, 7 `metadata`, 4 fuera del corpus), **10 guiones** (8 de test, 42 turnos) y **4 escenarios de memoria** (3 de test, 22 sondas). Las de identificador se derivan con código, no se escriben.
- **Cómo se escribió:** desde una lista de temas, con la respuesta esperada antes de buscar; la clave se copia del texto gold, la pregunta no. Hay 9 preguntas con fecha, y 4 "sí" y 4 "no" en test.
- **Contaminación:** ninguno de los 141 textos chequeados (preguntas, turnos, reescrituras oráculo y sondas) comparte 5 términos seguidos con su gold; el máximo es 4. BM25 no pone primera ninguna sección gold (sección@1 = 0 en todas las categorías).
- **Congelado** antes de cualquier resultado (`eval/FROZEN.json`, clave `99676007c42e6f6e`).

### Preguntas de la Etapa 2

- **¿Cómo se evitó que las preguntas copiaran al corpus, y qué dice el chequeo de contaminación?** Se escribieron desde temas, con la respuesta esperada antes de buscar; las claves se copiaron del gold y las preguntas no, y las de identificador se derivaron con código. El chequeo usa la misma normalización que el canal léxico: ningún texto comparte 5 términos seguidos con su gold, y BM25 no pone primera ninguna sección gold.
- **¿Qué fracción del gold son conjuntos en lugar de ids únicos, y por qué?** El 7% (7 de 97), todos porque la respuesta está en dos lugares: una ley estatal y una autonómica, o dos artículos. Ninguno por gemelos. Las secciones que solo remiten a otra norma no entran en el gold.
- **¿Qué habilidades de memoria cubren los escenarios, con cuántas sondas, y cuáles quedaron afuera?** Las seis, en las 22 sondas de test: 6 de actualización de conocimiento, 4 multi-sesión, y 3 de extracción, temporal, abstención y preferencia. Quedaron afuera las preferencias que cambian, los datos que dijo el asistente y el olvido a pedido.
- **Una pregunta que haría un usuario real y que el conjunto no puede medir:** "¿Me conviene más alquilar o comprar con la ayuda del Plan Estatal?". Pide un consejo y una cuenta con los datos del usuario, no una regla que esté en un artículo, y la contención de una clave no puede puntuarla.
- **Si `topical` e `identifier` no coinciden sobre una técnica, ¿en cuál se confía, y por qué decidirlo antes?** En `topical`, decidido antes de cualquier resultado: es lo que pregunta un usuario real. `identifier` mide la búsqueda literal de un artículo por su número, que depende del prefijo (el número de la norma está en el título del documento, no en la sección). Decidirlo antes evita elegir, después de ver la tabla, la columna que le da la razón a la técnica preferida.

## 4 · Etapa 3 · Recuperación

Ocho filas acumulativas, cada una con una sola variable de diferencia, todas con el mismo presupuesto de 3.000 tokens del generador. La primaria (pre-registrada) es **"clave entregada desde el gold"**: llegó al generador un pasaje del gold que contiene la clave. Sobre las 26 preguntas con respuesta de test:

| Fila | doc@10 | sección@10 | clave desde gold | pasajes | ms (mediana) | contra la anterior (signos) |
|---|---|---|---|---|---|---|
| bm25 | 22 | 4 | 4 | 10,2 | 7 | — |
| dense | 23 | 9 | 10 | 10,8 | 16 | 7-1, p = 0,070 (dos flags) |
| +prefix | 26 | 17 | 16 | 10,7 | 15 | 7-1, p = 0,070 |
| +rrf | 26 | 18 | 16 | 11,2 | 31 | 4-4, p = 1,00 |
| +rerank | 26 | 20 | 17 | 10,0 | 224 | 4-3, p = 1,00 |
| +parent | 26 | 20 | 17 | 4,9 | 226 | 3-3, p = 1,00 |
| +date | 25 | 22 | 19 | 5,0 | 230 | 5-3, p = 0,73 |
| +region | 25 | 22 | 19 | 5,0 | 235 | 0-0 |

- **Qué pagó su costo.** Ninguna pieza supera p = 0,05. El prefijo contextual (título, sección y vigencia antepuestos al hijo) es la única que se acerca: 7 victorias y 1 derrota. RRF, reranker, padres y filtros quedan dentro del ruido. Por la regla pre-registrada (entre las filas que no se distinguen de la de más aciertos, gana la de menos piezas), **la mejor configuración es `+prefix`**, aunque `+date` tenga 19. Es la que usan las Etapas 4-6: RRF, reranker, padres y filtros quedan afuera.
- **La elección se hizo sobre `test`.** `dev` tiene 8 preguntas con respuesta: con tan pocas, casi ninguna fila se distingue de otra, y la regla tiende a elegir la más simple. Elegir sobre los mismos datos en los que después se reporta es optimista; lo mitiga que la regla no elige la fila con más aciertos sino la más simple que no se distingue de ella. La Etapa 8 aplica la regla en `dev` y mide cuántas veces vuelve a elegir `+prefix` al remuestrear las 26 preguntas de `test`.
- **Lo que la regla dejó afuera importa más que lo que dice la tabla.** La ficha del corpus dice que un índice que no respeta la vigencia "le entrega al generador una versión que ya no rige, o que todavía no rige", y `+prefix` hace exactamente eso: busca entre todas las versiones. Contado con las filas guardadas de la Etapa 3, por id de sección (sin la equivalencia por texto idéntico que usa la primaria), en las 7 preguntas `metadata` de test `+prefix` entrega el gold sin ninguna otra versión de su artículo en 1 (md06), el gold junto con otras versiones en 3 (md01, md02, md05) y otras versiones sin el gold en 3 (md03, md04, md07). `+date` entrega el gold y ninguna otra versión en 6 de las 7; en md03 no llega ninguna versión de su artículo. A md02 `+prefix` le entrega cinco versiones del art. 9 de la LAU; `+date`, solo la de febrero de 2019. En un corpus versionado, el filtro de vigencia es parte de la definición del documento, no una pieza que tenga que ganarse su lugar en una tabla de 26 preguntas. Lo que la regla de parsimonia no podía ver es que la tabla mide la clave entregada, no la versión equivocada entregada al lado.
- **El movimiento más grande, abierto:** de BM25 a denso cambian 8 preguntas, 7 a favor del denso. En 4 de ellas BM25 no tiene la sección gold ni en su top 100: el usuario pregunta si puede "dejar el contrato" o quién arregla "la caldera", y la ley dice "desistimiento" y "reparaciones". BM25 cae en preámbulos de reales decretos-ley y en la Ley de Propiedad Horizontal por términos genéricos ("piso", "alquiler"); el denso encuentra los artículos de la LAU. **Gana por la razón que da su paper**, el desajuste de vocabulario entre la pregunta y el texto, pero no por ponerlo primero: el gold queda entre los puestos 1 y 15 del denso y llega por la entrega de 3.000 tokens. La única que pierde (md04) BM25 la pone segunda y el denso, 21.ª.
- **Documento contra sección:** con `+prefix`, el documento correcto llega en 26 de 26, pero la sección en 17 y la clave desde el gold en 16. Lo que predice para la Etapa 4: el techo de la generación con recuperación es de unas 16 preguntas.
- **Hijos contra padres, con tokens igualados:** empate (17 contra 17, 3-3). Los padres entregan la mitad de pasajes.
- **El filtro de fecha** mueve 5-3 a favor (`topical` de 6 a 8, `metadata` de 5 a 6), sin superar el ruido. El de región no cambia nada. La fila `+date` lo mide sobre `+parent` (con RRF, reranker y padres): la comparación de una variable contra la configuración elegida, `+prefix+date` contra `+prefix`, es la sección 1c de la Etapa 8.
- **¿Puede la recuperación sola decir que la respuesta no está?** No. El coseno máximo de las preguntas fuera del corpus (media 0,67) no es menor que el de las que tienen respuesta (0,64), y el puntaje del reranker tampoco.
- **La ventana del reranker:** 0 de 2.000 pares (consulta + hijo) pasan los 512 tokens.

## 5 · Etapa 4 · Respuestas fundamentadas

Tres brazos sobre las mismas 30 preguntas, en orden fijo y con el modelo descargado al empezar:

| Brazo | Acierto | Con respuesta | Abstención correcta | Citó sin recibir |
|---|---|---|---|---|
| Libro cerrado | 5/30 | 3/26 | 2/4 | — |
| Recuperación (`+prefix`) | 18/30 | 15/26 | 3/4 | 0 |
| Oráculo (secciones gold) | 25/30 | 22/26 | 3/4 | 0 |

- **Recuperación contra libro cerrado:** 14-1 (p = 0,001; +43 puntos, IC +23 a +63). **Oráculo contra recuperación:** 7-0 (p = 0,016; +23 puntos, IC +10 a +40). El oráculo recibe menos tokens (1.086 contra 2.968): su ventaja mezcla "el pasaje correcto" con "menos distractores". La Etapa 8 separa las dos cosas con un oráculo rellenado con distractores hasta los mismos 3.000 tokens (`oracle_padded`).
- **Libro cerrado: 3 de 26.** Lejos del umbral de §7.4: el eval set mide recuperación, no preentrenamiento.
- **La tabla 2×2** (fila de recuperación, 26 preguntas con respuesta):

| | respondió bien | respondió mal |
|---|---|---|
| **recuperó bien** (clave desde el gold) | 13 | 3 |
| **recuperó mal** | 2 | 8: 4 fieles al pasaje equivocado, 3 abstenciones, 1 sin apoyarse en lo entregado |

- **Las fallas con el pasaje correcto a la vista son de razonamiento temporal, y también de recuperación.** En md02 y md05 llegaron dos versiones del mismo artículo y el modelo aplicó la siguiente a una fecha anterior a su vigencia: para un contrato de febrero de 2019, el texto "vigente desde marzo de 2019". El modelo lee mal los intervalos, pero la versión equivocada se la dio el sistema: con el filtro de fecha, a md02 y md05 les llega solo la versión de su fecha (sección 4). Si con eso aciertan lo mide la Etapa 8 (`retrieval_date`, la mejor fila más el filtro de fecha). La tercera falla (tp05) es correcta en sustancia, pero sin la palabra de la clave.
- **Citas:** ningún brazo citó un bloque que no se le entregó (0 en 60 respuestas con pasajes). Con la respuesta en el contexto, se abstuvo 0 veces en recuperación y 1 en el oráculo.
- **Un prompt demasiado grande:** 11.239 tokens con `num_ctx` 8.192. Con la guarda, `ContextOverflow` antes de mandar. Sin ella, el servidor evaluó 4.098 (la mitad de la ventana: los primeros tokens y la cola) y el modelo respondió con seguridad, sin la clave. Solo lo delató la comparación de nuestra cuenta con `prompt_eval_count`.
- **Varianza:** 10 preguntas repetidas en frío, en otro orden y sin caché, cambiaron el texto en 7 casos y el puntaje en ninguno.
- **Posición del pasaje:** no se midió en la entrega. La Etapa 8 la mide: el mismo conjunto de ~6.000 tokens con el gold primero, en el medio o último, sobre las 26 preguntas con respuesta. Dicho antes de correr: a esa escala un nulo no refuta *Lost in the Middle*, y para que el test de signos dé p < 0,05 hacen falta al menos 6 preguntas discordantes, todas para el mismo lado.

## 6 · Etapa 5 · La conversación

**Parte A · con qué buscar** (37 turnos con respuesta, 8 guiones de test; el ruido se mide por guion):

| Consulta | doc@5 | sección@5 | clave desde gold (1.350 tokens) |
|---|---|---|---|
| último turno | 26 | 15 | 17 |
| conversación completa (el turno más nuevo primero) | 32 | 11 | 13 |
| reescritura del modelo | 26 | 16 | 16 |
| reescritura oráculo | 34 | 22 | 18 |

Por dependencia (doc@5 / sección@5):

| Dependencia | último turno | conversación | reescritura | oráculo |
|---|---|---|---|---|
| standalone (8) | 7 / 4 | 7 / 4 | 7 / 4 | 7 / 5 |
| coreference (10) | 5 / 2 | 9 / 3 | 5 / 1 | 9 / 4 |
| ellipsis (5) | 1 / 0 | 5 / 1 | 3 / 3 | 5 / 2 |
| topic_shift (6) | 6 / 5 | 4 / 0 | 3 / 3 | 6 / 6 |
| return (8) | 7 / 4 | 7 / 3 | 8 / 5 | 7 / 5 |

Por número de turno (doc@5 / sección@5):

| Turno | último turno | conversación | reescritura | oráculo |
|---|---|---|---|---|
| 1 (8) | 7 / 4 | 7 / 4 | 7 / 4 | 7 / 5 |
| 2 (8) | 2 / 1 | 7 / 3 | 4 / 2 | 8 / 4 |
| 3 (7) | 5 / 3 | 7 / 0 | 4 / 3 | 7 / 4 |
| 4 (6) | 6 / 3 | 3 / 0 | 3 / 1 | 6 / 5 |
| 5 (5) | 3 / 2 | 5 / 2 | 5 / 4 | 3 / 2 |
| 6 (3) | 3 / 2 | 3 / 2 | 3 / 2 | 3 / 2 |

- **Ninguna consulta supera el ruido** (la mayor diferencia por guion es oráculo contra último turno, 4-0, p = 0,125).
- **El ranking cambia con el turno, a nivel de documento.** En los turnos 2 y 3 la conversación completa encuentra la norma (7/8 y 7/7) y el último turno no (2/8 en el turno 2). En el turno 4 se invierte (último turno 6/6, conversación 3/6). A nivel de sección, la conversación completa es la peor desde el turno 3 (0/7 y 0/6). Con 3 a 8 turnos por fila, es una dirección y no evidencia.
- **La conversación completa encuentra la norma de la que se viene hablando, no el artículo nuevo:** gana en documento en las correferencias y elipsis, y pierde en los cambios de tema (0/6 a sección@5). El último turno falla donde se espera: el turno 2, la primera correferencia (2/8 a doc@5).
- **La reescritura oráculo no es mucho mejor que las del modelo** (18 contra 16-17), como señalan Ye et al. Por guion, contra la reescritura del modelo, queda 4-1 (p = 0,375): no hay diferencia que defender. Por dependencia (sección@5), le gana en los cambios de tema (6/6 contra 3/6) y en las correferencias (4/10 contra 1/10), pierde en las elipsis (2/5 contra 3/5) y empata en `return` (5/8). Por turno, pierde en el 5 (2/5 contra 4/5).
- **Las 20 reescrituras leídas:** 15 resuelven bien la referencia. Una agrega contenido de la ley que nadie dijo ("los doce meses del artículo 9 bis"), que es el efecto HyDE. Otra mezcla mal el historial, una arrastra el tema anterior a un cambio de tema, una no resuelve nada y una pierde la fecha que decide la respuesta.

**Parte B · qué conservar.** Cada guion se reproduce solo, con su historial guionado. El total de pasajes + historial queda fijo en 1.500 tokens y el historial tiene un tope: lo que no usa pasa a los pasajes, así el total es el mismo en todos los brazos (la diferencia de `prompt_eval_count` entre brazos es de 46 tokens en promedio). Consulta: último turno, por la regla pre-registrada.

| Brazo | Pierde el turno 1 (de 34) | Acierto (41 turnos comunes) | Clave desde gold | Llamadas por turno |
|---|---|---|---|---|
| ventana, tope 100 | 26 | 18/41 | 16/37 | 1,0 |
| ventana, tope 150 | 19 | 17/41 | 17/37 | 1,0 |
| ventana, tope 250 | 7 | 20/41 | 17/37 | 1,0 |
| resumen (MemGPT), tope 150 | — | 18/41 | 17/37 | 1,6 |
| buffer completo (referencia) | 0 | 19/41 | 17/37 | 1,0 |

El acierto se cuenta sobre los 41 turnos que puntuaron en todos los brazos: dl04.t4 dio error en tres de ellos (abajo) y queda afuera de los cinco. En los dos brazos en que no dio error, falló, así que ningún conteo cambia; cambian los denominadores, que antes eran 41 o 42 según el brazo.

- **El tope restringe de verdad, y el acierto casi no se mueve.** La mejor división fue el tope de 250 tokens de historial, con el resto de los 1.500 para pasajes (20/41), pero ninguna comparación supera el ruido (la mayor, 250 contra 150, es 3-0 por guion, p = 0,25). El turno 2 da 1/8 en todos los brazos: con el último turno como consulta, el pasaje que la correferencia necesita no llega, y conservar el turno 1 en el historial no lo trae. **A esta escala, con qué se busca pesa más que qué historial se guarda.**
- **La sonda del resumen** (una pregunta derivada del turno 1, sin pasajes) sí muestra el mecanismo. Con el mismo tope de 150, la ventana recuerda el turno 1 en 1 de 7 guiones y el resumen en 3 de 7, sin tenerlo a la vista. Con el buffer completo, 7 de 7; sin historial, 0 de 7. Con 7 guiones no es evidencia (p = 0,625).
- **La auditoría de los resúmenes** (25 actualizaciones, en las categorías de Wang et al.; hay una planilla ciega para que una persona la confirme, sección 10): 0 hechos inventados, 1 relación equivocada, 16 con detalles faltantes, 2 con texto corrupto (tokens en chino) y 6 correctas. Los faltantes tienen dos causas: el resumen **se congela en su tope de 220 caracteres** (30 de 58 lo tocaron) y repite el mismo texto cortado, o **reemplaza en lugar de fusionar** y pierde lo anterior (en dl01, la fecha de 2016). Lo que se probó es este resumen, con este tope.
- **Costo, y si vale la pena:** el resumen cuesta 1,6 llamadas por turno contra 1,0, y 247 tokens de salida contra 214. A cambio da 1 acierto más que la ventana del mismo tope (18/41 contra 17/41) y recuerda más en la sonda (3 de 7 contra 1 de 7), las dos cosas dentro del ruido. **Este resumidor** no paga su costo. La conclusión es sobre un resumidor que se congela en su tope, no sobre la estrategia: la Etapa 8 prueba un resumidor v2, con el mismo tope de historial, que pide reescribir comprimiendo y no deja que la gramática corte el texto.
- **Errores:** 3, los tres el mismo turno (dl04.t4): la respuesta llegó al tope de 1.200 caracteres y se contó como error de llamada.

## 7 · Etapa 6 · Memoria entre sesiones

**El régimen: el historial entra en la ventana.** Cada escenario de test tiene entre 820 y 947 tokens de historial, y quedan 3.794 para él; incluso los cuatro escenarios seguidos (3.629) entran. Para ver el régimen en el que no entra se impuso un presupuesto: `replay_600` recorta 28 de las 30 sondas. Con un presupuesto de 3.000 tokens, el historial dejaría de entrar en el cuarto escenario seguido; con 600, en la tercera sesión del primero. Lo que cambia entonces: con 600, la clave llega al espacio de memoria de la reproducción en 7 de 19 sondas en lugar de 13, y reproducción y almacén empatan (10 contra 10).

Cinco brazos con el mismo prompt, los mismos pasajes y la misma pregunta; solo cambia el espacio de memoria (y en `store_filter`, el filtro de los pasajes). Sobre las 22 sondas de test (conteos):

| Habilidad | sin memoria | replay (3.000) | replay_600 | store | store + filtro |
|---|---|---|---|---|---|
| extracción (3) | 0 | 2 | 1 | 1 | 1 |
| multi-sesión (4) | 1 | 2 | 1 | 1 | 1 |
| temporal (3) | 1 | 1 | 1 | 0 | 0 |
| actualización (6) | 2 | 2 | 2 | 4 | 5 |
| abstención (3) | 3 | 2 | 3 | 3 | 2 |
| preferencia (3) | 1 | 2 | 2 | 1 | 1 |
| **total (22)** | **8** | **11** | **10** | **10** | **10** |
| tokens de memoria por sonda | 0 | 847 | 550 | 404 | 404 |
| clave en el espacio de memoria | 0/19 | 13/19 | 7/19 | 5/19 | 5/19 |

- **Ninguna comparación supera el ruido, y con este tamaño ninguna podía superarlo.** Con 3 escenarios de test, el mejor resultado posible por escenario es 3-0 (p = 0,25); por sonda, 22 sondas que no son independientes entre sí. La etapa quedó dimensionada para mostrar mecanismos, no efectos: lo que sigue son direcciones, y la limitación es del diseño (cuatro escenarios escritos a mano, uno en `dev`), no de la suerte. La dirección más clara es reproducción contra sin memoria, 5-2 por sonda (p = 0,45).
- **La memoria gana en las actualizaciones de conocimiento** (2 → 4-5 de 6): es para lo que existe el camino de escritura. La reproducción ve los dos valores y no siempre elige el último. **La reproducción gana en la extracción y lo multi-sesión**, donde hace falta el dato literal: el almacén encuentra las sesiones con evidencia (25 de 31) pero la clave está en su espacio en 5 de 19. La extracción resume, y se pierde lo que se resume.
- **La predicción de Mem0, a esta escala:** donde la reproducción entra, el almacén gana en costo (404 tokens de memoria contra 847) y no en exactitud (10 contra 11), dentro del ruido. El ahorro tiene un costo que la lectura no ve: 66 llamadas de escritura para los 3 escenarios de test, unos 2.100 tokens por sonda. Con el mismo presupuesto (600), almacén y reproducción empatan (10 contra 10).
- **Una actualización de punta a punta** (sc01): "35 años" se guarda en la sesión 1; en la sesión 3 ("cumplí 36 el 8 de agosto") el UPDATE cierra el registro viejo (`valid_to`) y agrega el nuevo. La sonda del "antes" (15 de junio) ve 35 como vigente y la del "después" (octubre), 36.
- **Qué guardó que no era cierto** (auditoría de los 76 registros, hecha por un modelo, sección 10; en test, 56): en test, 50 bien, 3 mal y 3 que nunca se dijeron. En el escenario de dev, 10 bien, 7 mal y 3 nunca dichos. Los errores tienen tres formas:
  - **etiquetas equivocadas** que hacen que la regla de un valor vigente por *subject* cierre un dato verdadero: la ciudad de la usuaria guardada como la de la arrendadora;
  - **la fecha de la sesión tomada como dato** ("firmó el contrato el 11 de mayo de 2026");
  - **no-hechos** ("el usuario no especificó preferencia de citas") que **invalidan preferencias reales** dichas antes.

  **Los chequeos automáticos no ven ninguno de estos errores.** El de lenguaje de hipótesis marca 0. El de fugas del análisis marca 7, pero 5 de esos dicen lo que se dijo: el extractor parafrasea igual en su análisis y en el registro. Los no-hechos son las verdaderas "hipótesis guardadas como hechos", y solo se ven leyendo.
- **La memoria como filtro:** cambió los pasajes en 27 de 30 sondas y el resultado en 2 (ayudó en una actualización, perjudicó una abstención). Ningún efecto que leer.

## 8 · Etapa 7a · El juez

Dos jueces sobre las 407 respuestas de test de las Etapas 4-6: `qwen3:8b` (el mismo modelo que generó las respuestas) y `granite3.3:8b` (otra familia). Reciben la pregunta, la clave y la respuesta, sin el contexto ni el análisis del generador. Las abstenciones en preguntas con respuesta son incorrectas por regla, sin llamar al juez.

**Juez contra contención** (los 407 juicios):
- `qwen3:8b` es **más severo** que la contención: en las preguntas con respuesta acepta 58 de 216, contra 90 de la contención (33 contra 1 en los desacuerdos).
- `granite3.3:8b` es **más indulgente**: a libro cerrado acepta 14 de 30, contra 5.
- Entre ellos coinciden en 305 de 407, y granite acepta 97 respuestas que qwen rechaza, contra 5 al revés.
- Ninguno da vuelta el orden de los brazos de ninguna etapa.

**El auditor es un modelo, no una persona.** Los 82 juicios de la calibración (el 20%, muestreados con semilla y con los desacuerdos sobrerrepresentados) los auditó, sin ver los veredictos, un modelo de lenguaje más grande y de otra familia que los dos jueces. La consigna pide una persona. La tabla dice, entonces, qué instrumento se parece más a ese auditor, no cuál tiene razón:

| Instrumento contra el auditor-modelo | Acuerdo en la muestra | Reponderado | Acepta lo que el auditor rechaza | Rechaza lo que el auditor acepta |
|---|---|---|---|---|
| contención | 73/82 | 0,90 | 0 | 9 |
| `granite3.3:8b` | 69/82 | 0,86 | 8 | 5 |
| `qwen3:8b` | 42/82 | 0,75 | 4 | 36 |

- **Lo que se sostiene sin el auditor.** Los dos jueces difieren entre sí más que cualquier brazo de cualquier etapa (305 de 407 iguales), y `qwen3:8b` es más severo que la contención y que `granite3.3:8b`. Eso sale de comparar instrumentos sobre las mismas respuestas, y no depende de quién auditó.
- **Lo que depende del auditor.** Que la contención sea la vara más cercana a una lectura correcta (nunca aceptó una respuesta que el auditor rechazara; sus 9 desacuerdos son paráfrasis, como "al día en el pago" por "al corriente"), que por eso los números de las Etapas 4-6 sean una cota inferior, y que la autopreferencia no aparezca (`qwen3:8b` rechaza 36 respuestas que el auditor acepta). Las tres son plausibles, y las tres son hoy la lectura de un modelo.
- **Cómo se cierra.** Las planillas para que una persona rehaga la auditoría están en el zip (`judge_human/audit.csv`, ciega, con las mismas 82 filas; instrucciones en `AUDITORIA_HUMANA.md`). La Etapa 8 recalcula esta tabla contra la persona y mide su acuerdo con el auditor-modelo.
- **El framework (7b) no se hizo** (sección 10): no hay una comparación de lo que un framework decide por su cuenta.

## 9 · Lo que encontró el proyecto, más allá de las tablas

- **Un corpus legal se mueve más rápido que un trabajo práctico.** En ocho días cambiaron 6 de 177 normas, y una foto ya traía una versión con vigencia futura que cambió el gold de una pregunta al mover el día de referencia. Fijar la foto por hash y verificar el eval set al cargar no fue burocracia: es lo que detectó el problema.
- **El modelo lee mal la vigencia, y el sistema elegido le da con qué equivocarse.** Con el pasaje correcto y la versión siguiente del mismo artículo a la vista, aplica la versión equivocada a una fecha. La configuración elegida (`+prefix`, sin filtro de fecha) entrega versiones que no rigen en la fecha de la pregunta en 6 de las 7 preguntas `metadata` de test; con el filtro, en ninguna (sección 4). La lección es de diseño: en un corpus versionado, la vigencia es parte de la definición del documento (la ficha del corpus ya lo decía), y una regla de parsimonia sobre 26 preguntas no debería poder sacarla. Si con el filtro el modelo acierta lo mide la Etapa 8.
- **En la conversación, el cuello de botella es buscar bien el turno dependiente.** Guardar más historial no recuperó lo que la consulta no encontró.
- **Una memoria con "un valor vigente por *subject*" amplifica los errores del extractor:** un no-hecho o una etiqueta equivocada no solo agregan ruido, invalidan un dato verdadero. Y esos errores no los detecta ningún chequeo automático: hay que leer los registros.
- **Los jueces de 8B no se pueden usar sin calibrar.** Los dos difieren entre sí más que cualquier brazo de cualquier etapa, y el de la misma familia que el generador es el más severo, no el más indulgente (sección 8). La contención, la vara más simple, fue la que más coincidió con el auditor; como el auditor también es un modelo, esa parte queda abierta hasta la auditoría humana.
- **Temperatura 0 no es determinista,** pero a esta escala cambia la redacción, no el veredicto (7 de 10 textos, 0 de 10 puntajes).

## 10 · Prioridades, recortes y uso de IA

**El plan contra lo que pasó.** El plan inicial era hacer las ocho etapas y recortar desde el final.
- **Recortado:** 7b (reimplementar el loop en un framework), el experimento de posición del pasaje (Etapa 4) y el de §2.5. Las razones están en el registro.
- **El filtro por memoria** (Etapa 6) primero se recortó y después se hizo, cuando la entrega se movió una semana y el filtro de la Etapa 3 ya existía.
- **Dos cosas no estaban en el plan:** la foto se perdió y hubo que rehacer las Etapas 0-2 con una edición del eval set, y todas las corridas se hicieron en una sola máquina local (RTX 5070 Laptop de 8 GB, Ollama 0.34.4, la misma versión que Colab). Nada se corrió en Colab: que el pipeline corra en una T4 y que el build dé el mismo `corpus_sha` en otra máquina no está verificado.
- **El presupuesto real fue mucho menor que el estimado:** unas 1.420 llamadas al LLM y ~1,5 h de GPU para las Etapas 3-7, contra ~7-8 h de T4 estimadas. Los recortes se decidieron con el presupuesto estimado y no se revisaron con el real: con la GPU que sobró alcanzaba para la posición del pasaje, que la Etapa 8 corre. Lo que más poder le faltaba al proyecto, más escenarios de memoria (sección 7), costaba sobre todo tiempo de escritura, no de GPU. §2.5 sigue afuera por memoria, no por tiempo: con `num_ctx = 20.480`, `qwen3:8b` necesita ~8,2 GB entre pesos y caché KV, y no entra entero en 8 GB.

**El orden en el tiempo.** El repositorio tiene un solo commit (2026-10-24, el día antes de la entrega), así que no muestra por sí mismo que el eval set se congeló antes de los resultados. Lo muestran los artefactos: `eval/FROZEN.json` es de las 11:30 del 24 (hora de Buenos Aires) y la primera corrida de la Etapa 3 empezó a las 11:48 (`run_report.json`). Antes de esa hora, en la foto nueva, solo corrió el chequeo a libro cerrado de la Etapa 0 (11:10), cuyas 5 preguntas están en `dev` justamente porque se vieron sus salidas.

**Uso de IA (§2.3).**
- **Código.** Se usaron asistentes de código (Claude Code y Codex) para escribir el código de las etapas y sus tests, para revisarlo en busca de las trampas del enunciado antes de correr y para ejecutar las corridas.
- **Consultas de decisión.** Algunas decisiones de diseño se discutieron con el asistente. Las decidió el grupo: el corpus y el plan, los recortes, correr todo en una máquina local y rehacer la foto.
- **Auditorías.** La auditoría de los 82 juicios de la Etapa 7a y la de los 76 registros de memoria de la Etapa 6 las completó un modelo de lenguaje, no una persona (las dos planillas llevan el mismo auditor). La consigna pide que esa vara la ponga una persona, y es la falla metodológica más importante de la entrega: cada número de esas auditorías, y la calibración de los jueces, es hoy la lectura de un modelo. Las planillas ciegas para que una persona las rehaga, con las mismas filas, están en el zip, junto con las de los 25 resúmenes y las 20 reescrituras de la Etapa 5 (`AUDITORIA_HUMANA.md`); la Etapa 8 calcula el acuerdo persona-modelo cuando estén completas.
- **El eval set.** Las preguntas, los guiones y los escenarios se escribieron con el método de la sección 3 (temas primero, la respuesta esperada antes de buscar, la clave copiada del gold), las de identificador las derivó código, y el grupo revisó el conjunto completo antes de congelarlo.

## 11 · Revisión posterior a la entrega

Lo que cambió en esta versión, y por qué. El detalle de cada punto está en `decision_log.md`, sección 6.

| # | Lo que decía la entrega | Qué se corrigió | Dónde |
|---|---|---|---|
| 1 | La calibración era "contra el auditor", sin decir que era un modelo | Se dice; las conclusiones que dependen de él quedan marcadas; planillas ciegas para una persona | secciones 8 y 10, `AUDITORIA_HUMANA.md` |
| 2 | "Recuperar la versión correcta no alcanza si el generador no razona sobre intervalos" (nunca se probó) | La configuración elegida entrega versiones que no rigen; el filtro de fecha las saca (medido con las filas guardadas). La generación con el filtro, en la Etapa 8 | secciones 4, 5 y 9 |
| 3 | La mejor fila, elegida sobre `test` sin decirlo en el informe | Se dice, con su alcance; la Etapa 8 mide la regla en `dev` y su estabilidad | sección 4 |
| 4 | Etapa 5B con denominadores distintos (41 y 42) | Todos los brazos sobre los mismos 41 turnos; ningún conteo cambia | sección 6 |
| 5 | "El resumen no paga su costo" | Es este resumidor, que se congela en su tope; la Etapa 8 prueba un v2 con el mismo tope | sección 6 |
| 6 | Etapa 6 "dentro del ruido" | Con 3 escenarios no había cómo salir del ruido: es una limitación del diseño, y se dice así | sección 7 |
| 7 | Recortes justificados por costo, con 1,5 h usadas de 7-8 estimadas | Se dice que los recortes no se revisaron con el presupuesto real; la posición se corre en la Etapa 8 | sección 10 |

**La Etapa 8** (`08_followups.ipynb`) mide lo que la revisión dejó abierto, sin tocar la configuración, el eval set congelado ni las filas guardadas: exporta las métricas por ítem de la Etapa 3 (para recalcular su primaria desde el zip), aplica la regla de la mejor fila en `dev`, agrega a la Etapa 4 un brazo con el filtro de fecha y un oráculo con los tokens igualados, mide la posición del pasaje y prueba el resumidor v2. Antes de generar, verifica que el pipeline reproduzca exactamente las entregas guardadas de las Etapas 3, 4 y 5. Su diseño y sus reglas de lectura quedaron escritos antes de correrla (`decision_log.md`, sección 6). Está probada de punta a punta con datos falsos (`tests/test_followups_dryrun.py`); sus resultados necesitan la GPU y el índice de la corrida original (~280 llamadas, unos 10 minutos), y se agregan a este informe cuando se corra.
