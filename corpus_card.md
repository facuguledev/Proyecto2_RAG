# Corpus card · Legislación española de vivienda y alquiler (BOE, legislación consolidada)

*Etapa 0 (2026-10-16) y Etapa 1 (2026-10-17); foto rehecha el 2026-10-24 (`decision_log.md`). Los números son los de esa foto y salen de `01_documents.ipynb`.*

## Fuente y versión fijada

- **Fuente:** API de datos abiertos del BOE, colección de legislación consolidada: `https://www.boe.es/datosabiertos/api/legislacion-consolidada`. Documentación: `https://www.boe.es/datosabiertos/documentos/APIconsolidada.pdf` (versión del 2 de septiembre de 2025).
- **Qué trae cada norma:** metadatos (rango, ámbito, departamento, fechas, vigencia, estado de consolidación, ELI), análisis (materias, notas, referencias a normas anteriores y posteriores) y el texto consolidado con **todas las versiones fechadas** de cada bloque (`fecha_vigencia` y la norma que introdujo cada versión).
- **Versión fijada:** la descarga del **2026-10-24** (la del 2026-10-16 se perdió y no se podía volver a bajar: 6 normas habían cambiado). Las respuestas crudas están en `data/raw/boe_cache/` (fuera de git), con fecha de descarga y sha256 en `index.jsonl`. La lista de normas con la `fecha_actualizacion` de cada una está en `data/subset_v1.json` (en git). La fuente cambia seguido (la LAU tuvo versiones nuevas el 1, el 2 y el 8 de octubre de 2026); `?from=20261024` lista lo actualizado desde la foto.
- **Verificación de la foto** (Etapa 1):
  - Son 1.763 archivos. Ninguno falta ni difiere de su sha256 (comparado sobre el texto: la foto del 2026-10-16 se había guardado con CRLF en Windows).
  - Las 177 normas siguen en la `fecha_actualizacion` fijada.
  - El build lee solo la caché (offline).
- **Hash del corpus:** `corpus_sha = b01a2c0acc3508873861365d25b37d28ea0a8dacf7f868834ddb5207e28b526d`, en `artifacts/full/documents.manifest.json`. Un build en otra máquina (Colab) tiene que dar el mismo.
- **Idioma:** español jurídico peninsular; las normas autonómicas están en su versión castellana publicada en el BOE.

## Licencia y atribución

- **Marco:** reutilización de la información del sector público, Ley 37/2007. Se permite el uso comercial y no comercial. Condiciones (aviso legal del BOE): no desnaturalizar el sentido de la información, no sugerir carácter oficial ni patrocinio del BOE, y conservar los metadatos y las fechas de actualización.
- **Atribución en obras derivadas:** "Basado en datos de la Agencia Estatal Boletín Oficial del Estado".
- **Aviso obligatorio:** los textos consolidados tienen **carácter meramente informativo**; solo la publicación en el diario oficial tiene validez jurídica. Hay que indicarlo en cada texto reutilizado: va en el encabezado del informe.
- Los datos crudos no se commitean (assignment §6.1).

## Regla del subconjunto

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

## Documento y sección

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

## Mapeo de estado

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

## Defectos conocidos

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

## Limpieza (Etapa 1)

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

## Casi-duplicados: política

- **Norma ↔ norma** (todos los pares; Jaccard ≥ 0,3 o contención ≥ 0,8): 90 pares, ninguno es copia (Jaccard ≥ 0,9).
  - **87 son versiones:** una de las dos está reemplazada (RDL ↔ ley de convalidación, RDL derogado ↔ re-decretado, resolución anual ↔ la siguiente).
  - **3 son normas relacionadas** que comparten texto y rigen las dos.
  - Quedan en `metadata.near_duplicates`, y la reemplazada lleva `superseded_by`. No se borra ni se fusiona ninguna.
- **Sección ↔ sección de otra norma** (Jaccard ≥ 0,7): 434 secciones tienen un gemelo en `Section.twins`. 65 de esos pares rigen los dos el 2026-10-24.
- **Versión ↔ versión del mismo bloque:** son secciones distintas, porque dan respuestas distintas.
- **Para la Etapa 2:**
  - El gold de una pregunta sin fecha es la versión vigente, más sus gemelos vigentes como conjunto.
  - El de una pregunta a una fecha es la versión vigente en esa fecha.

## Qué pone a prueba bien este corpus

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

## Chequeo a libro cerrado

`00_closed_book.ipynb`: 5 preguntas sin recuperación, una por categoría más una posterior al corte. Corrida local del 2026-10-16 (modelo 63% en GPU): **0/5**, con 2 abstenciones y 3 alucinaciones seguras. Repetida el 2026-10-24 con el modelo 100% en GPU (Ollama 0.34.4): **1/5**. El resultado se mantiene con dos variantes de formato (schema sin la instrucción de sí/no, y texto libre).

`qwen3:8b` no conoce ni los datos más difundidos de este corpus (prórroga de la LAU, fianza del art. 36, umbral de gran tenedor). Un eval set sobre este corpus mide recuperación, no recuerdo.
