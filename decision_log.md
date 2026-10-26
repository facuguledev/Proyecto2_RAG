# Registro de decisiones

Proyecto 2 · Retrieval & Memory Lab · UTDT. Lo que pide §2.4: el presupuesto y el plan iniciales (sección 1), qué se probó (2), qué se recortó y por qué (3), dónde se terminó respecto del plan y qué hizo cambiar de opinión (4), y cada edición del eval set congelado (5).

---

## 1 · Presupuesto y plan iniciales

### Presupuesto

**Tiempo.** Diez días, cuatro integrantes, unas 3 h por persona en días de semana y 5 h en fin de semana: **~150 h-persona**. Un 25% para coordinación, informe y empaquetado; quedan **~110 h** para construir y medir.

**VRAM (T4, 15 GB).** Todo entra junto con `num_ctx = 8.192`, con ~4 GB de margen:

| Pieza | Memoria |
|---|---|
| `qwen3:8b` (Q4_K_M) | ~5,2 GB |
| caché KV con `num_ctx = 8.192` | ~1,2 GB |
| buffer del servidor | ~0,5-1 GB |
| `BAAI/bge-m3` fp16 (pesos + activaciones) | ~2-3 GB |
| reranker `mmarco-mMiniLMv2-L12-H384` | ~0,25-0,5 GB |
| **Total** | **~10-11 GB** |

**Parámetros fijos del generador.** `num_ctx = 8.192` en cada llamada, `num_predict = 768`: quedan **7.424 tokens** de prompt, contados con el tokenizer de `Qwen/Qwen3-8B` y verificados contra `prompt_eval_count`.

**Embeddings.** Dos pasadas (sin prefijo y con prefijo), estimadas en ~5-8 min cada una en la T4.

**Llamadas al LLM** (supuestos: ~15 s por respuesta, ~4 s por reescritura, ~8 s por resumen o escritura en memoria, ~6 s por juicio):

| Etapa | Cuenta | Llamadas | Tiempo T4 |
|---|---|---|---|
| 3 · recuperación | sin LLM | 0 | ~15 min de embeddings |
| 4 · respuestas | 40 ítems × 3 filas + varianza + overflow | ~127 | ~32 min |
| 5A · con qué buscar | 45 turnos × 1 reescritura | 45 | ~3 min |
| 5B · qué conservar | 45 turnos × 5 brazos + resúmenes + sonda del resumen | ~335 | ~77 min |
| 6 · memoria | escritura de 12 sesiones + 28 sondas × 4 brazos | ~172 | ~36 min |
| Re-corrida de 4-6 con el mejor recuperador | una vez | ~680 | ~2,5 h |
| Smoke y depuración | `qwen3:1.7b`, conjuntos de juguete | — | ~1 h |
| 7 · juez y framework | | ~470 | ~1-1,5 h |
| **Total** | | **~1.800** | **~7-8 h de T4** |

Repartido en las cuatro cuentas de Colab, ~2 h de T4 por cuenta. El cuello de botella no era la GPU: era escribir y verificar el eval set (~8 h-persona) y leer los resultados.

### Plan

- **Hacer todas las etapas (0-7).** Si algo se atrasa, recortar desde el final, en este orden: 7b → 7a → el filtro por memoria de la Etapa 6 → el brazo extra de compresión de la 5B → las filas "+ padre" y "+ filtro de metadatos" de la Etapa 3.
- **Lo que no se recorta nunca:** las Etapas 1-2, las tres filas de la Etapa 4, la Parte A de la 5 y el experimento de historial de la 5B.
- **Orden:** Etapas 0, 1 y 2 en serie; congelar el eval set antes de cualquier resultado; después cuatro líneas en paralelo (Etapas 3, 4, 5 y 6), cada una contra un *stub* de la anterior; enchufar el mejor recuperador y re-correr una vez las Etapas 4-6; Etapa 7 e informe al final.
- **Cómo se lee cada resultado:** pareados (victorias/derrotas), test de signos y bootstrap por la unidad independiente (pregunta, guion o escenario). Con n = 30, una fila tiene unos ±12 puntos de ruido.

---

## 2 · Qué se probó

### Etapa 0 · El corpus

- **Corpus: BOE, legislación consolidada de vivienda y alquiler.** Regla del subconjunto: 10 materias de vivienda (438 normas, 16,1 M tokens) → densidad temática ≥ 0,45 → consolidación finalizada. Quedan **177 normas, 3,09 M tokens**, fijadas en `data/subset_v1.json`.
  - Sin el filtro de densidad entran los Presupuestos, la LEC y las leyes ómnibus, que tocan la vivienda en un solo artículo. El umbral es 0,45 y no 0,5 para que entre el RDL 11/2020 (0,47).
  - Descartados: toda la unión de materias (casi todo es texto que no es de vivienda), solo la materia 305 (sigue trayendo ómnibus), filtrar por título (pierde el RDL 11/2020) y solo normas estatales (se pierde la región).
- **Foto fijada.** Cada respuesta de la API se guarda con su sha256, y el corpus tiene su propio hash (`corpus_sha`). El gold apunta a esa foto. Re-descargar en cada corrida daría un corpus distinto en cada notebook.
- **El análisis de cada norma se pide en XML.** El JSON de la API conserva solo la última materia: 436 de 438 normas pierden materias, sin error.
- **Chequeo a libro cerrado: 0/5** con el modelo partido entre CPU y GPU, **1/5** con el modelo entero en GPU. El eval set mide recuperación, no preentrenamiento. Tres lecciones para la Etapa 4: una instrucción de sí/no global contamina las preguntas que no son de sí/no; al abstenerse, el schema se degenera ("can_answer" en `answer`); la contención tiene que ser por palabra completa ("10" estaba en "100").
- **Temperatura 0 no alcanza.** La misma pregunta dio "1 año" o "3 años" según el estado de la caché de prompts de Ollama. Por eso cada corrida arranca con el modelo descargado, los brazos van en un orden fijo y la Etapa 4 mide la tasa de cambios. Apagar la caché no se puede por llamada, y descargar el modelo entre llamadas sumaba más de una hora.
- **Diez documentos crudos leídos de punta a punta**, elegidos para cubrir los casos que marcan los números: la LAU, la Ley 12/2023, el RDL 29/2026, la LPH, la ley balear cuyo metadato miente y el "Artículo único" de 25 K tokens, entre otros (la lista, en `00_corpus_audit`, sección 10).

### Etapa 1 · De la fuente a los documentos

- **Un documento por norma, una sección por versión:** `section_id = doc_id#bloque@AAAAMMDD`, con vigencia `[valid_from, valid_to)`. Permite preguntas "a una fecha" y un id nunca cambia de contenido. Cuesta: 8.687 secciones y 4,78 M tokens (2,89 M vigentes), un 65% más de embeddings que con solo la versión vigente.
- **Qué es texto:** el texto citado por una ley modificadora (`blockquote.sangrado`) es texto; las notas de modificación van a `Section.notes`; la firma y la fórmula de promulgación se descartan; las tablas se linealizan ("Encabezado: valor"), así cada fila se entiende sola; las imágenes quedan como `[imagen omitida]`.
- **Normalización dirigida, no NFKC.** NFKC convierte "1.º" en "1.o" (2.911 veces). Sí se unifican "m²" y "m2", que para BM25 eran dos palabras.
- **La versión vigente no es la última.** La última versión del art. 10 de la LAU todavía no regía en la foto. Las anomalías de la fuente (versiones desordenadas, del mismo día, que nunca rigieron) quedan marcadas en `version_flags`.
- **Metadatos validados contra el cuerpo.** Tres correcciones al estado de la fuente (una ley balear derogada que figuraba vigente, 12 resoluciones anuales reemplazadas, un RDL sin fecha de fin): 135 `current` y 42 `superseded`. `fecha_publicacion` falla en 17 de 22 autonómicas chequeables, así que las fechas salen siempre de `valid_from`/`valid_to`.
- **Casi-duplicados:** 0 copias, 87 pares de versiones y 3 relacionados; 434 secciones con un gemelo en otra norma. No se borra ni se fusiona nada: los gemelos vigentes entran en el gold.
- **Build offline y determinista**, con el `corpus_sha` en el manifiesto.
- **Una revisión de código** encontró errores que no tiran excepción (migas de pan heredadas, fechas re-asignadas en silencio, un hash que dependía de la versión de NumPy). Se corrigieron con tests antes de congelar el eval set.

### Etapa 2 · El conjunto de evaluación

- **Contrato propio** en `src/evalset.py`. El gold es una lista de secciones (vale cualquiera), `as_of` solo valida el gold (el pipeline no lo lee) y las claves se copian del texto gold y se comparan por palabra completa.
- **Cómo se escribió:** desde `eval/topics.md` (30 temas), con la pregunta y la respuesta esperada antes de buscar. La clave se copia del texto; la pregunta, no.
- **Tamaños:** 40 preguntas (30 test), 10 guiones (49 turnos, 42 en test) y 4 escenarios (30 sondas, 22 en test). En test, 4 "sí" y 4 "no".
- **Pre-registrado:** una pregunta fuera del corpus es correcta si y solo si `can_answer = false`; si `topical` e `identifier` discrepan, manda `topical`.
- **Preguntas de identificador derivadas con código**, no escritas. El título del artículo tiene que ser único en todo el corpus: con la primera regla entraban títulos como "Entrada en vigor", que se adivinan sin recuperar nada.
- **Guiones y escenarios.** El último turno de cada guion vuelve al primero, así el historial importa. Cada escenario tiene un hecho que cambia (un par antes/después con veredictos opuestos) y uno que nunca se dijo. Los escenarios los escribió un autor que no construyó la Etapa 6.
- **Lo que cambió al verificar:** se descartaron preguntas con gold ambiguo (el desahucio de la LEC sobrevive citado en otras leyes; las barreras arquitectónicas están en tres normas) y se ajustaron claves al texto, nunca la pregunta.
- **El canal léxico se definió antes de congelar** (`src/lexical.py`), y lo comparten la contaminación y BM25: stopwords en español, "29/1994" como un solo término, sin tildes, con numpy. Por defecto, `bm25s` usaba stopwords en inglés y tiraba el "9" de "artículo 9".
- **Contaminación:** 0 de 141 textos marcados (el n-grama compartido más largo es de 4). Solo BM25 no pone primera ninguna sección gold.
- **Congelado antes de cualquier resultado**, después de que el grupo revisó el conjunto completo (sección 5).

### El repo del curso

- **Entró por merge, con su historia.** Un fork sería público, y copiar los archivos sin historia haría chocar cada arreglo del curso.
- **El contrato sigue siendo el nuestro.** El `schema.py` del curso rechaza las 8.687 secciones (no admite los campos de vigencia) y los tres archivos del eval set. Adoptarlo obligaba a reconstruir el corpus sin versiones y a volver a congelar.
- **Nuestros módulos se renombraron** (`doc_schema.py`, `artifact_io.py`) para no tapar a los del curso: con dos `store.py` en el path, Python usa uno y el otro desaparece sin aviso.
- **requirements:** el del curso sin tocar, y lo nuestro en `requirements-group.txt`. `bm25s` bajó a la versión del curso: el top-100 de las 40 preguntas es idéntico.
- **Etapas como `.py` con `# %%`** y la celda de Colab del curso. Ollama en la versión del curso (0.34.4).

### Etapa 3 · Recuperación (pre-registrada)

- **Índice:** 7.595 secciones. Quedan afuera las 67 versiones que nunca rigieron y los 979 encabezados.
- **Unidades:** padres de hasta 1.200 tokens de Qwen; hijos de hasta 350 tokens del reranker, **con el prefijo incluido**, así el corte es el mismo con y sin prefijo.
- **Prefijo por plantilla, sin LLM:** norma, sección, vigencia (mes y año, sin día) y migas de pan, de lo más a lo menos importante.
- **Ocho filas, una variable por fila:** `bm25`, `dense`, `+prefix`, `+rrf`, `+rerank`, `+parent`, `+date`, `+region`. Sin HyDE.
- **Entrega por presupuesto:** 3.000 tokens del generador en todas las filas; lo que no entra se saltea y se sigue.
- **Primaria: "clave entregada desde el gold"**, con equivalencia por texto entre versiones idénticas. *answer-delivered* suelto se descartó como primaria: en 29 de 78 preguntas la clave está en 50 secciones o más.
- **Mejor configuración:** la de más aciertos en test; si una fila con menos piezas no se distingue de ella (signos, p ≥ 0,05), gana la de menos piezas.
- **En local, torch con CUDA y bge-m3 en fp32.** torch con CUDA se instala a mano en el `.venv` (no va a `requirements*.txt`: Colab trae el suyo). La regla (fp16 solo en Colab) se fijó midiendo en la máquina de desarrollo, una GTX 1660 Super sin tensor cores, donde fp16 procesaba 3,0 textos/s y fp32 14,2. La corrida completa fue en otra máquina, una RTX 5070 Laptop, que sí tiene tensor cores: ahí la regla también dio fp32 (22 min de embeddings), mientras que en Colab da fp16. Una corrida en Colab no reproduciría los embeddings bit a bit, y por eso el dtype va en la clave de la matriz.
- **Resultado:** ninguna pieza supera p = 0,05. El prefijo se acerca (7-1). La mejor configuración es **`+prefix`**.

### Etapa 4 · Respuestas fundamentadas

- **El pasaje es el de la Etapa 3**, con el id del bloque adelante: con ids reales, citar un artículo que no se entregó es una falla visible.
- **Puntaje pre-registrado:** acierto = `can_answer = true` + una clave en `answer` + el veredicto correcto en las de sí/no. Los errores de llamada son su propia categoría. El puntaje se recalcula al leer.
- **Tres brazos en orden fijo:** libro cerrado → recuperación → oráculo. Varianza: 10 preguntas repetidas en frío y en orden inverso. Guarda: un prompt de 11.000 tokens con `num_ctx` 8.192, con y sin la guarda.
- **Resultado:** 5/30, 18/30 y 25/30. La varianza cambió el texto de 7 respuestas y el puntaje de ninguna.

### Etapa 5 · La conversación

- **Cada guion se reproduce solo, con su historial guionado.** Con los guiones concatenados, un presupuesto de historial de 500 o más solo descartaba guiones ajenos, y la tabla habría dicho que el historial no importa. Lo encontró la revisión de código, antes de correr.
- **5A:** cuatro consultas (último turno, conversación completa, reescritura del modelo, reescritura oráculo), acotadas a la ventana del reranker. Se leyeron 20 reescrituras.
- **5B:** pasajes + historial fijos en 1.500 tokens, con topes de historial de 100, 150 y 250, un resumen estilo MemGPT con tope 150 y el buffer completo como referencia. Se descartaron las divisiones fijas (80/20, 50/50): el historial no usado se perdía y los totales diferían.
- **La sonda del resumen** pregunta por el turno 1 y se puntúa con las claves que no están en la pregunta, con un control sin historial.
- **La consulta de la Parte B** se eligió con una regla fijada antes de la corrida: salió el último turno.
- **Resultado:** ninguna consulta ni tope supera el ruido. A esta escala, con qué se busca pesa más que qué historial se guarda.

### Etapa 6 · Memoria entre sesiones

- **Cinco brazos:** sin memoria, reproducción (3.000 tokens), reproducción en 600, almacén (hasta 6 recuerdos en 600) y almacén + filtro. Entre brazos solo cambia el espacio de memoria.
- **Tiempo simulado; invalidar, no borrar.** Un valor vigente por *subject*, y cada sonda ve el almacén de su fecha.
- **Escritura:** extracción y decisión con schema; la extracción ve los *subjects* vigentes, para no inventar otra etiqueta para el mismo dato.
- **Lectura:** recencia + importancia + relevancia, con la recencia en escala logarítmica. Con meses entre sesiones, 0,995^h es un escalón que deja todo el peso en la relevancia: es un apartamiento explícito de Generative Agents.
- **La memoria como filtro de la recuperación**, con una regla fija y sin LLM.
- **Resultado:** ninguna comparación supera el ruido. La memoria gana en las actualizaciones; la reproducción, en la extracción. La auditoría de los 76 registros: 60 bien, 10 mal y 6 que nunca se dijeron, y los chequeos automáticos no detectan ninguno.

### Etapa 7a · El juez

- **El juez recibe la pregunta, la clave y `answer`**, nunca el `analysis` ni el contexto. Una rúbrica por tipo. Una abstención en una pregunta con respuesta es incorrecta por regla, sin preguntarle al juez.
- **Dos jueces:** `qwen3:8b` (la misma familia que el generador) y `granite3.3:8b`. Se descartaron `llama3.1`, `gemma2` y `aya-expanse` porque sus tokenizers son gated y sin ellos no se puede contar el prompt.
- **Calibración:** 82 juicios (el 20%), con semilla, estratificados, con los desacuerdos sobrerrepresentados y la planilla ciega a los veredictos.
- **Resultado:** contra el auditor, la contención coincide en 73/82, `granite3.3:8b` en 69/82 y `qwen3:8b` en 42/82.

---

## 3 · Qué se recortó y por qué

- **7b (el loop en un framework).** Era lo primero del orden de recorte. Cuesta una segunda implementación y volver a correr todas las sondas de la Etapa 6, y lo que mostraría (la ventana de `/v1`, `think`, cuándo resume) ya lo controla el loop propio. Se prefirió 7a porque la rúbrica pide la auditoría del juez.
- **El experimento de posición del pasaje (Etapa 4).** Con 3.000 tokens y unos 10 pasajes, un resultado nulo no refutaría *Lost in the Middle*.
- **El análisis de §2.5 (la ventana real, hasta 20.480 tokens).** Fuera del plan desde el principio: obligaba a sacar el embedder y el reranker de la GPU, y a esta escala tendría el mismo problema que el de posición.
- **HyDE en la Etapa 3.** Fuera del plan desde el principio: eran 40 llamadas más.

---

## 4 · Dónde se terminó respecto del plan

- **La entrega se movió una semana.** El orden de recorte no cambió, pero alcanzó para hacer **el filtro por memoria** (Etapa 6), que estaba tercero en la lista de recortes y que la primera versión de la etapa había dejado afuera porque el filtro de la Etapa 3 todavía no existía.
- **La foto se perdió entre máquinas y no se podía volver a bajar:** en ocho días habían cambiado 6 de las 177 normas, entre ellas la Ley 12/2023. Se rehicieron las Etapas 0 y 1 con la misma regla (salieron las mismas 177 normas) y se volvió a congelar el eval set, con una sola edición (sección 5).
- **Todo corrió en una sola máquina local** (RTX 5070 de 8 GB, Ollama 0.34.4, la versión de Colab), no en Colab. La foto nueva estaba ahí, y así no se mezclan índices ni cachés de dos entornos. Lo que cambió por eso:
  - bge-m3 corre en fp32 fuera de Colab (la regla de precisión de la Etapa 3), y la clave de la matriz lo dice;
  - el embedder y el reranker no conviven con `qwen3:8b` en 8 GB: cada etapa recupera en lote y libera la GPU antes de generar;
  - la Etapa 6 falló una vez por memoria de la GPU (Ollama devuelve antes de liberar la VRAM), y se agregó `index_loader.wait_for_vram`.
- **La Etapa 5 se rediseñó antes de correr**, por la revisión de código: guiones de a uno, total igualado por turno y topes de historial (sección 2).
- **El presupuesto real fue mucho menor que el estimado:**

| Etapa | Duración | Llamadas al LLM |
|---|---|---|
| 3 · recuperación | 24 min (22 de embeddings) | 0 |
| 4 · respuestas | 11 min | 133 |
| 5 · conversación | 15 min | 233 (124 de caché) |
| 6 · memoria | 19 min | 242 (153 de lectura, 89 de escritura) |
| 7a · juez | 20 min | 814 (dos jueces) |
| **Total** | **~1,5 h** | **~1.420** |

  Contra ~1.800 llamadas y ~7-8 h de T4 estimadas. Una respuesta tardó 1-2,5 s, no los ~15 s supuestos, y la re-corrida de las Etapas 4-6 no hizo falta: los *stubs* se retiraron antes de la primera corrida completa.

---

## 5 · Ediciones del eval set congelado

**El congelado original**, antes de cualquier resultado y después de la revisión del grupo:

| Qué | Valor |
|---|---|
| `single_turn.jsonl` | `17d8be8337c88d064137deb084eb79d49a39519ff9de275091b627e828af2a8b` |
| `dialogues.jsonl` | `6549988119801c6df0b0807ce14753ec725aa0348370b53c8a4813be9f4ffc01` |
| `scenarios.jsonl` | `9cfb5e7d1f66dbd51a0f53790499d26a08bd411b39d62bcb6dd582d4a867d281` |
| `corpus_sha` | `5713999a5dab5705d3de99c7e7ef75be40181d90f46b94f0643375582e8ddb58` |
| key | `f75b1e0658626e8e` |

La adopción del repo del curso no lo tocó (verificado contra los hashes).

**La única edición: dl03.t5, por la foto nueva.** Con la foto nueva, todo el eval set validó salvo un gold: el art. 10 de la LPH tiene una versión nueva (BOE-A-2026-20528) que ya rige el día de la foto, así que la anterior dejó de ser la vigente. Se leyó la versión nueva contra la pregunta ("con mi edad, ¿puedo exigir que se haga el ascensor aunque la junta no lo apruebe?", veredicto "sí", clave "setenta años"): la obligación sigue igual, y lo nuevo es sobre las ayudas.
- **Cambia solo el gold:** `BOE-A-1960-10906#adiez@20231229` → `BOE-A-1960-10906#adiez@20261023`. La pregunta, el veredicto, la clave y la respuesta guionada quedan como estaban.
- **Por qué es legítimo:** todavía no existía ningún resultado completo de las Etapas 3-7.
- **Qué cuesta:** nada de lo construido con la foto vieja (índices, embeddings, cachés) se reusa, porque el `corpus_sha` cambió.

**El congelado nuevo:** `dialogues.jsonl` = `13f7c449…` (los otros dos archivos sin cambios), `corpus_sha` = `b01a2c0acc3508873861365d25b37d28ea0a8dacf7f868834ddb5207e28b526d`, **key = `99676007c42e6f6e`**, que va en el `run_spec` de cada fila de resultados.

---

## 6 · Revisión posterior a la entrega (2026-10-26)

Una revisión del trabajo entregado encontró siete puntos en los que el informe se apoyaba en algo que no se midió, o que se midió con un instrumento que no correspondía. Ninguno cambia el eval set congelado, la configuración elegida ni las filas guardadas de las Etapas 3-7.

### Lo que se encontró

1. **Las auditorías las hizo un modelo, no una persona.** `07_second_read` lo dice en una línea ("un auditor que también es un modelo, aunque más grande y de otra familia"), pero el informe hablaba de "el auditor" sin decirlo, y la columna `auditor` de las planillas del juez y de la memoria remitía al informe. La consigna pide auditar a mano. Es la falla más importante: la calibración de los jueces, la tasa de registros de memoria mal guardados y, con ella, la afirmación de que la contención es una cota inferior, son hoy la lectura de un modelo.
2. **La configuración elegida entrega versiones que no rigen.** `+prefix` busca entre todas las versiones; contado con las filas guardadas de la Etapa 3 (por id de sección), entrega otras versiones del artículo gold en 6 de las 7 preguntas `metadata` de test, y `+date`, en ninguna. El informe atribuía md02 y md05 solo al razonamiento temporal del generador ("recuperar la versión correcta no alcanza") sin haber corrido nunca la Etapa 4 con el filtro.
3. **La mejor fila se eligió sobre `test`.** Lo decía la notebook 03, no el informe.
4. **Etapa 5B con denominadores distintos** (41 y 42, por los errores de dl04.t4). Sobre los 41 turnos comunes ningún conteo cambia.
5. **El resumen se congela en su tope**: "no paga su costo" era una conclusión sobre ese resumidor, no sobre la estrategia.
6. **Etapa 6 sin poder.** Con 3 escenarios de test, el mejor resultado posible es 3-0 (p = 0,25): ninguna comparación podía superar el ruido.
7. **Los recortes no se revisaron con el presupuesto real** (1,5 h de GPU contra 7-8 estimadas): la posición del pasaje entraba de sobra.

### Lo que se hizo sin LLM

- El informe dice cada una de estas cosas en su sección (y en la nueva sección 11).
- `tools/recompute_tables.py`: recalcula las tablas desde el zip, sin el corpus y con la biblioteca estándar. Reproduce las tablas principales del informe (aciertos, comparaciones, calibración); sus reglas de puntaje están verificadas contra las del proyecto (`tests/test_recompute_tables.py`).
- `tools/make_human_audits.py` y `AUDITORIA_HUMANA.md`: las cuatro planillas ciegas para una persona (82 juicios, 76 registros, 25 resúmenes, 20 reescrituras), con las mismas filas y sin lo que dijo el modelo.

### La Etapa 8, pre-registrada (antes de correrla)

`08_followups.py`. Reglas, fijadas antes de cualquier resultado:
- **No cambia nada de lo anterior.** La mejor fila sigue siendo `+prefix` para las Etapas 4-7, el eval set sigue congelado (`99676007c42e6f6e`), y lo nuevo va a archivos nuevos (`results/08_*.jsonl`, `results/03_retrieval_metrics.jsonl`).
- **Antes de generar, el pipeline tiene que reproducir lo guardado:** la recuperación `+prefix` de las 40 preguntas, las entregas de `retrieval` y `oracle` de la Etapa 4, la recuperación `last_turn` de los 49 turnos y las entregas de `window_h150` de la Etapa 5, unidad por unidad y con la misma cuenta de tokens. Si algo no coincide, la etapa se corta.
- **Cada brazo nuevo cambia una cosa** contra un brazo guardado, con el mismo prompt, `num_ctx = 8.192`, `num_predict = 768`, temperatura 0, orden fijo y el modelo descargado al empezar cada brazo:
  - `retrieval_date` contra `retrieval`: `+prefix+date` en lugar de `+prefix`, con el mismo presupuesto de 3.000 tokens;
  - `oracle_padded` contra `oracle`: los padres gold y, después, distractores (lo que recupera `+prefix`, sin el gold) hasta 3.000 tokens;
  - posición: el padre gold primero, en el medio o último entre los mismos distractores, hasta 6.000 tokens, sobre las 26 preguntas con respuesta de test;
  - `summary_v2_h150` contra `summary_h150` y contra `window_h150`: el mismo tope de 150 tokens de historial y otro resumidor (gramática hasta 600 caracteres, para que no corte; un prompt que pide reescribir juntando lo anterior con lo nuevo, nunca copiar, y sacar primero lo respondido en turnos viejos). Corre primero en los 2 guiones de `dev` como chequeo, sin ajustar nada, y después en los 8 de test.
- **La regla de la mejor fila** se aplica también en `dev` (8 preguntas con respuesta) y en 2.000 remuestreos bootstrap de `test`, con semilla. Se reporta; no cambia la configuración.
- **Cómo se lee:** sobre `test`; por ítem en la Etapa 4 y en la posición, por guion en la 5B. p < 0,05 en el test de signos es un efecto; si no, una dirección. Para la posición, dicho antes: hacen falta 6 discordantes del mismo lado para p < 0,05, y un nulo con ~6.000 tokens no refuta *Lost in the Middle*.
- **Costo estimado:** ~280 llamadas, unos 10 minutos en la RTX 5070.

### Lo que queda pendiente

- **Correr la Etapa 8** en la máquina con el índice y la GPU, y agregar sus resultados al informe.
- **La auditoría humana** de las cuatro planillas (unas 3 horas). La del juez es la que pide la consigna.
