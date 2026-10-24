# Temas del conjunto de evaluación

*Etapa 2, 2026-10-18.* Las preguntas, los guiones y los escenarios se escriben **desde esta lista**, no con el pasaje abierto (assignment, Etapa 2, primera trampa).

## Cómo se usa

1. **Se elige un tema.** La lista salió de tres fuentes, ninguna con el texto de los artículos:
   - la corpus card;
   - los **títulos** de las normas y de sus artículos;
   - lo que pregunta alguien que alquila, que es dueño de un piso o que vive en una comunidad de vecinos.
2. **Por tema, se escribe la pregunta**, como la diría un usuario en España (piso, casero, fianza, comunidad), y la **respuesta esperada**. Las dos cosas van antes de buscar en el corpus.
3. **Recién después se busca el gold** y se verifica la respuesta contra el texto. La clave de respuesta se copia del texto gold, y la pregunta no. Si el corpus dice otra cosa que la esperada, cambia la clave, nunca la redacción.
4. **Chequeo de contaminación y BM25** (`notebooks/02_eval.ipynb`). Lo marcado se reescribe **desde el tema**, no desde el pasaje.

Hay dos excepciones al paso 2:
- Las preguntas de **identificador** no se escriben: se derivan con `evalset.derive_identifier_items`.
- Las 5 preguntas del **chequeo a libro cerrado** de la Etapa 0 ya existían: van a `dev`.

## Temas

Columnas: **Conjuntos** = `U` preguntas de un turno · `G` guiones · `E` escenarios.

| # | Tema | Quién pregunta | Qué se puede preguntar | Conjuntos |
|---|---|---|---|---|
| 1 | Duración mínima del alquiler de vivienda y prórroga obligatoria | inquilino | cuánto dura como mínimo; si el casero puede no renovar al año; **cuánto duraba para un contrato firmado en otra fecha** (versiones) | U G E |
| 2 | Prórroga tácita al terminar la obligatoria | inquilino, propietario | qué pasa si nadie avisa; con cuánto preaviso | U G |
| 3 | Irse antes de tiempo (desistimiento del inquilino) | inquilino | desde cuándo puede irse; con cuánto aviso; indemnización | U G |
| 4 | El propietario necesita la vivienda | propietario | si puede recuperarla para él o un familiar, y con qué condiciones | U |
| 5 | Venta de la vivienda alquilada | inquilino | si el comprador tiene que respetar el contrato; derecho de compra preferente | U G |
| 6 | Muerte o separación del inquilino | familia del inquilino | quién puede seguir en la vivienda | U |
| 7 | Fianza y garantías | inquilino | cuánto se puede pedir para vivienda y para local; garantías adicionales; depósito de la fianza en la comunidad autónoma | U G E |
| 8 | Subidas de la renta | inquilino, propietario | cada cuánto y con qué índice se actualiza; topes; índice del INE | U G |
| 9 | Gastos e inmobiliaria | inquilino | quién paga la gestión inmobiliaria y la formalización del contrato; gastos de comunidad e IBI | U G |
| 10 | Obras | inquilino, propietario | conservación a cargo de quién; obras de mejora y subida de renta; obras del inquilino; adaptación por discapacidad | U G |
| 11 | Alquiler temporal (de temporada) | inquilino, propietario | duración máxima; qué pasa si se usa como vivienda habitual; conservación; sanciones (2026) | U G E |
| 12 | Locales de negocio | arrendatario de un local | fianza; cesión y subarriendo; indemnización al terminar | U G |
| 13 | Régimen sancionador de los alquileres (LAU, octubre de 2026) | propietario, plataforma | qué es infracción grave o muy grave; multas | U |
| 14 | Comunidad de propietarios | vecino, presidente | mayorías para limitar los pisos turísticos; obras de accesibilidad obligatorias; fondo de reserva; vecino moroso; actividades molestas | U G E |
| 15 | Gran tenedor | propietario | a partir de cuántas viviendas; qué obligaciones extra tiene | U E |
| 16 | Zonas de mercado residencial tensionado | inquilino, propietario | qué son; quién las declara; topes a la renta del nuevo contrato | U G E |
| 17 | Desahucios de personas vulnerables | inquilino vulnerable | si se suspenden y hasta cuándo; requisitos de vulnerabilidad | U G |
| 18 | Medidas de la pandemia (2020) | inquilino | moratoria de la deuda arrendaticia; prórroga extraordinaria | U |
| 19 | Bono Alquiler Joven | joven | requisitos (edad, ingresos, renta máxima); cuánto da; por cuánto tiempo | U G E |
| 20 | Plan Estatal de Vivienda 2026-2030 | inquilino, joven | qué ayudas al alquiler trae y quién las pide | U G |
| 21 | Registro único de arrendamientos | propietario de alquiler turístico o temporal | qué hay que registrar y para qué | U |
| 22 | Tipos de interés de los préstamos convenidos | comprador de vivienda protegida | cuál fue el tipo efectivo anual de un año dado (resoluciones anuales 2014-2026) | U |
| 23 | Deudores hipotecarios sin recursos | propietario endeudado | código de buenas prácticas; suspensión de lanzamientos | U |
| 24 | Barreras arquitectónicas | inquilino o propietario con discapacidad | si puede hacer obras de accesibilidad en la vivienda o el edificio | U G |
| 25 | Vivienda protegida (VPO) | comprador o inquilino de VPO | duración del régimen protegido; descalificación; precio máximo | U |
| 26 | Navarra: alquiler de habitaciones (Ley Foral 11/2026) | inquilino de una habitación | duración; renta; qué régimen se aplica | U G E |
| 27 | Illes Balears: vivienda y alquiler turístico | residente en Baleares | ley de vivienda balear; medidas urgentes | U E |
| 28 | Cataluña: contención de rentas e impuesto sobre viviendas vacías | inquilino, propietario en Cataluña | qué regula cada norma | U |
| 29 | Aragón: depósito de fianzas | propietario en Aragón | dónde y cuándo se deposita la fianza | U |
| 30 | Andalucía: Ley 5/2025 de Vivienda | residente en Andalucía | qué regula la ley nueva | U |

## Fuera del corpus: adyacentes, no absurdas

Cada una se confirma ausente mirando el top de BM25 (notebook, sección 6). **Regla de puntaje, decidida antes de cualquier resultado:** la respuesta es correcta si y solo si `can_answer = false`.

| Tema | Por qué es adyacente |
|---|---|
| Procedimiento de desahucio en la Ley de Enjuiciamiento Civil | se excluyó por densidad (0,13); el corpus tiene normas que la modifican, pero no el procedimiento |
| Ley catalana 18/2007 del derecho a la vivienda | excluida porque el BOE la marca desactualizada; hay otras 7 normas catalanas |
| Castilla-La Mancha | la única comunidad sin ninguna norma en el subconjunto |
| Un artículo que no existe (artículo 9 ter de la LAU) | el 9 y el 9 bis sí existen |
| El tipo de interés de los préstamos convenidos para 2027 | la serie llega hasta 2026: es "el mes que no tenés" |
| El Plan Estatal de Vivienda 2031-2035 | existen el 2018-2021, el 2022-2025 (Bono Alquiler Joven) y el 2026-2030 |
