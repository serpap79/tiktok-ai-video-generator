# tiktok-viral-generator

Agente que detecta temas de tendencia en **tech, IA y ciencia**, respalda cada afirmación con una
fuente verificable, redacta y produce un video corto en es-ES.

El recorte es deliberado. Ya existe código abierto maduro que transforma un tema en MP4 —
el [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) (MIT) lo hace muy
y bien. Lo que no existe es la mitad superior: **descubrir qué vale la pena decir y probar
que lo que se dice es cierto**. Esa mitad es la que este repositorio construye.

> Estado actual: **piloto automático (M6)** — cuatro publicaciones al día (09h, 12h, 16h y 19h
> hora de Madrid) sin comando manual, alternando entre corto (~25s) y largo (60-90s): radar
> de 8 fuentes → tema y tipo de contenido para el
> horario → investigación con fuente en cada hecho → **formato elegido por la información**  
> (video largo, corto o carrusel, con el motivo registrado) → guion y juez de familia  
> diferente → render propio (narración con pronunciación corregida, marca de agua,
> pista generada) → bandeja de entrada de TikTok. Consulte [Piloto automático](#piloto-automático).

## Por qué el grounding con citación no es adorno

Los Recompensas de Creadores de TikTok requieren videos de al menos 60 segundos, cuentan solo *visualizaciones
calificadas* y excluyen explícitamente el contenido "AI slop". El contenido asistido por IA es elegible
cuando es **original, transformador y etiquetado**.

Es decir: la etapa que extrae hechos con fuente y la etapa que exige un punto de vista propio
en el guion no son refinamientos de ingeniería — son lo que separa un canal monetizable de uno
no monetizable. Por eso son criterios de la rúbrica del juez, no sugerencias.

## Arquitectura

El agente es dueño del juicio; todo lo que es reemplazable queda detrás de una puerta.

```
radar → curador → investigador → redactor → juez → [Renderizador] → [Publicador]
                      ↑              ↑         ↑
                  [LLM] ────────────────────────┘
                      ↑
              memoria (SQLite: señales, temas, dossiers)
```

El `Renderizador` es una puerta porque la primera implementación delega al MoneyPrinterTurbo
ejecutado como servicio local. **No hacemos fork ni vendorizamos:** `scripts/setup_renderer.sh`
clona el proyecto en `.renderer/` (ignorado por git) y levanta su FastAPI. Nuestro agente pasa
guion y términos de búsqueda listos, lo que evita completamente el LLM del renderizador —
`task.py:generate_script` solo llama al LLM cuando `video_script` llega vacío.

## Restricción: costo cero

Todo el camino de producción funciona sin tarjeta de crédito:

| Capa | Implementación | Costo |
|------|----------------|-------|
| Radar | Hacker News, RSS es-ES y global, Hugging Face, Google Trends, Wikipedia | $0, sin clave |
| LLM | rutas por etapa sobre Gemini (3.8/3.5 flash, flash-lite) y Groq (gpt-oss, qwen) | $0, nivel gratuito |
| Narración | edge-tts, voces `es-ES-{Elvira,Alvaro}` + diccionario de pronunciación | $0, sin clave |
| Verificación de pronunciación | Whisper large-v3 en Groq | $0, nivel gratuito |
| Subtítulos karaoke | tiempos por palabra de edge-tts, ASS en la fuente de la marca | $0 |
| Material | Pexels (elegido por relevancia y oscuridad) | $0, registro gratuito |
| Pista | sintetizada localmente (numpy), según el clima del tipo de contenido | $0, sin derechos de autor |
| Montaje | ffmpeg (loudness −14 LUFS, ducking, tarjeta de gancho) | $0 |
| Publicación | API oficial de publicación de contenido de TikTok | $0 |

El `upload-post.com` que usa MoneyPrinterTurbo para publicar es un SaaS con nivel gratuito de
10 publicaciones/mes — por eso el publicador es nuestro.

## Ejecución

Requiere `ffmpeg`, `git` y [`uv`](https://docs.astral.sh/uv/) en el PATH.

```bash
# 1. dependencias del agente
uv sync

# 2. renderizador (clona ~340 MB en .renderer/, instala con Python 3.12)
#    La clave de Pexels es gratuita: https://www.pexels.com/api/
PEXELS_API_KEY=tu-clave ./scripts/setup_renderer.sh

# 3. inicia el renderizador (déjalo ejecutándose)
./scripts/setup_renderer.sh --serve

# 4. en otra terminal: renderiza el guion de referencia
uv run agent render --script fixtures/roteiro_manual.json
```

El comando imprime las dimensiones y la duración **medidas con `ffprobe`** y falla si el MP4 no
sale en 1080x1920 o fuera del rango de 60–90s. Acepta lo medido, no lo supuesto.

```bash
uv run pytest          # 322 pruebas, sin red y sin clave de LLM
uv run ruff check .
```

### El ciclo completo

```bash
uv run agent curate                            # elige el tema del día
uv run agent research                          # 3-5 fuentes, cada hecho con URL
uv run agent produce --out output/roteiro.json # escribe, juzga, revisa
uv run agent render --script output/roteiro.json
```

A partir de `research` se necesita una clave de LLM gratuita en `.env`
(`AGENT_GEMINI_API_KEY` o `AGENT_GROQ_API_KEY`); `uv run agent llm-health`
confirma que responde.

### Sin clave de Pexels

Se puede completar todo el camino sin ninguna clave, usando material local. Los clips
se generan con ffmpeg (no versionados — son ~9 MB de gradiente) y sirven solo para
probar la cadena de producción (narración, subtítulos, montaje) sin red:

```bash
./scripts/make_test_material.sh

AGENT_VIDEO_SOURCE=local \\
AGENT_LOCAL_MATERIALS='["fixtures/material/placeholder-1.mp4","fixtures/material/placeholder-2.mp4","fixtures/material/placeholder-3.mp4"]' \\
uv run agent render --script fixtures/roteiro_manual.json
```

Los archivos se suben por HTTP (`POST /api/v1/video_materials`) en lugar de copiarse
al disco del renderizador — es lo que mantiene la puerta válida si sale de esta máquina.

### Fedora: el ffmpeg del sistema no sirve

Fedora distribuye `ffmpeg-free`, compilado sin los codecs bajo patente — **no tiene
`libx264`**, que es el codificador predeterminado de MoviePy y del paso de concatenación de MPT. Sin
tratar esto, el renderizador falla al final, después de consumir todo el TTS y el montaje.

El `setup_renderer.sh` detecta y resuelve solo: el `imageio-ffmpeg`, que ya viene como
dependencia de MoviePy, trae un binario estático con `libx264`, y el
`utils.get_ffmpeg_binary()` de MPT honra `IMAGEIO_FFMPEG_EXE` antes del PATH. Ninguna
línea de su código se modifica.

## Piloto automático

```bash
./scripts/install_autopilot.sh                 # temporizadores 08:25/11:25/15:25/18:25 + persistencia
uv run agent autopilot-status --detail         # el día: espacios, motivos, cuota, tokens
journalctl --user -u 'seucanal-*' -f          # registro en vivo
uv run agent slot --slot 1900 --no-publish     # un espacio manual, sin publicar
uv run agent rerender --script <roteiro.json>  # vuelve a renderizar solo el video, sin gastar LLM
```

Cada horario es un proceso propio e idempotente (`slot_runs`, único por día+ranura):
disparo de nuevo no crea publicación duplicada, ranura con más de 2h de retraso se omite, y toda
falla se registra con el motivo y se notifica — nunca en silencio.

**Formato según la información.** Ninguna API pública dice qué se vuelve viral en TikTok, entonces la
selección es explícita y auditável: nota = hora (prior declarado: mañana de consumo
rápido, tarde para guardar, noche de atención larga) + tipo de contenido (tutorial y VS son
para guardar; historia y análisis requieren arco) + **lo que el dossier soporta** (medido: un
hecho con número cabe en 15s; 5 hechos con fecha sustentan 60–90s; elementos paralelos se vuelven
diapositive) − formato ya usado el día + desempeño medido del canal cuando haya muestra.
Un dossier con menos de 3 hechos no se vuelve largo ni carrusel — es regla, no nota.

**Cuota como recurso escaso.** El gemini-2.5-flash tiene 20 solicitudes/día en el nivel gratuito
(medido); cada modelo es su propio balde. El enrutador intenta la ruta de la etapa en orden,
espera el `retry-after` cuando el techo es por minuto, registra el agotado hasta el reset cuando es
por día, y descansa 10 min el modelo que responde 503. El juez prefiere una familia
diferente de la del redactor.

**Pronunciación.** La voz es-ES pronunciaba "Gemini" como "Zemini". El texto de voz recibe ortografía
fonética ("Djémini") y el subtítulo mantiene la original, con el tiempo alineado palabra a
palabra; el Whisper transcribe cada narración y acusa nombre propio que no reconoció.

**Límites de plataforma:** el video llega a la bandeja de entrada y se concluye en la app (publicar directamente
exige auditoría de la app); el carrusel solo sube por API desde un dominio verificado, entonces
sale como paquete listo para publicar.

## El radar

```bash
uv run agent radar
```

Cuatro fuentes gratuitas, ejecutándose en ~19s. Cada una falla aislada: la ventana de una tendencia es
de horas, entonces ninguna fuente caída derriba la recolección.

| Fuente | Unidad | Velocidad | Rol |
|--------|--------|-----------|-----|
| Hacker News (Algolia) | `puntos` | **nativa** — `puntos / edad` | primaria del nicho |
| Google Trends RSS | `búsquedas` | por diferencia | cobertura España + materiales ya asociados |
| Wikipedia pageviews | `visitas de página` | por diferencia | confirma interés real en es |
| GDELT DOC 2.0 | `artículos` | por diferencia | cobertura global, con disyuntor |

El radar **no normaliza** las unidades en una sola nota. Puntos de HN y visitas de página de
Wikipedia no son comparables, y convertir escalas diferentes a un solo número es juicio —
el juicio es trabajo del curador (M2). El radar recopila y mide.

La única magnitud comparable en forma es la **velocidad**, porque siempre es la misma derivada:
unidad por hora. Es `None` cuando se desconoce, nunca cero — `None` significa "no
medido" y cero significaría "medido y no se movió", que son afirmaciones diferentes y llevan a
decisiones diferentes.

Hacker News es la fuente primaria porque es la única gratuita que entrega velocidad **ya en
la primera recolección** (`puntos` + `created_at_i`). Las otras reportan nivel, no tasa, y
necesitan dos recolecciones para decir algo sobre movimiento — la serie queda en
SQLite.

## El curador

```bash
uv run agent curate
```

Tres puertas en orden, de lo más barato a lo más caro, y solo entonces el puntaje:

1. **Política** — bloquea antes de cualquier cálculo. Tema vetado no puede ganar en
   el ranking por estar subiendo rápido. Protege salud/medicamento, política partidaria,
   tragedia con víctima y menores.
2. **Nicho** — puerta, no adorno. Término fuera de tech/IA/ciencia se descarta incluso con
   velocidad altísima.
3. **Duplicata** — solo entre los que quedaron, porque comparar con el libro mayor cuesta.

El puntaje solo se calcula entre los sobrevivientes. Velocidad y volumen entran como
**percentil dentro de la propia fuente**: punto de Hacker News y visita de página de Wikipedia no
comparten escala, y sumar los números crudos haría que Wikipedia siempre ganara por tener
unidad mayor — no por tener mejor tema.

**Cada decisión se registra con motivo, incluidas las rechazadas.** Sin esto solo se sabe lo que
se eligió, nunca lo que se perdió, y calibrar el puntaje se vuelve adivinanza.

### Deduplicación: léxica, detrás de una puerta

El plan previa embeddings locales mediante `sentence-transformers`. Medido en 18/09/2026: ese
paquete arrastra torch con toda la pila CUDA — cudnn 527 MB, nccl 206 MB, cufft 204 MB,
cusolver 191 MB — más de 1,5 GB de bibliotecas NVIDIA en una máquina **sin GPU NVIDIA**. La
variante solo CPU no terminó de instalarse en 7 minutos.

Lo que necesita la deduplicación aquí es principalmente léxico: el mismo tema por
fuentes diferentes, o el mismo lanzamiento reformulado. Índice de Jaccard sobre tokens de
contenido lo resuelve, de forma determinista y comprobable, sin descarga y sin modelo.

Lo que **no** captura es paráfrasis sin palabra en común. Esa es la laguna que justificaría
embeddings — y, al estar detrás de la puerta `Deduplicator`, cambiar la técnica y medir contra la
misma base de temas es barato. Es el tipo de evidencia que el M5 produce.

## El investigador

```bash
uv run agent research                 # tema viene del curador
uv run agent research --topic "..." --url https://fuente/material
```

Es donde entra el primer LLM del proyecto. La etapa recibe un tema y devuelve un
dossier: de 3 a 5 fuentes leídas, y cada `Hecho` con afirmación, URL, nombre del medio
y el **trecho literal** que lo sustenta.

La decisión que sustenta el resto: **una llamada de modelo por fuente, y la URL estampada
por nosotros.** El modelo recibe el texto de una página y devuelve afirmaciones
sobre esa página; de dónde vino el texto es información que ya tenemos. Pedir
`source_url` al modelo invitaría al error más caro posible aquí — hecho real con
fuente cambiada, que parece anclado, pasa el juez y solo aparece cuando alguien
hace clic en el enlace. Con esto, "no existe `Hecho` sin URL verificable" deja de
depender de la honestidad del modelo y pasa a ser estructural.

### Descubrimiento de fuentes sin buscador de pago

No existe una API de búsqueda web gratuita que sirva: Google y Bing cobran, y raspar
SERP se rompe en una semana. Lo que existe de forma gratuita, y ya está en la pila:

| Estrategia | Lo que da | Costo |
|------------|-----------|-------|
| `news_item` del RSS de Google Trends RSS | título, medio y URL, ya asociados al tema | $0, viene en la recolección del radar |
| API de Algolia (`/items/{id}`) | el artículo detrás de la discusión de Hacker News | $0, sin clave |
| GDELT DOC 2.0 en `artlist` | quién escribió más sobre el tema | $0, sin clave, inestable |

Ninguna cubre todo tema — el artículo de HN no tiene material asociado, el tema de Trends
no pasa por Algolia, GDELT devuelve 429 con frecuencia — entonces las tres se ejecutan y
el informe dice cuáles fallaron. Límite de 2 páginas por dominio: cinco páginas
del mismo sitio no son cinco fuentes, y el juez no puede saber la diferencia
mirando solo el dossier.

### Dos puertas antes de que un hecho entre

Son determinísticos y no gastan token. Existen porque el modelo puede
parecer correcto estando equivocado, y porque pedirle que se audite no es
verificación.

1. **El trecho citado debe existir en la página.** El modelo devuelve, junto de cada
   afirmación, el pasaje literal que lo sustenta. Verificar el pasaje es `in` en una
   cadena — barato e imposible de engañar. Paráfrasis donde debería haber copia
   reprobara: no se puede saber si la afirmación es verdadera, y "no se puede saber"
   reprobara.
2. **Todo número de la afirmación debe estar en la fuente.** Número es lo que el guion
   usa para convencer, y es lo que un modelo inventa con más confianza.

La puerta numérica compara **dígitos**, no valores: es-ES escribe `5,9` e inglés
escribe `5.9` para lo mismo, y `1.500` es mil quinientos en portugués y uno
medio en inglés. Casar solo los dígitos resuelve ambos sentidos sin crear falso
negativo por coma.

El camino obvio — medir la superposición de vocabulario entre la afirmación y la página
— está equivocado aquí, y por una razón que solo aparece cuando se observan las fuentes
reales: casi todas están en inglés y la afirmación sale en es-ES. "Retiene el 98,2% del
rendimiento" y ""retiene el 98.2% del rendimiento"" no comparten una palabra. La
puerta reprobaría precisamente los hechos bien traducidos. **El número sobrevive a
la traducción; la palabra no.**

**Todo hecho descartado se registra con su motivo**, junto al dossier, y la CLI imprime
esos descartes antes del resultado. Es lo que indica si la puerta está calibrada o
estrangulando — sin esto, una puerta demasiado ajustada solo aparecería como "el
modelo está fallando hoy".

### Costo medido, no estimado

`Completion` carga tokens de entrada, de salida y latencia, y la tabla
`dossiers` guarda esto en columna propia. La evaluación del M5 compara nivel gratuito contra
modelo pago en la misma rúbrica, y esta comparación solo vale si el costo se mide en el
momento — el proveedor no devuelve consumo retrospectivo. En Gemini Flash, los tokens de
razonamiento interno entran en la cuenta: salen del mismo presupuesto, y ignorarlos
subestimaría el consumo.

### Los dos proveedores desde ya

| Proveedor | Formato estructurado | Papel |
|-----------|----------------------|-------|
| Gemini Flash (AI Studio) | `responseSchema` nativo | predeterminado; es-ES mejor |
| Groq (endpoint compatible con OpenAI) | `json_object` + esquema en el prompt | segundo brazo, mucho más rápido |

Una puerta con un único adaptador no es puerta, es indirección — por eso los dos entran
en conjunto, con prueba de conformidad. Las claves son gratuitas y están solo en `.env`:

```bash
AGENT_GEMINI_API_KEY=...   # aistudio.google.com/apikey
AGENT_GROQ_API_KEY=...     # console.groq.com/keys

uv run agent llm-health    # verifica qué ID de modelo aún responde, y a qué costo
```

`llm-health` existe porque el ID de modelo de nivel gratuito se descontinúa sin aviso, y
error aparecería en medio de una investigación, después de gastar tiempo leyendo páginas.
Gasta una llamada mínima y reporta quién respondió, en cuánto tiempo y por
cuántos tokens — verifica el artefacto, no la configuración.

## El redactor

```bash
uv run agent write --out output/roteiro.json
uv run agent render --script output/roteiro.json     # cierra el ciclo
```

La etapa lee el último dossier gravado y devuelve un `Guion` — el mismo contrato que
M0 ya renderiza, sin adaptación en el medio.

La separación que organiza la etapa: **el redactor corrige lo mecánico, el
juez juzga lo que es juicio.** Contar palabra, verificar si el término de búsqueda
tiene ASCII, verificar si el índice de hecho existe en el dossier, verificar si un
número en dígito de la narración está en algún hecho — nada de esto necesita de rúbrica,
y gastar una ronda de revisión del juez con error de conteo es quemar cuota de
nivel gratuito. Entonces el redactor tiene su propio bucle: hasta tres intentos, con el
**defecto medido devuelto al modelo en texto** ("la narración tiene 90 palabras y
necesita tener entre 150 y 225").

Solo llega al juez un guion que ya pasa todo lo verificable.

### El modelo señala el hecho, no lo reescribe

El guion no recibe los hechos como texto para reutilizar: recibe el dossier
**indexado**, y devuelve `used_facts: [0, 2]`. Los `Hecho` que van para el `Guion`
son los objetos del dossier, con la URL que el investigador estampó. Si el modelo
pudiera redactar el hecho, la afirmación del guion dejaría de ser rastreable a lo que
la fuente dice — que es todo el motivo de que exista el dossier.

### El rango de duración es requisito, no gusto

Un video por debajo de 60s no es elegible para las Recompensas de Creadores. El rango se estima aquí
por el ritmo de habla (2,5 palabras/s → 150 a 225 palabras) y **medida de verdad**
solo después del TTS, mediante `ffprobe`, en la aceptación del renderizador. La estimativa sirve
para no gastar un render completo descubriendo que el texto era corto; la medida
es la que manda.

Un límite conocido y asumido: la puerta de números solo ve lo escrito en
**dígito**. Una buena narración escribe el número por extenso para el TTS ("cinco coma
nueve gigabytes"), y verificar eso exigiría convertir numeral en portugués de
vuelta a dígito. Quien cubre ese caso es el criterio 2 de la rúbrica del juez, con el
dossier en mano.

### Todos los intentos se graban

Incluso cuando la primera ya pasa. Si toda ejecución gasta dos rondas en
el mismo defecto, el problema está en la instrucción y no en el modelo — y esto solo aparece
si el intervalo se registra en vez de descartarse en el éxito. La tabla
`scripts` guarda el número de intentos, las violaciones de cada una, el costo en
tokens y el `dossier_id` de origen.

Este vínculo con el dossier es lo que permitirá, en el M5, vincular la retención al material
que generó el guion. Sin él, "este video fue mejor" nunca se volverá "esta fuente
rinde mejor".

## El juez

```bash
uv run agent judge --script fixtures/roteiro_sem_fonte.json   # rechaza sin gastar nada
uv run agent produce --out output/roteiro.json                # escribe, juzga, revisa
```

Rúbrica de 7 criterios, 0–2 cada, corte en 11/14:

| # | Criterio | De dónde sale la nota |
|---|----------|-----------------------|
| 1 | el anzuelo abre un vacío en los primeros segundos | juzgado |
| 2 | toda afirmación tiene fuente en el dossier | **medida** (dígitos) + juzgado |
| 3 | duración hablada entre 60 y 90s | **medida** |
| 4 | punto de vista propio, no resumen de noticia | juzgado |
| 5 | ningún término de la lista de política | **medida** |
| 6 | es-ES hablado, frases cortas | juzgado |
| 7 | cierre con CTA que no sea "sigue para más" | juzgado |

### Lo que no se le pregunta al modelo

La duración es conteo de palabra. Preguntar a un LLM cuántos segundos tarda el texto
en hablarse es cambiar una medida por una conjetura. La política ya tiene un filtro escrito contra el
radar real, y el juez **reutiliza el mismo filtro** que bloqueó el tema — dos listas
divergirían con el tiempo, y el guion sería juzgado por una regla
distinta de la que decidió el tema.

La fuente tiene ambas mitades: la cuenta de dígitos es nuestra, la lectura es del modelo.
Número inventado es atrapado sin costo; afirmación que va más allá del dossier necesita de
un lector.

### Medir es barato, juzgar cuesta cuota

El orden es el mismo del curador: cuando la medida ya rechaza un criterio de
requisito, **el modelo no es llamado**. Los criterios de lectura se marcan
como *no evaluados* — cero allí significa "no sé", no "mal", y por eso no
vuelven al redactor como corrección.

Esto tiene un efecto práctico bueno: la fixture adversarial del M3 se rechaza **sin
ninguna clave de API y sin gastar un token**, lo que permite verificar ahora:

```
$ uv run agent judge --script fixtures/roteiro_sem_fonte.json
modelo    : no consultado (rechazó en la medida)
  [2/2] medido   duración        210 palabras, ~84s estimados
  [2/2] medido   política       ningún término de la lista de política
  [0/2] medido   fuente        número citado sin respaldo en el dossier: 12, 40
  [0/2] omitido  anzuelo      no evaluado: el guion rechazó antes en fuente
  ...
RECHAZADO: 4/14 (corte 11)
  veto en fuente: y requisito, no calidad — nota en otros criterios no compensa
costo     : 0 tokens
```

La fixture es el guion de referencia del M0 con **una sola** afirmación injertada
("12 mil GPUs y 40 millones de dólares", números que no existen en ningún hecho).
Todo lo demás es idéntico, a propósito: así no hay forma de que ella sea rechazada por
escritura mala, duración o política. La prueba de esto aún entrega al juez un parecer de
nota máxima en los cinco criterios juzgados — el escenario más favorable posible al
guion — y él la rechaza igual.

### La suma no decide sola

Aprobar requiere tres cosas: **11/14**, **ningún criterio en cero** y **ningún veto**.

La suma sola permite compensación errónea. Un guion que es puro resumen de
noticia (0 en punto de vista) llegaría a 12 de 14 con el resto perfecto y
pasaría — siendo exactamente el "AI slop" que las Recompensas de Creadores excluyen. Y fuente,
duración y política son **veto**: fallan en requisito, no en calidad, y nota
high en otros criterios no compra aprobación.

### Revisión: máximo dos

El bucle `redactor → juez → redactor` está fuera de las dos etapas
(`agent/pipeline.py`). Si el redactor supiera del juez, pasaría a escribir para
la rúbrica y la opinión dejaría de ser independiente; si el juez supiera del
redactor, juzgaría el intento y no el texto.

El límite de dos revisiones no es arbitrario: a partir de la tercera, lo que suele
ocurrir no es el guion mejorando — es el modelo empezando a cambiar de tema para
agradar la rúbrica. Mejor rechazar con el motivo anotado y elegir otro tema.

Las notas de revisión van ordenadas por costo: **veto primero**. No sirve de nada
mejorar el anzuelo de un guion que cita número sin fuente.

## Marca del canal — ejemplo (todas las capas obedecen)

Vector en `brand/brand.json` (transcrito del manual, revisar cada 90 días),
fonte única mediante `agent/brand/`. Lo que vale en todo video: fondo #0A0A0C, **1
acento** (verde #39FF88, ciano #00E0FF solo en ruptura), sin rostro, gancho ≤12
palabras, 1 número/frase, sin emoji/lema, promesa siempre cumplida. 6 pilares de
contenido (IA NOTICIAS, HECHO, ANÁLISIS, CÓMO HACER, 2030, VS) con fórmulas de gancho
 y CTA propios; subtítulo con las 5 hashtags `#ia #inteligenciaartificial
#tecnologia #ai #seucanal`. Las diapositivas y el avatar provienen de la paleta con
Space Grotesk/Plex Mono (`brand/assets/`).

> **La marca a continuación es un ejemplo — el método es lo que importa.** "Tu Canal",
> la Iris y el Theo son placeholders: cambia por nombre, paleta, handle y
> presentadores de tu propio canal y el agente entero obedece sin cambiar
> código. `brand/brand.json` muestra dónde cambiar cada decisión, y
> `uv run agent brand-avatar` imprime el prompt-modelo para generar tu
> presentador de IA.

La diapositiva del carrusel distribuye el contenido en cinco capas, diseño que vino de
una referencia de 20/09/2026: chip de la marca + contador en la parte superior, baño de acento
en diagonal sobre la foto, título grande con la **última línea (o última
palabra) en el acento**, apoyo marcado por un cuadrado, y pie de página con regla, arroba
 y la acción (deslizar, o guardar al final). El diseño anterior era honesto y
vacío: contador suelto, título y apoyo en el mismo peso, un tercio del cuadro sin
nada. El baño **solo entra con foto** — sobre el casi-negro puro se vuelve
verde dominante en todo el cuadro, y ahí la regla de la marca ya se ha ido.

Narración es-ES en CPU con Piper/VITS (`agent voice-*`): 176 palabras en ~5s
(RTF ~0,05–0,1 medido en i5 sin GPU). Biblioteca en `agent/voice/library.py`,
modelos en `data/voices/` (ignorados por git), muestras en `output/vozes/`.

| Modelo | ES-ES | Voces | Clonaje | Acento | CPU | Licencia | Veredicto |
|--------|-------|-------|---------|--------|-----|----------|-----------|
| Piper + razo | sí, tech | 1 masc. | no | neutro | RTF 0,08 | MIT | **base del canal** |
| Piper + faber-medium | sí | 1 | no | neutro | RTF 0,10 | MIT/CC0 | variedad |
| Piper + jeff-medium | sí | 1 | no | neutro | RTF 0,07 | MIT | suspense/misterio (lento: +20% duración) |
| Piper + edresson-low | sí | 1 | no | — | — | MIT | **rechazado**: nasal corrupta + arrastrada |
| F5-TTS es-es | sí | clonaje | sí | — | GPU | CC-BY-NC | **rechazado**: sin uso comercial |
| XTTS-v2 | sí | clonaje | sí | — | GPU | CPML | **rechazado**: sin uso comercial |
| Bark | sí | varias | no | — | RTF 10–20x | MIT | **rechazado**: 60s = 10–20min en CPU |
| Kokoro-82M | no verificado | — | — | — | — | Apache-2.0 | pendiente de voz es-ES |

Lagunas honestas: **sin voz femenina abierta** (no existe en los catálogos Piper;
edge-tts tiene, pero es externo sin SLA) y **sin acento regional** (ningún modelo
abierto reproduce; variedad = cambio de locutor). Estilos (`documental`,
`suspense`, `misterio`, `noticia`...) son interpretación sobre los 3 timbres
(velocidad/pausa/ruido), con `voice-list` mostrando todo. Marcado en el texto:
`[PAUSA CURTA/MEDIA/LONGA]`, `*énfasis*` (micro-pausa — el Piper no tiene SSML).

## Estudio de voces (TTS local, $0)

## Presentador animado (Iris y Theo, $0)

Los pilares de análisis, tutorial, hecho, VS y noticia se abren con un presentador
del elenco de la marca. No es un retrato pegado en la esquina: la silueta completa
aparece y **ella habla**, sintetizada cuadro a cuadro en
`agent/render/presenter.py`, sin ningún modelo y sin ninguna llamada de API.

Cuatro capas de movimiento, todas determinísticas:

| Capa | De dónde sale | Cómo se mide |
|------|---------------|--------------|
| **Boca** | envoltura RMS de la narración, ataque 25 ms / relajación 55 ms | perfil de desplazamiento de la mandíbula anclado en los puntos de la cara; apertura dibujada entre los labios |
| **Parpadeo** | sorteo con semilla fija (el tema), 2,6–6 s | párpado comprimiendo la zona ceja→pestaña |
| **Cabeza** | balanceo lento + acento en las sílabas fuertes | dos capas de máscara **complementarias** (cabeza + tronco = alfa original), pivote dentro del pecho |
| **Escenografía** | tiempo de palabra del TTS | grande en la llamada → canto durante el cuerpo → vuelta en el cierre |

El recorte y los puntos de la cara se generan **una vez**, fuera del bucle:

```bash
uv run --no-project --with rembg --with mediapipe --with pillow --with numpy \\
    python scripts/make_presenter_cutouts.py
```

Esto lee `brand/assets/presenters/source/<id>.jpg` y graba `<id>.png` (RGBA,
torso con degradé en la base) + `<id>.json` (ojos, base de la nariz, boca, mentón,
cuello, pivote). `rembg` y `mediapipe` suman más de 400 MB con onnxruntime y
opencv detrás — el agente funciona tres veces al día y nunca necesita esto, entonces
estos archivos quedan en un entorno efímero de `uv` y el repositorio sigue siendo ligero. Misma
decisión que la de `sentence-transformers`.

Para ver el artefacto sin gastar un slot ni una ronda de LLM:

```bash
uv run agent presenter-preview --script fixtures/roteiro_manual.json --presenter iris
# presentador.mp4 + hoja.png (llamada, recorrido, cuerpo, cierre)
```

Medido el 20/09/2026 en un i5 sin GPU: 40 ms por cuadro en la llamada (avatar al 51%
de la altura) y 22 ms en el canto, ~46 s de síntesis para un video de 67 s, capa de
5–9 MB. El costo es por píxel de **salida**, por lo que cada capa solo carga el
rectángulo donde tiene tinta — recortar así eliminó el 40% del tiempo de cada cuadro.

### Trampas de pago aquí

- **Ventana de `crop` no anima a nadie.** El primer diseño paseaba un
  rectángulo de 340x640 sobre el PNG: el recorte alfa no servía para nada, lo que
  aparecía era una caja de cara, y la persona dentro quedaba inmóvil. El
  prueba `test_o_quadro_nao_e_um_retangulo` lo atrapa midiendo el contorno
  (cara estreita, hombro amplio, esquinas vacías).
- **El recorte necesita el torso entero.** El `rembg` cortaba al sujeto en los
  bordes de la imagen de origen. Ahora el torso se corta *sobre* donde el hombro
  toca el borde y la última franja se vuelve transparente por degradé.
- **El labio de arriba no se mueve.** Rampa del subnasal al mentón hacía que el labio
  inferior bajara un tercio de lo debido y la boca se volviera una línea. El
  desplazamiento sube de cero a lleno justo debajo de la línea de los labios — y este
  tramo estirado *es* la boca abriendo.
- **Normalizar por p92 deja la boca escancarada.** Medido en la narración real de
  67 s: mediana de apertura 0,68. Descontando el suelo de ruido (p20) y llevando el
  medio hacia abajo (gamma 1,3), la mediana cae para 0,37, con 31% de los cuadros de
  boca casi cerrada — que es como se distribuye el habla real.
- **El VP9 con alfa de este ffmpeg devuelve el alfa opaco.** Probado antes de
  elegir. RGBA sin pérdida en `qtrle` da 1569 MB para 67 s; la capa sale en
  **dos pistas h264** (color pre-multiplicado + máscara en gris) con 8,7 MB, y
  el compuesto final difiere en promedio 0,7 de 255 por píxel. Pre-multiplicado
  porque, con alfa directo, el color salta de la cara al negro en el borde y el h264
  borra ese salto en franja oscura.
- **`presenter_for(pilar)` con el objeto en lugar del id devuelve `None`.** El
  ejecutor reasignaba `pilar` para el `ContentPillar` y pasaba el objeto para
  `presenter_for` y `accent_for`: ambos caían en el patrón **en silencio**.
  Ningún presentador entraba en ningún video, y todo video salía en verde
  incluso en los pilares de acento ciano. Bloqueado en `TestApresentadorNoSlot`.

## Hitos

| | Hito | Estado |
|---|------|--------|
| M0 | Puerta Renderer + aceptación medida del MP4 | **completado** |
| M1 | Radar (HN, Trends, Wikipedia, GDELT) | **completado** |
| M2 | Curador: puntaje, filtro de política, deduplicación por memoria | **completado** |
| M3 | Investigador + redactor + juez con rúbrica | **completado** |
| M4 | Publicador (TikTok, bandeja de entrada, etiqueta AIGC) | **piloto real en 19/09/2026** — MP4 de 69s en la bandeja (`SEND_TO_USER_INBOX`, completado en la app); falta aprobación de la app en producción |
| M5 | Evaluación: nivel gratuito x nivel gratuito en la misma rúbrica + métricas de la publicación | **en curso** — `agent eval` (offline) y `agent metrics-record` listos; 1ª ronda real abajo |
| M6 | Piloto automático: 3 espacios/día, formato según la información, enrutador de cuota, render propio | **activado en 19/09/2026** — systemd de usuario; carrusel aún publicado manualmente |
| M7 | OpenRouter delante de la ruta, escalón pago por tarea, presentador animado, carrusel con jerarquía | **en curso (20/09/2026)** — ~$0,01/video medido; presentador sintetizado cuadro a cuadro ($0); falta dominio verificado para publicar carrusel por API |

## Evaluación (M5) — primeros números

Primera ronda real en 19/09/2026, tema único (Bonsai 2 27B), `uv run agent eval`.
El brazo pagado (Claude) está fuera de alcance — sin API pagada, no hay número a
publicar. La comparación es nivel gratuito x nivel gratuito:

| | gemini (`gemini-2.5-flash`) | groq (`openai/gpt-oss-120b`) |
|---|-----------------------------|------------------------------|
| investigador | 4 hechos, 0 descartes, 2525 in / 489 out | 4 hechos, 0 descartes, 2537 in / 684 out |
| redactor | OK, 180 palabras (~72s), 1 intento | **fallo 2x**: `400 json_validate_failed` (no generó JSON válido antes del tope de tokens) |
| juez (mismo guion) | 12/14 APROBADO | **14/14 APROBADO** — 2 puntos más generoso (anzuelo 2x1, punto de vista 2x1) |

`produce` gemini punto a punto: APROBADO 12/14, 186 palabras, 2038 in / 648 out.
Ronda 3 formatos en 19/09/2026 (tema Bonsai 2, `output/rodada/`):
`corto` groq 42 palabras APROBADO 13/14; `carrusel` groq 8/8 con 5 diapositivas
1080x1920; `largo` groq 176 palabras APROBADO 12/14. Enrutamiento medido: gemini
estoura el corto (60–80 palabras, sin converger); groq escribe los tres de 1ª.
Humanizer (`writer/humanize.py`, adaptado del `blader/humanizer` MIT) se ejecuta
despues de la aceptación mecánica: escaneo de indicios es-ES, 1 reescritura detenida por
grounding, original intacto si se rompe número o rango.
Métricas de la publicación (`agent metrics-record`) se leen en la app manualmente: la bandeja de entrada
no expone endpoint de métricas y la API de Investigación está restringida a investigación académica.

## Límites conocidos

Publicados aquí a propósito, no ocultos.

- **Ninguna API pública dice qué se vuelve viral *en TikTok*.** El Centro Creativo no expone API
  y la API de Investigación está restringida a investigación académica. Detectamos tema de tendencia en internet
  e inferimos. Son cosas distintas, y el proyecto no pretende lo contrario.
- **El RSS de Google Trends no es API oficial** y puede cambiar sin aviso. La API oficial seguía en
  alfa con acceso por inscripción en ago/2026; el `pytrends` fue archivado en abr/2025.
- **No hay manera de anexar audio en tendencia de TikTok por API** — solo la Biblioteca de Música Comercial o
  el audio del propio archivo. Es un techo real de alcance.
- **El GDELT devuelve 429 con frecuencia** sin clave (2 de 3 intentos en las pruebas), por eso
  entra con retroceso y cortacircuitos, nunca como fuente única.
- **Reddit requiere OAuth**: JSON y RSS devuelven 403. Quedó fuera del M1.
- **El edge-tts usa el endpoint de lectura en voz alta de Edge.** Gratis, sin contrato, puede
  romperse. El plan de respaldo es el Kokoro-82M local (Apache 2.0, voces es-ES), que se ejecuta
  en CPU.
- **Los adaptadores de LLM fueron ejercitados contra la API real en 19/09/2026.**
  `llm-health` responde en ambos niveles gratuitos; investigador, redactor (gemini) y
  juez (gemini y groq) funcionaron de verdad en el tema Bonsai 2. El redactor groq
  (`gpt-oss-120b`) falla en `json_validate_failed` — registrado en la Evaluación arriba,
  no aquí.
- **La puerta de trecho rechaza paráfrasis.** Cuando el modelo reescribe donde debería
  copiar, el hecho cae incluso si es verdadero. El error es asimétrico a propósito:
  dossier corto con motivo anotado es calibrable, dossier lleno de hecho débil no.
  Los descartes se almacenan en la base precisamente para que esta calibración haya dado resultado.
- **La puerta de números no verifica unidad ni contexto.** "5,9 GB" coincide con una
  página que dice "5,9 millones de descargas". La alternativa sería pedir al propio
  modelo que se audite, lo cual no es verificación.
- **La puerta de números del redactor solo ve dígito.** Número inventado escrito
  por extenso ("nueve veces menor") escapa de ella; es el criterio 2 de la rúbrica del juez
  que cubre ese caso.
- **No hay búsqueda web gratuita.** El descubrimiento de fuentes depende de lo que el Trends
  RSS ya haya asociado, del enlace detrás del artículo de HN y del GDELT — que devuelve 429 con
  frecuencia. Un tema fuera de esas tres rutas puede no producir dossier alguno.
- **La viralidad no es predecible offline.** La rúbrica mide calidad de guion, no
  resultado. La única señal real es la retención post-publicación, y eso es lo que el M5 recopila.
- **Ninguna estimación de ingresos se publicará** hasta haber un número medido del propio canal.
- **La aceptación verifica el artefacto, no el código de salida.** La primera ejecución real
  produjo un MP4 mudo que pasaba en dimensión y duración: el adaptador descargaba
  `combined_videos` (la concatenación solo de video) en lugar de `videos` (el corte con narración y
  subtítulos). `has_audio` se mide con `ffprobe` y es `False` por defecto, para que el silencio
  nunca sea el predeterminado aprobado.

## Licencia y créditos

El renderizador es el [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)
de Harry, bajo licencia MIT. No está incluido en este repositorio: se clona en el momento de
setup y se consume por HTTP.