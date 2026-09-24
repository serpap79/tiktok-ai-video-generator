# Circuito Cero

Agente que detecta asuntos en tendencia en **tech, IA y ciencia**, ancla cada
afirmacion a una fuente verificable, escribe y produce video corto en
castellano (espana).

El recorte es deliberado. Ya existe open source maduro que transforma un tema
en MP4 — el [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)
(MIT) lo hace muy bien. Lo que no existe es la mitad de arriba: **descubrir de
que vale la pena hablar, y probar que lo que se habla es verdad.** Esa mitad es
la que construye este repositorio.

> Estado actual: **piloto automatico (M6)** — cuatro posts al dia (10:00, 13:00,
> 17:00 y 20:00 de Espana) sin comando manual, alternando **corto (~25s) y largo
> (60-90s)**: radar de 8 fuentes → tema y tipo de contenido para la hora →
> investigacion con fuente en cada hecho → **formato elegido por la informacion**
> (video largo, corto o carrusel, con el motivo grabado) → guion y juez de
> familia diferente → render propio (narracion con pronunciacion corregida,
> leyenda de la marca, pista generada) → inbox de TikTok. Ve
> [Piloto automatico](#piloto-automatico).

## Por que el grounding con cita no es adorno

El Creator Rewards de TikTok exige video de al menos 60 segundos, cuenta solo
*qualified views* y excluye explicitamente el contenido "AI slop". El contenido
asistido por IA es elegible cuando es **original, transformativo y etiquetado**.

O sea: la etapa que extrae hechos con fuente y la etapa que exige punto de vista
propio en el guion no son refinamiento de ingenieria — son lo que separa un canal
monetizable de un canal desmonetizado. Por eso son criterios de la rubrica del
juez, no sugerencias.

## Arquitectura

El agente es dueno del juicio; todo lo sustituible vive detras de un puerto.

```
radar → curador → investigador → guionista → juez → [Renderer] → [Publisher]
                        ↑             ↑          ↑
                    [LLM] ─────────────────────────┘
                        ↑
                    memoria (SQLite: senales, temas, dossiers)
```

El `Renderer` es un puerto porque la primera implementacion delega en
MoneyPrinterTurbo corriendo como servicio local. **No hacemos fork ni
vendorizamos:** `scripts/setup_renderer.sh` clona el proyecto en `.renderer/`
(git-ignored) y arranca su FastAPI. Nuestro agente pasa guion y terminos de
busqueda listos, lo que sortea por completo su LLM — `task.py:generate_script`
solo llama al LLM cuando `video_script` llega vacio.

## Restriccion: coste cero

Todo el camino de produccion corre sin tarjeta de credito:

| Capa | Implementacion | Coste |
|---|---|---|
| Radar | Hacker News, RSS espanol y global, Hugging Face, Google Trends, Wikipedia | $0, sin clave |
| LLM | rutas por etapa sobre Gemini (3.8/3.5 flash, flash-lite) y Groq (gpt-oss, qwen) | $0, free tier |
| Narracion | edge-tts, voces `es-ES-{Elvira,Alvaro,Ximena}` + diccionario de pronunciacion | $0, sin clave |
| Comprobacion de pronunciacion | Whisper large-v3 en Groq | $0, free tier |
| Leyenda karaoke | tiempos por palabra de edge-tts, ASS en la fuente de la marca | $0 |
| Material | Pexels (elegido por relevancia y oscuridad) | $0, registro gratis |
| Pista | sintetizada en local (numpy), por clima del tipo de contenido | $0, sin derechos de autor |
| Montaje | ffmpeg (loudness −14 LUFS, ducking, tarjeta del gancho) | $0 |
| Publicacion | Content Posting API oficial de TikTok | $0 |

El `upload-post.com` que MoneyPrinterTurbo usa para publicar es un SaaS con free
tier de 10 posts/mes — por eso el publicador es nuestro.

## Ejecutando

Requiere `ffmpeg`, `git` y [`uv`](https://docs.astral.sh/uv/) en el PATH.

```bash
# 1. dependencias del agente
uv sync

# 2. renderizador (clona ~340 MB en .renderer/, instala con Python 3.12)
#    La clave de Pexels es gratis: https://www.pexels.com/api/
PEXELS_API_KEY=tu-clave ./scripts/setup_renderer.sh

# 3. arranca el renderizador (dejalo corriendo)
./scripts/setup_renderer.sh --serve

# 4. en otra terminal: renderiza el guion de referencia
uv run agent render --script fixtures/guion_manual.json
```

El comando imprime dimensiones y duracion **medidas con `ffprobe`** y falla si el
MP4 no sale en 1080x1920 o fuera de la franja de 60–90s. Aceite medido, no
presumido.

```bash
uv run pytest          # 690 tests, sin red y sin clave de LLM
uv run ruff check .
```

### El ciclo entero

```bash
uv run agent curate                            # elige el tema del dia
uv run agent research                          # 3-5 fuentes, cada hecho con URL
uv run agent produce --out output/guion.json   # escribe, juzga, revisa
uv run agent render --script output/guion.json
```

Desde `research` hace falta una clave de LLM gratuita en el `.env`
(`AGENT_GEMINI_API_KEY` o `AGENT_GROQ_API_KEY`); `uv run agent llm-health`
confirma que responde.

### Sin clave de Pexels

Se puede cerrar el camino entero sin ninguna clave, usando material local. Los
clips se generan con ffmpeg (no versionados — son ~9 MB de gradiente) y sirven
solo para probar la cadena de produccion (narracion, leyenda, montaje) sin red:

```bash
./scripts/make_test_material.sh

AGENT_VIDEO_SOURCE=local \
AGENT_LOCAL_MATERIALS='["fixtures/material/placeholder-1.mp4","fixtures/material/placeholder-2.mp4","fixtures/material/placeholder-3.mp4"]' \
uv run agent render --script fixtures/guion_manual.json
```

Los archivos suben por HTTP (`POST /api/v1/video_materials`) en vez de copiarse
al disco del renderizador — es lo que mantiene el puerto valido si se va de esta
maquina.

### Fedora: el ffmpeg del sistema no sirve

Fedora distribuye `ffmpeg-free`, compilado sin los codecs con patente — **no
tiene `libx264`**, que es el encoder por defecto de MoviePy y del paso de
concatenacion del MPT. Sin tratarlo, el render muere al final, despues de gastar
todo el TTS y el montaje.

El `setup_renderer.sh` lo detecta y resuelve solo: el `imageio-ffmpeg`, que ya
viene como dependencia de MoviePy, trae un binario estatico con `libx264`, y el
`utils.get_ffmpeg_binary()` del MPT honra `IMAGEIO_FFMPEG_EXE` antes del PATH.
Ninguna linea de su codigo se modifica.

## Piloto automatico

```bash
./scripts/install_autopilot.sh                 # timers 09:25/12:25/16:25/19:25 + linger
uv run agent autopilot-status --detail         # el dia: slots, motivos, cuota, tokens
journalctl --user -u 'circuitocero-*' -f       # log en vivo
uv run agent slot --slot 2000 --no-publish     # un slot a mano, sin publicar
uv run agent rerender --script <guion.json>    # rehace solo el video, sin gastar LLM
```

Cada horario es un proceso propio e idempotente (`slot_runs`, unico por
dia+slot): disparar otra vez no se convierte en post duplicado, el slot con mas
de 2h de retraso se salta, y todo fallo queda registrado con el motivo y avisado
— nunca en silencio.

**Formato por la informacion.** Ninguna API publica dice que se viraliza en
TikTok, asi que la eleccion es explicita y auditable: nota = horario (prior
declarado: manana de consumo rapido, tarde guardable, noche de atencion larga) +
tipo de contenido (tutorial y VS se guardan; historia y analisis piden arco) +
**lo que el dossier aguanta** (medido: hecho unico con numero cabe en 15s; 5
hechos con fecha sostienen 60–90s; items paralelos se vuelven slides) − formato
ya usado en el dia + rendimiento medido del canal cuando haya muestra. Dossier
con menos de 3 hechos no se convierte en largo ni carrusel — es regla, no nota.

**Cuota como recurso escaso.** El gemini-2.5-flash tiene 20 peticiones/dia en el
free tier (medido); cada modelo es un cubo propio. El router prueba la ruta de la
etapa en orden, espera el `retry-after` cuando el techo es por minuto, graba el
agotado hasta el reset cuando es por dia, y saca 10 min el modelo que responde
503. El juez prefiere la familia distinta a la del guionista.

**Pronunciacion.** La voz es-ES decia "Gemini" como "Zemini". El texto de la voz
recibe grafia fonetica ("Yemini") y la leyenda mantiene la original, con el
tiempo alineado palabra a palabra; el Whisper transcribe cada narracion y acusa
el nombre propio que no reconocio.

**Limites de plataforma:** el video llega a la inbox y se completa en la app
(postar directo exige auditoria de la app); el carrusel solo sube por API desde
dominio verificado, asi que sale como paquete listo para postar.

## El radar

```bash
uv run agent radar
```

Ocho fuentes gratuitas. Cada una falla aislada: la ventana de una tendencia es de
horas, asi que ninguna fuente caida tira la recogida.

| Fuente | Unidad | Velocidad | Papel |
|---|---|---|---|
| Hacker News (Algolia) | `points` | **nativa** — `points / edad` | primaria del nicho |
| Google Trends RSS | `searches` | por diferencia | cobertura Espana + articulos ya asociados |
| Wikipedia pageviews | `pageviews` | por diferencia | confirma interes real en castellano |
| GDELT DOC 2.0 | `articles` | por diferencia | cobertura global, con disyuntor |

El radar **no normaliza** las unidades en una nota unica. Puntos del HN y
pageviews de la Wikipedia no son comparables, y convertir escalas distintas en un
numero solo es juicio — el juicio es trabajo del curador (M2). El radar recoge y
mide.

La unica grandeza comparable en forma es la **velocidad**, porque es siempre la
misma derivada: unidad por hora. Es `None` cuando se desconoce, nunca cero —
`None` significa "no medi" y cero significaria "medi y no se movio", que son
afirmaciones distintas y llevan a decisiones distintas.

Hacker News es la fuente primaria porque es la unica gratuita que entrega
velocidad **ya en la primera recogida** (`points` + `created_at_i`). Las otras
reportan nivel, no tasa, y necesitan dos recogidas para decir algo sobre
movimiento — la serie queda en SQLite.

## El curador

```bash
uv run agent curate
```

Tres puertas en orden, de la mas barata a la mas cara, y solo despues el score:

1. **Politica** — bloquea antes de cualquier calculo. Tema vetado no puede ganar
   en el ranking por estar subiendo rapido. Guarda salud/medicamento, politica
   partidista, tragedia con victima y menores.
2. **Nicho** — puerta, no aderezo. Termino fuera de tech/IA/ciencia se descarta
   incluso con velocidad altisima.
3. **Duplicado** — solo entre los que quedan, porque comparar con el libro cuesta.

El score solo se calcula entre los supervivientes. Velocidad y volumen entran como
**percentil dentro de la propia fuente**: punto del Hacker News y pageview de la
Wikipedia no comparten escala, y sumar los numeros crudos haria ganar siempre a la
Wikipedia por tener unidad mayor — no por tener asunto mejor.

**Toda decision se graba con motivo, incluidas las rechazadas.** Sin eso solo se
sabe lo que se eligio, nunca lo que se perdio, y calibrar el score se vuelve un
chute.

### Deduplicacion: lexica, detras de un puerto

El plan preveia embeddings locales via `sentence-transformers`. Medido el
18/09/2026: ese paquete arrastra el torch con la stack CUDA entera — cudnn
527 MB, nccl 206 MB, cufft 204 MB, cusolver 191 MB — mas de 1,5 GB de librerias
NVIDIA en una maquina **sin GPU NVIDIA**. La variante CPU-only no termino de
instalar en 7 minutos.

Lo que la deduplicacion necesita cazar aqui es mayoritariamente lexico: la misma
noticia por fuentes distintas, o el mismo lanzamiento reformulado. Jaccard sobre
tokens de contenido lo resuelve, de forma determinista y testeable, sin descarga
y sin modelo.

Lo que **no** caza es parafraseo sin palabra en comun. Esa es la laguna que
justificaria embeddings — y, por estar detras del puerto `Deduplicator`, cambiar
la tecnica y medir contra la misma base de temas es barato. Es el tipo de
evidencia que el M5 produce.

## El investigador

```bash
uv run agent research                 # el tema viene del curador
uv run agent research --topic "..." --url https://fuente/articulo
```

Es donde entra el primer LLM del proyecto. La etapa recibe un tema y devuelve un
dossier: de 3 a 5 fuentes leidas, y cada `Fact` con afirmacion, URL, nombre del
medio y el **pasaje literal** que lo sostiene.

La decision que sostiene el resto: **una llamada de modelo por fuente, y la URL
la estampamos nosotros.** El modelo recibe el texto de una pagina y devuelve
afirmaciones sobre esa pagina; de donde salio el texto es informacion que ya
tenemos. Pedir `source_url` al modelo invitaria al error mas caro posible aqui —
hecho real con fuente cambiada, que parece anclado, pasa al juez y solo aparece
cuando alguien hace clic en el enlace. Con eso, "no existe `Fact` sin URL
verificable" deja de depender de la honestidad del modelo y pasa a ser
estructural.

### Descubrimiento de fuentes sin buscador de pago

No existe API de busqueda web gratuita que sirva: Google y Bing cobran, y raspar
SERP se rompe en una semana. Lo que existe de gratis, y ya esta en la stack:

| Estrategia | Que da | Coste |
|---|---|---|
| `news_item` del Google Trends RSS | titulo, medio y URL, ya asociados al tema | $0, viene en la recogida del radar |
| API del Algolia (`/items/{id}`) | el articulo detras de la discusion del Hacker News | $0, sin clave |
| GDELT DOC 2.0 en `artlist` | quien mas escribio sobre el tema | $0, sin clave, inestable |

Ninguna cubre todo tema — item del HN no tiene articulo asociado, tema del Trends
no pasa por el Algolia, GDELT devuelve 429 con frecuencia — asi que las tres
corren y el informe dice cuales fallaron. Hay techo de 2 paginas por dominio:
cinco paginas del mismo sitio no son cinco fuentes, y el juez no tiene como saber
la diferencia mirando solo el dossier.

### Dos puertas antes de que un hecho entre

Las dos son deterministas y no gastan token. Existen porque el modelo puede
parecer cierto estando equivocado, y porque pedirle que se audite no es
verificacion.

1. **El pasaje citado tiene que existir en la pagina.** El modelo devuelve, junto
   a cada afirmacion, el pasaje literal que la sostiene. Comprobar el pasaje es
   `in` sobre una string — barato e imposible de enganar. Parafraseo donde habia
   que copiar reprueba: no se puede saber si la afirmacion es verdadera, y "no se
   puede saber" reprueba.
2. **Todo numero de la afirmacion tiene que estar en la fuente.** El numero es lo
   que el guion usa para convencer, y es lo que un modelo inventa con mas
   confianza.

La puerta numerica compara **digitos**, no valores: Espana escribe `5,9` y el
ingles escribe `5.9` para lo mismo, y `1.500` es mil quinientos en castellano y
uno y medio en ingles. Casar solo los digitos resuelve los dos sentidos sin crear
falso negativo por la coma.

El camino obvio — medir solapamiento de vocabulario entre la afirmacion y la
pagina — esta mal aqui, y por un motivo que solo aparece cuando se miran las
fuentes reales: son casi todas en ingles y la afirmacion sale en castellano.
"Retiene 98,2% del rendimiento" y *"retains 98.2% of performance"* no comparten
una palabra. La puerta reprobaria justamente los hechos bien traducidos. **El
numero sobrevive a la traduccion; la palabra no.**

**Todo hecho derribado se graba con el motivo**, junto al dossier, y la CLI
imprime los descartes antes del resultado. Es lo que dice si la puerta esta
calibrada o estrangulando — sin eso, una puerta demasiado apretada solo apareceria
como "el modelo esta mal hoy".

### Coste medido, no estimado

`Completion` lleva tokens de entrada, de salida y latencia, y la tabla
`dossiers` lo guarda en columna propia. El eval del M5 compara free tier contra
modelo pago en la misma rubrica, y esa comparacion solo vale si el coste se mide
en el momento — el proveedor no devuelve consumo retroactivo. En Gemini Flash,
los tokens de razonamiento interno entran en la cuenta: salen del mismo
presupuesto, e ignorarlos subestimaria el consumo.

### Los dos proveedores desde ya

| Proveedor | Formato estructurado | Papel |
|---|---|---|
| Gemini Flash (AI Studio) | `responseSchema` nativo | patron; castellano mejor |
| Groq (endpoint compatible con OpenAI) | `json_object` + schema en el prompt | segundo brazo, mucho mas rapido |

Puerto con un unico adaptador no es puerto, es indireccion — por eso los dos
entran juntos, con test de conformidad. Las claves son gratuitas y viven solo en
el `.env`:

```bash
AGENT_GEMINI_API_KEY=...   # aistudio.google.com/apikey
AGENT_GROQ_API_KEY=...     # console.groq.com/keys

uv run agent llm-health    # comprueba que id de modelo aun responde, y a que coste
```

`llm-health` existe porque el id de modelo de free tier se descontinua sin aviso,
y el error apareceria en medio de una investigacion, despues de gastar tiempo
leyendo paginas. Gasta una llamada minima y reporta quien respondio, en cuanto
tiempo y por cuantos tokens — comprueba el artefacto, no la configuracion.

## El guionista

```bash
uv run agent write --out output/guion.json
uv run agent render --script output/guion.json     # cierra el ciclo
```

La etapa lee el ultimo dossier grabado y devuelve un `Script` — el mismo
contrato que el M0 ya renderiza, sin adaptacion en medio.

La separacion que organiza la etapa: **el guionista corrige lo mecanico, el juez
juzga lo que es juicio.** Contar palabra, comprobar si el termino de busqueda
esta en ASCII, comprobar si el indice de hecho existe en el dossier, comprobar si
un numero en digito de la narracion esta en algun hecho — nada de eso necesita
rubrica, y gastar una ronda de revision del juez con error de conteo es quemar
cuota de free tier. Entonces el guionista tiene su propio lazo: hasta tres
intentos, con el **defecto medido devuelto al modelo en texto** ("la narracion
tiene 90 palabras y necesita tener entre 150 y 225").

Solo llega al juez un guion que ya pasa en todo lo verificable.

### El modelo apunta el hecho, no lo reescribe

El guion no recibe los hechos como texto para reutilizar: recibe el dossier
**indexado**, y devuelve `used_facts: [0, 2]`. Los `Fact` que van al `Script` son
los objetos del dossier, con la URL que el investigador estampo. Si el modelo
pudiera redactar el hecho, la afirmacion del guion dejaria de ser rastreable a lo
que la fuente dice — que es todo el motivo de que el dossier exista.

### La franja de duracion es requisito, no gusto

Video por debajo de 60s no es elegible al Creator Rewards. La franja se estima
aqui por el ritmo de habla (2,5 palabras/s → 150 a 225 palabras) y se **mide de
verdad** solo despues del TTS, por el `ffprobe`, en el aceite del renderizador.
La estimacion sirve para no gastar un render entero en descubrir que el texto era
corto; la medida es la que manda.

Un limite conocido y asumido: la puerta de numeros solo ve lo que esta escrito en
**digito**. La narracion buena escribe el numero por extenso para el TTS ("cinco
coma nueve gigabytes"), y comprobar eso exigiria convertir numeral en castellano
de vuelta a digito. Quien cubre ese caso es el criterio 2 de la rubrica del juez,
con el dossier en la mano.

### Todos los intentos quedan grabados

Incluso cuando el primero ya pasa. Si toda ejecucion gasta dos rondas en el mismo
defecto, el problema esta en la instruccion y no en el modelo — y eso solo aparece
si el intervalo se registra en vez de descartarse en el exito. La tabla `scripts`
guarda el numero de intentos, las violaciones de cada uno, el coste en tokens y
el `dossier_id` de origen.

Ese vinculo con el dossier es lo que permitira, en el M5, ligar retencion al
material que genero el guion. Sin el, "este video fue mejor" nunca se convierte en
"esta fuente rinde mejor".

## El juez

```bash
uv run agent judge --script fixtures/guion_sin_fuente.json   # reprueba sin gastar nada
uv run agent produce --out output/guion.json                 # escribe, juzga, revisa
```

Rubrica de 7 criterios, 0–2 cada uno, corte en 11/14:

| # | Criterio | De donde sale la nota |
|---|---|---|
| 1 | hook abre hueco en los primeros segundos | juzgada |
| 2 | toda afirmacion tiene fuente en el dossier | **medida** (digitos) + juzgada |
| 3 | duracion hablada entre 60 y 90s | **medida** |
| 4 | punto de vista propio, no resumen de noticia | juzgada |
| 5 | ningun termino de la lista de politica | **medida** |
| 6 | castellano hablado, frases cortas | juzgada |
| 7 | cierre con CTA que no sea "sigue para mas" | juzgada |

### Lo que no se le pregunta al modelo

Duracion es conteo de palabra. Preguntarle a un LLM cuantos segundos dura el
texto hablado es cambiar una medida por un chute. Politica ya tiene filtro escrito
contra el radar real, y el juez **reutiliza el mismo filtro** que veto el tema —
dos listas divergirian con el tiempo, y el guion pasaria a juzgarse por una regla
distinta de la que decidio el asunto.

Fuente tiene las dos mitades: la cuenta de digitos es nuestra, la lectura es del
modelo. Numero inventado se caza sin coste; afirmacion que va mas alla del
dossier necesita un lector.

### Medir es barato, juzgar cuesta cuota

El orden es el mismo del curador: cuando la medida ya reprueba en un criterio de
requisito, **el modelo no se llama**. Los criterios de lectura quedan marcados
como *no evaluados* — cero ahi significa "no se", no "malo", y por eso no vuelven
al guionista como correccion.

Esto tiene un efecto practico bueno: la fixture adversarial del M3 se reprueba
**sin ninguna clave de API y sin gastar un token**, y se puede comprobar ahora:

```
$ uv run agent judge --script fixtures/guion_sin_fuente.json
modelo    : no consultado (reprobo en la medida)
  [2/2] medido   duracion        210 palabras, ~84s estimados
  [2/2] medido   politica        ningun termino de la lista de politica
  [0/2] medido   fuente          numero citado sin respaldo en el dossier: 12, 40
  [0/2] saltado  hook            no evaluado: el guion reprobo antes en fuente
  ...
REPROBADO: 4/14 (corte 11)
  veto en fuente: es requisito, no calidad — nota en los otros criterios no compensa
coste     : 0 tokens
```

La fixture es el guion de referencia del M0 con **una unica** afirmacion
injertada ("12 mil GPUs y 40 millones de dolares", numeros que no existen en
ningun hecho). Todo lo demas es identico, a proposito: asi no hay como reprobarla
por escrita mala, duracion o politica. El test de eso aun le da al juez un
informe de nota maxima en los cinco criterios juzgados — el escenario mas
favorable posible al guion — y reprueba incluso asi.

### La suma no decide sola

Aprobar exige tres cosas: **11/14**, **ningun criterio a cero** y **ningun veto**.

La suma sola permite compensacion equivocada. Un guion que es puro resumen de
noticia (0 en punto de vista) llegaria a 12 de 14 con el resto perfecto y pasaria
— siendo exactamente el "AI slop" que el Creator Rewards excluye. Y fuente,
duracion y politica son **veto**: fallan en requisito, no en calidad, y nota alta
en los otros criterios no compra aprobacion.

### Revision: como maximo dos

El lazo `guionista → juez → guionista` vive fuera de las dos etapas
(`agent/pipeline.py`). Si el guionista supiera del juez, pasaria a escribir para
la rubrica y el informe dejaria de ser independiente; si el juez supiera del
guionista, juzgaria el intento y no el texto.

El techo de dos revisiones no es arbitrario: a partir de la tercera, lo que suele
pasar no es que el guion mejore — es que el modelo empieza a cambiar de asunto
para agradar a la rubrica. Mejor reprobar con el motivo grabado y elegir otro
tema.

Las notas de revision van ordenadas por coste: **veto primero**. No sirve de nada
mejorar el hook de un guion que cita numero sin fuente.

## Marca del canal — Circuito Cero (todas las capas obedecen)

Vector en `brand/brand.json` (transcrito de la guia, revisar cada 90 dias),
fuente unica via `agent/brand/`. Lo que vale en todo video: fondo #0A0A0C, **1
acento** (verde #39FF88, cian #00E0FF solo en ruptura), sin rostro, gancho ≤12
palabras, 1 numero/frase, sin emoji/muletilla, promesa siempre pagada. 7 pilares
de contenido (NEWS, DATO, ANALISIS, TUTORIAL, FUTURO, VS, HISTORIA) con formula
de gancho y CTA propios; leyenda con las 5 hashtags `#ia #inteligenciaartificial
#tecnologia #ai #circuitocero`. Slides y avatar salen de la paleta con Space
Grotesk/Plex Mono (`brand/assets/`).

> **El metodo es lo que importa.** "Circuito Cero", Atlas y Nova son la
> identidad actual: cambia nombre, paleta, handle y presentadores en
> `brand/brand.json` y el agente entero obedece sin tocar codigo. El json
> senala donde se cambia cada decision, y `uv run agent brand-avatar` imprime
> el prompt-modelo para generar tu presentador de IA.

El slide del carrusel distribuye el contenido en cinco capas, dibujo que vino de
una referencia del 20/09/2026: chip de la marca + contador arriba, bano de
acento en diagonal sobre la foto, titulo grande con la **ultima linea (o la
ultima palabra) en acento**, apoyo marcado con un cuadrado, y pie con regla,
arroba y la accion (deslizar, o guardar en el ultimo). El layout anterior era
honesto y vacio: contador suelto, titulo y apoyo al mismo peso, un tercio del
cuadro sin nada. El bano **solo entra con foto** — sobre el casi negro puro se
vuelve verde dominante en el cuadro entero, y ahi la regla de la marca ya se fue.

Narracion es-ES en CPU con Piper/VITS (`agent voice-*`): 176 palabras en ~5s
(RTF ~0,05–0,1 medido en un i5 sin GPU). Biblioteca en `agent/voice/library.py`,
modelos en `data/voices/` (git-ignored), muestras en `output/voces/`.

| Modelo | es-ES | Voces | Clonacion | Acento | CPU | Licencia | Veredicto |
|---|---|---|---|---|---|---|---|
| Piper + davefx | si, tech | 1 masc. | no | castellano neutro | RTF 0,08 | MIT | **base del canal** |
| Piper + carla | si | 1 | no | neutro | RTF 0,10 | MIT/CC0 | variedad |
| Piper + karen | si | 1 | no | neutro | RTF 0,07 | MIT | suspense/misterio |
| Piper + es-carlfm-x_low | si | 1 | no | — | — | MIT | **rechazado**: calidad de radio vieja |
| F5-TTS es | si | clonacion | si | — | GPU | CC-BY-NC | **rechazado**: sin uso comercial |
| XTTS-v2 | si | clonacion | si | — | GPU | CPML | **rechazado**: sin uso comercial |
| Bark | si | varias | no | — | RTF 10–20x | MIT | **rechazado**: 60s = 10–20min en CPU |
| Kokoro-82M | no verificado | — | — | — | — | Apache-2.0 | pendiente de voz es-ES |

Lagunas honestas: **sin voz femenina abierta** (no existe en los catalogos Piper;
edge-tts la tiene, pero es externo sin SLA) y **sin acento regional** (ningun
modelo abierto lo reproduce; variedad = cambio de locutor). Los estilos
(`documental`, `suspense`, `misterio`, `noticia`...) son interpretacion sobre los
3 timbres (velocidad/pausa/ruido), con `voice-list` mostrandolo todo. Marcas en
el texto: `[PAUSA CORTA/MEDIA/LARGA]`, `*enfasis*` (micro-pausa — el Piper no
tiene SSML).

## Estudio de voces (TTS local, $0)

## Presentador animado (Atlas y Nova, $0)

Los pilares de analisis, tutorial, dato, VS y noticia abren con un presentador
del elenco de la marca. No es un retrato pegado en la esquina: la silueta entera
aparece y **habla**, sintetizada cuadro a cuadro en `agent/render/presenter.py`,
sin ningun modelo y sin ninguna llamada de API.

Cuatro capas de movimiento, todas deterministas:

| Capa | De donde sale | Como se mide |
|---|---|---|
| **Boca** | envolvente RMS de la narracion, ataque 25 ms / relajacion 55 ms | perfil de desplazamiento de la mandibula anclado en los puntos del rostro; apertura dibujada entre los labios |
| **Parpadeo** | sorteo con semilla fija (el tema), 2,6–6 s | parpado comprimiendo la franja ceyas→pestanas |
| **Cabeza** | balanceo lento + acento en las silabas fuertes | dos capas de mascara **complementaria** (cabeza + torso = alfa original), pivote dentro del pecho |
| **Escenificacion** | tiempo de palabra del TTS | grande en la llamada → esquina durante el cuerpo → vuelve en el cierre |

El recorte y los puntos del rostro se generan **una vez**, fuera del lazo:

```bash
uv run --no-project --with rembg --with mediapipe --with pillow --with numpy \
    python scripts/make_presenter_cutouts.py
```

Esto lee `brand/assets/presenters/source/<id>.jpg` y graba `<id>.png` (RGBA,
busto con degrade en la base) + `<id>.json` (ojos, base de la nariz, boca,
menton, cuello, pivote). `rembg` y `mediapipe` suman mas de 400 MB con
onnxruntime y opencv detras — el agente corre tres veces al dia y nunca necesita
eso, asi que viven en un ambiente efimero de `uv` y el repositorio sigue ligero.
Misma decision que el `sentence-transformers`.

Para mirar el artefacto sin gastar un slot ni una ronda de LLM:

```bash
uv run agent presenter-preview --script fixtures/guion_manual.json --presenter theo
# presentador.mp4 + hoja.png (llamada, travesia, cuerpo, cierre)
```

Medido el 20/09/2026 en un i5 sin GPU: 40 ms por cuadro en la llamada (avatar al
51% de la altura) y 22 ms en la esquina, ~46 s de sintesis para un video de 67 s,
capa de 5–9 MB. El coste es por pixel de **salida**, por eso cada capa solo carga
el rectangulo en que tiene tinta — recortar asi quito el 40% del tiempo de cada
cuadro.

### Trampas pagadas aqui

- **La ventana de `crop` no anima a nadie.** El primer dibujo paseaba un
  rectangulo de 340x640 sobre el PNG: el recorte alfa no servia de nada, lo que
  se veia era una caja de rostro, y la persona dentro quedaba inmóvil. El test
  `test_el_cuadro_no_es_un_rectangulo` bloquea eso midiendo la silueta (testa
  estrecha, hombro ancho, esquinas vacias).
- **El recorte necesita el busto entero.** El `rembg` cortaba el asunto en los
  bordes de la imagen de origen. Ahora el busto se corta *por encima* de donde el
  hombro toca el borde y la ultima franja se vuelve transparente por degrada.
- **El labio de arriba no se mueve.** La rampa del subnasal al menton hacia bajar
  el labio inferior un tercio de lo que debia y la boca se volvia un risco. El
  desplazamiento sube de cero a lleno justo debajo de la linea de los labios — y
  ese trecho estirado *es* la boca abriendose.
- **Normalizar por el p92 deja la boca escancarada.** Medido en la narracion real
  de 67 s: mediana de apertura 0,68. Descontando el suelo de ruido (p20) y tirando
  del medio hacia abajo (gama 1,3), la mediana cae a 0,37, con 31% de los cuadros
  de boca casi cerrada — que es como de verdad se distribuye el habla.
- **El VP9 con alfa de este ffmpeg devuelve el alfa opaco.** Probado antes de
  elegir. RGBA sin perdida en `qtrle` da 1569 MB para 67 s; la capa sale en
  **dos pistas h264** (color premultiplicado + mascara en gris) con 8,7 MB, y el
  compuesto final difiere de media 0,7 de 255 por pixel. Premultiplicado porque,
  con alfa directo, el color salta del rostro al negro en el borde y el h264
  emborrona ese salto en franja oscura.
- **`presenter_for(pilar)` con el objeto en lugar del id devuelve `None`.** El
  runner reasignaba `pilar` al `ContentPillar` y pasaba el objeto a
  `presenter_for` y `accent_for`: las dos caian en el patron **en silencio**.
  Ningun presentador entraba en video alguno, y todo video salia en verde incluso
  en los pilares de acento cian. Bloqueado en `TestPresentadorEnSlot`.

## Hitos

| | Hito | Estado |
|---|---|---|
| M0 | Puerto Renderer + aceite medido del MP4 | **completado** |
| M1 | Radar (HN, Trends, Wikipedia, GDELT) | **completado** |
| M2 | Curador: score, filtro de politica, dedup por memoria | **completado** |
| M3 | Investigador + guionista + juez con rubrica | **completado** |
| M4 | Publicador (TikTok, inbox, etiqueta AIGC) | **piloto real el 19/09/2026** — MP4 69s en la inbox (`SEND_TO_USER_INBOX`, completado en la app); falta aprobacion de la app en produccion |
| M5 | Eval: free tier x free tier en la misma rubrica + metricas del post | **en curso** — `agent eval` (offline) y `agent metrics-record` listos; 1a ronda real abajo |
| M6 | Piloto automatico: 4 slots/dia en horario espanol, formato por la informacion, router de cuota, render propio | **encendido el 19/09/2026** — systemd de usuario; carrusel aun postado a mano |
| M7 | OpenRouter delante de la ruta, escalon pago por tarea, presentador animado, carrusel con jerarquia | **en curso (20/09/2026)** — ~$0,01/video medido; presentador sintetizado cuadro a cuadro ($0); falta dominio verificado para publicar carrusel por la API |

## Eval (M5) — primeros numeros

Primera ronda real el 19/09/2026, tema unico (Bonsai 2 27B), `uv run agent eval`.
El brazo pago (Claude) esta fuera de alcance — sin API de pago, no hay numero que
publicar. La comparacion es free tier x free tier:

| | gemini (`gemini-2.5-flash`) | groq (`openai/gpt-oss-120b`) |
|---|---|---|
| investigador | 4 hechos, 0 descartes, 2525 in / 489 out | 4 hechos, 0 descartes, 2537 in / 684 out |
| guionista | OK, 180 palabras (~72s), 1 intento | **fallo 2x**: `400 json_validate_failed` (no genero JSON valido antes del techo de tokens) |
| juez (mismo guion) | 12/14 APROBADO | **14/14 APROBADO** — 2 puntos mas generoso (hook 2x1, punto de vista 2x1) |

`produce` gemini punta a punta: APROBADO 12/14, 186 palabras, 2038 in / 648 out.
Ronda 3 formatos el 19/09/2026 (tema Bonsai 2, `output/rodada/`): `short` groq 42
palabras APROBADO 13/14; `carousel` groq 8/8 con 5 slides 1080x1920; `long` groq
176 palabras APROBADO 12/14. Routing medido: gemini agota el corto (60–80
palabras, sin convergir); groq escribe los tres al 1er intento. Humanizer
(`writer/humanize.py`, adaptado del `blader/humanizer` MIT) corre tras el aceite
mecanico: scan de muletillas del castellano, 1 reescrita bloqueada por grounding,
original intacto si rompe numero o franja. Metricas del post (`agent
metrics-record`) se leen en la app a mano: la inbox no expone endpoint de
metricas y la Research API esta restringida a investigacion academica.

## Limites conocidos

Publicados aqui a proposito, no escondidos.

- **Ninguna API publica dice que se viraliza *en TikTok*.** El Creative Center no
  expone API y la Research API esta restringida a investigacion academica.
  Detectamos asunto en tendencia en internet e inferimos. Son cosas distintas, y
  el proyecto no finge lo contrario.
- **El Google Trends RSS no es API oficial** y puede cambiar sin aviso. La API
  oficial seguia en alpha con acceso por inscripcion en ago/2026; `pytrends` fue
  archivado en abr/2025.
- **No hay como adjuntar sonido en tendencia de TikTok por API** — solo la
  Commercial Music Library o el audio del propio archivo. Es un techo real de
  alcance.
- **El GDELT devuelve 429 con frecuencia** sin clave (2 de 3 intentos en los
  tests), por eso entra con backoff y circuit breaker, nunca como fuente unica.
- **Reddit exige OAuth**: JSON y RSS devuelven 403. Quedo fuera del M1.
- **El edge-tts usa el endpoint de lectura en voz alta del Edge.** Gratis, sin
  contrato, puede romperse. El fallback planeado es el Kokoro-82M local (Apache
  2.0, voces es-ES), que corre en CPU.
- **Los adaptadores de LLM se ejercitaron contra la API real el 19/09/2026.**
  `llm-health` responde en los dos free tiers; investigador, guionista (gemini) y
  juez (gemini y groq) corrieron de verdad en el tema Bonsai 2. El guionista groq
  (`gpt-oss-120b`) falla en `json_validate_failed` — registrado en el Eval de
  arriba, no aqui.
- **La puerta de pasaje reprueba parafraseo.** Cuando el modelo reescribe donde
  habia que copiar, el hecho cae aunque sea verdadero. El error es asimetrico a
  proposito: dossier corto con motivo grabado es calibrable, dossier lleno de
  hecho flojo no. Los descartes quedan en la base justo para que esa calibracion
  tenga dato.
- **La puerta numerica no comprueba unidad ni contexto.** "5,9 GB" casa con una
  pagina que dice "5,9 millones de descargas". La alternativa seria pedirle al
  propio modelo que se audite, lo que no es verificacion.
- **La puerta de numeros del guionista solo ve digito.** Numero inventado
  escrito por extenso ("nueve veces menor") se le escapa; es el criterio 2 de la
  rubrica del juez quien cubre ese caso.
- **No hay busqueda web gratuita.** El descubrimiento de fuentes depende de lo
  que el Trends RSS ya asocio, del enlace detras del item del HN y del GDELT —
  que devuelve 429 con frecuencia. Tema fuera de esos tres caminos puede no
  rendir dossier alguno.
- **La viralidad no es predecible offline.** La rubrica mide calidad de guion,
  no resultado. La unica senal real es la retencion post-publicacion, y es eso lo
  que el M5 recoge.
- **Ninguna estimacion de ingresos se publicara** hasta haber numero medido del
  propio canal.
- **El aceite comprueba el artefacto, no el codigo de salida.** La primera
  ejecucion real produjo un MP4 mudo que pasaba en dimension y duracion: el
  adaptador descargaba `combined_videos` (el concat solo de video) en vez de
  `videos` (el corte con narracion y leyenda). `has_audio` se mide con `ffprobe`
  y es `False` por defecto, para que el silencio nunca sea el default aprobado.

## Licencia y creditos

El renderizador es el
[MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) de Harry, bajo
licencia MIT. No esta incluido en este repositorio: se clona en tiempo de setup y
se consume por HTTP.
