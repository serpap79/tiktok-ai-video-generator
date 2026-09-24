"""Encaje en el nicho: tech, IA y ciencia.

Lexico bilingue porque las fuentes son bilingues -- Hacker News entrega titulo
en ingles, Google Trends y Wikipedia entregan termino en castellano. Un lexico
solo en espanol descartaria la fuente primaria del nicho.

El encaje es una **puerta**, no un condimento del score: termino fuera del
nicho se descarta incluso con velocidad altisima. Sin esto, los asuntos mas
calientes de Espana (que rara vez son tech) dominarian toda recolecta, y la
memoria de rendimiento del M5 quedaria con temas incomparables entre si.

Dos cosas se aprendieron midiendo contra titulos reales, no en el papel:

1. `content_tokens` descartaba tokens con menos de 3 caracteres, lo que mataba
   "ai" e "ia" -- los dos terminos mas centrales del lexico -- antes de la
   comparacion. El nicho usa suelo 2; la deduplicacion mantiene 3, donde
   tokens cortos solo suman ruido.

2. La fuente carga informacion sobre el nicho. Hacker News es una comunidad de
   tecnologia: un titulo que llego alli ya paso por curadoria humana de
   asunto. Exigirle la misma prueba lexica que a una fuente generalista tira
   esa evidencia. Por eso existe un prior por fuente -- deliberadamente por
   debajo del umbral, para dar ventaja y no pase libre: "Warren Buffett deja
   la Berkshire" tambien esta en Hacker News y sigue siendo rechazado.
"""

from __future__ import annotations

from agent.text import content_tokens

# Terminos que solos ya caracterizan el nicho.
NUCLEO = frozenset("""
ia ai llm llms gpt chatbot bot agente agentes agent agents copilot
inteligencia artificial modelo modelos model models machine learning aprendizaje
neural neuronales neurona algoritmo algoritmos dataset embedding embeddings transformer
inference inferencia entrenamiento training cuantizacion quantization compresion
compression comprimido benchmark benchmarks prompt tokens parametros parameters
openai anthropic claude gemini qwen llama deepseek mistral gpt4 chatgpt copilot
huggingface nvidia amd intel tsmc arm apple microsoft google meta tesla waymo
chip chips gpu gpus cpu npu tpu semiconductor semiconductores procesador
risc x86 arm64 silicio wafer litografia nanometro
software codigo code coding programacion programming developer desarrollador
lenguaje language compiler compilador interprete kernel linux unix bsd
opensource api sdk framework biblioteca library runtime allocator
database base datos servidor server cloud nube container kubernetes docker
devops backend frontend browser navegador protocolo protocol
criptografia cryptography ciberseguridad cybersecurity vulnerabilidad
vulnerability exploit malware ransomware phishing overflow injection
privacidad seguridad security hacker breach passkey passkeys contrasena password
autenticacion authentication sso encryption cifrado
quantum cuantica cuantico qubit qubits superordenador ordenador computacion
computing
robot robots robotica robotics drone drones autonomo autonoma autonomous
espacio space cohete rocket satelite satelites nasa spacex esa orbita orbital
marte mars luna moon telescopio telescope astronomia astronomo galaxia galaxy
planeta exoplaneta asteroide sonda cometa nebulosa
ciencia science cientifico cientificos cientista investigacion research estudio
study experimento laboratorio fisica physics quimica chemistry biologia biology
genoma genome genetica dna adn rna proteina protein celula neurociencia
neuroscience microbioma
clima climate climatico energia energy fusion nuclear solar eolica
bateria baterias battery hidrogeno hydrogen fotovoltaico superconductor
internet web wifi bluetooth 5g 6g android ios iphone ipad macos windows
smartphone smartphones movil moviles aplicacion aplicaciones notebook laptop
hardware firmware
transistor microprocesador microchip semiconductor eniac arpanet cifrado
huggingface transformers deepfake deepfakes chatbots
""".split())

# Solos no dicen nada ("lanzamiento" puede ser de cualquier cosa), pero sumados
# a un termino del nucleo refuerzan que el asunto es del nicho.
APOYO = frozenset("""
lanzamiento lanzo anuncia anuncio anuncio descubrimiento descubren avance record
version update actualizacion beta release open source desempeno rendimiento
performance eficiencia eficiente billones millones billones trillones bits bytes
gigabytes terabytes latencia throughput cache memoria memory escala
escalabilidad
startup empresa laboratorio universidad instituto paper articulo preprint
launch launched announces announced discovery breakthrough record version
billion million research study engineering engineer build built tool tooling
""".split())

# Fuentes cuyo alcance ya es el nicho. El valor queda POR DEBAJO del umbral a
# proposito: es ventaja inicial, no aprobacion automatica.
PRIOR_POR_FUENTE: dict[str, float] = {
    "hacker_news": 0.22,
    # Redacciones de tech tambien cubren consola, concierto y celebridad:
    # ventaja, no pase libre -- mismo razonamiento que el HN.
    "rss_tech_es": 0.22,
    "rss_tech": 0.22,
    "rss_ciencia": 0.22,
    # Estas dos SON el nicho por construccion: el Hub solo lista modelos de IA,
    # y el archivo esta curado a mano para historia de la computacion y
    # ciencia. Aqui el prior aprueba solo, a proposito.
    "huggingface": 0.6,
    "archivo": 0.6,
}

# Por debajo de esto el termino se descarta. Calibrado contra titulos reales
# en tests/fixtures/radar/hacker_news.json -- ver test_niche_calibracion.
UMBRAL_DEFECTO = 0.34

# El nicho necesita ver "ai" e "ia"; la deduplicacion no gana nada con ello.
_MIN_TOKEN_NICHO = 2


def fit(termo: str, source: str | None = None) -> float:
    """Cuanto pertenece el termino al nicho, de 0 a 1.

    Un termino del nucleo ya garantiza 0,6 -- suficiente para pasar la puerta
    solo, porque "NASA" o "Linux" son asunto del canal aunque no haya mas
    pista. Terminos de apoyo suman poco y nunca aprueban solos.

    El prior de la fuente entra como suelo, no como suma: una fuente de nicho
    garantiza el minimo, y el lexico solo puede mejorar a partir de ahi.
    """
    palabras = content_tokens(termo, min_len=_MIN_TOKEN_NICHO)
    prior = PRIOR_POR_FUENTE.get(source or "", 0.0)
    if not palabras:
        return prior

    nucleo = len(palabras & NUCLEO)
    apoyo = len(palabras & APOYO)

    if nucleo == 0:
        # Solo apoyo no caracteriza nicho: "lanzamiento record" puede ser de futbol.
        lexico = min(0.15 * apoyo, 0.25)
    else:
        # Satura rapido: dos terminos del nucleo ya es senal fuerte, y el
        # tercero no deberia valer mas que la diferencia entre tener y no
        # tener nicho.
        lexico = min(0.6 + 0.2 * min(nucleo - 1, 2) + 0.05 * min(apoyo, 2), 1.0)

    return round(max(lexico, prior), 4)


def matched_terms(termo: str) -> tuple[set[str], set[str]]:
    """Que terminos casaron. Usado para justificar la decision en el registro."""
    palabras = content_tokens(termo, min_len=_MIN_TOKEN_NICHO)
    return palabras & NUCLEO, palabras & APOYO
