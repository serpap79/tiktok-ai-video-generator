"""Presentador animado: un retrato quieto se convierte en alguien hablando, sin modelo.

El dibujo anterior tomaba el PNG del presentador y paseaba una **ventana
rectangular** de 340x640 por encima de el. Dos cosas fallaban a la vez: la
ventana cortaba la silueta (el recorte alfa no servia para nada, lo que se veia
era un rectangulo de rostro) y el unico movimiento era el de la ventana -- la
persona dentro quedaba inmóvil. En pantalla eso se lee como pegatina, no como
presentador.

Aqui la silueta es siempre entera y quien se mueve es ella. Cuatro capas de
movimiento, todas deterministicas y gratis (ningun modelo, ninguna API):

1. **Boca por la voz.** La envolvente de energia de la narracion (RMS por
   cuadro, con ataque rapido y relajacion lenta, como un compresor) abre la
   mandibula: un campo de desplazamiento vertical que nace en cero en la base
   de la nariz, llega al maximo en el menton y vuelve a cero en el cuello. Por
   encima, la abertura de la boca se dibuja entre los labios, con diente solo
   en las silabas mas abiertas. No es visema -- es amplitud. A 3 metros de
   distancia de un feed vertical, la diferencia entre ambos es invisible; la
   diferencia entre boca parada y boca que acompana el audio es el video entero.
2. **Parpadeo.** El parpado baja de verdad: la franja entre la ceja y el
   parpado inferior se comprime contra la linea de las pestañas, y la piel de
   arriba cubre el ojo. Intervalo sorteado con semilla fija (2,6 a 6 s) -- el
   mismo guion parpadea siempre igual, lo que hace el cuadro verificable en test.
3. **Cabeza.** Cabeza y torso son dos capas con mascara complementaria (la suma
   de las dos da exactamente el alfa original, asi que con angulo cero el cuadro
   es identico al retrato). La cabeza gira en torno a un pivote dentro del
   pecho, con balanceo lento + acento en las silabas fuertes; el torso respira.
4. **Escenificacion por el guion.** Tamano y lugar del presentador cambian con
   lo que esta diciendo: grande y al centro en la llamada, pequeño en la esquina
   mientras el material de apoyo cuenta la historia, y de vuelta en el cierre.
   Eso es lo que `plan` calcula a partir de los tiempos de palabra -- movimiento
   *en relacion al contexto*, no movimiento por movimiento.

La capa sale en un solo archivo, con **dos pistas de video**: el color (ya
premultiplicado por el alfa) y la mascara en gris. No es capricho -- es lo que
cabe en disco. Medido en esta maquina: 67s de capa RGBA sin perdida en qtrle dan
1569 MB; las dos pistas en h264 dan 8,7 MB, y el compuesto final difiere de
media 0,7 de 255 por pixel. El VP9 con alfa de este ffmpeg entrega el alfa
opaco (probado antes de elegir), y RGB comun comprimido daria franja oscura en
la silueta -- por eso premultiplicado, que hace caer el color a cero junto con
la mascara.

Quedar en un archivo aparte es deliberado: se puede abrir el artefacto y mirar
el presentador solo, sin el video detras.

Nota sobre claves de datos: los nombres de los puntos del rostro (`subnasal`,
`labio_sup`, `queixo`, `mand_esq`...) son el contrato del JSON generado por
`scripts/make_presenter_cutouts.py` y no se renombran en codigo.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from agent.render.typography import MONO, font

W, H = 1080, 1920
FPS = 30

# ------------------------------------------------------------- escenificacion
# Alturas en pixel en pantalla (1920 de alto). La guia de la marca pide 30-36%
# para el presentador de esquina; la llamada y el cierre se pasan a proposito --
# ahi el es el asunto, no la firma.
ALTURA_LLAMADA = 980
ALTURA_CUERPO = 600
ALTURA_CIERRE = 640
BASE_LLAMADA = 1740
BASE_CUERPO = 1700
CX_LLAMADA = 540
CX_CUERPO = 320
CX_CIERRE = 340
# Tiempo de travesia entre dos escenificaciones (suavizado, sin corte seco).
TRANSICION_S = 0.7
# Area util de la capa: todo lo que la escenificacion alcanza, con margen para
# la rotacion y para el chip de nombre. Menor que el cuadro entero porque cada
# pixel de mas es un pixel copiado en cada uno de los ~2000 cuadros.
AREA = (0, 660, 1080, 1160)  # x, y, ancho, alto
# La leyenda sube cuando hay presentador. Comprobado en las dos escenificaciones
# de esquina: texto centrado en 940 (fuente 84, top en ~898) contra top del
# avatar en 1100 (cuerpo) y 1060 (cierre) -- sobra mas de 60px en ambas.
Y_LEYENDA_CON_PRESENTADOR = 940

# ---------------------------------------------------------------------- boca
# Caida maxima del menton, como fraccion de la altura del rostro. Habla normal
# abre la mandibula entre 10% y 15% de la altura del rostro; 10,5% porque por
# encima el rostro visiblemente se alarga en las silabas abiertas (mirado cuadro
# a cuadro).
MAXILAR = 0.105
# Abertura entre los labios, como fraccion de la caida del menton en ese cuadro.
ABERTURA = 0.80
# El diente solo aparece por encima de esta abertura (si no, todo "a" sonrie).
DIENTE_MIN = 0.52
# Supermuestreo del mosaico de la boca. El `ImageDraw` de Pillow no tiene
# antialiasing ninguno -- medido el 20/09/2026, la elipse de la boca salia con
# CERO pixel de borde parcial a 12, 26 y 44 px de altura. Dibujar 4x mayor y
# reducir por media de area da el borde suave por 16 muestras de un mosaico de
# ~150x70 px, lo que no aparece en el reloj del cuadro.
SS = 4
# Por debajo de esto la boca entra en degradado (ease cubico) en vez de
# aparecer de golpe con 2 px de altura, que era la unica discontinuidad restante.
BOCA_MIN_PX = 3.5
# Desenfoque del labio. Era 1,0 en el suelo para disimular el serrated; con el
# borde ya suave puede ser menor, y la boca queda menos empastada.
BOCA_BORRON_MIN = 0.6
ATAQUE_S = 0.035
RELAJACION_S = 0.085
# Cuantas muestras de envolvente por cuadro. El ataque de 35 ms es MAS CORTO
# que un cuadro de 33,3 ms: corriendo el filtro a la tasa de cuadro no filtraba
# nada y la boca se abria de un cuadro al siguiente. A 4x (120 Hz) el mismo
# ataque son 4 muestras y de verdad suaviza; solo al final la curva baja al
# cuadro.
ENV_SUB = 4
# La curva de la envolvente, calibrada en la narracion real de 67s del
# 20/09/2026. La primera version normalizaba por el p92 con gamma 0,7 y la boca
# quedaba en 0,68 de abertura en la mediana -- o sea, escancarada el video
# entero. Descontar el suelo de ruido (p20) y tirar del medio hacia abajo
# (gamma 1,3) devuelve la distribucion de habla real: mediana 0,37, 31% de los
# cuadros con boca casi cerrada, 10% abierta de par en par.
# Recalibrado el 20/09/2026 junto con la ventana de Hann. La ventana con
# solapamiento rellena los silencios cortos entre silabas -- que es justo lo
# que quita el temblor --, pero con ello levanta el suelo: en los mismos 20s de
# narracion la mediana subia de 0,39 a 0,44 y la boca pasaba a descansar mas
# abierta. p30/p92 devuelve la distribucion anterior: los picos clavan (16% de
# cuadros de boca abierta de par en par, identico) y la mediana queda en 0,43
# contra 0,39 -- 4 centesimas, que en una abertura maxima de 34 px son 1,4 px.
# Cambio aceptado a proposito: el pico es lo que se lee como expresivo.
ENV_PISO_PCT = 30
ENV_TECHO_PCT = 92
ENV_GAMMA = 1.3

# ------------------------------------------------------------------ parpadeo
# Parpadeo asimetrico: cierra rapido, sostiene un instante, abre despacio. La
# version anterior era `sin(pi*u)**0.6` en una ventana de 0,13 s -- medido, 2
# cuadros cerrando y 2 abriendo, con el primer cuadro saltando de 0,00 a 0,82.
# Simetrico es casi un corte: de ahi venia el aire de mecanismo.
PARPADEO_CIERRA_S = 0.055    # ~1,7 cuadros a 30 fps
PARPADEO_SOSTIENE_S = 0.035  # ~1 cuadro con el ojo cerrado
PARPADEO_ABRE_S = 0.115      # ~3,5 cuadros -- el doble del cierre
PARPADEO_S = PARPADEO_CIERRA_S + PARPADEO_SOSTIENE_S + PARPADEO_ABRE_S
PARPADEO_MIN_S = 2.6
PARPADEO_MAX_S = 6.0
# Cuanto puede un parpadeo estirarse hacia la frontera de frase mas cercana.
PARPADEO_ANCLA_S = 0.9
# Cuanto suben el parpado inferior y la mejilla en el cierre total, en fraccion
# de la altura del ojo. Parpadear no es solo el parpado de arriba: el orbicular
# tira de lo que hay en torno, y es la ausencia de eso lo que hace el ojo una
# persiana.
SUBE_VECINO = 0.20
# Hasta donde baja la caja del ojo, en alturas de ojo, para alcanzar la mejilla.
CAJA_MEJILLA = 2.4
# Semi-anchura de la sombra de la pestaña, en pixel.
CILIO_PX = 2.2
# Techo de compresion bajo la pestaña. Apretar una franja de altura fija en una
# franja que va a cero da pendiente infinita: con el ojo casi cerrado llegaba a
# 38x y saturaba la caja entera (medido: 47 lineas pegadas en la ultima). Por
# encima de este techo lo que queda del ojo ya esta cubierto por el parpado de
# todos modos -- no hay nada que apretar.
COMPRESION_MAX = 3.0
# En cuantas alturas de ojo la compresion vuelve a ser identidad, contadas
# desde la pestaña de abajo. Corto a proposito: bajo el ojo esta la mejilla, y
# la mejilla no comprime, solo sube.
SUELTA_ALTO = 0.6
# Margen lateral de la caja del ojo, en alturas de ojo. Es tambel la anchura de
# la pluma: la deformacion vale 1 en la esquina del ojo y 0 en el borde de la
# caja, asi que la columna de dentro y la de fuera se tocan sin salto.
PLUMA_OJO = 0.55

# -------------------------------------------------------------------- cabeza
GIRO_CABEZA = 1.7   # grados, balanceo lento
GIRO_TORSO = 0.8    # grados
ACENTO_CABEZA = 1.1  # grados de mas en las silabas fuertes
RESPIRO = 0.005     # fraccion de la altura


# ===========================================================================
# retrato medido
# ===========================================================================


@dataclass(frozen=True)
class Retrato:
    """El recorte del presentador con los puntos del rostro ya medidos.

    Viene de `scripts/make_presenter_cutouts.py`; aqui nada se adivina.
    """

    id: str
    imagen: Image.Image
    puntos: dict[str, tuple[float, float]]
    head_top: int
    neck_y: int
    face_height: float
    pivote: tuple[float, float]

    @property
    def size(self) -> tuple[int, int]:
        return self.imagen.size

    def p(self, nombre: str) -> tuple[float, float]:
        return self.puntos[nombre]


# Claves pt del JSON generado -> claves castellanas del codigo. Los assets ya
# horneados guardan los nombres de la primera guia; normalizar aqui deja el
# contrato del JSON intacto y el codigo hablando un solo idioma.
CLAVES_PT = {
    "boca_esq": "boca_izq", "boca_dir": "boca_der", "queixo": "menton",
    "nariz_ponta": "nariz_punta", "testa": "frente",
    "face_esq": "rostro_izq", "face_dir": "rostro_der",
    "mand_esq": "mandibula_izq", "mand_dir": "mandibula_der",
    "labio_sup_out": "labio_sup_ext", "labio_inf_out": "labio_inf_ext",
    "olho_esq_baixo": "ojo_izq_abajo", "olho_dir_baixo": "ojo_der_abajo",
    "olho_esq_cima": "ojo_izq_arriba", "olho_dir_cima": "ojo_der_arriba",
    "olho_esq_out": "ojo_izq_ext", "olho_esq_in": "ojo_izq_int",
    "olho_dir_out": "ojo_der_ext", "olho_dir_in": "ojo_der_int",
}


def _clave(nombre: str) -> str:
    return CLAVES_PT.get(nombre, nombre)


def cargar(png: Path) -> Retrato:
    meta_path = png.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} no existe -- corre scripts/make_presenter_cutouts.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return Retrato(
        id=meta.get("id", png.stem),
        imagen=Image.open(png).convert("RGBA"),
        puntos={_clave(k): (v[0], v[1]) for k, v in meta["points"].items()},
        head_top=int(meta["head_top"]),
        neck_y=int(meta["neck_y"]),
        face_height=float(meta["face_height"]),
        pivote=(float(meta["pivot"][0]), float(meta["pivot"][1])),
    )


# ===========================================================================
# tiempo: envolvente de la voz y latidos del guion
# ===========================================================================


def envolvente(audio: Path, n: int, fps: int = FPS, *, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Energia de la narracion por cuadro, de 0 (silencio) a 1 (silaba mas abierta).

    Ataque rapido y relajacion lenta porque la boca tiene inercia: cierra mas
    despacio de lo que abre, y sin eso la animacion tiembla en cada consonante.

    Medido el 20/09/2026: la version anterior leia el RMS en ventanas
    rectangulares de un cuadro entero (33,3 ms) **sin solapamiento y sin
    ventaneo**, y el ataque de 25 ms -- mas corto que un cuadro -- no filtraba
    nada. La senal de control llegaba a la geometria ya serrada en el tiempo, y
    la boca se abria de un cuadro al otro: de ahi venia la mitad del aire
    mecanico.

    Ahora son tres pasos, en el orden que importa: RMS con ventana de Hann a
    `ENV_SUB` veces la tasa de cuadro, ataque/relajacion **en esa** tasa (donde
    35 ms son 4 muestras y el filtro tiene que hacer), y solo al final la media
    del grupo baja la curva al cuadro. Filtrar antes de diezm es la misma idea
    de la supermuestreo de la boca, aplicada al eje del tiempo.
    """
    crudo = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(audio), "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"], capture_output=True, timeout=300).stdout
    x = np.frombuffer(crudo, dtype="<i2").astype(np.float32) / 32768.0
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    tasa = fps * ENV_SUB
    m = n * ENV_SUB
    paso = 16000 / tasa
    # Ventana de Hann con 50% de solape, en una sola convolucion del senal al
    # cuadrado: el RMS de todas las posiciones de golpe, sin bucle en Python.
    anchura = max(int(round(paso * 2)), 3)
    peso = np.hanning(anchura + 2)[1:-1].astype(np.float32)
    peso /= float(peso.sum())
    energia = np.convolve(np.square(x), peso, mode="same")
    centros = np.clip(((np.arange(m) + 0.5) * paso).astype(np.int64), 0, x.size - 1)
    rms = np.sqrt(np.maximum(energia[centros], 0.0)).astype(np.float32)
    habla = rms[rms > 1e-4]
    if not habla.size:
        return np.zeros(n, dtype=np.float32)
    piso = float(np.percentile(habla, ENV_PISO_PCT))
    techo = float(np.percentile(habla, ENV_TECHO_PCT))
    nivel = np.clip((rms - piso) / max(techo - piso, 1e-6), 0.0, 1.0) ** ENV_GAMMA
    return _ataque_relajacion(nivel, tasa).reshape(n, ENV_SUB).mean(axis=1)


def _ataque_relajacion(nivel: np.ndarray, tasa: int) -> np.ndarray:
    """Un polo por sentido, a la tasa en que vive la senal (no la del cuadro)."""
    sube = math.exp(-1.0 / max(ATAQUE_S * tasa, 1e-6))
    baja = math.exp(-1.0 / max(RELAJACION_S * tasa, 1e-6))
    salida = np.empty_like(nivel)
    y = 0.0
    for i, objetivo in enumerate(nivel):
        a = sube if objetivo > y else baja
        y = a * y + (1 - a) * float(objetivo)
        salida[i] = y
    return salida


@dataclass(frozen=True)
class Tiempos:
    """Donde el guion cambia de asunto, en segundos."""

    fin_gancho: float
    inicio_cierre: float
    duracion: float
    sentencias: tuple[float, ...] = ()


def tiempos_por_palabras(script_hook: str, script_closing: str, palabras, duracion: float
                         ) -> Tiempos:
    """Tiempos exactos a partir del tiempo de cada palabra que devuelve el TTS."""
    n_hook = len(script_hook.split())
    n_close = len(script_closing.split())
    total = len(palabras)
    if total == 0:
        return tiempos_estimados(n_hook, n_close, total, duracion)
    fin_gancho = palabras[min(n_hook, total) - 1].end
    inicio_cierre = palabras[max(0, total - n_close)].start
    sentencias = [p.start for i, p in enumerate(palabras)
                  if i and palabras[i - 1].text.endswith((".", "!", "?"))]
    return Tiempos(fin_gancho=min(fin_gancho, duracion),
                   inicio_cierre=min(max(inicio_cierre, fin_gancho), duracion),
                   duracion=duracion, sentencias=tuple(sentencias))


def tiempos_estimados(n_hook: int, n_close: int, total: int, duracion: float) -> Tiempos:
    """Reserva cuando no hay tiempo de palabra (camino del MPT): por proporcion."""
    total = max(total, n_hook + n_close, 1)
    return Tiempos(fin_gancho=duracion * n_hook / total,
                   inicio_cierre=duracion * (total - n_close) / total,
                   duracion=duracion)


# ===========================================================================
# escenificacion
# ===========================================================================


@dataclass(frozen=True)
class Pose:
    """Una pose de escenificacion: cuando, de que tamano y donde."""

    t: float
    altura: float
    cx: float
    base: float
    alfa: float = 1.0


@dataclass(frozen=True)
class Escenificacion:
    """El plan de escena entero: el presentador y lo que el empuja en pantalla.

    La leyenda y la tarjeta del gancho no son independientes del presentador --
    disputan el mismo espacio. Quien decide es la escenificacion, en un lugar
    unico.
    """

    poses: list[Pose]
    leyenda_y: int
    leyenda_inicio: float
    tarjeta_s: float


def plan(tiempos: Tiempos) -> Escenificacion:
    """Llamada grande al centro -> esquina durante el cuerpo -> vuelta en el cierre.

    El presentador sale del centro en cuanto el material de apoyo pasa a contar
    la historia, y vuelve cuando el texto vuelve a hablar con quien mira (el
    CTA). No es adorno: es la regla de hacia donde debe ir la mirada en cada
    trecho.

    Mientras el esta grande, quien escribe el gancho es la tarjeta -- la leyenda
    karaoke solo entra cuando el encoge. Sin eso el mismo texto apareceria dos
    veces en pantalla y la segunda caeria encima de su rostro.
    """
    fin_gancho = max(1.2, min(tiempos.fin_gancho, tiempos.duracion - 0.5))
    inicio_cierre = max(fin_gancho + TRANSICION_S,
                        min(tiempos.inicio_cierre, tiempos.duracion - 0.4))
    poses = [
        Pose(t=0.0, altura=ALTURA_LLAMADA, cx=CX_LLAMADA, base=BASE_LLAMADA, alfa=0.0),
        Pose(t=0.45, altura=ALTURA_LLAMADA, cx=CX_LLAMADA, base=BASE_LLAMADA),
        Pose(t=fin_gancho, altura=ALTURA_LLAMADA, cx=CX_LLAMADA, base=BASE_LLAMADA),
        Pose(t=fin_gancho + TRANSICION_S, altura=ALTURA_CUERPO, cx=CX_CUERPO,
             base=BASE_CUERPO),
        Pose(t=inicio_cierre, altura=ALTURA_CUERPO, cx=CX_CUERPO, base=BASE_CUERPO),
        Pose(t=inicio_cierre + TRANSICION_S, altura=ALTURA_CIERRE,
             cx=CX_CIERRE, base=BASE_CUERPO),
        Pose(t=tiempos.duracion, altura=ALTURA_CIERRE, cx=CX_CIERRE,
             base=BASE_CUERPO),
    ]
    poses = [p for i, p in enumerate(poses) if i == 0 or p.t > poses[i - 1].t]
    cambio = fin_gancho + TRANSICION_S
    return Escenificacion(poses=poses, leyenda_y=Y_LEYENDA_CON_PRESENTADOR,
                          leyenda_inicio=cambio, tarjeta_s=cambio)


def _suave(u: float) -> float:
    """Aceleracion y frenada (smoothstep): la escenificacion no cambia de sitio en corte."""
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _ease_cubico(u: float) -> float:
    """Ease-in-out cubico: sale de cero y llega a uno con velocidad cero.

    Mas lento al comienzo y al final que el smoothstep, y es eso lo que se
    quiere en un parpado: lo que denuncia mecanismo es la esquina en la salida
    y en la llegada.
    """
    u = min(1.0, max(0.0, u))
    return 4 * u * u * u if u < 0.5 else 1 - ((-2 * u + 2) ** 3) / 2


def pose(poses: list[Pose], t: float) -> Pose:
    if t <= poses[0].t:
        return poses[0]
    for a, b in zip(poses, poses[1:], strict=False):
        if t <= b.t:
            u = _suave((t - a.t) / max(b.t - a.t, 1e-6))
            return Pose(t=t,
                        altura=a.altura + (b.altura - a.altura) * u,
                        cx=a.cx + (b.cx - a.cx) * u,
                        base=a.base + (b.base - a.base) * u,
                        alfa=a.alfa + (b.alfa - a.alfa) * u)
    return poses[-1]


# ===========================================================================
# deformaciones en el retrato
# ===========================================================================


def _reinterp_y(bloque: np.ndarray, desplaz: np.ndarray) -> np.ndarray:
    """`salida[y] = bloque[y - desplaz[y]]`, con interpolacion lineal entre filas."""
    h = bloque.shape[0]
    ys = np.clip(np.arange(h, dtype=np.float32) - desplaz, 0, h - 1.001)
    lo = ys.astype(np.int32)
    fr = (ys - lo).astype(np.float32)[:, None, None]
    return (bloque[lo] * (1 - fr) + bloque[lo + 1] * fr)


@dataclass
class Maxilar:
    """Region del rostro que el habla mueve, precalculada una vez."""

    caja: tuple[int, int, int, int]
    perfil: np.ndarray          # desplazamiento por fila, para abertura 1.0
    peso_x: np.ndarray          # 1 en la mandibula, 0 fuera de ella (bordes suaves)
    boca_cx: float
    boca_y: float
    boca_w: float
    color_boca: tuple[int, int, int]
    color_diente: tuple[int, int, int]
    caida_labio: float          # desplazamiento de la linea de los labios, abertura 1.0


def preparar_maxilar(r: Retrato) -> Maxilar:
    """Donde el habla mueve el rostro, medido en los puntos -- nada es chute.

    El perfil no es una rampa de la nariz al menton: al hablar el labio de
    arriba queda quieto y el de abajo baja **junto con el menton**. Asi que el
    desplazamiento sube de cero a lleno en un trecho corto justo debajo de la
    linea de los labios (y ese trecho estirado es justamente la boca abriendo),
    sigue lleno hasta el menton y vuelve a cero en el cuello. La primera version
    rampaba del subnasal al menton y el labio inferior bajaba solo un tercio de
    lo que debia: la boca se volvia un risco.
    """
    sub_y = r.p("subnasal")[1]
    labio_y = r.p("labio_sup")[1]
    menton_y = r.p("menton")[1]
    fin = menton_y + 0.55 * (menton_y - sub_y)
    x0 = int(max(0, r.p("rostro_izq")[0] - 30))
    x1 = int(min(r.size[0], r.p("rostro_der")[0] + 30))
    y0 = int(max(0, sub_y - 6))
    y1 = int(min(r.size[1], fin + 8))

    filas = np.arange(y0, y1, dtype=np.float32)
    amp = MAXILAR * r.face_height
    abre = max(0.30 * (menton_y - labio_y), 8.0)
    sube = np.clip((filas - labio_y) / abre, 0, 1)
    sube = sube * sube * (3 - 2 * sube)      # sin arista en la linea de los labios
    vuelta = np.clip(1 - (filas - menton_y) / max(fin - menton_y, 1e-6), 0, 1)
    perfil = (amp * np.minimum(sube, vuelta)).astype(np.float32)

    columnas = np.arange(x0, x1, dtype=np.float32)
    m_izq, m_der = r.p("mandibula_izq")[0], r.p("mandibula_der")[0]
    pluma = 46.0
    peso = np.minimum(np.clip((columnas - (m_izq - pluma)) / pluma, 0, 1),
                      np.clip(((m_der + pluma) - columnas) / pluma, 0, 1))

    px = np.asarray(r.imagen.convert("RGB")).astype(np.float32)
    cx, cy = r.p("labio_sup")
    muestra = px[int(cy) - 4:int(cy) + 5, int(cx) - 14:int(cx) + 15].reshape(-1, 3)
    labio = muestra.mean(axis=0) if muestra.size else np.array([120.0, 80.0, 80.0])
    # El diente sale del tono claro del propio rostro (la mejilla iluminada),
    # no de un color fijo: cada presentador tiene su exposicion.
    rostro = px[int(r.p("ojo_izq_abajo")[1]):int(menton_y),
                int(r.p("mandibula_izq")[0]):int(r.p("mandibula_der")[0])].reshape(-1, 3)
    claro = (np.percentile(rostro, 88, axis=0) if rostro.size
             else np.array([200.0, 190.0, 185.0]))
    diente = claro * 0.55 + float(claro.mean()) * 0.52
    return Maxilar(
        caja=(x0, y0, x1, y1), perfil=perfil,
        peso_x=peso.astype(np.float32)[None, :, None],
        boca_cx=(r.p("boca_izq")[0] + r.p("boca_der")[0]) / 2,
        boca_y=labio_y,
        boca_w=r.p("boca_der")[0] - r.p("boca_izq")[0],
        color_boca=tuple(int(c) for c in np.clip(labio * 0.22, 6, 70)),
        color_diente=tuple(int(c) for c in np.clip(diente, 60, 244)),
        caida_labio=float(amp),
    )


def aplicar_maxilar(cuadro: np.ndarray, m: Maxilar, abertura: float) -> None:
    """Abre la mandibula en el sitio (el cuadro se modifica)."""
    if abertura <= 0.01:
        return
    x0, y0, x1, y1 = m.caja
    bloque = cuadro[y0:y1, x0:x1].astype(np.float32)
    movido = _reinterp_y(bloque, m.perfil * abertura)
    cuadro[y0:y1, x0:x1] = (bloque + (movido - bloque) * m.peso_x).astype(np.uint8)


def losa_boca(m: Maxilar, abertura: float) -> tuple[Image.Image, tuple[int, int]] | None:
    """La abertura entre los labios, dibujada en un mosaico pequeño.

    En un cuadro de 1072x1053 el desenfoque gaussiano de la imagen entera
    costaria mas que todo lo demas del cuadro sumado; aqui corre en una caja de
    ~300x200.

    Dos defectos murieron aqui el 20/09/2026, ambos medidos antes:

    1. **El `ImageDraw` de Pillow no tiene antialiasing.** La elipse salia con
       cero pixel de borde parcial (comprobado a 12, 26 y 44 px de altura) --
       escalon duro, que era el serrado que se veia en el labio. El desenfoque
       gaussiano despues de eso no lo resuelve: el escalon ya esta horneado.
    2. **La esquina del pegado se truncaba a entero.** Entre abertura 0,36 y
       0,39 la esquina saltaba 1 px de lado mientras la boca crecia 1 px de
       alto: temblaba en vez de crecer.

    Los dos desaparecen en la misma cuenta. Las mascaras se dibujan a `SS`
    veces el tamano y se reducen por media de area (`Image.BOX`, que para
    factor entero es la media exacta del bloque); y la **parte fraccionaria** de
    la posicion entra en las coordenadas del dibujo grande, asi que el mosaico
    ya nace desplazado un cuarto de pixel y el pegado sigue en entero.

    Solo la mascara se supermuestrea, nunca el color: reducir RGBA sobre fondo
    transparente tiraria el color al negro en el borde y devolveria la misma
    franja oscura que `encode_command` existe para evitar. El color del mosaico
    es plano en todo pixel -- incluido donde el alfa es cero.
    """
    altura = m.caida_labio * abertura * ABERTURA
    if altura < BOCA_MIN_PX * 0.5:
        return None
    # Entrada suave: la boca crece desde cero en vez de surgir de golpe con
    # 2 px. Es la ultima discontinuidad que quedaba en el camino de la abertura.
    surge = _ease_cubico(min(1.0, altura / BOCA_MIN_PX))
    # Boca abierta es mas estrecha que boca cerrada: el labio se recoge.
    anchura = m.boca_w * (0.78 - 0.12 * abertura)
    pad = 30

    # El agujero empieza en la linea del labio de arriba (el punto 13 de la
    # malla es el labio INTERNO), con 8% de solape para no dejar rendija. La
    # posicion exacta es fraccionaria: el entero va al pegado, el resto al dibujo.
    fx = m.boca_cx - anchura / 2 - pad
    fy = m.boca_y - altura * 0.08 - pad
    canto = (math.floor(fx), math.floor(fy))
    sx, sy = (fx - canto[0]) * SS, (fy - canto[1]) * SS
    w, h = int(anchura + 2 * pad) + 1, int(altura + 2 * pad) + 1

    caja = (pad * SS + sx, pad * SS + sy,
            pad * SS + sx + anchura * SS, pad * SS + sy + altura * SS)
    grande = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(grande).ellipse(caja, fill=255)
    alfa = np.asarray(grande.resize((w, h), Image.BOX), dtype=np.float32) / 255.0

    color = np.empty((h, w, 3), dtype=np.float32)
    color[:] = m.color_boca
    if abertura > DIENTE_MIN:
        fuerza = min(1.0, (abertura - DIENTE_MIN) / (1 - DIENTE_MIN))
        diente = (caja[0] + anchura * SS * 0.09, caja[1] + altura * SS * 0.02,
                  caja[2] - anchura * SS * 0.09,
                  caja[1] + altura * SS * (0.13 + 0.11 * fuerza))
        if diente[3] > diente[1] + 1.5 * SS:
            md = Image.new("L", (w * SS, h * SS), 0)
            ImageDraw.Draw(md).rounded_rectangle(
                diente, radius=max(1, int(altura * SS * 0.10)), fill=255)
            # El diente es translucido sobre la boca (190/255), como antes.
            peso = (np.asarray(md.resize((w, h), Image.BOX), dtype=np.float32)
                    / 255.0 * (190 / 255) * fuerza)[:, :, None]
            color += (np.asarray(m.color_diente, dtype=np.float32) - color) * peso

    buf = np.empty((h, w, 4), dtype=np.uint8)
    buf[:, :, :3] = color.astype(np.uint8)
    buf[:, :, 3] = (alfa * (255.0 * surge)).astype(np.uint8)
    losa = Image.fromarray(buf).filter(
        ImageFilter.GaussianBlur(max(BOCA_BORRON_MIN, altura * 0.055)))
    return losa, canto


@dataclass
class Ojo:
    caja: tuple[int, int, int, int]
    parpado_y: float
    base_y: float
    alto: float = 8.0


def preparar_ojos(r: Retrato) -> list[Ojo]:
    """La region que el parpadeo mueve, medida en los puntos del ojo.

    La caja baja hasta la mejilla (`CAJA_MEJILLA` alturas de ojo) y no solo
    hasta un poco debajo de la pestaña: parpadear tira del parpado inferior y de
    la mejilla juntos, y no habia pixel donde dibujar eso. Para antes del
    subnasal para nunca tocar la region que la mandibula ya mueve -- dos
    deformaciones en el mismo pixel se suman y se vuelven mueca.
    """
    ojos = []
    limite = r.p("subnasal")[1] - 4
    for lado in ("izq", "der"):
        arriba = r.p(f"ojo_{lado}_arriba")[1]
        abajo = r.p(f"ojo_{lado}_abajo")[1]
        xs = sorted((r.p(f"ojo_{lado}_ext")[0], r.p(f"ojo_{lado}_int")[0]))
        alto = max(abajo - arriba, 4.0)
        # La mejilla es mas ancha que el ojo; el margen lateral acompana, y es
        # en el que la pluma apaga la deformacion antes del borde de la caja.
        ojos.append(Ojo(
            caja=(int(max(0, xs[0] - PLUMA_OJO * alto)),
                  int(max(0, arriba - 2.4 * alto)),
                  int(min(r.size[0], xs[1] + PLUMA_OJO * alto)),
                  int(min(r.size[1], limite, abajo + CAJA_MEJILLA * alto))),
            parpado_y=arriba, base_y=abajo + 0.45 * alto, alto=alto))
    return ojos


def aplicar_parpadeo(rgb: np.ndarray, ojos: list[Ojo], cierre: float) -> None:
    """Comprime la franja entre la ceja y la pestaña: la piel de arriba cubre el ojo.

    Dos tramos lineales con la linea del parpado andando: por encima la piel se
    estira hacia abajo, por debajo lo que queda del ojo se aprieta. En cierre=1
    el ojo desaparecio y quedo parpado -- que es lo que es un parpadeo.

    Tres defectos salieron de aqui el 20/09/2026, los tres medidos antes:

    1. **La sombra de la pestaña saltaba de linea entera.** Era
       `int(round(linea))` y una franja dura de 2 px: entre el cuadro 60 y el 61
       saltaba 31 px de golpe. Ahora es un peso por linea centrado en la
       posicion **fraccionaria**, asi que baja continuamente -- fin del escalon
       en el parpado.
    2. **El tramo bajo la pestaña desbordaba.** Con el ojo casi cerrado el
       divisor `base_y - linea` iba a cero, la pendiente explotaba y las
       ultimas filas de la caja se pegaban todas a la misma linea de origen: un
       borron vertical de 5 px justo en el parpado inferior (medido en cierre
       0,9 y 1,0). Ahora la compresion se deshace hasta el borde de la caja.
    3. **Nada alrededor reaccionaba.** Parpadear no es solo el parpado de
       arriba -- el orbicular tira del parpado inferior y de la mejilla juntos.
       Un segundo campo sube lo que esta debajo del ojo y muere al final de la
       caja.
    4. **La caja tenia borde duro en los laterales.** La deformacion valia
       llena hasta la ultima columna y cero en la siguiente: medido, un salto de
       9,4/255 contra 2,8 tipico -- 3,4x, que en la piel lisa de la sien se lee
       como raya vertical. La mandibula ya lo resolvia con `peso_x`; el ojo no
       tenia pluma ninguna. Ahora la tiene, y cabe exactamente en el margen
       lateral de la caja.
    """
    if cierre <= 0.02:
        return
    for o in ojos:
        x0, y0, x1, y1 = o.caja
        if y1 <= y0 + 2 or x1 <= x0:
            continue
        bloque = rgb[y0:y1, x0:x1].astype(np.float32)
        h = bloque.shape[0]
        linea = o.parpado_y + (o.base_y - o.parpado_y) * cierre
        ys = np.arange(y0, y1, dtype=np.float32)

        arriba = y0 + (ys - y0) * (o.parpado_y - y0) / max(linea - y0, 1e-6)
        # Pendiente con techo: sin el va a infinito cuando la pestaña toca
        # `base_y`, y la caja entera satura en la ultima fila.
        tasa = min((o.base_y - o.parpado_y) / max(o.base_y - linea, 1e-6),
                   COMPRESION_MAX)
        comprime = o.parpado_y + (ys - linea) * tasa
        # Y la compresion vuelve a ser identidad justo bajo la pestaña: de ahi
        # para abajo es mejilla, y la mejilla no comprime.
        suelta = np.clip((ys - o.base_y) / max(SUELTA_ALTO * o.alto, 1e-6), 0, 1)
        suelta = suelta * suelta * (3 - 2 * suelta)
        debajo = comprime + (ys - comprime) * suelta

        # Parpado inferior y mejilla suben juntos, con el pico en la pestaña de
        # abajo y el efecto muriendo (cuadratico) en el borde de la caja.
        reaccion = np.clip((ys - o.parpado_y) / max(o.base_y - o.parpado_y, 1e-6), 0, 1)
        decae = np.clip(1 - (ys - o.base_y) / max(y1 - o.base_y, 1e-6), 0, 1)
        sube = SUBE_VECINO * o.alto * cierre * reaccion * decae * decae

        # Monotono en la fuerza. La suma de tres perfiles (estira, comprime,
        # sube) puede retroceder una fraccion de pixel donde se cruzan -- medido
        # en 0,12 a 0,20 px, invisible hoy. Pero retroceso es doblez: la imagen
        # vuelve atras y se vuelve pliegue. Una linea cierra la puerta a que
        # cualquier calibracion futura de SUBE_VECINO lo convierta en defecto
        # visible.
        origen = np.maximum.accumulate(
            np.where(ys <= linea, arriba, debajo) + sube)
        origen = np.clip(origen - y0, 0, h - 1.001)
        lo = origen.astype(np.int32)
        fr = (origen - lo).astype(np.float32)[:, None, None]
        salida = bloque[lo] * (1 - fr) + bloque[lo + 1] * fr

        # Sombra de la pestaña con centro fraccionario -- sin salto de linea.
        dist = np.abs(ys - linea) / CILIO_PX
        peso = np.clip(1.0 - dist * dist, 0.0, 1.0) ** 2
        salida *= (1 - 0.42 * cierre * peso)[:, None, None]

        # Pluma lateral: la deformacion se apaga antes del borde de la caja,
        # si no la ultima columna movida y la primera parada se tocan y se
        # vuelven raya.
        pluma = max(PLUMA_OJO * o.alto, 1.0)
        cols = np.arange(x0, x1, dtype=np.float32)
        lat = np.minimum(np.clip((cols - x0) / pluma, 0, 1),
                         np.clip((x1 - 1 - cols) / pluma, 0, 1))
        lat = (lat * lat * (3 - 2 * lat))[None, :, None]
        rgb[y0:y1, x0:x1] = np.clip(
            bloque + (salida - bloque) * lat, 0, 255).astype(np.uint8)


def _ancla(t: float, hitos: list[float]) -> float:
    """Tira de `t` hacia la frontera de frase mas cercana, si hay alguna al alcance.

    El parpadeo empieza un poco ANTES de la pausa: el ojo se cierra al terminar
    la frase, no despues de que la siguiente ya empezo.
    """
    if not hitos:
        return t
    i = bisect.bisect_left(hitos, t)
    cerca = [hitos[j] for j in (i - 1, i) if 0 <= j < len(hitos)]
    if not cerca:
        return t
    objetivo = min(cerca, key=lambda m: abs(m - t)) - PARPADEO_CIERRA_S
    return objetivo if abs(objetivo - t) <= PARPADEO_ANCLA_S else t


def parpadeos(duracion: float, semilla: str,
              sentencias: tuple[float, ...] = ()) -> list[tuple[float, float]]:
    """Ventanas de parpadeo, sorteadas con semilla fija (cuadro reproducible).

    El sorteo da el **ritmo**; las fronteras de frase dan el **lugar**. La gente
    parpadea en la puntuacion, en el respiro entre una idea y la siguiente, y no
    en medio de una palabra -- que es donde caia el sorteo puro. Eso es gratis:
    `tiempos.sentencias` ya sale del tiempo de palabra que devuelve el TTS.

    Sin tiempo de palabra (camino del MPT) `sentencias` viene vacio y el
    comportamiento es el antiguo, sorteado del inicio al final.
    """
    rng = random.Random(semilla)
    hitos = sorted(sentencias)
    ventanas: list[tuple[float, float]] = []
    t = rng.uniform(0.8, 2.2)
    while t < duracion:
        a = _ancla(t, hitos)
        ventanas.append((a, a + PARPADEO_S))
        if rng.random() < 0.18:      # parpadeo doble, como la gente
            t = a + PARPADEO_S + rng.uniform(0.12, 0.22)
            ventanas.append((t, t + PARPADEO_S))
        t += rng.uniform(PARPADEO_MIN_S, PARPADEO_MAX_S)
    return ventanas


def cierre_en(ventanas: list[tuple[float, float]], t: float) -> float:
    """Cuanto esta cerrado el ojo en `t`, de 0 (abierto) a 1 (cerrado).

    Asimetrico a proposito. La curva anterior era `sin(pi*u)**0.6` en una
    ventana de 0,13 s: medido, 2 cuadros cerrando y 2 abriendo, con el primer
    cuadro ya saltando de 0,00 a 0,82. El parpadeo humano no es simetrico ni
    entra en escalon -- el cierre es balistico y la apertura tarda el doble.
    Aqui: ~1,7 cuadros cerrando, ~1 sosteniendo, ~3,5 abriendo, cada tramo con
    ease cubico para no dejar esquina en la salida ni en la llegada.
    """
    for ini, _fin in ventanas:
        u = t - ini
        if u < 0.0 or u > PARPADEO_S:
            continue
        if u <= PARPADEO_CIERRA_S:
            return _ease_cubico(u / PARPADEO_CIERRA_S)
        if u <= PARPADEO_CIERRA_S + PARPADEO_SOSTIENE_S:
            return 1.0
        resta = (u - PARPADEO_CIERRA_S - PARPADEO_SOSTIENE_S) / PARPADEO_ABRE_S
        return 1.0 - _ease_cubico(resta)
    return 0.0


# ===========================================================================
# capas cabeza / torso
# ===========================================================================


def mascara_cabeza(r: Retrato) -> np.ndarray:
    """Peso de 1 (cabeza) a 0 (torso), con la costura en la linea de los hombros.

    Complementaria a proposito: cabeza + torso suman el alfa original, asi que
    con angulo cero el compuesto es identico al retrato -- y el test lo mide.
    La costura queda en el pecho, no en el cuello: cabeza que gira sobre cuello
    quieto es el efecto de muñeco de ventrilocuo.
    """
    # Medido en el recorte de la Iris: menton en 648, cuello en 682, hombro en
    # ~880. La costura tiene que caer entre el cuello y el hombro -- mas arriba
    # es cabeza girando sobre cuello quieto (muñeco de ventrilocuo); mas abajo
    # es el torso que se desvanece, porque el recorte acaba en 1053 y ya viene
    # esfumado.
    centro = r.neck_y + 0.35 * r.face_height
    media = 0.28 * r.face_height
    ys = np.arange(r.size[1], dtype=np.float32)
    u = np.clip((centro + media - ys) / max(2 * media, 1e-6), 0, 1)
    return (u * u * (3 - 2 * u)).astype(np.float32)[:, None]


def realce(img: Image.Image, accent: str) -> Image.Image:
    """Contraluz en el color del pilar + sombra detras: separa el presentador del fondo.

    La guia de identidad ya pedía "recorte #39FF88 en el hombro". Aqui ademas
    resuelve un problema medido: sobre clip claro, la silueta oscura sin
    contorno desaparece; sobre clip oscuro, la silueta oscura se vuelve borron.
    Calculado una vez -- depende solo del alfa, que el habla no cambia (boca y
    ojo son interior).
    """
    alfa = img.getchannel("A")
    nucleo = alfa.filter(ImageFilter.MinFilter(5))
    anillo = Image.fromarray(
        np.clip(np.asarray(alfa, dtype=np.int16) - np.asarray(nucleo, dtype=np.int16),
                0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3.0))
    rgb = tuple(int(accent.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    luz = Image.new("RGBA", img.size, (*rgb, 0))
    luz.putalpha(Image.fromarray(
        (np.asarray(anillo, dtype=np.float32) * 0.8).clip(0, 255).astype(np.uint8)))

    sombra = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sombra.putalpha(Image.fromarray(
        (np.asarray(alfa.filter(ImageFilter.GaussianBlur(13)), dtype=np.float32) * 0.38)
        .clip(0, 255).astype(np.uint8)))

    salida = Image.new("RGBA", img.size, (0, 0, 0, 0))
    salida.alpha_composite(sombra)
    salida.alpha_composite(img)
    salida.alpha_composite(luz)
    return salida


def _afin(origen: Image.Image, destino: tuple[int, int], *, escala: float,
          grados: float, pivote: tuple[float, float], objetivo: tuple[float, float],
          ) -> Image.Image:
    """Lleva `pivote` del origen al `objetivo` del destino, con escala y giro en una cuenta.

    Una transformacion sola en vez de resize + rotate: medido en 11 ms contra
    25 ms por cuadro en este retrato, y son miles de cuadros por video.
    """
    th = math.radians(grados)
    k = 1.0 / max(escala, 1e-6)
    cos, sin = math.cos(th), math.sin(th)
    a, b = k * cos, k * sin
    d, e = -k * sin, k * cos
    c = pivote[0] - (a * objetivo[0] + b * objetivo[1])
    f = pivote[1] - (d * objetivo[0] + e * objetivo[1])
    return origen.transform(destino, Image.AFFINE, (a, b, c, d, e, f),
                            resample=Image.BILINEAR)


def chip(nombre: str, accent: str) -> Image.Image:
    """Chip con el nombre del presentador -- entra y sale junto con la llamada."""
    texto = nombre.upper()
    f = font(MONO, 34)
    medida = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), texto, font=f)
    w = medida[2] - medida[0] + 54
    h = 62
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=14, fill=(10, 10, 12, 228))
    rgb = tuple(int(accent.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    d.rectangle((0, 12, 5, h - 13), fill=(*rgb, 255))
    d.text((22, (h - (medida[3] - medida[1])) / 2 - medida[1]), texto, font=f,
           fill=(233, 238, 241, 255))
    return img


# ===========================================================================
# el animador
# ===========================================================================


@dataclass(frozen=True)
class Capa:
    """Lo que `render/post.py` necesita saber para superponer la capa lista."""

    path: Path
    x: int
    y: int
    width: int
    height: int
    subtitle_y: int
    subtitle_start: float
    card_s: float
    frames: int
    fps: int
    seconds: float


class Animador:
    """Sintetiza los cuadros del presentador. Todo lo que no cambia va precalculado."""

    def __init__(self, retrato: Retrato, accent: str, nombre: str = "",
                 area: tuple[int, int, int, int] = AREA) -> None:
        self.retrato = retrato
        self.accent = accent
        self.area = area
        self.maxilar = preparar_maxilar(retrato)
        self.ojos = preparar_ojos(retrato)
        decorado = np.asarray(realce(retrato.imagen, accent)).copy()
        self._rgb = np.ascontiguousarray(decorado[:, :, :3])
        peso = mascara_cabeza(retrato)
        alfa = decorado[:, :, 3].astype(np.float32)
        self._alfa_cabeza = (alfa * peso).astype(np.uint8)
        self._alfa_torso = (alfa * (1 - peso)).astype(np.uint8)
        # Cada capa carga solo el rectangulo en el que tiene tinta. No es
        # detalle: la transformacion cuesta por pixel de SALIDA, y son dos mil
        # cuadros -- medido, recortar quito el 40% del tiempo de cada cuadro.
        self._cx_cabeza = _tinta(self._alfa_cabeza)
        self._cx_torso = _tinta(self._alfa_torso)
        self._chip = chip(nombre, accent) if nombre else None

    # ------------------------------------------------------------- cuadro

    def _capa(self, rgb: np.ndarray, alfa: np.ndarray,
              caja: tuple[int, int, int, int]) -> Image.Image:
        x0, y0, x1, y1 = caja
        buf = np.empty((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        buf[:, :, :3] = rgb[y0:y1, x0:x1]
        buf[:, :, 3] = alfa[y0:y1, x0:x1]
        return Image.fromarray(buf)

    def cuadro(self, t: float, abertura: float, cierre: float,
               poses: list[Pose]) -> Image.Image:
        """Un cuadro RGBA del tamano del area util de la capa."""
        ax, ay, aw, ah = self.area
        lienzo = Image.new("RGBA", (aw, ah), (0, 0, 0, 0))
        p = pose(poses, t)
        if p.alfa <= 0.004 or p.altura <= 1:
            return lienzo

        rgb = self._rgb.copy()
        aplicar_maxilar(rgb, self.maxilar, abertura)
        aplicar_parpadeo(rgb, self.ojos, cierre)
        boca = losa_boca(self.maxilar, abertura)

        rw, rh = self.retrato.size
        escala = (p.altura / rh) * (1 + RESPIRO * math.sin(2 * math.pi * t / 4.4))
        giro_torso = GIRO_TORSO * math.sin(2 * math.pi * t / 7.3 + 0.7)
        giro_cabeza = (GIRO_CABEZA * math.sin(2 * math.pi * t / 5.7 + 2.1)
                       + ACENTO_CABEZA * (abertura - 0.45))
        entrada = (1 - p.alfa) * 90     # desliza desde fuera mientras aparece

        # Donde cae el pivote del retrato en pantalla: la base del recorte toca
        # p.base y el centro horizontal en p.cx, girando en torno al pivote.
        pivote = self.retrato.pivote
        base = (p.cx - ax - entrada, p.base - ay)
        # Un pivote unico en pantalla para las dos capas: quien define donde
        # posa el cuerpo es el angulo del torso. Calcular un objetivo por capa
        # las separaria -- poco, pero las separaria, y la costura esta justo ahi.
        objetivo = _pivote_en_pantalla(base, pivote, (rw / 2, rh), escala, giro_torso)
        for alfa_capa, cx_fuente, grados in (
                (self._alfa_torso, self._cx_torso, giro_torso),
                (self._alfa_cabeza, self._cx_cabeza, giro_torso + giro_cabeza)):
            if cx_fuente is None:
                continue
            salida = _caja_salida(cx_fuente, pivote, objetivo, escala, grados)
            if salida is None:
                continue
            ox, oy, ow, oh = salida
            x0, y0 = cx_fuente[0], cx_fuente[1]
            img = self._capa(rgb, alfa_capa, cx_fuente)
            if boca is not None and y0 <= self.maxilar.boca_y < cx_fuente[3]:
                losa, canto = boca
                _pegar(img, losa, canto[0] - x0, canto[1] - y0)
            girada = _afin(img, (ow, oh), escala=escala, grados=grados,
                           pivote=(pivote[0] - x0, pivote[1] - y0),
                           objetivo=(objetivo[0] - ox, objetivo[1] - oy))
            if p.alfa < 0.999:
                girada.putalpha(girada.getchannel("A").point(
                    lambda v, a=p.alfa: int(v * a)))
            _pegar(lienzo, girada, ox, oy)
        self._chip_en(lienzo, p, base[0], base[1])
        return lienzo

    def _chip_en(self, lienzo: Image.Image, p: Pose, base_x: float,
                 base_y: float) -> None:
        """Solo en la llamada, donde se presenta -- desaparece con la tarjeta.

        En el cierre ya fue presentado; el chip reapareciendo al 10% de
        opacidad parecia defecto de composicion, no firma.
        """
        if self._chip is None:
            return
        franja = max(ALTURA_LLAMADA - ALTURA_CIERRE, 1.0)
        visible = min(1.0, max(0.0, (p.altura - ALTURA_CIERRE - 60) / franja)) * p.alfa
        if visible <= 0.02:
            return
        chip_img = self._chip
        if visible < 0.999:
            chip_img = chip_img.copy()
            chip_img.putalpha(chip_img.getchannel("A").point(
                lambda v, a=visible: int(v * a)))
        _pegar(lienzo, chip_img, int(base_x - chip_img.width / 2),
               int(base_y - p.altura * 0.115))


def _giro(dx: float, dy: float, grados: float, escala: float) -> tuple[float, float]:
    th = math.radians(grados)
    cos, sin = math.cos(th), math.sin(th)
    return (escala * (dx * cos - dy * sin), escala * (dx * sin + dy * cos))


def _pivote_en_pantalla(base: tuple[float, float], pivote: tuple[float, float],
                        ancla: tuple[float, float], escala: float,
                        grados: float) -> tuple[float, float]:
    """Posicion del pivote en pantalla, dado que el ancla (base del recorte) va en `base`."""
    dx, dy = _giro(ancla[0] - pivote[0], ancla[1] - pivote[1], grados, escala)
    return (base[0] - dx, base[1] - dy)


def _caja_salida(fuente: tuple[int, int, int, int], pivote: tuple[float, float],
                 objetivo: tuple[float, float], escala: float, grados: float,
                 margen: int = 3) -> tuple[int, int, int, int] | None:
    """Rectangulo que la capa ocupa en pantalla despues de girada y escalada."""
    xs, ys = [], []
    for px in (fuente[0], fuente[2]):
        for py in (fuente[1], fuente[3]):
            dx, dy = _giro(px - pivote[0], py - pivote[1], grados, escala)
            xs.append(objetivo[0] + dx)
            ys.append(objetivo[1] + dy)
    x0 = int(math.floor(min(xs))) - margen
    y0 = int(math.floor(min(ys))) - margen
    w = int(math.ceil(max(xs))) + margen - x0
    h = int(math.ceil(max(ys))) + margen - y0
    if w < 2 or h < 2:
        return None
    return (x0, y0, w, h)


def _tinta(alfa: np.ndarray) -> tuple[int, int, int, int] | None:
    """Rectangulo con tinta (margen de 2 px para que el filtro bilinear no corte)."""
    filas = np.flatnonzero(alfa.max(axis=1) > 0)
    columnas = np.flatnonzero(alfa.max(axis=0) > 0)
    if not filas.size or not columnas.size:
        return None
    return (max(0, int(columnas[0]) - 2), max(0, int(filas[0]) - 2),
            min(alfa.shape[1], int(columnas[-1]) + 3),
            min(alfa.shape[0], int(filas[-1]) + 3))


def _pegar(lienzo: Image.Image, pieza: Image.Image, x: int, y: int) -> None:
    """Compone recortando lo que cabe -- `alpha_composite` no acepta salir del lienzo."""
    sx0, sy0 = max(0, -x), max(0, -y)
    sx1 = min(pieza.width, lienzo.width - x)
    sy1 = min(pieza.height, lienzo.height - y)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    if (sx0, sy0, sx1, sy1) != (0, 0, pieza.width, pieza.height):
        pieza = pieza.crop((sx0, sy0, sx1, sy1))
    lienzo.alpha_composite(pieza, dest=(x + sx0, y + sy0))


def encode_command(out: Path, w: int, h: int, fps: int, *,
                   ffmpeg: str = "ffmpeg") -> list[str]:
    """Cuadros RGBA crudos a la entrada -> color premultiplicado + mascara, en un archivo.

    Premultiplicar antes de comprimir es lo que evita la franja: con alfa
    directo, el color salta del rostro al negro en el borde de la silueta, el
    h264 difumina ese salto y sobra un contorno oscuro. Premultiplicado, el
    color baja a cero en la misma rampa de la mascara, y `render/post.py`
    deshace la cuenta a la vuelta.
    """
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", "-",
            "-filter_complex",
            "[0:v]split=2[pm][al];"
            "[pm]premultiply=inplace=1,format=yuv420p[c];"
            "[al]alphaextract,format=gray[m]",
            "-map", "[c]", "-map", "[m]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


def render_capa(retrato: Retrato, audio: Path, out: Path, *, duracion: float,
                tiempos: Tiempos, accent: str, nombre: str = "",
                fps: int = FPS, ffmpeg: str = "ffmpeg",
                semilla: str = "ai-central") -> Capa:
    """Genera el `.mov` RGBA del presentador y devuelve donde superponerlo."""
    n = max(1, int(round(duracion * fps)))
    env = envolvente(audio, n, fps, ffmpeg=ffmpeg)
    ventanas = parpadeos(duracion, semilla, tiempos.sentencias)
    esc = plan(tiempos)
    anim = Animador(retrato, accent, nombre)
    ax, ay, aw, ah = anim.area

    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(encode_command(out, aw, ah, fps, ffmpeg=ffmpeg),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for k in range(n):
            t = k / fps
            cuadro = anim.cuadro(t, float(env[k]), cierre_en(ventanas, t), esc.poses)
            proc.stdin.write(cuadro.tobytes())
    except BrokenPipeError:  # pragma: no cover - solo con ffmpeg roto
        pass
    finally:
        proc.stdin.close()
    error = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg fallo al grabar el presentador: {error[-400:]}")
    return Capa(path=out, x=ax, y=ay, width=aw, height=ah,
                subtitle_y=esc.leyenda_y, subtitle_start=esc.leyenda_inicio,
                card_s=esc.tarjeta_s, frames=n, fps=fps, seconds=n / fps)


def conferir_presentador(final: Path, capa: Capa, trabajo: Path,
                         *, ffmpeg: str = "ffmpeg") -> str:
    """Comprueba en el MP4 **final** que el presentador poso de verdad en el cuadro.

    La capa puede generarse bien y no llegar a ningun lado: basta un `overlay`
    equivocado, una etiqueta de flujo cambiada o un `alphamerge` que el ffmpeg
    descarta en silencio. Nada de eso levanta error -- sale un MP4 valido, con
    la duracion correcta, sin presentador. Es el mismo defecto del MP4 mudo, y
    la regla del proyecto es medir el artefacto.

    **La medida cambio el 20/09/2026, porque la antigua habia empezado a
    mentir.** Comparaba el cuadro final con el cuadro *crudo* y pedía que la
    diferencia dentro de la mascara fuera el doble de la diferencia en un anillo
    alrededor. Eso mide "algo cambio aqui", y lo que cambio puede ser cualquier
    cosa: con el presentador fotorrealista y la tarjeta del gancho justo encima,
    el anillo paso a cambiar tanto como la mascara y el veredicto se volvio NO
    LLEGO AL CUADRO en videos que tenian al presentador en el cuadro, comprobado
    a ojo. Alarma que dispara siempre es alarma que nadie lee -- y el dia que
    tenga razon pasa inadvertida tambien.

    La medida nueva compara el final con la **propia capa**: dentro de la
    mascara, el cuadro final tiene que ser el presentador, y no solo "distinto
    del crudo". Correlacion (y no diferencia absoluta) porque la posproduccion
    aplica eq y vineta por encima -- las dos cambian el nivel del pixel y
    ninguna deshace la estructura del rostro. Un avatar ausente deja b-roll ahi,
    que no correlaciona con rostro ninguno.

    Solo entran los pixels de alfa casi lleno: la capa se graba con el color
    premultiplicado, asi que es donde la mascara esta llena que el color grabado
    equivale al color de verdad.

    Devuelve la frase para el log; no levanta -- video sin avatar sigue siendo
    publicable, y el motivo tiene que quedar registrado.
    """
    t = max(0.3, capa.card_s * 0.5)
    try:
        cuadro = _un_cuadro(final, t, trabajo / "_aceite_final.png", ffmpeg=ffmpeg)
        color = _un_cuadro(capa.path, t, trabajo / "_aceite_color.png",
                           ffmpeg=ffmpeg, stream="0:v:0")
        mascara = _un_cuadro(capa.path, t, trabajo / "_aceite_mascara.png",
                             ffmpeg=ffmpeg, stream="0:v:1")
    except (OSError, ValueError) as exc:
        return f"presentador: verificacion del cuadro indisponible ({exc})"
    if cuadro is None or color is None or mascara is None:
        return "presentador: verificacion del cuadro indisponible (cuadro vacio)"

    plano_mascara = np.zeros(cuadro.shape, dtype=np.uint8)
    plano_color = np.zeros(cuadro.shape, dtype=np.uint8)
    h = min(mascara.shape[0], cuadro.shape[0] - capa.y)
    w = min(mascara.shape[1], cuadro.shape[1] - capa.x)
    if h <= 0 or w <= 0:
        return "presentador: capa fuera del cuadro"
    plano_mascara[capa.y:capa.y + h, capa.x:capa.x + w] = mascara[:h, :w]
    plano_color[capa.y:capa.y + h, capa.x:capa.x + w] = color[:h, :w]

    solido = plano_mascara > 230
    if solido.sum() < 4000:
        return "presentador: mascara vacia en el instante de la llamada"
    a = cuadro[solido].astype(np.float64)
    b = plano_color[solido].astype(np.float64)
    if a.std() < 4 or b.std() < 4:
        return "presentador: sin estructura para comparar en la mascara"
    corr = float(np.corrcoef(a, b)[0, 1])
    veredicto = "OK" if corr > 0.55 else "NO LLEGO AL CUADRO"
    return (f"presentador en cuadro: {veredicto} "
            f"(t={t:.1f}s, correlacion con la capa {corr:.2f}, "
            f"{int(solido.sum())} px)")


def _un_cuadro(video: Path, t: float, destino: Path, *, ffmpeg: str = "ffmpeg",
               stream: str = "0:v:0") -> np.ndarray | None:
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}",
         "-i", str(video), "-map", stream, "-frames:v", "1", str(destino)],
        capture_output=True, timeout=120)
    if r.returncode != 0 or not destino.exists():
        return None
    with Image.open(destino) as img:
        return np.asarray(img.convert("L")).copy()


__all__ = ["AREA", "Animador", "Capa", "Escenificacion", "Maxilar", "Ojo",
           "Pose", "Retrato", "Tiempos", "Y_LEYENDA_CON_PRESENTADOR",
           "aplicar_maxilar", "aplicar_parpadeo", "cargar", "cierre_en",
           "chip", "conferir_presentador", "encode_command", "envolvente",
           "losa_boca", "mascara_cabeza", "parpadeos", "plan", "pose",
           "preparar_maxilar", "preparar_ojos", "realce", "render_capa",
           "tiempos_estimados", "tiempos_por_palabras"]
