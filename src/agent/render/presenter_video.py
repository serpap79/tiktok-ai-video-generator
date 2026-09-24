"""Presentador de video: el clip humano da la persona, nosotros ponemos la boca.

`render/presenter.py` sintetiza todo a partir de un retrato quieto -- parpadeo,
balanceo de cabeza, respiracion. Funciona y su limite esta declarado ahi: es una
silueta convincente, no una persona. Desde el 20/09/2026 existe un clip
fotorrealista de THEO (Hailuo, identidad fija del `brand.json`) en el que el
parpadeo y el balanceo **ya son humanos**, porque se generaron como video. Lo
unico que no puede venir hecho es la boca, que depende de la narracion de ese
video concreto.

Asi que la division de trabajo aqui es la opuesta a la del modulo antiguo:

| lo que                      | de donde viene                                |
|-----------------------------|-----------------------------------------------|
| parpadeo, cabeza, respiracion | del clip base, cuadro a cuadro (humano)     |
| boca                        | sintetizada aqui, al ritmo de la narracion    |
| tamano y lugar en pantalla  | de la escenificacion del guion (`presenter.plan`) |

Nada de parpadeo sorteado ni giro por seno: seria movimiento nuestro peleando
con el movimiento que ya esta en el cuadro.

**La boca sigue al rostro.** `scripts/make_presenter_video.py` grabo los puntos
de la malla en *cada* cuadro del clip; la cabeza se mueve, y una boca dibujada
en coordenadas fijas se despega del rostro en el primer balanceo. Aqui cada
cuadro lee los puntos de su propio cuadro.

**La boca tiene forma, no solo tamano.** La abertura sale de
`render/visemes.py` (visema sobre el tiempo de palabra del TTS) modulada por la
envolvente de energia de la narracion: la letra dice que forma, el audio dice
con cuanta fuerza. En un rostro fotorrealista esto dejo de ser refinamiento --
boca abierta en la /m/ es gatillo de valle inquietante, y en primer plano nadie
lo perdona.

El clip base es corto y la narracion no: la lectura es en **vaiven** (hacia
adelante hasta el final, hacia atras hasta el principio). La union por corte se
midio y se descarto -- el mejor par de cuadros no vecinos del clip de THEO
difiere 6,74/255, contra 1,13 entre vecinos: seis veces el movimiento normal,
daria un golpe visible en cada vuelta. En vaiven el giro usa cuadros vecinos y
es invisible. El precio, declarado: el parpadeo se oye al reves una vez por
ciclo (cierra despacio, abre rapido -- lo inverso del humano). Con un clip base
mas largo que la narracion el vaiven nunca llega a girar.

Nota sobre claves de datos: los nombres de los puntos (`labio_sup`, `queixo`,
`mand_esq`...) y los arcos `labio_in_baixo_NN` / `labio_in_cima_NN` son el
contrato del JSON generado por `scripts/make_presenter_video.py` y no se
renombran en codigo.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from agent.render.presenter import (
    AREA,
    CLAVES_PT,
    FPS,
    Capa,
    Pose,
    Tiempos,
    _afin,
    _ease_cubico,
    _pegar,
    chip,
    encode_command,
    envolvente,
    plan,
    pose,
)
from agent.render.visemes import pista


def _clave(nombre: str) -> str:
    """Normaliza las claves pt de los JSON horneados al contrato castellano."""
    return CLAVES_PT.get(nombre, nombre)

# Caida maxima del menton como fraccion de la altura del rostro -- la misma
# medida del presentador quieto, que salio de mirar cuadro a cuadro: por encima
# de ~10% el rostro visiblemente se alarga en las silabas abiertas.
MAXILAR = 0.105
# Longitud de la rampa de la mandibula en el centro de la boca. Tiene que ser
# un escalon: es ahi donde la boca abre. Con los 18 px de la primera version el
# labio de abajo casi no bajaba y la boca salia aplastada, una raya horizontal
# en vez de un hueco.
ESCALON_PX = 3.0
# Por encima de esta abertura aparece diente. Mas bajo que en el presentador
# quieto (0,52): en un rostro real el diente es parte de la boca abierta, y la
# falta se lee como agujero negro. Por debajo la boca esta solo entreabierta y
# no aparece diente.
DIENTE_MIN = 0.30
# Supermuestreo de las mascaras. El `ImageDraw` de Pillow no tiene antialiasing
# ninguno (medido el 20/09/2026: cero pixel de borde parcial); dibujar 4x mayor
# y reducir por media de area (`Image.BOX`) es lo que da el borde suave.
SS = 4
# Por debajo de esta altura en pixel la boca entra en degradado en vez de surgir.
BOCA_MIN_PX = 3.0
# Margen en torno al poligono del hueco, para que el borron tenga donde caer.
MARGEN_HUECO = 14
# Cuanto la anchura del visema estira o recoge la boca. 0,12 salio de mirar: a
# 0,20 la /u/ se vuelve pico de dibujo animado, a 0,06 la /i/ no se distingue.
ANCHURA_MAX = 0.12
# Pluma lateral del campo de la mandibula: sin ella la deformacion muere de una
# linea a la siguiente en el borde de la caja y aparece una costura vertical en
# la mejilla.
PLUMA_X = 46.0
# Suelo de la modulacion por la energia. La forma del visema nunca se anula por
# el audio: la envolvente puede estar baja en una consonante sorda que la boca
# hace igualmente.
ENERGIA_PISO = 0.55


# ===========================================================================
# el clip base horneado
# ===========================================================================


@dataclass(frozen=True)
class Base:
    """El clip del presentador con los puntos del rostro ya medidos por cuadro."""

    id: str
    video: Path
    size: tuple[int, int]
    fps: float
    frames: int
    head_top: int
    neck_y: int
    face_height: float
    pivote: tuple[float, float]
    pistas: dict[str, np.ndarray] = field(default_factory=dict)
    n_inferior: int = 0
    n_superior: int = 0

    def puntos(self, k: int) -> dict[str, tuple[float, float]]:
        k = min(max(k, 0), self.frames - 1)
        return {nombre: (float(v[k, 0]), float(v[k, 1]))
                for nombre, v in self.pistas.items()}

    def contorno(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Los dos arcos del labio interno en este cuadro: (inferior, superior).

        Es lo que sustituyo a la elipse dibujada a mano. La boca tiene esquina
        en punta y la elipse tiene tangente vertical en la esquina: la elipse
        cubria el borde del labio de arriba y dejaba un contorno fantasma por
        debajo -- visto en la ampliacion 2x del artefacto del 20/09/2026.
        """
        k = min(max(k, 0), self.frames - 1)
        inferior = np.array([self.pistas[f"labio_in_baixo_{j:02d}"][k]
                             for j in range(self.n_inferior)], dtype=np.float32)
        superior = np.array([self.pistas[f"labio_in_cima_{j:02d}"][k]
                             for j in range(self.n_superior)], dtype=np.float32)
        return inferior, superior


def cargar_base(meta_json: Path) -> Base:
    """Lee `<id>_base.json` y comprueba que el mp4 hermano existe."""
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    video = meta_json.with_name(meta_json.stem + ".mp4")
    if not video.exists():
        raise FileNotFoundError(
            f"{video} no existe -- corre scripts/make_presenter_video.py")
    if not meta.get("contorno"):
        raise ValueError(
            f"{meta_json.name} se horneo sin el contorno del labio -- "
            "corre scripts/make_presenter_video.py otra vez")
    contorno = meta.get("contorno", {})
    return Base(
        id=meta.get("id", meta_json.stem),
        video=video,
        size=(int(meta["size"][0]), int(meta["size"][1])),
        fps=float(meta["fps"]),
        frames=int(meta["frames"]),
        head_top=int(meta["head_top"]),
        neck_y=int(meta["neck_y"]),
        face_height=float(meta["face_height"]),
        pivote=(float(meta["pivot"][0]), float(meta["pivot"][1])),
        pistas={_clave(nombre): np.asarray(v, dtype=np.float32)
                for nombre, v in meta["tracks"].items()},
        n_inferior=int(contorno.get("inferior", contorno.get("baixo", 0))),
        n_superior=int(contorno.get("superior", contorno.get("cima", 0))),
    )


def indice_vaiven(t: float, base: Base) -> int:
    """Que cuadro del clip toca en el instante `t`, yendo y volviendo.

    Fuera del vaiven no hay union invisible: medido en el clip de THEO, el
    mejor corte entre cuadros no vecinos cuesta 6x el movimiento normal.
    """
    if base.frames <= 1:
        return 0
    pos = t * base.fps
    ciclo = 2 * (base.frames - 1)
    r = pos % ciclo
    if r > base.frames - 1:
        r = ciclo - r
    return int(round(r))


class Cuadros:
    """Acceso aleatorio a los cuadros del clip horneado.

    El mp4 viene en dos pistas (color premultiplicado + mascara). Decodificar el
    clip entero una vez a un archivo mapeado en memoria cuesta segundos y
    cambia CPU por disco; decodificar bajo demanda costaria un proceso de ffmpeg
    por cuadro de salida, que son miles.
    """

    def __init__(self, base: Base, trabajo: Path, *, ffmpeg: str = "ffmpeg") -> None:
        w, h = base.size
        self.base = base
        trabajo.mkdir(parents=True, exist_ok=True)
        destino = trabajo / f"{base.id}_base_rgba.raw"
        crudo = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(base.video),
             "-filter_complex",
             "[0:v:0][0:v:1]alphamerge,unpremultiply=inplace=1,format=rgba[o]",
             "-map", "[o]", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
            capture_output=True, check=True, timeout=600).stdout
        esperado = base.frames * h * w * 4
        if len(crudo) < esperado:
            raise RuntimeError(
                f"clip base incompleto: {len(crudo)} bytes, esperaba {esperado}")
        destino.write_bytes(crudo[:esperado])
        self._datos = np.memmap(destino, dtype=np.uint8, mode="r",
                                shape=(base.frames, h, w, 4))

    def __getitem__(self, k: int) -> np.ndarray:
        return np.asarray(self._datos[min(max(k, 0), self.base.frames - 1)])


# ===========================================================================
# la boca
# ===========================================================================


@dataclass(frozen=True)
class Tonos:
    """Colores de la boca, muestreados una vez en el clip.

    Una vez, y no por cuadro, a proposito: la luz del clip esta fija, y un color
    re-muestreado en cada cuadro temblaria con el ruido del grano -- boca
    parpadeando de tono es peor que boca de tono levemente equivocado.
    """

    interior: tuple[float, float, float]
    diente: tuple[float, float, float]
    lengua: tuple[float, float, float]


def medir_tonos(cuadro: np.ndarray, pts: dict[str, tuple[float, float]]) -> Tonos:
    px = cuadro[:, :, :3].astype(np.float32)
    cx, cy = pts["labio_sup"]
    muestra = px[int(cy) - 4:int(cy) + 5, int(cx) - 14:int(cx) + 15].reshape(-1, 3)
    labio = muestra.mean(axis=0) if muestra.size else np.array([120.0, 80.0, 80.0])
    rostro = px[int(pts["ojo_izq_abajo"][1]):int(pts["menton"][1]),
                int(pts["mandibula_izq"][0]):int(pts["mandibula_der"][0])].reshape(-1, 3)
    # p97 y no p88: el recorte del rostro toma barba y sombra del menton, y la
    # media demasiado alta tiraba del "claro" por debajo del tono del labio --
    # el diente salia mas oscuro que la boca, que es lo contrario de un diente.
    claro = (np.percentile(rostro, 97, axis=0) if rostro.size
             else np.array([200.0, 190.0, 185.0]))
    # El diente no es blanco ni es piel: es casi neutro, sacado del brillo del
    # propio rostro (cada presentador tiene su exposicion) y rebajado porque
    # esta dentro de la boca, en sombra. Tomar el color de la piel directo
    # dejaba el diente rosado -- se leia como plastico, no como diente.
    luz = float(np.mean(claro))
    diente = np.clip(luz * 1.18 * np.array([1.0, 0.985, 0.95]), 60, 242)
    # La boca por dentro es oscura, no negra: a 0,20 el hueco se volvia un
    # agujero plano y el diente no tenia contra que contrastar.
    # La lengua existe para que el hueco deje de ser un agujero negro. No
    # necesita forma: basta un volumen mate y rojizo al fondo, que es lo que se
    # ve de reojo en una boca hablando.
    return Tonos(interior=tuple(np.clip(labio * 0.38, 20, 78)),
                 diente=tuple(diente),
                 lengua=tuple(np.clip(labio * 0.62, 40, 150)))


def _reinterp_y2(bloque: np.ndarray, desplaz: np.ndarray) -> np.ndarray:
    """`salida[y,x] = bloque[y - desplaz[y,x], x]`, lineal entre filas.

    Desplazamiento por PIXEL, no por fila: la mandibula no baja en bloque (ver
    `abrir_maxilar`), asi que cada columna tiene su propio perfil.
    """
    h = bloque.shape[0]
    ys = np.clip(np.arange(h, dtype=np.float32)[:, None] - desplaz, 0, h - 1.001)
    lo = ys.astype(np.int32)
    fr = (ys - lo)[:, :, None].astype(np.float32)
    columnas = np.arange(bloque.shape[1])[None, :]
    return bloque[lo, columnas] * (1 - fr) + bloque[lo + 1, columnas] * fr


def campo_maxilar(pts: dict[str, tuple[float, float]], face_h: float,
                  abertura: float, shape: tuple[int, int],
                  ) -> tuple[tuple[int, int, int, int], np.ndarray, float] | None:
    """El desplazamiento vertical, en pixel, de cada punto de la mitad de abajo.

    Sale como funcion propia porque **el campo es el artefacto**: mirar la
    imagen deformada y creer que esta bien fue como la raya horizontal en la
    mejilla paso inadvertida. Con el campo en la mano se puede medir el salto
    entre lineas vecinas y bloquearlo en test.

    El modelo tiene tres factores, y cada uno viene de un defecto visto en la
    ampliacion:

    - `S(y)` -- la mandibula es hueso: de la linea de los labios al menton baja
      entera (escalon corto), y solo el cuello absorbe la diferencia. Una rampa
      larga aqui apretaba el labio de abajo y la boca se volvia una raya.
    - `amp(x)` -- el hueso GIRA en torno a la articulacion cerca de la oreja, asi
      que el menton baja todo y la esquina de la mandibula casi nada. Sin eso
      sobraba un escalon de 61 px cruzando la mejilla.
    - `abre(x, y)` -- **solo la boca abre en la linea de los labios.** A la
      altura del labio el desplazamiento vale el perfil de abertura (cero en las
      comisuras); bajando hacia el menton se vuelve 1 en toda la anchura. Es lo
      que hace la esquina de la boca quedar pegada mientras el centro abre, y a
      la vez no deja ningun escalon en la mejilla, porque ahi el campo empieza
      en cero y crece.
    """
    h, w = shape
    labio_y = (pts["labio_sup"][1] + pts["labio_inf"][1]) / 2
    menton_y = pts["menton"][1]
    fin = menton_y + 0.5 * face_h
    y0 = int(max(0, labio_y - 6))
    y1 = int(min(h, fin + 8))
    x0 = int(max(0, pts["rostro_izq"][0] - PLUMA_X))
    x1 = int(min(w, pts["rostro_der"][0] + PLUMA_X))
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None

    caida = MAXILAR * face_h * abertura
    columnas = np.arange(x0, x1, dtype=np.float32)
    filas = np.arange(y0, y1, dtype=np.float32)[:, None]

    sube = np.clip((filas - labio_y) / ESCALON_PX, 0, 1)
    sube = sube * sube * (3 - 2 * sube)
    vuelta = np.clip(1 - (filas - menton_y) / max(fin - menton_y, 1e-6), 0, 1)
    vuelta = vuelta * vuelta * (3 - 2 * vuelta)
    campo = caida * np.minimum(sube, vuelta) * _amplitud_x(pts, columnas)[None, :]

    # Profundidad: 0 en la linea de los labios, 1 en el menton. En la linea del
    # labio quien manda es el perfil de abertura; en el menton, la mandibula
    # entera.
    prof = np.clip((filas - labio_y) / max(menton_y - labio_y, 1e-6), 0, 1)
    prof = prof * prof * (3 - 2 * prof)
    abre = _abre_x(pts, columnas)[None, :]
    campo *= abre + (1.0 - abre) * prof

    # Pluma lateral: el campo ya muere en el rostro, pero la caja es mas ancha
    # que el y una costura vertical en el borde de la caja fue defecto pagado en
    # el presentador quieto -- aqui no vuelve.
    peso = np.minimum(
        np.clip((columnas - (pts["mandibula_izq"][0] - PLUMA_X)) / PLUMA_X, 0, 1),
        np.clip(((pts["mandibula_der"][0] + PLUMA_X) - columnas) / PLUMA_X, 0, 1))
    campo *= (peso * peso * (3 - 2 * peso))[None, :]
    return (x0, y0, x1, y1), campo, float(caida)


def abrir_maxilar(cuadro: np.ndarray, pts: dict[str, tuple[float, float]],
                  face_h: float, abertura: float) -> float:
    """Baja la mandibula en el sitio. Devuelve la caida en pixel en el menton."""
    if abertura <= 0.01:
        return 0.0
    hecho = campo_maxilar(pts, face_h, abertura, cuadro.shape[:2])
    if hecho is None:
        return 0.0
    (x0, y0, x1, y1), campo, caida = hecho
    bloque = cuadro[y0:y1, x0:x1].astype(np.float32)
    cuadro[y0:y1, x0:x1] = _reinterp_y2(bloque, campo).astype(np.uint8)
    return caida


def extender_labios(cuadro: np.ndarray, pts: dict[str, tuple[float, float]],
                    anchura: float) -> None:
    """Estira (/i/) o recoge (/u/) la boca en horizontal, en el sitio.

    El campo muere en los bordes de la caja a proposito: sin eso la mejilla
    entera se moveria junto y el rostro cambiaria de anchura en cada silaba.
    """
    if abs(anchura) < 0.02:
        return
    h, w = cuadro.shape[:2]
    cx = (pts["boca_izq"][0] + pts["boca_der"][0]) / 2
    cy = (pts["labio_sup"][1] + pts["labio_inf"][1]) / 2
    boca_w = max(pts["boca_der"][0] - pts["boca_izq"][0], 8.0)
    # La caja llega hasta donde llega el LABIO, y no un tanto arbitrario de la
    # anchura de la boca. Con `boca_w * 0,62` media 121 px de alto -- de la
    # nariz al menton -- y el estiramiento ondulaba la barba y la mejilla en
    # cada silaba, bien visible en la ampliacion 2x. El labio es lo que se
    # estira; la piel en torno, no.
    labio_h = abs(pts["labio_inf_ext"][1] - pts["labio_sup_ext"][1])
    rx, ry = boca_w * 0.85, max(labio_h * 0.65, 20.0)
    x0, x1 = int(max(0, cx - rx)), int(min(w, cx + rx))
    y0, y1 = int(max(0, cy - ry)), int(min(h, cy + ry))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return

    escala = 1.0 + ANCHURA_MAX * anchura
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    # Desplazamiento que llevaria a la escala exacta, apagado en los bordes.
    desplaz = (xs - cx) * (1.0 / max(escala, 1e-6) - 1.0)
    tx = np.clip(1 - np.abs(xs - cx) / max(rx, 1e-6), 0, 1)
    ty = np.clip(1 - np.abs(ys - cy) / max(ry, 1e-6), 0, 1)
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    campo = desplaz[None, :] * tx[None, :] * ty[:, None]

    bloque = cuadro[y0:y1, x0:x1].astype(np.float32)
    src = np.clip(np.arange(x1 - x0, dtype=np.float32)[None, :] + campo,
                  0, (x1 - x0) - 1.001)
    lo = src.astype(np.int32)
    fr = (src - lo)[:, :, None].astype(np.float32)
    filas = np.arange(y1 - y0)[:, None]
    cuadro[y0:y1, x0:x1] = (bloque[filas, lo] * (1 - fr)
                            + bloque[filas, lo + 1] * fr).astype(np.uint8)


def _abre_x(pts: dict[str, tuple[float, float]], xs: np.ndarray) -> np.ndarray:
    """Cuanto abre la boca en cada columna: lleno en el centro, cero en las comisuras.

    La esquina de la boca no abre -- es donde el labio de arriba encuentra al de
    abajo. Sin este perfil el hueco salia como una **barra rectangular** de
    esquina en escuadra, porque el labio de abajo bajaba lo mismo en el centro
    y en la punta. Con el, el hueco se vuelve lente: lleno en el centro,
    cerrando en punta en las dos esquinas.

    El exponente por debajo de 1 ensancha el centro: la boca abierta es casi
    igual a lo largo del centro y solo cierra cerca de la punta, que es
    distinto de un seno puro.
    """
    izq, der = pts["boca_izq"][0], pts["boca_der"][0]
    u = np.clip((xs - izq) / max(der - izq, 1e-6), 0.0, 1.0)
    # El clip del seno no es paranoia: `sin(pi)` devuelve -8,7e-17 en coma
    # flotante, y base negativa con exponente fraccionario es NaN -- que baja
    # entero hasta el indice de la reinterpolacion y tira el cuadro.
    return np.clip(np.sin(np.pi * u), 0.0, 1.0) ** 0.65


def _amplitud_x(pts: dict[str, tuple[float, float]], xs: np.ndarray) -> np.ndarray:
    """Cuanto del giro de la mandibula llega a cada columna (el mismo perfil del campo).

    El labio de abajo es hueso: baja con la mandibula. Para que el hueco pintado
    case con el pixel que la deformacion movio, los dos tienen que usar este
    mismo perfil -- si divergen, sobra rendija de un lado y el hueco cubre labio
    del otro.
    """
    cx = (pts["boca_izq"][0] + pts["boca_der"][0]) / 2
    boca_media = max((pts["boca_der"][0] - pts["boca_izq"][0]) / 2 * 0.84, 6.0)
    rostro_medio = max((pts["rostro_der"][0] - pts["rostro_izq"][0]) / 2, boca_media + 8.0)
    fuera = np.clip((np.abs(xs - cx) - boca_media) / (rostro_medio - boca_media), 0, 1)
    fuera = fuera * fuera * (3 - 2 * fuera)
    return 1.0 - 0.68 * fuera


def hueco_boca(pts: dict[str, tuple[float, float]],
               contorno: tuple[np.ndarray, np.ndarray], caida: float,
               abertura: float, anchura: float, tonos: Tonos
               ) -> tuple[Image.Image, tuple[int, int]] | None:
    """El hueco entre los labios, recortado por el contorno REAL de la boca.

    La version anterior dibujaba una elipse entre los labios. En la ampliacion
    2x del artefacto del 20/09/2026 se veian los tres defectos que eso cuesta:

    1. la elipse tiene tangente vertical en la esquina y la boca tiene esquina
       en punta, asi que avanzaba por encima del borde del labio de arriba;
    2. donde no llegaba quedaba el labio de arriba **duplicado** por la
       deformacion, un contorno fantasma justo debajo del verdadero;
    3. la boca tenia la misma forma en toda silaba, porque la elipse solo
       cambiaba de tamano.

    Aqui el hueco es el poligono entre el arco interno de ARRIBA (que queda
    quieto, es del craneo) y el arco interno de ABAJO desplazado por la caida de
    la mandibula (que es hueso y baja). Es decir: el hueco es exactamente el
    area que el labio de abajo despejo, y la forma viene de la boca de THEO, no
    de una elipse nuestra.

    El diente cuelga del techo del poligono columna a columna, asi que nace con
    la curva del labio de arriba gratis.
    """
    inferior, superior = contorno
    if inferior.size < 3 or superior.size < 3 or caida <= 0:
        return None
    cx = (pts["boca_izq"][0] + pts["boca_der"][0]) / 2
    # Boca abierta recoge en las esquinas, y el visema aun estira o redondea.
    escala_x = (1.0 - 0.10 * abertura) * (1.0 + ANCHURA_MAX * anchura)
    # Los dos arcos corren en sentidos opuestos; alineados, su media es la
    # **linea de costura** de los labios -- donde la boca de hecho se abre.
    #
    # El hueco nace de esa costura, y no del arco inferior crudo, porque en la
    # malla los dos arcos quedan 1 a 3 px separados incluso con la boca cerrada
    # (grosor del labio y ruido de medida). Con el arco crudo, una boca cerrada
    # empezaba ya con area y el resultado era una raya oscura permanente entre
    # los labios. Desde la costura, area cero en reposo, por construccion.
    superior_a = superior[::-1]
    costura = np.column_stack([cx + ((superior_a[:, 0] + inferior[:, 0]) / 2 - cx) * escala_x,
                               (superior_a[:, 1] + inferior[:, 1]) / 2])
    # El labio de abajo baja con el hueso, por el MISMO perfil que el campo usa
    # en la linea de los labios -- si divergen, sobra rendija de un lado y el
    # hueco cubre labio del otro. Se afina un poco al abrir, porque el labio se
    # estira: sin eso se desliza como una losa rigida.
    desplaz = (caida * _abre_x(pts, costura[:, 0])
               * _amplitud_x(pts, costura[:, 0]) * (1.0 - 0.12 * abertura))
    if float(desplaz.max()) < 0.6:
        # Menos de medio pixel de hueco: no hay boca abierta, y dibujar aqui
        # gastaria un mosaico invisible en todo cuadro de consonante cerrada.
        return None
    superior_e = costura
    inferior_e = np.column_stack([costura[:, 0], costura[:, 1] + desplaz])

    # El borde de abajo de izquierda a derecha y el de arriba a la vuelta: es
    # el bucle cerrado. Apilar los dos en el mismo sentido hace que el poligono
    # se cruce en el medio y la boca salga con una X dentro -- ocurrio, se ve.
    poli = np.vstack([inferior_e, superior_e[::-1]])
    x0 = math.floor(poli[:, 0].min()) - MARGEN_HUECO
    y0 = math.floor(poli[:, 1].min()) - MARGEN_HUECO
    w = int(math.ceil(poli[:, 0].max()) - x0) + MARGEN_HUECO
    h = int(math.ceil(poli[:, 1].max()) - y0) + MARGEN_HUECO
    if w < 4 or h < 4 or w > 4000 or h > 4000:
        return None

    # Mascara a SS veces el tamano: el `ImageDraw` de Pillow no tiene
    # antialiasing ninguno (medido: cero pixel de borde parcial). La parte
    # fraccionaria de la posicion entra en las coordenadas del dibujo grande,
    # asi que el hueco crece en pasos menores que un pixel en vez de saltar.
    grande = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(grande).polygon(
        [((px - x0) * SS, (py - y0) * SS) for px, py in poli], fill=255)
    alfa = np.asarray(grande.resize((w, h), Image.BOX), dtype=np.float32) / 255.0
    if alfa.max() <= 0.02:
        return None
    # Entrada suave: el hueco nace de cero en vez de aparecer con 2 px de golpe.
    altura = float((alfa > 0.5).sum(axis=0).max())
    alfa *= _ease_cubico(min(1.0, max(altura, 0.1) / BOCA_MIN_PX))

    # --- profundidad: donde empieza y acaba el hueco, columna a columna
    dentro = alfa > 0.35
    hay = dentro.any(axis=0)
    techo = np.where(hay, dentro.argmax(axis=0), 0).astype(np.float32)
    fondo = np.where(hay, h - 1 - dentro[::-1].argmax(axis=0), 0).astype(np.float32)
    hueco_h = np.maximum(fondo - techo, 1.0)
    filas = np.arange(h, dtype=np.float32)[:, None]
    prof = np.clip((filas - techo[None, :]) / hueco_h[None, :], 0, 1)

    color = np.empty((h, w, 3), dtype=np.float32)
    color[:] = tonos.interior
    # Fondo de la boca mas oscuro que el frente, y esquina mas honda que el centro.
    color *= (1.0 - 0.34 * prof)[:, :, None]
    borde = np.clip(np.abs(np.arange(w, dtype=np.float32) - (cx - x0))
                    / max(w * 0.5, 1e-6), 0, 1)
    color *= (1.0 - 0.30 * borde * borde)[None, :, None]

    # --- lengua: el fondo del hueco no es negro
    lengua = np.clip((prof - 0.46) / 0.22, 0, 1)
    lengua *= np.clip(1 - borde[None, :] / 0.82, 0, 1)
    lengua = lengua * lengua * (3 - 2 * lengua) * 0.75
    color += (np.asarray(tonos.lengua, dtype=np.float32) - color) * lengua[:, :, None]

    if abertura > DIENTE_MIN:
        fuerza = min(1.0, (abertura - DIENTE_MIN) / (1 - DIENTE_MIN))
        # El diente cuelga del labio de ARRIBA (es del craneo, no baja con el
        # hueso): nace pegado al techo del hueco, con una linea de sombra
        # antes, y termina en una ARISTA. Con la bajada suave que estaba aqui
        # antes se leia como una barra de metal pulido dentro de un agujero --
        # lo que identifica un diente es el borde de corte recto abajo, no el
        # brillo.
        sube = np.clip((prof - 0.03) / 0.09, 0, 1)
        baja = 1.0 - np.clip((prof - (0.30 + 0.10 * fuerza)) / 0.07, 0, 1)
        vy = np.clip(sube, 0, 1) * np.clip(baja, 0, 1)
        vy = vy * vy * (3 - 2 * vy)
        vx = np.clip(1 - borde / 0.90, 0, 1)
        vx = vx * vx * (3 - 2 * vx)
        # Separacion entre dientes: poca, solo para romper el gradiente liso.
        # Un diente de THEO mide ~14 px en pantalla, de ahi sale el periodo.
        fase = (np.arange(w, dtype=np.float32) - (cx - x0)) / 14.0
        surco = 1.0 - 0.10 * (0.5 + 0.5 * np.cos(2 * np.pi * fase))
        peso = (alfa * vy * (vx * surco)[None, :] * (0.55 + 0.45 * fuerza))[:, :, None]
        color += (np.asarray(tonos.diente, dtype=np.float32) - color) * peso

    buf = np.empty((h, w, 4), dtype=np.uint8)
    buf[:, :, :3] = np.clip(color, 0, 255).astype(np.uint8)
    buf[:, :, 3] = np.clip(alfa * 255.0, 0, 255).astype(np.uint8)
    losa = Image.fromarray(buf).filter(
        ImageFilter.GaussianBlur(max(0.6, altura * 0.045)))
    return losa, (x0, y0)


def hablar(cuadro: np.ndarray, pts: dict[str, tuple[float, float]],
           contorno: tuple[np.ndarray, np.ndarray], face_h: float,
           abertura: float, anchura: float, tonos: Tonos) -> None:
    """Una silaba en el rostro: anchura del labio, mandibula y el hueco encima.

    En este orden a proposito: el estiramiento mueve el labio cerrado, la
    mandibula baja lo que esta debajo de el, y el hueco se pinta al final, en el
    agujero que los dos dejaron.
    """
    extender_labios(cuadro, pts, anchura)
    caida = abrir_maxilar(cuadro, pts, face_h, abertura)
    if caida <= 0:
        return
    hecho = hueco_boca(pts, contorno, caida, abertura, anchura, tonos)
    if hecho is None:
        return
    losa, canto = hecho
    _componer(cuadro, losa, canto[0], canto[1])


def _componer(cuadro: np.ndarray, pieza: Image.Image, x: int, y: int) -> None:
    """Superpone el mosaico en el color del cuadro, preservando el alfa de la silueta.

    El alfa del cuadro es el recorte del presentador y no se toca: la boca esta
    *dentro* de la silueta, y tocar el alfa ahi abriria un agujero por el que el
    video de fondo apareceria en mitad del rostro.
    """
    h, w = cuadro.shape[:2]
    px = np.asarray(pieza, dtype=np.float32)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + pieza.width), min(h, y + pieza.height)
    if x1 <= x0 or y1 <= y0:
        return
    sub = px[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (sub[:, :, 3:4] / 255.0)
    objetivo = cuadro[y0:y1, x0:x1, :3].astype(np.float32)
    cuadro[y0:y1, x0:x1, :3] = (objetivo + (sub[:, :, :3] - objetivo) * a).astype(np.uint8)


# ===========================================================================
# el animador
# ===========================================================================


class AnimadorVideo:
    """Monta cada cuadro: el clip base con boca, puesto en la escenificacion."""

    def __init__(self, base: Base, cuadros: Cuadros, accent: str, nombre: str = "",
                 area: tuple[int, int, int, int] = AREA) -> None:
        self.base = base
        self.cuadros = cuadros
        self.area = area
        self.tonos = medir_tonos(cuadros[base.frames // 2],
                                 base.puntos(base.frames // 2))
        self._chip = chip(nombre, accent) if nombre else None

    def cuadro(self, t: float, abertura: float, anchura: float,
               poses: list[Pose]) -> Image.Image:
        ax, ay, aw, ah = self.area
        lienzo = Image.new("RGBA", (aw, ah), (0, 0, 0, 0))
        p = pose(poses, t)
        if p.alfa <= 0.004 or p.altura <= 1:
            return lienzo

        k = indice_vaiven(t, self.base)
        cuadro = self.cuadros[k].copy()
        hablar(cuadro, self.base.puntos(k), self.base.contorno(k),
               self.base.face_height, abertura, anchura, self.tonos)

        bw, bh = self.base.size
        escala = p.altura / bh
        entrada = (1 - p.alfa) * 90        # desliza desde fuera mientras aparece
        # La base del recorte toca p.base y el centro horizontal en p.cx.
        objetivo_x = p.cx - ax - entrada
        objetivo_y = p.base - ay
        ox = int(math.floor(objetivo_x - bw * escala / 2)) - 2
        oy = int(math.floor(objetivo_y - bh * escala)) - 2
        ow = int(math.ceil(bw * escala)) + 5
        oh = int(math.ceil(bh * escala)) + 5

        img = Image.fromarray(cuadro)
        girada = _afin(img, (ow, oh), escala=escala, grados=0.0,
                       pivote=(bw / 2, bh), objetivo=(objetivo_x - ox, objetivo_y - oy))
        if p.alfa < 0.999:
            girada.putalpha(girada.getchannel("A").point(
                lambda v, a=p.alfa: int(v * a)))
        _pegar(lienzo, girada, ox, oy)
        self._chip_en(lienzo, p, objetivo_x, objetivo_y)
        return lienzo

    def _chip_en(self, lienzo: Image.Image, p: Pose, base_x: float,
                 base_y: float) -> None:
        """Solo mientras esta grande, donde se presenta."""
        from agent.render.presenter import ALTURA_CIERRE, ALTURA_LLAMADA

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


def modular(abertura_visema: np.ndarray, energia: np.ndarray) -> np.ndarray:
    """Forma del fonema x fuerza del audio.

    La letra sabe *que* boca hacer, el audio sabe *con cuanta fuerza*. Sola, la
    pista de visemas recita la frase entera a la misma intensidad; sola, la
    envolvente no sabe cerrar el labio en la /m/. El suelo existe porque el
    fonema sordo tiene poca energia y la boca lo hace igual.
    """
    n = min(len(abertura_visema), len(energia))
    ganancia = ENERGIA_PISO + (1.0 - ENERGIA_PISO) * energia[:n]
    return np.clip(abertura_visema[:n] * ganancia, 0.0, 1.0)


def render_capa(base: Base, audio: Path, out: Path, *, duracion: float,
                tiempos: Tiempos, accent: str, palabras_habladas,
                nombre: str = "", fps: int = FPS, ffmpeg: str = "ffmpeg",
                trabajo: Path | None = None) -> Capa:
    """Genera la capa del presentador a partir del clip base y devuelve donde posar."""
    n = max(1, int(round(duracion * fps)))
    energia = envolvente(audio, n, fps, ffmpeg=ffmpeg)
    abertura, anchura = pista(palabras_habladas, n, fps)
    abertura = modular(abertura, energia)
    esc = plan(tiempos)

    cuadros = Cuadros(base, trabajo or out.parent, ffmpeg=ffmpeg)
    anim = AnimadorVideo(base, cuadros, accent, nombre)
    ax, ay, aw, ah = anim.area

    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(encode_command(out, aw, ah, fps, ffmpeg=ffmpeg),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for k in range(n):
            t = k / fps
            img = anim.cuadro(t, float(abertura[k]), float(anchura[k]), esc.poses)
            proc.stdin.write(img.tobytes())
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


__all__ = ["AnimadorVideo", "Base", "DIENTE_MIN", "MAXILAR", "MARGEN_HUECO",
           "Cuadros", "Tonos", "abrir_maxilar", "cargar_base", "extender_labios",
           "campo_maxilar", "hablar", "hueco_boca", "indice_vaiven", "medir_tonos",
           "modular", "render_capa"]
