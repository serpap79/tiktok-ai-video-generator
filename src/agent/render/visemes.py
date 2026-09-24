"""Boca por visema, no por volumen: forma de la boca sobre el tiempo del TTS.

El presentador anterior abria la boca por la envolvente de energia de la
narracion. En una silueta oscura eso pasaba -- el limite estaba declarado y la
 decision de pulir amplitud primero fue del autor. En un rostro fotorrealista
no pasa: la energia no distingue "mama" de "papa", y la boca queda **abierta
en la /m/ y en la /b/**, que es el gatillo clasico del valle inquietante.
Quien lee labios sin saber que lee labios es todo el mundo; el error no
necesita nombre para incomodar.

Aqui la forma de la boca viene de la letra y el tamano del audio:

- **forma** -- cada palabra hablada se convierte en visemas (abertura +
  anchura) por reglas de grafema castellano, y los visemas se reparten en la
  duracion que edge-tts devolvio para esa palabra. Determinista y $0: no hay
  modelo, no hay llamada, el mismo texto da siempre la misma boca.
- **tamano** -- la envolvente RMS modula la abertura. Sin ella la boca recita
  la frase entera a la misma intensidad; con ella sola, la boca no sabe
  cerrarse. Una gobierna a la otra: `presenter_video` multiplica las dos.

El visema es del texto **hablado** (el respelling de `voice/pronounce.py`, que
escribe "Yemini"), nunca del texto de la leyenda -- es la voz la que la boca
acompana, y la voz dice "Yemini".

Donde esto para, declarado: son visemas de grafema, sin diccionario de
excepcion ni silaba tonica de verdad. "Excepto" o "subrayar" salen
aproximados. La diferencia que importa en un feed vertical -- labio pegado en
la consonante cerrada, boca redonda en la /o/, boca estirada en la /i/ -- esa
sale bien.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------- visemas
# (abertura 0..1, anchura -1 redondeada .. +1 estirada, peso de duracion).
# La abertura es relativa: 1,0 es la silaba mas abierta que la boca hace, y
# quien lo convierte en pixel es el `presenter_video` con la altura del rostro
# medida.


@dataclass(frozen=True)
class Visema:
    abertura: float
    anchura: float
    peso: float
    cerrado: bool = False   # bilabial: labio pegado, cueste lo que cueste


REPOSO = Visema(0.0, 0.0, 1.0)

VISEMAS: dict[str, Visema] = {
    # vocales orales
    "a": Visema(1.00, 0.15, 1.00),
    "e": Visema(0.62, 0.45, 1.00),
    "i": Visema(0.34, 0.85, 0.90),
    "o": Visema(0.62, -0.55, 1.00),
    "u": Visema(0.30, -0.95, 0.90),
    # vocales nasales: la misma boca, un poco menos abierta (el aire sale por la nariz)
    "a~": Visema(0.80, 0.10, 1.05),
    "e~": Visema(0.50, 0.35, 1.05),
    "i~": Visema(0.28, 0.70, 0.95),
    "o~": Visema(0.50, -0.50, 1.05),
    "u~": Visema(0.26, -0.85, 0.95),
    # consonantes
    "P": Visema(0.00, 0.00, 0.42, cerrado=True),   # p, b, m
    "F": Visema(0.16, 0.25, 0.75),                 # f, v
    "T": Visema(0.28, 0.10, 0.45),                 # t, d, n, l, r simple
    "s": Visema(0.18, 0.50, 0.75),                 # s, z, c cedilla
    "S": Visema(0.30, -0.60, 0.80),                # ch, j, g antes de e/i
    "K": Visema(0.35, 0.05, 0.45),                 # c, g, qu, r fuerte, h
    "N": Visema(0.24, 0.10, 0.55),                 # ñ
    "L": Visema(0.30, 0.15, 0.55),                 # ll
    ".": REPOSO,                                  # silencio
}

VOCALES = set("aeiouáéíóúâêôàãõ")
# Vocal anterior, con o sin acento: es lo que ablanda `c` y `g` ("voz" y
# "gente" llevan /s/ y /x/, no /k/). Comprobar contra la string "ei" cruda
# dejaba "vez" y "inteligencia" con la [k] dura -- y la boca de la [k] es
# visiblemente mas abierta que la de la [s].
_ANTERIOR = frozenset("eiéêíî")
_ACENTO_ABRE = {"á": "a", "à": "a", "â": "a", "é": "e", "ê": "e",
                "í": "i", "ó": "o", "ô": "o", "ú": "u"}
_NASAL = {"ã": "a~", "õ": "o~"}


def _limpa(palabra: str) -> str:
    """Minusculas, sin puntuacion, acento conservado (cambia el fonema)."""
    return re.sub(r"[^a-záéíóúâêôàãõçñ]", "", palabra.lower())


def fonemas(palabra: str) -> list[str]:
    """Grafema -> visema, al nivel que la boca ve.

    Reglas que valen la pena, porque son las que cambian la forma del labio:

    - `m`/`n` **antes de consonante o al final** nasalizan la vocal y NO pegan
      el labio ("dar" no cierra la boca; "can**t**ar" tampoco, la nasal va con
      la vocal). Tratar todo `m` como bilabial ponia un labio pegado donde el
      audio no tiene ninguno -- error peor que el que este modulo vino a
      corregir. "Yo**m**" cierra: ahi el m cierra la silaba.
    - las consonantes que el castellano comparte con pt-BR mantienen su boca;
      la diferencia castellana (la /x/ de "gente" en vez de la /S/)
      comparte labio, y por eso la forma sigue siendo correcta.
    - `l` **en final de silaba** se vuelve semivocal [u] ("sal" -> "sau"):
      labio redondeado, no lengua en el diente.
    """
    p = _limpa(palabra)
    if not p:
        return []
    salida: list[str] = []
    i, n = 0, len(p)
    while i < n:
        c = p[i]
        sig = p[i + 1] if i + 1 < n else ""
        luego = p[i + 2] if i + 2 < n else ""

        # --- digrafos
        if c == "c" and sig == "h":
            salida.append("S")
            i += 2
            continue
        if c == "l" and sig == "h":
            salida.append("L")
            i += 2
            continue
        if c == "n" and sig == "h":
            salida.append("N")
            i += 2
            continue
        if c == "r" and sig == "r":
            salida.append("K")
            i += 2
            continue
        if c == "s" and sig == "s":
            salida.append("s")
            i += 2
            continue
        if c in "sx" and sig == "c" and luego in _ANTERIOR:
            salida.append("s")
            i += 2
            continue
        if c == "q" and sig == "u":
            salida.append("K")
            if luego in ("a", "o", "á", "ó", "ô"):   # "cuatro": sobra el [w]
                salida.append("u")
            i += 2
            continue
        if c == "g" and sig == "u" and luego in _ANTERIOR:
            salida.append("K")
            i += 2
            continue

        # --- vocales
        if c in VOCALES:
            base = _NASAL.get(c) or _ACENTO_ABRE.get(c, c)
            # vocal + m/n cerrando silaba = vocal nasal, y el m/n se consume
            if sig in ("m", "n") and luego not in VOCALES:
                base = base.rstrip("~") + "~"
                i += 1                    # consume tambien el m/n
            elif c not in _NASAL and _fim_atono(p, i, c):
                base = "i" if base == "e" else "u"
            salida.append(base)
            i += 1
            continue

        # --- consonantes
        if c in "pb":
            salida.append("P")
        elif c == "m":
            # solo llega aqui el m que la vocal NO consumio como nasal: bilabial.
            salida.append("P")
        elif c in "fv":
            salida.append("F")
        elif c in "tdn":
            salida.append("T")
        elif c == "l":
            # final de silaba (sin vocal a continuacion) se vuelve [u] velarizado
            salida.append("T" if sig in VOCALES else "u")
        elif c == "r":
            # inicio de palabra o tras n/l/s = r fuerte; entre vocales = tap
            fuerte = i == 0 or (i > 0 and p[i - 1] in "nls")
            salida.append("K" if fuerte else "T")
        elif c in "zç":
            salida.append("s")
        elif c == "s":
            # entre vocales el sonido es [z], pero el labio hace lo mismo en ambos.
            salida.append("s")
        elif c == "c":
            salida.append("s" if sig in _ANTERIOR else "K")
        elif c == "g":
            salida.append("S" if sig in _ANTERIOR else "K")
        elif c in "jx":
            salida.append("S")
        elif c in "kw":
            salida.append("K")
        elif c == "y":
            salida.append("i")
        elif c == "ñ":
            salida.append("N")
        # la 'h' muda cae fuera sola
        i += 1
    return salida


def _fim_atono(p: str, i: int, c: str) -> bool:
    """`e`/`o` sin acento en la ultima letra de la palabra (se reduce a [i]/[u])."""
    return c in "eo" and i == len(p) - 1 and len(p) > 2


def visemas(palabra: str) -> list[Visema]:
    return [VISEMAS.get(f, VISEMAS["T"]) for f in fonemas(palabra)]


# ------------------------------------------------------------------ pista
# Muestras por cuadro en el calculo de la dinamica. No es capricho: es la misma
# leccion ya pagada en la envolvente del presentador quieto -- una constante de
# tiempo de 40 ms es MAS CORTA que el cuadro de 33,3 ms, asi que correr el
# filtro a la tasa de cuadro no filtra nada. A 4x (120 Hz) vale 5 muestras y de
# verdad suaviza; solo despues la curva baja al cuadro, por media.
SUB = 4
# Constante de tiempo de la mandibula y del labio. Son distintas a proposito:
# la mandibula es hueso con masa y llega despacio; el labio (estirar en la /i/,
# redondear en la /u/) es ligero y llega antes. Usar un numero solo para los
# dos dejaba la boca entera con la misma cadencia, que es la mitad de lo que
# delata marioneta.
TAU_MAXILAR = 0.035
TAU_LABIO = 0.024
# La consonante cerrada tiene que LLEGAR a cero, si no no se lee como labio
# pegado. Se reimpone a la tasa de SUB (no a la de cuadro) y con ventana de
# coseno elevado: a 30 fps una "V" de un cuadro solo se leia como parpadeo de
# la boca, un defecto en el lugar del otro. En sub-cuadro el cierre es
# continuo, y la media del cuadro se convierte sola en el borron de un gesto
# rapido -- que es lo que haria la camara.
CIERRE_S = 0.055
# Hueco entre palabras que ya cuenta como pausa: por debajo de eso la boca pasa
# de una palabra a la otra sin pasar por el reposo, que es como se habla.
PAUSA_S = 0.12


def _inercia(objetivo: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Respuesta criticamente amortiguada de `objetivo`: boca con masa.

    El interpolador anterior sostenia el visema quieto y saltaba al siguiente
    en 70 ms. Medido en narracion real de 16,5 s: salto medio de 0,109 entre
    cuadros y 0,381 en el p95 -- la boca recorría 38% del curso en 33 ms. Eso
    se lee como chasquido, no como habla, y ningun ajuste de forma del visema
    lo arregla, porque el defecto esta en la DINAMICA y no en la pose.

    Masa-muelle criticamente amortiguada llega rapido, no oscila, y sobre todo
    **no alcanza el objetivo cuando el objetivo cambia deprisa**. Esa falta de
    alcance es la coarticulacion: en el habla corrida nadie articula cada
    fonema por entero, y era exactamente eso lo que la escalera hacia --
    pronunciaba todo con la misma perfeccion, que es el modo mas rapido de
    sonar a robot.
    """
    y = np.empty_like(objetivo)
    pos = float(objetivo[0])
    vel = 0.0
    k = 1.0 / (tau * tau)
    c = 2.0 / tau
    for i in range(len(objetivo)):
        vel += dt * (k * (float(objetivo[i]) - pos) - c * vel)
        pos += dt * vel
        y[i] = pos
    return y


def _escalones(objetivos: list[tuple[float, float, Visema]], campo: str,
               t: np.ndarray) -> np.ndarray:
    """Objetivo constante por visema: cada uno vale su DURACION, no un punto.

    Sostener el visema solo en el centro del fonema e interpolar entre centros
    daba una rampa permanente, sin escalon ninguno; con el escalon, quien decide
    si el fonema se alcanza o no pasa a ser la inercia -- que es quien decide en
    una boca de verdad.
    """
    fuera = np.zeros(len(t), dtype=np.float32)
    for ini, fin, v in objetivos:
        fuera[(t >= ini) & (t < fin)] = getattr(v, campo)
    return fuera


def pista(palabras, n: int, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """Abertura y anchura de la boca en cada uno de los `n` cuadros.

    `palabras` son los tiempos del TTS (`voice/edge.WordTiming`) del texto
    HABLADO. Entre dos palabras la boca vuelve al reposo si la pausa pasa de
    `PAUSA_S` -- es lo que separa palabra de palabra visualmente.
    """
    if n <= 0:
        return (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))

    # --- 1. lo que la boca deberia hacer, visema a visema, con duracion
    objetivos: list[tuple[float, float, Visema]] = []
    cierres: list[tuple[float, float]] = []
    fin_anterior = 0.0
    for w in palabras:
        ini = float(getattr(w, "start_s", getattr(w, "start", 0.0)))
        fin = float(getattr(w, "end_s", getattr(w, "end", 0.0)))
        vs = visemas(getattr(w, "text", ""))
        if not vs or fin <= ini:
            continue
        # Descanso al FINAL de la palabra anterior, y no solo un poco antes de
        # la siguiente: con el descanso solo a la entrada, el ultimo visema de
        # la palabra valia hasta que la palabra siguiente empezaba -- en una
        # pausa de 2,4 s entre frases la boca quedaba escancarada en la /a/
        # final todo el tiempo.
        if fin_anterior > 0 and ini - fin_anterior > PAUSA_S:
            objetivos.append((fin_anterior + 0.04, ini - 0.04, REPOSO))
        total = sum(v.peso for v in vs)
        t = ini
        for v in vs:
            dur = (fin - ini) * v.peso / total
            objetivos.append((t, t + dur, v))
            if v.cerrado:
                cierres.append((t, t + dur))
            t += dur
        fin_anterior = fin
    if not objetivos:
        return (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))

    # --- 2. la dinamica, en sub-cuadro
    m = n * SUB
    dt = 1.0 / (fps * SUB)
    t_sub = np.arange(m, dtype=np.float32) * dt
    abertura = _inercia(_escalones(objetivos, "abertura", t_sub), dt, TAU_MAXILAR)
    anchura = _inercia(_escalones(objetivos, "anchura", t_sub), dt, TAU_LABIO)

    # --- 3. el labio pegado, reimpuesto despues de la dinamica
    # Entre dos vocales abiertas la inercia nunca llega a cero en una /p/ de
    # 40 ms: sobraria un respiro de abertura justo donde el espectador comprueba
    # el labio sin saber que lo comprueba.
    for ini, fin in cierres:
        centro = (ini + fin) / 2
        k0 = max(0, int((centro - CIERRE_S) / dt))
        k1 = min(m, int((centro + CIERRE_S) / dt) + 1)
        if k1 <= k0:
            continue
        d = (t_sub[k0:k1] - centro) / CIERRE_S
        # coseno elevado: vale 0 en el centro y vuelve a 1 en las puntas, sin
        # esquina
        abertura[k0:k1] *= (1 - np.cos(np.pi * np.clip(np.abs(d), 0, 1))) / 2

    # --- 4. de vuelta al cuadro, por media (el borron de quien filma)
    abertura = np.clip(abertura, 0.0, 1.0).reshape(n, SUB).mean(axis=1)
    anchura = np.clip(anchura, -1.0, 1.0).reshape(n, SUB).mean(axis=1)
    return abertura.astype(np.float32), anchura.astype(np.float32)


def texto_hablado(respelled) -> str:
    """El texto que la voz dice, para verificacion en test y en el log."""
    return " ".join(getattr(respelled, "spoken", []) or [])


def _sem_acento(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    return "".join(c for c in d if not unicodedata.combining(c))


__all__ = ["CIERRE_S", "PAUSA_S", "REPOSO", "SUB", "TAU_LABIO", "TAU_MAXILAR",
           "VISEMAS", "Visema",
           "fonemas", "pista", "texto_hablado", "visemas"]
