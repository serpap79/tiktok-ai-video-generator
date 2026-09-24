"""Leyenda karaoke en ASS: grafia original, tiempo de la voz, palabra activa destacada.

La voz dice "Djemini"; la pantalla muestra "Gemini". El tiempo de cada palabra
viene de los eventos WordBoundary del TTS, que hablan de la grafia HABLADA --
`align` casa esos eventos con las palabras habladas y devuelve el tiempo a las
palabras originales por el alineamiento de `voice/pronounce.py`.

Estilo: bloque de hasta 3 palabras (una sola por pantalla era el MPT: "coma",
"punto" sueltos en medio del cuadro, sin contexto), la palabra que se esta
diciendo en el acento de la marca, fuente de la marca (Space Grotesk 700),
contorno grueso para leer sobre cualquier clip, al 60% de la altura -- debajo
de la tarjeta del gancho, por encima de la interfaz de TikTok.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from agent.voice.edge import WordTiming
from agent.voice.pronounce import Respelled

MAX_PALABRAS = 3
MAX_CARACTERES = 22
Y_LEYENDA = 1150
TAMANO = 84


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


def _norm(texto: str) -> str:
    sin = unicodedata.normalize("NFD", texto.lower())
    sin = "".join(c for c in sin if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", sin)


def align(respelled: Respelled, timings: list[WordTiming]) -> list[TimedWord]:
    """Tiempo de cada palabra ORIGINAL a partir de los limites del habla."""
    hablado = respelled.spoken
    dueno: list[int] = []
    for i, n in enumerate(respelled.groups):
        dueno.extend([i] * n)
    inicio: list[float | None] = [None] * len(hablado)
    fin: list[float | None] = [None] * len(hablado)

    j = 0
    for t in timings:
        objetivo = _norm(t.text)
        if not objetivo:
            continue
        for k in range(j, min(j + 8, len(hablado))):
            fk = _norm(hablado[k])
            if fk and (fk == objetivo or fk.startswith(objetivo) or objetivo.startswith(fk)):
                inicio[k] = t.start_s if inicio[k] is None else inicio[k]
                fin[k] = t.end_s
                j = k + 1
                break

    _rellenar(inicio, fin)
    palabras: list[TimedWord] = []
    for i, original in enumerate(respelled.original):
        ks = [k for k, d in enumerate(dueno) if d == i]
        if ks:
            s = min(inicio[k] for k in ks if inicio[k] is not None)
            e = max(fin[k] for k in ks if fin[k] is not None)
        elif palabras:
            # Parte de nombre compuesto ("Street" en "Wall Street Journal"): se
            # enciende junto con la primera palabra del nombre.
            s, e = palabras[-1].start, palabras[-1].end
        else:
            s = e = 0.0
        palabras.append(TimedWord(original, s, e))
    return palabras


def _rellenar(inicio: list[float | None], fin: list[float | None]) -> None:
    """Palabra hablada sin evento de limite gana el tiempo entre las vecinas."""
    n = len(inicio)
    for k in range(n):
        if inicio[k] is not None:
            continue
        ant = next((fin[x] for x in range(k - 1, -1, -1) if fin[x] is not None), 0.0)
        prox = next((inicio[x] for x in range(k + 1, n) if inicio[x] is not None), None)
        prox = prox if prox is not None else ant + 0.3
        inicio[k] = ant
        fin[k] = max(ant, prox)


def chunks(palabras: list[TimedWord]) -> list[list[TimedWord]]:
    """Bloques de hasta 3 palabras, cortando en puntuacion."""
    bloques: list[list[TimedWord]] = []
    actual: list[TimedWord] = []
    for p in palabras:
        texto = " ".join(w.text for w in [*actual, p])
        if actual and (len(actual) >= MAX_PALABRAS or len(texto) > MAX_CARACTERES):
            bloques.append(actual)
            actual = []
        actual.append(p)
        if re.search(r"[.!?;:,…]$", p.text):
            bloques.append(actual)
            actual = []
    if actual:
        bloques.append(actual)
    return bloques


def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _t(segundos: float) -> str:
    segundos = max(0.0, segundos)
    h = int(segundos // 3600)
    m = int(segundos % 3600 // 60)
    s = segundos % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escapar(texto: str) -> str:
    return texto.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def build_ass(palabras: list[TimedWord], destino: Path, *, accent: str,
              font_family: str, duration: float, y: int = Y_LEYENDA,
              start_at: float = 0.0) -> Path:
    """Un evento por palabra activa: el bloque entero en pantalla, la actual en color.

    `y` y `start_at` vienen de la escenificacion del presentador cuando hay uno:
    mientras el esta grande en la llamada, quien escribe el gancho es la
    tarjeta -- la leyenda entra despues, mas arriba, para pasar por encima de
    la cabeza de el en la esquina. Sin esto el mismo texto apareceria escrito
    dos veces, y la segunda sobre el rostro.
    """
    blanco = _ass_color("#FFFFFF")
    destaque = _ass_color(accent)
    cabecera = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
        " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
        " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Fala,{font_family},{TAMANO},{blanco},{blanco},&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,6,2,5,60,60,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lineas: list[str] = []
    bloques = chunks(palabras)
    for b, bloque in enumerate(bloques):
        fin_bloque = (bloques[b + 1][0].start if b + 1 < len(bloques)
                      else min(duration, bloque[-1].end + 0.4))
        if fin_bloque <= start_at:
            continue
        for i, palabra in enumerate(bloque):
            ini = max(palabra.start, start_at)
            fin = bloque[i + 1].start if i + 1 < len(bloque) else fin_bloque
            if fin <= ini:
                fin = ini + 0.05
            partes = []
            for k, w in enumerate(bloque):
                color = destaque if k == i else blanco
                partes.append(f"{{\\c{color}}}{_escapar(w.text)}")
            texto = f"{{\\an5\\pos(540,{y})}}" + " ".join(partes)
            lineas.append(f"Dialogue: 0,{_t(ini)},{_t(fin)},Fala,,0,0,0,,{texto}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(cabecera + "\n".join(lineas) + "\n", encoding="utf-8")
    return destino


__all__ = ["TimedWord", "align", "build_ass", "chunks"]
