"""Narración con edge-tts, incluido el tiempo de cada palabra.

edge-tts usa el endpoint de lectura en voz alta de Edge: es gratuito, no requiere
clave y ofrece voces neuronales es-ES (Álvaro, Elvira y Ximena). Al no tener
contrato, puede dejar de funcionar; por eso la narración dispone de una reserva
local (`local_fallback`, Piper en CPU, `voice/engine.py`).

Además del audio, los eventos `WordBoundary` son importantes: indican el inicio
y la duración de cada palabra hablada. De ellos se genera la leyenda palabra a
palabra, sin Whisper ni estimaciones.

Licencia: edge-tts es LGPLv3 y se usa como biblioteca sin modificaciones.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

# Voces es-ES del endpoint gratuito (comprobadas con `edge-tts --list-voices`).
# Las dos monolingües son el estándar: la multilingüe cambia el acento a mitad de frase.
VOZ_FEMENINA = "es-ES-ElviraNeural"
VOZ_MASCULINA = "es-ES-AlvaroNeural"
VOZ_MULTILINGUE = "es-ES-XimenaMultilingualNeural"

TICKS_POR_S = 10_000_000

# Velocidad por voz para que todas hablen al mismo ritmo. Referencia: las voces
# es-ES rondan las 2,5-2,7 palabras/s; se recalibra con la primera medida real.
# Con ritmos desiguales, un guion largo de 150 palabras quedaría por debajo de los
# 60 s de Creator Rewards. Las tres apuntan a ~2,57 palabras/s, el valor de
# `WORDS_PER_SECOND` del guionista.
RATE_POR_VOZ = {VOZ_FEMENINA: "+0%", VOZ_MASCULINA: "+0%", VOZ_MULTILINGUE: "+0%"}


@dataclass(frozen=True)
class WordTiming:
    text: str
    start_s: float
    end_s: float


class TTSUnavailable(RuntimeError):
    """El endpoint no respondió o devolvió audio vacío."""


async def _sintetizar(texto: str, voz: str, destino: Path, rate: str,
                      pitch: str) -> list[WordTiming]:
    import edge_tts

    comunicacion_tts = edge_tts.Communicate(
        texto,
        voz,
        rate=rate,
        pitch=pitch,
        boundary="WordBoundary",
    )
    tempos: list[WordTiming] = []
    with destino.open("wb") as fh:
        async for trozo in comunicacion_tts.stream():
            if trozo["type"] == "audio":
                fh.write(trozo["data"])
            elif trozo["type"] == "WordBoundary":
                inicio = trozo["offset"] / TICKS_POR_S
                tempos.append(
                    WordTiming(
                        trozo["text"],
                        inicio,
                        inicio + trozo["duration"] / TICKS_POR_S,
                    )
                )
    return tempos


def synthesize(texto: str, destino: Path, *, voice: str = VOZ_FEMENINA,
               rate: str | None = None, pitch: str = "+0Hz",
               retries: int = 2) -> list[WordTiming]:
    """Graba el MP3 en `destino` y devuelve los tiempos de las palabras habladas.

    `rate` procede por defecto de RATE_POR_VOZ para mantener el ritmo de las voces.
    Se admiten dos reintentos porque el endpoint es intermitente.
    """
    rate = rate or RATE_POR_VOZ.get(voice, "+0%")
    destino.parent.mkdir(parents=True, exist_ok=True)
    ultimo: Exception | None = None
    for _ in range(retries + 1):
        try:
            tempos = asyncio.run(_sintetizar(texto, voice, destino, rate, pitch))
        except Exception as exc:  # noqa: BLE001 -- red, websocket o protocolo
            ultimo = exc
            continue
        if destino.exists() and destino.stat().st_size > 1000 and tempos:
            return tempos
        ultimo = TTSUnavailable("audio vacío o sin tiempos de palabra")
    raise TTSUnavailable(f"edge-tts no disponible con {voice}: {ultimo}")


def local_fallback(texto: str, destino: Path, *, voice_model: Path,
                   speed: float = 1.0) -> list[WordTiming]:
    """Reserva sin conexión: Piper en CPU y tiempos por palabra estimados por frase.

    Piper no informa tiempos por palabra. Dentro de cada frase, el tiempo se reparte
    de forma proporcional a la longitud de las palabras; es suficiente para la
    leyenda y permite terminar el vídeo aunque edge-tts no esté disponible.
    """
    import re
    import wave

    from agent.voice.engine import PiperTTS

    motor = PiperTTS(voice_model)
    frases = [f for f in re.split(r"(?<=[.!?…])\s+|\n+", texto) if f.strip()]
    pcm = bytearray()
    taxa = 22050
    tempos: list[WordTiming] = []
    cursor = 0.0
    for frase in frases:
        fala = motor.speak(frase, speed=speed)
        taxa = fala.sample_rate
        dur = fala.duration_s
        palavras = frase.split()
        pesos = [max(1, len(p)) for p in palavras]
        total = sum(pesos) or 1
        t = cursor
        for p, w in zip(palavras, pesos, strict=True):
            d = dur * w / total
            tempos.append(WordTiming(p, t, t + d))
            t += d
        pcm += fala.pcm16 + b"\x00" * int(taxa * 0.25) * 2
        cursor += dur + 0.25
    wav = destino.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(taxa)
        w.writeframes(bytes(pcm))
    return tempos


__all__ = ["RATE_POR_VOZ", "TTSUnavailable", "VOZ_FEMENINA", "VOZ_MASCULINA",
           "VOZ_MULTILINGUE", "VOICE_FEMENINA", "WordTiming", "local_fallback",
           "synthesize"]

# Alias con otro nombre: algunos imports del nombre original fallan de forma
# intermitente en el import machinery del renderizador; el alias evita la colisión.
VOICE_FEMENINA = VOZ_FEMENINA
