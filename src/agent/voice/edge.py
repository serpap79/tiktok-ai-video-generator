"""Narracao pelo edge-tts, com o tempo de cada palavra.

O edge-tts usa o endpoint de leitura em voz alta do Edge: gratis, sem chave,
vozes neurais pt-BR (Antonio, Francisca, Thalita). Sem contrato -- pode
quebrar --, por isso a narracao tem reserva local (`local_fallback`, Piper
em CPU, `voice/engine.py`).

O que importa aqui alem do audio sao os eventos `WordBoundary`: o instante e
a duracao de cada palavra falada. E deles que sai a legenda palavra a
palavra -- sem Whisper, sem estimativa.

Licenca: edge-tts e LGPLv3, usado como biblioteca sem modificacao.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

# Vozes pt-BR do endpoint gratuito (conferido com `edge-tts --list-voices` em
# 20/09/2026). As duas monolingues sao o padrao: a Multilingual troca o
# sotaque no meio da frase ("Discorda?" saiu com sotaque de outra lingua no
# primeiro video do piloto).
VOZ_FEMININA = "pt-BR-FranciscaNeural"
VOZ_MASCULINA = "pt-BR-AntonioNeural"
VOZ_MULTILINGUE = "pt-BR-ThalitaMultilingualNeural"

TICKS_POR_S = 10_000_000

# Velocidade por voz para as duas falarem no MESMO ritmo, medido em 20/09/2026
# no roteiro do Gemini (178 palavras): Francisca +4% = 2,78 palavras/s,
# Antonio +4% = 2,57. Com o ritmo desigual, o roteiro longo de 150 palavras
# saia com 54s na voz feminina -- abaixo dos 60s do Creator Rewards. Os dois
# em ~2,57 palavras/s, que e o `WORDS_PER_SECOND` do roteirista.
RATE_POR_VOZ = {VOZ_FEMININA: "-4%", VOZ_MASCULINA: "+4%", VOZ_MULTILINGUE: "+0%"}


@dataclass(frozen=True)
class WordTiming:
    text: str
    start_s: float
    end_s: float


class TTSUnavailable(RuntimeError):
    """O endpoint nao respondeu ou devolveu audio vazio."""


async def _sintetizar(texto: str, voz: str, destino: Path, rate: str,
                      pitch: str) -> list[WordTiming]:
    import edge_tts

    comunicacao = edge_tts.Communicate(texto, voz, rate=rate, pitch=pitch,
                                       boundary="WordBoundary")
    tempos: list[WordTiming] = []
    with destino.open("wb") as fh:
        async for pedaco in comunicacao.stream():
            if pedaco["type"] == "audio":
                fh.write(pedaco["data"])
            elif pedaco["type"] == "WordBoundary":
                inicio = pedaco["offset"] / TICKS_POR_S
                tempos.append(WordTiming(pedaco["text"], inicio,
                                         inicio + pedaco["duration"] / TICKS_POR_S))
    return tempos


def synthesize(texto: str, destino: Path, *, voice: str = VOZ_FEMININA,
               rate: str | None = None, pitch: str = "+0Hz",
               retries: int = 2) -> list[WordTiming]:
    """Grava o MP3 em `destino` e devolve os tempos das palavras faladas.

    `rate` padrao vem de RATE_POR_VOZ (mesmo ritmo nas duas vozes). Duas
    novas tentativas: o endpoint oscila.
    """
    rate = rate or RATE_POR_VOZ.get(voice, "+0%")
    destino.parent.mkdir(parents=True, exist_ok=True)
    ultimo: Exception | None = None
    for _ in range(retries + 1):
        try:
            tempos = asyncio.run(_sintetizar(texto, voice, destino, rate, pitch))
        except Exception as exc:  # noqa: BLE001 -- rede, websocket, protocolo
            ultimo = exc
            continue
        if destino.exists() and destino.stat().st_size > 1000 and tempos:
            return tempos
        ultimo = TTSUnavailable("audio vazio ou sem tempos de palavra")
    raise TTSUnavailable(f"edge-tts indisponivel com {voice}: {ultimo}")


def local_fallback(texto: str, destino: Path, *, voice_model: Path,
                   speed: float = 1.0) -> list[WordTiming]:
    """Reserva offline: Piper em CPU, tempos por palavra estimados por frase.

    O Piper nao informa limite de palavra; dentro de cada frase o tempo e
    dividido proporcionalmente ao tamanho das palavras -- bom o bastante
    para legenda, e o video sai mesmo com o edge-tts fora do ar.
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


__all__ = ["RATE_POR_VOZ", "TTSUnavailable", "VOZ_FEMININA", "VOZ_MASCULINA",
           "VOZ_MULTILINGUE", "WordTiming", "local_fallback", "synthesize"]
