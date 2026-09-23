"""Roteiro para wav: frase a frase, com pausa e enfase marcadas no texto.

Marcacao (opcional, some antes da sintese):
- `[PAUSA CURTA]` 0,3s · `[PAUSA MEDIA]` 0,6s · `[PAUSA LONGA]` 1,0s
- `*palavra*` poe micro-pausa (0,12s) antes e depois -- aproximacao honesta
  de enfase: o Piper nao tem SSML, e a pausa e o que o ouvido percebe como
  destaque sem mudar o timbre.

Pontuacao vira pausa sozinha: ponto final e quebra de linha respiram pelo
`pausa_frase_s` do estilo. Variacao entre geracoes vem do proprio VITS
(amostragem com noise_scale): o mesmo roteiro nunca sai byte-identico, sem
parametro de seed porque o motor nao expoe um.
"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass

from agent.voice.engine import TTS, Utterance

PAUSAS = {
    "pausa curta": 0.3,
    "pausa media": 0.6,
    "pausa longa": 1.0,
}
_MICRO_PAUSA = 0.12
_MARCADOR = re.compile(r"\[(pausa curta|pausa media|pausa longa)\]", re.IGNORECASE)
_ENFASE = re.compile(r"\*([^*]{1,60})\*")
_FRASE = re.compile(r"(?<=[.!?…])\s+|\n+")


@dataclass
class Narration:
    sample_rate: int
    pcm16: bytes
    utterances: int = 0
    pauses_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return len(self.pcm16) / 2 / self.sample_rate if self.sample_rate else 0.0

    def write_wav(self, path: str) -> str:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.pcm16)
        return path


def _silencio(rate: int, segundos: float) -> bytes:
    return b"\x00" * int(rate * segundos) * 2


@dataclass
class Narrator:
    tts: TTS
    sample_rate: int = 22050
    pausa_frase_s: float = 0.3

    def narrate(self, text: str, *, speed: float = 1.0,
                noise: float = 0.667) -> Narration:
        """Texto (ou narracao hook+body+closing) para audio concatenado."""
        blocos = _expandir_marcadores(text)
        pcm = bytearray()
        falas = 0
        pausas = 0.0
        for kind, valor in blocos:
            if kind == "pausa":
                pcm += _silencio(self.sample_rate, valor)
                pausas += valor
            else:
                for frase in [f for f in _FRASE.split(valor) if f.strip()]:
                    u = self._tts_speak(frase, speed, noise)
                    pcm += u.pcm16
                    pcm += _silencio(self.sample_rate, self.pausa_frase_s)
                    pausas += self.pausa_frase_s
                    falas += 1
        return Narration(sample_rate=self.sample_rate, pcm16=bytes(pcm),
                         utterances=falas, pauses_s=round(pausas, 2))

    def _tts_speak(self, frase: str, speed: float, noise: float) -> Utterance:
        # Enfase vira pausa ao redor do trecho, nao tag enviada ao motor.
        partes = _ENFASE.split(frase)
        if len(partes) == 1:
            return self.tts.speak(frase, speed=speed, noise=noise)
        saida = bytearray()
        rate = self.sample_rate
        for i, p in enumerate(partes):
            if not p:
                continue
            if i % 2 == 1:
                saida += _silencio(rate, _MICRO_PAUSA)
                u = self.tts.speak(p, speed=speed, noise=noise)
                saida += u.pcm16
                saida += _silencio(rate, _MICRO_PAUSA)
                rate = u.sample_rate
            else:
                u = self.tts.speak(p, speed=speed, noise=noise)
                saida += u.pcm16
                rate = u.sample_rate
        return Utterance(sample_rate=rate, pcm16=bytes(saida))


def _expandir_marcadores(text: str) -> list[tuple[str, object]]:
    """Texto para [('fala', trecho), ('pausa', segundos), ...]."""
    saida: list[tuple[str, object]] = []
    pos = 0
    for m in _MARCADOR.finditer(text):
        if m.start() > pos:
            saida.append(("fala", text[pos:m.start()]))
        saida.append(("pausa", PAUSAS[m.group(1).lower()]))
        pos = m.end()
    if pos < len(text):
        saida.append(("fala", text[pos:]))
    return [(k, v) for k, v in saida if (k == "pausa" or str(v).strip())]


__all__ = ["Narration", "Narrator", "PAUSAS"]
