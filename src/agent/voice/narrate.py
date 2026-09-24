"""Guion a wav: frase a frase, con pausa y enfasis marcadas en el texto.

Marcacion (opcional, desaparece antes de la sintesis):
- `[PAUSA CORTA]` 0,3s · `[PAUSA MEDIA]` 0,6s · `[PAUSA LARGA]` 1,0s
- `*palabra*` mete micro-pausa (0,12s) antes y despues -- aproximacion honesta
  de enfasis: el Piper no tiene SSML, y la pausa es lo que el oido percibe como
  destaque sin cambiar el timbre.

Puntuacion se vuelve pausa sola: punto final y salto de linea respiran por el
`pausa_frase_s` del estilo. La variacion entre generaciones viene del propio
VITS (muestreo con noise_scale): el mismo guion nunca sale byte-identico, sin
parametro de seed porque el motor no expone uno.
"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass

from agent.voice.engine import TTS, Utterance

PAUSAS = {
    "pausa corta": 0.3,
    "pausa media": 0.6,
    "pausa larga": 1.0,
}
_MICRO_PAUSA = 0.12
_MARCADOR = re.compile(r"\[(pausa corta|pausa media|pausa larga)\]", re.IGNORECASE)
_ENFASIS = re.compile(r"\*([^*]{1,60})\*")
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
        """Texto (o narracion hook+body+closing) a audio concatenado."""
        bloques = _expandir_marcadores(text)
        pcm = bytearray()
        hablas = 0
        pausas = 0.0
        for kind, valor in bloques:
            if kind == "pausa":
                pcm += _silencio(self.sample_rate, valor)
                pausas += valor
            else:
                for frase in [f for f in _FRASE.split(valor) if f.strip()]:
                    u = self._tts_speak(frase, speed, noise)
                    pcm += u.pcm16
                    pcm += _silencio(self.sample_rate, self.pausa_frase_s)
                    pausas += self.pausa_frase_s
                    hablas += 1
        return Narration(sample_rate=self.sample_rate, pcm16=bytes(pcm),
                         utterances=hablas, pauses_s=round(pausas, 2))

    def _tts_speak(self, frase: str, speed: float, noise: float) -> Utterance:
        # El enfasis se vuelve pausa alrededor del pasaje, no tag enviada al motor.
        partes = _ENFASIS.split(frase)
        if len(partes) == 1:
            return self.tts.speak(frase, speed=speed, noise=noise)
        salida = bytearray()
        rate = self.sample_rate
        for i, p in enumerate(partes):
            if not p:
                continue
            if i % 2 == 1:
                salida += _silencio(rate, _MICRO_PAUSA)
                u = self.tts.speak(p, speed=speed, noise=noise)
                salida += u.pcm16
                salida += _silencio(rate, _MICRO_PAUSA)
                rate = u.sample_rate
            else:
                u = self.tts.speak(p, speed=speed, noise=noise)
                salida += u.pcm16
                rate = u.sample_rate
        return Utterance(sample_rate=rate, pcm16=bytes(salida))


def _expandir_marcadores(text: str) -> list[tuple[str, object]]:
    """Texto a [('habla', pasaje), ('pausa', segundos), ...]."""
    salida: list[tuple[str, object]] = []
    pos = 0
    for m in _MARCADOR.finditer(text):
        if m.start() > pos:
            salida.append(("habla", text[pos:m.start()]))
        salida.append(("pausa", PAUSAS[m.group(1).lower()]))
        pos = m.end()
    if pos < len(text):
        salida.append(("habla", text[pos:]))
    return [(k, v) for k, v in salida if (k == "pausa" or str(v).strip())]


__all__ = ["Narration", "Narrator", "PAUSAS"]
