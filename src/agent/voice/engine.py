"""Porta TTS + adaptador Piper (ONNX local, CPU).

O adaptador e fino de proposito: carrega o `.onnx` (+ `.onnx.json` irmao),
sintetiza um trecho e devolve PCM. Pausa, ordem e concatenacao moram em
`narrate.py`, que e testavel com qualquer backend -- inclusive um falso, o
que mantem a suite hermetica sem baixar 200 MB de modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Utterance:
    sample_rate: int
    pcm16: bytes

    @property
    def duration_s(self) -> float:
        return len(self.pcm16) / 2 / self.sample_rate if self.sample_rate else 0.0


class TTS(Protocol):
    def speak(self, text: str, *, speed: float = 1.0, noise: float = 0.667,
              seed: int | None = None) -> Utterance: ...


class PiperTTS:
    """Voz Piper via `piper-tts`. Exige espeak-ng no sistema (Fedora: dnf)."""

    def __init__(self, model_path: str | Path, sample_rate: int = 22050):
        from piper import PiperVoice  # import tardio: suite nao paga o import
        self._voice = PiperVoice.load(str(model_path))
        self._rate = sample_rate

    def speak(self, text: str, *, speed: float = 1.0, noise: float = 0.667,
              seed: int | None = None) -> Utterance:
        from piper import config as piper_config
        # length_scale e o inverso da velocidade; noise_scale da a variacao
        # entre geracoes. seed nao existe no Piper: variacao natural do VITS.
        _ = seed
        chunks = self._voice.synthesize(
            text,
            syn_config=piper_config.SynthesisConfig(
                length_scale=max(0.5, min(2.0, 1.0 / speed)),
                noise_scale=noise,
                noise_w_scale=0.8,
            ),
        )
        pcm = bytearray()
        rate = self._rate
        for c in chunks:
            rate = c.sample_rate
            pcm += bytes(c.audio_int16_bytes)
        return Utterance(sample_rate=rate, pcm16=bytes(pcm))


__all__ = ["PiperTTS", "TTS", "Utterance"]
