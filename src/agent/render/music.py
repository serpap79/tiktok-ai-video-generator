"""Pista de fondo generada aqui: dark ambient por clima, sin derechos de autor.

Por que generar en vez de descargar. Las musicas que usaba MoneyPrinterTurbo
vienen de videos de YouTube, y el propio README de ellos pide borrarlas si hay
problema de derechos de autor -- en un canal que busca el Creator Rewards,
audio sin licencia es riesgo de video mudo o reclamado. El Lyria (musica de
Gemini) se midio en 19/09/2026: cuota cero en free tier. Bibliotecas libres
(YouTube Audio Library, Pixabay) no tienen API.

Lo que queda, a $0 y sin riesgo: sintetizar. El canal es "dark" y la pista se
queda ~20 dB por debajo de la voz, asi que lo que necesita es textura y pulso,
no melodia: drone grave con filtro que respira, acorde de pad en progresion
menor, pulso de bombo discreto y, en los climas de descubrimiento, un arpegio
con eco. Todo con numpy, determinista por la semilla (el mismo tema genera la
misma pista -- reproducible), en ~1s para 90s de audio.

Clima por tipo de contenido (pilar de la marca): noticia pulsa, curiosidad
descubre, analisis tensa, tutorial enfoca, futuro abre horizonte, VS duele,
historia suena a archivo.
"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 44100


@dataclass(frozen=True)
class Mood:
    id: str
    root_hz: float
    bpm: float
    drone: float
    pad: float
    pulse: float
    ticks: float
    arp: float
    air: float
    # Pulso a cada N tempos (1 = seminima, 2 = minima).
    pulse_every: int = 1
    harmonics: int = 8


MOODS: dict[str, Mood] = {
    "pulso": Mood("pulso", 55.0, 88, drone=0.55, pad=0.35, pulse=0.45, ticks=0.10,
                  arp=0.0, air=0.10),
    "descubrimiento": Mood("descubrimiento", 73.42, 96, drone=0.35, pad=0.40, pulse=0.20,
                           ticks=0.0, arp=0.35, air=0.10),
    "tension": Mood("tension", 69.30, 70, drone=0.70, pad=0.30, pulse=0.40, ticks=0.0,
                    arp=0.0, air=0.15, pulse_every=2),
    "foco": Mood("foco", 82.41, 100, drone=0.30, pad=0.45, pulse=0.25, ticks=0.06,
                 arp=0.18, air=0.05),
    "horizonte": Mood("horizonte", 87.31, 80, drone=0.35, pad=0.55, pulse=0.0, ticks=0.0,
                      arp=0.22, air=0.20),
    "duelo": Mood("duelo", 98.0, 110, drone=0.45, pad=0.25, pulse=0.50, ticks=0.14,
                  arp=0.0, air=0.05),
    "archivo": Mood("archivo", 73.42, 66, drone=0.40, pad=0.55, pulse=0.0, ticks=0.0,
                    arp=0.12, air=0.25, harmonics=4),
}

MOOD_BY_PILLAR: dict[str, str] = {
    "news": "pulso", "dato": "descubrimiento", "analisis": "tension", "tutorial": "foco",
    "futuro": "horizonte", "vs": "duelo", "historia": "archivo",
}

# Progresion menor cinematografica (i - VI - III - VII), en semitonos sobre la raiz.
PROGRESION = ((0, 3, 7), (-4, 0, 3), (3, 7, 10), (-2, 2, 5))


def mood_for(pillar: str) -> Mood:
    return MOODS[MOOD_BY_PILLAR.get(pillar, "pulso")]


def _semente(texto: str) -> int:
    return int(hashlib.sha256(texto.encode()).hexdigest()[:8], 16)


def synthesize(duration_s: float, mood: Mood, seed: str = "") -> np.ndarray:
    """Estereo float32 em [-1, 1], com fade de entrada e saida."""
    rng = np.random.default_rng(_semente(seed or mood.id))
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float64) / SR
    # Variacion por tema: la raiz se desplaza hasta 2 semitonos, la progresion gira.
    raiz = mood.root_hz * 2 ** (int(rng.integers(-2, 3)) / 12)
    giro = int(rng.integers(0, len(PROGRESION)))
    batida = 60.0 / mood.bpm
    compas = 4 * batida
    esq = np.zeros(n)
    der = np.zeros(n)

    if mood.drone:
        # Filtro que respira: quantos harmonicos passam oscila devagar.
        corte = 2.5 + 2.0 * (1 + np.sin(2 * np.pi * 0.045 * t + rng.uniform(0, 6))) / 2
        for voz, mult in ((0, 1.0), (1, 1.5)):
            for k in range(1, mood.harmonics + 1):
                amp = (1 / k) * np.exp(-(k - 1) / corte)
                fase = rng.uniform(0, 2 * np.pi)
                onda = amp * np.sin(2 * np.pi * raiz * mult * k * t + fase)
                ganho = mood.drone * (0.55 if voz else 1.0) / 2.2
                esq += onda * ganho * (1.0 if voz == 0 else 0.8)
                der += onda * ganho * (0.8 if voz == 0 else 1.0)

    if mood.pad:
        # Un acorde cada 2 compases, con ataque y suelta lentos (crossfade).
        dur_acorde = 2 * compas
        n_acordes = int(np.ceil(duration_s / dur_acorde)) + 1
        for i in range(n_acordes):
            inicio = i * dur_acorde
            a = max(0, int((inicio - 1.0) * SR))
            b = min(n, int((inicio + dur_acorde + 1.0) * SR))
            if b <= a:
                continue
            tt = t[a:b]
            env = _ventana(tt, inicio - 1.0, inicio + dur_acorde + 1.0, ataque=1.5)
            acorde = PROGRESION[(i + giro) % len(PROGRESION)]
            for semi in acorde:
                f = raiz * 4 * 2 ** (semi / 12)
                for det, pan in ((-0.004, 0.2), (0.0, 0.5), (0.004, 0.8)):
                    onda = np.sin(2 * np.pi * f * (1 + det) * tt + rng.uniform(0, 6))
                    onda += 0.25 * np.sin(4 * np.pi * f * (1 + det) * tt)
                    ganho = mood.pad * env / (3 * len(acorde)) * 0.9
                    esq[a:b] += onda * ganho * (1 - pan)
                    der[a:b] += onda * ganho * pan

    if mood.pulse:
        paso = batida * mood.pulse_every
        bombo = _bombo()
        for inicio in np.arange(0.0, duration_s, paso):
            _sumar(esq, der, bombo * mood.pulse, int(inicio * SR))

    if mood.ticks:
        tick = _tick(rng)
        for inicio in np.arange(batida / 2, duration_s, batida / 2):
            _sumar(esq, der, tick * mood.ticks, int(inicio * SR), pan=0.65)

    if mood.arp:
        nota_dur = batida / 2
        i = 0
        for inicio in np.arange(0.0, duration_s, nota_dur):
            acorde = PROGRESION[(int(inicio // (2 * compas)) + giro) % len(PROGRESION)]
            semi = acorde[i % len(acorde)] + (12 if (i // len(acorde)) % 2 else 0)
            nota = _pluck(raiz * 8 * 2 ** (semi / 12)) * mood.arp
            pan = 0.35 if i % 2 else 0.65
            _sumar(esq, der, nota, int(inicio * SR), pan=pan)
            # Eco de 3/8 de tiempo, dos repeticiones.
            for rep, caida in ((1, 0.35), (2, 0.12)):
                _sumar(esq, der, nota * caida, int((inicio + rep * 0.75 * batida) * SR),
                       pan=1 - pan)
            i += 1

    if mood.air:
        ruido = np.cumsum(rng.standard_normal(n)) / 400
        ruido -= _media_movil(ruido, 2048)
        esq += ruido * mood.air * 0.6
        der += np.roll(ruido, 1103) * mood.air * 0.6

    estereo = np.stack([esq, der], axis=1)
    pico = float(np.max(np.abs(estereo))) or 1.0
    estereo = estereo / pico * 0.89
    return _fades(estereo, 1.5, 2.5).astype(np.float32)


def write_wav(path: Path | str, audio: np.ndarray) -> Path:
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(destino), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return destino


def render_bed(path: Path | str, duration_s: float, pillar: str, seed: str) -> Path:
    """Pista del video: clima del pilar, semilla del tema, duracion del video."""
    return write_wav(path, synthesize(duration_s + 0.5, mood_for(pillar), seed))


# ---------------------------------------------------------------- piezas

def _ventana(t: np.ndarray, inicio: float, fin: float, ataque: float) -> np.ndarray:
    sube = np.clip((t - inicio) / ataque, 0, 1)
    baja = np.clip((fin - t) / ataque, 0, 1)
    return np.minimum(sube, baja)


def _media_movil(x: np.ndarray, k: int) -> np.ndarray:
    """Media movil centrada en O(n) (la convolucion directa costaba ~5s en 90s)."""
    c = np.cumsum(np.concatenate([np.zeros(1), x]))
    media = (c[k:] - c[:-k]) / k
    frente = (k - 1) // 2
    return np.concatenate([np.full(frente, media[0]), media,
                           np.full(len(x) - len(media) - frente, media[-1])])


def _bombo() -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    freq = 45 + 50 * np.exp(-t * 28)
    fase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(fase) * np.exp(-t * 9) * 0.9


def _tick(rng: np.random.Generator) -> np.ndarray:
    n = int(0.03 * SR)
    ruido = rng.standard_normal(n)
    ruido = np.diff(ruido, prepend=0.0)  # agudo: tira o grave do ruido
    return ruido * np.exp(-np.arange(n) / SR * 160) * 0.25


def _pluck(freq: float) -> np.ndarray:
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * 7) * np.clip(t / 0.004, 0, 1)
    return (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)) * env * 0.5


def _sumar(esq: np.ndarray, der: np.ndarray, sonido: np.ndarray, inicio: int,
           pan: float = 0.5) -> None:
    if inicio >= len(esq):
        return
    fin = min(len(esq), inicio + len(sonido))
    trozo = sonido[: fin - inicio]
    esq[inicio:fin] += trozo * (1 - pan) * 2 * 0.5
    der[inicio:fin] += trozo * pan * 2 * 0.5


def _fades(audio: np.ndarray, entrada_s: float, saida_s: float) -> np.ndarray:
    n = len(audio)
    env = np.ones(n)
    a = min(n, int(entrada_s * SR))
    b = min(n, int(saida_s * SR))
    if a:
        env[:a] = np.linspace(0, 1, a)
    if b:
        env[n - b:] = np.minimum(env[n - b:], np.linspace(1, 0, b))
    return audio * env[:, None]


__all__ = ["MOODS", "MOOD_BY_PILLAR", "Mood", "SR", "mood_for", "render_bed",
           "synthesize", "write_wav"]
