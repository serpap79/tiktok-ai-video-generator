"""Trilha de fundo gerada aqui: dark ambient por clima, sem direito autoral.

Por que gerar em vez de baixar. As musicas que o MoneyPrinterTurbo usava vem
de videos do YouTube, e o proprio README deles pede para apagar se houver
problema de direito autoral -- num canal que busca o Creator Rewards, audio
sem licenca e risco de video mudo ou reivindicado. O Lyria (musica do
Gemini) foi medido em 19/09/2026: cota zero no free tier. Bibliotecas livres
(YouTube Audio Library, Pixabay) nao tem API.

O que sobra, a $0 e sem risco: sintetizar. O canal e "dark" e a trilha fica
~20 dB abaixo da voz, entao o que ela precisa ter e textura e pulso, nao
melodia: drone grave com filtro que respira, acorde de pad em progressao
menor, pulso de bumbo discreto e, nos climas de descoberta, um arpejo com
eco. Tudo com numpy, deterministico pela semente (o mesmo tema gera a mesma
trilha -- reproduzivel), em ~1s para 90s de audio.

Clima por tipo de conteudo (pilar da marca): noticia pulsa, curiosidade
descobre, analise tensiona, tutorial foca, futuro abre horizonte, VS duela,
historia soa como arquivo.
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
    "descoberta": Mood("descoberta", 73.42, 96, drone=0.35, pad=0.40, pulse=0.20,
                       ticks=0.0, arp=0.35, air=0.10),
    "tensao": Mood("tensao", 69.30, 70, drone=0.70, pad=0.30, pulse=0.40, ticks=0.0,
                   arp=0.0, air=0.15, pulse_every=2),
    "foco": Mood("foco", 82.41, 100, drone=0.30, pad=0.45, pulse=0.25, ticks=0.06,
                 arp=0.18, air=0.05),
    "horizonte": Mood("horizonte", 87.31, 80, drone=0.35, pad=0.55, pulse=0.0, ticks=0.0,
                      arp=0.22, air=0.20),
    "duelo": Mood("duelo", 98.0, 110, drone=0.45, pad=0.25, pulse=0.50, ticks=0.14,
                  arp=0.0, air=0.05),
    "arquivo": Mood("arquivo", 73.42, 66, drone=0.40, pad=0.55, pulse=0.0, ticks=0.0,
                    arp=0.12, air=0.25, harmonics=4),
}

MOOD_BY_PILLAR: dict[str, str] = {
    "news": "pulso", "fato": "descoberta", "analise": "tensao", "tutorial": "foco",
    "futuro": "horizonte", "vs": "duelo", "historia": "arquivo",
}

# Progressao menor cinematica (i - VI - III - VII), em semitons sobre a raiz.
PROGRESSAO = ((0, 3, 7), (-4, 0, 3), (3, 7, 10), (-2, 2, 5))


def mood_for(pillar: str) -> Mood:
    return MOODS[MOOD_BY_PILLAR.get(pillar, "pulso")]


def _semente(texto: str) -> int:
    return int(hashlib.sha256(texto.encode()).hexdigest()[:8], 16)


def synthesize(duration_s: float, mood: Mood, seed: str = "") -> np.ndarray:
    """Estereo float32 em [-1, 1], com fade de entrada e saida."""
    rng = np.random.default_rng(_semente(seed or mood.id))
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float64) / SR
    # Variacao por tema: raiz desloca ate 2 semitons, progressao gira.
    raiz = mood.root_hz * 2 ** (int(rng.integers(-2, 3)) / 12)
    giro = int(rng.integers(0, len(PROGRESSAO)))
    batida = 60.0 / mood.bpm
    compasso = 4 * batida
    esq = np.zeros(n)
    dir_ = np.zeros(n)

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
                dir_ += onda * ganho * (0.8 if voz == 0 else 1.0)

    if mood.pad:
        # Um acorde a cada 2 compassos, com ataque e soltura lentos (crossfade).
        dur_acorde = 2 * compasso
        n_acordes = int(np.ceil(duration_s / dur_acorde)) + 1
        for i in range(n_acordes):
            inicio = i * dur_acorde
            a = max(0, int((inicio - 1.0) * SR))
            b = min(n, int((inicio + dur_acorde + 1.0) * SR))
            if b <= a:
                continue
            tt = t[a:b]
            env = _janela(tt, inicio - 1.0, inicio + dur_acorde + 1.0, ataque=1.5)
            acorde = PROGRESSAO[(i + giro) % len(PROGRESSAO)]
            for semi in acorde:
                f = raiz * 4 * 2 ** (semi / 12)
                for det, pan in ((-0.004, 0.2), (0.0, 0.5), (0.004, 0.8)):
                    onda = np.sin(2 * np.pi * f * (1 + det) * tt + rng.uniform(0, 6))
                    onda += 0.25 * np.sin(4 * np.pi * f * (1 + det) * tt)
                    ganho = mood.pad * env / (3 * len(acorde)) * 0.9
                    esq[a:b] += onda * ganho * (1 - pan)
                    dir_[a:b] += onda * ganho * pan

    if mood.pulse:
        passo = batida * mood.pulse_every
        bumbo = _bumbo()
        for inicio in np.arange(0.0, duration_s, passo):
            _somar(esq, dir_, bumbo * mood.pulse, int(inicio * SR))

    if mood.ticks:
        tick = _tick(rng)
        for inicio in np.arange(batida / 2, duration_s, batida / 2):
            _somar(esq, dir_, tick * mood.ticks, int(inicio * SR), pan=0.65)

    if mood.arp:
        nota_dur = batida / 2
        i = 0
        for inicio in np.arange(0.0, duration_s, nota_dur):
            acorde = PROGRESSAO[(int(inicio // (2 * compasso)) + giro) % len(PROGRESSAO)]
            semi = acorde[i % len(acorde)] + (12 if (i // len(acorde)) % 2 else 0)
            nota = _pluck(raiz * 8 * 2 ** (semi / 12)) * mood.arp
            pan = 0.35 if i % 2 else 0.65
            _somar(esq, dir_, nota, int(inicio * SR), pan=pan)
            # Eco de 3/8 de tempo, duas repeticoes.
            for rep, queda in ((1, 0.35), (2, 0.12)):
                _somar(esq, dir_, nota * queda, int((inicio + rep * 0.75 * batida) * SR),
                       pan=1 - pan)
            i += 1

    if mood.air:
        ruido = np.cumsum(rng.standard_normal(n)) / 400
        ruido -= _media_movel(ruido, 2048)
        esq += ruido * mood.air * 0.6
        dir_ += np.roll(ruido, 1103) * mood.air * 0.6

    estereo = np.stack([esq, dir_], axis=1)
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
    """Trilha do video: clima do pilar, semente do tema, duracao do video."""
    return write_wav(path, synthesize(duration_s + 0.5, mood_for(pillar), seed))


# ---------------------------------------------------------------- pecas

def _janela(t: np.ndarray, inicio: float, fim: float, ataque: float) -> np.ndarray:
    sobe = np.clip((t - inicio) / ataque, 0, 1)
    desce = np.clip((fim - t) / ataque, 0, 1)
    return np.minimum(sobe, desce)


def _media_movel(x: np.ndarray, k: int) -> np.ndarray:
    """Media movel centrada em O(n) (convolucao direta custava ~5s em 90s)."""
    c = np.cumsum(np.concatenate([np.zeros(1), x]))
    media = (c[k:] - c[:-k]) / k
    frente = (k - 1) // 2
    return np.concatenate([np.full(frente, media[0]), media,
                           np.full(len(x) - len(media) - frente, media[-1])])


def _bumbo() -> np.ndarray:
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


def _somar(esq: np.ndarray, dir_: np.ndarray, som: np.ndarray, inicio: int,
           pan: float = 0.5) -> None:
    if inicio >= len(esq):
        return
    fim = min(len(esq), inicio + len(som))
    trecho = som[: fim - inicio]
    esq[inicio:fim] += trecho * (1 - pan) * 2 * 0.5
    dir_[inicio:fim] += trecho * pan * 2 * 0.5


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
