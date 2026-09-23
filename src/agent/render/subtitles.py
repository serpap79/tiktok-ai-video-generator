"""Legenda karaoke em ASS: grafia original, tempo da voz, palavra ativa em destaque.

A voz le "Djemini"; a tela mostra "Gemini". O tempo de cada palavra vem dos
eventos WordBoundary do TTS, que falam da grafia FALADA -- `align` casa esses
eventos com as palavras faladas e devolve o tempo as palavras originais pelo
alinhamento de `voice/pronounce.py`.

Estilo: bloco de ate 3 palavras (uma so por tela era o MPT: "virgula",
"ponto" soltos no meio do quadro, sem contexto), palavra que esta sendo dita
no acento da marca, fonte da marca (Space Grotesk 700), contorno grosso para
ler sobre qualquer clipe, a 60% da altura -- abaixo do cartao do gancho,
acima da interface do TikTok.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from agent.voice.edge import WordTiming
from agent.voice.pronounce import Respelled

MAX_PALAVRAS = 3
MAX_CARACTERES = 22
Y_LEGENDA = 1150
TAMANHO = 84


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


def _norm(texto: str) -> str:
    sem = unicodedata.normalize("NFD", texto.lower())
    sem = "".join(c for c in sem if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", sem)


def align(respelled: Respelled, timings: list[WordTiming]) -> list[TimedWord]:
    """Tempo de cada palavra ORIGINAL a partir dos limites da fala."""
    falado = respelled.spoken
    dono: list[int] = []
    for i, n in enumerate(respelled.groups):
        dono.extend([i] * n)
    inicio: list[float | None] = [None] * len(falado)
    fim: list[float | None] = [None] * len(falado)

    j = 0
    for t in timings:
        alvo = _norm(t.text)
        if not alvo:
            continue
        for k in range(j, min(j + 8, len(falado))):
            fk = _norm(falado[k])
            if fk and (fk == alvo or fk.startswith(alvo) or alvo.startswith(fk)):
                inicio[k] = t.start_s if inicio[k] is None else inicio[k]
                fim[k] = t.end_s
                j = k + 1
                break

    _preencher(inicio, fim)
    palavras: list[TimedWord] = []
    for i, original in enumerate(respelled.original):
        ks = [k for k, d in enumerate(dono) if d == i]
        if ks:
            s = min(inicio[k] for k in ks if inicio[k] is not None)
            e = max(fim[k] for k in ks if fim[k] is not None)
        elif palavras:
            # Parte de nome composto ("Street" em "Wall Street Journal"): acende
            # junto com a primeira palavra do nome.
            s, e = palavras[-1].start, palavras[-1].end
        else:
            s = e = 0.0
        palavras.append(TimedWord(original, s, e))
    return palavras


def _preencher(inicio: list[float | None], fim: list[float | None]) -> None:
    """Palavra falada sem evento de limite ganha o tempo entre as vizinhas."""
    n = len(inicio)
    for k in range(n):
        if inicio[k] is not None:
            continue
        ant = next((fim[x] for x in range(k - 1, -1, -1) if fim[x] is not None), 0.0)
        prox = next((inicio[x] for x in range(k + 1, n) if inicio[x] is not None), None)
        prox = prox if prox is not None else ant + 0.3
        inicio[k] = ant
        fim[k] = max(ant, prox)


def chunks(palavras: list[TimedWord]) -> list[list[TimedWord]]:
    """Blocos de ate 3 palavras, quebrando em pontuacao."""
    blocos: list[list[TimedWord]] = []
    atual: list[TimedWord] = []
    for p in palavras:
        texto = " ".join(w.text for w in [*atual, p])
        if atual and (len(atual) >= MAX_PALAVRAS or len(texto) > MAX_CARACTERES):
            blocos.append(atual)
            atual = []
        atual.append(p)
        if re.search(r"[.!?;:,…]$", p.text):
            blocos.append(atual)
            atual = []
    if atual:
        blocos.append(atual)
    return blocos


def _ass_cor(hex_rgb: str, alpha: int = 0) -> str:
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


def build_ass(palavras: list[TimedWord], destino: Path, *, accent: str,
              font_family: str, duration: float, y: int = Y_LEGENDA,
              start_at: float = 0.0) -> Path:
    """Um evento por palavra ativa: o bloco inteiro na tela, a atual colorida.

    `y` e `start_at` vem da encenacao do apresentador quando ha um: enquanto ele
    esta grande na chamada, quem escreve o gancho e o cartao -- a legenda entra
    depois, mais alta, para passar acima da cabeca dele no canto. Sem isso o
    mesmo texto apareceria escrito duas vezes, e a segunda em cima do rosto.
    """
    branco = _ass_cor("#FFFFFF")
    destaque = _ass_cor(accent)
    cabecalho = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
        " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
        " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Fala,{font_family},{TAMANHO},{branco},{branco},&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,6,2,5,60,60,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    linhas: list[str] = []
    blocos = chunks(palavras)
    for b, bloco in enumerate(blocos):
        fim_bloco = (blocos[b + 1][0].start if b + 1 < len(blocos)
                     else min(duration, bloco[-1].end + 0.4))
        if fim_bloco <= start_at:
            continue
        for i, palavra in enumerate(bloco):
            ini = max(palavra.start, start_at)
            fim = bloco[i + 1].start if i + 1 < len(bloco) else fim_bloco
            if fim <= ini:
                fim = ini + 0.05
            partes = []
            for k, w in enumerate(bloco):
                cor = destaque if k == i else branco
                partes.append(f"{{\\c{cor}}}{_escapar(w.text)}")
            texto = f"{{\\an5\\pos(540,{y})}}" + " ".join(partes)
            linhas.append(f"Dialogue: 0,{_t(ini)},{_t(fim)},Fala,,0,0,0,,{texto}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(cabecalho + "\n".join(linhas) + "\n", encoding="utf-8")
    return destino


__all__ = ["TimedWord", "align", "build_ass", "chunks"]
