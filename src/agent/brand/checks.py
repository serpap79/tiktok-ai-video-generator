"""Portoes da marca: o que da para medir sem julgamento, em texto de correcao.

Todo portao aqui vem de regra literal do vetor (`brand/brand.json`): gancho
ate 12 palavras, um numero por frase, sem emoji, sem bordao. Promessa que o
video nao paga e julgamento -- mora no juiz, nao aqui.
"""

from __future__ import annotations

import re

HOOK_MAX_WORDS = 12

_DIGITO = re.compile(r"\d+(?:[.,]\d+)?")
_INTERVALO = re.compile(r"\d+(?:[.,]\d+)?\s*(?:a|até|ate|-)\s*\d+(?:[.,]\d+)?")


def _sem_intervalos(text: str) -> str:
    """'25 a 300' e uma quantidade so: mantem o primeiro numero."""
    return _INTERVALO.sub(lambda m: re.findall(r"\d+(?:[.,]\d+)?", m.group(0))[0],
                          text)
_NOME_COLADO = re.compile(r"\b(?=[\w.,]*[A-Za-zÀ-ÿ])[\w.,]*\d[\w.,]*\b")
_NOME_SIGLA_NUMERO = re.compile(r"[A-ZÀ-Þ]{2,}\s+\d+(?:[.,]\d+)?")


def _sem_nomes(text: str) -> str:
    """Tira nomes proprios da conta: 27B, FP16 e RTX 5090 sao uma unidade
    lexical para o publico tech, nao duas quantidades na mesma frase."""
    text = _NOME_COLADO.sub("", text)
    return _NOME_SIGLA_NUMERO.sub(lambda m: m.group(0).split()[0], text)
_SENTENCA = re.compile(r"(?<=[.!?…])\s+|\n+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]")
_BORDAO = ("fala galera", "fala, galera", "se inscreva", "se inscrevam",
           "inscreva-se")


def hook_words(hook: str) -> int:
    return len(hook.split())


def check_hook(hook: str) -> str | None:
    n = hook_words(hook)
    if n > HOOK_MAX_WORDS:
        return (f"gancho com {n} palavras (teto {HOOK_MAX_WORDS} da marca): "
                "corte ate a lacuna caber numa frase.")
    return None


def check_numbers(text: str, ignorar_ate: int = 0) -> list[str]:
    """Uma frase, um numero: dois numeros pedem duas frases.

    `ignorar_ate` isenta contagens estruturais pequenas (o "5" da promessa do
    carrossel e o formato, nao afirmacao -- mesma isencao do portao de
    ancoragem). Na narracao de video vale 0: la todo numero e afirmacao.
    """
    problemas = []
    for sent in [s.strip() for s in _SENTENCA.split(_sem_intervalos(text))
                 if s.strip()]:
        nums = [n for n in _DIGITO.findall(_sem_nomes(sent))
                if not (n.isdigit() and int(n) <= ignorar_ate)]
        if len(nums) > 1:
            problemas.append(
                f"frase com {len(nums)} numeros ('{sent[:60]}...'): "
                "maximo um por frase -- quebre em duas.")
    return problemas


def check_emoji_bordao(text: str) -> list[str]:
    problemas = []
    if _EMOJI.search(text):
        problemas.append("emoji no texto: a marca nunca usa.")
    baixa = text.lower()
    for b in _BORDAO:
        if b in baixa:
            problemas.append(f"bordao banido pela marca: {b!r}.")
            break
    return problemas


__all__ = ["HOOK_MAX_WORDS", "check_emoji_bordao", "check_hook",
           "check_numbers", "hook_words"]
