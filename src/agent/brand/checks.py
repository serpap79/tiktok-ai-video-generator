"""Puertas de la marca: lo que se puede medir sin juicio, en texto de correccion.

Toda puerta aqui viene de regla literal del vector (`brand/brand.json`): gancho
hasta 12 palabras, un numero por frase, sin emoji, sin muletilla. Promesa que
el video no paga es juicio -- vive en el juez, no aqui.
"""

from __future__ import annotations

import re

HOOK_MAX_WORDS = 12

_DIGITO = re.compile(r"\d+(?:[.,]\d+)?")
_INTERVALO = re.compile(r"\d+(?:[.,]\d+)?\s*(?:a|al|hasta|-)\s*\d+(?:[.,]\d+)?")


def _sin_intervalos(text: str) -> str:
    """'25 a 300' es una cantidad sola: se queda el primer numero."""
    return _INTERVALO.sub(lambda m: re.findall(r"\d+(?:[.,]\d+)?", m.group(0))[0],
                          text)
_NOMBRE_PEGADO = re.compile(r"\b(?=[\w.,]*[A-Za-zÀ-ÿ])[\w.,]*\d[\w.,]*\b")
_NOMBRE_SIGLA_NUMERO = re.compile(r"[A-ZÀ-Þ]{2,}\s+\d+(?:[.,]\d+)?")


def _sin_nombres(text: str) -> str:
    """Quita nombres propios de la cuenta: 27B, FP16 y RTX 5090 son una unidad
    lexica para el publico tech, no dos cantidades en la misma frase."""
    text = _NOMBRE_PEGADO.sub("", text)
    return _NOMBRE_SIGLA_NUMERO.sub(lambda m: m.group(0).split()[0], text)
_SENTENCIA = re.compile(r"(?<=[.!?…])\s+|\n+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]")
_MULETILLA = ("hola a todos", "que pasa gente", "os suscribis", "suscribete",
              "dame like", "deja tu like")


def hook_words(hook: str) -> int:
    return len(hook.split())


def check_hook(hook: str) -> str | None:
    n = hook_words(hook)
    if n > HOOK_MAX_WORDS:
        return (f"gancho con {n} palabras (techo {HOOK_MAX_WORDS} de la marca): "
                "corta hasta que el hueco quepa en una frase.")
    return None


def check_numbers(text: str, ignorar_hasta: int = 0) -> list[str]:
    """Una frase, un numero: dos numeros piden dos frases.

    `ignorar_hasta` exime conteos estructurales pequenos (el "5" de la promesa
    del carrusel es el formato, no afirmacion -- misma exencion de la puerta de
    anclaje). En la narracion de video vale 0: ahi todo numero es afirmacion.
    """
    problemas = []
    for sent in [s.strip() for s in _SENTENCIA.split(_sin_intervalos(text))
                 if s.strip()]:
        nums = [n for n in _DIGITO.findall(_sin_nombres(sent))
                if not (n.isdigit() and int(n) <= ignorar_hasta)]
        if len(nums) > 1:
            problemas.append(
                f"frase con {len(nums)} numeros ('{sent[:60]}...'): "
                "maximo uno por frase -- rompela en dos.")
    return problemas


_TRANS = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")


def check_emoji_muletilla(text: str) -> list[str]:
    """La comparacion ignora tildes: en Espana la gente escribe igual
    '¿Que pasa gente!' que '¿Qué pasa gente!' y ambas son muletilla."""
    problemas = []
    if _EMOJI.search(text):
        problemas.append("emoji en el texto: la marca nunca los usa.")
    baja = text.lower().translate(_TRANS)
    for b in _MULETILLA:
        if b in baja:
            problemas.append(f"muletilla vetada por la marca: {b!r}.")
            break
    return problemas


__all__ = ["HOOK_MAX_WORDS", "check_emoji_muletilla", "check_hook",
           "check_numbers", "hook_words"]
