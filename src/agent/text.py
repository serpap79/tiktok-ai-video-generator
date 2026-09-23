"""Normalizacao de texto compartilhada pelos estagios.

Fica separada porque quatro lugares precisam do mesmo tratamento: o filtro de
politica, o portao de nicho, a deduplicacao e a montagem da consulta de busca do
pesquisador. Se cada um normalizasse a seu modo, "Inteligência" casaria numa
regra e escaparia da outra.
"""

from __future__ import annotations

import re
import unicodedata

# Palavras sem carga tematica em pt e en. Entram na deduplicacao (dois titulos
# sobre o mesmo assunto nao devem parecer parecidos so por compartilharem "the")
# e ficam de fora da contagem de nicho.
STOPWORDS = frozenset("""
a as o os um uma uns umas de do da dos das em no na nos nas por para com sem sob
e ou mas que se ao aos à às pelo pela pelos pelas seu sua seus suas este esta
isso isto esse essa aquele aquela mais menos muito muita ja nao sim como quando
onde qual quais quem cujo entre apos ate desde sobre contra durante
the a an of in on at to for with without and or but that which who whom this
these those is are was were be been being it its as by from into over under
""".split())

_NAO_ALFANUM = re.compile(r"[^a-z0-9]+")


def strip_accents(texto: str) -> str:
    """Remove diacriticos: "inteligência" -> "inteligencia".

    As fontes misturam pt e en e nem sempre acentuam de forma consistente; casar
    regra com acento faria o filtro de politica falhar por um til.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c)
    )


def normalize(texto: str) -> str:
    """Minusculas, sem acento, sem pontuacao, espaco unico."""
    return " ".join(_NAO_ALFANUM.sub(" ", strip_accents(texto).lower()).split())


def tokens(texto: str, drop_stopwords: bool = True) -> list[str]:
    palavras = normalize(texto).split()
    if drop_stopwords:
        palavras = [p for p in palavras if p not in STOPWORDS]
    return palavras


def content_tokens(texto: str, min_len: int = 3) -> set[str]:
    """Tokens que carregam assunto: sem stopword e sem fragmento curto.

    Numeros escapam do piso de tamanho porque costumam ser o proprio assunto
    ("27B", "5090", "GPT 6") e sao exatamente o que distingue duas materias
    parecidas sobre lancamentos diferentes.
    """
    return {t for t in tokens(texto) if len(t) >= min_len or t.isdigit()}
