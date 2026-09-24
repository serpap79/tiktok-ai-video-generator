"""Normalizacion de texto compartida por las etapas.

Vive aparte porque cuatro sitios necesitan el mismo tratamiento: el filtro de
politica, la puerta de nicho, la deduplicacion y la composicion de la consulta
de busqueda del investigador. Si cada uno normalizara a su manera,
"Inteligencia" casaria en una regla y se escaparia de la otra.
"""

from __future__ import annotations

import re
import unicodedata

# Palabras sin carga tematica en castellano e ingles. Entran en la deduplicacion
# (dos titulos sobre el mismo asunto no deben parecerse solo por compartir "the")
# y quedan fuera del conteo de nicho.
STOPWORDS = frozenset("""
el la los las un una unos unas de del al en y o u es son era fue ser sido este
esta esto estos estas ese esa eso esos esas aquel aquella aquellos aquellas
lo le les su sus mi mis tu tus nos vos mas menos muy mucho muchos mucha ya no
si como cuando donde cual cuales quien quienes cuyo entre tras hasta desde
sobre contra durante por para con sin bajo ante segun tal tan tanto tampoco
the a an of in on at to for with without and or but that which who whom this
these those is are was were be been being it its as by from into over under
""".split())

_NO_ALFANUM = re.compile(r"[^a-z0-9]+")


def strip_accents(texto: str) -> str:
    """Quita diacriticos: "inteligencia" -> "inteligencia".

    Las fuentes mezclan castellano e ingles y no siempre acentuan de forma
    consistente; casar la regla con acento haria fallar el filtro de politica
    por una tilde.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c)
    )


def normalize(texto: str) -> str:
    """Minusculas, sin acento, sin puntuacion, espacio unico."""
    return " ".join(_NO_ALFANUM.sub(" ", strip_accents(texto).lower()).split())


def tokens(texto: str, drop_stopwords: bool = True) -> list[str]:
    palabras = normalize(texto).split()
    if drop_stopwords:
        palabras = [p for p in palabras if p not in STOPWORDS]
    return palabras


def content_tokens(texto: str, min_len: int = 3) -> set[str]:
    """Tokens que cargan asunto: sin stopword y sin fragmento corto.

    Los numeros escapan del suelo de tamano porque suelen ser el propio asunto
    ("27B", "5090", "GPT 6") y son exactamente lo que distingue dos titulos
    parecidos sobre lanzamientos diferentes.
    """
    return {t for t in tokens(texto) if len(t) >= min_len or t.isdigit()}
