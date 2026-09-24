"""Más fuentes para el tema, halladas en la propia recolección del radar y sin búsqueda de pago.

El primer slot real (20/09) creó el expediente de «Gemini went rogue» con UNA
fuente (The Verge). En la misma recolección había otros artículos de la misma historia,
incluso en un medio extranjero, que la agrupación de RSS no juntó porque
o titulo estava em outro idioma: "Gemini invade empresas" e "Gemini went
rogue" dividem pouco vocabulario comum.

Lo que sobrevive a la traducción son los NOMBRES: Gemini, Google, GPT-6, RTX 5090.
Dos nombres propios en común (o uno con dígito, que es un nombre de producto)
bastan para tratarlas como la misma historia y ofrecer la página al
investigador. La comprobación de Fragmento literal sigue vigente página a página,
así, un artículo incorrecto cuesta una llamada y ningún hecho falso.
"""

from __future__ import annotations

import re

from agent.models import Decision, NewsItem, Signal
from agent.text import strip_accents

_PALAVRA = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9.+-]*")
# Palavras que abrem titulo com maiuscula sem ser nome.
_COMUNS = frozenset("""
the a an how why what when who this that these new now its it is are was
o a os as um uma como por que quando quem este esta novo nova agora
""".split())


def names(titulo: str) -> set[str]:
    """Nomes proprios e codinomes de produto do titulo, normalizados."""
    saida: set[str] = set()
    for i, bruto in enumerate(_PALAVRA.findall(titulo)):
        palavra = bruto.strip(".-+")
        if len(palavra) < 2:
            continue
        chave = strip_accents(palavra).lower()
        tem_digito = any(c.isdigit() for c in palavra)
        maiuscula = palavra[:1].isupper()
        if chave in _COMUNS:
            continue
        if tem_digito and any(c.isalpha() for c in palavra):
            saida.add(chave)
        elif maiuscula and (i > 0 or len(palavra) >= 4) and len(palavra) >= 3:
            saida.add(chave)
    return saida


def related_items(decision: Decision, signals: list[Signal], limit: int = 3) -> list[NewsItem]:
    """Artículos de la misma recolección que tratan la misma historia en otro medio."""
    alvo = names(decision.term)
    if not alvo:
        return []
    ja = {str(decision.url or "")} | {str(n.url) for n in decision.news_items}
    achados: list[tuple[int, NewsItem]] = []
    for s in signals:
        # Solo fuentes cuya URL es el artículo: RSS y tarjetas de modelo. La URL de HN es la
        # discusión, y la de Wikipedia más visitada es la entrada, no la noticia.
        if (s.url is None or str(s.url) in ja
                or not (s.source.startswith("rss") or s.source == "huggingface")):
            continue
        comuns = alvo & names(s.term)
        forte = any(any(c.isdigit() for c in n) for n in comuns)
        if len(comuns) >= 2 or (forte and comuns):
            try:
                item = NewsItem(title=s.term, url=str(s.url), source_name=s.source)
            except ValueError:
                continue
            achados.append((len(comuns), item))
            ja.add(str(s.url))
    achados.sort(key=lambda x: -x[0])
    return [item for _, item in achados[:limit]]


__all__ = ["names", "related_items"]
