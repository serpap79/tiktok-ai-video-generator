"""Mais fontes para o tema, garimpadas na propria coleta do radar -- sem busca paga.

O primeiro slot real (20/09) montou o dossie do "Gemini went rogue" com UMA
fonte (The Verge). Na mesma coleta havia outras materias da mesma historia,
inclusive em veiculo brasileiro, que o agrupamento do RSS nao juntou porque
o titulo estava em outro idioma: "Gemini invade empresas" e "Gemini went
rogue" dividem pouco vocabulario comum.

O que sobrevive a traducao sao os NOMES: Gemini, Google, GPT-6, RTX 5090.
Dois nomes proprios em comum (ou um com digito, que e nome de produto)
bastam para tratar como a mesma historia e oferecer a pagina ao
pesquisador. O portao de trecho literal continua valendo pagina a pagina,
entao uma materia errada custa uma chamada e nenhum fato falso.
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
    """Materias da mesma coleta que falam da mesma historia, em outro veiculo."""
    alvo = names(decision.term)
    if not alvo:
        return []
    ja = {str(decision.url or "")} | {str(n.url) for n in decision.news_items}
    achados: list[tuple[int, NewsItem]] = []
    for s in signals:
        # So fonte cuja URL e a materia: RSS e model card. A URL do HN e a
        # discussao, e a da Wikipedia mais vista e o verbete, nao a noticia.
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
