"""Tipo de conteudo de um tema: noticia, curiosidade, analise, tutorial...

Os tipos sao os pilares da marca (`brand/brand.json`), e cada um carrega a
propria formula de gancho, a batida e o CTA -- e isso que faz "informativo",
"curiosidade", "tutorial" e "historia" sairem diferentes, e nao o mesmo
roteiro com rotulo trocado.

A classificacao e lexica e deterministica, de proposito: roda antes da
pesquisa, para CADA candidato do radar, e gastar uma chamada de modelo por
candidato so para rotular queimaria a cota que a escrita precisa. O custo
de errar e baixo -- o pilar vira orientacao de prompt, e o juiz le o texto
final --, e cada palpite leva o motivo gravado para calibrar depois.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.text import strip_accents

PILARES = ("news", "fato", "analise", "tutorial", "futuro", "vs", "historia")

# Pistas por pilar, em pt e en (o radar e bilingue). Peso maior = pista mais
# especifica: "how to" quase so aparece em tutorial; "new" aparece em tudo.
_PISTAS: dict[str, tuple[tuple[str, float], ...]] = {
    "tutorial": (
        ("how to", 3.0), ("como usar", 3.0), ("como fazer", 3.0), ("passo a passo", 3.0),
        ("tutorial", 3.0), ("guide", 2.0), ("guia", 2.0), ("tips", 1.5), ("dicas", 1.5),
        ("cheat sheet", 2.5), ("step by step", 3.0), ("prompt", 1.0), ("comando", 1.5),
        ("command", 1.0), ("workflow", 1.5), ("setup", 1.0), ("configurar", 1.5),
        ("show hn", 1.5), ("open source tool", 1.5), ("cli", 1.0), ("extension", 1.0),
        ("agents.md", 2.0), ("claude.md", 2.0), ("vscode", 1.0), ("plugin", 1.0),
    ),
    "vs": (
        (" vs ", 3.0), (" vs. ", 3.0), ("versus", 3.0), (" x ", 1.5), ("compared", 2.0),
        ("comparison", 2.0), ("comparacao", 2.0), ("beats", 1.5), ("outperforms", 2.0),
        ("supera", 2.0), ("melhor que", 2.0), ("better than", 2.0), ("faster than", 1.5),
        ("mais rapido que", 1.5), ("duelo", 2.0), ("against", 1.0),
    ),
    "historia": (
        ("history", 2.5), ("historia", 2.5), ("anniversary", 2.5), ("aniversario", 2.5),
        ("years ago", 2.5), ("anos atras", 2.5), ("invented", 2.0), ("inventou", 2.0),
        ("invencao", 2.0), ("first ever", 2.0), ("o primeiro", 1.5), ("a primeira", 1.5),
        ("origin", 1.5), ("origem", 1.5), ("decada", 1.5), ("legacy", 1.0),
        ("museum", 1.5), ("vintage", 1.5), ("retro", 1.0), ("1940", 2.0), ("1950", 2.0),
        ("1960", 2.0), ("1970", 2.0), ("1980", 1.5), ("1990", 1.0),
    ),
    "futuro": (
        ("future", 2.0), ("futuro", 2.0), ("2030", 3.0), ("2035", 3.0), ("2040", 3.0),
        ("2050", 3.0), ("next decade", 2.5), ("proxima decada", 2.5), ("predict", 2.0),
        ("previsao", 2.0), ("roadmap", 1.5), ("will replace", 2.0), ("vai substituir", 2.0),
        ("agi", 1.5), ("superintelligence", 2.0), ("forecast", 1.5),
    ),
    "analise": (
        ("why ", 1.5), ("por que", 1.5), ("i don't", 2.0), ("i do not", 2.0),
        ("the case for", 2.5), ("the case against", 2.5), ("should", 1.0),
        ("myth", 2.0), ("mito", 2.0), ("problem with", 2.0), ("problema", 1.0),
        ("is dead", 2.0), ("overrated", 2.0), ("bubble", 2.0), ("bolha", 2.0),
        ("ethic", 1.5), ("etica", 1.5), ("jobs", 1.5), ("empregos", 1.5),
        ("regulation", 1.5), ("regulacao", 1.5), ("lei ", 1.0), ("lawsuit", 1.5),
        ("processo judicial", 1.5), ("theft", 1.5), ("debate", 1.5), ("risk", 1.0),
        ("risco", 1.0), ("hid it", 1.5), ("escondeu", 1.5), ("went rogue", 1.5),
    ),
    "fato": (
        ("study", 2.0), ("estudo", 2.0), ("researchers", 2.0), ("pesquisadores", 2.0),
        ("scientists", 2.0), ("cientistas", 2.0), ("discover", 2.0), ("descobr", 2.0),
        ("found that", 1.5), ("record", 1.5), ("recorde", 1.5), ("largest", 1.5),
        ("maior ", 1.0), ("smallest", 1.5), ("menor ", 1.0), ("fastest", 1.5),
        ("billion", 1.0), ("bilh", 1.0), ("million", 0.5), ("milh", 0.5),
        ("quadrillion", 1.5), ("brain", 1.0), ("cerebro", 1.0), ("dna", 1.0),
        ("telescope", 1.0), ("telescopio", 1.0), ("planet", 1.0), ("planeta", 1.0),
    ),
    "news": (
        ("launch", 1.5), ("lanca", 1.5), ("launched", 1.5), ("announces", 1.5),
        ("anuncia", 1.5), ("released", 1.5), ("release", 1.0), ("introduces", 1.5),
        ("apresenta", 1.0), ("new ", 0.5), ("novo", 0.5), ("nova ", 0.5),
        ("update", 1.0), ("atualizacao", 1.0), ("now ", 1.0), ("agora", 1.0),
        ("raises", 1.0), ("acquires", 1.0), ("compra", 1.0), ("open-sources", 1.5),
        ("available", 1.0), ("disponivel", 1.0), ("chega", 1.0), ("beta", 1.0),
    ),
}

# A fonte tambem diz algo: o arquivo historico so traz historia; ciencia de
# release academico tende a curiosidade; o Hugging Face em alta e noticia de
# modelo. Prior, nao veredito: soma a pista lexica.
_PRIOR_FONTE: dict[str, dict[str, float]] = {
    "wikipedia_onthisday": {"historia": 4.0},
    "arquivo": {"historia": 3.0, "fato": 1.0},
    "huggingface": {"news": 2.0},
    "hacker_news": {"news": 0.5},
    "rss_ciencia": {"fato": 1.5},
    "rss_tech_br": {"news": 1.0},
    "rss_tech": {"news": 1.0},
}

# Versao de modelo/produto no titulo ("GPT-6", "27B", "V4.1") e marca forte
# de lancamento.
_VERSAO = re.compile(r"\b(?:v?\d+(?:\.\d+)+|\d+b|gpt-?\d|[a-z]+-\d+(?:\.\d+)?)\b")


@dataclass(frozen=True)
class PillarGuess:
    pillar: str
    score: float
    reason: str


def classify(topic: str, source: str = "") -> list[PillarGuess]:
    """Pilares em ordem de encaixe, cada um com o motivo. Nunca vazio.

    Sem pista nenhuma, o tema e noticia: e o que o radar traz por construcao.
    """
    texto = f" {strip_accents(topic).lower()} "
    pontos: dict[str, float] = {p: 0.0 for p in PILARES}
    motivos: dict[str, list[str]] = {p: [] for p in PILARES}

    for pilar, pistas in _PISTAS.items():
        for pista, peso in pistas:
            if pista in texto:
                pontos[pilar] += peso
                motivos[pilar].append(pista.strip())
    for pilar, peso in _PRIOR_FONTE.get(source, {}).items():
        pontos[pilar] += peso
        motivos[pilar].append(f"fonte {source}")
    if _VERSAO.search(texto):
        pontos["news"] += 1.0
        motivos["news"].append("versao/modelo no titulo")

    if all(v == 0 for v in pontos.values()):
        return [PillarGuess("news", 0.1, "sem pista: noticia por padrao do radar")]

    ordenados = sorted(PILARES, key=lambda p: (-pontos[p], PILARES.index(p)))
    return [
        PillarGuess(p, round(pontos[p], 2),
                    "pistas: " + ", ".join(motivos[p][:4]) if motivos[p] else "sem pista")
        for p in ordenados if pontos[p] > 0
    ]


def best(topic: str, source: str = "") -> PillarGuess:
    return classify(topic, source)[0]


__all__ = ["PILARES", "PillarGuess", "best", "classify"]
