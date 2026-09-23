"""Planejamento do slot: qual tema, com qual tipo de conteudo, antes de pesquisar.

O curador ja diz "o que esta subindo" (velocidade, volume, nicho). O
planejador cruza isso com o horario e com o que o dia ja publicou, porque o
melhor tema do radar as 9h nao e necessariamente o melhor tema para as 20h:
de noite o slot pede arco (historia, analise), de manha pede o fato do dia.

O formato NAO e decidido aqui: ele depende do que a pesquisa encontrar
(`editorial/formats.py`), e decidir antes seria prometer um video longo para
um dossie de dois fatos.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.editorial.content import PillarGuess, classify
from agent.models import Decision

# Preferencia de tipo por horario, 0-1. Mesma natureza de SLOT_PRIOR: hipotese
# declarada, gravada no motivo de cada escolha.
SLOT_PILLAR_PREF: dict[str, dict[str, float]] = {
    "0900": {"news": 1.0, "fato": 0.85, "futuro": 0.6, "vs": 0.55, "analise": 0.5,
             "tutorial": 0.45, "historia": 0.35},
    "1500": {"tutorial": 1.0, "vs": 0.9, "fato": 0.75, "news": 0.6, "futuro": 0.5,
             "analise": 0.5, "historia": 0.5},
    "2000": {"historia": 1.0, "analise": 0.95, "futuro": 0.85, "news": 0.6, "fato": 0.6,
             "vs": 0.5, "tutorial": 0.45},
}

PESO_RADAR = 0.6
PESO_HORARIO = 0.4
PENALIDADE_TIPO_REPETIDO = 0.15
# Palpite alternativo so e aproveitado se tiver pelo menos esta fracao da
# nota do melhor -- trocar "tutorial" forte por "news" fraco so para variar
# faria o roteiro seguir uma formula que o tema nao sustenta.
FRACAO_ALTERNATIVA = 0.6


@dataclass
class TopicChoice:
    decision: Decision
    pillar: str
    score: float
    reason: str
    guesses: list[PillarGuess]


def choose_pillar(guesses: list[PillarGuess], used_today: list[str]) -> PillarGuess:
    """O melhor palpite, trocando por um alternativo forte se o melhor ja saiu hoje."""
    melhor = guesses[0]
    if melhor.pillar not in used_today:
        return melhor
    for g in guesses[1:]:
        if g.pillar not in used_today and g.score >= FRACAO_ALTERNATIVA * melhor.score:
            return g
    return melhor


def rank_topics(decisions: list[Decision], slot_id: str,
                pillars_used_today: list[str] | None = None,
                limit: int = 8) -> list[TopicChoice]:
    """Candidatos elegiveis do curador reordenados para este horario."""
    usados = list(pillars_used_today or [])
    preferencia = SLOT_PILLAR_PREF.get(slot_id, SLOT_PILLAR_PREF["1500"])
    saida: list[TopicChoice] = []
    for d in decisions:
        palpites = classify(d.term, d.source)
        pilar = choose_pillar(palpites, usados)
        pref = preferencia.get(pilar.pillar, 0.5)
        repetido = PENALIDADE_TIPO_REPETIDO if pilar.pillar in usados else 0.0
        nota = round(PESO_RADAR * d.score + PESO_HORARIO * pref - repetido, 4)
        motivo = (f"radar {d.score:.2f} x{PESO_RADAR} + horario {slot_id} gosta de "
                  f"{pilar.pillar} ({pref:.2f}) x{PESO_HORARIO}"
                  + (f" - tipo ja usado hoje ({repetido:.2f})" if repetido else "")
                  + f"; tipo {pilar.pillar} por {pilar.reason}")
        saida.append(TopicChoice(d, pilar.pillar, nota, motivo, palpites))
    saida.sort(key=lambda c: -c.score)
    return saida[:limit]


__all__ = ["SLOT_PILLAR_PREF", "TopicChoice", "choose_pillar", "rank_topics"]
