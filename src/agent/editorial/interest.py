"""Interesse do publico: o radar mede movimento, nao apelo.

O score do curador sabe que um tema esta subindo (pontos por hora, veiculos
cobrindo, curtidas) e que e do nicho. Nao sabe se ele prende o publico
brasileiro GERAL do TikTok: "Bend, uma linguagem que bloqueia erro de IA com
prova formal" sobe rapido no Hacker News e interessa a programador; "Como
usar agentes de IA para automatizar planilhas" interessa a quase todo mundo.

Uma chamada de modelo POR SLOT (nao por tema) da nota de 0 a 10 aos ~10
melhores do radar, com motivo curto. A nota entra combinada com o score
deterministico -- nunca sozinha: o modelo nao ve a velocidade real, e o
radar nao ve o apelo. Sem modelo disponivel, vale so o radar (e o motivo
diz isso).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.editorial.planner import TopicChoice
from agent.ports.llm import LLM, LLMError, parse_json_object

PESO_INTERESSE = 0.5
MAX_CANDIDATOS = 10

SISTEMA = (
    "Voce e editor de pauta de um canal brasileiro de tecnologia e IA no TikTok, "
    "para publico geral de 18 a 35 anos. Voce sabe o que faz alguem parar de rolar "
    "o feed. Responde so com o JSON pedido."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "notas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "motivo": {"type": "string"},
                    "interesse": {"type": "integer"},
                },
                "required": ["i", "motivo", "interesse"],
            },
        }
    },
    "required": ["notas"],
}


@dataclass
class InterestReport:
    applied: bool
    scores: dict[int, tuple[int, str]]
    note: str = ""


def build_prompt(escolhas: list[TopicChoice]) -> str:
    linhas = "\n".join(f"[{i}] ({c.pillar}) {c.decision.term}"
                       for i, c in enumerate(escolhas))
    return (
        f"TEMAS EM ALTA AGORA (tipo de conteudo entre parenteses):\n{linhas}\n\n"
        "Para CADA tema, de uma nota de 0 a 10 para o quanto ele prende a atencao "
        "do publico brasileiro geral hoje. Pese: impacto no dia a dia de quem "
        "assiste, surpresa, nome conhecido (Google, WhatsApp, ChatGPT...), e se da "
        "para explicar em 60 segundos. Tire pontos de: tema so para programador, "
        "jargao sem traducao, assunto que ja saturou, politica partidaria, guia de "
        "compra, promocao ou produto financeiro. O canal e de IA e tecnologia: tema "
        "de IA, ciencia ou tecnologia que muda o dia a dia vale mais.\n"
        "Motivo em ate 12 palavras, escrito antes da nota. Devolva "
        '{"notas": [{"i": 0, "motivo": "...", "interesse": 7}, ...]} com todos os indices.'
    )


def rate(escolhas: list[TopicChoice], llm: LLM) -> InterestReport:
    candidatos = escolhas[:MAX_CANDIDATOS]
    if not candidatos:
        return InterestReport(False, {}, "sem candidatos")
    try:
        resposta = llm.complete(build_prompt(candidatos), system=SISTEMA, schema=SCHEMA,
                                temperature=0.2, max_output_tokens=1200)
        corpo = parse_json_object(resposta.text)
    except LLMError as exc:
        return InterestReport(False, {}, f"nota de interesse indisponivel: {exc}"[:200])
    notas: dict[int, tuple[int, str]] = {}
    for item in corpo.get("notas") or []:
        if not isinstance(item, dict):
            continue
        i, nota = item.get("i"), item.get("interesse")
        if (isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(candidatos)
                and isinstance(nota, int) and not isinstance(nota, bool)):
            notas[i] = (max(0, min(10, nota)), " ".join(str(item.get("motivo", "")).split())[:120])
    if len(notas) < len(candidatos) // 2:
        return InterestReport(False, notas, "nota de interesse incompleta; vale o radar")
    return InterestReport(True, notas)


def rerank(escolhas: list[TopicChoice], llm: LLM) -> tuple[list[TopicChoice], str]:
    """Reordena combinando radar+horario com o interesse; devolve o motivo."""
    rel = rate(escolhas, llm)
    if not rel.applied:
        return escolhas, rel.note
    reordenadas: list[TopicChoice] = []
    for i, c in enumerate(escolhas):
        if i in rel.scores:
            nota, motivo = rel.scores[i]
            combinado = round((1 - PESO_INTERESSE) * c.score + PESO_INTERESSE * nota / 10, 4)
            c.reason += f"; interesse {nota}/10 ({motivo})"
            c.score = combinado
        reordenadas.append(c)
    reordenadas.sort(key=lambda c: -c.score)
    return reordenadas, f"interesse aplicado a {len(rel.scores)} temas"


__all__ = ["InterestReport", "build_prompt", "rate", "rerank"]
