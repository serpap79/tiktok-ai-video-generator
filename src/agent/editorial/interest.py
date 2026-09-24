"""Interés de la audiencia: el radar mide movimiento, no atractivo.

El score del curador sabe que un tema está subiendo (puntos por hora, medios
cubriéndolo, me gusta) y que es del nicho. No sabe si retiene al público
GENERAL de España en TikTok: "Bend, un lenguaje que bloquea errores de IA con
prueba formal" sube rápido en Hacker News e interesa a programadores; "Cómo
usar agentes de IA para automatizar hojas de cálculo" interesa a casi todo el
mundo.

Una llamada de modelo POR SLOT (no por tema) da la nota de 0 a 10 a los ~10
mejores del radar, con motivo corto. La nota entra combinada con el score
determinista -- nunca sola: el modelo no ve la velocidad real, y el radar no
ve el atractivo. Sin modelo disponible, vale solo el radar (y el motivo lo
dice).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.editorial.planner import TopicChoice
from agent.ports.llm import LLM, LLMError, parse_json_object

PESO_INTERES = 0.5
MAX_CANDIDATOS = 10

SISTEMA = (
    "Eres editor de pauta de un canal español de tecnología e IA en TikTok, "
    "para público general de 18 a 35 años. Sabes qué hace que alguien deje de "
    "deslizar el feed. Respondes solo con el JSON pedido."
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
                    "interes": {"type": "integer"},
                },
                "required": ["i", "motivo", "interes"],
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


def build_prompt(elecciones: list[TopicChoice]) -> str:
    lineas = "\n".join(f"[{i}] ({c.pillar}) {c.decision.term}"
                       for i, c in enumerate(elecciones))
    return (
        f"TEMAS EN ALTA AHORA (tipo de contenido entre paréntesis):\n{lineas}\n\n"
        "Para CADA tema, da una nota de 0 a 10 sobre cuánto retiene la atención "
        "del público español general hoy. Pondera: impacto en el día a día de "
        "quien mira, sorpresa, nombre conocido (Google, WhatsApp, ChatGPT...), "
        "y si se puede explicar en 60 segundos. Quita puntos a: tema solo para "
        "programadores, jerga sin traducción, asunto ya saturado, política "
        "partidista, guía de compra, promoción o producto financiero. El canal "
        "es de IA y tecnología: un tema de IA, ciencia o tecnología que cambia "
        "el día a día vale más.\n"
        "Motivo en hasta 12 palabras, escrito antes de la nota. Devuelve "
        '{"notas": [{"i": 0, "motivo": "...", "interes": 7}, ...]} con todos '
        "los índices."
    )


def rate(elecciones: list[TopicChoice], llm: LLM) -> InterestReport:
    candidatos = elecciones[:MAX_CANDIDATOS]
    if not candidatos:
        return InterestReport(False, {}, "sin candidatos")
    try:
        respuesta = llm.complete(build_prompt(candidatos), system=SISTEMA, schema=SCHEMA,
                                 temperature=0.2, max_output_tokens=1200)
        cuerpo = parse_json_object(respuesta.text)
    except LLMError as exc:
        return InterestReport(False, {}, f"nota de interés no disponible: {exc}"[:200])
    notas: dict[int, tuple[int, str]] = {}
    for item in cuerpo.get("notas") or []:
        if not isinstance(item, dict):
            continue
        i, nota = item.get("i"), item.get("interes")
        if (isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(candidatos)
                and isinstance(nota, int) and not isinstance(nota, bool)):
            notas[i] = (max(0, min(10, nota)), " ".join(str(item.get("motivo", "")).split())[:120])
    if len(notas) < len(candidatos) // 2:
        return InterestReport(False, notas, "nota de interés incompleta; vale el radar")
    return InterestReport(True, notas)


def rerank(elecciones: list[TopicChoice], llm: LLM) -> tuple[list[TopicChoice], str]:
    """Reordena combinando radar+horario con el interés; devuelve el motivo."""
    rel = rate(elecciones, llm)
    if not rel.applied:
        return elecciones, rel.note
    reordenadas: list[TopicChoice] = []
    for i, c in enumerate(elecciones):
        if i in rel.scores:
            nota, motivo = rel.scores[i]
            combinado = round((1 - PESO_INTERES) * c.score + PESO_INTERES * nota / 10, 4)
            c.reason += f"; interés {nota}/10 ({motivo})"
            c.score = combinado
        reordenadas.append(c)
    reordenadas.sort(key=lambda c: -c.score)
    return reordenadas, f"interés aplicado a {len(rel.scores)} temas"


__all__ = ["InterestReport", "build_prompt", "rate", "rerank"]
