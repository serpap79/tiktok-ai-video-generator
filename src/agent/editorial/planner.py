"""Planificacion del slot: que tema, con que tipo de contenido, antes de investigar.

El curador ya dice "que esta subiendo" (velocidad, volumen, nicho). El
planificador cruza eso con la hora y con lo que el dia ya ha publicado, porque
el mejor tema del radar a las 9h no es necesariamente el mejor tema para las
20h: por la noche el slot pide arco (historia, analisis), por la manana pide
el dato del dia.

El formato NO se decide aqui: depende de lo que la investigacion encuentre
(`editorial/formats.py`), y decidir antes seria prometer un video largo para
un dossier de dos hechos.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.editorial.content import PillarGuess, classify
from agent.models import Decision

# Preferencia de tipo por horario, 0-1. Misma naturaleza de SLOT_PRIOR: hipotesis
# declarada, grabada en el motivo de cada eleccion.
SLOT_PILLAR_PREF: dict[str, dict[str, float]] = {
    "0900": {"news": 1.0, "dato": 0.85, "futuro": 0.6, "vs": 0.55, "analisis": 0.5,
             "tutorial": 0.45, "historia": 0.35},
    "1500": {"tutorial": 1.0, "vs": 0.9, "dato": 0.75, "news": 0.6, "futuro": 0.5,
             "analisis": 0.5, "historia": 0.5},
    "2000": {"historia": 1.0, "analisis": 0.95, "futuro": 0.85, "news": 0.6, "dato": 0.6,
             "vs": 0.5, "tutorial": 0.45},
}

PESO_RADAR = 0.6
PESO_HORARIO = 0.4
PENALIZACION_TIPO_REPETIDO = 0.15
# La alternativa solo se aprovecha si tiene al menos esta fraccion de la nota
# del mejor -- cambiar "tutorial" fuerte por "news" debil solo para variar
# haria que el guion siga una formula que el tema no sostiene.
FRACCION_ALTERNATIVA = 0.6


@dataclass
class TopicChoice:
    decision: Decision
    pillar: str
    score: float
    reason: str
    guesses: list[PillarGuess]


def choose_pillar(guesses: list[PillarGuess], used_today: list[str]) -> PillarGuess:
    """La mejor apuesta, cambiando por una alternativa fuerte si la mejor ya salio hoy."""
    mejor = guesses[0]
    if mejor.pillar not in used_today:
        return mejor
    for g in guesses[1:]:
        if g.pillar not in used_today and g.score >= FRACCION_ALTERNATIVA * mejor.score:
            return g
    return mejor


def rank_topics(decisions: list[Decision], slot_id: str,
                pillars_used_today: list[str] | None = None,
                limit: int = 8) -> list[TopicChoice]:
    """Candidatos elegibles del curador reordenados para esta hora."""
    usados = list(pillars_used_today or [])
    preferencia = SLOT_PILLAR_PREF.get(slot_id, SLOT_PILLAR_PREF["1500"])
    salida: list[TopicChoice] = []
    for d in decisions:
        apuestas = classify(d.term, d.source)
        pilar = choose_pillar(apuestas, usados)
        pref = preferencia.get(pilar.pillar, 0.5)
        repetido = PENALIZACION_TIPO_REPETIDO if pilar.pillar in usados else 0.0
        nota = round(PESO_RADAR * d.score + PESO_HORARIO * pref - repetido, 4)
        motivo = (f"radar {d.score:.2f} x{PESO_RADAR} + horario {slot_id} le gusta "
                  f"{pilar.pillar} ({pref:.2f}) x{PESO_HORARIO}"
                  + (f" - tipo ya usado hoy ({repetido:.2f})" if repetido else "")
                  + f"; tipo {pilar.pillar} por {pilar.reason}")
        salida.append(TopicChoice(d, pilar.pillar, nota, motivo, apuestas))
    salida.sort(key=lambda c: -c.score)
    return salida[:limit]


__all__ = ["SLOT_PILLAR_PREF", "TopicChoice", "choose_pillar", "rank_topics"]
