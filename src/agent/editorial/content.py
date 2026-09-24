"""Tipo de contenido de un tema: noticia, curiosidad, analisis, tutorial...

Los tipos son los pilares de la marca (`brand/brand.json`), y cada uno lleva
su propia fórmula de gancho, la batida y el CTA -- y eso es lo que hace que
"informativo", "curiosidad", "tutorial" e "historia" salgan diferentes, y no
el mismo guion con etiqueta cambiada.

La clasificación es léxica y determinista, a propósito: corre antes de la
investigación, para CADA candidato del radar, y gastar una llamada de modelo
por candidato solo para etiquetar quemaría la cuota que la escrita necesita.
El coste de errar es bajo -- el pilar se convierte en orientación de prompt,
y el juez lee el texto final --, y cada apuesta lleva el motivo grabado para
calibrar después.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.text import strip_accents

PILARES = ("news", "dato", "analisis", "tutorial", "futuro", "vs", "historia")

# Pistas por pilar, en español e inglés (el radar es bilingüe). Peso mayor =
# pista más específica: "how to" casi solo aparece en tutorial; "new" aparece
# en todo.
_PISTAS: dict[str, tuple[tuple[str, float], ...]] = {
    "tutorial": (
        ("how to", 3.0), ("como usar", 3.0), ("como hacer", 3.0), ("paso a paso", 3.0),
        ("tutorial", 3.0), ("guide", 2.0), ("guia", 2.0), ("tips", 1.5), ("trucos", 1.5),
        ("cheat sheet", 2.5), ("step by step", 3.0), ("prompt", 1.0), ("comando", 1.5),
        ("command", 1.0), ("workflow", 1.5), ("setup", 1.0), ("configurar", 1.5),
        ("show hn", 1.5), ("open source tool", 1.5), ("cli", 1.0), ("extension", 1.0),
        ("agents.md", 2.0), ("claude.md", 2.0), ("vscode", 1.0), ("plugin", 1.0),
    ),
    "vs": (
        (" vs ", 3.0), (" vs. ", 3.0), ("versus", 3.0), (" x ", 1.5), ("compared", 2.0),
        ("comparison", 2.0), ("comparacion", 2.0), ("beats", 1.5), ("outperforms", 2.0),
        ("supera", 2.0), ("mejor que", 2.0), ("better than", 2.0), ("faster than", 1.5),
        ("mas rapido que", 1.5), ("duelo", 2.0), ("against", 1.0),
    ),
    "historia": (
        ("history", 2.5), ("historia", 2.5), ("anniversary", 2.5), ("aniversario", 2.5),
        ("years ago", 2.5), ("anos atras", 2.5), ("invented", 2.0), ("invento", 2.0),
        ("invencion", 2.0), ("first ever", 2.0), ("el primero", 1.5), ("la primera", 1.5),
        ("origin", 1.5), ("origen", 1.5), ("decada", 1.5), ("legacy", 1.0),
        ("museum", 1.5), ("vintage", 1.5), ("retro", 1.0), ("1940", 2.0), ("1950", 2.0),
        ("1960", 2.0), ("1970", 2.0), ("1980", 1.5), ("1990", 1.0),
    ),
    "futuro": (
        ("future", 2.0), ("futuro", 2.0), ("2030", 3.0), ("2035", 3.0), ("2040", 3.0),
        ("2050", 3.0), ("next decade", 2.5), ("proxima decada", 2.5), ("predict", 2.0),
        ("prediccion", 2.0), ("roadmap", 1.5), ("will replace", 2.0), ("va a substituir", 2.0),
        ("agi", 1.5), ("superintelligence", 2.0), ("forecast", 1.5), ("va a sustituir", 2.0),
    ),
    "analisis": (
        ("why ", 1.5), ("por que", 1.5), ("i don't", 2.0), ("i do not", 2.0),
        ("the case for", 2.5), ("the case against", 2.5), ("should", 1.0),
        ("myth", 2.0), ("mito", 2.0), ("problem with", 2.0), ("problema", 1.0),
        ("is dead", 2.0), ("overrated", 2.0), ("bubble", 2.0), ("burbuja", 2.0),
        ("ethic", 1.5), ("etica", 1.5), ("jobs", 1.5), ("empleos", 1.5),
        ("regulation", 1.5), ("regulacion", 1.5), ("ley ", 1.0), ("lawsuit", 1.5),
        ("demanda", 1.5), ("robo", 1.5), ("debate", 1.5), ("risk", 1.0),
        ("riesgo", 1.0), ("esconder", 1.5), ("oculto", 1.5), ("paso de todo", 1.5),
    ),
    "dato": (
        ("study", 2.0), ("estudio", 2.0), ("researchers", 2.0), ("investigadores", 2.0),
        ("scientists", 2.0), ("cientificos", 2.0), ("discover", 2.0), ("descubr", 2.0),
        ("found that", 1.5), ("record", 1.5), ("record", 1.5), ("largest", 1.5),
        ("mayor ", 1.0), ("smallest", 1.5), ("menor ", 1.0), ("fastest", 1.5),
        ("billion", 1.0), ("mill", 1.0), ("million", 0.5), ("millon", 0.5),
        ("quadrillion", 1.5), ("brain", 1.0), ("cerebro", 1.0), ("dna", 1.0),
        ("telescope", 1.0), ("telescopio", 1.0), ("planet", 1.0), ("planeta", 1.0),
    ),
    "news": (
        ("launch", 1.5), ("lanza", 1.5), ("launched", 1.5), ("announces", 1.5),
        ("anuncia", 1.5), ("released", 1.5), ("release", 1.0), ("introduces", 1.5),
        ("presenta", 1.0), ("new ", 0.5), ("nuevo", 0.5), ("nueva ", 0.5),
        ("update", 1.0), ("actualizacion", 1.0), ("now ", 1.0), ("ahora", 1.0),
        ("raises", 1.0), ("acquires", 1.0), ("compra", 1.0), ("open-sources", 1.5),
        ("available", 1.0), ("disponible", 1.0), ("llega", 1.0), ("beta", 1.0),
    ),
}

# La fuente también dice algo: el archivo histórico solo trae historia; la
# ciencia de release académico tiende a curiosidad; el Hugging Face en alta es
# noticia de modelo. Prior, no veredicto: suma a la pista léxica.
_PRIOR_FUENTE: dict[str, dict[str, float]] = {
    "wikipedia_onthisday": {"historia": 4.0},
    "arquivo": {"historia": 3.0, "dato": 1.0},
    "huggingface": {"news": 2.0},
    "hacker_news": {"news": 0.5},
    "rss_ciencia": {"dato": 1.5},
    "rss_tech_br": {"news": 1.0},
    "rss_tech": {"news": 1.0},
}

# Versión de modelo/producto en el título ("GPT-6", "27B", "V4.1") es marca
# fuerte de lanzamiento.
_VERSION = re.compile(r"\b(?:v?\d+(?:\.\d+)+|\d+b|gpt-?\d|[a-z]+-\d+(?:\.\d+)?)\b")


@dataclass(frozen=True)
class PillarGuess:
    pillar: str
    score: float
    reason: str


def classify(topic: str, source: str = "") -> list[PillarGuess]:
    """Pilares en orden de encaje, cada uno con el motivo. Nunca vacío.

    Sin pista ninguna, el tema es noticia: es lo que el radar trae por
    construcción.
    """
    texto = f" {strip_accents(topic).lower()} "
    puntos: dict[str, float] = {p: 0.0 for p in PILARES}
    motivos: dict[str, list[str]] = {p: [] for p in PILARES}

    for pilar, pistas in _PISTAS.items():
        for pista, peso in pistas:
            if pista in texto:
                puntos[pilar] += peso
                motivos[pilar].append(pista.strip())
    for pilar, peso in _PRIOR_FUENTE.get(source, {}).items():
        puntos[pilar] += peso
        motivos[pilar].append(f"fuente {source}")
    if _VERSION.search(texto):
        puntos["news"] += 1.0
        motivos["news"].append("versión/modelo en el título")

    if all(v == 0 for v in puntos.values()):
        return [PillarGuess("news", 0.1, "sin pista: noticia por defecto del radar")]

    ordenados = sorted(PILARES, key=lambda p: (-puntos[p], PILARES.index(p)))
    return [
        PillarGuess(p, round(puntos[p], 2),
                    "pistas: " + ", ".join(motivos[p][:4]) if motivos[p] else "sin pista")
        for p in ordenados if puntos[p] > 0
    ]


def best(topic: str, source: str = "") -> PillarGuess:
    return classify(topic, source)[0]


__all__ = ["PILARES", "PillarGuess", "best", "classify"]
