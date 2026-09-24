"""Que formato sirve a este tema: vídeo largo, corto o carrusel.

La pregunta del autor fue: "el agente debe saber la prioridad de formato según
la información que recibe". Aquí la respuesta es una nota por formato con tres
partes, cada una grabada en el motivo:

1. **lo que la información aguanta** (`fit`, medido en el dossier): dato único
   con número fuerte cabe en 15s; cinco datos con fecha y fuente sostienen
   60-90s; cuatro elementos paralelos (números, pasos) se vuelven cinco
   diapositivas. Dossier con menos de 3 datos NO se convierte en largo ni
   carrusel -- es regla, no nota.
2. **lo que el tipo de contenido pide** (`affinity`): tutorial y comparación
   se guardan (carrusel); historia y análisis piden arco (largo); curiosidad
   y noticia caliente caben en el corto.
3. **lo que la hora favorece** (`slot`): hipótesis declarada en
   `editorial/slots.py`.

Por encima, dos correcciones: formato ya usado hoy pierde puntos (variedad
para la audiencia Y muestra para la eval) y el rendimiento MEDIDO del canal
entra cuando exista.

Desde 20/09/2026 la parrilla de los cuatro slots **fuerza** el formato (corto
por la mañana y por la tarde, largo a mediodía y de noche), así que la nota de
abajo solo desempata dentro de lo que la hora permite -- y la penalización de
repetición no cambia resultado ninguno cuando hay un formato solo permitido.
Sigue valiendo para `slot-extra` y para el día en que la parrilla vuelva a ser
libre.

Nada de esto dice qué se viraliza en TikTok: no hay API pública para eso. Los
pesos 2 y 3 son priors declarados; el de rendimiento es el único medido, y es
el que debe crecer con el tiempo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.models import Dossier

FORMATS = ("long", "short", "carousel")

# Orden forzada por hora (decisión del autor en 20/09/2026, segunda versión):
# cuatro posts por día, todos VÍDEO, alternando corto y largo. El carrusel
# salió de la parrilla -- sigue implementado y alcanzable por
# `slot-extra --format carrusel`, pero ya no ocupa hora.
#
# El segundo elemento es emergencia, no alternativa: dossier con menos de 3
# datos no sostiene un largo (`MIN_HECHOS`), y en ese caso el slot sale corto
# en vez de fallar -- y el motivo grabado dice que fue emergencia. Los slots
# cortos no necesitan reserva: corto exige 1 dato, que es el mínimo que el
# investigador entrega.
FORMATO_FORZADO: dict[str, tuple[str, ...]] = {
    "1000": ("short",),
    "1300": ("long", "short"),
    "1700": ("short",),
    "2000": ("long", "short"),
}

# Mínimo de hechos por formato. Por debajo de 3 el guionista ya rechaza largo
# y carrusel (writer.MIN_HECHOS_PARA_GUION); aquí la regla solo queda
# explícita.
MIN_HECHOS = {"long": 3, "short": 1, "carousel": 3}

# Prior por hora (ver slots.py). Suma 1 en cada línea.
SLOT_PRIOR: dict[str, dict[str, float]] = {
    "1000": {"short": 0.55, "long": 0.30, "carousel": 0.15},
    "1300": {"long": 0.60, "short": 0.25, "carousel": 0.15},
    "1700": {"short": 0.50, "long": 0.30, "carousel": 0.20},
    "2000": {"long": 0.60, "short": 0.25, "carousel": 0.15},
}
# Prior de quien no está en la parrilla (`slot-extra`, o un slot nuevo antes
# de ganar línea propia). Era `SLOT_PRIOR["1500"]` escrito a mano, y ese id
# dejó de existir cuando la parrilla se volvió 09/12/16/19 -- un `KeyError`
# esperando el primer `slot-extra`.
PRIOR_POR_DEFECTO: dict[str, float] = {"short": 0.40, "long": 0.35, "carousel": 0.25}

# Afinidad tipo de contenido x formato. Suma 1 en cada línea.
AFFINITY: dict[str, dict[str, float]] = {
    "news": {"short": 0.50, "long": 0.35, "carousel": 0.15},
    "dato": {"short": 0.55, "carousel": 0.25, "long": 0.20},
    "analisis": {"long": 0.60, "carousel": 0.25, "short": 0.15},
    "tutorial": {"carousel": 0.60, "long": 0.30, "short": 0.10},
    "futuro": {"long": 0.45, "short": 0.35, "carousel": 0.20},
    "vs": {"carousel": 0.50, "short": 0.30, "long": 0.20},
    "historia": {"long": 0.60, "carousel": 0.20, "short": 0.20},
}

PESO_SLOT = 0.30
PESO_AFINIDAD = 0.30
PESO_ENCAJE = 0.40
PENALIZACION_REPETIDO = 0.25
# Rendimiento medido solo entra con muestra: menos que eso por formato es ruido.
MUESTRA_MINIMA = 3
TECHO_RENDIMIENTO = 0.15

_ANO = re.compile(r"\b(1[6-9]\d\d|20\d\d)\b")
_DIGITO = re.compile(r"\d")
_PASO = re.compile(
    r"\b(paso|etapa|step|primer|segundo|tercer|instal|configur|ejecut|lanza|usa |"
    r"usar|pulsa|escribe|comando|command|activa|habilita)", re.IGNORECASE)


@dataclass(frozen=True)
class DossierFeatures:
    facts: int
    with_numbers: int
    domains: int
    steps: int
    dated: int

    @property
    def listness(self) -> float:
        """0-1: cuánto los hechos se comportan como elementos de lista."""
        return min(1.0, (self.with_numbers + self.steps) / 4)


def features(dossier: Dossier) -> DossierFeatures:
    claims = [f"{f.claim} {f.quote}" for f in dossier.facts]
    dominios = {str(f.source_url).split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")
                for f in dossier.facts}
    return DossierFeatures(
        facts=len(dossier.facts),
        with_numbers=sum(1 for f in dossier.facts if _DIGITO.search(f.claim)),
        domains=len(dominios),
        steps=sum(1 for c in claims if _PASO.search(c)),
        dated=sum(1 for f in dossier.facts if _ANO.search(f.claim)),
    )


def fit(feat: DossierFeatures, formato: str) -> float:
    """Cuánto la información sostiene el formato, de 0 a 1."""
    if feat.facts < MIN_HECHOS[formato]:
        return 0.0
    if formato == "long":
        return round(0.5 * min(1.0, feat.facts / 5)
                     + 0.25 * min(1.0, feat.domains / 2)
                     + 0.25 * (1.0 if feat.dated or feat.facts >= 5 else 0.4), 3)
    if formato == "carousel":
        return round(0.5 * min(1.0, feat.facts / 4) + 0.5 * feat.listness, 3)
    # short: un dato fuerte basta; el número es lo que retiene en 15s.
    return round(0.6 + (0.4 if feat.with_numbers else 0.12), 3)


@dataclass
class FormatDecision:
    format: str
    scores: dict[str, float]
    parts: dict[str, dict[str, float]] = field(default_factory=dict)
    reason: str = ""

    def ranked(self) -> list[str]:
        return sorted(self.scores, key=lambda f: -self.scores[f])


def choose_format(feat: DossierFeatures, pillar: str, slot_id: str,
                  used_today: list[str] | None = None,
                  performance: dict[str, float] | None = None,
                  allowed: tuple[str, ...] = FORMATS,
                  en_orden: bool = False) -> FormatDecision:
    """Formato con la nota más alta; el motivo dice de dónde salió cada parte.

    Con `en_orden=True` la lista `allowed` deja de ser un conjunto de
    permitidos y pasa a ser **preferencia**: vale el primero que el dossier
    sostiene, y los siguientes solo entran como emergencia. La nota sigue
    calculándose y grabándose en el motivo -- simplemente no decide.

    Por qué existe: `FORMATO_FORZADO` siempre se describió como orden ("el
    segundo elemento es emergencia, no alternativa") y nunca lo fue. La
    función elegía por nota dentro del permitido, y eso pasó desapercibido
    mientras el único slot forzado era el carrusel de las 20h, que puntuaba
    alto solo. Cuando la parrilla de 20/09/2026 forzó `("long", "short")` en
    el mediodía y la noche, el corto pasaba a ganar al largo en pilar de
    noticia (afinidad 0,50 contra 0,35) y los dos slots largos del día saldrían
    cortos -- exactamente lo contrario de lo que la parrilla pide. Descubierto
    revisando la parrilla después de montarla, no por test.
    """
    usados = list(used_today or [])
    rendimiento = performance or {}
    prior_slot = SLOT_PRIOR.get(slot_id, PRIOR_POR_DEFECTO)
    afinidad = AFFINITY.get(pillar, AFFINITY["news"])

    notas: dict[str, float] = {}
    partes: dict[str, dict[str, float]] = {}
    for f in allowed:
        encaje = fit(feat, f)
        if encaje == 0.0:
            continue
        p = {
            "slot": round(PESO_SLOT * prior_slot.get(f, 0.0), 3),
            "tipo": round(PESO_AFINIDAD * afinidad.get(f, 0.0), 3),
            "encaje": round(PESO_ENCAJE * encaje, 3),
            "repetido": round(-PENALIZACION_REPETIDO * usados.count(f), 3),
            "medido": round(max(-TECHO_RENDIMIENTO,
                                min(TECHO_RENDIMIENTO, rendimiento.get(f, 0.0))), 3),
        }
        partes[f] = p
        notas[f] = round(sum(p.values()), 3)

    if not notas:
        # Solo llega aquí con dossier sin hechos (el investigador no entrega
        # eso) o `allowed` vacío. Corto es el formato que menos exige.
        return FormatDecision("short", {"short": 0.0}, {},
                              "ningún formato sostenido por el dossier; corto por defecto")

    if en_orden:
        elegido = next((f for f in allowed if f in notas), None)
        if elegido is None:
            elegido = max(notas, key=lambda f: (notas[f], -FORMATS.index(f)))
    else:
        elegido = max(notas, key=lambda f: (notas[f], -FORMATS.index(f)))
    p = partes[elegido]
    otros = ", ".join(f"{f} {notas[f]:.2f}" for f in sorted(notas, key=lambda x: -notas[x])
                      if f != elegido)
    medido = ("rendimiento medido del canal entró" if any(rendimiento.values())
              else "sin métrica propia con muestra aún: los pesos son prior declarado")
    reason = (
        f"{elegido} ({notas[elegido]:.2f}) = horario {p['slot']:+.2f}, "
        f"tipo {pillar} {p['tipo']:+.2f}, dossier {p['encaje']:+.2f} "
        f"({feat.facts} hechos, {feat.with_numbers} con número, {feat.domains} "
        f"fuente(s), {feat.steps} paso(s), {feat.dated} con fecha)"
        + (f", repetido hoy {p['repetido']:+.2f}" if p["repetido"] else "")
        + (f", medido {p['medido']:+.2f}" if p["medido"] else "")
        + (f"; otros: {otros}" if otros else "")
        + f"; {medido}"
    )
    return FormatDecision(elegido, notas, partes, reason)


def performance_from_metrics(rows: list[dict]) -> dict[str, float]:
    """Bonus por formato a partir de las métricas leídas en la app, con muestra.

    `rows`: una línea por post con `format`, `views` y `completion_rate`
    (la última recolección de cada publish_id). Cada formato recibe la
    diferencia relativa de su mediana de conclusión respecto a la mediana
    general, limitada a +-TECHO_RENDIMIENTO. Formato con menos de
    MUESTRA_MINIMA posts queda en 0: dos vídeos buenos no son tendencia.
    """
    por_formato: dict[str, list[float]] = {}
    for r in rows:
        tasa = r.get("completion_rate")
        if tasa is None or r.get("format") not in FORMATS:
            continue
        por_formato.setdefault(r["format"], []).append(float(tasa))
    todos = [v for vs in por_formato.values() for v in vs]
    if not todos:
        return {}
    general = _mediana(todos)
    salida: dict[str, float] = {}
    for f, valores in por_formato.items():
        if len(valores) < MUESTRA_MINIMA or general <= 0:
            continue
        mediana = _mediana(valores)
        salida[f] = round(max(-TECHO_RENDIMIENTO, min(TECHO_RENDIMIENTO,
                                                      (mediana - general) / general)), 3)
    return salida


def _mediana(valores: list[float]) -> float:
    ordenados = sorted(valores)
    medio = len(ordenados) // 2
    if len(ordenados) % 2:
        return ordenados[medio]
    return (ordenados[medio - 1] + ordenados[medio]) / 2


__all__ = ["AFFINITY", "FORMATS", "FORMATO_FORZADO", "MIN_HECHOS", "PRIOR_POR_DEFECTO",
           "SLOT_PRIOR", "DossierFeatures",
           "FormatDecision", "choose_format", "features", "fit",
           "performance_from_metrics"]
