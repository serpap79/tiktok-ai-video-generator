"""Guionista de carrusel: dossier para 5 diapositivas + caption.

Receta validada de los canales (2026): diapositiva 1 con promesa numerada,
revelación progresiva con bomba de valor en medio, diapositiva 5 con
conclusión + CTA de guardado, 10-15 palabras por diapositiva, caption con
pregunta para arrastrar comentario. Todo lo que es contable se convierte en
puerta mecánica aquí; gancho y progresión son del juez.

Mismo bucle que el guionista de vídeo: hasta 3 intentos, defecto medido de
vuelta al modelo en texto, intento reprobado grabado para calibrar el prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent.brand.brand import pillar_brief, voice_brief
from agent.brand.checks import check_emoji_muletilla, check_numbers
from agent.models import (
    CAROUSEL_MAX_WORDS_PER_SLIDE,
    CAROUSEL_SLIDES,
    Carousel,
    Dossier,
    Slide,
)
from agent.ports.llm import LLM, Completion, LLMError, Usage, parse_json_object
from agent.research.grounding import missing_numbers
from agent.research.subject import missing_subject
from agent.writer.humanize import scan as scan_tells
from agent.writer.visuals import compact_brief, suggest_pillar, validate_broll, validate_terms
from agent.writer.writer import _resolver_hechos

MAX_INTENTOS = 3

SISTEMA = (
    "Escribes guiones de carruseles de un canal español dark de tech, IA y "
    "ciencia. Cada diapositiva es una frase corta que se lee en 3 segundos. "
    "Solo afirmas lo que está en el dossier que recibes."
)

SCHEMA_CARRUSEL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "slides": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"n": {"type": "integer"},
                                     "headline": {"type": "string"},
                                     "text": {"type": "string"},
                                     "visual": {"type": "string"}},
                      "required": ["n", "headline", "text", "visual"]},
        },
        "caption": {"type": "string"},
        "broll": {"type": "array", "items": {"type": "string"}},
        "used_facts": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["slides", "caption", "broll", "used_facts"],
}

_DIGITO = re.compile(r"\d")


@dataclass
class CarouselAttempt:
    violations: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    # Las diapositivas reprobadas en texto, para que la corrección ajuste en
    # vez de rehacer.
    text: str = ""


@dataclass
class CarouselReport:
    topic: str
    carousel: Carousel | None = None
    attempts: list[CarouselAttempt] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    refusal: str = ""

    @property
    def ok(self) -> bool:
        return self.carousel is not None

    @property
    def usage(self) -> Usage:
        total = Usage()
        for a in self.attempts:
            total = total + a.usage
        return total

    @property
    def latency_s(self) -> float:
        return round(sum(a.latency_s for a in self.attempts), 3)


def write_carousel(dossier: Dossier, llm: LLM,
                   max_attempts: int = MAX_INTENTOS,
                   notes: list[str] | None = None,
                   pillar: str = "") -> CarouselReport:
    """Escribe el carrusel con autocorrección de lo que es mecánico."""
    report = CarouselReport(
        topic=dossier.topic,
        model=getattr(llm, "model", ""),
        provider=getattr(llm, "provider", ""),
    )
    from agent.writer.writer import thin_dossier_reason
    report.refusal = thin_dossier_reason(dossier, "carousel")
    if report.refusal:
        return report

    correccion: list[str] = list(notes or [])
    anterior = ""
    for _ in range(max_attempts):
        try:
            respuesta = llm.complete(
                build_prompt(dossier, correccion, pillar=pillar,
                             previous=anterior if correccion else ""),
                system=SISTEMA,
                schema=SCHEMA_CARRUSEL,
                temperature=0.6,
                max_output_tokens=2048,
            )
        except LLMError:
            raise
        report.model, report.provider = respuesta.model, respuesta.provider
        intento, carrusel = _evaluar(respuesta, dossier, pillar)
        report.attempts.append(intento)
        if not intento.violations:
            report.carousel = carrusel
            return report
        correccion = intento.violations
        anterior = intento.text or anterior
    return report


def _evaluar(respuesta: Completion, dossier: Dossier, pillar: str = ""
             ) -> tuple[CarouselAttempt, Carousel | None]:
    intento = CarouselAttempt(usage=respuesta.usage, latency_s=respuesta.latency_s)
    if respuesta.truncated:
        intento.violations.append(
            "la respuesta se cortó por límite de tokens; escribe más corto")
        return intento, None
    try:
        cuerpo = parse_json_object(respuesta.text)
    except LLMError as exc:
        intento.violations.append(f"la respuesta no llegó como objeto JSON: {exc}")
        return intento, None

    usados, fuera = _resolver_hechos(cuerpo.get("used_facts"), dossier.facts)
    try:
        slides = [Slide(n=int(s.get("n", i + 1)),
                        headline=_texto(s.get("headline")),
                        text=_texto(s.get("text")),
                        visual=_texto(s.get("visual")))
                  for i, s in enumerate(cuerpo.get("slides") or [])]
        broll = [_texto(t) for t in (cuerpo.get("broll") or []) if isinstance(t, str)]
        carrusel = Carousel(topic=dossier.topic, slides=slides,
                            caption=_texto(cuerpo.get("caption")), facts=usados,
                            broll=[t for t in broll if t][:3], pillar=pillar)
    except (ValidationError, ValueError, AttributeError) as exc:
        intento.violations.append(f"el carrusel no respeta el contrato: {exc}")
        return intento, None
    intento.text = "\n".join(
        f"[{s.n}] {s.headline} / {s.text} ~ {s.visual}" for s in carrusel.slides
    ) + f"\ncaption: {carrusel.caption}"

    intento.violations.extend(_violaciones(carrusel, dossier, fuera))
    return intento, (carrusel if not intento.violations else None)


def _violaciones(carrusel: Carousel, dossier: Dossier, fuera: list[int]) -> list[str]:
    problemas: list[str] = []
    for s in carrusel.slides:
        if s.word_count > CAROUSEL_MAX_WORDS_PER_SLIDE:
            problemas.append(
                f"la diapositiva {s.n} tiene {s.word_count} palabras (techo "
                f"{CAROUSEL_MAX_WORDS_PER_SLIDE}): corta "
                f"{s.word_count - CAROUSEL_MAX_WORDS_PER_SLIDE} palabras, "
                "la diapositiva se lee en 3 segundos.")
    s1 = carrusel.slides[0]
    if not _DIGITO.search(f"{s1.headline} {s1.text}"):
        problemas.append(
            "diapositiva 1 sin número: la promesa numerada ('5 IAs que...') es "
            "lo que hace deslizar a la gente. Sin número no hay recompensa "
            "finita.")
    s5 = carrusel.slides[-1]
    if not any(palabra in f"{s5.headline} {s5.text}".lower()
               for palabra in ("guard", "salv", "graba")):
        problemas.append(
            "diapositiva 5 sin CTA de guardado ('guárdalo'): save/view es el "
            "indicador líder del formato; sin él el carrusel no acumula "
            "distribución.")
    if "?" not in carrusel.caption:
        problemas.append(
            "caption sin pregunta: la caption lleva la invitación al "
            "comentario, y el comentario es donde el carrusel gana al vídeo.")
    if fuera:
        problemas.append(
            f"used_facts apunta a un índice que no existe en el dossier: {fuera}.")
    if not carrusel.facts:
        problemas.append("used_facts vacío: el carrusel también se ancla en fuente.")
    problemas.extend(
        "visual: " + v for v in validate_terms([s.visual for s in carrusel.slides]))
    problemas.extend(validate_broll(carrusel.broll, "carousel"))
    texto = "\n".join(f"{s.headline} {s.text}" for s in carrusel.slides)
    fuentes = "\n".join(f"{f.claim}\n{f.quote}" for f in dossier.facts)
    sin_sujeto = missing_subject(texto, dossier.topic, minimum=1)
    if sin_sujeto:
        problemas.append(
            "carrusel sin sujeto: ninguna diapositiva nombra "
            + ", ".join(f"{t!r}" for t in sin_sujeto) + ". Quien mira "
            "necesita saber sobre QUÉ son las 5 diapositivas.")
    # El recuento estructural (el "5" de la promesa) no es afirmación factual:
    # son las propias diapositivas, verificadas arriba por el orden 1-5. Solo
    # un número por encima de eso debe existir en el dossier.
    sueltos = [n for n in missing_numbers(texto, fuentes)
               if not (n.isdigit() and int(n) <= CAROUSEL_SLIDES)]
    if sueltos:
        problemas.append(
            f"la diapositiva cita un número que no está en el dossier: "
            f"{', '.join(sueltos)}.")
    tells = scan_tells(texto)
    if tells:
        problemas.append(
            "diapositiva con vicio de IA (" + "; ".join(tells[:3]) + "): "
            "reescribe como habla corta de persona.")
    problemas.extend(check_numbers(texto, ignorar_hasta=CAROUSEL_SLIDES))
    problemas.extend(check_emoji_muletilla(texto))
    return problemas


def build_prompt(dossier: Dossier, correcciones: list[str] | None = None,
                 pillar: str = "", previous: str = "") -> str:
    hechos = "\n".join(
        f"[{i}] {f.claim}\n    fuente: {f.source_name}"
        for i, f in enumerate(dossier.facts))
    partes = [
        f"TEMA: {dossier.topic}\n",
        f"DOSSIER (usa el índice para citar):\n{hechos}\n",
        "TAREA\nEscribe un carrusel de 5 diapositivas para TikTok photo mode.\n"
        "- diapositiva 1: promesa NUMERADA ('5 IAs que...', '3 comandos...'). "
        "Sin número, sin swipe.\n"
        "- diapositivas 2-4: revelación progresiva, un dato nuevo por "
        "diapositiva; el mejor dato en la 3 o 4 (bomba de valor).\n"
        "- diapositiva 5: conclusión + 'guárdalo para después'.\n"
        "- cada diapositiva: headline de 2 a 5 palabras + text de hasta 7 "
        "palabras. Juntas, como máximo 12 (techo de la marca): cuenta antes "
        "de responder. Medido en 20/09: pedir 'apunta a 10 en total' dio "
        "diapositivas de 13-14 palabras en tres intentos seguidos.\n"
        "- caption: una línea con la palabra clave + UNA pregunta.\n"
        "- visual: etiqueta COPIADA de la lista de ESTÉTICA, un pilar solo.\n"
        "- broll: términos EN INGLÉS del objeto concreto del tema, para la "
        "foto de portada.\n"
        "- used_facts: índices del dossier.\n"
        "- Nombra el tema (nombre + quién lo construyó, SI el dossier lo "
        "dice) ya en la diapositiva 1 o 2: diapositiva anónima no tiene "
        "búsqueda ni credibilidad.\n"
        "- Hilo: cada diapositiva continúa la anterior Y se entiende sola -- "
        "nombra el sujeto o retómalo ('esa técnica', 'el modelo') + dato "
        "nuevo. La diapositiva que solo tiene sentido pegada a la vecina "
        "('Segundo artículo en 2025') reprueba en el juez: di QUÉ pasó, no "
        "solo CUÁNDO.\n"
        "- UN dato numérico por línea: '143 tokens/s en la RTX 5090' rompe la "
        "regla de la marca (dos números en una frase) -- pon '143 tokens por "
        "segundo' en una línea y el nombre de la gráfica en la otra.\n",
        pillar_brief(pillar or "news", "carousel"),
        compact_brief(suggest_pillar(dossier.topic), "carousel"),
        voice_brief(),
        "REGLAS\n"
        "- Solo dato del dossier. Sin emoji, sin hashtag en la diapositiva.\n"
        "- Castellano de España hablado, frase corta.",
    ]
    if correcciones:
        bloque = "CORRIGE EL INTENTO ANTERIOR\n" + "\n".join(f"- {c}" for c in correcciones)
        if previous:
            bloque += (f"\nDIAPOSITIVAS ANTERIORES (ajusta estas, no empieces "
                       f"desde cero):\n{previous}")
        partes.append(bloque + "\nMantén lo que estaba bien y arregla solo lo señalado.")
    return "\n".join(partes)


def _texto(valor: object) -> str:
    return " ".join(str(valor).split()) if isinstance(valor, str) else ""


__all__ = ["CarouselAttempt", "CarouselReport", "build_prompt", "write_carousel"]
