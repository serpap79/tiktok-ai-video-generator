"""Juez de carrusel: politica medido, hook/fuente/cta/flujo leidos.

El mecanico (5 slides, techo de palabras, save en el 5, numero en el 1, numeros
anclados) ya paso en el guionista de carrusel. Lo que queda para lectura:
el slide 1 abre hueco, la progresion paga la promesa sin ir mas alla del
dossier, cada slide se sostiene solo en la secuencia (flujo), y el cierre pide
guardar con motivo -- no el slide, el acto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent.curator import policy
from agent.models import (
    Carousel,
    CarouselReview,
    Criterion,
    CriterionScore,
    Dossier,
)
from agent.ports.llm import LLM, LLMError, Usage, parse_json_object

MAX_INTENTOS = 2

JULGADOS = (Criterion.hook, Criterion.fuente, Criterion.flujo, Criterion.cta)

DESCRIPCIONES: dict[Criterion, str] = {
    Criterion.hook: (
        "El slide 1 abre hueco de informacion con promesa numerada? 2 = dan "
        "ganas de arrastrar; 1 = interesa pero regala el asunto; 0 = titulo "
        "generico o payoff ya en el slide 1. Promesa que los slides no pagan "
        "(ej. '5 IAs' con una sola mostrada) puntua cero."
    ),
    Criterion.fuente: (
        "Todo lo que los slides afirman esta en el dossier? 2 = todo sostenido; "
        "1 = algun slide va mas alla; 0 = afirmacion sin apoyo en el dossier."
    ),
    Criterion.flujo: (
        "CADA slide del medio se entiende solo? Test: tapa el resto y lee "
        "solo el slide 3 -- se puede decir de QUE habla? 2 = si en "
        "todos, y la secuencia tiene arco; 1 = se puede seguir con esfuerzo; "
        "0 = fragmento que solo tiene sentido pegado al vecino (ej. 'Lanzado "
        "en 2026 con la misma idea' -- la misma idea de QUE?). No aceptes "
        "promesa en lugar de historia: 'revelacion progresiva' con slides "
        "que no dicen nada solos es 0, no 1."
    ),
    Criterion.cta: (
        "El slide 5 cierra con conclusion + motivo para guardar? 2 = save con "
        "motivo concreto; 1 = pide save sin motivo; 0 = sin CTA."
    ),
}

SCHEMA_INFORME: dict[str, Any] = {
    "type": "object",
    "properties": {
        criterio.value: {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["reason", "score"],
        }
        for criterio in JULGADOS
    },
    "required": [c.value for c in JULGADOS],
}

SISTEMA = (
    "Juzgas carruseles de un canal espanol de tech, IA y ciencia. "
    "Nota de 0 a 2 por criterio, con motivo de una frase en castellano. "
    "Sé economico con el 2: nota maxima es para quien ejecuta el criterio, "
    "no para quien no falla."
)


@dataclass
class CarouselReviewReport:
    review: CarouselReview | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0

    @property
    def approved(self) -> bool:
        return self.review is not None and self.review.approved


def judge_carousel(carrossel: Carousel, dossier: Dossier, llm: LLM,
                   ) -> CarouselReviewReport:
    """Informe del carrusel, con politica medido antes de gastar cuota."""
    texto = "\n".join(f"{s.headline} {s.text}" for s in carrossel.slides)
    veredicto = policy.check(texto + " " + carrossel.caption)
    nota_politica = CriterionScore(
        criterion=Criterion.politica, measured=True,
        score=2 if veredicto.allowed else 0,
        reason=("ningun termino de la lista de politica en los slides"
                if veredicto.allowed else f"politica/{veredicto.rule}"),
    )
    if not veredicto.allowed:
        motivo = "no evaluado: reprobado en politica (medido)"
        no_evaluados = [
            CriterionScore(criterion=c, score=0, reason=motivo,
                           evaluated=False) for c in JULGADOS]
        review = CarouselReview(
            topic=carrossel.topic,
            scores=[nota_politica] + no_evaluados,
            reviewed_at=datetime.now(UTC),
            model=getattr(llm, "model", ""),
            provider=getattr(llm, "provider", ""))
        return CarouselReviewReport(review=review)

    uso = Usage()
    latencia = 0.0
    ultimo = ""
    for _ in range(MAX_INTENTOS):
        respuesta = llm.complete(
            build_prompt(carrossel, dossier),
            system=SISTEMA,
            schema=SCHEMA_INFORME,
            temperature=0.0,
            max_output_tokens=1024,
        )
        uso = uso + respuesta.usage
        latencia = round(latencia + respuesta.latency_s, 3)
        try:
            cuerpo = parse_json_object(respuesta.text)
            juzgados = [_nota(c, cuerpo.get(c.value)) for c in JULGADOS]
        except (LLMError, ValueError) as exc:
            ultimo = str(exc)
            continue
        return CarouselReviewReport(
            review=CarouselReview(
                topic=carrossel.topic,
                scores=[nota_politica] + juzgados,
                reviewed_at=datetime.now(UTC),
                model=getattr(llm, "model", ""),
                provider=getattr(llm, "provider", "")),
            usage=uso, latency_s=latencia)
    raise LLMError(f"informe de carrusel invalido ({ultimo})")


def _nota(criterio: Criterion, crudo: Any) -> CriterionScore:
    if not isinstance(crudo, dict):
        raise ValueError(f"informe sin el criterio {criterio.value!r}")
    score = crudo.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 2:
        raise ValueError(f"criterio {criterio.value!r} con nota invalida: {score!r}")
    motivo = " ".join(str(crudo.get("reason") or "").split())
    if len(motivo) < 3:
        # "ok" rebasa el min_length del contrato y tiraria abajo el slot.
        motivo = (f"el modelo no justifico la nota ({motivo})" if motivo
                  else "el modelo no justifico la nota")
    return CriterionScore(criterion=criterio, score=score, reason=motivo)


def build_prompt(carrossel: Carousel, dossier: Dossier) -> str:
    slides = "\n".join(
        f"[{s.n}] {s.headline} / {s.text}" for s in carrossel.slides)
    hechos = "\n".join(
        f"[{i}] {f.claim} (fuente: {f.source_name})" for i, f in enumerate(dossier.facts))
    rubrica = "\n".join(f"- {c.value}: {DESCRIPCIONES[c]}" for c in JULGADOS)
    return (
        f"TEMA: {carrossel.topic}\n\n"
        f"DOSSIER:\n{hechos}\n\n"
        f"CARRUSEL:\n{slides}\n\nLEYENDA: {carrossel.caption}\n\n"
        f"RUBRICA\n{rubrica}\n\n"
        "Antes de puntuar el flujo, escribe en el propio 'reason' de UNA linea "
        "de que habla cada slide 2-4, usando SOLO lo que esta escrito en el. "
        "Si alguno no se puede resolver ('la misma idea' de QUE?), flujo es 0.\n"
        "Devuelve JSON con un campo por criterio ('reason' + 'score'). "
        "No evalues politica: ya fue medida fuera de tu informe."
    )


__all__ = ["CarouselReviewReport", "build_prompt", "judge_carousel"]
