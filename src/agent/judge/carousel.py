"""Juiz de carrossel: politica medido, hook/fonte/cta/fluxo lidos.

O mecanico (5 slides, teto de palavras, save no 5, numero no 1, numeros
ancorados) ja passou no roteirista de carrossel. O que sobra para leitura:
o slide 1 abre lacuna, a progressao paga a promessa sem ir alem do dossie,
cada slide se sustenta sozinho na sequencia (fluxo), e o fechamento pede
save com motivo -- nao o slide, o ato.
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

MAX_TENTATIVAS = 2

JULGADOS = (Criterion.hook, Criterion.fonte, Criterion.fluxo, Criterion.cta)

DESCRICOES: dict[Criterion, str] = {
    Criterion.hook: (
        "O slide 1 abre lacuna de informacao com promessa numerada? 2 = da "
        "vontade de arrastar; 1 = interessa mas entrega o assunto; 0 = titulo "
        "generico ou payoff ja no slide 1. Promessa que os slides nao pagam "
        "(ex. '5 IAs' com uma so mostrada) zera."
    ),
    Criterion.fonte: (
        "Tudo que os slides afirmam esta no dossie? 2 = tudo sustentado; 1 = "
        "algum slide vai alem; 0 = afirmacao sem apoio no dossie."
    ),
    Criterion.fluxo: (
        "CADA slide do meio se entende sozinho? Teste: tape o resto e leia "
        "so o slide 3 -- da para dizer sobre O QUE ele fala? 2 = sim em "
        "todos, e a sequencia tem arco; 1 = da para seguir com esforco; "
        "0 = fragmento que so faz sentido colado no vizinho (ex. 'Lancado "
        "em 2026 com a mesma ideia' -- a mesma ideia do QUE?). Nao aceite "
        "promessa no lugar de historia: 'revelacao progressiva' com slides "
        "que nao dizem nada sozinhos e 0, nao 1."
    ),
    Criterion.cta: (
        "O slide 5 fecha com conclusao + motivo para salvar? 2 = save com "
        "motivo concreto; 1 = pede save sem motivo; 0 = sem CTA."
    ),
}

SCHEMA_PARECER: dict[str, Any] = {
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
    "Voce julga carrosseis de um canal brasileiro de tech, IA e ciencia. "
    "Nota de 0 a 2 por critério, com motivo de uma frase em portugues. "
    "Seja economico com 2: nota maxima e para quem executa o critério, "
    "nao para quem nao erra."
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
    """Parecer do carrossel, com politica medido antes de gastar cota."""
    texto = "\n".join(f"{s.headline} {s.text}" for s in carrossel.slides)
    veredito = policy.check(texto + " " + carrossel.caption)
    nota_politica = CriterionScore(
        criterion=Criterion.politica, measured=True,
        score=2 if veredito.allowed else 0,
        reason=("nenhum termo da lista de politica nos slides"
                if veredito.allowed else f"politica/{veredito.rule}"),
    )
    if not veredito.allowed:
        motivo = "nao avaliado: reprovado em politica (medido)"
        nao_avaliados = [
            CriterionScore(criterion=c, score=0, reason=motivo,
                           evaluated=False) for c in JULGADOS]
        review = CarouselReview(
            topic=carrossel.topic,
            scores=[nota_politica] + nao_avaliados,
            reviewed_at=datetime.now(UTC),
            model=getattr(llm, "model", ""),
            provider=getattr(llm, "provider", ""))
        return CarouselReviewReport(review=review)

    uso = Usage()
    latencia = 0.0
    ultimo = ""
    for _ in range(MAX_TENTATIVAS):
        resposta = llm.complete(
            build_prompt(carrossel, dossier),
            system=SISTEMA,
            schema=SCHEMA_PARECER,
            temperature=0.0,
            max_output_tokens=1024,
        )
        uso = uso + resposta.usage
        latencia = round(latencia + resposta.latency_s, 3)
        try:
            corpo = parse_json_object(resposta.text)
            julgadas = [_nota(c, corpo.get(c.value)) for c in JULGADOS]
        except (LLMError, ValueError) as exc:
            ultimo = str(exc)
            continue
        return CarouselReviewReport(
            review=CarouselReview(
                topic=carrossel.topic,
                scores=[nota_politica] + julgadas,
                reviewed_at=datetime.now(UTC),
                model=getattr(llm, "model", ""),
                provider=getattr(llm, "provider", "")),
            usage=uso, latency_s=latencia)
    raise LLMError(f"parecer de carrossel invalido ({ultimo})")


def _nota(criterio: Criterion, bruto: Any) -> CriterionScore:
    if not isinstance(bruto, dict):
        raise ValueError(f"parecer sem o criterio {criterio.value!r}")
    score = bruto.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 2:
        raise ValueError(f"criterio {criterio.value!r} com nota invalida: {score!r}")
    motivo = " ".join(str(bruto.get("reason") or "").split())
    if len(motivo) < 3:
        # "ok" estoura o min_length do contrato e derrubaria o slot.
        motivo = (f"o modelo nao justificou a nota ({motivo})" if motivo
                  else "o modelo nao justificou a nota")
    return CriterionScore(criterion=criterio, score=score, reason=motivo)


def build_prompt(carrossel: Carousel, dossier: Dossier) -> str:
    slides = "\n".join(
        f"[{s.n}] {s.headline} / {s.text}" for s in carrossel.slides)
    fatos = "\n".join(
        f"[{i}] {f.claim} (fonte: {f.source_name})" for i, f in enumerate(dossier.facts))
    rubrica = "\n".join(f"- {c.value}: {DESCRICOES[c]}" for c in JULGADOS)
    return (
        f"TEMA: {carrossel.topic}\n\n"
        f"DOSSIE:\n{fatos}\n\n"
        f"CARROSSEL:\n{slides}\n\nLEGENDA: {carrossel.caption}\n\n"
        f"RUBRICA\n{rubrica}\n\n"
        "Antes de notar o fluxo, escreva no proprio 'reason' de UMA linha "
        "sobre o que fala cada slide 2-4, usando SO o que esta escrito nele. "
        "Se algum nao der para resolver ('a mesma ideia' do QUE?), fluxo e 0.\n"
        "Devolva JSON com um campo por critério ('reason' + 'score'). "
        "Nao avalie politica: ja foi medida fora do seu parecer."
    )


__all__ = ["CarouselReviewReport", "build_prompt", "judge_carousel"]
