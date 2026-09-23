"""Roteirista de carrossel: dossie para 5 slides + legenda.

Receita validada dos canais (2026): slide 1 com promessa numerada, revelacao
progressiva com value bomb no meio, slide 5 com conclusao + CTA de save, 10-15
palavras por slide, legenda com pergunta para puxar comentario. Tudo que e
contavel vira portao mecanico aqui; gancho e progressao sao do juiz.

Mesmo laco do roteirista de video: ate 3 tentativas, defeito medido de volta
ao modelo em texto, tentativa reprovada gravada para calibrar o prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent.brand.brand import pillar_brief, voice_brief
from agent.brand.checks import check_emoji_bordao, check_numbers
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
from agent.writer.writer import _resolver_fatos

MAX_TENTATIVAS = 3

SISTEMA = (
    "Voce roteiriza carrosseis de um canal brasileiro dark de tech, IA e "
    "ciencia. Cada slide e uma frase curta que se le em 3 segundos. Voce so "
    "afirma o que esta no dossie que recebe."
)

SCHEMA_CARROSSEL: dict[str, Any] = {
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
    # Os slides reprovados em texto, para a correcao ajustar em vez de refazer.
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
                   max_attempts: int = MAX_TENTATIVAS,
                   notes: list[str] | None = None,
                   pillar: str = "") -> CarouselReport:
    """Escreve o carrossel com correcao propria do que e mecanico."""
    report = CarouselReport(
        topic=dossier.topic,
        model=getattr(llm, "model", ""),
        provider=getattr(llm, "provider", ""),
    )
    from agent.writer.writer import thin_dossier_reason
    report.refusal = thin_dossier_reason(dossier, "carousel")
    if report.refusal:
        return report

    correcao: list[str] = list(notes or [])
    anterior = ""
    for _ in range(max_attempts):
        try:
            resposta = llm.complete(
                build_prompt(dossier, correcao, pillar=pillar,
                             previous=anterior if correcao else ""),
                system=SISTEMA,
                schema=SCHEMA_CARROSSEL,
                temperature=0.6,
                max_output_tokens=2048,
            )
        except LLMError:
            raise
        report.model, report.provider = resposta.model, resposta.provider
        tentativa, carrossel = _avaliar(resposta, dossier, pillar)
        report.attempts.append(tentativa)
        if not tentativa.violations:
            report.carousel = carrossel
            return report
        correcao = tentativa.violations
        anterior = tentativa.text or anterior
    return report


def _avaliar(resposta: Completion, dossier: Dossier, pillar: str = ""
             ) -> tuple[CarouselAttempt, Carousel | None]:
    tentativa = CarouselAttempt(usage=resposta.usage, latency_s=resposta.latency_s)
    if resposta.truncated:
        tentativa.violations.append(
            "a resposta foi cortada por limite de tokens; escreva mais curto")
        return tentativa, None
    try:
        corpo = parse_json_object(resposta.text)
    except LLMError as exc:
        tentativa.violations.append(f"a resposta nao veio como objeto JSON: {exc}")
        return tentativa, None

    usados, fora = _resolver_fatos(corpo.get("used_facts"), dossier.facts)
    try:
        slides = [Slide(n=int(s.get("n", i + 1)),
                        headline=_texto(s.get("headline")),
                        text=_texto(s.get("text")),
                        visual=_texto(s.get("visual")))
                  for i, s in enumerate(corpo.get("slides") or [])]
        broll = [_texto(t) for t in (corpo.get("broll") or []) if isinstance(t, str)]
        carrossel = Carousel(topic=dossier.topic, slides=slides,
                             caption=_texto(corpo.get("caption")), facts=usados,
                             broll=[t for t in broll if t][:3], pillar=pillar)
    except (ValidationError, ValueError, AttributeError) as exc:
        tentativa.violations.append(f"o carrossel nao respeita o contrato: {exc}")
        return tentativa, None
    tentativa.text = "\n".join(
        f"[{s.n}] {s.headline} / {s.text} ~ {s.visual}" for s in carrossel.slides
    ) + f"\nlegenda: {carrossel.caption}"

    tentativa.violations.extend(_violacoes(carrossel, dossier, fora))
    return tentativa, (carrossel if not tentativa.violations else None)


def _violacoes(carrossel: Carousel, dossier: Dossier, fora: list[int]) -> list[str]:
    problemas: list[str] = []
    for s in carrossel.slides:
        if s.word_count > CAROUSEL_MAX_WORDS_PER_SLIDE:
            problemas.append(
                f"slide {s.n} tem {s.word_count} palavras (teto "
                f"{CAROUSEL_MAX_WORDS_PER_SLIDE}): corte "
                f"{s.word_count - CAROUSEL_MAX_WORDS_PER_SLIDE} palavras, "
                "slide se le em 3 segundos.")
    s1 = carrossel.slides[0]
    if not _DIGITO.search(f"{s1.headline} {s1.text}"):
        problemas.append(
            "slide 1 sem numero: a promessa numerada ('5 IAs que...') e o que "
            "faz a pessoa arrastar. Sem numero nao ha payoff finito.")
    s5 = carrossel.slides[-1]
    if "salv" not in f"{s5.headline} {s5.text}".lower():
        problemas.append(
            "slide 5 sem CTA de save ('salve'): save/view e o indicador lider "
            "do formato; sem ele o carrossel nao acumula distribuicao.")
    if "?" not in carrossel.caption:
        problemas.append(
            "legenda sem pergunta: a legenda carrega o convite ao comentario, "
            "e comentario e onde o carrossel ganha do video.")
    if fora:
        problemas.append(
            f"used_facts aponta indice que nao existe no dossie: {fora}.")
    if not carrossel.facts:
        problemas.append("used_facts vazio: carrossel tambem ancora em fonte.")
    problemas.extend(
        "visual: " + v for v in validate_terms([s.visual for s in carrossel.slides]))
    problemas.extend(validate_broll(carrossel.broll, "carousel"))
    texto = "\n".join(f"{s.headline} {s.text}" for s in carrossel.slides)
    fontes = "\n".join(f"{f.claim}\n{f.quote}" for f in dossier.facts)
    sem_sujeito = missing_subject(texto, dossier.topic, minimum=1)
    if sem_sujeito:
        problemas.append(
            "carrossel sem sujeito: nenhum slide nomeia "
            + ", ".join(f"{t!r}" for t in sem_sujeito) + ". Quem assiste "
            "precisa saber sobre O QUE sao os 5 slides.")
    # Contagem estrutural (o "5" da promessa) nao e afirmacao factual: sao os
    # proprios slides, verificados acima pela ordem 1-5. So numero acima disso
    # precisa existir no dossie.
    soltos = [n for n in missing_numbers(texto, fontes)
              if not (n.isdigit() and int(n) <= CAROUSEL_SLIDES)]
    if soltos:
        problemas.append(
            f"slide cita numero que nao esta no dossie: {', '.join(soltos)}.")
    tells = scan_tells(texto)
    if tells:
        problemas.append(
            "slide com vicio de IA (" + "; ".join(tells[:3]) + "): reescreva "
            "como fala curta de pessoa.")
    problemas.extend(check_numbers(texto, ignorar_ate=CAROUSEL_SLIDES))
    problemas.extend(check_emoji_bordao(texto))
    return problemas


def build_prompt(dossier: Dossier, correcoes: list[str] | None = None,
                 pillar: str = "", previous: str = "") -> str:
    fatos = "\n".join(
        f"[{i}] {f.claim}\n    fonte: {f.source_name}"
        for i, f in enumerate(dossier.facts))
    partes = [
        f"TEMA: {dossier.topic}\n",
        f"DOSSIE (use o indice para citar):\n{fatos}\n",
        "TAREFA\nEscreva um carrossel de 5 slides para TikTok photo mode.\n"
        "- slide 1: promessa NUMERADA ('5 IAs que...', '3 comandos...'). Sem numero, sem swipe.\n"
        "- slides 2-4: revelacao progressiva, um dado novo por slide; o melhor "
        "dado no 3 ou 4 (value bomb).\n"
        "- slide 5: conclusao + 'salve para depois'.\n"
        "- cada slide: headline de 2 a 5 palavras + text de ate 7 palavras. Juntos, no "
        "maximo 12 (teto da marca): conte antes de responder. Medido em 20/09: pedir "
        "'mire 10 no total' deu slides de 13-14 palavras em tres tentativas seguidas.\n"
        "- caption: uma linha com a palavra-chave + UMA pergunta.\n"
        "- visual: tag COPIADA da lista de ESTETICA, um pilar so.\n"
        "- broll: termos EM INGLES do objeto concreto do assunto, para a foto da capa.\n"
        "- used_facts: indices do dossie.\n"
        "- Nomeie o assunto (nome + quem construiu, SE o dossie disser) ja no "
        "slide 1 ou 2: slide anonimo nao tem busca nem credibilidade.\n"
        "- Fio: cada slide continua o anterior E se entende sozinho -- nomeie "
        "o sujeito ou retome ('essa tecnica', 'o modelo') + dado novo. Slide "
        "que so faz sentido colado no vizinho ('Segundo artigo em 2025') "
        "reprova no juiz: diga O QUE aconteceu, nao so QUANDO.\n"
        "- UM dado numerico por linha: '143 tokens/s na RTX 5090' quebra a "
        "regra da marca (dois numeros numa frase) -- ponha '143 tokens por "
        "segundo' numa linha e o nome da placa na outra.\n",
        pillar_brief(pillar or "news", "carousel"),
        compact_brief(suggest_pillar(dossier.topic), "carousel"),
        voice_brief(),
        "REGRAS\n"
        "- So dado do dossie. Sem emoji, sem hashtag no slide.\n"
        "- pt-BR falado, frase curta.",
    ]
    if correcoes:
        bloco = "CORRIJA A TENTATIVA ANTERIOR\n" + "\n".join(f"- {c}" for c in correcoes)
        if previous:
            bloco += f"\nSLIDES ANTERIORES (ajuste estes, nao comece do zero):\n{previous}"
        partes.append(bloco + "\nMantenha o que estava bom e conserte apenas o apontado.")
    return "\n".join(partes)


def _texto(valor: object) -> str:
    return " ".join(str(valor).split()) if isinstance(valor, str) else ""


__all__ = ["CarouselAttempt", "CarouselReport", "build_prompt", "write_carousel"]
