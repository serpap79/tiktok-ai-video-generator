"""Humanizacao: tira o vicio de texto de IA da narracao, sem inventar nada.

Integracao do `blader/humanizer` (MIT, 50k estrelas) como estagio do
roteirista -- nao como dependencia: la e um skill de prompt para agente, aqui
vira uma passada com travas de grounding. Atribuicao e ideia de la; as regras
de seguranca sao nossas.

Duas metades, como o resto do roteirista:

1. **Scan deterministico** (`scan`): os tells fortes e de baixo falso-positivo,
   adaptados para pt-BR falado. Sem achado, sem chamada -- quota de free tier
   nao se gasta com texto que ja soa humano.
2. **Reescrita em uma chamada** (`humanize`): o modelo recebe os trechos
   marcados e reescreve hook/body/closing sob as mesmas travas do roteiro
   (fatos, numeros, faixa de palavras). Se a reescrita quebra numero ou faixa,
   vale o original com motivo -- reescrita que inventa dado e pior que vicio
   de estilo.

O que NAO passa por aqui: `search_terms` (vocabulario fechado, nao prosa) e
slides de carrossel (linhas curtas de impacto, nao paragrafos).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.models import Dossier
from agent.ports.llm import LLM, Completion, LLMError, Usage, parse_json_object
from agent.research.grounding import missing_numbers

# Tells fortes em pt-BR falado. Lista curta de proposito: cada item aqui
# dispara UMA chamada de modelo, entao falso-positivo custa quota. So entra o
# que quase nunca aparece em fala natural de canal tech.
_TELLS: tuple[tuple[str, str], ...] = (
    ("não é apenas", "contraste encenado (não-é-X-é-Y)"),
    ("nao e apenas", "contraste encenado (não-é-X-é-Y)"),
    ("não se trata de", "contraste encenado"),
    ("nao se trata de", "contraste encenado"),
    ("mergulh", "palavra de IA (mergulhar/delve)"),
    ("paisagem", "palavra de IA (landscape figurado)"),
    ("testemunho", "palavra de IA (testament)"),
    ("deslumbrante", "linguagem de anuncio"),
    ("vibrante", "linguagem de anuncio"),
    ("jornada", "dizer profundo (saying)"),
    ("em resumo", "fechamento de cartilha"),
    ("para concluir", "fechamento de cartilha"),
    ("concluindo", "fechamento de cartilha"),
    ("é importante notar", "enchimento (filler)"),
    ("e importante notar", "enchimento (filler)"),
    ("vale ressaltar", "enchimento (filler)"),
    ("no mundo de hoje", "preambulo encenado"),
    ("na era digital", "preambulo encenado"),
    ("siga para mais", "CTA generico (a rubrica reprova)"),
)

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]"
)


def scan(narration: str) -> list[str]:
    """Trechos com vicio, em texto que vai ao modelo. Vazio = nada a fazer."""
    baixa = narration.lower()
    achados = [f"{t!r} ({motivo})" for t, motivo in _TELLS if t in baixa]
    if _EMOJI.search(narration):
        achados.append("emoji no texto (TTS nao fala emoji)")
    return achados


HUMANIZE_SYSTEM = (
    "Voce revisa narracao de video curto de canal brasileiro de tech, IA e "
    "ciencia. Soa como pessoa falando, nao como chatbot escrevendo. Nunca "
    "inventa fato, numero, nome ou data: tudo que afirma ja esta no texto."
)


def build_prompt(hook: str, body: str, closing: str,
                 flags: list[str], min_words: int, max_words: int) -> str:
    alvos = "\n".join(f"- {f}" for f in flags)
    return (
        "TAREFA\nReescreva a narracao abaixo mantendo TODOS os fatos, numeros e "
        "nomes com os mesmos valores. Pode cortar enrolacao, juntar frase e "
        "trocar palavra engessada por fala natural curta.\n"
        f"Trechos marcados:\n{alvos}\n"
        "REGRAS\n"
        f"- Entre {min_words} e {max_words} palavras no total (hook + body + closing).\n"
        "- pt-BR falado, frase curta, voz ativa. Sem emoji.\n"
        "- Sem contraste encenado ('nao e X, e Y'), sem preambulo ('mergulhe', "
        "'no mundo de hoje'), sem fechamento de cartilha ('em resumo').\n"
        "- Termine no ultimo fato concreto, nao em otimismo vago.\n"
        "- NAO escreva indices como [0] no texto.\n"
        "NARRACAO\n"
        f"hook: {hook}\nbody: {body}\nclosing: {closing}\n"
        'Devolva JSON {"hook": ..., "body": ..., "closing": ...} e nada mais.'
    )


@dataclass
class HumanizeReport:
    hook: str
    body: str
    closing: str
    changed: bool = False
    notes: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


def humanize(hook: str, body: str, closing: str, dossier: Dossier,
             llm: LLM, min_words: int, max_words: int) -> HumanizeReport:
    """Uma passada de humanizacao. Original intacto se a reescrita falhar."""
    original = HumanizeReport(hook=hook, body=body, closing=closing)
    flags = scan("\n".join((hook, body, closing)))
    if not flags:
        return original
    try:
        resposta: Completion = llm.complete(
            build_prompt(hook, body, closing, flags, min_words, max_words),
            system=HUMANIZE_SYSTEM,
            schema={"type": "object",
                    "properties": {"hook": {"type": "string"},
                                   "body": {"type": "string"},
                                   "closing": {"type": "string"}},
                    "required": ["hook", "body", "closing"]},
            temperature=0.7,
            max_output_tokens=2048,
        )
    except LLMError as exc:
        original.notes.append(f"humanizacao pulada (provedor): {exc}")
        return original
    original.usage = resposta.usage
    original.latency_s = resposta.latency_s
    if resposta.truncated:
        original.notes.append("reescrita cortada; vale o original")
        return original
    try:
        corpo = parse_json_object(resposta.text)
    except LLMError as exc:
        original.notes.append(f"reescrita fora de formato; vale o original: {exc}")
        return original
    novo = HumanizeReport(
        hook=" ".join(str(corpo.get("hook", "")).split()),
        body=" ".join(str(corpo.get("body", "")).split()),
        closing=" ".join(str(corpo.get("closing", "")).split()),
        usage=resposta.usage, latency_s=resposta.latency_s,
    )
    if not (novo.hook and novo.body and novo.closing):
        novo.notes.append("reescrita com campo vazio; vale o original")
        return original
    palavras = len(f"{novo.hook} {novo.body} {novo.closing}".split())
    if not (min_words <= palavras <= max_words):
        novo.notes.append(
            f"reescrita com {palavras} palavras (faixa {min_words}-{max_words}); "
            "vale o original")
        novo.hook, novo.body, novo.closing = hook, body, closing
        return novo
    fontes = "\n".join(f"{f.claim}\n{f.quote}" for f in dossier.facts)
    soltos = missing_numbers(f"{novo.hook} {novo.body} {novo.closing}", fontes)
    if soltos:
        novo.notes.append(
            f"reescrita mudou numero ({', '.join(soltos)}); vale o original")
        novo.hook, novo.body, novo.closing = hook, body, closing
        return novo
    novo.changed = True
    novo.notes.append(f"tells corrigidos: {len(flags)}")
    return novo


__all__ = ["HumanizeReport", "HUMANIZE_SYSTEM", "build_prompt", "humanize", "scan"]
