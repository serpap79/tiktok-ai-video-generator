"""Juiz: decide se o roteiro merece ser renderizado.

Dois critérios da rubrica **nao sao perguntados ao modelo**, e isso e o desenho e
nao economia:

- **duracao** e contagem de palavra. Perguntar a um LLM quantos segundos o texto
  leva falado e trocar uma medida por um chute.
- **politica** ja tem filtro escrito contra o radar real (`curator/policy.py`).
  Reusar o mesmo filtro garante que o roteiro seja julgado pela mesma regra que
  barrou o tema, em vez de duas listas que divergem com o tempo.

Um terceiro, **fonte**, tem uma parte medida: se a narracao cita numero em digito
que nao esta em nenhum fato do dossie, o critério vai a zero sem consultar
ninguem. O resto do critério -- afirmacao que vai alem do dossie, numero escrito
por extenso -- continua sendo leitura.

E a ordem entre as duas metades e a mesma do curador: **medir e barato, julgar
custa cota**. Quando a medida ja reprova em criterio de requisito, o modelo nao e
chamado, e os criterios de leitura ficam marcados como nao avaliados. Pagar um
parecer para confirmar uma reprovacao ja decidida seria queimar free tier -- e,
de graca, isso torna a reprovacao demonstravel sem nenhuma chave de API.

Sobram quatro critérios que sao mesmo julgamento -- hook, ponto de vista,
portugues falado e fechamento -- e para esses nao existe alternativa a um leitor.

O corte nao e so a soma. Soma sozinha permite compensacao errada: um roteiro que
e resumo de noticia (zero em ponto de vista) chegaria a 12 de 14 com o resto
perfeito, sendo exatamente o "AI slop" que o Creator Rewards exclui. Por isso
aprovar exige 11/14, **nenhum critério zerado** e nenhum veto violado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent.curator import policy
from agent.models import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    SHORT_MAX_DURATION_S,
    SHORT_MIN_DURATION_S,
    VETO_MINIMO,
    Criterion,
    CriterionScore,
    Dossier,
    Review,
    Script,
)
from agent.ports.llm import LLM, Completion, LLMError, Usage, parse_json_object
from agent.research.grounding import missing_numbers

# Duas chances para o parecer. Truncamento e JSON malformado sao intermitentes, e
# perder um roteiro ja escrito por causa disso custaria a proxima execucao inteira.
MAX_TENTATIVAS = 2

# Criterios que dependem de leitura. A ordem e a da rubrica original.
JULGADOS = (Criterion.hook, Criterion.fonte, Criterion.ponto_de_vista,
            Criterion.pt_br, Criterion.cta)

DESCRICOES: dict[Criterion, str] = {
    Criterion.hook: (
        "A primeira frase abre uma lacuna de informacao e nao a responde? "
        "2 = da vontade de continuar ouvindo; 1 = interessa mas entrega o assunto "
        "de graca; 0 = anuncia o tema ('hoje vou falar sobre'). "
        "Promessa que o video nao paga (ex. anunciar '5 IAs' e mostrar uma) "
        "zera, mesmo com frase boa."
    ),
    Criterion.fonte: (
        "Toda afirmacao factual da narracao esta sustentada por um fato do "
        "dossie? 2 = todas; 1 = alguma afirmacao vai alem do que o dossie diz; "
        "0 = ha afirmacao factual sem nenhum apoio no dossie."
    ),
    Criterion.ponto_de_vista: (
        "O roteiro tem ponto de vista proprio, ou e resumo de noticia? 2 = "
        "oferece leitura, contraste ou aponta o que as fontes NAO dizem; 1 = "
        "quase so recontagem, com um comentario; 0 = resumo da materia."
    ),
    Criterion.pt_br: (
        "E portugues do Brasil falado? 2 = frase curta, voz ativa, soa natural "
        "lido em voz alta; 1 = passagens escritas demais, jargao nao explicado "
        "ou numeros em sequencia (ficha tecnica lida em voz alta); 0 = travado, "
        "traduzido ao pe da letra."
    ),
    Criterion.cta: (
        "O fechamento chama para algo especifico, que nao seja 'siga para "
        "mais'? 2 = pergunta ou convite concreto ligado ao tema; 1 = generico; "
        "0 = pede seguidor ou nao fecha."
    ),
}

SISTEMA = (
    "Voce e editor de um canal brasileiro de tech, IA e ciencia, e avalia roteiros "
    "antes da producao. Voce e severo e especifico: nota alta e excecao, e cada "
    "nota vem com o motivo escrito de forma que o roteirista saiba o que mudar. "
    "Voce nao reescreve o roteiro, apenas julga."
)

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


@dataclass
class ReviewReport:
    """O parecer e o custo de produzi-lo."""

    review: Review | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.review is not None

    @property
    def approved(self) -> bool:
        return self.review is not None and self.review.approved


class Judge:
    def __init__(self, llm: LLM):
        self._llm = llm

    def review(self, script: Script, dossier: Dossier) -> ReviewReport:
        antecipado = review_measured_only(
            script, dossier,
            model=getattr(self._llm, "model", ""),
            provider=getattr(self._llm, "provider", ""),
        )
        if antecipado is not None:
            # Reprovado na medida: nenhuma chamada, custo zero.
            return ReviewReport(review=antecipado)

        uso = Usage()
        latencia = 0.0
        ultimo = ""

        for _ in range(MAX_TENTATIVAS):
            resposta = self._llm.complete(
                build_prompt(script, dossier),
                system=SISTEMA,
                schema=SCHEMA_PARECER,
                temperature=0.0,  # parecer precisa ser reproduzivel para o M5 comparar
                max_output_tokens=1536,
            )
            uso = uso + resposta.usage
            latencia = round(latencia + resposta.latency_s, 3)
            try:
                julgadas = _notas_julgadas(resposta)
            except LLMError as exc:
                # Resposta malformada e truncamento sao intermitentes: o roteiro
                # ja escrito nao pode ser perdido por causa de um JSON que abriu
                # e nao fechou. Uma segunda chance custa menos que refazer o
                # roteiro inteiro na proxima execucao.
                ultimo = str(exc)
                continue

            return ReviewReport(
                review=self._montar(script, _notas_medidas(script) + julgadas),
                usage=uso,
                latency_s=latencia,
            )

        raise LLMError(
            f"parecer invalido em {MAX_TENTATIVAS} tentativas ({ultimo}); "
            f"gastos {uso.total_tokens} tokens"
        )

    def _montar(self, script: Script, notas: list[CriterionScore]) -> Review:
        return Review(
            topic=script.topic,
            scores=notas,
            reviewed_at=datetime.now(UTC),
            model=getattr(self._llm, "model", ""),
            provider=getattr(self._llm, "provider", ""),
        )


def review_measured_only(
    script: Script, dossier: Dossier, *, model: str = "", provider: str = ""
) -> Review | None:
    """Parecer completo sem consultar modelo, quando a medida ja reprova.

    Devolve None quando o roteiro passa nos critérios de requisito -- ai o
    parecer depende de leitura e nao ha como produzi-lo sem modelo.

    E funcao livre, e nao metodo, para que quem chama possa descobrir que o
    roteiro ja reprovou **antes** de construir um adaptador e exigir chave de
    API. E o que permite reprovar a fixture adversarial do M3 sem nenhuma conta
    em provedor nenhum.
    """
    medidas = measured_scores(script, dossier)
    bloqueio = [s for s in medidas if s.score < VETO_MINIMO.get(s.criterion, 1)]
    if not bloqueio:
        return None
    return Review(
        topic=script.topic,
        scores=medidas + _nao_avaliados(bloqueio),
        reviewed_at=datetime.now(UTC),
        model=model,
        provider=provider,
    )


def _notas_julgadas(resposta: Completion) -> list[CriterionScore]:
    corpo = parse_json_object(resposta.text)
    notas: list[CriterionScore] = []
    for criterio in JULGADOS:
        bruto = corpo.get(criterio.value)
        if not isinstance(bruto, dict):
            raise LLMError(f"parecer sem o criterio {criterio.value!r}")

        score = bruto.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 2:
            # Nota fora da escala nao e arredondada para dentro: a rubrica e de
            # 0 a 2, e aceitar 5 seria deixar o modelo redefinir o corte.
            raise LLMError(
                f"criterio {criterio.value!r} veio com nota invalida: {score!r}"
            )
        motivo = " ".join(str(bruto.get("reason") or "").split())
        # Motivo curto demais ("ok") estourava o min_length do CriterionScore
        # como ValidationError -- fora do LLMError que o laco trata -- e
        # derrubava o slot inteiro. Achado no teste do piloto (20/09).
        if len(motivo) < 3:
            motivo = (f"o modelo nao justificou a nota ({motivo})" if motivo
                      else "o modelo nao justificou a nota")
        notas.append(CriterionScore(criterion=criterio, score=score, reason=motivo))
    return notas


def measured_scores(script: Script, dossier: Dossier) -> list[CriterionScore]:
    """Os critérios de requisito, medidos sem consultar modelo.

    Devolve duracao e politica sempre, e fonte **apenas quando a ancoragem
    numerica falha** -- porque so nesse caso a medida decide o critério sozinha.
    Quando os numeros conferem, fonte continua sendo leitura e sai do parecer do
    modelo.
    """
    notas = _notas_medidas(script)
    fora = missing_numbers(script.narration, _texto_do_dossie(dossier))
    if fora:
        notas.append(CriterionScore(
            criterion=Criterion.fonte,
            score=0,
            reason=f"numero citado sem respaldo no dossie: {', '.join(fora[:4])}",
            measured=True,
        ))
    return notas


def _nao_avaliados(bloqueio: list[CriterionScore]) -> list[CriterionScore]:
    """Preenche a rubrica com os critérios que nao foram julgados, e diz por que.

    A rubrica continua completa -- parecer com criterio faltando nao valida --
    mas zero aqui significa "nao sei", e `evaluated=False` marca isso para que a
    nota nao volte ao roteirista como se fosse critica ao texto.
    """
    motivo = (
        "nao avaliado: o roteiro reprovou antes em "
        + ", ".join(sorted(s.criterion.value for s in bloqueio))
        + " (medido), e o parecer do modelo nao mudaria o resultado"
    )
    ja_medidos = {s.criterion for s in bloqueio}
    return [
        CriterionScore(criterion=c, score=0, reason=motivo, evaluated=False)
        for c in JULGADOS
        if c not in ja_medidos
    ]


def _notas_medidas(script: Script) -> list[CriterionScore]:
    """Duracao e politica: medida, nunca leitura."""
    duracao = script.estimated_duration_s
    faixa = ((SHORT_MIN_DURATION_S, SHORT_MAX_DURATION_S) if script.format == "short"
             else (MIN_DURATION_S, MAX_DURATION_S))
    na_faixa = faixa[0] <= duracao <= faixa[1]
    # Nao existe meio ponto para duracao: ou o video esta na faixa do formato,
    # ou nao e.
    nota_duracao = CriterionScore(
        criterion=Criterion.duracao,
        score=2 if na_faixa else 0,
        reason=(
            f"{script.word_count} palavras, ~{duracao:.0f}s estimados "
            f"(faixa exigida: {faixa[0]}-{faixa[1]}s)"
            + ("" if na_faixa else "; fora da faixa do formato")
        ),
        measured=True,
    )

    veredito = policy.check(script.narration)
    nota_politica = CriterionScore(
        criterion=Criterion.politica,
        score=2 if veredito.allowed else 0,
        reason=(
            "nenhum termo da lista de politica na narracao"
            if veredito.allowed
            else f"politica/{veredito.rule}: '{veredito.matched}' — {veredito.reason}"
        ),
        measured=True,
    )
    return [nota_duracao, nota_politica]


def _texto_do_dossie(dossier: Dossier) -> str:
    return "\n".join(f"{f.claim}\n{f.quote}" for f in dossier.facts)


def build_prompt(script: Script, dossier: Dossier) -> str:
    """Monta o prompt do parecer. Funcao livre para o teste inspecionar o texto."""
    fatos = "\n".join(
        f"[{i}] {f.claim} (fonte: {f.source_name})" for i, f in enumerate(dossier.facts)
    )
    rubrica = "\n".join(f"- {c.value}: {DESCRICOES[c]}" for c in JULGADOS)
    loop = (
        "\nFECHAMENTO EM LOOP: video curto vive de replay automatico; o "
        "criterio cta vale 2 quando o fechamento reconecta com a pergunta do "
        "hook, e 0 quando e CTA generico."
        if script.format == "short" else ""
    )
    tipo = _expectativa(script.pillar)
    return (
        f"TEMA: {script.topic}\n\n"
        f"DOSSIE DISPONIVEL AO ROTEIRISTA:\n{fatos}\n\n"
        "ROTEIRO A AVALIAR\n"
        f"HOOK: {script.hook}\n\n"
        f"CORPO: {script.body}\n\n"
        f"FECHAMENTO: {script.closing}\n\n"
        "RUBRICA (nota de 0 a 2 em cada critério)\n"
        f"{rubrica}{loop}{tipo}\n\n"
        "Devolva um objeto json com um campo por critério, cada um com 'reason' "
        "(uma frase dizendo o que precisa mudar, em portugues) e 'score' (0, 1 ou 2). "
        "Escreva a razao antes da nota. Nao avalie duracao nem politica: "
        "esses dois sao medidos fora do seu parecer."
    )


def _expectativa(pillar: str) -> str:
    """A formula do tipo de conteudo, para o juiz cobrar o que o roteirista recebeu.

    Sem isso o juiz julgaria um tutorial pela regua de noticia: o ponto de
    vista de um tutorial e "o erro comum que anula tudo", nao uma opiniao.
    """
    if not pillar:
        return ""
    from agent.brand.brand import load

    p = load().pillars.get(pillar)
    if p is None:
        return ""
    return (f"\nTIPO {p.tag}: gancho esperado = {p.hook_formula}; batidas = "
            + " -> ".join(p.beats) + ". Use isso para ler hook e ponto_de_vista.")

