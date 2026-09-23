"""Pesquisador: de um tema escolhido para um dossie com fonte em cada fato.

A decisao de projeto que sustenta todo o resto: **uma chamada de modelo por
fonte, e a URL e estampada por nos, nao pedida a ele.** O modelo recebe o texto
de uma pagina e devolve afirmacoes sobre aquela pagina; de onde veio o texto e
informacao que ja temos. Pedir `source_url` ao modelo convidaria o erro mais
caro possivel neste projeto -- fato real com fonte trocada, que parece ancorado,
passa no juiz e so aparece quando alguem clica no link.

Com isso, "nao existe Fact sem URL verificavel" deixa de depender do modelo ter
sido honesto e passa a ser estrutural. O que ainda depende dele e a fidelidade
da afirmacao ao texto, e e ai que entram os dois portoes deterministicos:

1. **o trecho citado precisa existir na pagina.** O modelo devolve, junto de
   cada afirmacao, a passagem literal que a sustenta. Conferir passagem e
   `in` numa string -- barato e impossivel de enganar.
2. **todo numero da afirmacao precisa estar na fonte** (`grounding.py`).

Os dois derrubam o fato com motivo gravado, nunca em silencio. Dossie curto com
motivo registrado e calibravel; dossie cheio de fato frouxo nao e.

Ha ainda uma terceira regra, e ela veio de execucao real: **um trecho sustenta um
fato so**. Sem isso, o modelo divide uma frase de changelog em quatro afirmacoes
e entrega um dossie que parece cheio e nao da assunto para 60 segundos de video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import ValidationError

from agent.models import Decision, Dossier, Fact
from agent.ports.llm import LLM, LLMError, Usage, parse_json_object
from agent.research import grounding, sources
from agent.research.fetch import Page, PageFetcher, PageUnavailable
from agent.research.sources import Candidate
from agent.text import tokens

SISTEMA = (
    "Voce e pesquisador de um canal de tech, IA e ciencia. Sua unica funcao e "
    "extrair afirmacoes factuais do texto que recebe, sem acrescentar nada que o "
    "texto nao diga. Voce nao opina, nao contextualiza com conhecimento proprio e "
    "nao completa lacuna com o que costuma ser verdade. Se o texto nao tratar do "
    "tema pedido, devolva lista vazia."
)

# O schema e o mesmo para os dois provedores: o Gemini o recebe como
# responseSchema nativo e o Groq como texto no system. Pedir `quote` ANTES de
# `claim` e deliberado -- a ordem das chaves e a ordem em que o modelo escreve, e
# escolher a passagem primeiro e o que faz a afirmacao sair dela, em vez de a
# passagem ser procurada depois para justificar o que ele ja tinha escrito.
SCHEMA_FATOS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["quote", "claim"],
            },
        }
    },
    "required": ["facts"],
}

_ESPACOS = re.compile(r"\s+")

# Trecho muito curto casa com qualquer coisa e nao prova nada ("5,9 GB" aparece
# no menu tambem). Muito longo vira a pagina inteira colada.
MIN_TRECHO = 25
MAX_TRECHO = 400

# Caracteres da pagina que vao ao modelo depois do foco no tema (~1,2K
# tokens). A pagina lida continua com `research_page_chars` para o portao.
FOCO_CHARS = 5000


@dataclass
class Discarded:
    """Fato que o modelo produziu e um portao derrubou, com o motivo."""

    claim: str
    reason: str
    source_url: str


@dataclass
class ResearchReport:
    """O que a pesquisa produziu, o que descartou e quanto custou.

    Custo entra no relatorio porque o M3 e o primeiro estagio que gasta cota, e
    o eval do M5 compara provedores por qualidade **e** por consumo. Numero
    medido na hora e mais confiavel que reconstruido depois do log.
    """

    topic: str
    dossier: Dossier | None = None
    pages: list[Page] = field(default_factory=list)
    discarded: list[Discarded] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    model: str = ""
    # `provedor:modelo` que respondeu por ultimo quando o LLM e roteado.
    route: str = ""

    @property
    def facts(self) -> list[Fact]:
        return list(self.dossier.facts) if self.dossier else []

    @property
    def source_count(self) -> int:
        """Fontes distintas que sustentam o dossie, por dominio."""
        return len({_dominio(str(f.source_url)) for f in self.facts})

    @property
    def ok(self) -> bool:
        return self.dossier is not None


class Researcher:
    def __init__(
        self,
        llm: LLM,
        fetcher: PageFetcher | None = None,
        client: httpx.Client | None = None,
        max_sources: int = 5,
        max_facts_per_source: int = 4,
    ):
        self._llm = llm
        self._fetcher = fetcher or PageFetcher()
        self._client = client
        self._max_sources = max_sources
        self._max_facts = max_facts_per_source

    def research(
        self, decision: Decision, candidates: list[Candidate] | None = None,
        target_facts: int | None = None,
    ) -> ResearchReport:
        """Le as fontes em ordem e extrai fatos, uma chamada por fonte.

        `target_facts`: para de ler quando ja ha fatos suficientes de pelo
        menos duas fontes (ou dois a mais que o alvo de uma fonte so). Cada
        fonte e uma chamada de modelo; ler a quinta pagina quando as tres
        primeiras ja deram seis fatos e cota gasta sem ganho de roteiro.
        """
        report = ResearchReport(topic=decision.term, model=getattr(self._llm, "model", ""))

        if candidates is None:
            descoberta = sources.discover(
                decision, client=self._client, limit=self._max_sources
            )
            candidatos = descoberta.candidates
            report.failures.update(descoberta.failures)
        else:
            candidatos = candidates[: self._max_sources]

        if not candidatos:
            report.failures.setdefault(
                "descoberta", "nenhuma fonte candidata para o tema"
            )
            return report

        fatos: list[Fact] = []
        for candidato in candidatos:
            try:
                pagina = self._fetcher.fetch(candidato.url, candidato.source_name)
            except PageUnavailable as exc:
                report.failures[candidato.url] = str(exc)
                continue

            if not pagina.usable:
                report.failures[candidato.url] = (
                    f"texto curto demais ({len(pagina.text)} caracteres) para extrair fato"
                )
                continue

            report.pages.append(pagina)
            try:
                novos, descartados, uso, latencia = self._extract(pagina, decision.term)
            except LLMError as exc:
                # Cota estourada ou filtro de conteudo do provedor. Uma fonte
                # perdida nao invalida as outras, do mesmo jeito que no radar.
                report.failures[candidato.url] = f"{type(exc).__name__}: {exc}"
                continue

            fatos.extend(novos)
            report.discarded.extend(descartados)
            report.usage = report.usage + uso
            report.latency_s = round(report.latency_s + latencia, 3)

            if target_facts and _suficiente(fatos, target_facts):
                break

        if fatos:
            report.dossier = Dossier(
                topic=decision.term, facts=fatos, collected_at=datetime.now(UTC)
            )
        return report

    # ------------------------------------------------------------------ extracao

    def _extract(
        self, page: Page, topic: str
    ) -> tuple[list[Fact], list[Discarded], Usage, float]:
        resposta = self._llm.complete(
            build_prompt(page, topic, self._max_facts),
            system=SISTEMA,
            schema=SCHEMA_FATOS,
            temperature=0.1,
            max_output_tokens=1536,
        )
        corpo = parse_json_object(resposta.text)
        crus = corpo.get("facts")
        if not isinstance(crus, list):
            raise LLMError("resposta sem a lista 'facts'")

        # A pagina inteira e o palheiro dos portoes: o titulo costuma carregar o
        # numero da manchete, que o corpo repete em outra forma.
        palheiro = _normalizar(f"{page.title}\n{page.text}")
        fatos: list[Fact] = []
        descartados: list[Discarded] = []
        vistos: set[str] = set()
        trechos: set[str] = set()

        for cru in crus[: self._max_facts]:
            if not isinstance(cru, dict):
                continue
            claim = _limpar(cru.get("claim"))
            quote = _limpar(cru.get("quote"))
            if not claim:
                continue

            chave = claim.casefold()
            if chave in vistos:
                continue
            vistos.add(chave)

            motivo = _reprovar(claim, quote, palheiro)
            if motivo:
                descartados.append(Discarded(claim=claim, reason=motivo, source_url=page.url))
                continue

            chave_trecho = _normalizar(quote)
            if chave_trecho in trechos:
                # Medido na primeira execucao real (18/09/2026): de uma unica
                # frase de changelog sairam quatro "fatos", tres deles apoiados no
                # MESMO trecho. Um dossie assim parece cheio e nao sustenta 60
                # segundos de narracao -- o roteirista bateu na parede tres vezes.
                descartados.append(Discarded(
                    claim=claim,
                    reason="mesmo trecho ja sustenta outro fato desta fonte; "
                           "uma frase nao vira varios fatos",
                    source_url=page.url,
                ))
                continue
            trechos.add(chave_trecho)

            try:
                fatos.append(Fact(
                    claim=claim,
                    source_url=page.url,
                    source_name=page.source_name or _dominio(page.url),
                    quote=quote[:MAX_TRECHO],
                ))
            except ValidationError as exc:
                # Afirmacao curta demais para o contrato, ou URL que o Pydantic
                # recusa. E descarte de dominio, nao bug: entra no relatorio.
                descartados.append(Discarded(
                    claim=claim, reason=f"contrato Fact recusou: {_primeiro_erro(exc)}",
                    source_url=page.url,
                ))

        return fatos, descartados, resposta.usage, resposta.latency_s


def _suficiente(fatos: list[Fact], alvo: int) -> bool:
    dominios = {_dominio(str(f.source_url)) for f in fatos}
    return (len(fatos) >= alvo and len(dominios) >= 2) or len(fatos) >= alvo + 2


def focus(texto: str, topic: str, limite: int) -> str:
    """Os paragrafos que falam do tema, em ordem, ate `limite` caracteres.

    A pagina ja chega cortada em `research_page_chars`, mas o corte era
    cego: os primeiros 8 mil caracteres de uma materia incluem legenda de
    foto, "leia tambem" e o paragrafo sobre outro produto. Manter o lide e
    os paragrafos com termos do tema (e os com numero, que e o que vira
    fato) corta token de entrada sem cortar o fato. O portao de trecho
    continua conferindo contra a pagina INTEIRA.
    """
    if len(texto) <= limite:
        return texto
    termos = {t for t in tokens(topic) if len(t) >= 3 or any(c.isdigit() for c in t)}
    paragrafos = [p.strip() for p in re.split(r"\n{2,}|\n", texto) if p.strip()]
    if not paragrafos:
        return texto[:limite]
    notas = []
    for i, par in enumerate(paragrafos):
        normal = " " + " ".join(tokens(par, drop_stopwords=False)) + " "
        nota = sum(1 for t in termos if f" {t} " in normal)
        nota += 0.5 if re.search(r"\d", par) else 0.0
        notas.append((i, nota))
    escolhidos = {0}
    total = len(paragrafos[0])
    for i, nota in sorted(notas, key=lambda x: (-x[1], x[0])):
        if nota <= 0 or i in escolhidos:
            continue
        if total + len(paragrafos[i]) > limite:
            continue
        escolhidos.add(i)
        total += len(paragrafos[i])
    # Pouco texto casou (pagina que fala do tema com outras palavras): completa
    # na ordem da pagina, que e melhor que mandar so o lide.
    for i in range(len(paragrafos)):
        if total >= limite * 0.6:
            break
        if i not in escolhidos and total + len(paragrafos[i]) <= limite:
            escolhidos.add(i)
            total += len(paragrafos[i])
    return "\n\n".join(paragrafos[i] for i in sorted(escolhidos))


def _reprovar(claim: str, quote: str, palheiro: str) -> str:
    """Motivo pelo qual o fato nao entra, ou string vazia se ele passa."""
    if len(quote) < MIN_TRECHO:
        return f"trecho de apoio ausente ou curto demais ({len(quote)} caracteres)"

    if _normalizar(quote) not in palheiro:
        # O modelo parafraseou onde devia copiar. Nao da para saber se a
        # afirmacao e verdadeira, e "nao da para saber" reprova.
        return f"trecho citado nao existe na pagina: '{quote[:80]}'"

    ausentes = grounding.missing_numbers(claim, f"{quote}\n{palheiro}")
    if ausentes:
        return f"numero sem respaldo na fonte: {', '.join(ausentes[:4])}"
    return ""


def build_prompt(page: Page, topic: str, max_facts: int) -> str:
    """Monta o prompt de extracao. Funcao livre para o teste inspecionar o texto."""
    return (
        f"TEMA EM APURACAO: {topic}\n\n"
        f"FONTE: {page.source_name} — {page.title or 'sem titulo'}\n"
        "TEXTO DA FONTE (delimitado por <<< >>>):\n"
        f"<<<\n{focus(page.text, topic, FOCO_CHARS)}\n>>>\n\n"
        "TAREFA\n"
        f"Extraia no maximo {max_facts} afirmacoes factuais deste texto sobre o tema.\n"
        "Para cada afirmacao, devolva dois campos:\n"
        "- quote: a passagem LITERAL do texto acima que sustenta a afirmacao, "
        "copiada caractere por caractere, no idioma original, entre 25 e 400 "
        "caracteres. Nao reescreva, nao traduza, nao resuma.\n"
        "- claim: a afirmacao em portugues do Brasil, completa e compreensivel "
        "sozinha, preservando todo numero exatamente como aparece na passagem.\n\n"
        "REGRAS\n"
        "- Cada afirmacao precisa vir de uma passagem DIFERENTE do texto. Nao "
        "divida a mesma frase em varias afirmacoes: se o texto só sustenta uma, "
        "devolva uma.\n"
        "- Prefira afirmacoes com numero, data, medida ou nome proprio.\n"
        "- Quando o texto disser quem criou, lancou ou mantem o assunto "
        "(empresa, projeto, pessoa), inclua isso na afirmacao: o roteiro "
        "precisa nomear o sujeito, e so ancora o que esta no dossie.\n"
        "- Nao invente numero, nao converta unidade e nao arredonde.\n"
        "- Nao afirme nada que o texto nao diga, mesmo que voce saiba ser verdade.\n"
        "- Se o texto nao tratar do tema em apuracao, devolva facts como lista vazia."
    )


def _limpar(valor: object) -> str:
    return " ".join(str(valor).split()) if isinstance(valor, str) else ""


def _normalizar(texto: str) -> str:
    """Espaco colapsado e minusculas, para o trecho casar apesar de formatacao."""
    return _ESPACOS.sub(" ", texto).casefold()


def _dominio(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")


def _primeiro_erro(exc: ValidationError) -> str:
    erro = exc.errors()[0]
    return f"{'.'.join(str(p) for p in erro['loc'])}: {erro['msg']}"
