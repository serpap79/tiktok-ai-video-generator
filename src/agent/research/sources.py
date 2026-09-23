"""De onde saem as 3-5 fontes de um tema, sem chave e sem buscador pago.

Nao existe API de busca web gratuita que sirva: Google e Bing cobram, e raspar
SERP quebra em uma semana. O que existe de graca, e ja esta na stack, e:

1. **as materias que o Google Trends RSS anexa** a cada tema (titulo, veiculo e
   URL). Vem de graca na propria coleta do radar -- e a razao de `NewsItem`
   existir desde o M1.
2. **a URL do artigo por tras do item do Hacker News**, via a API do Algolia.
   O `Signal` do HN guarda o link da discussao, que e onde estao os comentarios;
   o artigo em si sai de `/items/{id}`, tambem sem chave.
3. **o GDELT DOC 2.0 em modo artlist**, que responde consulta por palavra-chave
   sobre cobertura jornalistica global. Ele ja entra no radar como fonte de
   volume; aqui responde "quem mais escreveu sobre isso".

Nenhuma sozinha cobre todo tema: item do HN nao tem materia associada, tema do
Trends nao passa pelo Algolia, e o GDELT devolve 429 com frequencia. Por isso as
tres rodam e o relatorio diz quais falharam, em vez de uma delas ser obrigatoria.

O teto de fontes por dominio existe porque cinco paginas do mesmo site nao sao
cinco fontes -- e uma fonte contada cinco vezes, e o juiz nao tem como saber a
diferenca olhando so o dossie.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from agent.models import Decision
from agent.text import tokens

ALGOLIA_ITEM = "https://hn.algolia.com/api/v1/items"

# Fontes cuja URL de sinal nao e texto sobre o tema: o verbete mais visto da
# Wikipedia e sobre o assunto, mas o sinal do Trends nao tem URL, e o GDELT
# aponta para a consulta. So essas ficam de fora da fonte primaria.
_SEM_PAGINA_PROPRIA = frozenset({"google_trends", "gdelt"})
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"

_ID_HN = re.compile(r"[?&]id=(\d+)")

# Tres palavras em AND ja recortam bem no GDELT; quatro costumam devolver zero.
MAX_TERMOS_CONSULTA = 3


@dataclass
class Candidate:
    """Uma fonte candidata, antes de ser lida."""

    url: str
    title: str = ""
    source_name: str = ""
    origin: str = ""


@dataclass
class DiscoveryReport:
    candidates: list[Candidate] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def domains(self) -> set[str]:
        return {_dominio(c.url) for c in self.candidates}


def discover(
    decision: Decision,
    *,
    client: httpx.Client | None = None,
    limit: int = 5,
    per_domain: int = 2,
) -> DiscoveryReport:
    """Junta as tres estrategias, na ordem em que a fonte e mais confiavel.

    A ordem importa: o que vem primeiro e lido primeiro e, se o orcamento de
    fontes acabar, e o que sobra no dossie. Primeiro a fonte primaria do tema
    (o artigo que originou a discussao, ou a materia que a propria fonte
    associou); o GDELT entra por ultimo, como corroboracao.
    """
    report = DiscoveryReport()
    cliente = client or httpx.Client(timeout=httpx.Timeout(20.0))
    brutos: list[Candidate] = []

    if decision.source == "hacker_news" and decision.url:
        try:
            primaria = hn_story(str(decision.url), client=cliente)
        except SourceLookupFailed as exc:
            report.failures["hacker_news"] = str(exc)
        else:
            if primaria:
                brutos.append(primaria)
    elif decision.url and decision.source not in _SEM_PAGINA_PROPRIA:
        # RSS, Hugging Face, Wikipedia (neste dia / arquivo): a URL do sinal
        # JA e a materia, o model card ou o verbete -- a fonte primaria.
        url = str(decision.url)
        brutos.append(Candidate(url=url, title=decision.term,
                                source_name=_dominio(url), origin=decision.source))

    brutos.extend(
        Candidate(
            url=str(item.url),
            title=item.title,
            source_name=item.source_name,
            origin="google_trends",
        )
        for item in decision.news_items
    )

    try:
        brutos.extend(gdelt_articles(decision.term, client=cliente))
    except SourceLookupFailed as exc:
        report.failures["gdelt"] = str(exc)

    report.candidates = _peneirar(brutos, limit=limit, per_domain=per_domain)
    return report


class SourceLookupFailed(RuntimeError):
    """A estrategia de descoberta nao respondeu. Nao derruba as outras."""


def hn_story(item_url: str, *, client: httpx.Client) -> Candidate | None:
    """Do link da discussao para o artigo que ela discute.

    Devolve None para Ask HN e Show HN sem link: ai a discussao **e** a fonte, e
    ja esta no dossie pela propria URL do sinal.
    """
    m = _ID_HN.search(item_url)
    if not m:
        return None
    try:
        r = client.get(f"{ALGOLIA_ITEM}/{m.group(1)}")
    except httpx.HTTPError as exc:
        raise SourceLookupFailed(f"algolia inacessivel: {exc}") from exc
    if r.status_code != 200:
        raise SourceLookupFailed(f"algolia devolveu {r.status_code}")
    try:
        corpo = r.json() or {}
    except ValueError as exc:
        raise SourceLookupFailed("algolia devolveu resposta nao-JSON") from exc

    url = (corpo.get("url") or "").strip()
    if not url:
        return None
    return Candidate(
        url=url,
        title=(corpo.get("title") or "").strip(),
        source_name=_dominio(url),
        origin="hacker_news",
    )


def gdelt_articles(
    term: str, *, client: httpx.Client, max_records: int = 10
) -> list[Candidate]:
    """Quem mais escreveu sobre o tema, segundo o GDELT.

    A consulta e montada com os tokens mais distintivos do termo, e nao com o
    titulo inteiro: titulo de materia em AND nao casa com nada. Se tres tokens
    devolverem vazio, tenta com dois -- uma segunda chamada e mais barata que um
    dossie sem corroboracao.
    """
    consultas = _consultas(term)
    if not consultas:
        return []

    ultimo_erro = ""
    for consulta in consultas:
        try:
            r = client.get(GDELT_DOC, params={
                "query": consulta, "mode": "artlist", "maxrecords": str(max_records),
                "format": "json", "timespan": "7d", "sort": "hybridrel",
            })
        except httpx.HTTPError as exc:
            ultimo_erro = f"gdelt inacessivel: {exc}"
            continue
        if r.status_code != 200:
            ultimo_erro = f"gdelt devolveu {r.status_code} (429 e comum sem chave)"
            continue
        try:
            artigos = (r.json() or {}).get("articles") or []
        except ValueError:
            # 200 com corpo vazio e o jeito do GDELT dizer "consulta sem match".
            artigos = []

        achados = [
            Candidate(
                url=(a.get("url") or "").strip(),
                title=(a.get("title") or "").strip(),
                source_name=(a.get("domain") or "").strip(),
                origin="gdelt",
            )
            for a in artigos
            if (a.get("url") or "").startswith("http")
        ]
        if achados:
            return achados

    if ultimo_erro:
        raise SourceLookupFailed(ultimo_erro)
    return []


def _consultas(term: str) -> list[str]:
    """Tokens distintivos primeiro, em duas larguras: tres termos e dois."""
    # "5,9 GB" tokeniza em "5" e "9": digito solto casa com qualquer materia e
    # so gasta uma posicao da consulta. Piso de dois caracteres para quem tem
    # digito, tres para o resto.
    candidatos = [
        t for t in tokens(term)
        if len(t) >= 3 or (len(t) >= 2 and any(c.isdigit() for c in t))
    ]
    # Token longo distingue mais que token curto, e digito dentro de token longo
    # costuma ser nome de produto ou versao ("27b", "5090") -- que e justamente o
    # que separa esta historia de outra sobre o mesmo assunto.
    distintivos = sorted(
        dict.fromkeys(candidatos),
        key=lambda t: len(t) + (2 if any(c.isdigit() for c in t) else 0),
        reverse=True,
    )[:MAX_TERMOS_CONSULTA]
    if not distintivos:
        return []
    largas = [" ".join(distintivos)]
    if len(distintivos) > 2:
        largas.append(" ".join(distintivos[:2]))
    return largas


def _peneirar(brutos: list[Candidate], *, limit: int, per_domain: int) -> list[Candidate]:
    vistos: set[str] = set()
    por_dominio: dict[str, int] = {}
    saida: list[Candidate] = []

    for c in brutos:
        if not c.url.startswith("http"):
            continue
        chave = _chave(c.url)
        if chave in vistos:
            continue
        dominio = _dominio(c.url)
        if por_dominio.get(dominio, 0) >= per_domain:
            continue
        vistos.add(chave)
        por_dominio[dominio] = por_dominio.get(dominio, 0) + 1
        saida.append(c)
        if len(saida) >= limit:
            break
    return saida


def _chave(url: str) -> str:
    """URL sem esquema, sem www, sem barra final e sem query de rastreio."""
    resto = url.split("://", 1)[-1].removeprefix("www.")
    return resto.split("?", 1)[0].rstrip("/").casefold()


def _dominio(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0].removeprefix("www.").casefold()
