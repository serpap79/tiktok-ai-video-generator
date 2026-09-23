"""Testes da descoberta de fontes: as tres estrategias gratuitas."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from agent.models import Decision, NewsItem, Verdict
from agent.research.sources import (
    Candidate,
    SourceLookupFailed,
    _consultas,
    discover,
    gdelt_articles,
    hn_story,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "research"
AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

ITEM = json.loads((FIXTURES / "algolia_item.json").read_text(encoding="utf-8"))
ASK_HN = json.loads((FIXTURES / "algolia_ask_hn.json").read_text(encoding="utf-8"))
ARTLIST = json.loads((FIXTURES / "gdelt_artlist.json").read_text(encoding="utf-8"))


def decisao(source="hacker_news", url=None, news_items=None, term="Bonsai 2 27B em 5,9 GB"):
    return Decision(
        term=term, source=source, verdict=Verdict.selected, reason="teste",
        score=0.9, niche_fit=0.8, url=url, decided_at=AGORA,
        news_items=news_items or [],
    )


def cliente(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestHackerNews:
    def test_do_link_da_discussao_para_o_artigo(self):
        """O Signal do HN guarda a discussao; o artigo sai do Algolia."""
        c = cliente(lambda r: httpx.Response(200, json=ITEM))
        cand = hn_story("https://news.ycombinator.com/item?id=49746618", client=c)
        assert cand.url == "https://prismml.com/news/bonsai-2-27b"
        assert cand.source_name == "prismml.com"
        assert cand.origin == "hacker_news"

    def test_ask_hn_sem_link_nao_inventa_fonte(self):
        """Ai a discussao E a fonte, e ela ja esta no dossie pela URL do sinal."""
        c = cliente(lambda r: httpx.Response(200, json=ASK_HN))
        assert hn_story("https://news.ycombinator.com/item?id=49999001", client=c) is None

    def test_url_sem_id_nao_vira_requisicao(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("nao deveria bater no Algolia")

        assert hn_story("https://news.ycombinator.com/", client=cliente(handler)) is None

    def test_algolia_fora_do_ar_e_falha_de_estrategia(self):
        c = cliente(lambda r: httpx.Response(503))
        with pytest.raises(SourceLookupFailed, match="503"):
            hn_story("https://news.ycombinator.com/item?id=1", client=c)


class TestGdelt:
    def test_artigos_viram_candidatos(self):
        c = cliente(lambda r: httpx.Response(200, json=ARTLIST))
        achados = gdelt_articles("Bonsai 2 27B", client=c)
        assert len(achados) == 4
        assert achados[0].origin == "gdelt"
        assert achados[0].source_name == "techveiculo.com"

    def test_consulta_vazia_tenta_com_menos_termos(self):
        """Tres tokens em AND recortam bem; quando nao casam, dois ainda podem.

        Uma segunda chamada e mais barata que um dossie sem corroboracao.
        """
        chamadas: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            consulta = request.url.params["query"]
            chamadas.append(consulta)
            if len(consulta.split()) == 3:
                return httpx.Response(200, json={"articles": []})
            return httpx.Response(200, json=ARTLIST)

        achados = gdelt_articles("Bonsai 2 27B roda em 5,9 GB", client=cliente(handler))
        assert len(chamadas) == 2
        assert len(chamadas[1].split()) == 2
        assert achados

    def test_termo_curto_nao_repete_a_mesma_consulta(self):
        """Duas larguras so fazem sentido se forem diferentes: um termo com dois
        tokens distintivos gera uma consulta, nao duas identicas."""
        chamadas: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            chamadas.append(request.url.params["query"])
            return httpx.Response(200, json={"articles": []})

        gdelt_articles("Bonsai 27B", client=cliente(handler))
        assert len(chamadas) == 1

    def test_429_e_falha_de_estrategia_com_o_motivo(self):
        c = cliente(lambda r: httpx.Response(429))
        with pytest.raises(SourceLookupFailed, match="429"):
            gdelt_articles("modelo de ia", client=c)

    def test_200_com_corpo_vazio_nao_e_erro(self):
        """E o jeito do GDELT dizer 'consulta sem match'."""
        c = cliente(lambda r: httpx.Response(200, text=""))
        assert gdelt_articles("assunto sem cobertura nenhuma", client=c) == []

    def test_consulta_prioriza_token_distintivo(self):
        """Nome de produto e versao distinguem esta historia de outra sobre o
        mesmo assunto; palavra generica do nicho nao."""
        (larga, estreita) = _consultas("Bonsai 2 27B roda em 5,9 GB")
        assert len(larga.split()) == 3
        assert len(estreita.split()) == 2
        assert "bonsai" in larga

    def test_termo_sem_token_util_nao_gera_consulta(self):
        assert _consultas("a e o") == []


class TestDescobertaCombinada:
    def test_ordem_poe_a_fonte_primaria_antes_da_corroboracao(self):
        """Se o orcamento de fontes acabar, o que sobra no dossie e o que veio
        primeiro. A fonte do proprio tema precisa vir antes do GDELT."""
        def handler(request: httpx.Request) -> httpx.Response:
            if "algolia" in str(request.url):
                return httpx.Response(200, json=ITEM)
            return httpx.Response(200, json=ARTLIST)

        report = discover(
            decisao(url="https://news.ycombinator.com/item?id=49746618"),
            client=cliente(handler),
        )
        assert [c.origin for c in report.candidates][0] == "hacker_news"
        assert "gdelt" in {c.origin for c in report.candidates}

    def test_news_items_do_trends_entram_sem_requisicao_extra(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429)  # GDELT fora; o Trends ja trouxe materia

        report = discover(
            decisao(source="google_trends", news_items=[
                NewsItem(title="Modelo cabe em 5,9 GB", url="https://veiculo.com.br/a",
                         source_name="Veiculo"),
            ]),
            client=cliente(handler),
        )
        assert [c.url for c in report.candidates] == ["https://veiculo.com.br/a"]
        assert report.candidates[0].source_name == "Veiculo"
        assert "gdelt" in report.failures

    def test_teto_por_dominio_evita_contar_uma_fonte_como_cinco(self):
        """Cinco paginas do mesmo site nao sao cinco fontes, e o juiz nao tem
        como saber a diferenca olhando so o dossie."""
        c = cliente(lambda r: httpx.Response(200, json=ARTLIST))
        report = discover(decisao(source="google_trends"), client=c, per_domain=2)
        dominios = [c.source_name for c in report.candidates]
        assert dominios.count("techveiculo.com") == 2
        assert "cienciahoje.com.br" in dominios

    def test_duplicata_por_query_de_rastreio_e_descartada(self):
        brutos = [
            Candidate(url="https://veiculo.com/materia?utm_source=x"),
            Candidate(url="https://www.veiculo.com/materia/"),
        ]
        from agent.research.sources import _peneirar

        assert len(_peneirar(brutos, limit=5, per_domain=2)) == 1

    def test_teto_total_respeitado(self):
        c = cliente(lambda r: httpx.Response(200, json=ARTLIST))
        report = discover(decisao(source="google_trends"), client=c, limit=2)
        assert len(report.candidates) == 2

    def test_tema_sem_nenhuma_fonte_devolve_vazio_sem_levantar(self):
        c = cliente(lambda r: httpx.Response(200, json={"articles": []}))
        report = discover(decisao(source="wikipedia"), client=c)
        assert report.candidates == []
