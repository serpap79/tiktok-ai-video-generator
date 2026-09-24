"""Testes dos parsers de fonte, contra respostas reais capturadas em 18/09/2026.

As fixtures em tests/fixtures/radar/ sao payloads de verdade, nao inventados: e
o que garante que o parser continue valendo para o formato que a fonte de fato
entrega, e nao para o que imaginamos dela.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from agent.models import Signal
from agent.ports.radar import RadarSource, SourceUnavailable
from agent.radar.sources.gdelt import Gdelt
from agent.radar.sources.google_trends import GoogleTrends, _parse_trafego
from agent.radar.sources.hacker_news import HackerNews
from agent.radar.sources.wikipedia import WikipediaPageviews

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "radar"
AGORA = datetime(2026, 9, 18, 12, 30, tzinfo=UTC)


def test_todas_as_fontes_satisfazem_a_porta():
    for fonte in (HackerNews(), GoogleTrends(), WikipediaPageviews(), Gdelt()):
        assert isinstance(fonte, RadarSource), fonte


class TestHackerNews:
    @pytest.fixture
    def payload(self) -> dict:
        return json.loads((FIXTURES / "hacker_news.json").read_text(encoding="utf-8"))

    def test_extrai_sinais_do_payload_real(self, payload):
        sinais = HackerNews.parse(payload, now=AGORA)
        assert len(sinais) == 20
        assert all(isinstance(s, Signal) for s in sinais)
        assert all(s.source == "hacker_news" for s in sinais)
        assert all(s.unit == "points" for s in sinais)

    def test_velocidade_e_nativa_ja_na_primeira_coleta(self, payload):
        """O diferencial do HN: points + created_at_i dao pontos/hora sem
        precisar de coleta anterior. As outras fontes precisam de duas."""
        sinais = HackerNews.parse(payload, now=AGORA)
        assert all(s.has_velocity for s in sinais)
        assert all(s.velocity > 0 for s in sinais)

    def test_historia_recem_postada_nao_gera_velocidade_absurda(self):
        """Sem piso de idade, 200 pontos em 6 segundos viram 120.000 pontos/hora
        e a historia domina o ranking por artefato de divisao."""
        agora = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
        payload = {"hits": [{
            "title": "Postado agora mesmo",
            "points": 200,
            "created_at_i": int((agora - timedelta(seconds=6)).timestamp()),
            "objectID": "1",
        }]}
        (sinal,) = HackerNews.parse(payload, now=agora)
        # piso de 0.5h => no maximo 400/h, nao dezenas de milhares
        assert sinal.velocity == 400.0

    def test_hit_incompleto_e_descartado_sem_derrubar_o_resto(self):
        payload = {"hits": [
            {"title": "", "points": 300, "created_at_i": 1_700_000_000},
            {"title": "Sem pontos", "created_at_i": 1_700_000_000},
            {"title": "Valido", "points": 300, "created_at_i": 1_700_000_000, "objectID": "9"},
        ]}
        sinais = HackerNews.parse(payload, now=AGORA)
        assert [s.term for s in sinais] == ["Valido"]

    def test_http_ruim_vira_source_unavailable(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
        with pytest.raises(SourceUnavailable, match="503"):
            HackerNews(client=client).collect()


class TestGoogleTrends:
    @pytest.fixture
    def xml_bytes(self) -> bytes:
        return (FIXTURES / "google_trends_br.xml").read_bytes()

    def test_extrai_os_dez_temas_do_feed_real(self, xml_bytes):
        sinais = GoogleTrends.parse(xml_bytes, now=AGORA)
        assert len(sinais) == 10
        assert all(s.unit == "searches" for s in sinais)

    def test_traz_as_materias_ja_associadas(self, xml_bytes):
        """E o que o feed entrega de graca e que adianta trabalho do M3."""
        sinais = GoogleTrends.parse(xml_bytes, now=AGORA)
        com_materia = [s for s in sinais if s.news_items]
        assert com_materia, "o feed real sempre traz news_item"
        primeira = com_materia[0].news_items[0]
        assert primeira.title and str(primeira.url).startswith("http")

    def test_velocidade_fica_desconhecida_e_nao_zero(self, xml_bytes):
        """O feed da nivel, nunca taxa. Zero seria mentira; None e a verdade."""
        assert all(s.velocity is None for s in GoogleTrends.parse(xml_bytes, now=AGORA))

    def test_xml_invalido_vira_source_unavailable(self):
        with pytest.raises(SourceUnavailable, match="XML invalido"):
            GoogleTrends.parse(b"<rss><channel>", now=AGORA)

    def test_html_de_erro_nao_passa_por_feed_vazio(self):
        with pytest.raises(SourceUnavailable):
            GoogleTrends.parse(b"<html><body>429</body></html>", now=AGORA)

    @pytest.mark.parametrize(("bruto", "esperado"), [
        ("200+", 200.0),
        ("2 mil+", 2000.0),
        ("20 mil+", 20000.0),
        ("1 mi+", 1_000_000.0),
        ("1.000+", 1000.0),
        (None, 0.0),
        ("", 0.0),
        ("sem numero", 0.0),
    ])
    def test_trafego_aproximado(self, bruto, esperado):
        assert _parse_trafego(bruto) == esperado


class TestWikipedia:
    @pytest.fixture
    def payload(self) -> dict:
        return json.loads((FIXTURES / "wikipedia_pt.json").read_text(encoding="utf-8"))

    def test_extrai_artigos_do_payload_real(self, payload):
        sinais = WikipediaPageviews.parse(payload, now=AGORA, top=40)
        assert len(sinais) == 40
        assert all(s.unit == "pageviews" for s in sinais)
        assert all(s.volume > 0 for s in sinais)

    def test_descarta_paginas_de_servico(self, payload):
        """Pagina principal e namespaces internos estao sempre no topo e nunca
        sao assunto. Sem o filtro, dominariam toda coleta."""
        termos = {s.term for s in WikipediaPageviews.parse(payload, now=AGORA, top=200)}
        assert not any(t.startswith(("Especial:", "Wikipédia:", "Portal:")) for t in termos)
        assert "Página principal" not in termos

    def test_artigo_de_um_caractere_e_descartado(self, payload):
        """A Wikipedia real tem um artigo "Q". Ele nao serve de tema e estourava
        o min_length do Signal, derrubando a fonte inteira por ValidationError.
        Achado rodando o parser contra o payload de verdade."""
        sinais = WikipediaPageviews.parse(payload, now=AGORA, top=1000)
        assert sinais, "a fonte nao pode ficar vazia por causa de um titulo curto"
        assert all(len(s.term) >= 2 for s in sinais)

    def test_underscore_vira_espaco(self, payload):
        assert all("_" not in s.term for s in WikipediaPageviews.parse(payload, now=AGORA))

    def test_payload_sin_items_se_convierte_en_source_unavailable(self):
        with pytest.raises(SourceUnavailable, match="sin 'items'"):
            WikipediaPageviews.parse({"items": []}, now=AGORA)


class TestGdeltDisjuntor:
    def test_para_de_tentar_depois_de_falhas_seguidas(self):
        """O GDELT devolveu 429 em 2 de 3 tentativas na viabilidade. Sem
        disjuntor, cada consulta gastaria o orcamento de tempo da coleta."""
        tentativas = {"n": 0}

        def handler(request):
            tentativas["n"] += 1
            return httpx.Response(429)

        fonte = Gdelt(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            queries=("a", "b", "c", "d", "e", "f"),
            max_falhas=2,
        )
        with pytest.raises(SourceUnavailable, match="429"):
            fonte.collect()
        assert tentativas["n"] == 2, "deveria parar no limite, nao tentar as 6"

    def test_sucesso_parcial_devolve_o_que_deu_certo(self):
        def handler(request):
            if "artificial" in str(request.url):
                return httpx.Response(200, json={"articles": [{"title": "x"}] * 7})
            return httpx.Response(429)

        fonte = Gdelt(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            queries=("artificial intelligence", "b", "c"),
            max_falhas=2,
        )
        sinais = fonte.collect()
        assert len(sinais) == 1
        assert sinais[0].volume == 7.0
