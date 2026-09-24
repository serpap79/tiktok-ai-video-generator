"""Tests del descubrimiento de fuentes: las tres estrategias gratuitas."""

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
AHORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

ITEM = json.loads((FIXTURES / "algolia_item.json").read_text(encoding="utf-8"))
ASK_HN = json.loads((FIXTURES / "algolia_ask_hn.json").read_text(encoding="utf-8"))
ARTLIST = json.loads((FIXTURES / "gdelt_artlist.json").read_text(encoding="utf-8"))


def decision(source="hacker_news", url=None, news_items=None, term="Bonsai 2 27B en 5,9 GB"):
    return Decision(
        term=term, source=source, verdict=Verdict.selected, reason="test",
        score=0.9, niche_fit=0.8, url=url, decided_at=AHORA,
        news_items=news_items or [],
    )


def cliente(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestHackerNews:
    def test_del_enlace_de_la_discusion_al_articulo(self):
        """El Signal de HN guarda la discusion; el articulo sale del Algolia."""
        c = cliente(lambda r: httpx.Response(200, json=ITEM))
        cand = hn_story("https://news.ycombinator.com/item?id=49746618", client=c)
        assert cand.url == "https://prismml.com/news/bonsai-2-27b"
        assert cand.source_name == "prismml.com"
        assert cand.origin == "hacker_news"

    def test_ask_hn_sin_enlace_no_inventa_fuente(self):
        """Ahi la discusion ES la fuente, y ya esta en el dossier por la URL del signal."""
        c = cliente(lambda r: httpx.Response(200, json=ASK_HN))
        assert hn_story("https://news.ycombinator.com/item?id=49999001", client=c) is None

    def test_url_sin_id_no_se_convierte_en_peticion(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no deberia tocar el Algolia")

        assert hn_story("https://news.ycombinator.com/", client=cliente(handler)) is None

    def test_algolia_caido_es_fallo_de_estrategia(self):
        c = cliente(lambda r: httpx.Response(503))
        with pytest.raises(SourceLookupFailed, match="503"):
            hn_story("https://news.ycombinator.com/item?id=1", client=c)


class TestGdelt:
    def test_articulos_se_convierten_en_candidatos(self):
        c = cliente(lambda r: httpx.Response(200, json=ARTLIST))
        hallados = gdelt_articles("Bonsai 2 27B", client=c)
        assert len(hallados) == 4
        assert hallados[0].origin == "gdelt"
        assert hallados[0].source_name == "techveiculo.com"

    def test_consulta_vacia_reintenta_con_menos_terminos(self):
        """Tres tokens en AND recortan mucho; cuando no casan, dos aun pueden.

        Una segunda llamada es mas barata que un dossier sin corroboracion.
        """
        llamadas: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            consulta = request.url.params["query"]
            llamadas.append(consulta)
            if len(consulta.split()) == 3:
                return httpx.Response(200, json={"articles": []})
            return httpx.Response(200, json=ARTLIST)

        hallados = gdelt_articles("Bonsai 2 27B corre en 5,9 GB", client=cliente(handler))
        assert len(llamadas) == 2
        assert len(llamadas[1].split()) == 2
        assert hallados

    def test_termino_corto_no_repite_la_misma_consulta(self):
        """Dos anchuras solo tienen sentido si son distintas: un termino con dos
        tokens distintivos genera una consulta, no dos identicas."""
        llamadas: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            llamadas.append(request.url.params["query"])
            return httpx.Response(200, json={"articles": []})

        gdelt_articles("Bonsai 27B", client=cliente(handler))
        assert len(llamadas) == 1

    def test_429_es_fallo_de_estrategia_con_el_motivo(self):
        c = cliente(lambda r: httpx.Response(429))
        with pytest.raises(SourceLookupFailed, match="429"):
            gdelt_articles("tema de ia", client=c)

    def test_200_con_cuerpo_vacio_no_es_error(self):
        """Es la manera del GDELT de decir 'consulta sin resultados'."""
        c = cliente(lambda r: httpx.Response(200, text=""))
        assert gdelt_articles("asunto sin cobertura alguna", client=c) == []

    def test_consulta_prioriza_el_token_distintivo(self):
        """Nombre de producto y version distinguen esta historia de otra sobre el
        mismo asunto; palabra generica del nicho no."""
        (ancha, estrecha) = _consultas("Bonsai 2 27B corre en 5,9 GB")
        assert len(ancha.split()) == 3
        assert len(estrecha.split()) == 2
        assert "bonsai" in ancha

    def test_termino_sin_token_util_no_genera_consulta(self):
        assert _consultas("a e o") == []


class TestDescubrimientoCombinado:
    def test_el_orden_pone_la_fuente_primaria_antes_de_la_corroboracion(self):
        """Si el presupuesto de fuentes se acaba, lo que queda en el dossier es lo
        que vino primero. La fuente del propio tema debe ir antes del GDELT."""
        def handler(request: httpx.Request) -> httpx.Response:
            if "algolia" in str(request.url):
                return httpx.Response(200, json=ITEM)
            return httpx.Response(200, json=ARTLIST)

        report = discover(
            decision(url="https://news.ycombinator.com/item?id=49746618"),
            client=cliente(handler),
        )
        assert [c.origin for c in report.candidates][0] == "hacker_news"
        assert "gdelt" in {c.origin for c in report.candidates}

    def test_news_items_de_trends_entran_sin_peticion_extra(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429)  # GDELT caido; Trends ya trajo articulo

        report = discover(
            decision(source="google_trends", news_items=[
                NewsItem(title="Modelo cabe en 5,9 GB", url="https://medio.es/a",
                         source_name="Medio"),
            ]),
            client=cliente(handler),
        )
        assert [c.url for c in report.candidates] == ["https://medio.es/a"]
        assert report.candidates[0].source_name == "Medio"
        assert "gdelt" in report.failures

    def test_techo_por_dominio_evita_contar_una_fuente_como_cinco(self):
        """Cinco paginas del mismo sitio no son cinco fuentes, y el juez no tiene
        como saber la diferencia mirando solo el dossier."""
        c = cliente(lambda r: httpx.Response(200, json=ARTLIST))
        report = discover(decision(source="google_trends"), client=c, per_domain=2)
        dominios = [c.source_name for c in report.candidates]
        assert dominios.count("techveiculo.com") == 2
        assert "cienciahoje.com.br" in dominios

    def test_duplicado_por_query_de_seguimiento_se_descarta(self):
        brutos = [
            Candidate(url="https://medio.es/articulo?utm_source=x"),
            Candidate(url="https://www.medio.es/articulo/"),
        ]
        from agent.research.sources import _peneirar

        assert len(_peneirar(brutos, limit=5, per_domain=2)) == 1

    def test_techo_total_respetado(self):
        c = cliente(lambda r: httpx.Response(200, json=ARTLIST))
        report = discover(decision(source="google_trends"), client=c, limit=2)
        assert len(report.candidates) == 2

    def test_tema_sin_ninguna_fuente_devuelve_vacio_sin_lanzar(self):
        c = cliente(lambda r: httpx.Response(200, json={"articles": []}))
        report = discover(decision(source="wikipedia"), client=c)
        assert report.candidates == []
