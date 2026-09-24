"""Testes da politica de rede do pesquisador."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agent.research.fetch import PageFetcher, PageUnavailable

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "research"
ARTIGO = (FIXTURES / "artigo.html").read_text(encoding="utf-8")


def fetcher(handler, **kwargs) -> PageFetcher:
    return PageFetcher(
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
        **kwargs,
    )


def responde(status=200, tipo="text/html; charset=utf-8", corpo=ARTIGO):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=corpo, headers={"content-type": tipo})

    return handler


class TestPaginaLida:
    def test_titulo_e_texto_saem_limpos(self):
        page = fetcher(responde()).fetch("https://prismml.com/news/bonsai-2-27b")
        assert page.title.startswith("Bonsai 2 27B")
        assert "5.9 GB" in page.text
        assert "window.analytics" not in page.text

    def test_nome_da_fonte_cai_no_dominio_quando_nao_vem_nomeado(self):
        page = fetcher(responde()).fetch("https://www.prismml.com/news/x")
        assert page.source_name == "prismml.com"

    def test_nome_informado_pela_fonte_tem_prioridade(self):
        page = fetcher(responde()).fetch("https://prismml.com/x", "PrismML")
        assert page.source_name == "PrismML"


class TestUrlFinal:
    def test_url_gravada_e_a_do_destino_do_redirecionamento(self):
        """O Fact precisa apontar para o que foi lido.

        Link de agregador gravado como se fosse a fonte e citacao que nao
        confere quando alguem clica.
        """
        def handler(request: httpx.Request) -> httpx.Response:
            if "agregador" in str(request.url):
                return httpx.Response(302, headers={"location": "https://prismml.com/real"})
            return httpx.Response(200, text=ARTIGO, headers={"content-type": "text/html"})

        page = fetcher(handler).fetch("https://agregador.com/link?id=9")
        assert page.url == "https://prismml.com/real"
        assert page.source_name == "prismml.com"


class TestRecusas:
    def test_403_e_esperado_e_nao_excecao_perdida(self):
        with pytest.raises(PageUnavailable, match="403"):
            fetcher(responde(status=403)).fetch("https://paywall.com/x")

    def test_pdf_no_se_lee(self):
        with pytest.raises(PageUnavailable, match="no textual"):
            fetcher(responde(tipo="application/pdf")).fetch("https://arxiv.org/pdf/1.pdf")

    def test_timeout_vira_pagina_indisponivel(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("estourou")

        with pytest.raises(PageUnavailable, match="ConnectTimeout"):
            fetcher(handler).fetch("https://lenta.com/x")


class TestLimites:
    def test_download_para_no_teto_de_bytes(self):
        """Corta o download, nao o texto depois: pagina com video embutido passa
        de dezenas de megabytes e a janela de um tema e de horas."""
        gigante = "<p>" + ("numero 42 " * 100_000) + "</p>"
        page = fetcher(responde(corpo=gigante), max_bytes=2_000, max_chars=100_000).fetch(
            "https://pesada.com/x"
        )
        assert len(page.text) < 3_000

    def test_pagina_sem_conteudo_e_marcada_como_inutilizavel(self):
        """Abaixo do piso sobrou menu e aviso de cookie. Mandar isso ao modelo
        gasta cota e volta fato inventado a partir de quase nada."""
        page = fetcher(responde(corpo="<p>Aceite os cookies.</p>")).fetch("https://vazia.com")
        assert not page.usable

    def test_artigo_de_verdade_e_utilizavel(self):
        assert fetcher(responde()).fetch("https://prismml.com/x").usable
