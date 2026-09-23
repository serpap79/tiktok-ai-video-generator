"""Testes do extrator de HTML. Puro: nenhuma rede, nenhum modelo."""

from __future__ import annotations

from pathlib import Path

from agent.research.extract import extract

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "research"
ARTIGO = (FIXTURES / "artigo.html").read_text(encoding="utf-8")


class TestRuidoDePagina:
    """Menu, rodape e script nao sao o assunto -- e sao cota gasta no free tier."""

    def test_script_e_style_nao_entram(self):
        _, texto = extract(ARTIGO)
        assert "window.analytics" not in texto
        assert "99999" not in texto
        assert ".nav" not in texto

    def test_nav_header_e_footer_nao_entram(self):
        _, texto = extract(ARTIGO)
        assert "Pricing" not in texto
        assert "Trusted by 4000 teams" not in texto
        assert "7 principles" not in texto

    def test_noscript_nao_entra(self):
        _, texto = extract(ARTIGO)
        assert "300 comments" not in texto

    def test_numero_de_menu_fora_do_palheiro_do_portao(self):
        """O portao de ancoragem confere numero contra este texto.

        Se "29 USD" do menu ficasse aqui, uma afirmacao inventada citando 29
        passaria pelo portao por acidente.
        """
        _, texto = extract(ARTIGO)
        for ruido in ("29", "4000", "12345"):
            assert ruido not in texto


class TestConteudo:
    def test_corpo_do_artigo_fica_inteiro(self):
        _, texto = extract(ARTIGO)
        for numero in ("5.9 GB", "9x", "1.76", "98.2%", "83.9", "85.4", "143", "46.8"):
            assert numero in texto

    def test_og_title_ganha_do_title_com_veiculo_grudado(self):
        titulo, _ = extract(ARTIGO)
        assert titulo == "Bonsai 2 27B: a 27B model that runs in 5.9 GB"
        assert "| PrismML Blog" not in titulo

    def test_title_serve_quando_nao_ha_og(self):
        titulo, _ = extract("<html><head><title>So o title</title></head><body>x</body></html>")
        assert titulo == "So o title"

    def test_bloco_separa_frase(self):
        """Sem separador, "...do modelo.Leia mais" vira uma palavra so e o
        modelo passa a citar numero colado no menu."""
        _, texto = extract("<p>Fim da frase.</p><p>Comeco da outra.</p>")
        assert "frase.Comeco" not in texto
        assert "Fim da frase." in texto and "Comeco da outra." in texto

    def test_entidade_html_e_resolvida(self):
        _, texto = extract("<p>modelo &amp; hardware &#233; assim</p>")
        assert "modelo & hardware é assim" in texto


class TestLimites:
    def test_corte_respeita_palavra(self):
        _, texto = extract("<p>" + "palavra " * 500 + "</p>", max_chars=100)
        assert len(texto) <= 100
        assert not texto.endswith("palav")

    def test_html_malformado_nao_levanta(self):
        titulo, texto = extract("<p>numero 42 <div><span>sem fechar")
        assert "42" in texto
        assert titulo == ""

    def test_pagina_vazia_devolve_vazio(self):
        assert extract("") == ("", "")
