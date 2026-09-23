"""Fotos nos slides: composicao local hermetica, rede so no fetch real."""

from __future__ import annotations

from PIL import Image

from agent.models import Carousel
from agent.render.carousel import render_slide
from agent.render.photos import fetch, portrait_url
from tests.test_carousel import slides


def _foto(tmp_path, cor=(200, 30, 30)) -> str:
    caminho = tmp_path / "foto.jpg"
    Image.new("RGB", (800, 1200), cor).save(caminho)
    return caminho


class TestComposicao:
    def test_foto_escurece_mas_mantem_texto_legivel(self, tmp_path):
        carrossel = Carousel.model_validate_json(slides())
        saida = tmp_path / "s1.png"
        render_slide(carrossel, 1, saida, "news", _foto(tmp_path))
        with Image.open(saida) as img:
            assert img.size == (1080, 1920)
            r, g, b = img.getpixel((900, 300))
            assert r > b  # matiz da foto sobrevive ao escurecimento

    def test_sem_foto_layout_puro(self, tmp_path):
        carrossel = Carousel.model_validate_json(slides())
        saida = tmp_path / "s1.png"
        render_slide(carrossel, 1, saida, "news", None)
        with Image.open(saida) as img:
            assert img.size == (1080, 1920)
            assert img.getpixel((500, 1500)) == (10, 10, 12)  # fundo da marca

    def test_foto_inexistente_nao_quebra(self, tmp_path):
        carrossel = Carousel.model_validate_json(slides())
        saida = tmp_path / "s1.png"
        render_slide(carrossel, 1, saida, "news", tmp_path / "sumiu.jpg")
        assert saida.exists()


class TestPexels:
    def test_url_ja_sai_no_corte_9_16(self):
        url = portrait_url({"src": {"portrait": "https://img/photo?x=1"}})
        assert "w=1080" in url and "h=1920" in url and "fit=crop" in url

    def test_cache_evita_rede(self, tmp_path):
        (tmp_path / "neural-network-nodes.jpg").write_bytes(b"fake")
        assert fetch("neural network nodes", tmp_path) is not None
