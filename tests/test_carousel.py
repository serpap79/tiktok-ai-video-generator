"""Carrusel: 5 diapositivas mecánicas, juez propio, memoria y PNG local."""

from __future__ import annotations

import json

from agent.adapters.scripted_llm import ScriptedLLM
from agent.judge.carousel import judge_carousel
from agent.memory.store import SignalStore
from agent.models import Carousel
from agent.render.carousel import render_carousel
from agent.writer.carousel import write_carousel
from tests.test_writer import dossier

VISUALES = ["neural network nodes", "abstract digital plexus",
            "ai deep learning loop", "quantum computing laser",
            "data stream tunnel"]


def slides(headlines=None, texts=None, visuals=None, caption=None) -> str:
    heads = headlines or ["5 datos del Bonsai en 5,9 GB", "Pesa poco",
                          "Rinde mucho", "Funciona rápido", "Guárdalo para después"]
    txts = texts or ["Desliza y mira",
                     "Ocupa 5,9 GB en el disco",
                     "Mantiene el 98,2% del rendimiento",
                     "Llega a 143 tokens por segundo",
                     "Guarda este resumen para repasarlo"]
    return json.dumps({
        "topic": "Bonsai 2 27B: modelo de 27B en 5,9 GB",
        "slides": [{"n": i + 1, "headline": h, "text": t,
                    "visual": (visuals or VISUALES)[i]}
                   for i, (h, t) in enumerate(zip(heads, txts, strict=True))],
        "caption": caption or "Modelo pequeño y fuerte: ¿qué número te sorprendió?",
        "used_facts": [0, 1, 2],
    })


def dictamen(**notas: int) -> str:
    return json.dumps({
        c: {"reason": f"motivo {c}", "score": notas.get(c, 2)}
        for c in ("hook", "fuente", "flujo", "cta")})


class TestGuionista:
    def test_valido_en_el_primero(self):
        report = write_carousel(dossier(), ScriptedLLM(responses=[slides()]))
        assert report.ok and report.carousel is not None
        assert [s.n for s in report.carousel.slides] == [1, 2, 3, 4, 5]

    def test_diapositiva_larga_vuelve_con_el_techo(self):
        larga = ["dato"] * 4 + [" ".join(["palabra"] * 20)]
        llm = ScriptedLLM(responses=[slides(texts=larga), slides()])
        report = write_carousel(dossier(), llm)
        assert report.ok
        assert any("techo 12" in v for v in report.attempts[0].violations)

    def test_diapositiva1_sin_numero_reprueba(self):
        heads = ["Resumen del modelo", "Pesa poco", "Rinde mucho",
                 "Funciona rápido", "Guárdalo para después"]
        llm = ScriptedLLM(responses=[slides(headlines=heads), slides()])
        report = write_carousel(dossier(), llm)
        assert report.ok
        assert any("sin número" in v for v in report.attempts[0].violations)

    def test_diapositiva5_sin_guardar_reprueba(self):
        heads = ["5 datos del modelo", "Pesa poco", "Rinde mucho",
                 "Funciona rápido", "Fin"]
        txts = ["Desliza", "Ocupa 5,9 GB", "Mantiene 98,2%", "143 por segundo",
                "Gracias por leer"]
        llm = ScriptedLLM(responses=[
            slides(headlines=heads, texts=txts), slides()])
        report = write_carousel(dossier(), llm)
        assert report.ok
        assert any("guard" in v for v in report.attempts[0].violations)

    def test_caption_sin_pregunta_reprueba(self):
        llm = ScriptedLLM(responses=[
            slides(caption="Resumen del modelo pequeño."), slides()])
        report = write_carousel(dossier(), llm)
        assert report.ok
        assert any("pregunta" in v for v in report.attempts[0].violations)
        # El mensaje avisa en la caption y pide la pregunta en el texto.

    def test_visual_fuera_del_pool_reprueba(self):
        llm = ScriptedLLM(responses=[
            slides(visuals=["innovation"] * 5), slides()])
        report = write_carousel(dossier(), llm)
        assert report.ok
        assert any("vocabulario" in v for v in report.attempts[0].violations)

    def test_prompt_pide_hilo_narrativo(self):
        from agent.writer.carousel import build_prompt
        prompt = build_prompt(dossier())
        assert "Hilo" in prompt and "sola" in prompt


class TestJuez:
    def test_aprueba_con_rubrica_buena(self):
        carrusel = Carousel.model_validate_json(slides())
        report = judge_carousel(carrusel, dossier(),
                                ScriptedLLM(responses=[dictamen()]))
        assert report.approved and report.review is not None
        assert report.review.total == 10

    def test_flujo_a_cero_reprueba_y_vuelve_como_nota(self):
        """Diapositiva suelta ('Segundo artículo en 2025') pone el flujo a cero y
        no pasa, incluso con hook/fuente/cta al máximo -- y la nota vuelve al
        guionista."""
        carrusel = Carousel.model_validate_json(slides())
        report = judge_carousel(carrusel, dossier(),
                                ScriptedLLM(responses=[dictamen(flujo=0)]))
        assert not report.approved
        assert report.review is not None
        assert any("flujo" in n for n in report.review.revision_notes)

    def test_politica_reprueba_sin_modelo(self):
        txts = ["Desliza y mira", "Muerte en el laboratorio", "Mantiene 98,2%",
                "143 por segundo", "Guarda este resumen"]
        carrusel = Carousel.model_validate_json(slides(texts=txts))
        llm = ScriptedLLM(responses=[])
        report = judge_carousel(carrusel, dossier(), llm)
        assert not report.approved and llm.calls == []
        assert report.review is not None and report.review.short_circuited


class TestMemoria:
    def test_ida_y_vuelta_con_dictamen(self, tmp_path):
        from agent.judge.carousel import judge_carousel as jc
        store = SignalStore(tmp_path / "agent.db")
        carrusel = Carousel.model_validate_json(slides())
        review = jc(carrusel, dossier(),
                    ScriptedLLM(responses=[dictamen()])).review
        linea = store.record_carousel(
            carrusel, model="m", provider="p", usage=(10, 5),
            latency_s=1.0, attempts=[], review=review)
        assert store.latest_carousel().topic == carrusel.topic
        assert store.list_carousels()[0]["approved"] == 1
        assert linea == store.latest_carousel_id(carrusel.topic)


class TestDiapositivas:
    def test_cinco_png_1080x1920(self, tmp_path):
        from PIL import Image
        carrusel = Carousel.model_validate_json(slides())
        salidas = render_carousel(carrusel, tmp_path / "car", "news")
        assert len(salidas) == 5 and (tmp_path / "car" / "caption.txt").exists()
        for s in salidas:
            with Image.open(s) as img:
                assert img.size == (1080, 1920)

    def test_titulo_legible_medido_en_el_png(self, tmp_path):
        """El carrusel del 19/09 salió con título de ~10px y pasó la aceptación
        de dimensión. La aceptación ahora mide tinta en el PNG."""
        from agent.render.carousel import ALTURA_MINIMA_TITULO_PX, ink_height, legible
        carrusel = Carousel.model_validate_json(slides())
        for s in render_carousel(carrusel, tmp_path / "car", "news"):
            assert legible(s), f"{s.name}: {ink_height(s)}px"
            assert ink_height(s) >= ALTURA_MINIMA_TITULO_PX

    def test_titulo_minusculo_reprueba(self, tmp_path):
        from PIL import Image, ImageDraw

        from agent.render.carousel import legible
        png = tmp_path / "bitmap.png"
        img = Image.new("RGB", (1080, 1920), (10, 10, 12))
        ImageDraw.Draw(img).text((70, 1000), "Título en fuente bitmap", fill=(233, 238, 241))
        img.save(png)
        assert not legible(png)

    def test_sin_fonttools_usa_la_fuente_de_la_marca(self, monkeypatch):
        """'No sé si cubre' no puede convertirse en fuente bitmap."""
        import builtins

        from PIL import ImageFont

        from agent.render import typography
        typography._codepoints.cache_clear()
        original = builtins.__import__

        def sin_fonttools(name, *args, **kwargs):
            if name.startswith("fontTools"):
                raise ImportError(name)
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", sin_fonttools)
        try:
            f = typography.font(typography.DISPLAY, 96, "Configuración ñá")
        finally:
            typography._codepoints.cache_clear()
        assert isinstance(f, ImageFont.FreeTypeFont) and f.size == 96

    def test_ultima_diapositiva_credita_la_fuente_en_la_caption(self, tmp_path):
        carrusel = Carousel.model_validate_json(slides())
        carrusel = carrusel.model_copy(update={"facts": dossier().facts})
        render_carousel(carrusel, tmp_path / "car", "news")
        caption = (tmp_path / "car" / "caption.txt").read_text(encoding="utf-8")
        assert "Fuentes:" in caption and "#circuitocero" in caption


class TestLayoutDeLaDiapositiva:
    """La distribución de contenido que vino de la referencia de 20/09/2026."""

    def test_el_acento_siempre_toca_el_titulo(self):
        from agent.render.carousel import _destaque

        # Dos líneas o más: la última línea entera.
        assert _destaque(["Peso mínimo,", "rendimiento máximo"]) == (1, 0)
        # Última línea demasiado corta para sostenerse: enciende la de arriba.
        assert _destaque(["Un título que", "cae"]) == (0, 0)
        # Una línea solo: la última palabra, si se sostiene.
        assert _destaque(["Abre tu Gmail"]) == (0, 1)
        assert _destaque(["Todo bien"]) == (-1, 0)

    def test_chip_de_la_marca_y_contador_arriba(self, tmp_path):
        """Quien lee debe saber de quién es la pieza y cuánto falta antes del título."""
        import numpy as np
        from PIL import Image

        from agent.render.carousel import TOPO_CHIP, render_slide
        carrusel = Carousel.model_validate_json(slides())
        png = render_slide(carrusel, 2, tmp_path / "s2.png", "news")
        with Image.open(png) as img:
            topo = np.array(img.convert("RGB"))[TOPO_CHIP:TOPO_CHIP + 42]
        # Barra de acento a la izquierda (verde fuerte) y contador a la derecha.
        verde = (topo[:, :, 1].astype(int) - topo[:, :, 0]) > 80
        assert verde[:, 80:95].any(), "sin barra de acento en el chip"
        assert verde[:, 800:].any(), "sin contador en la esquina derecha"

    def test_ultima_diapositiva_pide_guardar_en_lugar_de_deslizar(self, tmp_path):
        from agent.render.carousel import render_carousel as rc
        carrusel = Carousel.model_validate_json(slides())
        salidas = rc(carrusel, tmp_path / "car", "news")
        assert len(salidas) == 5
        # El texto se dibuja, así que lo que se puede medir es que las dos
        # últimas diapositivas difieren en el pie -- misma altura, contenido
        # distinto.
        import numpy as np
        from PIL import Image

        from agent.render.carousel import RODAPIE_TEXTO
        with Image.open(salidas[3]) as a, Image.open(salidas[4]) as b:
            fa = np.array(a.convert("L"))[RODAPIE_TEXTO:RODAPIE_TEXTO + 40, 600:]
            fb = np.array(b.convert("L"))[RODAPIE_TEXTO:RODAPIE_TEXTO + 40, 600:]
        assert not np.array_equal(fa, fb)

    def test_sin_foto_el_fondo_sigue_casi_negro(self, tmp_path):
        """La primera versión del baño de acento teñía el cuadro entero de verde.

        La regla de la marca es fondo casi-negro; el baño ata la FOTO al pilar,
        y sin foto no tiene nada que atar.
        """
        import numpy as np
        from PIL import Image

        from agent.render.carousel import BARRA_IZQ, render_slide
        carrusel = Carousel.model_validate_json(slides())
        png = render_slide(carrusel, 2, tmp_path / "s2.png", "news")
        with Image.open(png) as img:
            # Franja vacía entre el chip y el título, fuera de la cinta del borde.
            franja = np.array(img.convert("RGB"))[300:900, BARRA_IZQ + 40:]
        assert franja.max() <= 20, f"fondo teñido: max {franja.max()}"
