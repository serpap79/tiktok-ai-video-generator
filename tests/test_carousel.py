"""Carrossel: 5 slides mecanicos, juiz proprio, memoria e PNG local."""

from __future__ import annotations

import json

from agent.adapters.scripted_llm import ScriptedLLM
from agent.judge.carousel import judge_carousel
from agent.memory.store import SignalStore
from agent.models import Carousel
from agent.render.carousel import render_carousel
from agent.writer.carousel import write_carousel
from tests.test_writer import dossie

VISUAIS = ["neural network nodes", "abstract digital plexus",
           "ai deep learning loop", "quantum computing laser",
           "data stream tunnel"]


def slides(headlines=None, texts=None, visuals=None, caption=None) -> str:
    heads = headlines or ["5 dados do Bonsai em 5,9 GB", "Pesa pouco",
                          "Rende muito", "Roda rapido", "Salve para depois"]
    txts = texts or ["Arraste e veja",
                     "Ocupa 5,9 GB no disco",
                     "Mantem 98,2% do desempenho",
                     "Chega a 143 tokens por segundo",
                     "Salve este resumo para rever"]
    return json.dumps({
        "topic": "Bonsai 2 27B: modelo de 27B em 5,9 GB",
        "slides": [{"n": i + 1, "headline": h, "text": t,
                    "visual": (visuals or VISUAIS)[i]}
                   for i, (h, t) in enumerate(zip(heads, txts, strict=True))],
        "caption": caption or "Modelo pequeno e forte: qual numero te surpreendeu?",
        "used_facts": [0, 1, 2],
    })


def parecer(**notas: int) -> str:
    return json.dumps({
        c: {"reason": f"motivo {c}", "score": notas.get(c, 2)}
        for c in ("hook", "fonte", "fluxo", "cta")})


class TestRoteirista:
    def test_valido_na_primeira(self):
        report = write_carousel(dossie(), ScriptedLLM(responses=[slides()]))
        assert report.ok and report.carousel is not None
        assert [s.n for s in report.carousel.slides] == [1, 2, 3, 4, 5]

    def test_slide_longo_volta_com_o_teto(self):
        longo = ["dado"] * 4 + [" ".join(["palavra"] * 20)]
        llm = ScriptedLLM(responses=[slides(texts=longo), slides()])
        report = write_carousel(dossie(), llm)
        assert report.ok
        assert any("teto 12" in v for v in report.attempts[0].violations)

    def test_slide1_sem_numero_reprova(self):
        heads = ["Resumo do modelo", "Pesa pouco", "Rende muito",
                 "Roda rapido", "Salve para depois"]
        llm = ScriptedLLM(responses=[slides(headlines=heads), slides()])
        report = write_carousel(dossie(), llm)
        assert report.ok
        assert any("numero" in v for v in report.attempts[0].violations)

    def test_slide5_sem_save_reprova(self):
        heads = ["5 dados do modelo", "Pesa pouco", "Rende muito",
                 "Roda rapido", "Fim"]
        txts = ["Arraste", "Ocupa 5,9 GB", "Mantem 98,2%", "143 por segundo",
                "Obrigado por ler"]
        llm = ScriptedLLM(responses=[
            slides(headlines=heads, texts=txts), slides()])
        report = write_carousel(dossie(), llm)
        assert report.ok
        assert any("save" in v for v in report.attempts[0].violations)

    def test_legenda_sem_pergunta_reprova(self):
        llm = ScriptedLLM(responses=[
            slides(caption="Resumo do modelo pequeno."), slides()])
        report = write_carousel(dossie(), llm)
        assert report.ok
        assert any("pergunta" in v for v in report.attempts[0].violations)

    def test_visual_fora_do_pool_reprova(self):
        llm = ScriptedLLM(responses=[
            slides(visuals=["innovation"] * 5), slides()])
        report = write_carousel(dossie(), llm)
        assert report.ok
        assert any("vocabulario" in v for v in report.attempts[0].violations)

    def test_prompt_pede_fio_narrativo(self):
        from agent.writer.carousel import build_prompt
        prompt = build_prompt(dossie())
        assert "Fio" in prompt and "sozinho" in prompt


class TestJuiz:
    def test_aprova_com_rubrica_boa(self):
        carrossel = Carousel.model_validate_json(slides())
        report = judge_carousel(carrossel, dossie(),
                                ScriptedLLM(responses=[parecer()]))
        assert report.approved and report.review is not None
        assert report.review.total == 10

    def test_fluxo_zerado_reprova_e_volta_como_nota(self):
        """Slide solto ('Segundo artigo em 2025') zera fluxo e nao passa,
        mesmo com hook/fonte/cta no maximo -- e a nota volta ao roteirista."""
        carrossel = Carousel.model_validate_json(slides())
        report = judge_carousel(carrossel, dossie(),
                                ScriptedLLM(responses=[parecer(fluxo=0)]))
        assert not report.approved
        assert report.review is not None
        assert any("fluxo" in n for n in report.review.revision_notes)

    def test_politica_reprova_sem_modelo(self):
        txts = ["Arraste e veja", "Morte no laboratorio", "Mantem 98,2%",
                "143 por segundo", "Salve este resumo"]
        carrossel = Carousel.model_validate_json(slides(texts=txts))
        llm = ScriptedLLM(responses=[])
        report = judge_carousel(carrossel, dossie(), llm)
        assert not report.approved and llm.calls == []
        assert report.review is not None and report.review.short_circuited


class TestMemoria:
    def test_ida_e_volta_com_parecer(self, tmp_path):
        from agent.judge.carousel import judge_carousel as jc
        store = SignalStore(tmp_path / "agent.db")
        carrossel = Carousel.model_validate_json(slides())
        review = jc(carrossel, dossie(),
                    ScriptedLLM(responses=[parecer()])).review
        linha = store.record_carousel(
            carrossel, model="m", provider="p", usage=(10, 5),
            latency_s=1.0, attempts=[], review=review)
        assert store.latest_carousel().topic == carrossel.topic
        assert store.list_carousels()[0]["approved"] == 1
        assert linha == store.latest_carousel_id(carrossel.topic)


class TestSlides:
    def test_cinco_png_1080x1920(self, tmp_path):
        from PIL import Image
        carrossel = Carousel.model_validate_json(slides())
        saidas = render_carousel(carrossel, tmp_path / "car", "news")
        assert len(saidas) == 5 and (tmp_path / "car" / "caption.txt").exists()
        for s in saidas:
            with Image.open(s) as img:
                assert img.size == (1080, 1920)

    def test_titulo_legivel_medido_no_png(self, tmp_path):
        """O carrossel de 19/09 saiu com titulo de ~10px e passou no aceite
        de dimensao. O aceite agora mede tinta no PNG."""
        from agent.render.carousel import ALTURA_MINIMA_TITULO_PX, ink_height, legible
        carrossel = Carousel.model_validate_json(slides())
        for s in render_carousel(carrossel, tmp_path / "car", "news"):
            assert legible(s), f"{s.name}: {ink_height(s)}px"
            assert ink_height(s) >= ALTURA_MINIMA_TITULO_PX

    def test_titulo_minusculo_reprova(self, tmp_path):
        from PIL import Image, ImageDraw

        from agent.render.carousel import legible
        png = tmp_path / "bitmap.png"
        img = Image.new("RGB", (1080, 1920), (10, 10, 12))
        ImageDraw.Draw(img).text((70, 1000), "Titulo em fonte bitmap", fill=(233, 238, 241))
        img.save(png)
        assert not legible(png)

    def test_sem_fonttools_usa_a_fonte_da_marca(self, monkeypatch):
        """'Nao sei se cobre' nao pode virar fonte bitmap."""
        import builtins

        from PIL import ImageFont

        from agent.render import typography
        typography._codepoints.cache_clear()
        original = builtins.__import__

        def sem_fonttools(name, *args, **kwargs):
            if name.startswith("fontTools"):
                raise ImportError(name)
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", sem_fonttools)
        try:
            f = typography.font(typography.DISPLAY, 96, "Configuração ção")
        finally:
            typography._codepoints.cache_clear()
        assert isinstance(f, ImageFont.FreeTypeFont) and f.size == 96

    def test_ultimo_slide_credita_a_fonte_na_legenda(self, tmp_path):
        carrossel = Carousel.model_validate_json(slides())
        carrossel = carrossel.model_copy(update={"facts": dossie().facts})
        render_carousel(carrossel, tmp_path / "car", "news")
        legenda = (tmp_path / "car" / "caption.txt").read_text(encoding="utf-8")
        assert "Fontes:" in legenda and "#seucanal" in legenda


class TestLayoutDoSlide:
    """A distribuicao de conteudo que veio da referencia de 20/09/2026."""

    def test_o_acento_sempre_toca_o_titulo(self):
        from agent.render.carousel import _destaque

        # Duas linhas ou mais: a ultima linha inteira.
        assert _destaque(["Peso minimo,", "performance maxima"]) == (1, 0)
        # Ultima linha curta demais para se sustentar: acende a de cima.
        assert _destaque(["Um titulo que", "cai"]) == (0, 0)
        # Uma linha so: a ultima palavra, se ela se sustentar.
        assert _destaque(["Abra seu Gmail"]) == (0, 1)
        assert _destaque(["Tudo bem"]) == (-1, 0)

    def test_chip_da_marca_e_contador_no_topo(self, tmp_path):
        """Quem le tem de saber de quem e a peca e quanto falta antes do titulo."""
        import numpy as np
        from PIL import Image

        from agent.render.carousel import TOPO_CHIP, render_slide
        carrossel = Carousel.model_validate_json(slides())
        png = render_slide(carrossel, 2, tmp_path / "s2.png", "news")
        with Image.open(png) as img:
            topo = np.array(img.convert("RGB"))[TOPO_CHIP:TOPO_CHIP + 42]
        # Barra de acento a esquerda (verde forte) e contador a direita.
        verde = (topo[:, :, 1].astype(int) - topo[:, :, 0]) > 80
        assert verde[:, 80:95].any(), "sem barra de acento no chip"
        assert verde[:, 800:].any(), "sem contador no canto direito"

    def test_ultimo_slide_pede_salvar_em_vez_de_deslizar(self, tmp_path):
        from agent.render.carousel import render_carousel as rc
        carrossel = Carousel.model_validate_json(slides())
        saidas = rc(carrossel, tmp_path / "car", "news")
        assert len(saidas) == 5
        # O texto e desenhado, entao o que da para medir e que os dois ultimos
        # slides diferem no rodape -- mesma altura, conteudo diferente.
        import numpy as np
        from PIL import Image

        from agent.render.carousel import RODAPE_TEXTO
        with Image.open(saidas[3]) as a, Image.open(saidas[4]) as b:
            fa = np.array(a.convert("L"))[RODAPE_TEXTO:RODAPE_TEXTO + 40, 600:]
            fb = np.array(b.convert("L"))[RODAPE_TEXTO:RODAPE_TEXTO + 40, 600:]
        assert not np.array_equal(fa, fb)

    def test_sem_foto_o_fundo_continua_quase_preto(self, tmp_path):
        """A primeira versao do banho de acento tingia o quadro inteiro de verde.

        A regra da marca e fundo quase-preto; o banho amarra a FOTO ao pilar, e
        sem foto ele nao tem o que amarrar.
        """
        import numpy as np
        from PIL import Image

        from agent.render.carousel import BARRA_ESQ, render_slide
        carrossel = Carousel.model_validate_json(slides())
        png = render_slide(carrossel, 2, tmp_path / "s2.png", "news")
        with Image.open(png) as img:
            # Faixa vazia entre o chip e o titulo, fora da fita da borda.
            faixa = np.array(img.convert("RGB"))[300:900, BARRA_ESQ + 40:]
        assert faixa.max() <= 20, f"fundo tingido: max {faixa.max()}"
