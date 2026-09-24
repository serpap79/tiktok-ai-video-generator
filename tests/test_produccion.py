"""Produccion: clips elegidos, pista generada, postproduccion -- sin red y sin ffmpeg."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from agent.models import Script
from agent.render.footage import (
    Clip,
    best_file,
    darkness_of,
    needed_clips,
    relevance,
    score,
    timeline_terms,
)
from agent.render.music import MOOD_BY_PILLAR, MOODS, SR, mood_for, synthesize
from agent.render.post import (
    TOPO_TARJETA,
    TOPO_TARJETA_CON_PRESENTADOR,
    PostSpec,
    build_command,
    tarjeta_gancho,
)
from agent.render.presenter import Capa


def guion(broll: list[str] | None = None, palabras: int = 180) -> Script:
    return Script(topic="Tema de test", hook="Un gancho que abre hueco.",
                  body=" ".join(["palabra"] * palabras), closing="Cierre con pregunta?",
                  search_terms=["matrix code rain", "cyberpunk hacking terminal",
                                "server room blinking lights"],
                  broll=broll or [])


def clip(**kw) -> Clip:
    base = dict(term="t", video_id=1, page_url="https://www.pexels.com/video/x-1/",
                file_url="https://x", width=1080, height=1920, duration=10.0, thumb_url="")
    base.update(kw)
    return Clip(**base)


class TestClips:
    def test_relevancia_por_el_slug_de_la_pagina(self):
        url = "https://www.pexels.com/video/a-graphics-card-on-a-motherboard-123/"
        assert relevance("graphics card", url) == 1.0
        assert relevance("human brain model", url) == 0.0

    def test_oscuridad_de_la_miniatura(self):
        def jpeg(color):
            buf = io.BytesIO()
            Image.new("RGB", (40, 70), color).save(buf, "JPEG")
            return buf.getvalue()
        assert darkness_of(jpeg((10, 10, 12))) > 0.9 > darkness_of(jpeg((240, 235, 220)))

    def test_broll_pesa_relevancia_y_el_pilar_pesa_oscuridad(self):
        relevante_claro = clip(relevance=1.0, darkness=0.1)
        irrelevante_oscuro = clip(relevance=0.0, darkness=0.95)
        assert (score(relevante_claro, is_broll=True, reused=False)
                > score(irrelevante_oscuro, is_broll=True, reused=False))
        assert (score(irrelevante_oscuro, is_broll=False, reused=False)
                > score(relevante_claro, is_broll=False, reused=False))

    def test_clip_repetido_pierde_puntos(self):
        c = clip(relevance=0.5, darkness=0.5)
        assert score(c, is_broll=False, reused=True) < score(c, is_broll=False, reused=False)

    def test_archivo_retrato_mas_cercano_a_1080(self):
        video = {"video_files": [
            {"width": 2160, "height": 3840, "link": "4k"},
            {"width": 1080, "height": 1920, "link": "fhd"},
            {"width": 1920, "height": 1080, "link": "paisaje"},
        ]}
        assert best_file(video) == ("fhd", 1080, 1920)
        assert best_file({"video_files": [{"width": 540, "height": 960, "link": "sd"}]}) is None

    def test_el_broll_abre_el_video_y_el_segundo_entra_en_medio(self):
        linea = timeline_terms(guion(["graphics card", "data center"]))
        assert linea[0] == ("graphics card", True)
        assert ("data center", True) in linea[1:-1]

    def test_clips_suficientes_para_no_repetir(self):
        # 180 palabras ~ 72s, con margen y corte de 5s: ~19 clips.
        assert 17 <= needed_clips(guion()) <= 21


class TestPista:
    def test_duracion_formato_y_fade(self):
        audio = synthesize(6.0, MOODS["pulso"], seed="tema")
        assert audio.shape == (int(6.0 * SR), 2) and audio.dtype == np.float32
        assert float(np.abs(audio).max()) <= 0.9
        assert float(np.abs(audio[:100]).max()) < 0.05  # fade de entrada

    def test_mismo_tema_misma_pista(self):
        a = synthesize(2.0, MOODS["archivo"], seed="ENIAC")
        b = synthesize(2.0, MOODS["archivo"], seed="ENIAC")
        assert np.array_equal(a, b)

    def test_todo_pilar_tiene_clima(self):
        for pilar in ("news", "dato", "analisis", "tutorial", "futuro", "vs", "historia"):
            assert mood_for(pilar).id == MOOD_BY_PILLAR[pilar]


class TestPostproduccion:
    def test_el_grafo_tiene_gancho_marca_ducking_y_loudness(self, tmp_path):
        cmd = build_command(tmp_path / "v.mp4", tmp_path / "o.mp4", hook_png=tmp_path / "h.png",
                            mark_png=tmp_path / "m.png", music_wav=tmp_path / "t.wav",
                            duration_s=65.0)
        grafo = cmd[cmd.index("-filter_complex") + 1]
        assert "sidechaincompress" in grafo and "loudnorm=I=-14" in grafo
        assert "fade=t=out" in grafo and "vignette" in grafo
        assert "libx264" in cmd

    def test_sin_pista_igualmente_normaliza(self, tmp_path):
        cmd = build_command(tmp_path / "v.mp4", tmp_path / "o.mp4", hook_png=tmp_path / "h.png",
                            mark_png=tmp_path / "m.png", music_wav=None, duration_s=15.0)
        grafo = cmd[cmd.index("-filter_complex") + 1]
        assert "sidechaincompress" not in grafo and "loudnorm" in grafo

    def test_la_tarjeta_del_gancho_tiene_texto_legible(self, tmp_path):
        png = tarjeta_gancho(PostSpec(
            hook="Dicen que la IA es segura, pero Gemini ha probado lo contrario.",
            tag="ANALISIS",
            accent="#39FF88",
            handle="@circuitocero",
        ),
        tmp_path / "hook.png",
    )
        with Image.open(png) as img:
            alpha = np.array(img)[:, :, 3]
        lineas = np.where(alpha[300:900].max(axis=1) > 0)[0]
        assert img.size == (1080, 1920) and len(lineas) > 150

    def test_el_presentador_entra_como_capa_entera_sin_crop(self, tmp_path):
        """El defecto que este test bloquea: `crop` recortaba la silueta.

        La capa ya viene posicionada y con alfa; la postproduccion solo elige
        donde posa. Cualquier `crop` de vuelta en el grafo es la pegatina
        rectangular otra vez.
        """
        capa = Capa(path=tmp_path / "apr.mov", x=0, y=660, width=1080,
                    height=1160, subtitle_y=940, subtitle_start=4.1,
                    card_s=4.1, frames=1950, fps=30, seconds=65.0)
        cmd = build_command(tmp_path / "v.mp4", tmp_path / "o.mp4",
                            hook_png=tmp_path / "h.png", mark_png=tmp_path / "m.png",
                            music_wav=None, duration_s=65.0, presenter=capa,
                            card_s=capa.card_s)
        grafo = cmd[cmd.index("-filter_complex") + 1]
        assert "crop=" not in grafo
        assert "overlay=0:660" in grafo
        assert str(capa.path) in cmd
        # La tarjeta del gancho ahora dura hasta que entra la leyenda, no 3,2s fijos.
        assert "-t 4.10" in " ".join(cmd)

    def test_la_tarjeta_sube_con_el_presentador_y_no_firma_nombre(self, tmp_path):
        """El cartel paso a ser parte del presentador: camina con el, sale con el."""
        capa = Capa(path=tmp_path / "apr.mov", x=0, y=660, width=1080,
                    height=1160, subtitle_y=940, subtitle_start=4.1,
                    card_s=4.1, frames=100, fps=30, seconds=3.3)
        hook = "Dicen que la IA es segura, pero el Gemini probó lo contrario."
        con = tarjeta_gancho(PostSpec(hook=hook, tag="ANALISIS", accent="#39FF88",
                                      handle="@circuitocero", presenter=capa),
                             tmp_path / "hook-apr.png")
        sin = tarjeta_gancho(PostSpec(hook=hook, tag="ANALISIS", accent="#39FF88",
                                      handle="@circuitocero"), tmp_path / "hook.png")
        with Image.open(con) as img:
            a_con = np.array(img)[:, :, 3]
        with Image.open(sin) as img:
            a_sin = np.array(img)[:, :, 3]
        tope_con = int(np.flatnonzero(a_con.max(axis=1) > 0)[0])
        tope_sin = int(np.flatnonzero(a_sin.max(axis=1) > 0)[0])
        assert tope_con == TOPO_TARJETA_CON_PRESENTADOR < tope_sin == TOPO_TARJETA
        # El panel usa todo el ancho en los dos casos (el presentador queda debajo).
        assert a_con.max(axis=0).nonzero()[0].max() == a_sin.max(axis=0).nonzero()[0].max()
        # Nada escrito debajo de 900: la cabeza del presentador empieza en ~815.
        assert a_con[900:].max() == 0


class TestBrollSinRelevancia:
    def test_broll_sin_palabra_del_termino_se_descarta(self):
        "«Thyles Rupes» devolvió flores y oruga: mejor ningún clip."""
        import tempfile

        import httpx

        from agent.render.footage import PexelsFootage

        videos = {"videos": [{"id": 1, "duration": 10, "url": "https://www.pexels.com/video/white-flowers-1/",
                              "image": "", "video_files": [{"width": 1080, "height": 1920,
                                                            "link": "https://x/1.mp4"}]}]}
        cliente = httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=videos)))
        f = PexelsFootage("clave", tempfile.mkdtemp(), client=cliente)
        assert f._rank("Thyles Rupes", True, set()) == []
        assert len(f._rank("dark tech laboratory", False, set())) == 1
