"""Producao: clipes escolhidos, trilha gerada, pos-producao -- sem rede e sem ffmpeg."""

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
    TOPO_CARTAO,
    TOPO_CARTAO_COM_APRESENTADOR,
    PostSpec,
    build_command,
    hook_card,
)
from agent.render.presenter import Camada


def roteiro(broll: list[str] | None = None, palavras: int = 180) -> Script:
    return Script(topic="Tema de teste", hook="Um gancho que abre lacuna.",
                  body=" ".join(["palavra"] * palavras), closing="Fechamento com pergunta?",
                  search_terms=["matrix code rain", "cyberpunk hacking terminal",
                                "server room blinking lights"],
                  broll=broll or [])


def clip(**kw) -> Clip:
    base = dict(term="t", video_id=1, page_url="https://www.pexels.com/video/x-1/",
                file_url="https://x", width=1080, height=1920, duration=10.0, thumb_url="")
    base.update(kw)
    return Clip(**base)


class TestClipes:
    def test_relevancia_pelo_slug_da_pagina(self):
        url = "https://www.pexels.com/video/a-graphics-card-on-a-motherboard-123/"
        assert relevance("graphics card", url) == 1.0
        assert relevance("human brain model", url) == 0.0

    def test_escuridao_da_miniatura(self):
        def jpeg(cor):
            buf = io.BytesIO()
            Image.new("RGB", (40, 70), cor).save(buf, "JPEG")
            return buf.getvalue()
        assert darkness_of(jpeg((10, 10, 12))) > 0.9 > darkness_of(jpeg((240, 235, 220)))

    def test_broll_pesa_relevancia_e_pilar_pesa_escuridao(self):
        relevante_claro = clip(relevance=1.0, darkness=0.1)
        irrelevante_escuro = clip(relevance=0.0, darkness=0.95)
        assert (score(relevante_claro, is_broll=True, reused=False)
                > score(irrelevante_escuro, is_broll=True, reused=False))
        assert (score(irrelevante_escuro, is_broll=False, reused=False)
                > score(relevante_claro, is_broll=False, reused=False))

    def test_clipe_repetido_perde_pontos(self):
        c = clip(relevance=0.5, darkness=0.5)
        assert score(c, is_broll=False, reused=True) < score(c, is_broll=False, reused=False)

    def test_arquivo_retrato_mais_proximo_de_1080(self):
        video = {"video_files": [
            {"width": 2160, "height": 3840, "link": "4k"},
            {"width": 1080, "height": 1920, "link": "fhd"},
            {"width": 1920, "height": 1080, "link": "paisagem"},
        ]}
        assert best_file(video) == ("fhd", 1080, 1920)
        assert best_file({"video_files": [{"width": 540, "height": 960, "link": "sd"}]}) is None

    def test_broll_abre_o_video_e_o_segundo_entra_no_meio(self):
        linha = timeline_terms(roteiro(["graphics card", "data center"]))
        assert linha[0] == ("graphics card", True)
        assert ("data center", True) in linha[1:-1]

    def test_clipes_suficientes_para_nao_repetir(self):
        # 180 palavras ~ 72s, com folga e corte de 5s: ~19 clipes.
        assert 17 <= needed_clips(roteiro()) <= 21


class TestTrilha:
    def test_duracao_formato_e_fade(self):
        audio = synthesize(6.0, MOODS["pulso"], seed="tema")
        assert audio.shape == (int(6.0 * SR), 2) and audio.dtype == np.float32
        assert float(np.abs(audio).max()) <= 0.9
        assert float(np.abs(audio[:100]).max()) < 0.05  # fade de entrada

    def test_mesmo_tema_mesma_trilha(self):
        a = synthesize(2.0, MOODS["arquivo"], seed="ENIAC")
        b = synthesize(2.0, MOODS["arquivo"], seed="ENIAC")
        assert np.array_equal(a, b)

    def test_todo_pilar_tem_clima(self):
        for pilar in ("news", "fato", "analise", "tutorial", "futuro", "vs", "historia"):
            assert mood_for(pilar).id == MOOD_BY_PILLAR[pilar]


class TestPosProducao:
    def test_grafo_tem_gancho_marca_ducking_e_loudness(self, tmp_path):
        cmd = build_command(tmp_path / "v.mp4", tmp_path / "o.mp4", hook_png=tmp_path / "h.png",
                            mark_png=tmp_path / "m.png", music_wav=tmp_path / "t.wav",
                            duration_s=65.0)
        grafo = cmd[cmd.index("-filter_complex") + 1]
        assert "sidechaincompress" in grafo and "loudnorm=I=-14" in grafo
        assert "fade=t=out" in grafo and "vignette" in grafo
        assert "libx264" in cmd

    def test_sem_trilha_ainda_normaliza(self, tmp_path):
        cmd = build_command(tmp_path / "v.mp4", tmp_path / "o.mp4", hook_png=tmp_path / "h.png",
                            mark_png=tmp_path / "m.png", music_wav=None, duration_s=15.0)
        grafo = cmd[cmd.index("-filter_complex") + 1]
        assert "sidechaincompress" not in grafo and "loudnorm" in grafo

    def test_cartao_do_gancho_tem_texto_legivel(self, tmp_path):
        png = hook_card(PostSpec(hook="Dizem que a IA é segura, mas o Gemini provou o contrário.",
                                 tag="ANÁLISE", accent="#39FF88", handle="@seucanal"),
                        tmp_path / "hook.png")
        with Image.open(png) as img:
            alpha = np.array(img)[:, :, 3]
        linhas = np.where(alpha[300:900].max(axis=1) > 0)[0]
        assert img.size == (1080, 1920) and len(linhas) > 150

    def test_apresentador_entra_como_camada_inteira_sem_crop(self, tmp_path):
        """O defeito que este teste tranca: `crop` recortava a silhueta.

        A camada ja vem posicionada e com alfa; a pos-producao so escolhe onde
        ela pousa. Qualquer `crop` de volta no grafo e a figurinha retangular
        de novo.
        """
        camada = Camada(path=tmp_path / "apr.mov", x=0, y=660, width=1080,
                        height=1160, subtitle_y=940, subtitle_start=4.1,
                        card_s=4.1, frames=1950, fps=30, seconds=65.0)
        cmd = build_command(tmp_path / "v.mp4", tmp_path / "o.mp4",
                            hook_png=tmp_path / "h.png", mark_png=tmp_path / "m.png",
                            music_wav=None, duration_s=65.0, presenter=camada,
                            card_s=camada.card_s)
        grafo = cmd[cmd.index("-filter_complex") + 1]
        assert "crop=" not in grafo
        assert "overlay=0:660" in grafo
        assert str(camada.path) in cmd
        # O cartao do gancho agora dura ate a legenda entrar, nao 3,2s fixos.
        assert "-t 4.10" in " ".join(cmd)

    def test_cartao_sobe_com_apresentador_e_nao_assina_nome(self, tmp_path):
        """A plaquinha virou parte do apresentador: anda com ele, sai com ele."""
        camada = Camada(path=tmp_path / "apr.mov", x=0, y=660, width=1080,
                        height=1160, subtitle_y=940, subtitle_start=4.1,
                        card_s=4.1, frames=100, fps=30, seconds=3.3)
        hook = "Dizem que a IA é segura, mas o Gemini provou o contrário."
        com = hook_card(PostSpec(hook=hook, tag="ANÁLISE", accent="#39FF88",
                                 handle="@seucanal", presenter=camada),
                        tmp_path / "hook-apr.png")
        sem = hook_card(PostSpec(hook=hook, tag="ANÁLISE", accent="#39FF88",
                                 handle="@seucanal"), tmp_path / "hook.png")
        with Image.open(com) as img:
            a_com = np.array(img)[:, :, 3]
        with Image.open(sem) as img:
            a_sem = np.array(img)[:, :, 3]
        topo_com = int(np.flatnonzero(a_com.max(axis=1) > 0)[0])
        topo_sem = int(np.flatnonzero(a_sem.max(axis=1) > 0)[0])
        assert topo_com == TOPO_CARTAO_COM_APRESENTADOR < topo_sem == TOPO_CARTAO
        # O painel usa a largura toda nos dois casos (o apresentador fica abaixo).
        assert a_com.max(axis=0).nonzero()[0].max() == a_sem.max(axis=0).nonzero()[0].max()
        # Nada escrito abaixo de 900: a cabeca do apresentador comeca em ~815.
        assert a_com[900:].max() == 0


class TestBrollSemRelevancia:
    def test_broll_sem_palavra_do_termo_e_descartado(self):
        """'Thyles Rupes' devolveu flores e lagarta: melhor nenhum clipe."""
        import httpx

        from agent.render.footage import PexelsFootage

        videos = {"videos": [{"id": 1, "duration": 10, "url": "https://www.pexels.com/video/white-flowers-1/",
                              "image": "", "video_files": [{"width": 1080, "height": 1920,
                                                            "link": "https://x/1.mp4"}]}]}
        cliente = httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=videos)))
        import tempfile
        f = PexelsFootage("chave", tempfile.mkdtemp(), client=cliente)
        assert f._rank("Thyles Rupes", True, set()) == []
        assert len(f._rank("dark tech laboratory", False, set())) == 1
