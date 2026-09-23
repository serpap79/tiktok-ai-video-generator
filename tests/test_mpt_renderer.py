"""Testes do adaptador do renderizador, contra um MPT simulado.

O contrato aqui foi lido do codigo do MoneyPrinterTurbo, nao suposto:
  - prefixo /api/v1 (app/controllers/v1/base.py)
  - envelope {"status": ..., "data": {...}} (app/utils/utils.py:get_response)
  - estados -1 falha / 1 completo / 4 processando (app/models/const.py)
  - artefato como URI relativa /tasks/<...> quando `endpoint` esta vazio
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent.adapters.mpt_renderer import MptRenderer
from agent.config import Settings
from agent.models import RenderState, Script
from agent.ports.renderer import Renderer, RendererError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "roteiro_manual.json"


def _noop(request: httpx.Request) -> httpx.Response:
    """Handler que nunca deveria ser chamado: usado so para montar payload."""
    return httpx.Response(200)


@pytest.fixture
def script() -> Script:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return Script.model_validate(raw)


def make_renderer(handler, tmp_path: Path) -> MptRenderer:
    settings = Settings(
        renderer_url="http://mpt.test",
        renderer_api_key="chave-de-teste",
        renderer_poll_interval_s=0.0,
        renderer_timeout_s=5.0,
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
    )
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://mpt.test",
        headers={"x-api-key": "chave-de-teste"},
    )
    return MptRenderer(settings=settings, client=client)


def test_adaptador_satisfaz_a_porta(tmp_path):
    def _ok(request):
        return httpx.Response(200)

    assert isinstance(make_renderer(_ok, tmp_path), Renderer)


class TestPayload:
    def test_roteiro_pronto_contorna_o_llm_do_mpt(self, script, tmp_path):
        """task.py:generate_script so chama o LLM quando video_script vem vazio.

        Se este campo parar de ser enviado, o MPT passa a gerar roteiro proprio
        em silencio - o agente perde o controle sem nenhum erro aparecer.
        """
        payload = make_renderer(_noop, tmp_path).build_payload(script)
        assert payload["video_script"].strip()
        assert payload["video_script"] == script.narration
        assert payload["video_terms"] == script.search_terms

    def test_pede_vertical_e_legenda_karaoke(self, script, tmp_path):
        payload = make_renderer(_noop, tmp_path).build_payload(script)
        assert payload["video_aspect"] == "9:16"
        assert payload["subtitle_display_mode"] == "word_by_word"
        assert payload["subtitle_enabled"] is True

    def test_voz_e_fonte_compativeis_com_ptbr(self, script, tmp_path):
        """Voz sem sufixo -V2 roteia para o edge-tts gratuito, nao para o Azure pago.

        E BeVietnamPro-Bold e a unica fonte do MPT que tem os acentos do pt-BR;
        as outras renderizam tofu no lugar de "c-cedilha" e "a-til".
        """
        payload = make_renderer(_noop, tmp_path).build_payload(script)
        assert payload["voice_name"].startswith("pt-BR-")
        assert not payload["voice_name"].endswith("-V2")
        assert payload["font_name"] == "BeVietnamPro-Bold.ttf"


class TestFluxoFeliz:
    def test_renderiza_e_mede_o_resultado(self, script, tmp_path, monkeypatch):
        chamadas = {"post": 0, "get": 0, "download": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                chamadas["post"] += 1
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-1"}})
            if request.url.path == "/api/v1/tasks/t-1":
                chamadas["get"] += 1
                # primeiro processando, depois completo: garante que o poll roda
                if chamadas["get"] < 2:
                    return httpx.Response(
                        200, json={"status": 200, "data": {"state": 4, "progress": 40}}
                    )
                return httpx.Response(
                    200,
                    json={
                        "status": 200,
                        "data": {
                            "state": 1,
                            "progress": 100,
                            "videos": ["/tasks/t-1/final-1.mp4"],
                        },
                    },
                )
            if request.url.path == "/api/v1/download/t-1/final-1.mp4":
                chamadas["download"] += 1
                return httpx.Response(200, content=b"\x00" * 1024)
            return httpx.Response(404)

        renderer = make_renderer(handler, tmp_path)
        monkeypatch.setattr(
            "agent.adapters.mpt_renderer.probe_video",
            lambda p: {
                "width": 1080, "height": 1920, "duration_s": 78.4, "has_audio": True,
            },
        )

        result = renderer.render(script)

        assert result.state is RenderState.complete
        assert result.is_portrait_1080x1920
        assert result.duration_in_monetizable_range
        assert chamadas == {"post": 1, "get": 2, "download": 1}
        assert Path(result.video_path).read_bytes() == b"\x00" * 1024

    def test_baixa_o_final_e_nao_o_combinado(self, script, tmp_path, monkeypatch):
        """Regressao: `combined_videos` NAO e o corte final.

        No MPT, combined-N.mp4 e o concat so de video e final-N.mp4 e o corte com
        narracao e legenda. Preferir "combined" pelo nome entrega um MP4 mudo que
        passa em qualquer checagem de dimensao e duracao. Foi exatamente o que
        aconteceu na primeira execucao real contra o renderizador.
        """
        baixado = {"path": None}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-2"}})
            if request.url.path.startswith("/api/v1/tasks/"):
                return httpx.Response(
                    200,
                    json={
                        "status": 200,
                        "data": {
                            "state": 1,
                            "videos": ["/tasks/t-2/final-1.mp4"],
                            "combined_videos": ["/tasks/t-2/combined-1.mp4"],
                        },
                    },
                )
            baixado["path"] = request.url.path
            return httpx.Response(200, content=b"x")

        monkeypatch.setattr(
            "agent.adapters.mpt_renderer.probe_video",
            lambda p: {
                "width": 1080, "height": 1920, "duration_s": 70.0, "has_audio": True,
            },
        )
        result = make_renderer(handler, tmp_path).render(script)
        assert result.state is RenderState.complete
        assert baixado["path"] == "/api/v1/download/t-2/final-1.mp4"
        assert result.has_audio


class TestFalhas:
    def test_falha_de_dominio_vira_resultado_e_nao_excecao(self, script, tmp_path):
        """Render que falha e fato do dominio: precisa ser gravavel na memoria."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-3"}})
            return httpx.Response(
                200,
                json={"status": 200, "data": {"state": -1, "error": "pexels sem material"}},
            )

        result = make_renderer(handler, tmp_path).render(script)
        assert result.state is RenderState.failed
        assert "pexels" in result.error

    def test_fila_cheia_e_erro_de_infraestrutura(self, script, tmp_path):
        def handler(request):
            return httpx.Response(429, json={"status": 429, "message": "queue full"})

        with pytest.raises(RendererError, match="fila"):
            make_renderer(handler, tmp_path).render(script)

    def test_chave_errada_e_reportada_como_tal(self, script, tmp_path):
        def handler(request):
            return httpx.Response(401, json={"status": 401})

        with pytest.raises(RendererError, match="x-api-key"):
            make_renderer(handler, tmp_path).render(script)

    def test_timeout_nao_trava_para_sempre(self, script, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-4"}})
            return httpx.Response(200, json={"status": 200, "data": {"state": 4, "progress": 10}})

        renderer = make_renderer(handler, tmp_path)
        renderer.settings.renderer_timeout_s = 0.01
        with pytest.raises(RendererError, match="timeout"):
            renderer.render(script)

    def test_completo_sem_artefato_nao_passa_por_sucesso(self, script, tmp_path):
        """O pior modo de falha seria reportar sucesso sem video."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-5"}})
            return httpx.Response(200, json={"status": 200, "data": {"state": 1, "videos": []}})

        result = make_renderer(handler, tmp_path).render(script)
        assert result.state is RenderState.failed
        assert "sem video" in result.error

    def test_resposta_fora_do_envelope_e_erro_claro(self, script, tmp_path):
        def handler(request):
            return httpx.Response(200, text="<html>nginx</html>")

        with pytest.raises(RendererError, match="nao-JSON"):
            make_renderer(handler, tmp_path).render(script)


class TestHealth:
    def test_servico_fora_do_ar_nao_levanta(self, tmp_path):
        def handler(request):
            raise httpx.ConnectError("recusado")

        assert make_renderer(handler, tmp_path).health() is False

    def test_pong(self, tmp_path):
        def pong(request):
            return httpx.Response(200, json="pong")

        assert make_renderer(pong, tmp_path).health()
