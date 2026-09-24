"""Tests del adaptador del renderizador, contra un MPT simulado.

El contrato aquí se leyó del código de MoneyPrinterTurbo, no se supuso:
  - prefijo /api/v1 (app/controllers/v1/base.py)
  - envoltorio {"status": ..., "data": {...}} (app/utils/utils.py:get_response)
  - estados -1 fallo / 1 completo / 4 procesando (app/models/const.py)
  - artefacto como URI relativa /tasks/<...> cuando `endpoint` está vacío
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

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "guion_manual.json"


def _noop(request: httpx.Request) -> httpx.Response:
    """Handler que nunca debería llamarse: usado solo para montar payload."""
    return httpx.Response(200)


@pytest.fixture
def script() -> Script:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return Script.model_validate(raw)


def make_renderer(handler, tmp_path: Path) -> MptRenderer:
    settings = Settings(
        renderer_url="http://mpt.test",
        renderer_api_key="clave-de-prueba",
        renderer_poll_interval_s=0.0,
        renderer_timeout_s=5.0,
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
    )
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://mpt.test",
        headers={"x-api-key": "clave-de-prueba"},
    )
    return MptRenderer(settings=settings, client=client)


def test_adaptador_satisface_el_puerto(tmp_path):
    def _ok(request):
        return httpx.Response(200)

    assert isinstance(make_renderer(_ok, tmp_path), Renderer)


class TestPayload:
    def test_guion_listo_sortea_el_llm_del_mpt(self, script, tmp_path):
        """task.py:generate_script solo llama al LLM cuando video_script llega vacío.

        Si este campo dejara de enviarse, el MPT pasaría a generar guion propio
        en silencio -- el agente pierde el control sin que aparezca ningún error.
        """
        payload = make_renderer(_noop, tmp_path).build_payload(script)
        assert payload["video_script"].strip()
        assert payload["video_script"] == script.narration
        assert payload["video_terms"] == script.search_terms

    def test_pide_vertical_y_leyenda_karaoke(self, script, tmp_path):
        payload = make_renderer(_noop, tmp_path).build_payload(script)
        assert payload["video_aspect"] == "9:16"
        assert payload["subtitle_display_mode"] == "word_by_word"
        assert payload["subtitle_enabled"] is True

    def test_voz_y_fuente_compatibles_con_es(self, script, tmp_path):
        """Voz es-ES del edge-tts gratuito: el canal es de España.

        Y BeVietnamPro-Bold es la única fuente del MPT que tiene los acentos
        del castellano; las otras renderizan tofu en lugar de la "ñ" y los
        acentos.
        """
        payload = make_renderer(_noop, tmp_path).build_payload(script)
        assert payload["voice_name"].startswith("es-ES-")
        assert payload["font_name"] == "BeVietnamPro-Bold.ttf"


class TestFlujoFeliz:
    def test_renderiza_y_mide_el_resultado(self, script, tmp_path, monkeypatch):
        llamadas = {"post": 0, "get": 0, "download": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                llamadas["post"] += 1
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-1"}})
            if request.url.path == "/api/v1/tasks/t-1":
                llamadas["get"] += 1
                # primero procesando, después completo: garantiza que el poll corre
                if llamadas["get"] < 2:
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
                llamadas["download"] += 1
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
        assert llamadas == {"post": 1, "get": 2, "download": 1}
        assert Path(result.video_path).read_bytes() == b"\x00" * 1024

    def test_descarga_el_final_y_no_el_combinado(self, script, tmp_path, monkeypatch):
        """Regresión: `combined_videos` NO es el corte final.

        En el MPT, combined-N.mp4 es el concat solo de vídeo y final-N.mp4 es el
        corte con narración y leyenda. Preferir "combined" por el nombre entrega
        un MP4 mudo que pasa cualquier comprobación de dimensión y duración. Fue
        exactamente lo que pasó en la primera ejecución real contra el
        renderizador.
        """
        bajado = {"path": None}

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
            bajado["path"] = request.url.path
            return httpx.Response(200, content=b"x")

        monkeypatch.setattr(
            "agent.adapters.mpt_renderer.probe_video",
            lambda p: {
                "width": 1080, "height": 1920, "duration_s": 70.0, "has_audio": True,
            },
        )
        result = make_renderer(handler, tmp_path).render(script)
        assert result.state is RenderState.complete
        assert bajado["path"] == "/api/v1/download/t-2/final-1.mp4"
        assert result.has_audio


class TestFallos:
    def test_fallo_de_dominio_se_vuelve_resultado_y_no_excepcion(self, script, tmp_path):
        """Render que falla es un hecho del dominio: debe ser grabable en memoria."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-3"}})
            return httpx.Response(
                200,
                json={"status": 200, "data": {"state": -1, "error": "pexels sin material"}},
            )

        result = make_renderer(handler, tmp_path).render(script)
        assert result.state is RenderState.failed
        assert "pexels" in result.error

    def test_cola_llena_es_error_de_infraestructura(self, script, tmp_path):
        def handler(request):
            return httpx.Response(429, json={"status": 429, "message": "queue full"})

        with pytest.raises(RendererError, match="cola"):
            make_renderer(handler, tmp_path).render(script)

    def test_clave_errada_se_reporta_como_tal(self, script, tmp_path):
        def handler(request):
            return httpx.Response(401, json={"status": 401})

        with pytest.raises(RendererError, match="x-api-key"):
            make_renderer(handler, tmp_path).render(script)

    def test_timeout_no_se_cuelga_para_siempre(self, script, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-4"}})
            return httpx.Response(200, json={"status": 200, "data": {"state": 4, "progress": 10}})

        renderer = make_renderer(handler, tmp_path)
        renderer.settings.renderer_timeout_s = 0.01
        with pytest.raises(RendererError, match="timeout"):
            renderer.render(script)

    def test_completo_sin_artefacto_no_pasa_por_exito(self, script, tmp_path):
        """El peor modo de fallo sería reportar éxito sin vídeo."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/videos":
                return httpx.Response(200, json={"status": 200, "data": {"task_id": "t-5"}})
            return httpx.Response(200, json={"status": 200, "data": {"state": 1, "videos": []}})

        result = make_renderer(handler, tmp_path).render(script)
        assert result.state is RenderState.failed
        assert "sin video" in result.error

    def test_respuesta_fuera_del_envoltorio_es_error_claro(self, script, tmp_path):
        def handler(request):
            return httpx.Response(200, text="<html>nginx</html>")

        with pytest.raises(RendererError, match="no es JSON"):
            make_renderer(handler, tmp_path).render(script)


class TestHealth:
    def test_servicio_caido_no_levanta(self, tmp_path):
        def handler(request):
            raise httpx.ConnectError("rechazado")

        assert make_renderer(handler, tmp_path).health() is False

    def test_pong(self, tmp_path):
        def pong(request):
            return httpx.Response(200, json="pong")

        assert make_renderer(pong, tmp_path).health()
