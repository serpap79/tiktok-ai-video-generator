"""Pruebas del adaptador OpenRouter y de la ruta principal, sin red.

Mismo contrato que el resto de adaptadores: payload que sale, traducción de error en
acción (402/429 piden caminos opuestos en el enrutador), tokens para el coste del M5.
Dos puntos propios más: la ruta por defecto abre con OpenRouter y cierra con
Gemini (la cuota de AI Studio compite con otras automatizaciones del autor), y el
hueco de las 20:00 tiene orden de formato forzado.
"""

from __future__ import annotations

import httpx
import pytest

from agent.adapters.llm_factory import DEFAULT_ROUTES, build_llm, routes_for
from agent.adapters.openrouter import OpenRouter, quota_error
from agent.config import Settings
from agent.editorial.formats import FORMATO_FORZADO, FORMATS, choose_format, features
from agent.ports.llm import LLMBlocked, LLMError
from tests.test_editorial import dossier


def _cliente(status: int, body: dict | None = None, text: str | None = None,
             headers: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text, headers=headers or {})
        return httpx.Response(status, json=body or {}, headers=headers or {})

    return httpx.Client(
        base_url="https://ejemplo", transport=httpx.MockTransport(handler)
    )


def openrouter_prueba(status: int = 200, body: dict | None = None,
                      text: str | None = None,
                      headers: dict | None = None, **kwargs) -> OpenRouter:
    return OpenRouter(api_key="clave-de-prueba",
                      client=_cliente(status, body, text, headers), **kwargs)


def _ok(texto: str = '{"ok": "sí"}') -> dict:
    return {"choices": [{"message": {"content": texto}, "finish_reason": "stop"}],
            "model": "deepseek/deepseek-v4-flash",
            "usage": {"prompt_tokens": 12, "completion_tokens": 5}}


class TestClave:
    def test_sin_clave_dice_donde_conseguirla(self):
        with pytest.raises(LLMError, match="openrouter.ai/keys"):
            OpenRouter(api_key="")

    def test_build_llm_sin_clave_explicita_no_lee_el_entorno(self):
        cfg = Settings(openrouter_api_key="", groq_api_key="", gemini_api_key="")
        with pytest.raises(LLMError):
            build_llm("openrouter", settings=cfg)


class TestPayload:
    SCHEMA = {"type": "object", "properties": {"ok": {"type": "string"}},
              "required": ["ok"]}

    def test_el_schema_va_en_el_prompt_y_json_object(self):
        p = OpenRouter.build_payload("di ok", system="sé breve", schema=self.SCHEMA,
                                     model="modelo/cualquier-sin-medida",
                                     temperature=0.2, max_output_tokens=64)
        assert p["response_format"] == {"type": "json_object"}
        assert "json" in p["messages"][0]["content"].lower()
        assert "schema" in p["messages"][0]["content"].lower()

    def test_modelo_medido_usa_json_schema_estricto(self):
        from agent.adapters.openrouter import STRICT_SCHEMA_MODELS

        assert {"deepseek/deepseek-v4-flash",
                "google/gemini-2.5-flash-lite",
                "google/gemini-2.5-flash",
                "deepseek/deepseek-v4-pro"} <= set(STRICT_SCHEMA_MODELS)
        p = OpenRouter.build_payload("di ok", system="sé breve", schema=self.SCHEMA,
                                     model="deepseek/deepseek-v4-flash",
                                     temperature=0.2, max_output_tokens=64)
        rf = p["response_format"]
        assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
        # En modo estricto el schema sale del prompt: menos tokens de entrada.
        assert "schema" not in p["messages"][0]["content"].lower()

    def test_sin_schema_sin_response_format(self):
        p = OpenRouter.build_payload("hola", system="", schema=None,
                                     model="deepseek/deepseek-v4-flash",
                                     temperature=0.2, max_output_tokens=64)
        assert "response_format" not in p
        assert [m["role"] for m in p["messages"]] == ["user"]

    def test_el_esfuerzo_se_convierte_en_reasoning(self):
        p = OpenRouter.build_payload("hola", system="", schema=None,
                                     model="deepseek/deepseek-v4-flash",
                                     temperature=0.2, max_output_tokens=64,
                                     reasoning_effort="low")
        assert p["reasoning"] == {"effort": "low"}

    def test_sin_esfuerzo_sin_reasoning(self):
        p = OpenRouter.build_payload("hola", system="", schema=None,
                                     model="deepseek/deepseek-v4-flash",
                                     temperature=0.2, max_output_tokens=64)
        assert "reasoning" not in p

    def test_referer_opcional(self):
        sin = OpenRouter(api_key="x")
        assert "HTTP-Referer" not in sin._client.headers
        con = OpenRouter(api_key="x", app_url="https://ejemplo.app")
        assert con._client.headers["HTTP-Referer"] == "https://ejemplo.app"


class TestRespuesta:
    def test_ok_cuenta_tokens(self):
        r = openrouter_prueba(body=_ok()).complete("di ok")
        assert r.usage.total_tokens == 17
        assert r.provider == "openrouter"

    def test_sin_choice_con_error_del_proveedor(self):
        r = openrouter_prueba(body={"error": {"message": "provider overloaded"}})
        with pytest.raises(LLMError, match="provider overloaded"):
            r.complete("hola")

    def test_el_filtro_de_contenido_sube(self):
        cuerpo = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
        with pytest.raises(LLMBlocked):
            openrouter_prueba(body=cuerpo).complete("hola")

    def test_500_es_indisponibilidad(self):
        from agent.ports.llm import LLMUnavailable

        with pytest.raises(LLMUnavailable):
            openrouter_prueba(status=503, text="busy").complete("hola")


class TestCuota:
    def test_402_es_cuota_del_dia(self):
        r = httpx.Response(402, json={"error": {"message": "Insufficient credits"}})
        error = quota_error(r, "m")
        assert error.scope == "day" and error.retry_after_s == 86400.0

    def test_429_respeta_retry_after(self):
        r = httpx.Response(429, json={"error": {"message": "Rate limit"}},
                           headers={"retry-after": "7"})
        error = quota_error(r, "m")
        assert error.scope == "minute" and error.retry_after_s == 7.0


class TestRutas:
    @pytest.mark.parametrize(
        "etapa", sorted(set(DEFAULT_ROUTES) - {"writer_short"}))
    def test_openrouter_abre_y_gemini_cierra(self, etapa):
        rutas = routes_for(etapa)
        assert rutas[0].provider == "openrouter"
        pos_gemini = [i for i, r in enumerate(rutas) if r.provider == "gemini"]
        pos_or = [i for i, r in enumerate(rutas) if r.provider == "openrouter"]
        assert pos_gemini and max(pos_or) < min(pos_gemini)

    def test_el_guionista_abre_en_gemini_25_flash_y_el_juez_en_pro(self):
        assert routes_for("writer")[0].key == "openrouter:google/gemini-2.5-flash"
        assert routes_for("humanize")[0].key == "openrouter:google/gemini-2.5-flash"
        assert routes_for("judge")[0].key == "openrouter:deepseek/deepseek-v4-pro"
        assert routes_for("research")[0].key == "openrouter:deepseek/deepseek-v4-flash"

    def test_el_corto_abre_en_groq_y_no_disputa_con_gemini(self):
        rutas = routes_for("writer_short")
        assert rutas[0].key == "groq:openai/gpt-oss-120b"
        assert rutas[0].reasoning_effort == "low"
        pos_gemini = [i for i, r in enumerate(rutas) if r.provider == "gemini"]
        pos_or = [i for i, r in enumerate(rutas) if r.provider == "openrouter"]
        assert pos_gemini and max(pos_or) < min(pos_gemini)


class TestCuadranteForzado:
    def test_cuatro_huecos_alternan_corto_y_largo(self):
        """El cuadrante del 20/09/2026: 4 publicaciones al día, todas en vídeo.

        Corto por la mañana y por la tarde (alcance), largo a la comida y por la
        noche (monetización -- Rewards exige 60 s). El carrusel salió de los horarios
        fijos: ningún hueco lo fuerza, y pasa a salir solo por `slot-extra`.
        """
        assert set(FORMATO_FORZADO) == {"1000", "1300", "1700", "2000"}
        assert FORMATO_FORZADO["1000"] == ("short",)
        assert FORMATO_FORZADO["1700"] == ("short",)
        # Largo con corto de emergencia: un dossier con menos de 3 hechos no
        # sostiene un largo, y el hueco sale corto en lugar de fallar.
        assert FORMATO_FORZADO["1300"] == ("long", "short")
        assert FORMATO_FORZADO["2000"] == ("long", "short")
        assert not any("carousel" in v for v in FORMATO_FORZADO.values())

    def test_el_orden_forzado_es_orden_y_no_lista_de_permitidos(self):
        """Con un dossier que sostiene largo, el hueco de largo TIENE que salir largo.

        Fallo silencioso: `FORMATO_FORZADO` siempre se describió como orden ("el
        segundo elemento es emergencia, no alternativa") y la función elegía
        por la NOTA dentro de lo permitido. Pasó desapercibido mientras el único
        hueco forzado era el carrusel de las 20:00, que puntuaba alto por sí solo. Con
        `("long", "short")` a la comida y por la noche, el corto gana al largo en
        pilar de noticia (afinidad 0,50 x 0,35) y los dos huecos largos del día
        saldrían cortos -- al contrario de lo que pide el cuadrante.
        """
        rico = features(dossier(
            "El modelo ocupa 5,9 GB y mantiene el 98% del rendimiento",
            "La versión anterior pesaba 54 GB en disco",
            "El entrenamiento llevó 3 semanas en 2026",
            "La licencia permite uso comercial sin tasa"))
        for hueco in ("1300", "2000"):
            d = choose_format(rico, "news", hueco,
                              allowed=FORMATO_FORZADO[hueco], en_orden=True)
            assert d.format == "long", f"{hueco}: {d.reason}"

    def test_la_emergencia_entra_cuando_el_dossier_no_sostiene(self):
        """El orden no es terquedad: un dossier fino sale corto en vez de fallar."""
        fino = features(dossier("El modelo ocupa 5,9 GB y mantiene el 98% del rendimiento"))
        d = choose_format(fino, "news", "2000",
                          allowed=FORMATO_FORZADO["2000"], en_orden=True)
        assert d.format == "short"

    def test_sin_cuadrante_forzado_decide_la_nota(self):
        """`en_orden` solo vale para el cuadrante; fuera de él manda la nota."""
        rico = features(dossier(
            "El modelo ocupa 5,9 GB y mantiene el 98% del rendimiento",
            "La versión anterior pesaba 54 GB en disco",
            "El entrenamiento llevó 3 semanas en 2026",
            "La licencia permite uso comercial sin tasa"))
        d = choose_format(rico, "news", "1000")
        assert d.format in FORMATS and d.scores

    def test_todo_hueco_del_cuadrante_tiene_prior(self):
        """Un hueco sin priori caía en un `SLOT_PRIOR["1500"]` que dejó de existir."""
        from agent.editorial.formats import SLOT_PRIOR
        from agent.editorial.slots import SLOTS

        assert set(SLOTS) <= set(SLOT_PRIOR)

    def test_allowed_se_respeta_en_la_eleccion(self):
        feat = features(dossier(
            "Paso 1: instala la CLI con una orden",
            "Paso 2: configura el archivo en la raíz",
            "Paso 3: ejecuta el agente en la terminal",
            "El error habitual es olvidar habilitar la lectura"))
        d = choose_format(feat, "tutorial", "2000", allowed=("carousel", "short"))
        assert d.format == "carousel"
        assert set(d.scores) <= {"carousel", "short"}
