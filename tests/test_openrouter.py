"""Testes do adaptador OpenRouter e da rota principal, sem rede.

Mesmo contrato dos outros adaptadores: payload que sai, traducao de erro em
acao (402/429 pedem caminhos opostos no roteador), tokens para o custo do M5.
Mais dois pontos proprios: a rota padrao abre com OpenRouter e fecha com
Gemini (a cota do AI Studio disputa com outras automacoes do autor), e o
slot das 20h tem ordem de formato forcada (carrossel por ultimo no dia).
"""

from __future__ import annotations

import httpx
import pytest

from agent.adapters.llm_factory import DEFAULT_ROUTES, build_llm, routes_for
from agent.adapters.openrouter import OpenRouter, quota_error
from agent.config import Settings
from agent.editorial.formats import FORMATO_FORCADO, FORMATS, choose_format, features
from agent.ports.llm import LLMBlocked, LLMError
from tests.test_editorial import dossie


def _cliente(status: int, body: dict | None = None, text: str | None = None,
             headers: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text, headers=headers or {})
        return httpx.Response(status, json=body or {}, headers=headers or {})

    return httpx.Client(
        base_url="https://exemplo", transport=httpx.MockTransport(handler)
    )


def orouter(status: int = 200, body: dict | None = None, text: str | None = None,
            headers: dict | None = None, **kwargs) -> OpenRouter:
    return OpenRouter(api_key="chave-de-teste",
                      client=_cliente(status, body, text, headers), **kwargs)


def _ok(texto: str = '{"ok": "sim"}') -> dict:
    return {"choices": [{"message": {"content": texto}, "finish_reason": "stop"}],
            "model": "deepseek/deepseek-v4-flash",
            "usage": {"prompt_tokens": 12, "completion_tokens": 5}}


class TestChave:
    def test_sem_chave_diz_onde_pegar(self):
        with pytest.raises(LLMError, match="openrouter.ai/keys"):
            OpenRouter(api_key="")

    def test_build_llm_sem_chave_explicita_nao_le_env(self):
        cfg = Settings(openrouter_api_key="", groq_api_key="", gemini_api_key="")
        with pytest.raises(LLMError):
            build_llm("openrouter", settings=cfg)


class TestPayload:
    SCHEMA = {"type": "object", "properties": {"ok": {"type": "string"}},
              "required": ["ok"]}

    def test_schema_vai_no_prompt_e_json_object(self):
        p = OpenRouter.build_payload("diga ok", system="seja breve", schema=self.SCHEMA,
                                     model="modelo/qualquer-sem-medida",
                                     temperature=0.2, max_output_tokens=64)
        assert p["response_format"] == {"type": "json_object"}
        assert "json" in p["messages"][0]["content"].lower()
        assert "schema" in p["messages"][0]["content"].lower()

    def test_modelo_medidio_usa_json_schema_estrito(self):
        from agent.adapters.openrouter import STRICT_SCHEMA_MODELS

        assert {"deepseek/deepseek-v4-flash",
                "google/gemini-2.5-flash-lite",
                "google/gemini-2.5-flash",
                "deepseek/deepseek-v4-pro"} <= set(STRICT_SCHEMA_MODELS)
        p = OpenRouter.build_payload("diga ok", system="seja breve", schema=self.SCHEMA,
                                     model="deepseek/deepseek-v4-flash",
                                     temperature=0.2, max_output_tokens=64)
        rf = p["response_format"]
        assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
        # No modo estrito o schema sai do prompt: menos token de entrada.
        assert "schema" not in p["messages"][0]["content"].lower()

    def test_sem_schema_sem_response_format(self):
        p = OpenRouter.build_payload("oi", system="", schema=None,
                                     model="deepseek/deepseek-v4-flash",
                                     temperature=0.2, max_output_tokens=64)
        assert "response_format" not in p
        assert [m["role"] for m in p["messages"]] == ["user"]

    def test_effort_vira_reasoning(self):
        p = OpenRouter.build_payload("oi", system="", schema=None,
                                     model="deepseek/deepseek-v4-flash",
                                     temperature=0.2, max_output_tokens=64,
                                     reasoning_effort="low")
        assert p["reasoning"] == {"effort": "low"}

    def test_sem_effort_sem_reasoning(self):
        p = OpenRouter.build_payload("oi", system="", schema=None,
                                     model="deepseek/deepseek-v4-flash",
                                     temperature=0.2, max_output_tokens=64)
        assert "reasoning" not in p

    def test_referer_opcional(self):
        sem = OpenRouter(api_key="x")
        assert "HTTP-Referer" not in sem._client.headers
        com = OpenRouter(api_key="x", app_url="https://exemplo.app")
        assert com._client.headers["HTTP-Referer"] == "https://exemplo.app"


class TestResposta:
    def test_ok_conta_tokens(self):
        r = orouter(body=_ok()).complete("diga ok")
        assert r.usage.total_tokens == 17
        assert r.provider == "openrouter"

    def test_sem_choice_com_erro_do_provedor(self):
        r = orouter(body={"error": {"message": "provider overloaded"}})
        with pytest.raises(LLMError, match="provider overloaded"):
            r.complete("oi")

    def test_filtro_de_conteudo_sobe(self):
        corpo = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
        with pytest.raises(LLMBlocked):
            orouter(body=corpo).complete("oi")

    def test_500_e_indisponibilidade(self):
        from agent.ports.llm import LLMUnavailable

        with pytest.raises(LLMUnavailable):
            orouter(status=503, text="busy").complete("oi")


class TestCota:
    def test_402_e_cota_do_dia(self):
        r = httpx.Response(402, json={"error": {"message": "Insufficient credits"}})
        erro = quota_error(r, "m")
        assert erro.scope == "day" and erro.retry_after_s == 86400.0

    def test_429_respeita_retry_after(self):
        r = httpx.Response(429, json={"error": {"message": "Rate limit"}},
                           headers={"retry-after": "7"})
        erro = quota_error(r, "m")
        assert erro.scope == "minute" and erro.retry_after_s == 7.0


class TestRotas:
    @pytest.mark.parametrize(
        "estagio", sorted(set(DEFAULT_ROUTES) - {"writer_short"}))
    def test_openrouter_abre_e_gemini_fecha(self, estagio):
        rotas = routes_for(estagio)
        assert rotas[0].provider == "openrouter"
        pos_gemini = [i for i, r in enumerate(rotas) if r.provider == "gemini"]
        pos_or = [i for i, r in enumerate(rotas) if r.provider == "openrouter"]
        assert pos_gemini and max(pos_or) < min(pos_gemini)

    def test_escritor_abre_no_gemini_25_flash_e_juiz_no_pro(self):
        assert routes_for("writer")[0].key == "openrouter:google/gemini-2.5-flash"
        assert routes_for("humanize")[0].key == "openrouter:google/gemini-2.5-flash"
        assert routes_for("judge")[0].key == "openrouter:deepseek/deepseek-v4-pro"
        assert routes_for("research")[0].key == "openrouter:deepseek/deepseek-v4-flash"

    def test_curto_abre_no_groq_e_sem_disputar_gemini(self):
        rotas = routes_for("writer_short")
        assert rotas[0].key == "groq:openai/gpt-oss-120b"
        assert rotas[0].reasoning_effort == "low"
        pos_gemini = [i for i, r in enumerate(rotas) if r.provider == "gemini"]
        pos_or = [i for i, r in enumerate(rotas) if r.provider == "openrouter"]
        assert pos_gemini and max(pos_or) < min(pos_gemini)


class TestGradeForcada:
    def test_quatro_slots_alternam_curto_e_longo(self):
        """A grade de 20/09/2026: 4 posts por dia, todos video.

        Curto de manha e na tarde (alcance), longo no almoco e a noite
        (monetizacao -- o Rewards exige 60s). O carrossel saiu dos horarios
        fixos: nenhum slot o forca, e ele passa a sair so por `slot-extra`.
        """
        assert set(FORMATO_FORCADO) == {"0900", "1200", "1600", "1900"}
        assert FORMATO_FORCADO["0900"] == ("short",)
        assert FORMATO_FORCADO["1600"] == ("short",)
        # Longo com curto de emergencia: dossie com menos de 3 fatos nao
        # sustenta um longo, e o slot sai curto em vez de falhar.
        assert FORMATO_FORCADO["1200"] == ("long", "short")
        assert FORMATO_FORCADO["1900"] == ("long", "short")
        assert not any("carousel" in v for v in FORMATO_FORCADO.values())

    def test_ordem_forcada_e_ordem_e_nao_lista_de_permitidos(self):
        """Com dossie que sustenta longo, o slot de longo TEM de sair longo.

        Bug silencioso: `FORMATO_FORCADO` sempre foi descrito como ordem ("o
        segundo elemento e emergencia, nao alternativa") e a funcao escolhia
        pela NOTA dentro do permitido. Passou despercebido enquanto o unico
        slot forcado era o carrossel das 20h, que pontuava alto sozinho. Com
        `("long", "short")` no almoco e na noite, o curto ganha do longo em
        pilar de noticia (afinidade 0,50 x 0,35) e os dois slots longos do dia
        sairiam curtos -- o contrario do que a grade pede.
        """
        rico = features(dossie(
            "O modelo ocupa 5,9 GB e mantem 98% do desempenho",
            "A versao anterior pesava 54 GB em disco",
            "O treino levou 3 semanas em 2026",
            "A licenca permite uso comercial sem taxa"))
        for slot in ("1200", "1900"):
            d = choose_format(rico, "news", slot,
                              allowed=FORMATO_FORCADO[slot], em_ordem=True)
            assert d.format == "long", f"{slot}: {d.reason}"

    def test_emergencia_entra_quando_o_dossie_nao_sustenta(self):
        """Ordem nao e teimosia: dossie fino vira curto em vez de falhar."""
        fino = features(dossie("O modelo ocupa 5,9 GB e mantem 98% do desempenho"))
        d = choose_format(fino, "news", "1900",
                          allowed=FORMATO_FORCADO["1900"], em_ordem=True)
        assert d.format == "short"

    def test_sem_grade_forcada_a_nota_decide(self):
        """`em_ordem` vale so para a grade; fora dela a nota continua mandando."""
        rico = features(dossie(
            "O modelo ocupa 5,9 GB e mantem 98% do desempenho",
            "A versao anterior pesava 54 GB em disco",
            "O treino levou 3 semanas em 2026",
            "A licenca permite uso comercial sem taxa"))
        d = choose_format(rico, "news", "0900")
        assert d.format in FORMATS and d.scores

    def test_todo_slot_da_grade_tem_prior(self):
        """Slot sem prior caia num `SLOT_PRIOR["1500"]` que deixou de existir."""
        from agent.editorial.formats import SLOT_PRIOR
        from agent.editorial.slots import SLOTS

        assert set(SLOTS) <= set(SLOT_PRIOR)

    def test_allowed_respeitado_na_escolha(self):
        feat = features(dossie(
            "Passo 1: instale o CLI com um comando",
            "Passo 2: configure o arquivo na raiz",
            "Passo 3: execute o agente no terminal",
            "O erro comum e esquecer de habilitar a leitura"))
        d = choose_format(feat, "tutorial", "1900", allowed=("carousel", "short"))
        assert d.format == "carousel"
        assert set(d.scores) <= {"carousel", "short"}
