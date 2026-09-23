"""Roteador de LLM: cada negativa de cota vira a acao certa, sem rede."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from agent.adapters.gemini_free import quota_error as gemini_quota
from agent.adapters.groq import Groq, duration_s, strict_schema
from agent.adapters.groq import quota_error as groq_quota
from agent.adapters.router import Route, RoutedLLM
from agent.memory.llm_ledger import LLMLedger, next_pacific_midnight
from agent.ports.llm import (
    Completion,
    LLMBlocked,
    LLMError,
    LLMQuotaExhausted,
    LLMUnavailable,
    Usage,
)

AGORA = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


@dataclass
class Falso:
    """LLM que devolve/levanta o que o roteiro de teste mandar, em ordem."""

    nome: str
    roteiro: list = field(default_factory=list)
    chamadas: int = 0
    provider: str = "fake"
    model: str = "fake"

    def complete(self, prompt, *, system="", schema=None, temperature=0.2,
                 max_output_tokens=2048):
        self.chamadas += 1
        item = self.roteiro.pop(0) if self.roteiro else "ok"
        if isinstance(item, Exception):
            raise item
        return Completion(text=f"{self.nome}:{item}", model=self.nome, provider="fake",
                          usage=Usage(input_tokens=10, output_tokens=5))


def roteador(falsos: dict[str, Falso], ledger=None, esperas=None) -> RoutedLLM:
    rotas = [Route.parse(k) for k in falsos]
    dormidas = esperas if esperas is not None else []
    return RoutedLLM(stage="teste", routes=rotas, builder=lambda r: falsos[r.key],
                     ledger=ledger, sleeper=dormidas.append, clock=lambda: AGORA)


@pytest.fixture
def livro(tmp_path) -> LLMLedger:
    return LLMLedger(tmp_path / "agent.db")


class TestRotas:
    def test_primeiro_que_responde_ganha(self, livro):
        a, b = Falso("a"), Falso("b")
        r = roteador({"gemini:a": a, "groq:b": b}, livro)
        assert r.complete("x").text == "a:ok"
        assert (r.provider, r.model) == ("gemini", "a") and b.chamadas == 0

    def test_cota_do_dia_troca_de_modelo_e_fica_no_livro(self, livro):
        a = Falso("a", [LLMQuotaExhausted("dia", scope="day", limit="20")])
        b = Falso("b")
        r = roteador({"gemini:a": a, "groq:b": b}, livro)
        assert r.complete("x").text == "b:ok"
        # O proximo slot (outro processo, mesmo banco) nem tenta o esgotado.
        ate = livro.exhausted_until("gemini:a", AGORA)
        assert ate == next_pacific_midnight(AGORA)
        r2 = roteador({"gemini:a": Falso("a"), "groq:b": Falso("b")}, livro)
        assert r2.complete("x").text == "b:ok"
        assert "esgotado" in r.trail[0] or "dia" in r.trail[0]

    def test_cota_do_minuto_espera_e_repete_o_mesmo(self, livro):
        esperas: list[float] = []
        a = Falso("a", [LLMQuotaExhausted("tpm", scope="minute", retry_after_s=7.0)])
        b = Falso("b")
        r = roteador({"groq:a": a, "gemini:b": b}, livro, esperas)
        assert r.complete("x").text == "a:ok"
        assert esperas == [7.5] and b.chamadas == 0

    def test_espera_longa_demais_segue_a_rota(self, livro):
        a = Falso("a", [LLMQuotaExhausted("tpm", scope="minute", retry_after_s=600)])
        r = roteador({"groq:a": a, "gemini:b": Falso("b")}, livro)
        assert r.complete("x").text == "b:ok"
        assert livro.exhausted_until("groq:a", AGORA) is not None

    def test_pedido_grande_demais_nao_condena_o_modelo(self, livro):
        a = Falso("a", [LLMQuotaExhausted("413", scope="request", limit="TPM")])
        r = roteador({"groq:a": a, "gemini:b": Falso("b")}, livro)
        assert r.complete("x").text == "b:ok"
        assert livro.exhausted_until("groq:a", AGORA) is None

    def test_instabilidade_de_rede_tem_uma_segunda_chance(self, livro):
        a = Falso("a", [LLMUnavailable("gemini inacessivel: timeout")])
        r = roteador({"gemini:a": a, "groq:b": Falso("b")}, livro)
        assert r.complete("x").text == "a:ok" and a.chamadas == 2

    def test_sobrecarga_503_tira_o_modelo_por_minutos(self, livro):
        """Medido no primeiro slot: o 503 do Gemini 3.x repetia na segunda
        chance e cada um custava 10-20s. Um 503 basta para seguir a rota."""
        a = Falso("a", [LLMUnavailable("gemini instavel (503)")])
        r = roteador({"gemini:a": a, "groq:b": Falso("b")}, livro)
        assert r.complete("x").text == "b:ok" and a.chamadas == 1
        ate = livro.exhausted_until("gemini:a", AGORA)
        assert ate is not None and ate <= AGORA + timedelta(minutes=11)

    def test_filtro_de_conteudo_sobe_sem_trocar_de_modelo(self, livro):
        b = Falso("b")
        r = roteador({"gemini:a": Falso("a", [LLMBlocked("safety")]), "groq:b": b}, livro)
        with pytest.raises(LLMBlocked):
            r.complete("x")
        assert b.chamadas == 0

    def test_modelo_inexistente_fica_fora_um_dia(self, livro):
        a = Falso("a", [LLMError("gemini devolveu 404: models/x not found")])
        r = roteador({"gemini:a": a, "groq:b": Falso("b")}, livro)
        assert r.complete("x").text == "b:ok"
        assert livro.exhausted_until("gemini:a", AGORA) == AGORA + timedelta(hours=24)

    def test_todos_fora_e_indisponibilidade_com_motivo(self, livro):
        r = roteador({
            "gemini:a": Falso("a", [LLMQuotaExhausted("d", scope="day")]),
            "groq:b": Falso("b", [LLMQuotaExhausted("d", scope="day")]),
        }, livro)
        with pytest.raises(LLMUnavailable, match="gemini:a"):
            r.complete("x")

    def test_chave_ausente_pula_a_rota(self, livro):
        def builder(rota):
            if rota.provider == "gemini":
                raise LLMError("AGENT_GEMINI_API_KEY vazia")
            return Falso("b")
        r = RoutedLLM(stage="t", routes=[Route.parse("gemini:a"), Route.parse("groq:b")],
                      builder=builder, ledger=livro, clock=lambda: AGORA)
        assert r.complete("x").text == "b:ok"

    def test_toda_chamada_vai_para_o_livro(self, livro):
        a = Falso("a", [LLMQuotaExhausted("d", scope="day")])
        r = roteador({"gemini:a": a, "groq:b": Falso("b")}, livro)
        r.complete("x")
        uso = livro.usage_by_route(AGORA - timedelta(minutes=1))
        assert uso["gemini:a"]["failed"] == 1 and uso["groq:b"]["tokens"] == 15

    def test_juiz_prefere_outra_familia(self, livro):
        r = roteador({"groq:a": Falso("a"), "gemini:b": Falso("b")}, livro)
        juiz = r.prefer_other_than("groq")
        assert [x.provider for x in juiz.routes] == ["gemini", "groq"]

    def test_parse_da_rota(self):
        rota = Route.parse("groq:openai/gpt-oss-120b@low")
        assert (rota.provider, rota.model, rota.reasoning_effort) == (
            "groq", "openai/gpt-oss-120b", "low")
        with pytest.raises(ValueError):
            Route.parse("sem-provedor")


def _resposta(status: int, corpo: dict, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=corpo, headers=headers or {},
                          request=httpx.Request("POST", "https://exemplo"))


class TestTraducaoDeCota:
    def test_gemini_dia_com_teto_real(self):
        """O 429 real de 19/09: gemini-2.5-flash, 20 pedidos/dia."""
        exc = gemini_quota(_resposta(429, {"error": {"code": 429, "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                 "quotaValue": "20"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "11s"},
        ]}}), "gemini-2.5-flash")
        assert (exc.scope, exc.limit, exc.retry_after_s) == ("day", "20", 11.0)

    def test_gemini_minuto(self):
        exc = gemini_quota(_resposta(429, {"error": {"details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                 "quotaValue": "10"}]}]}}), "m")
        assert exc.scope == "minute"

    def test_gemini_modelo_fora_do_free_tier(self):
        exc = gemini_quota(_resposta(429, {"error": {"details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                 "quotaValue": "0"}]}]}}), "lyria")
        assert exc.scope == "day"

    def test_groq_tpm_com_espera(self):
        exc = groq_quota(_resposta(429, {"error": {"message": (
            "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute "
            "(TPM): Limit 8000, Used 5120, Requested 3811. Please try again in 7.07s."),
            "code": "rate_limit_exceeded"}}), "openai/gpt-oss-120b")
        assert (exc.scope, exc.limit) == ("minute", "TPM")
        assert exc.retry_after_s == pytest.approx(7.07)

    def test_groq_dia(self):
        exc = groq_quota(_resposta(429, {"error": {"message": (
            "Rate limit reached on requests per day (RPD): Limit 1000, Used 1000. "
            "Please try again in 1h2m3s.")}}), "m")
        assert exc.scope == "day" and exc.retry_after_s == 3723

    def test_groq_413_e_pedido_grande(self):
        exc = groq_quota(_resposta(413, {"error": {"message": (
            "Request too large for model on tokens per minute (TPM): Limit 8000, "
            "Requested 9500")}}), "m")
        assert exc.scope == "request"

    def test_duracao_composta(self):
        assert duration_s("2m59.56s") == pytest.approx(179.56)
        assert duration_s("510ms") == pytest.approx(0.51)


class TestSchemaEstritoDoGroq:
    SCHEMA = {"type": "object",
              "properties": {"facts": {"type": "array", "items": {
                  "type": "object",
                  "properties": {"quote": {"type": "string"}, "claim": {"type": "string"}},
                  "required": ["quote", "claim"]}}},
              "required": ["facts"]}

    def test_objetos_fechados_e_tudo_obrigatorio(self):
        s = strict_schema(self.SCHEMA)
        assert s["additionalProperties"] is False
        item = s["properties"]["facts"]["items"]
        assert item["additionalProperties"] is False and item["required"] == ["quote", "claim"]
        # O original fica intacto: o Gemini recebe o mesmo dict.
        assert "additionalProperties" not in self.SCHEMA

    def test_modelo_estrito_nao_leva_schema_no_prompt(self):
        p = Groq.build_payload("x", system="sys", schema=self.SCHEMA,
                               model="openai/gpt-oss-120b", temperature=0.1,
                               max_output_tokens=64)
        assert p["response_format"]["type"] == "json_schema"
        assert p["response_format"]["json_schema"]["strict"] is True
        assert '"facts"' not in p["messages"][0]["content"]

    def test_esforco_de_raciocinio_so_no_gpt_oss(self):
        enviados: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            import json
            enviados.append(json.loads(req.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})

        g = Groq(api_key="k", model="qwen/qwen3.8-27b", reasoning_effort="low",
                 client=httpx.Client(base_url="https://exemplo",
                                     transport=httpx.MockTransport(handler)))
        assert g.complete("x").text == "{}"
        # O parametro nao vai para o qwen (que devolveria 400).
        assert "reasoning_effort" not in enviados[0]
