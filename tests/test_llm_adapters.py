"""Testes dos adaptadores Gemini e Groq, sem rede.

O que importa aqui nao e "o modelo respondeu bem" -- isso nao e testavel offline
e nem seria teste de unidade. E o contrato: o payload que sai, a traducao do
schema, a contagem de tokens que entra no custo do M5, e o tipo de excecao para
cada falha, porque cota negada e filtro de conteudo pedem acoes opostas.
"""

from __future__ import annotations

import httpx
import pytest

from agent.adapters.gemini_free import GeminiFree, to_openapi_schema
from agent.adapters.groq import Groq
from agent.ports.llm import LLM, LLMBlocked, LLMError, LLMUnavailable

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"quote": {"type": "string"}, "claim": {"type": "string"}},
                "required": ["quote", "claim"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
}


def _cliente(status: int, body: dict | None, text: str | None,
             headers: dict[str, str] | None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text, headers=headers or {})
        return httpx.Response(status, json=body or {}, headers=headers or {})

    return httpx.Client(
        base_url="https://exemplo", transport=httpx.MockTransport(handler)
    )


def gemini(status: int = 200, body: dict | None = None, text: str | None = None,
           headers: dict[str, str] | None = None) -> GeminiFree:
    return GeminiFree(api_key="chave-de-teste", client=_cliente(status, body, text, headers))


def groq(status: int = 200, body: dict | None = None, text: str | None = None,
         headers: dict[str, str] | None = None) -> Groq:
    return Groq(api_key="chave-de-teste", client=_cliente(status, body, text, headers))


class TestChaveFaltando:
    def test_gemini_diz_onde_pegar_a_chave(self):
        with pytest.raises(LLMError, match="aistudio.google.com"):
            GeminiFree(api_key="")

    def test_groq_diz_onde_pegar_a_chave(self):
        with pytest.raises(LLMError, match="console.groq.com"):
            Groq(api_key="")


class TestSchemaDoGemini:
    """O responseSchema e um subconjunto do OpenAPI, nao JSON Schema."""

    def test_tipos_viram_maiusculas(self):
        convertido = to_openapi_schema(SCHEMA)
        assert convertido["type"] == "OBJECT"
        assert convertido["properties"]["facts"]["type"] == "ARRAY"
        assert convertido["properties"]["facts"]["items"]["type"] == "OBJECT"

    def test_chave_que_o_endpoint_recusa_e_removida(self):
        item = to_openapi_schema(SCHEMA)["properties"]["facts"]["items"]
        assert "additionalProperties" not in item

    def test_ordem_das_chaves_e_preservada_explicitamente(self):
        """A ordem muda o que o modelo escreve: trecho antes de afirmacao faz a
        afirmacao sair do trecho, e nao o contrario."""
        item = to_openapi_schema(SCHEMA)["properties"]["facts"]["items"]
        assert item["propertyOrdering"] == ["quote", "claim"]

    def test_required_passa_intacto(self):
        assert to_openapi_schema(SCHEMA)["required"] == ["facts"]


class TestPayloadGemini:
    def test_system_vai_em_campo_proprio(self):
        p = GeminiFree.build_payload(
            "pergunta", system="instrucao", schema=None,
            temperature=0.1, max_output_tokens=256,
        )
        assert p["systemInstruction"]["parts"][0]["text"] == "instrucao"
        assert p["contents"][0]["parts"][0]["text"] == "pergunta"
        assert "responseSchema" not in p["generationConfig"]

    def test_schema_liga_o_modo_json(self):
        p = GeminiFree.build_payload(
            "x", system="", schema=SCHEMA, temperature=0.1, max_output_tokens=256,
        )
        assert p["generationConfig"]["responseMimeType"] == "application/json"
        assert p["generationConfig"]["responseSchema"]["type"] == "OBJECT"


class TestRespostaGemini:
    def test_texto_tokens_e_latencia(self):
        r = gemini(body={
            "candidates": [{
                "content": {"parts": [{"text": '{"facts": []}'}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 80},
            "modelVersion": "gemini-2.5-flash-001",
        }).complete("x", schema=SCHEMA)
        assert r.text == '{"facts": []}'
        assert r.model == "gemini-2.5-flash-001"
        assert r.provider == "gemini"
        assert (r.usage.input_tokens, r.usage.output_tokens) == (1200, 80)
        assert r.latency_s >= 0

    def test_tokens_de_raciocinio_entram_no_custo(self):
        """O Flash cobra o thinking separado, e sai do mesmo orcamento.

        Ignorar isso subestimaria o consumo na comparacao do M5 -- exatamente a
        medida que o eval existe para produzir.
        """
        r = gemini(body={
            "candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 10, "candidatesTokenCount": 20, "thoughtsTokenCount": 300,
            },
        }).complete("x")
        assert r.usage.output_tokens == 320

    def test_cota_negada_e_indisponibilidade_e_nao_erro_de_uso(self):
        with pytest.raises(LLMUnavailable, match="37"):
            gemini(status=429, headers={"retry-after": "37"}).complete("x")

    def test_instabilidade_do_provedor_e_indisponibilidade(self):
        with pytest.raises(LLMUnavailable):
            gemini(status=503).complete("x")

    def test_modelo_inexistente_e_erro_de_uso(self):
        """404 e id de modelo descontinuado -- repetir nao resolve, e a mensagem
        precisa chegar inteira para dar para corrigir o .env."""
        with pytest.raises(LLMError, match="not found") as exc:
            gemini(status=404, text="models/gemini-x not found").complete("x")
        assert not isinstance(exc.value, LLMUnavailable)

    def test_prompt_bloqueado_tem_tipo_proprio(self):
        with pytest.raises(LLMBlocked):
            gemini(body={"promptFeedback": {"blockReason": "SAFETY"}}).complete("x")

    def test_resposta_bloqueada_tem_tipo_proprio(self):
        with pytest.raises(LLMBlocked):
            gemini(body={"candidates": [{"finishReason": "SAFETY"}]}).complete("x")

    def test_candidato_sem_texto_diz_o_motivo(self):
        """Acontece com MAX_TOKENS depois do thinking: candidato sem parte de
        texto. Devolver string vazia faria o parser reclamar de outra coisa."""
        with pytest.raises(LLMError, match="MAX_TOKENS"):
            gemini(body={
                "candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}],
            }).complete("x")


class TestPayloadGroq:
    def test_schema_vai_no_system_porque_o_endpoint_nao_aceita(self):
        p = Groq.build_payload(
            "pergunta", system="instrucao", schema=SCHEMA, model="m",
            temperature=0.1, max_output_tokens=256,
        )
        system = p["messages"][0]
        assert system["role"] == "system"
        assert "instrucao" in system["content"]
        assert '"facts"' in system["content"]
        # O modo json do endpoint exige a palavra "json" no prompt.
        assert "json" in system["content"].lower()
        assert p["response_format"] == {"type": "json_object"}
        assert p["messages"][1] == {"role": "user", "content": "pergunta"}

    def test_sem_schema_nao_liga_o_modo_json(self):
        p = Groq.build_payload(
            "x", system="", schema=None, model="m", temperature=0.1, max_output_tokens=256,
        )
        assert "response_format" not in p
        assert [m["role"] for m in p["messages"]] == ["user"]


class TestRespostaGroq:
    def test_texto_e_tokens(self):
        r = groq(body={
            "model": "openai/gpt-oss-120b",
            "choices": [{"message": {"content": '{"facts": []}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40},
        }).complete("x", schema=SCHEMA)
        assert r.text == '{"facts": []}'
        assert r.provider == "groq"
        assert (r.usage.input_tokens, r.usage.output_tokens) == (900, 40)

    def test_cota_negada(self):
        with pytest.raises(LLMUnavailable):
            groq(status=429).complete("x")

    def test_filtro_de_conteudo(self):
        with pytest.raises(LLMBlocked):
            groq(body={
                "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
            }).complete("x")

    def test_truncamento_chega_ao_chamador(self):
        r = groq(body={
            "choices": [{"message": {"content": '{"facts": ['}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 512},
        }).complete("x")
        assert r.truncated


class TestConformidadeComAPorta:
    """Os dois adaptadores existem desde o M3 porque porta com um adaptador so
    e indirecao, nao porta. O teste guarda isso."""

    def test_ambos_satisfazem_a_porta(self):
        assert isinstance(gemini(), LLM)
        assert isinstance(groq(), LLM)
