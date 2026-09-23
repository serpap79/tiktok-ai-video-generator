"""Testes da porta LLM: o que todo adaptador precisa respeitar."""

from __future__ import annotations

import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.ports.llm import LLM, Completion, LLMError, Usage, parse_json_object


class TestParseJsonObject:
    """Modelo de free tier embrulha JSON de varias formas. Todas caem aqui."""

    def test_objeto_limpo(self):
        assert parse_json_object('{"facts": []}') == {"facts": []}

    @pytest.mark.parametrize("bruto", [
        '```json\n{"ok": 1}\n```',
        '```\n{"ok": 1}\n```',
        'Aqui esta o resultado:\n{"ok": 1}',
        '  \n {"ok": 1} \n\n',
        '{"ok": 1}\nEspero ter ajudado!',
    ])
    def test_embrulho_comum_e_tolerado(self, bruto):
        assert parse_json_object(bruto) == {"ok": 1}

    def test_resposta_sem_objeto_levanta(self):
        with pytest.raises(LLMError, match="sem objeto JSON"):
            parse_json_object("Nao posso ajudar com isso.")

    def test_lista_no_topo_nao_vira_o_primeiro_elemento(self):
        """Pedimos objeto; lista solta significa que o modelo ignorou o schema.

        A tentacao e pegar o primeiro elemento e seguir -- e isso devolveria um
        dossie com um fato so, sem nenhum aviso de que os outros foram perdidos.
        """
        with pytest.raises(LLMError, match="veio uma lista"):
            parse_json_object('[{"claim": "x"}, {"claim": "y"}]')

    def test_json_truncado_nao_e_remendado(self):
        """Resposta cortada e erro de orcamento de token.

        Consertar com regex produziria um dossie com metade dos fatos e nenhuma
        pista do motivo -- que e exatamente o tipo de falha silenciosa que este
        projeto evita.
        """
        with pytest.raises(LLMError, match="truncada"):
            parse_json_object('{"facts": [{"claim": "o modelo ocupa 5,9 GB"')

    def test_json_corrompido_no_meio_tambem_nao_e_remendado(self):
        with pytest.raises(LLMError, match="JSON invalido"):
            parse_json_object('{"facts": [{"claim": 5,9 GB"}]}')


class TestUsage:
    def test_soma_acumula_os_dois_lados(self):
        total = Usage(input_tokens=10, output_tokens=5) + Usage(input_tokens=1, output_tokens=2)
        assert (total.input_tokens, total.output_tokens, total.total_tokens) == (11, 7, 18)

    def test_zero_e_desconhecido_e_nao_mente_sobre_custo(self):
        assert Usage().total_tokens == 0


class TestCompletion:
    @pytest.mark.parametrize("motivo,truncado", [
        ("stop", False), ("length", True), ("MAX_TOKENS", True), ("", False),
    ])
    def test_truncamento_e_detectado_nos_dois_dialetos(self, motivo, truncado):
        c = Completion(text="x", model="m", provider="p", finish_reason=motivo)
        assert c.truncated is truncado


class TestScriptedLLM:
    def test_satisfaz_a_porta(self):
        assert isinstance(ScriptedLLM(responses=["{}"]), LLM)

    def test_devolve_na_ordem_e_registra_a_chamada(self):
        llm = ScriptedLLM(responses=['{"n": 1}', '{"n": 2}'])
        assert llm.complete("a", system="s").text == '{"n": 1}'
        assert llm.complete("b").text == '{"n": 2}'
        assert [c.prompt for c in llm.calls] == ["a", "b"]
        assert llm.calls[0].system == "s"

    def test_chamada_extra_falha_alto(self):
        """Chamada alem do previsto e erro de teste, nao resposta vazia."""
        llm = ScriptedLLM(responses=["{}"])
        llm.complete("a")
        with pytest.raises(LLMError, match="sem resposta restante"):
            llm.complete("b")
