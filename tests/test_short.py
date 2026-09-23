"""Modo short (~15s, alcance): faixa propria, loop no fechamento, sem polish quebrado."""

from __future__ import annotations

import json

from agent.adapters.scripted_llm import ScriptedLLM
from agent.judge.judge import _notas_medidas
from agent.models import SHORT_MAX_DURATION_S, SHORT_MIN_DURATION_S, Script
from agent.writer.writer import BANDS, Screenwriter, build_prompt
from tests.test_writer import dossie

# Tres termos: o curto de ~25s pede de 3 a 5 (um clipe de b-roll cobre ~5s).
TERMOS_SHORT = ["neural network nodes", "data stream tunnel",
                "abstract digital plexus"]


def resposta_short(total: int = 64) -> str:
    hook = "Um modelo de 27 bilhões cabe em cinco vírgula nove gigabytes?"
    closing = "Gigante no bolso: sera que cabe?"
    sujeito = "O Bonsai 27B prova. "
    fixos = len((hook + " " + sujeito + closing).split())
    return json.dumps({
        "hook": hook,
        "body": sujeito + " ".join(["detalhe"] * max(total - fixos, 1)),
        "closing": closing,
        "search_terms": TERMOS_SHORT,
        "caption": "Gigante no bolso.\nO Bonsai 27B cabe num pendrive.",
        "used_facts": [0],
    })


class TestShort:
    def test_faixa_curta_aprova_o_tamanho_de_25s(self):
        """58-70 palavras = 23-27s, a faixa da grade de 20/09/2026.

        Era 30-50 (~15s) ate a grade subir para quatro posts com o curto
        pedindo 25s.
        """
        report = Screenwriter(ScriptedLLM(responses=[resposta_short()])).write(
            dossie(), mode="short", polish=False)
        assert report.ok
        assert report.script is not None and report.script.format == "short"
        assert 58 <= report.script.word_count <= 70
        assert 23 <= report.script.estimated_duration_s <= 28

    def test_tamanho_do_curto_antigo_agora_reprova(self):
        """40 palavras (~15s) ficaram curtas demais para a grade nova."""
        report = Screenwriter(ScriptedLLM(responses=[resposta_short(40)] * 3)).write(
            dossie(), mode="short", polish=False)
        assert not report.ok

    def test_180_palavras_reprova_no_modo_short(self):
        from tests.test_writer import resposta as resposta_long
        report = Screenwriter(ScriptedLLM(responses=[resposta_long()] * 3)).write(
            dossie(), mode="short", polish=False)
        assert not report.ok

    def test_termos_de_longo_sao_muitos_para_short(self):
        report = Screenwriter(ScriptedLLM(responses=[
            resposta_short().replace('"data stream tunnel"',
                                     '"a", "b", "c", "d", "e", "f", "g"'),
            resposta_short()])).write(dossie(), mode="short", polish=False)
        assert report.ok and len(report.attempts) == 2

    def test_prompt_pede_loop_e_faixa(self):
        texto = build_prompt(dossie(), None, "short")
        minimo, maximo = BANDS["short"]
        assert "reconecta" in texto
        assert str(minimo) in texto and str(maximo) in texto
        assert "25s" in texto      # o exemplo de tamanho, nao mais 15s
        assert "implicacao" in texto

    def test_modo_desconhecido_falha_cedo(self):
        import pytest
        with pytest.raises(ValueError):
            Screenwriter(ScriptedLLM(responses=[])).write(dossie(), mode="tv")

    def test_duracao_medida_por_formato(self):
        """A faixa do curto vem de `models`, nao de um numero cravado no juiz.

        Era `(10, 20)` escrito dentro do juiz e `(30, 50)` palavras dentro do
        roteirista: dois lugares para o mesmo fato. Quando a grade pediu 25s,
        so um foi atualizado e o juiz passou a reprovar todo curto que o
        roteirista aprovava -- com a mensagem generica "nenhum formato
        aprovado pelo juiz", que nao aponta para lugar nenhum.
        """
        curta = Script(topic="Tema curto de teste", hook="h " * 7, body="b " * 50,
                       closing="c " * 7,
                       search_terms=["neural network nodes", "data stream tunnel",
                                     "abstract digital plexus"],
                       format="short")
        (duracao, _) = _notas_medidas(curta)
        assert duracao.score == 2, duracao.reason
        assert f"{SHORT_MIN_DURATION_S}-{SHORT_MAX_DURATION_S}s" in duracao.reason
