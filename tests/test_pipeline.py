"""Testes do laco roteirista <-> juiz."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from agent.adapters.scripted_llm import ScriptedLLM
from agent.judge.judge import JULGADOS, Judge
from agent.models import Dossier, Fact
from agent.pipeline import produce
from agent.ports.llm import LLMUnavailable
from agent.writer.writer import Screenwriter

AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

HOOK = "Um modelo gigante agora cabe num pendrive."
FECHAMENTO = "Que numero de anuncio voce vai conferir na fonte hoje?"


def dossie() -> Dossier:
    return Dossier(
        topic="Bonsai 2 27B em 5,9 GB",
        facts=[
            Fact(claim="O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor",
                 source_url="https://prismml.com/news/bonsai-2-27b", source_name="PrismML",
                 quote="that occupies 5.9 GB on disk, more than 9x smaller"),
            Fact(claim="Retem 98,2% do desempenho nos benchmarks",
                 source_url="https://prismml.com/news/bonsai-2-27b", source_name="PrismML",
                 quote="the model retains 98.2% of the original score"),
            Fact(claim="Roda a 143 tokens por segundo numa RTX 5090",
                 source_url="https://cienciahoje.com.br/ia/bonsai",
                 source_name="Ciencia Hoje",
                 quote="Throughput reaches 143 tokens per second on a single RTX 5090"),
        ],
        collected_at=AGORA,
    )


def roteiro_json(total: int = 187) -> str:
    fixos = len((HOOK + " " + FECHAMENTO).split())
    corpo = "O Bonsai 27B e um modelo. " + " ".join(["detalhe"] * max(total - fixos - 6, 1))
    return json.dumps({
        "hook": HOOK,
        "body": corpo,
        "closing": FECHAMENTO,
        "search_terms": ["neural network nodes", "ai deep learning loop",
                         "data stream tunnel", "abstract digital plexus"],
        "caption": "Modelo gigante, disco pequeno.\nO Bonsai 27B mostra o que a compressao ja faz.",
        "used_facts": [0, 1],
    }, ensure_ascii=False)


def parecer_json(**notas: int) -> str:
    return json.dumps({
        c.value: {"reason": f"motivo para {c.value}", "score": notas.get(c.value, 2)}
        for c in JULGADOS
    })


def rodar(respostas: list[str], **kwargs):
    """Um unico modelo atende roteirista e juiz, na ordem em que sao chamados."""
    llm = ScriptedLLM(responses=respostas)
    return produce(dossie(), Screenwriter(llm), Judge(llm), **kwargs), llm


class TestCaminhoAprovado:
    def test_aprovado_na_primeira_rodada(self):
        report, llm = rodar([roteiro_json(), parecer_json()])
        assert report.approved
        assert len(report.rounds) == 1
        assert len(llm.calls) == 2  # um roteiro, um parecer
        assert report.script.word_count == 187
        assert report.review.approved

    def test_custo_soma_roteirista_e_juiz(self):
        report, _ = rodar([roteiro_json(), parecer_json()])
        rodada = report.rounds[0]
        assert report.usage.input_tokens == (
            rodada.write.usage.input_tokens + rodada.review.usage.input_tokens
        )


class TestRevisao:
    def test_reprovado_volta_com_as_notas_do_juiz_no_prompt(self):
        """A nota do juiz entra no mesmo canal das violacoes mecanicas: o modelo
        recebe defeitos a corrigir e nao precisa saber qual foi contado."""
        report, llm = rodar([
            roteiro_json(), parecer_json(hook=0),
            roteiro_json(), parecer_json(),
        ])
        assert report.approved
        assert len(report.rounds) == 2
        # A segunda chamada de roteiro (3a do modelo) traz a nota do juiz.
        prompt_revisao = llm.calls[2].prompt
        assert "CORRIJA A TENTATIVA ANTERIOR" in prompt_revisao
        assert "[hook 0/2]" in prompt_revisao
        assert report.rounds[1].notes_in

    def test_teto_de_revisoes_para_o_laco(self):
        """Depois da segunda revisao o modelo costuma trocar de assunto para
        agradar a rubrica, em vez de melhorar o roteiro."""
        respostas = [roteiro_json(), parecer_json(hook=0)] * 3
        report, llm = rodar(respostas, max_revisions=2)
        assert not report.approved
        assert len(report.rounds) == 3
        assert len(llm.calls) == 6

    def test_reprovado_mantem_o_ultimo_roteiro_e_parecer(self):
        """Sem isso nao daria para mostrar por que reprovou."""
        report, _ = rodar([roteiro_json(), parecer_json(fonte=1)], max_revisions=0)
        assert not report.approved
        assert report.script is not None
        assert report.review is not None
        assert report.review.vetoed

    def test_revisao_zerada_julga_uma_vez_so(self):
        report, llm = rodar([roteiro_json(), parecer_json(hook=0)], max_revisions=0)
        assert len(report.rounds) == 1
        assert len(llm.calls) == 2


class TestFalhaAntesDoJuiz:
    def test_roteiro_que_nao_passa_no_mecanico_nao_vai_ao_juiz(self):
        """Pagar por um parecer sobre texto que ja se sabe fora da faixa de
        duracao seria queimar cota."""
        report, llm = rodar([roteiro_json(total=90)] * 3)
        assert not report.approved
        assert len(report.rounds) == 1
        assert report.rounds[0].review is None
        assert len(llm.calls) == 3  # tres tentativas do roteirista, zero pareceres

    def test_cota_estourada_encerra_o_laco_dizendo_o_estagio(self):
        """Falha de provedor e dado registrado, nao excecao perdida -- a mesma
        regra do radar. O estagio entra no texto porque "falha do provedor" sem
        dizer onde nao ajuda a decidir o que fazer."""
        def responder(prompt: str) -> str:
            raise LLMUnavailable("cota diaria estourada (429)")

        llm = ScriptedLLM(responder=responder)
        report = produce(dossie(), Screenwriter(llm), Judge(llm))
        assert not report.approved
        assert report.failure.startswith("roteirista: LLMUnavailable")
        assert len(report.rounds) == 1

    def test_falha_do_juiz_e_registrada_com_o_roteiro_preservado(self):
        """O roteiro ja escrito nao se perde: ele custou tokens e serve para a
        proxima execucao nao comecar do zero."""
        llm = ScriptedLLM(responses=[roteiro_json(), "nao e json", "tambem nao"])
        report = produce(dossie(), Screenwriter(llm), Judge(llm))
        assert not report.approved
        assert report.failure.startswith("juiz: LLMError")
        assert report.script is not None
        assert report.review is None
