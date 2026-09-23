"""Testes do juiz: a rubrica de 7 critérios e o que ela recusa.

Dois criterios nao sao perguntados ao modelo (duracao e politica) e um tem teto
medido (fonte). Os testes cobrem principalmente isso, porque e a parte que da
para afirmar sem depender de um leitor -- e porque e onde um erro deixaria passar
roteiro nao monetizavel ou com numero inventado.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.judge.judge import JULGADOS, Judge, build_prompt
from agent.models import (
    RUBRIC_CUTOFF,
    RUBRIC_MAX,
    Criterion,
    CriterionScore,
    Dossier,
    Fact,
    Review,
    Script,
)
from agent.ports.llm import LLMError

AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

# 180 palavras => 72s estimados, dentro da faixa de monetizacao.
CORPO = ("O modelo guarda cada peso como um de tres valores. " + "detalhe " * 150).strip()


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
        ],
        collected_at=AGORA,
    )


def roteiro(body: str = CORPO, hook: str = "Um modelo enorme agora cabe num pendrive.",
            closing: str = "Que numero voce deixaria de acreditar hoje?") -> Script:
    return Script(
        topic="Bonsai 2 27B em 5,9 GB",
        hook=hook,
        body=body,
        closing=closing,
        search_terms=["memory chip macro", "server rack lights", "binary code screen"],
        facts=dossie().facts,
    )


def parecer(**notas: int) -> str:
    """Parecer do modelo para os cinco critérios julgados; 2 em tudo por padrao."""
    return json.dumps({
        c.value: {
            "reason": f"motivo de exemplo para {c.value}",
            "score": notas.get(c.value, 2),
        }
        for c in JULGADOS
    })


def julgar(resposta: str, script: Script | None = None, dossier: Dossier | None = None):
    # A mesma resposta duas vezes: o juiz tem direito a uma segunda tentativa
    # quando a primeira vem malformada, e o teste precisa poder exercitar as duas.
    llm = ScriptedLLM(responses=[resposta, resposta])
    return Judge(llm).review(script or roteiro(), dossier or dossie()), llm


class TestRubricaCompleta:
    def test_parecer_tem_os_sete_criterios(self):
        report, _ = julgar(parecer())
        # A rubrica do video tem 7 criterios; `fluxo` e so do carrossel.
        assert set(report.review.by_criterion) == set(Criterion) - {Criterion.fluxo}
        assert report.review.total == RUBRIC_MAX

    def test_criterio_faltando_no_contrato_e_erro(self):
        """Parecer incompleto nao vira aprovacao por omissao."""
        with pytest.raises(ValueError, match="parecer incompleto"):
            Review(topic="t", scores=[CriterionScore(
                criterion=Criterion.hook, score=2, reason="hook abre bem")], reviewed_at=AGORA)

    def test_criterio_repetido_e_erro(self):
        notas = [CriterionScore(criterion=c, score=2, reason="nota cheia") for c in Criterion]
        notas.append(CriterionScore(criterion=Criterion.hook, score=0, reason="de novo"))
        with pytest.raises(ValueError, match="criterio repetido"):
            Review(topic="t", scores=notas, reviewed_at=AGORA)


class TestCriteriosMedidos:
    def test_duracao_sai_de_contagem_e_nao_do_modelo(self):
        """Perguntar a um LLM quantos segundos o texto leva falado seria trocar
        uma medida por um chute."""
        report, _ = julgar(parecer())
        nota = report.review.by_criterion[Criterion.duracao]
        assert nota.measured
        assert nota.score == 2
        assert "palavras" in nota.reason

    def test_roteiro_curto_demais_e_vetado(self):
        curto = roteiro(body="Corpo curto que nao chega perto de sessenta segundos falados.")
        report, _ = julgar(parecer(), script=curto)
        nota = report.review.by_criterion[Criterion.duracao]
        assert nota.score == 0
        assert "fora da faixa" in nota.reason
        assert not report.approved

    def test_politica_reusa_o_filtro_do_curador(self):
        """Mesma regra que barrou o tema barra o roteiro: duas listas divergiriam."""
        com_politica = roteiro(
            body=CORPO + " O senado discutiu o assunto na eleicao passada."
        )
        report, _ = julgar(parecer(), script=com_politica)
        nota = report.review.by_criterion[Criterion.politica]
        assert nota.score == 0
        assert nota.measured
        assert "politica/politica" in nota.reason
        assert not report.approved

    def test_narracao_limpa_passa_na_politica(self):
        report, _ = julgar(parecer())
        assert report.review.by_criterion[Criterion.politica].score == 2

    def test_modelo_nao_e_consultado_sobre_duracao_nem_politica(self):
        _, llm = julgar(parecer())
        pedidos = set(llm.calls[0].schema["properties"])
        assert "duracao" not in pedidos and "politica" not in pedidos
        assert pedidos == {c.value for c in JULGADOS}


class TestMedirAntesDeJulgar:
    """Medida barata primeiro, parecer pago depois -- a mesma ordem do curador."""

    def test_numero_inventado_reprova_sem_consultar_o_modelo(self):
        inventado = roteiro(body=CORPO + " O treinamento usou 12000 GPUs.")
        report, llm = julgar(parecer(), script=inventado)
        nota = report.review.by_criterion[Criterion.fonte]
        assert nota.score == 0
        assert nota.measured
        assert "12000" in nota.reason
        assert not report.approved
        assert llm.calls == []

    def test_reprovacao_medida_custa_zero_token(self):
        """Pagar um parecer para confirmar reprovacao ja decidida queimaria cota
        do free tier -- e, de graca, torna o caso demonstravel sem chave."""
        curto = roteiro(body="Corpo curto que nao chega a sessenta segundos falados.")
        report, llm = julgar(parecer(), script=curto)
        assert report.usage.total_tokens == 0
        assert report.latency_s == 0.0
        assert llm.calls == []

    def test_criterios_de_leitura_ficam_marcados_como_nao_avaliados(self):
        """Zero num critério nao avaliado significa "nao sei", nao "ruim"."""
        inventado = roteiro(body=CORPO + " O treinamento usou 12000 GPUs.")
        report, _ = julgar(parecer(), script=inventado)
        nao_avaliados = [s.criterion for s in report.review.scores if not s.evaluated]
        assert set(nao_avaliados) == {
            Criterion.hook, Criterion.ponto_de_vista, Criterion.pt_br, Criterion.cta,
        }
        assert report.review.short_circuited
        assert "reprovou antes em fonte" in report.review.by_criterion[Criterion.hook].reason

    def test_nota_nao_avaliada_nao_volta_como_correcao_ao_roteirista(self):
        """Mandar o roteirista "melhorar o hook" por causa de um zero que nunca
        foi julgado desperdicaria a revisao no lugar errado."""
        inventado = roteiro(body=CORPO + " O treinamento usou 12000 GPUs.")
        report, _ = julgar(parecer(), script=inventado)
        assert report.review.revision_notes == ["[fonte 0/2] " + (
            report.review.by_criterion[Criterion.fonte].reason
        )]

    def test_numero_do_dossie_deixa_o_criterio_para_o_modelo(self):
        """Quando os numeros conferem, fonte volta a ser leitura: a medida nao
        tem como julgar afirmacao que vai alem do dossie."""
        ancorado = roteiro(body=CORPO + " Ele ocupa 5,9 GB e retem 98,2% da nota.")
        report, llm = julgar(parecer(), script=ancorado)
        nota = report.review.by_criterion[Criterion.fonte]
        assert nota.score == 2
        assert not nota.measured
        assert len(llm.calls) == 1

    def test_modelo_ainda_pode_reprovar_a_fonte_que_a_medida_aprovou(self):
        report, _ = julgar(parecer(fonte=0))
        nota = report.review.by_criterion[Criterion.fonte]
        assert nota.score == 0
        assert not nota.measured
        assert not report.approved


class TestRegraDeAprovacao:
    def test_corte_no_limite_aprova(self):
        """11/14 com nenhum zero e nenhum veto."""
        report, _ = julgar(parecer(hook=1, ponto_de_vista=1, pt_br=1))
        assert report.review.total == RUBRIC_CUTOFF
        assert report.approved

    def test_abaixo_do_corte_reprova(self):
        report, _ = julgar(parecer(hook=1, ponto_de_vista=1, pt_br=1, cta=1))
        assert report.review.total == RUBRIC_CUTOFF - 1
        assert not report.approved

    def test_criterio_zerado_reprova_mesmo_com_a_soma_no_corte(self):
        """Resumo de noticia com o resto perfeito chegaria a 12 de 14 e passaria.

        E exatamente o "AI slop" que o Creator Rewards exclui, entao a soma
        sozinha nao decide.
        """
        report, _ = julgar(parecer(ponto_de_vista=0))
        assert report.review.total == 12
        assert report.review.total >= RUBRIC_CUTOFF
        assert not report.approved
        assert report.review.zeroed[0].criterion is Criterion.ponto_de_vista

    def test_veto_em_fonte_reprova_mesmo_com_nota_alta(self):
        """Meia fonte e fonte faltando: 1/2 em fonte ja e veto."""
        report, _ = julgar(parecer(fonte=1))
        assert report.review.total == 13
        assert not report.approved
        assert [s.criterion for s in report.review.vetoed] == [Criterion.fonte]

    def test_qualidade_fraca_compensavel_nao_e_veto(self):
        """O corte existe para isso: hook fraco pode ser compensado, fonte nao."""
        report, _ = julgar(parecer(hook=1, cta=1))
        assert report.approved


class TestNotasDeRevisao:
    def test_veto_vem_antes_do_resto(self):
        """Nao adianta melhorar o hook de um roteiro que cita numero sem fonte."""
        report, _ = julgar(parecer(fonte=1, hook=0))
        notas = report.review.revision_notes
        assert notas[0].startswith("[fonte 1/2]")
        assert any(n.startswith("[hook 0/2]") for n in notas)

    def test_criterio_com_nota_maxima_nao_entra_na_revisao(self):
        report, _ = julgar(parecer(hook=1))
        assert len(report.review.revision_notes) == 1
        assert "hook" in report.review.revision_notes[0]

    def test_roteiro_perfeito_nao_tem_nota_de_revisao(self):
        report, _ = julgar(parecer())
        assert report.review.revision_notes == []


class TestRespostaDefeituosa:
    @pytest.mark.parametrize("nota", [5, -1, "dois", None, True])
    def test_nota_fora_da_escala_nao_e_arredondada(self, nota):
        """Aceitar 5 seria deixar o modelo redefinir o corte da rubrica."""
        bruto = json.loads(parecer())
        bruto["hook"]["score"] = nota
        with pytest.raises(LLMError, match="nota invalida"):
            julgar(json.dumps(bruto))

    def test_criterio_ausente_na_resposta_levanta(self):
        bruto = json.loads(parecer())
        del bruto["cta"]
        with pytest.raises(LLMError, match="cta"):
            julgar(json.dumps(bruto))

    def test_resposta_malformada_ganha_uma_segunda_chance(self):
        """Medido em 18/09/2026: o Flash truncou o JSON do parecer no meio, e o
        roteiro ja escrito se perdia por causa disso. Uma segunda chamada custa
        menos que refazer o roteiro inteiro na proxima execucao."""
        llm = ScriptedLLM(responses=['{"hook": {"reason": "cortou aqui', parecer()])
        report = Judge(llm).review(roteiro(), dossie())
        assert report.approved
        assert len(llm.calls) == 2

    def test_custo_das_duas_tentativas_entra_no_relatorio(self):
        llm = ScriptedLLM(responses=["nao e json", parecer()])
        report = Judge(llm).review(roteiro(), dossie())
        # As duas chamadas foram cobradas; contar so a que funcionou
        # subestimaria o custo do parecer no eval do M5.
        assert report.usage.output_tokens > len(parecer().split())

    def test_erro_final_diz_quanto_foi_gasto(self):
        """Sob restricao de $0, saber o custo de uma execucao que nao entregou
        nada e parte do resultado."""
        llm = ScriptedLLM(responses=["nao e json", "tambem nao e"])
        with pytest.raises(LLMError, match="gastos [0-9]+ tokens"):
            Judge(llm).review(roteiro(), dossie())

    def test_motivo_vazio_nao_quebra_o_contrato(self):
        """`reason` e obrigatorio no contrato; parecer sem motivo ganha um texto
        que diz isso, em vez de derrubar o julgamento inteiro."""
        bruto = json.loads(parecer())
        bruto["hook"]["reason"] = ""
        report, _ = julgar(json.dumps(bruto))
        assert "nao justificou" in report.review.by_criterion[Criterion.hook].reason


class TestPrompt:
    def test_dossie_e_roteiro_vao_no_prompt(self):
        prompt = build_prompt(roteiro(), dossie())
        assert "[0] O Bonsai 2 27B ocupa 5,9 GB" in prompt
        assert "HOOK:" in prompt and "FECHAMENTO:" in prompt

    def test_pede_a_razao_antes_da_nota(self):
        """A ordem muda o julgamento: nota primeiro faz o modelo justificar o que
        ja decidiu."""
        prompt = build_prompt(roteiro(), dossie())
        assert "razao antes da nota" in prompt

    def test_avisa_que_duracao_e_politica_sao_medidas_fora(self):
        prompt = build_prompt(roteiro(), dossie())
        assert "Nao avalie duracao nem politica" in prompt

    def test_custo_do_parecer_e_medido(self):
        report, _ = julgar(parecer())
        assert report.usage.input_tokens > 0
        assert report.latency_s >= 0
        assert report.review.model == "scripted-1"
