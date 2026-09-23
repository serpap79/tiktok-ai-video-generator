"""Aceite do M3, como o plano o definiu.

  "roteiro aprovado com 100% das afirmacoes rastreaveis a uma URL; fixture
   adversarial com afirmacao sem fonte e reprovado pelo juiz"

As duas fixtures sao o mesmo roteiro, e a diferenca entre elas e uma unica
afirmacao inventada. E o que torna o teste conclusivo: nao ha como o roteiro
adversarial ser reprovado por escrita ruim, duracao ou politica, porque nesses
tres ele e identico ao aprovado.

O teste nao depende de nenhum modelo real. O juiz recebe um parecer de nota
maxima nos cinco criterios julgados -- ou seja, o cenario mais favoravel
possivel ao roteiro adversarial. Ele e reprovado mesmo assim, porque o criterio
de fonte tem teto medido.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.judge.judge import JULGADOS, Judge
from agent.models import RUBRIC_CUTOFF, Criterion, Dossier, Script

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def carregar(nome: str) -> Script:
    bruto = json.loads((FIXTURES / nome).read_text(encoding="utf-8"))
    bruto.pop("_comment", None)
    return Script.model_validate(bruto)


def dossie_do_roteiro(script: Script) -> Dossier:
    """O Script carrega os fatos que o roteirista usou, entao ele se autojulga."""
    return Dossier(topic=script.topic, facts=script.facts, collected_at=datetime.now(UTC))


def parecer_nota_maxima() -> str:
    return json.dumps({
        c.value: {"reason": f"sem ressalva em {c.value}", "score": 2} for c in JULGADOS
    })


@pytest.fixture
def referencia() -> Script:
    return carregar("roteiro_manual.json")


@pytest.fixture
def adversarial() -> Script:
    return carregar("roteiro_sem_fonte.json")


class TestRoteiroAprovado:
    def test_cem_por_cento_das_afirmacoes_tem_url(self, referencia):
        assert referencia.facts
        assert all(str(f.source_url).startswith("http") for f in referencia.facts)
        assert all(f.source_name for f in referencia.facts)

    def test_passa_nos_tres_criterios_medidos_sem_llm_nenhum(self, referencia):
        """Duracao, politica e ancoragem numerica sao medidas nossas: o roteiro
        de referencia passa nelas sem que nenhum modelo opine."""
        report = Judge(ScriptedLLM(responses=[parecer_nota_maxima()])).review(
            referencia, dossie_do_roteiro(referencia)
        )
        medidos = [s for s in report.review.scores if s.measured]
        assert {s.criterion for s in medidos} == {Criterion.duracao, Criterion.politica}
        assert all(s.score == 2 for s in medidos)

    def test_e_aprovado(self, referencia):
        report = Judge(ScriptedLLM(responses=[parecer_nota_maxima()])).review(
            referencia, dossie_do_roteiro(referencia)
        )
        assert report.approved
        assert report.review.total >= RUBRIC_CUTOFF


class TestFixtureAdversarial:
    def test_e_reprovado_apesar_do_parecer_de_nota_maxima(self, adversarial):
        report = Judge(ScriptedLLM(responses=[parecer_nota_maxima()])).review(
            adversarial, dossie_do_roteiro(adversarial)
        )
        assert not report.approved

    def test_reprova_por_fonte_e_nao_por_outro_criterio(self, adversarial):
        """A afirmacao inventada precisa aparecer no criterio certo. Reprovar
        pelo motivo errado esconderia o defeito que se quer pegar."""
        report = Judge(ScriptedLLM(responses=[parecer_nota_maxima()])).review(
            adversarial, dossie_do_roteiro(adversarial)
        )
        assert [s.criterion for s in report.review.vetoed] == [Criterion.fonte]
        nota = report.review.by_criterion[Criterion.fonte]
        assert nota.score == 0
        assert "12" in nota.reason and "40" in nota.reason

    def test_a_diferenca_com_o_aprovado_e_so_a_afirmacao_inventada(
        self, referencia, adversarial
    ):
        assert adversarial.hook == referencia.hook
        assert adversarial.closing == referencia.closing
        assert adversarial.search_terms == referencia.search_terms
        assert [f.claim for f in adversarial.facts] == [f.claim for f in referencia.facts]
        assert adversarial.body != referencia.body
        # Duracao e politica seguem iguais: nao ha outro motivo de reprovacao.
        assert 60 <= adversarial.estimated_duration_s <= 90

    def test_nota_de_revisao_aponta_a_fonte_primeiro(self, adversarial):
        """E o que volta ao roteirista: nao adianta mexer no hook."""
        report = Judge(ScriptedLLM(responses=[parecer_nota_maxima()])).review(
            adversarial, dossie_do_roteiro(adversarial)
        )
        assert report.review.revision_notes[0].startswith("[fonte 0/2]")
