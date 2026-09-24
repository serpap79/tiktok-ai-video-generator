"""Aceite del M3, tal como el plan lo definio.

  "guion aprobado con 100% de las afirmaciones rastreables a una URL; fixture
   adversarial con afirmacion sin fuente y reprobada por el juez"

Las dos fixtures son el mismo guion, y la diferencia entre ellas es una unica
afirmacion inventada. Es lo que hace el test concluyente: no hay manera de que
el guion adversarial sea reprobado por escrita mala, duracion o politica, porque
en esos tres es identico al aprobado.

El test no depende de ningun modelo real. El juez recibe un informe de nota
maxima en los cinco criterios juzgados -- o sea, el escenario mas favorable
posible al guion adversarial. Es reprobado aun asi, porque el criterio de fuente
tiene techo medido.
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


def cargar(nombre: str) -> Script:
    bruto = json.loads((FIXTURES / nombre).read_text(encoding="utf-8"))
    bruto.pop("_comment", None)
    return Script.model_validate(bruto)


def dossier_del_guion(script: Script) -> Dossier:
    """El Script lleva los hechos que el guionista uso, entonces se autojuzga."""
    return Dossier(topic=script.topic, facts=script.facts, collected_at=datetime.now(UTC))


def informe_nota_maxima() -> str:
    return json.dumps({
        c.value: {"reason": f"sin salvedad en {c.value}", "score": 2} for c in JULGADOS
    })


@pytest.fixture
def referencia() -> Script:
    return cargar("guion_manual.json")


@pytest.fixture
def adversarial() -> Script:
    return cargar("guion_sin_fuente.json")


class TestGuionAprobado:
    def test_cien_por_cien_de_las_afirmaciones_tienen_url(self, referencia):
        assert referencia.facts
        assert all(str(f.source_url).startswith("http") for f in referencia.facts)
        assert all(f.source_name for f in referencia.facts)

    def test_pasa_en_los_tres_criterios_medidos_sin_llm_ninguno(self, referencia):
        """Duracion, politica y anclaje numerico son medidas nuestras: el guion
        de referencia pasa en ellas sin que ningun modelo opine."""
        report = Judge(ScriptedLLM(responses=[informe_nota_maxima()])).review(
            referencia, dossier_del_guion(referencia)
        )
        medidos = [s for s in report.review.scores if s.measured]
        assert {s.criterion for s in medidos} == {Criterion.duracion, Criterion.politica}
        assert all(s.score == 2 for s in medidos)

    def test_es_aprobado(self, referencia):
        report = Judge(ScriptedLLM(responses=[informe_nota_maxima()])).review(
            referencia, dossier_del_guion(referencia)
        )
        assert report.approved
        assert report.review.total >= RUBRIC_CUTOFF


class TestFixtureAdversarial:
    def test_es_reprobado_apesar_del_informe_de_nota_maxima(self, adversarial):
        report = Judge(ScriptedLLM(responses=[informe_nota_maxima()])).review(
            adversarial, dossier_del_guion(adversarial)
        )
        assert not report.approved

    def test_reprueba_por_fuente_y_no_por_otro_criterio(self, adversarial):
        """La afirmacion inventada necesita aparecer en el criterio correcto.
        Reprobar por el motivo equivocado esconderia el defecto que se quiere cazar."""
        report = Judge(ScriptedLLM(responses=[informe_nota_maxima()])).review(
            adversarial, dossier_del_guion(adversarial)
        )
        assert [s.criterion for s in report.review.vetoed] == [Criterion.fuente]
        nota = report.review.by_criterion[Criterion.fuente]
        assert nota.score == 0
        assert "12" in nota.reason and "40" in nota.reason

    def test_la_diferencia_con_el_aprobado_es_solo_la_afirmacion_inventada(
        self, referencia, adversarial
    ):
        assert adversarial.hook == referencia.hook
        assert adversarial.closing == referencia.closing
        assert adversarial.search_terms == referencia.search_terms
        assert [f.claim for f in adversarial.facts] == [f.claim for f in referencia.facts]
        assert adversarial.body != referencia.body
        # Duracion y politica siguen iguales: no hay otro motivo de reprobacion.
        assert 60 <= adversarial.estimated_duration_s <= 90

    def test_nota_de_revision_apunta_a_la_fuente_primero(self, adversarial):
        """Es lo que vuelve al guionista: no sirve de nada tocar el hook."""
        report = Judge(ScriptedLLM(responses=[informe_nota_maxima()])).review(
            adversarial, dossier_del_guion(adversarial)
        )
        assert report.review.revision_notes[0].startswith("[fuente 0/2]")
