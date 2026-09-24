"""Tests del juez: la rubrica de 7 criterios y lo que rechaza.

Dos criterios no se preguntan al modelo (duracion y politica) y uno tiene techo
medido (fuente). Los tests cubren sobre todo eso, porque es la parte que se
puede afirmar sin depender de un lector -- y porque es donde un error dejaria
pasar guion no monetizable o con numero inventado.
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

AHORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

# 180 palabras => ~72s estimados, dentro de la franja de monetizacion.
CUERPO = ("El modelo guarda cada peso en uno de tres valores. " + "detalle " * 150).strip()


def dossier() -> Dossier:
    return Dossier(
        topic="Bonsai 2 27B en 5,9 GB",
        facts=[
            Fact(claim="El Bonsai 2 27B ocupa 5,9 GB, mas de 9 veces menor",
                 source_url="https://prismml.com/news/bonsai-2-27b", source_name="PrismML",
                 quote="that occupies 5.9 GB on disk, more than 9x smaller"),
            Fact(claim="Retiene 98,2% del rendimiento en los benchmarks",
                 source_url="https://prismml.com/news/bonsai-2-27b", source_name="PrismML",
                 quote="the model retains 98.2% of the original score"),
        ],
        collected_at=AHORA,
    )


def guion(body: str = CUERPO, hook: str = "Un modelo enorme cabe ahora en un pendrive.",
          closing: str = "¿Que numero dejaras de creer hoy?") -> Script:
    return Script(
        topic="Bonsai 2 27B en 5,9 GB",
        hook=hook,
        body=body,
        closing=closing,
        search_terms=["memory chip macro", "server rack lights", "binary code screen"],
        facts=dossier().facts,
    )


def informe(**notas: int) -> str:
    """Informe del modelo para los cinco criterios juzgados; 2 en todo por defecto."""
    return json.dumps({
        c.value: {
            "reason": f"motivo de ejemplo para {c.value}",
            "score": notas.get(c.value, 2),
        }
        for c in JULGADOS
    })


def juzgar(respuesta: str, script: Script | None = None, dossier_param: Dossier | None = None):
    # La misma respuesta dos veces: el juez tiene derecho a un segundo intento
    # cuando el primero viene malformado, y el test necesita poder ejercitar ambos.
    llm = ScriptedLLM(responses=[respuesta, respuesta])
    return Judge(llm).review(script or guion(), dossier_param or dossier()), llm


class TestRubricaCompleta:
    def test_informe_tiene_los_siete_criterios(self):
        informe_r, _ = juzgar(informe())
        # La rubrica del video tiene 7 criterios; `flujo` es solo del carrusel.
        assert set(informe_r.review.by_criterion) == set(Criterion) - {Criterion.flujo}
        assert informe_r.review.total == RUBRIC_MAX

    def test_criterio_faltante_en_el_contrato_es_error(self):
        """Informe incompleto no se convierte en aprobacion por omision."""
        with pytest.raises(ValueError, match="informe incompleto"):
            Review(topic="t", scores=[CriterionScore(
                criterion=Criterion.hook, score=2, reason="hook abre bien")], reviewed_at=AHORA)

    def test_criterio_repetido_es_error(self):
        notas = [CriterionScore(criterion=c, score=2, reason="nota llena") for c in Criterion]
        notas.append(CriterionScore(criterion=Criterion.hook, score=0, reason="de nuevo"))
        with pytest.raises(ValueError, match="criterio repetido"):
            Review(topic="t", scores=notas, reviewed_at=AHORA)


class TestCriteriosMedidos:
    def test_duracion_sale_del_conteo_y_no_del_modelo(self):
        """Preguntarle a un LLM cuantos segundos dura el texto hablado seria
        cambiar una medida por un chute."""
        informe_r, _ = juzgar(informe())
        nota = informe_r.review.by_criterion[Criterion.duracion]
        assert nota.measured
        assert nota.score == 2
        assert "palabras" in nota.reason

    def test_guion_demasiado_corto_es_vetado(self):
        corto = guion(body="Cuerpo corto que no llega ni de lejos a sesenta segundos hablados.")
        informe_r, _ = juzgar(informe(), script=corto)
        nota = informe_r.review.by_criterion[Criterion.duracion]
        assert nota.score == 0
        assert "fuera de la franja" in nota.reason
        assert not informe_r.approved

    def test_politica_reutiliza_el_filtro_del_curador(self):
        """La misma regla que veto el tema veto el guion: dos listas divergirian."""
        con_politica = guion(
            body=CUERPO + " El senado discutio el asunto en la eleccion pasada."
        )
        informe_r, _ = juzgar(informe(), script=con_politica)
        nota = informe_r.review.by_criterion[Criterion.politica]
        assert nota.score == 0
        assert nota.measured
        assert "politica/politica" in nota.reason
        assert not informe_r.approved

    def test_narracion_limpia_pasa_la_politica(self):
        informe_r, _ = juzgar(informe())
        assert informe_r.review.by_criterion[Criterion.politica].score == 2

    def test_modelo_no_es_consultado_sobre_duracion_ni_politica(self):
        _, llm = juzgar(informe())
        pedidos = set(llm.calls[0].schema["properties"])
        assert "duracion" not in pedidos and "politica" not in pedidos
        assert pedidos == {c.value for c in JULGADOS}


class TestMedirAntesDeJuzgar:
    """Medida barata primero, informe pago despues -- el mismo orden del curador."""

    def test_numero_inventado_reprueba_sin_consultar_al_modelo(self):
        inventado = guion(body=CUERPO + " El entrenamiento uso 12000 GPUs.")
        informe_r, llm = juzgar(informe(), script=inventado)
        nota = informe_r.review.by_criterion[Criterion.fuente]
        assert nota.score == 0
        assert nota.measured
        assert "12000" in nota.reason
        assert not informe_r.approved
        assert llm.calls == []

    def test_reprobacion_medida_cuesta_cero_tokens(self):
        """Pagar un informe para confirmar una reprobacion ya decidida quemaria
        cuota del free tier -- y, de gratis, vuelve el caso demostrable sin clave."""
        corto = guion(body="Cuerpo corto que no llega a sesenta segundos hablados.")
        informe_r, llm = juzgar(informe(), script=corto)
        assert informe_r.usage.total_tokens == 0
        assert informe_r.latency_s == 0.0
        assert llm.calls == []

    def test_criterios_de_lectura_quedan_marcados_como_no_evaluados(self):
        """Cero en un criterio no evaluado significa "no se", no "malo"."""
        inventado = guion(body=CUERPO + " El entrenamiento uso 12000 GPUs.")
        informe_r, _ = juzgar(informe(), script=inventado)
        no_evaluados = [s.criterion for s in informe_r.review.scores if not s.evaluated]
        assert set(no_evaluados) == {
            Criterion.hook, Criterion.punto_de_vista, Criterion.es_es, Criterion.cta,
        }
        assert informe_r.review.short_circuited
        assert "reprobo antes en fuente" in informe_r.review.by_criterion[Criterion.hook].reason

    def test_nota_no_evaluada_no_vuelve_como_correccion_al_guionista(self):
        """Mandar al guionista "mejora el hook" por un cero que nunca fue juzgado
        gastaria la revision en el lugar equivocado."""
        inventado = guion(body=CUERPO + " El entrenamiento uso 12000 GPUs.")
        informe_r, _ = juzgar(informe(), script=inventado)
        assert informe_r.review.revision_notes == ["[fuente 0/2] " + (
            informe_r.review.by_criterion[Criterion.fuente].reason
        )]

    def test_numero_del_dossier_deja_el_criterio_al_modelo(self):
        """Cuando los numeros cuadran, fuente vuelve a ser lectura: la medida no
        tiene como juzgar una afirmacion que va mas alla del dossier."""
        anclado = guion(body=CUERPO + " Ocupa 5,9 GB y retiene 98,2% del rendimiento.")
        informe_r, llm = juzgar(informe(), script=anclado)
        nota = informe_r.review.by_criterion[Criterion.fuente]
        assert nota.score == 2
        assert not nota.measured
        assert len(llm.calls) == 1

    def test_el_modelo_aun_puede_reprobar_la_fuente_que_la_medida_aprobo(self):
        informe_r, _ = juzgar(informe(fuente=0))
        nota = informe_r.review.by_criterion[Criterion.fuente]
        assert nota.score == 0
        assert not nota.measured
        assert not informe_r.approved


class TestReglaDeAprobacion:
    def test_corte_en_el_limite_aprueba(self):
        """11/14 con ningun cero y ningun veto."""
        informe_r, _ = juzgar(informe(hook=1, punto_de_vista=1, es_es=1))
        assert informe_r.review.total == RUBRIC_CUTOFF
        assert informe_r.approved

    def test_debajo_del_corte_reprueba(self):
        informe_r, _ = juzgar(informe(hook=1, punto_de_vista=1, es_es=1, cta=1))
        assert informe_r.review.total == RUBRIC_CUTOFF - 1
        assert not informe_r.approved

    def test_criterio_a_cero_reprueba_incluso_con_la_suma_en_el_corte(self):
        """Resumen de noticia con el resto perfecto llegaria a 12 de 14 y pasaria.

        Es exactamente el "AI slop" que el Creator Rewards excluye, asi que la
        suma sola no decide.
        """
        informe_r, _ = juzgar(informe(punto_de_vista=0))
        assert informe_r.review.total == 12
        assert informe_r.review.total >= RUBRIC_CUTOFF
        assert not informe_r.approved
        assert informe_r.review.zeroed[0].criterion is Criterion.punto_de_vista

    def test_veto_en_fuente_reprueba_incluso_con_nota_alta(self):
        """Media fuente es fuente faltando: 1/2 en fuente ya es veto."""
        informe_r, _ = juzgar(informe(fuente=1))
        assert informe_r.review.total == 13
        assert not informe_r.approved
        assert [s.criterion for s in informe_r.review.vetoed] == [Criterion.fuente]

    def test_calidad_debil_compensable_no_es_veto(self):
        """El corte existe para eso: hook debil se puede compensar, fuente no."""
        informe_r, _ = juzgar(informe(hook=1, cta=1))
        assert informe_r.approved


class TestNotasDeRevision:
    def test_el_veto_va_antes_del_resto(self):
        """No sirve de nada mejorar el hook de un guion que cita numero sin fuente."""
        informe_r, _ = juzgar(informe(fuente=1, hook=0))
        notas = informe_r.review.revision_notes
        assert notas[0].startswith("[fuente 1/2]")
        assert any(n.startswith("[hook 0/2]") for n in notas)

    def test_criterio_con_nota_maxima_no_entra_en_la_revision(self):
        informe_r, _ = juzgar(informe(hook=1))
        assert len(informe_r.review.revision_notes) == 1
        assert "hook" in informe_r.review.revision_notes[0]

    def test_guion_perfecto_no_tiene_nota_de_revision(self):
        informe_r, _ = juzgar(informe())
        assert informe_r.review.revision_notes == []


class TestRespuestaDefectuosa:
    @pytest.mark.parametrize("nota", [5, -1, "dos", None, True])
    def test_nota_fuera_de_la_escala_no_se_redondea(self, nota):
        """Aceptar 5 seria dejar que el modelo redefina el corte de la rubrica."""
        bruto = json.loads(informe())
        bruto["hook"]["score"] = nota
        with pytest.raises(LLMError, match="nota invalida"):
            juzgar(json.dumps(bruto))

    def test_criterio_ausente_en_la_respuesta_lanza(self):
        bruto = json.loads(informe())
        del bruto["cta"]
        with pytest.raises(LLMError, match="cta"):
            juzgar(json.dumps(bruto))

    def test_respuesta_malformada_gana_una_segunda_oportunidad(self):
        """Medido el 18/09/2026: el Flash corto el JSON del informe a la mitad, y
        el guion ya escrito se perdia por culpa de eso. Una segunda llamada cuesta
        menos que rehacer el guion entero en la siguiente ejecucion."""
        llm = ScriptedLLM(responses=['{"hook": {"reason": "corto aqui', informe()])
        informe_r = Judge(llm).review(guion(), dossier())
        assert informe_r.approved
        assert len(llm.calls) == 2

    def test_coste_de_los_dos_intentos_entra_en_el_informe(self):
        llm = ScriptedLLM(responses=["no es json", informe()])
        informe_r = Judge(llm).review(guion(), dossier())
        # Las dos llamadas se cobraron; contar solo la que funciono
        # subestimaria el coste del informe en el eval del M5.
        assert informe_r.usage.output_tokens > len(informe().split())

    def test_error_final_dice_cuanto_se_gasto(self):
        """Bajo restriccion de $0, saber el coste de una ejecucion que no entrego
        nada es parte del resultado."""
        llm = ScriptedLLM(responses=["no es json", "tampoco es"])
        with pytest.raises(LLMError, match="gastados [0-9]+ tokens"):
            Judge(llm).review(guion(), dossier())

    def test_motivo_vacio_no_rompe_el_contrato(self):
        """`reason` es obligatorio en el contrato; informe sin motivo gana un texto
        que lo dice, en vez de tumbar el juicio entero."""
        bruto = json.loads(informe())
        bruto["hook"]["reason"] = ""
        informe_r, _ = juzgar(json.dumps(bruto))
        assert "no justifico" in informe_r.review.by_criterion[Criterion.hook].reason


class TestPrompt:
    def test_dossier_y_guion_van_en_el_prompt(self):
        prompt = build_prompt(guion(), dossier())
        assert "[0] El Bonsai 2 27B ocupa 5,9 GB" in prompt
        assert "HOOK:" in prompt and "CIERRE:" in prompt

    def test_pide_la_razon_antes_de_la_nota(self):
        """El orden cambia el juicio: nota primero hace que el modelo justifique
        lo que ya decidio."""
        prompt = build_prompt(guion(), dossier())
        assert "razon antes de la nota" in prompt

    def test_avisa_que_duracion_y_politica_se_miden_fuera(self):
        prompt = build_prompt(guion(), dossier())
        assert "No evalues duracion ni politica" in prompt

    def test_coste_del_informe_se_mide(self):
        informe_r, _ = juzgar(informe())
        assert informe_r.usage.input_tokens > 0
        assert informe_r.latency_s >= 0
        assert informe_r.review.model == "scripted-1"
