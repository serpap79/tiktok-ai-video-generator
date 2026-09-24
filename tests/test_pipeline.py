"""Tests del lazo guionista <-> juez."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from agent.adapters.scripted_llm import ScriptedLLM
from agent.judge.judge import JULGADOS, Judge
from agent.models import Dossier, Fact
from agent.pipeline import produce
from agent.ports.llm import LLMUnavailable
from agent.writer.writer import Screenwriter

AHORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

HOOK = "Un modelo gigante cabe ahora en un pendrive."
CIERRE = "En que numero del anuncio confiarias en la fuente hoy?"


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
            Fact(claim="Corre a 143 tokens por segundo en una RTX 5090",
                 source_url="https://cienciaes.com/ia/bonsai",
                 source_name="Ciencia Es",
                 quote="Throughput reaches 143 tokens per second on a single RTX 5090"),
        ],
        collected_at=AHORA,
    )


def guion_json(total: int = 187) -> str:
    fijos = len((HOOK + " " + CIERRE).split())
    cuerpo = "El Bonsai 27B es un modelo. " + " ".join(["detalle"] * max(total - fijos - 6, 1))
    return json.dumps({
        "hook": HOOK,
        "body": cuerpo,
        "closing": CIERRE,
        "search_terms": ["neural network nodes", "ai deep learning loop",
                         "data stream tunnel", "abstract digital plexus"],
        "caption": (
            "Modelo gigante, disco pequeño.\n"
            "El Bonsai 27B muestra lo que la compresión ya hace."
        ),
        "used_facts": [0, 1],
    }, ensure_ascii=False)


def informe_json(**notas: int) -> str:
    return json.dumps({
        c.value: {"reason": f"motivo para {c.value}", "score": notas.get(c.value, 2)}
        for c in JULGADOS
    })


def rodar(respuestas: list[str], **kwargs):
    """Un unico modelo atiende guionista y juez, en el orden en que se les llama."""
    llm = ScriptedLLM(responses=respuestas)
    return produce(dossier(), Screenwriter(llm), Judge(llm), **kwargs), llm


class TestCaminoAprobado:
    def test_aprobado_en_la_primera_ronda(self):
        report, llm = rodar([guion_json(), informe_json()])
        assert report.approved
        assert len(report.rounds) == 1
        assert len(llm.calls) == 2  # un guion, un informe
        assert report.script.word_count == 187
        assert report.review.approved

    def test_el_coste_suma_guionista_y_juez(self):
        report, _ = rodar([guion_json(), informe_json()])
        ronda = report.rounds[0]
        assert report.usage.input_tokens == (
            ronda.write.usage.input_tokens + ronda.review.usage.input_tokens
        )


class TestRevision:
    def test_reprobado_vuelve_con_las_notas_del_juez_en_el_prompt(self):
        """La nota del juez entra en el mismo canal de las violaciones mecanicas:
        el modelo recibe defectos que corregir y no necesita saber cual se conto."""
        report, llm = rodar([
            guion_json(), informe_json(hook=0),
            guion_json(), informe_json(),
        ])
        assert report.approved
        assert len(report.rounds) == 2
        # La segunda llamada de guion (3a del modelo) trae la nota del juez.
        prompt_revision = llm.calls[2].prompt
        assert "CORRIGE EL INTENTO ANTERIOR" in prompt_revision
        assert "[hook 0/2]" in prompt_revision
        assert report.rounds[1].notes_in

    def test_techo_de_revisiones_para_el_lazo(self):
        """Despues de la segunda revision el modelo suele cambiar de asunto para
        agradar a la rubrica, en vez de mejorar el guion."""
        respuestas = [guion_json(), informe_json(hook=0)] * 3
        report, llm = rodar(respuestas, max_revisions=2)
        assert not report.approved
        assert len(report.rounds) == 3
        assert len(llm.calls) == 6

    def test_reprobado_mantiene_el_ultimo_guion_y_el_ultimo_informe(self):
        """Sin eso no se podria mostrar por que reprobo."""
        report, _ = rodar([guion_json(), informe_json(fuente=1)], max_revisions=0)
        assert not report.approved
        assert report.script is not None
        assert report.review is not None
        assert report.review.vetoed

    def test_revision_cero_juzga_una_sola_vez(self):
        report, llm = rodar([guion_json(), informe_json(hook=0)], max_revisions=0)
        assert len(report.rounds) == 1
        assert len(llm.calls) == 2


class TestFalloAntesDelJuez:
    def test_guion_que_no_pasa_en_lo_mecanico_no_va_al_juez(self):
        """Pagar un informe sobre texto que ya se sabe fuera de la franja de
        duracion seria quemar cuota."""
        report, llm = rodar([guion_json(total=90)] * 3)
        assert not report.approved
        assert len(report.rounds) == 1
        assert report.rounds[0].review is None
        assert len(llm.calls) == 3  # tres intentos del guionista, cero informes

    def test_cuota_agotada_cierra_el_lazo_diciendo_la_etapa(self):
        """Fallo de proveedor es dato registrado, no excepcion perdida -- la misma
        regla del radar. La etapa entra en el texto porque "fallo del proveedor"
        sin decir donde no ayuda a decidir que hacer."""
        def responder(prompt: str) -> str:
            raise LLMUnavailable("cuota diaria agotada (429)")

        llm = ScriptedLLM(responder=responder)
        report = produce(dossier(), Screenwriter(llm), Judge(llm))
        assert not report.approved
        assert report.failure.startswith("guionista: LLMUnavailable")
        assert len(report.rounds) == 1

    def test_fallo_del_juez_se_registra_con_el_guion_preservado(self):
        """El guion ya escrito no se pierde: costo tokens y sirve para que la
        siguiente ejecucion no empiece de cero."""
        llm = ScriptedLLM(responses=[guion_json(), "no es json", "tampoco"])
        report = produce(dossier(), Screenwriter(llm), Judge(llm))
        assert not report.approved
        assert report.failure.startswith("juez: LLMError")
        assert report.script is not None
        assert report.review is None
