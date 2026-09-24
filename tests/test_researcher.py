"""Tests del investigador -- y el aceite de la primera tajada del M3.

El criterio no es "el dossier quedo bien": es que **ningun hecho entre sin URL
verificable**, y que la URL venga de nosotros y no del modelo. El resto de los
tests existe porque cada uno corresponde a una forma conocida de que el modelo
parezca cierto estando equivocado.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.models import Decision, Verdict
from agent.ports.llm import LLMBlocked, LLMUnavailable
from agent.research.fetch import PageFetcher
from agent.research.researcher import Researcher
from agent.research.sources import Candidate

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "research"
ARTICULO = (FIXTURES / "artigo.html").read_text(encoding="utf-8")
AHORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

URL_REAL = "https://prismml.com/news/bonsai-2-27b"

# Pasajes copiados del texto que el extractor produce para la fixture. Es lo que
# el modelo tendria que devolver para que la puerta de pasaje acepte.
PASAJE_TAMANO = "that occupies 5.9 GB on disk, more than 9x smaller than"
PASAJE_BENCH = "the model retains 98.2% of the original score"
PASAJE_THROUGHPUT = "Throughput reaches 143 tokens per second on a single RTX 5090"


def respuesta(*hechos: tuple[str, str]) -> str:
    """Respuesta del modelo en el formato del esquema: pares (pasaje, afirmacion)."""
    return json.dumps(
        {"facts": [{"quote": q, "claim": c} for q, c in hechos]}, ensure_ascii=False
    )


def decision(term: str = "Bonsai 2 27B en 5,9 GB", source: str = "hacker_news") -> Decision:
    return Decision(
        term=term, source=source, verdict=Verdict.selected,
        reason="test", score=0.9, niche_fit=0.8, decided_at=AHORA,
    )


def fetcher(handler=None) -> PageFetcher:
    handler = handler or (
        lambda r: httpx.Response(200, text=ARTICULO, headers={"content-type": "text/html"})
    )
    return PageFetcher(
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    )


def investigar(llm, candidatos=None, **kwargs) -> tuple:
    r = Researcher(llm, fetcher=fetcher(kwargs.pop("handler", None)), **kwargs).research(
        decision(), candidates=candidatos or [Candidate(url=URL_REAL, source_name="PrismML")]
    )
    return r


class TestAceiteDeLaTajada:
    def test_todo_hecho_tiene_url_rastreable(self):
        llm = ScriptedLLM(responses=[respuesta(
            (PASAJE_TAMANO, "El Bonsai 2 27B ocupa 5,9 GB, mas de 9 veces menor que el original"),
            (PASAJE_BENCH, "El modelo retiene 98,2% de la nota original en los benchmarks"),
        )])
        report = investigar(llm)

        assert report.ok
        assert len(report.facts) == 2
        assert all(str(f.source_url) == URL_REAL for f in report.facts)
        assert all(f.source_name == "PrismML" for f in report.facts)
        assert report.dossier.source_urls == {URL_REAL}

    def test_la_url_la_estampamos_nosotros_y_no_se_pide_al_modelo(self):
        """El error mas caro posible aqui es un hecho real con fuente cambiada.

        Parece anclado, pasa al juez, y solo aparece cuando alguien hace clic en
        el enlace. Por eso el modelo ni tiene como opinar: el campo que mande se
        ignora, porque de donde salio el texto es informacion que ya tenemos.
        """
        inventado = json.dumps({"facts": [{
            "quote": PASAJE_TAMANO,
            "claim": "El Bonsai 2 27B ocupa 5,9 GB, mas de 9 veces menor que el original",
            "source_url": "https://fuente-que-el-modelo-invento.com/post",
            "source_name": "Medio Inexistente",
        }]})
        report = investigar(ScriptedLLM(responses=[inventado]))

        (hecho,) = report.facts
        assert str(hecho.source_url) == URL_REAL
        assert hecho.source_name == "PrismML"

    def test_esquema_pedido_al_modelo_no_tiene_campo_de_fuente(self):
        """No sirve ignorar la URL del modelo y pedirala ademas: pedir invita al
        modelo a rellenarla, y alguien luego usaria el campo por error."""
        llm = ScriptedLLM(responses=[respuesta((PASAJE_BENCH, "Retiene 98,2% de la nota"))])
        investigar(llm)
        propiedades = llm.calls[0].schema["properties"]["facts"]["items"]["properties"]
        assert set(propiedades) == {"quote", "claim"}


class TestUnaLlamadaPorFuente:
    def test_cada_fuente_recibe_una_sola_llamada(self):
        """Una llamada por fuente es lo que permite estampar la URL correcta.

        Mandar tres paginas juntas ahorraria cuota y devolveria hecho sin dueno.
        """
        llm = ScriptedLLM(responses=[
            respuesta((PASAJE_TAMANO, "Ocupa 5,9 GB, mas de 9 veces menor")),
            respuesta((PASAJE_BENCH, "Retiene 98,2% de la nota original")),
        ])
        report = investigar(llm, candidatos=[
            Candidate(url="https://prismml.com/a", source_name="PrismML"),
            Candidate(url="https://cienciaes.com/b", source_name="Ciencia Es"),
        ])
        assert len(llm.calls) == 2
        assert {f.source_name for f in report.facts} == {"PrismML", "Ciencia Es"}

    def test_el_texto_de_la_pagina_va_en_el_prompt(self):
        llm = ScriptedLLM(responses=[respuesta((PASAJE_BENCH, "Retiene 98,2% de la nota"))])
        investigar(llm)
        prompt = llm.calls[0].prompt
        assert "5.9 GB" in prompt
        assert "Bonsai 2 27B en 5,9 GB" in prompt  # el tema en investigacion

    def test_techo_de_fuentes_se_respeta(self):
        llm = ScriptedLLM(responses=[respuesta((PASAJE_BENCH, "Retiene 98,2% de la nota"))])
        investigar(llm, max_sources=1, candidatos=[
            Candidate(url="https://prismml.com/a"),
            Candidate(url="https://otro.com/b"),
        ])
        assert len(llm.calls) == 1


class TestPuertaDePasaje:
    def test_parafraseo_en_lugar_de_copia_reprueba(self):
        """El modelo debia copiar y reescribio. No se puede saber si la afirmacion
        es verdadera -- y "no se puede saber" reprueba."""
        report = investigar(ScriptedLLM(responses=[respuesta((
            "the model keeps 98.2 percent of the original score",
            "El modelo retiene 98,2% de la nota original",
        ))]))
        assert not report.ok
        (descartado,) = report.discarded
        assert "no existe en la página" in descartado.reason
        assert descartado.source_url == URL_REAL

    def test_pasaje_ausente_reprueba(self):
        report = investigar(ScriptedLLM(responses=[respuesta(
            ("", "El modelo retiene 98,2% de la nota original"),
        )]))
        assert not report.ok
        assert "pasaje de apoyo ausente" in report.discarded[0].reason

    def test_pasaje_demasiado_corto_reprueba(self):
        """"5.9 GB" aparece en el menu tambien: pasaje corto casa con cualquier
        cosa y no prueba nada."""
        report = investigar(ScriptedLLM(responses=[respuesta(
            ("5.9 GB", "El modelo ocupa 5,9 GB"),
        )]))
        assert not report.ok
        assert "demasiado corto" in report.discarded[0].reason

    def test_espacio_estrecho_dentro_del_numero_no_reprueba(self):
        """Medido el 18/09/2026: el `groq/compound-mini` devuelve "5,9\u202fGB" --
        espacio estrecho sin salto dentro del numero. Es formateo tipografico, no
        parafraseo, y reprobar por eso tiraria copia literal correcta."""
        estrecho = "that occupies 5.9\u202fGB on disk, more than 9x smaller than"
        report = investigar(ScriptedLLM(responses=[respuesta(
            (estrecho, "El Bonsai 2 27B ocupa 5,9\u202fGB, mas de 9 veces menor"),
        )]))
        assert report.ok
        assert report.facts[0].quote.startswith("that occupies 5.9")

    def test_diferencia_de_espacio_y_salto_de_linea_no_reprueba(self):
        """El pasaje atraviesa un salto de linea en el HTML; exigir formateo
        identico reprobaria copia literal correcta."""
        atraviesa = "a ternary-weight version of Qwen3.8 27B     that occupies 5.9 GB"
        report = investigar(ScriptedLLM(responses=[respuesta(
            (atraviesa, "El Bonsai 2 27B es una version de pesos ternarios del Qwen3.8 27B"),
        )]))
        assert report.ok


class TestPuertaNumerica:
    def test_numero_inventado_reprueba_con_el_numero_en_el_motivo(self):
        report = investigar(ScriptedLLM(responses=[respuesta((
            PASAJE_TAMANO,
            "El Bonsai 2 27B ocupa 5,9 GB y fue entrenado con 12000 GPUs",
        ))]))
        assert not report.ok
        assert "12000" in report.discarded[0].reason

    def test_numero_en_notacion_espanola_pasa(self):
        report = investigar(ScriptedLLM(responses=[respuesta((
            PASAJE_THROUGHPUT,
            "Corre a 143 tokens por segundo en una RTX 5090",
        ))]))
        assert report.ok

    def test_hecho_bueno_sobrevive_al_lado_de_hecho_reprobado(self):
        """Una afirmacion mala no contamina la fuente entera."""
        report = investigar(ScriptedLLM(responses=[respuesta(
            (PASAJE_TAMANO, "El Bonsai 2 27B ocupa 5,9 GB, mas de 9 veces menor"),
            (PASAJE_BENCH, "Retiene 98,2% de la nota y cuesta 500 mil dolares"),
        )]))
        assert len(report.facts) == 1
        assert len(report.discarded) == 1


class TestFallosAislados:
    def test_cuota_agotada_en_una_fuente_no_tira_las_otras(self):
        """Misma regla del radar: la ventana de un tema es de horas."""
        def responder(prompt: str) -> str:
            if "cienciaes" in prompt or "Ciencia Es" in prompt:
                raise LLMUnavailable("cuota diaria agotada (429)")
            return respuesta((PASAJE_BENCH, "Retiene 98,2% de la nota original"))

        report = investigar(ScriptedLLM(responder=responder), candidatos=[
            Candidate(url="https://prismml.com/a", source_name="PrismML"),
            Candidate(url="https://cienciaes.com/b", source_name="Ciencia Es"),
        ])
        assert report.ok
        assert len(report.facts) == 1
        assert any("LLMUnavailable" in m for m in report.failures.values())

    def test_filtro_de_contenido_del_proveedor_se_registra_y_no_explota(self):
        def responder(prompt: str) -> str:
            raise LLMBlocked("filtro de contenido")

        report = investigar(ScriptedLLM(responder=responder))
        assert not report.ok
        assert any("LLMBlocked" in m for m in report.failures.values())

    def test_pagina_caida_se_registra_con_la_url(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403)

        report = investigar(ScriptedLLM(responses=[]), handler=handler)
        assert not report.ok
        assert "403" in report.failures[URL_REAL]

    def test_pagina_sin_texto_no_gasta_llamada_de_modelo(self):
        """La cuota es el recurso escaso del free tier: pagina sin contenido no
        se convierte en prompt, y el motivo queda grabado."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<p>Acepta las cookies</p>",
                                  headers={"content-type": "text/html"})

        llm = ScriptedLLM(responses=[])
        report = investigar(llm, handler=handler)
        assert llm.calls == []
        assert "demasiado corto" in report.failures[URL_REAL]

    def test_respuesta_del_modelo_fuera_del_contrato_es_fallo_de_la_fuente(self):
        report = investigar(ScriptedLLM(responses=["no puedo ayudar con eso"]))
        assert not report.ok
        assert any("LLMError" in m for m in report.failures.values())

    def test_tema_sin_fuente_candidata_no_llama_al_modelo(self):
        llm = ScriptedLLM(responses=[])
        report = Researcher(llm, fetcher=fetcher()).research(decision(), candidates=[])
        assert llm.calls == []
        assert not report.ok
        assert "descubrimiento" in report.failures


class TestCosteMedido:
    def test_tokens_y_latencia_se_suman_entre_las_fuentes(self):
        """El M5 compara proveedores por calidad y por consumo. Numero medido en
        el momento es mas fiable que reconstruido del log despues."""
        llm = ScriptedLLM(responses=[
            respuesta((PASAJE_TAMANO, "Ocupa 5,9 GB, mas de 9 veces menor")),
            respuesta((PASAJE_BENCH, "Retiene 98,2% de la nota original")),
        ])
        report = investigar(llm, candidatos=[
            Candidate(url="https://prismml.com/a"),
            Candidate(url="https://cienciaes.com/b"),
        ])
        assert report.usage.input_tokens > 0
        assert report.usage.output_tokens > 0
        assert report.latency_s >= 0
        assert report.model == "scripted-1"

    def test_fuente_distinta_se_cuenta_por_dominio(self):
        llm = ScriptedLLM(responses=[
            respuesta((PASAJE_TAMANO, "Ocupa 5,9 GB, mas de 9 veces menor")),
            respuesta((PASAJE_BENCH, "Retiene 98,2% de la nota original")),
        ])
        report = investigar(llm, candidatos=[
            Candidate(url="https://prismml.com/a"),
            Candidate(url="https://prismml.com/b"),
        ])
        assert len(report.facts) == 2
        assert report.source_count == 1


class TestHigieneDeLaRespuesta:
    def test_hecho_repetido_en_la_misma_fuente_entra_una_vez(self):
        afirmacion = "El Bonsai 2 27B ocupa 5,9 GB, mas de 9 veces menor"
        report = investigar(ScriptedLLM(responses=[respuesta(
            (PASAJE_TAMANO, afirmacion), (PASAJE_TAMANO, afirmacion),
        )]))
        assert len(report.facts) == 1

    def test_techo_de_hechos_por_fuente(self):
        llm = ScriptedLLM(responses=[respuesta(
            (PASAJE_TAMANO, "Ocupa 5,9 GB, mas de 9 veces menor que el original"),
            (PASAJE_BENCH, "Retiene 98,2% de la nota original en los benchmarks"),
            (PASAJE_THROUGHPUT, "Corre a 143 tokens por segundo en una RTX 5090"),
        )])
        report = investigar(llm, max_facts_per_source=2)
        assert len(report.facts) == 2

    def test_afirmacion_demasiado_corta_para_el_contrato_se_descarta_con_motivo(self):
        report = investigar(ScriptedLLM(responses=[respuesta((PASAJE_TAMANO, "5,9 GB"))]))
        assert not report.ok
        assert "contrato Fact rechazo" in report.discarded[0].reason

    def test_lista_vacia_del_modelo_es_respuesta_valida(self):
        """Pagina que no trata el tema debe devolver nada, no inventar."""
        report = investigar(ScriptedLLM(responses=['{"facts": []}']))
        assert not report.ok
        assert report.discarded == []
        assert report.pages  # la pagina se leyo; solo no rindio hechos


def test_dossier_sin_hecho_no_autoriza_guion():
    """El contrato Dossier ya lo prohibe; el investigador no intenta burlarlo."""
    report = investigar(ScriptedLLM(responses=['{"facts": []}']))
    assert report.dossier is None
    with pytest.raises(AttributeError):
        _ = report.dossier.facts


class TestUnPasajeUnHecho:
    """Regla que vino de una ejecucion real, no de una suposicion.

    El 18/09/2026, de UNA frase de changelog el modelo saco cuatro "hechos", tres
    de ellos apoyados en el mismo pasaje. El dossier parecia lleno (4 hechos) y no
    daba materia para 60 segundos: el guionista lo intento tres veces y nunca paso
    de 157 palabras.
    """

    def test_pasaje_repetido_en_la_misma_fuente_viene_un_hecho_solo(self):
        report = investigar(ScriptedLLM(responses=[respuesta(
            (PASAJE_TAMANO, "El Bonsai 2 27B ocupa 5,9 GB, mas de 9 veces menor"),
            (PASAJE_TAMANO, "El modelo es mas de nueve veces menor que el original"),
            (PASAJE_BENCH, "Retiene 98,2% de la nota original en los benchmarks"),
        )]))
        assert len(report.facts) == 2
        assert "mismo pasaje ya sostiene otro hecho" in report.discarded[0].reason

    def test_diferencia_de_formateo_no_burla_la_regla(self):
        espaciado = PASAJE_TAMANO.replace(" ", "   ")
        report = investigar(ScriptedLLM(responses=[respuesta(
            (PASAJE_TAMANO, "El Bonsai 2 27B ocupa 5,9 GB, mas de 9 veces menor"),
            (espaciado, "Afirmacion distinta sobre el mismo pasaje reformateado"),
        )]))
        assert len(report.facts) == 1

    def test_pasajes_distintos_rinden_hechos_distintos(self):
        report = investigar(ScriptedLLM(responses=[respuesta(
            (PASAJE_TAMANO, "El Bonsai 2 27B ocupa 5,9 GB, mas de 9 veces menor"),
            (PASAJE_BENCH, "Retiene 98,2% de la nota original en los benchmarks"),
            (PASAJE_THROUGHPUT, "Corre a 143 tokens por segundo en una RTX 5090"),
        )]))
        assert len(report.facts) == 3
        assert report.discarded == []

    def test_el_prompt_pide_pasaje_distinto(self):
        llm = ScriptedLLM(responses=[respuesta((PASAJE_BENCH, "Retiene 98,2% de la nota"))])
        investigar(llm)
        assert "pasaje DIFERENTE" in llm.calls[0].prompt
