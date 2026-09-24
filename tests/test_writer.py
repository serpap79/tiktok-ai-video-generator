"""Tests del guionista -- porción 2 del M3.

El criterio aquí es que **nada mecánico llegue al juez**: conteo de palabras
fuera de la franja de monetización, término de búsqueda en portugués, índice de
hecho inexistente y número que no está en el dossier son defectos verificables
sin juicio. Gastar una ronda de revisión del juez con esos errores sería
quemar cuota del free tier.

Lo que este archivo NO comprueba, a propósito: si el hook es bueno, si el
cierre tiene punto de vista propio, si el castellano suena hablado. Eso es la
rúbrica del juez (porción 3) y no se puede afirmar por regla.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.models import Dossier, Fact, Script
from agent.ports.llm import LLMUnavailable
from agent.writer.writer import (
    MAX_PALABRAS,
    MIN_PALABRAS,
    Screenwriter,
    build_prompt,
)

AHORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

HOOK = "Un modelo gigante ahora cabe en tu bolsillo."
CIERRE = ("Número sin base de comparación no es medición, es marketing. Lleva esa "
          "pregunta al próximo anuncio que leas.")
TERMINOS = [
    "neural network nodes",
    "ai deep learning loop",
    "data stream tunnel",
    "abstract digital plexus",
]


def dossier() -> Dossier:
    return Dossier(
        topic="Bonsai 2 27B: modelo de 27B en 5,9 GB",
        facts=[
            Fact(claim="El Bonsai 2 27B ocupa 5,9 GB, más de 9x menor que el original",
                 source_url="https://prismml.com/news/bonsai-2-27b", source_name="PrismML",
                 quote="that occupies 5.9 GB on disk, more than 9x smaller"),
            Fact(claim="Retiene el 98,2% del rendimiento agregado en los benchmarks",
                 source_url="https://prismml.com/news/bonsai-2-27b", source_name="PrismML",
                 quote="the model retains 98.2% of the original score"),
            Fact(claim="Funciona a 143 tokens por segundo en una RTX 5090",
                 source_url="https://cienciahoje.com.br/ia/bonsai",
                 source_name="Ciencia Hoy",
                 quote="Throughput reaches 143 tokens per second on a single RTX 5090"),
        ],
        collected_at=AHORA,
    )


def respuesta(
    total: int = 187,
    used: tuple[int, ...] = (0, 1),
    terminos: list[str] | None = None,
    extra: str = "",
    hook: str = HOOK,
    closing: str = CIERRE,
) -> str:
    """Respuesta del modelo en el formato del schema, con conteo de palabras exacto."""
    sujeto = "El Bonsai 27B es un modelo. "
    fijas = len((hook + " " + sujeto + closing + " " + extra).split())
    cuerpo = sujeto + " ".join(["detalle"] * max(total - fijas, 1))
    return json.dumps({
        "hook": hook,
        "body": (extra + " " + cuerpo).strip(),
        "closing": closing,
        "search_terms": TERMINOS if terminos is None else terminos,
        "caption": "Modelo gigante, disco pequeño.\nEl Bonsai 27B muestra lo que la "
                   "compresión ya hace.",
        "used_facts": list(used),
    }, ensure_ascii=False)


def escribir(llm, **kwargs):
    return Screenwriter(llm, **kwargs).write(dossier())


class TestAceptacionDeLaPorcion:
    def test_guion_valido_en_el_primer_intento(self):
        report = escribir(ScriptedLLM(responses=[respuesta()]))
        assert report.ok
        assert len(report.attempts) == 1
        assert MIN_PALABRAS <= report.script.word_count <= MAX_PALABRAS
        assert 60 <= report.script.estimated_duration_s <= 90

    def test_el_modelo_apunta_el_hecho_y_no_lo_reescribe(self):
        """Si el modelo pudiera redactar el hecho, la afirmación del guion dejaría
        de ser rastreable a lo que la fuente dice -- todo el punto de haber
        dossier."""
        report = escribir(ScriptedLLM(responses=[respuesta(used=(0, 2))]))
        d = dossier()
        assert [f.claim for f in report.script.facts] == [
            d.facts[0].claim, d.facts[2].claim,
        ]
        assert all(str(f.source_url).startswith("http") for f in report.script.facts)

    def test_guion_sirve_de_entrada_para_el_renderizador(self):
        """El contrato Script es el mismo que el M0 ya renderiza: lo que sale de
        aquí entra en el `render` sin adaptación."""
        report = escribir(ScriptedLLM(responses=[respuesta()]))
        assert report.script.narration.startswith(HOOK.split(".")[0])
        assert len(report.script.search_terms) >= 3
        assert all(t.isascii() for t in report.script.search_terms)


class TestFranjaDeDuracion:
    def test_demasiado_corto_vuelve_con_el_conteo_medido(self):
        """Por debajo de 60s el vídeo no es elegible al Creator Rewards. El defecto
        es contable, así que el guionista corrige solo en vez de ocupar al juez."""
        llm = ScriptedLLM(responses=[respuesta(total=90), respuesta(total=180)])
        report = escribir(llm)
        assert report.ok
        assert len(report.attempts) == 2
        (violacion,) = report.attempts[0].violations
        assert "90 palabras" in violacion
        assert f"{MIN_PALABRAS}" in violacion and f"{MAX_PALABRAS}" in violacion
        # La corrección medida vuelve al modelo en texto, en la segunda llamada.
        assert "90 palabras" in llm.calls[1].prompt
        assert "CORRIGE EL INTENTO ANTERIOR" in llm.calls[1].prompt

    def test_demasiado_largo_tambien_se_corrige(self):
        report = escribir(ScriptedLLM(responses=[respuesta(total=400), respuesta(total=200)]))
        assert report.ok
        assert "400 palabras" in report.attempts[0].violations[0]

    def test_techo_de_intentos_no_entrega_guion_malo(self):
        """Tres intentos errados son problema de instrucción, no de suerte.
        Insistir quema cuota; entregar fuera de la franja rompería la
        monetización."""
        llm = ScriptedLLM(responses=[respuesta(total=90)] * 3)
        report = escribir(llm)
        assert not report.ok
        assert len(report.attempts) == 3
        assert len(llm.calls) == 3
        assert report.violations  # el motivo de la última reprobación queda en el informe


class TestAnclajeEnElDossier:
    def test_numero_en_digito_fuera_del_dossier_reprueba(self):
        llm = ScriptedLLM(responses=[
            respuesta(extra="El entrenamiento usó 12000 GPUs."),
            respuesta(),
        ])
        report = escribir(llm)
        assert report.ok
        assert "12000" in report.attempts[0].violations[0]

    def test_numero_del_dossier_pasa_incluso_en_notacion_es(self):
        report = escribir(ScriptedLLM(responses=[
            respuesta(extra="Ocupa 5,9 GB en el disco. Retiene casi todo."),
        ]))
        assert report.ok

    def test_indice_inexistente_reprueba_diciendo_la_franja_valida(self):
        llm = ScriptedLLM(responses=[respuesta(used=(0, 9)), respuesta()])
        report = escribir(llm)
        (violacion,) = llm_violaciones(report, 0)
        assert "[9]" in violacion
        assert "0 a 2" in violacion

    def test_guion_sin_hecho_reprueba(self):
        """Guion sin fuente es exactamente para lo que existe este proyecto."""
        llm = ScriptedLLM(responses=[respuesta(used=()), respuesta()])
        report = escribir(llm)
        assert any("used_facts esta vacio" in v for v in llm_violaciones(report, 0))

    def test_indice_repetido_no_duplica_el_hecho(self):
        report = escribir(ScriptedLLM(responses=[respuesta(used=(1, 1, 1))]))
        assert len(report.script.facts) == 1


class TestContratoDelScript:
    def test_termino_de_busqueda_en_portugues_reprueba(self):
        """Los términos van directos a Pexels, sin traducción: acento casi siempre
        significa que el modelo respondió en pt-BR, y el material vuelve mal."""
        llm = ScriptedLLM(responses=[
            respuesta(terminos=["placa de vídeo", "chip de memória", "sala de servidores"]),
            respuesta(),
        ])
        report = escribir(llm)
        assert report.ok
        assert any("search_terms" in v for v in llm_violaciones(report, 0))

    def test_terminos_de_menos_reprueban(self):
        llm = ScriptedLLM(responses=[respuesta(terminos=["one term"]), respuesta()])
        report = escribir(llm)
        assert any("search_terms" in v for v in llm_violaciones(report, 0))

    def test_campo_faltante_reprueba_con_el_nombre_del_campo(self):
        llm = ScriptedLLM(responses=[json.dumps({"hook": "corto"}), respuesta()])
        report = escribir(llm)
        violaciones = " ".join(llm_violaciones(report, 0))
        assert "body" in violaciones and "search_terms" in violaciones


class TestRespuestaDefectuosa:
    def test_json_roto_es_violacion_corregible_y_no_fallo_de_etapa(self):
        report = escribir(ScriptedLLM(responses=["no puedo ayudar", respuesta()]))
        assert report.ok
        assert "no vino como objeto JSON" in report.attempts[0].violations[0]

    def test_respuesta_truncada_pide_texto_mas_corto(self):
        """Truncamiento es presupuesto de token, no error de escritura: mandar al
        modelo 'corregir el JSON' no resolvería nada."""

        class Truncado(ScriptedLLM):
            def complete(self, prompt, **kwargs):
                c = super().complete(prompt, **kwargs)
                return c.model_copy(update={"finish_reason": "length"})

        report = escribir(Truncado(responses=[respuesta()] * 3))
        assert not report.ok
        assert "cortada por limite de tokens" in report.attempts[0].violations[0]

    def test_cuota_agotada_sube_a_quien_llamo(self):
        """No hay qué corregir en el prompt: repetir solo gasta la cuota que falta."""
        def responder(prompt: str) -> str:
            raise LLMUnavailable("cuota diaria agotada (429)")

        with pytest.raises(LLMUnavailable):
            escribir(ScriptedLLM(responder=responder))


class TestCosteEHistorial:
    def test_coste_suma_los_intentos_que_fallaron(self):
        """El intento reprobado también se cobró. Contar solo el que pasó
        subestimaría el coste del guion en la eval del M5."""
        report = escribir(ScriptedLLM(responses=[respuesta(total=90), respuesta()]))
        primero, segundo = report.attempts
        assert report.usage.input_tokens == (
            primero.usage.input_tokens + segundo.usage.input_tokens
        )
        assert report.usage.output_tokens > 0

    def test_intentos_quedan_en_el_informe_incluso_con_exito(self):
        """Es lo que revela prompt débil: si toda ejecución gasta dos rondas en el
        mismo defecto, el problema es la instrucción, no el modelo."""
        report = escribir(ScriptedLLM(responses=[respuesta(total=90), respuesta()]))
        assert len(report.attempts) == 2
        assert report.attempts[0].violations and not report.attempts[1].violations
        assert report.attempts[0].word_count == 90

    def test_modelo_y_proveedor_quedan_registrados(self):
        report = escribir(ScriptedLLM(responses=[respuesta()]))
        assert report.model == "scripted-1"
        assert report.provider == "scripted"


class TestPrompt:
    def test_hechos_van_indexados_con_fuente_y_fragmento(self):
        prompt = build_prompt(dossier())
        assert "[0] El Bonsai 2 27B ocupa 5,9 GB" in prompt
        assert "fuente: PrismML" in prompt
        assert "that occupies 5.9 GB on disk" in prompt

    def test_franja_de_palabras_es_la_regla_de_monetizacion_en_el_texto(self):
        prompt = build_prompt(dossier())
        assert str(MIN_PALABRAS) in prompt and str(MAX_PALABRAS) in prompt
        assert "monetizacion" in prompt

    def test_terminos_en_ingles_y_orden_cronologico_se_piden(self):
        prompt = build_prompt(dossier())
        assert "EN INGLES" in prompt
        assert "cronologico" in prompt

    def test_primer_intento_no_tiene_seccion_de_correccion(self):
        assert "CORRIGE" not in build_prompt(dossier())


def llm_violaciones(report, indice: int) -> list[str]:
    return report.attempts[indice].violations


class TestMarcadorDeCita:
    """Defecto medido en ejecución real, no imaginado.

    El modelo escribió "...en tu proyecto [0]." y "...en un proyecto [0, 3]." --
    dejando escapar en el texto HABLADO el índice que debía ir solo en
    used_facts. El sintetizador diría "cero" y "tres" en voz alta en el vídeo.
    """

    def test_marcador_en_el_texto_reprueba_con_el_motivo_correcto(self):
        llm = ScriptedLLM(responses=[
            respuesta(extra="La lectura cambia sin CLAUDE.md [0] y eso vale para todos [1, 3]."),
            respuesta(),
        ])
        report = escribir(llm)
        assert report.ok
        (violacion,) = [v for v in report.attempts[0].violations if "marcador" in v]
        assert "[0]" in violacion and "[1, 3]" in violacion
        assert "voz alta" in violacion

    def test_marcador_no_se_convierte_en_acusacion_de_numero_inventado(self):
        """El mensaje equivocado era "número 0, 1, 2 sin respaldo": verdad e
        inútil para saber qué hacer."""
        llm = ScriptedLLM(responses=[respuesta(extra="Vale para el proyecto [0]."),
                                     respuesta()])
        report = escribir(llm)
        assert not any("numero que no esta en el dossier" in v
                       for v in report.attempts[0].violations)

    def test_numero_inventado_de_verdad_sigue_siendo_cazado(self):
        llm = ScriptedLLM(responses=[
            respuesta(extra="Según el hecho [0], fueron 12000 GPUs."), respuesta(),
        ])
        report = escribir(llm)
        violaciones = " ".join(report.attempts[0].violations)
        assert "marcador" in violaciones
        assert "12000" in violaciones

    def test_ano_en_el_texto_no_se_confunde_con_marcador(self):
        llm = ScriptedLLM(responses=[
            respuesta(extra="Ocupa 5,9 GB desde 2026."),
            respuesta(),
        ])
        report = escribir(llm)
        assert report.ok
        violaciones = " ".join(report.attempts[0].violations)
        assert "marcador" not in violaciones
        assert "2026" in violaciones

    def test_el_prompt_prohibe_el_marcador(self):
        from agent.writer.writer import build_prompt
        prompt = build_prompt(dossier())
        assert "'[0]'" in prompt and "used_facts" in prompt


class TestDossierFino:
    """Medir antes de pagar, como hacen el curador y el juez.

    Un dossier de 4 hechos sacados de UNA frase llevó al guionista a tres
    intentos, todos entre 104 y 157 palabras: faltaba asunto, no instrucción.
    """

    def test_dossier_con_menos_de_tres_hechos_no_gasta_llamada(self):
        from agent.models import Dossier
        from agent.writer.writer import Screenwriter

        fino = Dossier(topic="t", facts=dossier().facts[:2], collected_at=AHORA)
        llm = ScriptedLLM(responses=[])
        report = Screenwriter(llm).write(fino)

        assert not report.ok
        assert llm.calls == []
        assert report.attempts == []
        assert "dossier fino: 2 hecho(s)" in report.refusal
        assert report.usage.total_tokens == 0

    def test_recusa_diciendo_que_hacer(self):
        from agent.models import Dossier
        from agent.writer.writer import Screenwriter

        fino = Dossier(topic="t", facts=dossier().facts[:1], collected_at=AHORA)
        report = Screenwriter(ScriptedLLM()).write(fino)
        assert "Investiga otras fuentes" in report.refusal

    def test_tres_hechos_ya_autorizan_el_intento(self):
        report = escribir(ScriptedLLM(responses=[respuesta()]))
        assert report.ok
        assert report.refusal == ""

    def test_el_guion_de_referencia_del_m0_no_seria_recusado(self):
        """Cinco hechos de una única fuente rinden guion: el número de FUENTES no
        entra en la regla, el de hechos distintos sí."""
        import json
        from pathlib import Path

        from agent.models import Dossier, Script
        from agent.writer.writer import thin_dossier_reason

        bruto = json.loads(
            (Path(__file__).resolve().parent.parent / "fixtures" / "guion_manual.json")
            .read_text(encoding="utf-8")
        )
        bruto.pop("_comment", None)
        referencia = Script.model_validate(bruto)
        assert thin_dossier_reason(
            Dossier(topic=referencia.topic, facts=referencia.facts, collected_at=AHORA)
        ) == ""


class TestCaptionDelPost:
    """Guía de la marca: gancho escrito SIN repetir el audio + contexto; sin link/hashtag."""

    def _script(self, caption: str) -> Script:
        return Script(topic="Tema", hook="Un modelo gigante ahora cabe en un pendrive.",
                      body=" ".join(["palabra"] * 60), closing="Y tu, confias?",
                      search_terms=["neural network nodes", "data stream tunnel"],
                      caption=caption)

    def test_caption_buena_pasa(self):
        from agent.writer.writer import caption_problems
        assert caption_problems(self._script(
            "Disco pequeño, modelo enorme.\nEl Bonsai 27B reduce 9x el tamaño.")) == []

    def test_repetir_el_hook_reprueba(self):
        from agent.writer.writer import caption_problems
        (p,) = caption_problems(self._script("Un modelo gigante ahora cabe en un pendrive."))
        assert "repite el hook" in p

    def test_hashtag_link_y_tres_lineas_reprueban(self):
        from agent.writer.writer import caption_problems
        problemas = " ".join(caption_problems(self._script(
            "Linea uno #ia\nmira en prismml.com\nlinea tres")))
        assert "hashtag" in problemas and "link" in problemas and "linea" in problemas

    def test_vacia_reprueba_con_instruccion(self):
        from agent.writer.writer import caption_problems
        assert "2 lineas" in caption_problems(self._script(""))[0]
