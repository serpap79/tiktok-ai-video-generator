"""Modo short (~25s, alcance): franja propia, bucle en el cierre, sin polish roto."""

from __future__ import annotations

import json

from agent.adapters.scripted_llm import ScriptedLLM
from agent.judge.judge import _notas_medidas
from agent.models import SHORT_MAX_DURATION_S, SHORT_MIN_DURATION_S, Script
from agent.writer.writer import BANDS, Screenwriter, build_prompt
from tests.test_writer import dossier

# Tres términos: el corto de ~25s pide de 3 a 5 (un clip de b-roll cubre ~5s).
TERMINOS_SHORT = ["neural network nodes", "data stream tunnel",
                  "abstract digital plexus"]


def respuesta_short(total: int = 64) -> str:
    hook = "¿Un modelo de 27 mil millones cabe en cinco coma nueve gigabytes?"
    closing = "Gigante en el bolsillo: cabe?"
    sujeto = "El Bonsai 27B lo prueba. "
    fijas = len((hook + " " + sujeto + closing).split())
    return json.dumps({
        "hook": hook,
        "body": sujeto + " ".join(["detalle"] * max(total - fijas, 1)),
        "closing": closing,
        "search_terms": TERMINOS_SHORT,
        "caption": "Gigante en el bolsillo.\nEl Bonsai 27B cabe en un pendrive.",
        "used_facts": [0],
    })


class TestShort:
    def test_franja_corta_aprueba_el_tamano_de_25s(self):
        """58-70 palabras = 23-27s, la franja de la parrilla de 20/09/2026.

        Era 30-50 (~15s) hasta que la parrilla subió a cuatro posts con el
        corto pidiendo 25s.
        """
        report = Screenwriter(ScriptedLLM(responses=[respuesta_short()])).write(
            dossier(), mode="short", polish=False)
        assert report.ok
        assert report.script is not None and report.script.format == "short"
        assert 58 <= report.script.word_count <= 70
        assert 23 <= report.script.estimated_duration_s <= 28

    def test_tamano_del_corto_antiguo_ahora_reprueba(self):
        """40 palabras (~15s) quedaron demasiado cortas para la parrilla nueva."""
        report = Screenwriter(ScriptedLLM(responses=[respuesta_short(40)] * 3)).write(
            dossier(), mode="short", polish=False)
        assert not report.ok

    def test_180_palabras_reprueban_en_modo_short(self):
        from tests.test_writer import respuesta as respuesta_long
        report = Screenwriter(ScriptedLLM(responses=[respuesta_long()] * 3)).write(
            dossier(), mode="short", polish=False)
        assert not report.ok

    def test_terminos_de_largo_son_demas_para_short(self):
        report = Screenwriter(ScriptedLLM(responses=[
            respuesta_short().replace('"data stream tunnel"',
                                      '"a", "b", "c", "d", "e", "f", "g"'),
            respuesta_short()])).write(dossier(), mode="short", polish=False)
        assert report.ok and len(report.attempts) == 2

    def test_prompt_pide_bucle_y_franja(self):
        texto = build_prompt(dossier(), None, "short")
        minimo, maximo = BANDS["short"]
        assert "reconecta" in texto
        assert str(minimo) in texto and str(maximo) in texto
        assert "25s" in texto      # el ejemplo de tamaño, ya no 15s
        assert "implicacion" in texto

    def test_modo_desconocido_falla_pronto(self):
        import pytest
        with pytest.raises(ValueError):
            Screenwriter(ScriptedLLM(responses=[])).write(dossier(), mode="tv")

    def test_duracion_medida_por_formato(self):
        """La franja del corto viene de `models`, no de un número clavado en el juez.

        Era `(10, 20)` escrito dentro del juez y `(30, 50)` palabras dentro del
        guionista: dos sitios para el mismo hecho. Cuando la parrilla pidió 25s,
        solo uno se actualizó y el juez pasó a reprobar todo corto que el
        guionista aprobaba -- con el mensaje genérico "ningún formato aprobado
        por el juez", que no apunta a ningún sitio.
        """
        corta = Script(topic="Tema corto de prueba", hook="h " * 7, body="b " * 50,
                       closing="c " * 7,
                       search_terms=["neural network nodes", "data stream tunnel",
                                     "abstract digital plexus"],
                       format="short")
        (duracion, _) = _notas_medidas(corta)
        assert duracion.score == 2, duracion.reason
        assert f"{SHORT_MIN_DURATION_S}-{SHORT_MAX_DURATION_S}s" in duracion.reason
