"""Tests de los contratos entre etapas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.models import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    Dossier,
    RenderResult,
    RenderState,
    Script,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "guion_manual.json"


def load_fixture() -> Script:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return Script.model_validate(raw)


class TestScript:
    def test_fixture_es_valido(self):
        script = load_fixture()
        assert script.topic
        assert script.facts, "el fixture necesita cargar hechos con fuente"

    def test_fixture_cabe_en_la_franja_monetizable(self):
        """La estimación pre-TTS debe caer en la franja del Creator Rewards.

        No sustituye la medición del MP4 -- solo evita mandar a render un
        guion que ya nace fuera de la franja.
        """
        script = load_fixture()
        assert MIN_DURATION_S <= script.estimated_duration_s <= MAX_DURATION_S, (
            f"{script.word_count} palabras => ~{script.estimated_duration_s:.0f}s, "
            f"fuera de {MIN_DURATION_S}-{MAX_DURATION_S}s"
        )

    def test_narracion_usa_castellano_acentuado(self):
        """El fixture es el modelo de lo que el guionista debe generar en el M3.

        Castellano sin acento aparece en la leyenda palabra por palabra y parece
        descuido en un canal. La fuente (BeVietnamPro-Bold) cubre todos los
        acentos, así que no hay excusa técnica para escribir sin ellos.
        """
        import unicodedata

        narracion = load_fixture().narration
        acentos = sum(
            1 for c in unicodedata.normalize("NFD", narracion) if unicodedata.combining(c)
        )
        assert acentos >= 20, f"solo {acentos} acentos en {len(narracion)} caracteres"

    def test_narracion_preserva_el_orden(self):
        script = load_fixture()
        narration = script.narration
        assert narration.index(script.hook) < narration.index(script.closing)

    def test_termino_de_busqueda_con_acento_es_rechazado(self):
        """Término en pt-BR va directo a Pexels y devuelve resultado vacío.

        Fallar en la validación cuesta milisegundos; descubrirlo después cuesta
        un render entero con material equivocado.
        """
        with pytest.raises(ValidationError, match="ASCII"):
            Script(
                topic="prueba",
                hook="un hook cualquiera con tamaño suficiente",
                body="un cuerpo de guion con al menos cincuenta caracteres para pasar",
                closing="un cierre cualquiera",
                search_terms=["placa de vídeo", "data center", "código"],
            )

    def test_pocos_terminos_son_rechazados(self):
        with pytest.raises(ValidationError):
            Script(
                topic="prueba",
                hook="un hook cualquiera con tamaño suficiente",
                body="un cuerpo de guion con al menos cincuenta caracteres para pasar",
                closing="un cierre cualquiera",
                search_terms=["only one"],
            )

    def test_verificacion_de_fuente_no_es_del_modelo(self):
        """Casar afirmación con fuente exige juicio semántico, no string match."""
        with pytest.raises(NotImplementedError):
            _ = load_fixture().unsourced


class TestDossier:
    def test_dossier_sin_hecho_es_rechazado(self):
        from datetime import datetime

        with pytest.raises(ValidationError, match="sin hecho"):
            Dossier(topic="t", facts=[], collected_at=datetime.now())

    def test_hecho_sin_url_es_rechazado(self):
        from datetime import datetime

        with pytest.raises(ValidationError):
            Dossier.model_validate(
                {
                    "topic": "t",
                    "collected_at": datetime.now(),
                    "facts": [{"claim": "una afirmación cualquiera", "source_name": "Fuente"}],
                }
            )


class TestRenderResult:
    def test_completo_sin_camino_es_incoherente(self):
        with pytest.raises(ValidationError, match="sin video_path"):
            RenderResult(state=RenderState.complete)

    def test_fallo_sin_motivo_es_incoherente(self):
        with pytest.raises(ValidationError, match="sin mensaje de error"):
            RenderResult(state=RenderState.failed)

    def test_aceptacion_del_m0(self):
        ok = RenderResult(
            state=RenderState.complete,
            video_path="/tmp/x.mp4",
            width=1080,
            height=1920,
            duration_s=78.4,
            has_audio=True,
        )
        assert ok.is_portrait_1080x1920
        assert ok.duration_in_monetizable_range
        assert ok.has_audio

    def test_video_mudo_no_es_aceptacion(self):
        """Regresión de un caso real: el primer render salió sin pista de audio
        y pasó las comprobaciones de dimensión y duración. `has_audio` es falso
        por defecto justamente para que el silencio nunca sea el estado
        aprobado por defecto."""
        mudo = RenderResult(
            state=RenderState.complete,
            video_path="/tmp/x.mp4",
            width=1080,
            height=1920,
            duration_s=78.4,
        )
        assert mudo.is_portrait_1080x1920
        assert mudo.duration_in_monetizable_range
        assert not mudo.has_audio

    def test_video_demasiado_corto_reprueba(self):
        """59s no es elegible al Creator Rewards, por más bonito que esté."""
        corto = RenderResult(
            state=RenderState.complete,
            video_path="/tmp/x.mp4",
            width=1080,
            height=1920,
            duration_s=59.0,
        )
        assert corto.is_portrait_1080x1920
        assert not corto.duration_in_monetizable_range

    def test_duracion_ausente_no_cuenta_como_aprobada(self):
        sin_medida = RenderResult(state=RenderState.complete, video_path="/tmp/x.mp4")
        assert not sin_medida.duration_in_monetizable_range
