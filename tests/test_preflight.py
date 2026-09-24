"""Preflight: la ultima puerta comprueba eslabones, no re-juzga texto."""

from __future__ import annotations

from datetime import datetime

from agent.paths import run_dir, slugify
from agent.publish.preflight import preflight_carousel, preflight_video


def _guion(palabras: int = 180) -> dict:
    return {"topic": "Bonsai 27B", "hook": "h", "body": " ".join(["b"] * palabras),
            "closing": "c", "facts": [{"claim": "x"}], "format": "long"}


def _linea(id: int = 7, topic: str = "Bonsai 27B", words: int = 182) -> dict:
    return {"id": id, "topic": topic, "word_count": words}


def _informe(id: int = 9, script_id: int = 7, ok: bool = True) -> dict:
    return {"id": id, "topic": "Bonsai 27B", "script_id": script_id,
            "model": "m", "approved": ok}


def _probe(**kw) -> dict:
    base = {"width": 1080, "height": 1920, "has_audio": True, "duration_s": 69.4}
    base.update(kw)
    return base


class TestVideo:
    def test_paquete_completo_aprueba(self):
        # hook(1) + body(180) + closing(1) = 182
        r = preflight_video(_guion(), [_linea()], [_informe()], _probe())
        assert r.approved

    def test_sin_informe_enlazado_reprueba(self):
        r = preflight_video(_guion(), [_linea()], [_informe(ok=False)], _probe())
        assert not r.approved
        assert any("informe" in g.label for g in r.gates if not g.passed)

    def test_guion_reescrito_invalida_informe_viejo(self):
        r = preflight_video(_guion(palabras=190), [_linea()], [_informe()],
                            _probe())
        assert not r.approved

    def test_mp4_mudo_y_dimension_reprueban(self):
        r = preflight_video(_guion(), [_linea()], [_informe()],
                            _probe(has_audio=False, width=720, height=1280,
                                   duration_s=69.4))
        assert not r.approved
        assert sum(1 for g in r.gates if not g.passed) == 2

    def test_sin_mp4_reprueba_sin_romper(self):
        r = preflight_video(_guion(), [_linea()], [_informe()], None)
        assert not r.approved


class TestCarrusel:
    def test_aprobado_con_slides(self):
        raw = {"topic": "T", "facts": [{"claim": "x"}]}
        assert preflight_carousel(raw, True, True).approved

    def test_sin_informe_o_slides_reprueba(self):
        raw = {"topic": "T", "facts": []}
        r = preflight_carousel(raw, False, False)
        assert not r.approved and len(r.gates) == 3


class TestPaths:
    def test_slug_y_directorio(self):
        assert slugify("Bonsai 2 27B: modelo!") == "bonsai-2-27b-modelo"
        d = run_dir("short", "Bonsai 2 27B", ahora=datetime(2026, 9, 19, 14, 30))
        assert str(d) == "output/2026-09-19/1430-bonsai-2-27b-short"
