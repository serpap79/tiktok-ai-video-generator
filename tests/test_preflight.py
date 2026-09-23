"""Preflight: a ultima porta confere elos, nao rejulga texto."""

from __future__ import annotations

from datetime import datetime

from agent.paths import run_dir, slugify
from agent.publish.preflight import preflight_carousel, preflight_video


def _roteiro(palavras: int = 180) -> dict:
    return {"topic": "Bonsai 27B", "hook": "h", "body": " ".join(["b"] * palavras),
            "closing": "c", "facts": [{"claim": "x"}], "format": "long"}


def _linha(id: int = 7, topic: str = "Bonsai 27B", words: int = 182) -> dict:
    return {"id": id, "topic": topic, "word_count": words}


def _parecer(id: int = 9, script_id: int = 7, ok: bool = True) -> dict:
    return {"id": id, "topic": "Bonsai 27B", "script_id": script_id,
            "model": "m", "approved": ok}


def _probe(**kw) -> dict:
    base = {"width": 1080, "height": 1920, "has_audio": True, "duration_s": 69.4}
    base.update(kw)
    return base


class TestVideo:
    def test_pacote_completo_aprova(self):
        # hook(1) + body(180) + closing(1) = 182
        r = preflight_video(_roteiro(), [_linha()], [_parecer()], _probe())
        assert r.approved

    def test_sem_parecer_ligado_reprova(self):
        r = preflight_video(_roteiro(), [_linha()], [_parecer(ok=False)], _probe())
        assert not r.approved
        assert any("parecer" in g.label for g in r.gates if not g.passed)

    def test_roteiro_reescrito_invalida_parecer_velho(self):
        r = preflight_video(_roteiro(palavras=190), [_linha()], [_parecer()],
                            _probe())
        assert not r.approved

    def test_mp4_mudo_e_dimensao_reprovam(self):
        r = preflight_video(_roteiro(), [_linha()], [_parecer()],
                            _probe(has_audio=False, width=720, height=1280,
                                   duration_s=69.4))
        assert not r.approved
        assert sum(1 for g in r.gates if not g.passed) == 2

    def test_sem_mp4_reprova_sem_quebrar(self):
        r = preflight_video(_roteiro(), [_linha()], [_parecer()], None)
        assert not r.approved


class TestCarrossel:
    def test_aprovado_com_slides(self):
        raw = {"topic": "T", "facts": [{"claim": "x"}]}
        assert preflight_carousel(raw, True, True).approved

    def test_sem_parecer_ou_slides_reprova(self):
        raw = {"topic": "T", "facts": []}
        r = preflight_carousel(raw, False, False)
        assert not r.approved and len(r.gates) == 3


class TestPaths:
    def test_slug_e_diretorio(self):
        assert slugify("Bonsai 2 27B: modelo!") == "bonsai-2-27b-modelo"
        d = run_dir("short", "Bonsai 2 27B", agora=datetime(2026, 9, 19, 14, 30))
        assert str(d) == "output/2026-09-19/1430-bonsai-2-27b-short"
