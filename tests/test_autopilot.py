"""Piloto automático: un slot completo con piezas falsas -- sin red, sin render.

Traba lo que importa para que el día corra solo: idempotencia (slot publicado
no se convierte en dos posts), caída al siguiente tema/formato, espera por la
hora, registro del motivo, y el slot que falla sin derribar el proceso.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.autopilot.runner import SlotRunner
from agent.autopilot.runs import SlotRuns
from agent.config import Settings
from agent.editorial.slots import SLOTS, TZ
from agent.models import Decision, Dossier, Fact, PublishResult, PublishState, Verdict
from agent.research.researcher import ResearchReport
from tests.test_writer import respuesta

DIA = date(2026, 9, 20)


def hechos(n: int) -> list[Fact]:
    return [Fact(claim=f"El Bonsai 27B tiene el dato número {i} medido por la fuente",
                 source_url=f"https://fuente{i % 2}.com/a", source_name="fuente",
                 quote=f"the Bonsai 27B has the measured figure number {i}")
            for i in range(n)]


def dictamen_ok() -> str:
    from agent.judge.judge import JULGADOS
    return json.dumps({c.value: {"reason": "ok", "score": 2} for c in JULGADOS})


class RunnerFalso(SlotRunner):
    """Radar, investigación, LLM, render y publicación sustituidos por fakes."""

    def __init__(self, tmp: Path, *, temas: list[str], n_hechos: int = 6,
                 guiones: list[str] | None = None, ahora: datetime | None = None,
                 publicacion: PublishResult | None = None,
                 formatos: dict[str, tuple[str, ...]] | None = None):
        cfg = Settings(_env_file=None, data_dir=tmp / "data", output_dir=tmp / "out",
                       groq_api_key="", gemini_api_key="")
        self.dormidas: list[float] = []
        super().__init__(cfg, now=lambda: ahora or datetime(2026, 9, 20, 19, 30, tzinfo=TZ),
                         sleep=self.dormidas.append, log=lambda m: None,
                         formatos=formatos)
        self._temas = temas
        self._n_hechos = n_hechos
        self._guiones = list(guiones or [])
        self._publicacion = publicacion
        self.publicados: list[Path] = []

    def candidates(self, usados):
        return [Decision(term=t, source="hacker_news", verdict=Verdict.not_selected,
                         reason="prueba", score=0.9 - i * 0.1, niche_fit=0.8,
                         decided_at=datetime.now(UTC)) for i, t in enumerate(self._temas)
                if t not in usados["topics"]]

    def research(self, decision, llm):
        rep = ResearchReport(topic=decision.term)
        if self._n_hechos:
            rep.dossier = Dossier(topic=decision.term, facts=hechos(self._n_hechos),
                                  collected_at=datetime.now(UTC))
        return rep

    def llm(self, stage):
        if stage in ("writer", "writer_short"):
            return ScriptedLLM(responses=self._guiones, provider="groq")
        if stage == "ranker":
            return ScriptedLLM(responder=lambda _p: '{"notas": []}', provider="groq")
        return ScriptedLLM(responder=lambda _p: dictamen_ok(), provider="gemini")

    def render_video(self, script, carpeta, pillar):
        video = carpeta / "video.mp4"
        video.write_bytes(b"mp4")
        duracion = 70.0 if script.format == "long" else 18.0
        return video, {"width": 1080, "height": 1920, "duration_s": duracion,
                       "has_audio": True}

    def render_carousel(self, carrusel, carpeta, pillar):
        from PIL import Image, ImageDraw

        carpeta.mkdir(parents=True, exist_ok=True)
        slides = []
        for i in range(1, 6):
            slide = carpeta / f"slide-{i}.png"
            img = Image.new("RGB", (1080, 1920), (20, 20, 30))
            # Franja clara en la zona del título (y 960-1440, x 70-1010): la
            # aceptación de slides mide tinta clara allí
            # (`render/carousel.py:ink_height`).
            ImageDraw.Draw(img).rectangle([80, 1000, 1000, 1300], fill=(255, 255, 255))
            img.save(slide)
            slides.append(slide)
        (carpeta / "caption.txt").write_text("caption de prueba #ia #tech\n",
                                             encoding="utf-8")
        return slides

    def publish_video(self, video, paquete, slot):
        self.publicados.append(video)
        return self._publicacion or PublishResult(state=PublishState.uploaded,
                                                  publish_id="v_inbox_prueba",
                                                  video_path=str(video))


@pytest.fixture(autouse=True)
def sin_aviso(monkeypatch):
    monkeypatch.setattr("agent.autopilot.runner.notify", lambda *a, **k: [])


def guion_largo() -> str:
    return respuesta(total=190)


class TestSlotCompleto:
    def test_produce_espera_la_hora_y_publica(self, tmp_path):
        from tests.test_short import respuesta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"],
                        guiones=[respuesta_short()] * 3,
                        ahora=datetime(2026, 9, 20, 16, 30, tzinfo=TZ))
        res = r.run(SLOTS["1700"], DIA)
        assert res.state == "published" and res.publish_id == "v_inbox_prueba"
        assert res.format == "short"
        # Produjo a las 16:30 y durmió hasta las 17:00 para publicar.
        assert r.dormidas and r.dormidas[0] == pytest.approx(30 * 60)
        linea = SlotRuns(r.cfg.db_path).get(DIA.isoformat(), "1700")
        plan = json.loads(linea["plan_json"])
        assert linea["state"] == "published"
        assert "format_reason" in plan and "topic_reason" in plan
        assert Path(linea["package_dir"], "caption.txt").exists()

    def test_slot_publicado_no_se_convierte_en_dos_posts(self, tmp_path):
        from tests.test_short import respuesta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"],
                        guiones=[respuesta_short()] * 6)
        r.run(SLOTS["1700"], DIA, wait=False)
        de_nuevo = r.run(SLOTS["1700"], DIA, wait=False)
        assert de_nuevo.state == "skipped" and len(r.publicados) == 1

    def test_tema_sin_dossier_cae_al_siguiente(self, tmp_path):
        from tests.test_short import respuesta_short

        class Selectivo(RunnerFalso):
            def research(self, decision, llm):
                if decision.term.startswith("Sin"):
                    return ResearchReport(topic=decision.term)
                return super().research(decision, llm)

        r = Selectivo(tmp_path, temas=["Sin fuente ninguna", "Bonsai 27B cabe en el bolsillo"],
                      guiones=[respuesta_short()] * 3)
        res = r.run(SLOTS["1700"], DIA, wait=False)
        assert res.state == "published" and res.topic.startswith("Bonsai")

    def test_dossier_fino_se_vuelve_corto_y_no_largo(self, tmp_path):
        from tests.test_short import respuesta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"], n_hechos=2,
                        guiones=[respuesta_short()] * 3)
        res = r.run(SLOTS["2000"], DIA, wait=False)
        assert res.format == "short"

    def test_la_noche_sale_video_largo_y_no_carrusel(self, tmp_path):
        """La parrilla de 20/09/2026 sacó el carrusel de las horas fijas.

        Hasta la tarde de ese día las 20h eran carrusel (paquete manual). Con
        cuatro posts al día y todos vídeo, la noche se volvió el largo -- que
        es el formato que monetiza. El carrusel sigue implementado y sale por
        `slot-extra --format carrusel`.
        """
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"], n_hechos=4,
                        guiones=[guion_largo()] * 3)
        res = r.run(SLOTS["2000"], DIA, wait=False)
        assert res.format == "long"
        assert res.state != "ready_manual"

    def test_carrusel_sigue_alcanzable_fuera_de_la_parrilla(self, tmp_path):
        """Sacarlo de la parrilla no es eliminarlo: el formato debe seguir llamable."""
        from datetime import time as _time

        from agent.editorial.slots import Slot
        from tests.test_carousel import dictamen as dictamen_carrusel
        from tests.test_carousel import slides

        class Manual(RunnerFalso):
            def llm(self, stage):
                if stage == "judge":
                    return ScriptedLLM(responder=lambda _p: dictamen_carrusel(),
                                       provider="groq")
                return super().llm(stage)

        r = Manual(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"],
                   formatos={"extra": ("carousel",)},
                   guiones=[slides(
                       headlines=["5 datos del Bonsai en disco", "Pesa poco",
                                  "Rinde mucho", "Funciona rápido", "Guárdalo para después"],
                       texts=["Desliza y mira", "Ocupa 2 partes del disco",
                              "Mantiene 4 de 5 puntos", "Llega a 3 veces más",
                              "Guarda este resumen para repasarlo"])] * 3)
        extra = Slot("extra", _time(12, 0), "extra", "ronda manual")
        res = r.run(extra, DIA, wait=False)
        assert res.format == "carousel" and res.state == "ready_manual"

    def test_fallo_de_publicacion_queda_registrado(self, tmp_path):
        from tests.test_short import respuesta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"],
                        guiones=[respuesta_short()] * 3,
                        publicacion=PublishResult(state=PublishState.failed,
                                                  error="access_token_invalid"))
        res = r.run(SLOTS["1700"], DIA, wait=False)
        assert res.state == "failed" and "access_token_invalid" in res.error
        assert SlotRuns(r.cfg.db_path).get(DIA.isoformat(), "1700")["state"] == "failed"

    def test_sin_ningun_tema_falla_con_motivo_y_sin_excepcion(self, tmp_path):
        r = RunnerFalso(tmp_path, temas=[])
        res = r.run(SLOTS["1000"], DIA, wait=False)
        assert res.state == "failed" and "ningún tema" in res.error

    def test_slot_muy_atrasado_se_salta(self, tmp_path):
        """Máquina encendida a las 14h: el slot de las 10h no sale pegado al de las 17h."""
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"],
                        ahora=datetime(2026, 9, 20, 14, 0, tzinfo=TZ))
        res = r.run(SLOTS["1000"], DIA)
        assert res.state == "skipped" and "después de la hora" in res.error

    def test_sin_publicar_deja_el_paquete_listo(self, tmp_path):
        from tests.test_short import respuesta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"],
                        guiones=[respuesta_short()] * 3)
        res = r.run(SLOTS["1700"], DIA, publish=False, wait=False)
        assert res.state == "produced" and not r.publicados

    def test_dia_varia_el_formato(self, tmp_path):
        """El segundo slot del día ve el formato del primero y pierde puntos en él."""
        runs = SlotRuns(tmp_path / "x.db")
        runs.begin("2026-09-20", "1000")
        runs.update("2026-09-20", "1000", state="published", format="short",
                    pillar="news", topic="Tema A")
        usados = runs.used_today("2026-09-20", except_slot="1700")
        assert usados == {"formats": ["short"], "pillars": ["news"], "topics": ["Tema A"]}

    def test_slot_extra_corre_fuera_de_la_parrilla(self, tmp_path):
        """Ronda extra: fila propia en el día, sin tocar los slots de los timers."""
        from datetime import time

        from agent.editorial.slots import Slot
        from tests.test_short import respuesta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"],
                        guiones=[respuesta_short()] * 3)
        res = r.run(Slot("extra", time(10, 45), "extra", "prueba"), DIA, wait=False)
        assert res.state == "published" and res.slot == "extra"
        linea = SlotRuns(r.cfg.db_path).get(DIA.isoformat(), "extra")
        assert linea["state"] == "published"
        assert SlotRunner._hora(SLOTS["1000"]) == "10h"
        assert SlotRunner._hora(
            Slot("extra", time(10, 45), "extra", "prueba")) == "extra"

    def test_formatos_fuera_de_la_parrilla_respetan_orden(self, tmp_path):
        """`slot-extra --format video` prueba long antes que short."""
        from datetime import time

        from agent.editorial.slots import Slot
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B cabe en el bolsillo"],
                        guiones=[guion_largo()] * 3,
                        formatos={"extra": ("long", "short")})
        res = r.run(Slot("extra", time(10, 45), "extra", "prueba"), DIA, wait=False)
        assert res.state == "published" and res.format == "long"


class TestPresentadorEnSlot:
    """El reparto y el acento salen del ID del pilar, no de su objeto.

    Bug pagado en 20/09/2026: el runner reasignaba `pilar` al objeto
    `ContentPillar` y luego llamaba `presenter_for(pilar)` y
    `accent_for(pilar)`. Los dos reciben un id (string), así que caían en el
    defecto en silencio -- ningún presentador entraba y todo vídeo salía en
    verde, incluso en los pilares de acento cian. Ningún test lo pillaba
    porque los dos caminos degradan sin error.
    """

    def test_el_pilar_correcto_trae_el_presentador_correcto(self, tmp_path):
        from agent.brand.brand import load as load_brand

        runner = RunnerFalso(tmp_path, temas=["tema"])
        marca = load_brand()
        for pilar in ("analisis", "tutorial", "futuro", "historia"):
            pedido = runner._pedido_presentador(marca, pilar)
            assert pedido is not None and pedido.name == "Atlas", pilar

    def test_pilar_sin_clip_base_en_el_repo_pide_el_retrato(self, tmp_path):
        """Sin `theo_base.mp4` en el repo, el pedido cae al retrato parado.

        Mide el PEDIDO, no el código de salida: si `base` llega vacío el vídeo
        sale con el presentador sintetizado -- y el motivo queda en el log, no
        en silencio.
        """
        from agent.brand.brand import load as load_brand

        runner = RunnerFalso(tmp_path, temas=["tema"])
        pedido = runner._pedido_presentador(load_brand(), "news")
        assert pedido is not None
        hay_clip = pedido.base is not None
        if hay_clip:
            assert pedido.base.name == "theo_base.json"
            assert pedido.base.with_suffix(".mp4").exists()
        else:
            # Reserva: retrato parado sintetizado, con su PNG y sus metadatos.
            assert pedido.cutout is not None and pedido.cutout.name == "theo.png"

    def test_objeto_del_pilar_en_lugar_del_id_no_resuelve_nada(self):
        """El síntoma exacto del bug, trabado: objeto en lugar del id devuelve None."""
        from agent.brand.brand import load as load_brand

        marca = load_brand()
        assert marca.presenter_for("analisis") is not None
        assert marca.presenter_for(marca.pillars["analisis"]) is None

    def test_acento_del_pilar_de_ruptura_no_es_el_verde_por_defecto(self):
        from agent.brand.brand import load as load_brand

        marca = load_brand()
        assert marca.accent_for("dato") != marca.accent_for("news")
        assert marca.accent_for(marca.pillars["dato"]) == marca.accent_primary
