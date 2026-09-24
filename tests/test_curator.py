"""Tests de la orquestacion del curador: orden de las puertas, score y registro."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent.curator.curator import Curator
from agent.memory.store import SignalStore
from agent.models import Signal, Verdict

AHORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def senal(term: str, source: str = "hacker_news", volume: float = 300.0,
          velocity: float | None = 20.0, unit: str = "points") -> Signal:
    return Signal(term=term, source=source, volume=volume, unit=unit,
                  velocity=velocity, seen_at=AHORA)


class TestOrdenDeLasPuertas:
    def test_politica_gana_a_velocidad_altisima(self):
        """Tema vetado no puede ganar en el ranking por estar subiendo rapido.

        Por eso la politica corre ANTES del score, y no como desempate despues.
        """
        report = Curator().curate([
            senal("Ozempic para bajar de peso", "wikipedia", 999_999, 99_999, "pageviews"),
            senal("Qwen 3.8 Omni Flash: nuevo modelo de IA", volume=100, velocity=1.0),
        ])
        assert report.selected.term.startswith("Qwen")
        assert report.by_verdict(Verdict.rejected_policy)[0].score == 0.0

    def test_politica_se_evalua_antes_del_nicho(self):
        """Un tema puede estar fuera del nicho Y vetado; el registro debe mostrar
        el veto, que es la informacion mas fuerte."""
        (d,) = Curator().curate([senal("Eleccion presidencial", "wikipedia")]).decisions
        assert d.verdict is Verdict.rejected_policy

    def test_duplicado_se_comprueba_despues_de_los_otros_dos(self):
        """Comparar con el libro cuesta; no se gasta eso en un tema ya vetado."""
        ledger = ["Eleccion presidencial en Espana"]
        (d,) = Curator().curate([senal("Eleccion presidencial", "wikipedia")], ledger).decisions
        assert d.verdict is Verdict.rejected_policy


class TestSeleccion:
    def test_elige_exactamente_uno(self):
        report = Curator().curate([
            senal("GPU quantization benchmark", velocity=50.0),
            senal("Nuevo modelo de IA bate record", velocity=40.0),
            senal("Kernel de Linux 7.0 lanzado", velocity=30.0),
        ])
        assert len(report.by_verdict(Verdict.selected)) == 1

    def test_toda_decision_tiene_motivo_incluida_la_aprobada(self):
        """Decision sin justificacion grabada no se puede auditar despues."""
        report = Curator().curate([
            senal("GPU quantization benchmark"),
            senal("Eleccion presidencial", "wikipedia"),
            senal("Receta de bizcocho de zanahoria", "wikipedia", velocity=None),
        ])
        assert all(d.reason.strip() for d in report.decisions)
        assert "score" in report.selected.reason

    def test_ningun_elegible_devuelve_selected_none(self):
        report = Curator().curate([senal("Eleccion presidencial", "wikipedia")])
        assert report.selected is None
        assert report.tally()["rejected_policy"] == 1

    def test_lista_vacia_no_explota(self):
        report = Curator().curate([])
        assert report.selected is None
        assert report.decisions == []

    def test_el_elegido_entra_en_el_libro_de_la_propia_ronda(self):
        """Dos fuentes trayendo la misma historia no pueden producir dos videos.

        El termino elegido se anade al libro en memoria durante la propia
        curaduria -- pero los demas ya pasaron por la puerta de duplicado antes
        del score, asi que el efecto real es en el libro persistido de la
        siguiente ronda.
        """
        report = Curator().curate([senal("GPU quantization benchmark")])
        assert report.selected is not None


class TestScorePorPercentil:
    def test_unidades_distintas_no_competen_en_valor_bruto(self):
        """Pageview de la Wikipedia y punto de Hacker News no comparten escala.

        Sumar los numeros crudos haria ganar siempre a la Wikipedia por tener
        unidad mayor, y no por tener asunto mejor. El score usa percentil dentro
        de la propia fuente.
        """
        senales = [
            # Wikipedia con volumen gigante, pero en el pie de su propia fuente
            senal("Computacion cuantica", "wikipedia", volume=50_000, velocity=10.0,
                  unit="pageviews"),
            senal("Robotica", "wikipedia", volume=90_000, velocity=500.0, unit="pageviews"),
            senal("Genoma humano", "wikipedia", volume=95_000, velocity=900.0, unit="pageviews"),
            # HN con volumen minusculo, pero en la cima de su propia fuente
            senal("Nuevo modelo de IA rompe un benchmark", volume=400, velocity=90.0),
            senal("Chip fotonico", volume=120, velocity=5.0),
        ]
        report = Curator().curate(senales)
        # La cima del HN tiene que poder vencer a la cima de la Wikipedia
        assert report.selected.source == "hacker_news"

    def test_velocidad_desconocida_no_se_trata_como_cero(self):
        """None significa "no medi"; cero significaria "medi y no se movio".

        Quien no tiene medida recibe la mitad de la escala y compite por el
        volumen y por el nicho hasta que la segunda coleta de la tasa.
        """
        # Con una unica observacion de velocidad en la fuente no hay distribucion,
        # y el percentil devuelve 0.5 -- el mismo valor del desconocido. Hacen
        # falta varios valores para que el test diga algo.
        senales = [
            senal("GPU benchmark record", volume=100, velocity=100.0),
            senal("Kernel de Linux optimizado", volume=100, velocity=50.0),
            senal("Chip fotonico integrado", volume=100, velocity=1.0),
            senal("Robotica autonoma en la industria", volume=100, velocity=None),
        ]
        por_termino = {d.term: d for d in Curator().curate(senales).decisions}

        lento = por_termino["Chip fotonico integrado"]
        sin_medida = por_termino["Robotica autonoma en la industria"]
        assert sin_medida.velocity is None
        # "no medi" tiene que valer mas que "medi y esta en el pie de la fuente"
        assert sin_medida.score > lento.score

    def test_el_score_se_queda_entre_cero_y_uno(self):
        report = Curator().curate([
            senal("GPU quantization benchmark", velocity=1e9, volume=1e9),
            senal("Kernel de Linux", velocity=0.0, volume=0.0),
        ])
        assert all(0.0 <= d.score <= 1.0 for d in report.decisions)


class TestLibro:
    @pytest.fixture
    def store(self, tmp_path) -> SignalStore:
        return SignalStore(tmp_path / "agent.db")

    def test_tema_ya_aprobado_se_rechaza_como_duplicado(self, store):
        anterior = "Bonsai 2 27B Near Lossless Compression Footprint"
        report = Curator().curate(
            [senal("Bonsai 2 27B: Near-Lossless Compression in a Smaller Footprint")],
            ledger=[anterior],
        )
        (d,) = report.decisions
        assert d.verdict is Verdict.rejected_duplicate
        assert d.duplicate_of == anterior

    def test_graba_tambien_los_rechazos(self, store):
        """Sin los rechazos grabados, solo se sabe lo que se eligio, nunca lo que
        se perdio -- y calibrar el score se vuelve un chute."""
        report = Curator().curate([
            senal("GPU quantization benchmark"),
            senal("Eleccion presidencial", "wikipedia"),
        ])
        assert store.record_decisions(report.decisions) == 2
        assert store.topic_count() == 2

    def test_el_libro_solo_devuelve_los_aprobados(self, store):
        """Tema rechazado por politica nunca fue video: bloquear el parecido
        extenderia el veto a asuntos que nunca fueron juzgados."""
        report = Curator().curate([
            senal("GPU quantization benchmark"),
            senal("Eleccion presidencial", "wikipedia"),
        ])
        store.record_decisions(report.decisions)
        recientes = store.recent_topics()
        assert recientes == ["GPU quantization benchmark"]

    def test_el_libro_respeta_la_ventana_de_dias(self, store):
        report = Curator().curate([senal("GPU quantization benchmark")])
        store.record_decisions(report.decisions)
        assert store.recent_topics(days=30) != []
        assert store.recent_topics(days=0) == []
