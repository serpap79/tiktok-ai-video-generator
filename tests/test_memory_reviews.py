"""Tests de la memoria de informes.

Tabla propia, y no columnas en `scripts`, porque el mismo guion puede ser
juzgado mas de una vez: el eval del M5 compara jueces de modelos diferentes sobre
el MISMO texto, y eso es una relacion de uno a muchos.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent.memory.store import SignalStore
from agent.models import Criterion, CriterionScore, Review

AHORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path) -> SignalStore:
    return SignalStore(tmp_path / "agent.db")


def informe(nota_hook: int = 2, model: str = "gemini-2.5-flash",
            provider: str = "gemini") -> Review:
    notas = [
        CriterionScore(criterion=c, score=2, reason=f"sin salvedad en {c.value}")
        for c in Criterion if c not in (Criterion.hook, Criterion.flujo)
    ]
    notas.append(CriterionScore(
        criterion=Criterion.hook, score=nota_hook, reason="el hook entrega el asunto",
    ))
    return Review(topic="Bonsai 2 27B", scores=notas, reviewed_at=AHORA,
                  model=model, provider=provider)


class TestGrabacion:
    def test_el_informe_vuelve_con_nota_y_motivo_de_cada_criterio(self, store):
        store.record_review(informe(), usage=(1500, 200), latency_s=2.2)
        leido = store.latest_review()
        assert set(leido.by_criterion) == set(Criterion) - {Criterion.flujo}
        assert leido.by_criterion[Criterion.hook].reason == "el hook entrega el asunto"
        assert leido.total == informe().total

    def test_suma_y_aprobacion_se_quedan_en_columna(self, store):
        """Agregar por SQL sin abrir el JSON de cada informe."""
        store.record_review(informe(nota_hook=0), usage=(1, 1), latency_s=0.1)
        with store._conn() as conn:
            row = conn.execute("SELECT total, approved, provider FROM reviews").fetchone()
        assert row["total"] == 12
        assert row["approved"] == 0
        assert row["provider"] == "gemini"

    def test_dos_informes_para_el_mismo_guion(self, store):
        """Y es el formato del eval del M5: jueces distintos, texto identico."""
        from agent.models import Fact, Script

        script_id = store.record_script(
            Script(
                topic="Bonsai 2 27B",
                hook="Un modelo enorme cabe en un pendrive.",
                body="Cuerpo del guion con tamano suficiente para el contrato del modelo.",
                closing="En que numero confiaras hoy?",
                search_terms=["memory chip", "server rack", "binary code"],
                facts=[Fact(claim="Ocupa 5,9 GB, mas de 9 veces menor",
                            source_url="https://prismml.com/x", source_name="PrismML")],
            ),
            model="m", provider="p", usage=(1, 1), latency_s=0.1,
            attempts=[{"violations": [], "word_count": 180}],
        )
        store.record_review(informe(model="gemini-2.5-flash", provider="gemini"),
                            usage=(1, 1), latency_s=0.1, script_id=script_id)
        store.record_review(informe(nota_hook=1, model="llama-3.3-70b", provider="groq"),
                            usage=(1, 1), latency_s=0.1, script_id=script_id)

        with store._conn() as conn:
            filas = conn.execute(
                "SELECT provider, total FROM reviews WHERE script_id = ? ORDER BY id",
                (script_id,),
            ).fetchall()
        assert [(r["provider"], r["total"]) for r in filas] == [("gemini", 14), ("groq", 13)]
        assert store.review_count() == 2


class TestLectura:
    def test_el_mas_reciente_por_tema(self, store):
        store.record_review(informe(nota_hook=0), usage=(1, 1), latency_s=0.1)
        store.record_review(informe(nota_hook=2), usage=(1, 1), latency_s=0.1)
        assert store.latest_review("Bonsai 2 27B").approved

    def test_memoria_vacia_devuelve_none(self, store):
        assert store.latest_review() is None
        assert store.review_count() == 0


def test_informe_interrumpido_es_identificable_despues(store):
    """El eval del M5 necesita filtrar informe interrumpido antes de agregar nota:
    4/14 de un guion reprobado en la medida no es comparable con 4/14 juzgado."""
    notas = [
        CriterionScore(criterion=Criterion.duracion, score=2, reason="210 palabras"),
        CriterionScore(criterion=Criterion.politica, score=2, reason="sin termino vetado"),
        CriterionScore(criterion=Criterion.fuente, score=0, reason="numero 12 sin respaldo",
                       measured=True),
    ]
    notas += [
        CriterionScore(criterion=c, score=0, reason="no evaluado: reprobo antes",
                       evaluated=False)
        for c in (Criterion.hook, Criterion.punto_de_vista, Criterion.es_es, Criterion.cta)
    ]
    store.record_review(
        Review(topic="t", scores=notas, reviewed_at=AHORA), usage=(0, 0), latency_s=0.0
    )
    leido = store.latest_review()
    assert leido.short_circuited
    assert leido.total == 4
    assert leido.revision_notes == ["[fuente 0/2] numero 12 sin respaldo"]
