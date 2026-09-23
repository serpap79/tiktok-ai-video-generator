"""Testes da memoria de pareceres.

Tabela propria, e nao colunas em `scripts`, porque o mesmo roteiro pode ser
julgado mais de uma vez: o eval do M5 compara juizes de modelos diferentes sobre
o MESMO texto, e isso e uma relacao de um para muitos.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent.memory.store import SignalStore
from agent.models import Criterion, CriterionScore, Review

AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path) -> SignalStore:
    return SignalStore(tmp_path / "agent.db")


def parecer(nota_hook: int = 2, model: str = "gemini-2.5-flash",
            provider: str = "gemini") -> Review:
    notas = [
        CriterionScore(criterion=c, score=2, reason=f"sem ressalva em {c.value}")
        for c in Criterion if c not in (Criterion.hook, Criterion.fluxo)
    ]
    notas.append(CriterionScore(
        criterion=Criterion.hook, score=nota_hook, reason="o hook entrega o assunto",
    ))
    return Review(topic="Bonsai 2 27B", scores=notas, reviewed_at=AGORA,
                  model=model, provider=provider)


class TestGravacao:
    def test_parecer_volta_com_nota_e_motivo_de_cada_criterio(self, store):
        store.record_review(parecer(), usage=(1500, 200), latency_s=2.2)
        lido = store.latest_review()
        assert set(lido.by_criterion) == set(Criterion) - {Criterion.fluxo}
        assert lido.by_criterion[Criterion.hook].reason == "o hook entrega o assunto"
        assert lido.total == parecer().total

    def test_soma_e_aprovacao_ficam_em_coluna(self, store):
        """Agregar por SQL sem abrir o JSON de cada parecer."""
        store.record_review(parecer(nota_hook=0), usage=(1, 1), latency_s=0.1)
        with store._conn() as conn:
            row = conn.execute("SELECT total, approved, provider FROM reviews").fetchone()
        assert row["total"] == 12
        assert row["approved"] == 0
        assert row["provider"] == "gemini"

    def test_dois_pareceres_para_o_mesmo_roteiro(self, store):
        """E o formato do eval do M5: juizes diferentes, texto identico."""
        from agent.models import Fact, Script

        script_id = store.record_script(
            Script(
                topic="Bonsai 2 27B",
                hook="Um modelo enorme cabe num pendrive.",
                body="Corpo do roteiro com tamanho suficiente para o contrato do modelo.",
                closing="Que numero voce confere hoje?",
                search_terms=["memory chip", "server rack", "binary code"],
                facts=[Fact(claim="Ocupa 5,9 GB, mais de 9x menor",
                            source_url="https://prismml.com/x", source_name="PrismML")],
            ),
            model="m", provider="p", usage=(1, 1), latency_s=0.1,
            attempts=[{"violations": [], "word_count": 180}],
        )
        store.record_review(parecer(model="gemini-2.5-flash", provider="gemini"),
                            usage=(1, 1), latency_s=0.1, script_id=script_id)
        store.record_review(parecer(nota_hook=1, model="llama-3.3-70b", provider="groq"),
                            usage=(1, 1), latency_s=0.1, script_id=script_id)

        with store._conn() as conn:
            linhas = conn.execute(
                "SELECT provider, total FROM reviews WHERE script_id = ? ORDER BY id",
                (script_id,),
            ).fetchall()
        assert [(r["provider"], r["total"]) for r in linhas] == [("gemini", 14), ("groq", 13)]
        assert store.review_count() == 2


class TestLeitura:
    def test_mais_recente_por_tema(self, store):
        store.record_review(parecer(nota_hook=0), usage=(1, 1), latency_s=0.1)
        store.record_review(parecer(nota_hook=2), usage=(1, 1), latency_s=0.1)
        assert store.latest_review("Bonsai 2 27B").approved

    def test_memoria_vazia_devolve_none(self, store):
        assert store.latest_review() is None
        assert store.review_count() == 0


def test_parecer_interrompido_e_identificavel_depois(store):
    """O eval do M5 precisa filtrar parecer interrompido antes de agregar nota:
    4/14 de um roteiro reprovado na medida nao e comparavel com 4/14 julgado."""
    notas = [
        CriterionScore(criterion=Criterion.duracao, score=2, reason="210 palavras"),
        CriterionScore(criterion=Criterion.politica, score=2, reason="sem termo vetado"),
        CriterionScore(criterion=Criterion.fonte, score=0, reason="numero 12 sem respaldo",
                       measured=True),
    ]
    notas += [
        CriterionScore(criterion=c, score=0, reason="nao avaliado: reprovou antes",
                       evaluated=False)
        for c in (Criterion.hook, Criterion.ponto_de_vista, Criterion.pt_br, Criterion.cta)
    ]
    store.record_review(
        Review(topic="t", scores=notas, reviewed_at=AGORA), usage=(0, 0), latency_s=0.0
    )
    lido = store.latest_review()
    assert lido.short_circuited
    assert lido.total == 4
    assert lido.revision_notes == ["[fonte 0/2] numero 12 sem respaldo"]
