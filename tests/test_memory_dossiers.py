"""Testes da memoria de dossies.

O dossie e gravado com o custo medido e com os fatos que os portoes derrubaram.
A regra e a mesma do ledger de temas do M2: sem o descartado, so se sabe o que
entrou, nunca o que foi perdido -- e calibrar portao vira chute.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent.memory.store import SignalStore
from agent.models import Dossier, Fact

AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path) -> SignalStore:
    return SignalStore(tmp_path / "agent.db")


def dossie(topic: str = "Bonsai 2 27B", claim: str = "O modelo ocupa 5,9 GB") -> Dossier:
    return Dossier(
        topic=topic,
        facts=[Fact(
            claim=claim,
            source_url="https://prismml.com/news/bonsai-2-27b",
            source_name="PrismML",
            quote="that occupies 5.9 GB on disk",
        )],
        collected_at=AGORA,
    )


class TestGravacao:
    def test_grava_e_devolve_o_id(self, store):
        linha = store.record_dossier(
            dossie(), model="gemini-2.5-flash", provider="gemini",
            usage=(1200, 80), latency_s=3.4, source_count=2,
        )
        assert linha > 0
        assert store.dossier_count() == 1

    def test_custo_medido_fica_consultavel_por_sql(self, store):
        """O eval do M5 compara provedores por consumo: o numero precisa estar
        em coluna, nao enterrado no JSON."""
        store.record_dossier(
            dossie(), model="openai/gpt-oss-120b", provider="groq",
            usage=(900, 40), latency_s=1.2, source_count=3,
        )
        with store._conn() as conn:
            row = conn.execute(
                "SELECT provider, input_tokens, output_tokens, latency_s, source_count"
                " FROM dossiers"
            ).fetchone()
        assert row["provider"] == "groq"
        assert (row["input_tokens"], row["output_tokens"]) == (900, 40)
        assert row["latency_s"] == 1.2
        assert row["source_count"] == 3

    def test_descartado_e_gravado_junto(self, store):
        store.record_dossier(
            dossie(), model="m", provider="p", usage=(1, 1), latency_s=0.1, source_count=1,
            discarded=[{"claim": "treinado com 12000 GPUs",
                        "reason": "numero sem respaldo na fonte: 12000",
                        "source_url": "https://prismml.com/news/bonsai-2-27b"}],
            failures={"https://paywall.com/x": "HTTP 403"},
        )
        with store._conn() as conn:
            row = conn.execute("SELECT discarded_json, failures_json FROM dossiers").fetchone()
        assert "12000" in row["discarded_json"]
        assert "403" in row["failures_json"]


class TestLeitura:
    def test_dossie_volta_identico_ao_gravado(self, store):
        """Guardar o contrato serializado e o que mantem o dossie reproduzivel
        palavra por palavra -- o juiz do M5 rele o mesmo material."""
        original = dossie()
        store.record_dossier(
            original, model="m", provider="p", usage=(1, 1), latency_s=0.1, source_count=1,
        )
        lido = store.latest_dossier()
        assert lido.topic == original.topic
        assert lido.facts[0].claim == original.facts[0].claim
        assert str(lido.facts[0].source_url) == str(original.facts[0].source_url)
        assert lido.facts[0].quote == original.facts[0].quote

    def test_mais_recente_do_tema_pedido(self, store):
        for claim in ("O modelo ocupa 5,9 GB", "O modelo retem 98,2% da nota"):
            store.record_dossier(
                dossie(claim=claim), model="m", provider="p",
                usage=(1, 1), latency_s=0.1, source_count=1,
            )
        store.record_dossier(
            dossie(topic="Outro tema"), model="m", provider="p",
            usage=(1, 1), latency_s=0.1, source_count=1,
        )
        assert store.latest_dossier("Bonsai 2 27B").facts[0].claim.startswith("O modelo retem")
        assert store.latest_dossier("Outro tema").topic == "Outro tema"

    def test_memoria_vazia_devolve_none(self, store):
        assert store.latest_dossier() is None
        assert store.latest_dossier("tema que nunca existiu") is None


def test_tabela_nova_nao_quebra_banco_do_m2(tmp_path):
    """SignalStore aplica o schema em cada abertura: um agent.db criado no M2
    precisa ganhar a tabela de dossies sem migracao na mao."""
    caminho = tmp_path / "agent.db"
    SignalStore(caminho)
    segunda = SignalStore(caminho)
    assert segunda.dossier_count() == 0
