"""Testes da memoria de roteiros.

O roteiro e gravado com o custo, o numero de tentativas e o dossie que o
originou. O vinculo com o dossie e o que vai permitir, no M5, ligar metrica de
retencao ao material que gerou o roteiro -- sem ele, "este video foi melhor" nao
tem como virar "esta fonte rende melhor".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent.memory.store import SignalStore
from agent.models import Dossier, Fact, Script

AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path) -> SignalStore:
    return SignalStore(tmp_path / "agent.db")


def fato() -> Fact:
    return Fact(
        claim="O Bonsai 2 27B ocupa 5,9 GB, mais de 9x menor",
        source_url="https://prismml.com/news/bonsai-2-27b",
        source_name="PrismML",
        quote="that occupies 5.9 GB on disk",
    )


def roteiro(topic: str = "Bonsai 2 27B",
            hook: str = "Um modelo enorme cabe num pendrive.") -> Script:
    return Script(
        topic=topic,
        hook=hook,
        body="Corpo do roteiro com tamanho suficiente para respeitar o contrato do modelo.",
        closing="Numero sem base de comparacao e marketing.",
        search_terms=["memory chip macro", "server rack lights", "binary code screen"],
        facts=[fato()],
    )


class TestGravacao:
    def test_roteiro_volta_identico(self, store):
        store.record_script(
            roteiro(), model="gemini-2.5-flash", provider="gemini",
            usage=(2000, 300), latency_s=5.1, attempts=[{"violations": [], "word_count": 180}],
        )
        lido = store.latest_script()
        assert lido.hook == roteiro().hook
        assert lido.facts[0].quote == fato().quote
        assert lido.search_terms == roteiro().search_terms

    def test_tentativas_reprovadas_ficam_gravadas(self, store):
        """Elas dizem onde o prompt esta fraco, e desaparecem se so o sucesso
        for gravado."""
        store.record_script(
            roteiro(), model="m", provider="p", usage=(1, 1), latency_s=0.1,
            attempts=[
                {"violations": ["a narracao tem 90 palavras"], "word_count": 90},
                {"violations": [], "word_count": 180},
            ],
        )
        with store._conn() as conn:
            row = conn.execute(
                "SELECT attempts, attempts_json, word_count FROM scripts"
            ).fetchone()
        assert row["attempts"] == 2
        assert "90 palavras" in row["attempts_json"]
        assert row["word_count"] == roteiro().word_count

    def test_vinculo_com_o_dossie_de_origem(self, store):
        dossier_id = store.record_dossier(
            Dossier(topic="Bonsai 2 27B", facts=[fato()], collected_at=AGORA),
            model="m", provider="p", usage=(1, 1), latency_s=0.1, source_count=1,
        )
        store.record_script(
            roteiro(), model="m", provider="p", usage=(1, 1), latency_s=0.1,
            attempts=[{"violations": [], "word_count": 180}], dossier_id=dossier_id,
        )
        with store._conn() as conn:
            row = conn.execute("SELECT dossier_id FROM scripts").fetchone()
        assert row["dossier_id"] == dossier_id

    def test_id_do_dossie_mais_recente_por_tema(self, store):
        dossie = Dossier(topic="Bonsai 2 27B", facts=[fato()], collected_at=AGORA)
        primeiro = store.record_dossier(
            dossie, model="m", provider="p", usage=(1, 1), latency_s=0.1, source_count=1,
        )
        segundo = store.record_dossier(
            dossie, model="m", provider="p", usage=(1, 1), latency_s=0.1, source_count=1,
        )
        assert primeiro != segundo
        assert store.latest_dossier_id("Bonsai 2 27B") == segundo
        assert store.latest_dossier_id("tema inexistente") is None


class TestLeitura:
    def test_mais_recente_por_tema(self, store):
        for hook in ("Primeira versao do hook.", "Segunda versao do hook."):
            store.record_script(
                roteiro(hook=hook), model="m", provider="p", usage=(1, 1),
                latency_s=0.1, attempts=[{"violations": [], "word_count": 180}],
            )
        assert store.latest_script("Bonsai 2 27B").hook == "Segunda versao do hook."

    def test_memoria_vazia_devolve_none(self, store):
        assert store.latest_script() is None
        assert store.script_count() == 0

    def test_custo_consultavel_por_sql(self, store):
        store.record_script(
            roteiro(), model="openai/gpt-oss-120b", provider="groq",
            usage=(1800, 260), latency_s=0.9,
            attempts=[{"violations": [], "word_count": 180}],
        )
        with store._conn() as conn:
            row = conn.execute(
                "SELECT provider, input_tokens, output_tokens, latency_s FROM scripts"
            ).fetchone()
        assert row["provider"] == "groq"
        assert (row["input_tokens"], row["output_tokens"]) == (1800, 260)
        assert row["latency_s"] == 0.9


def test_roteiro_gravado_e_o_mesmo_que_o_render_consome(store, tmp_path):
    """O JSON gravado tem de validar de volta no contrato que o M0 renderiza."""
    store.record_script(
        roteiro(), model="m", provider="p", usage=(1, 1), latency_s=0.1,
        attempts=[{"violations": [], "word_count": 180}],
    )
    with store._conn() as conn:
        bruto = conn.execute("SELECT script_json FROM scripts").fetchone()["script_json"]

    caminho = tmp_path / "roteiro.json"
    caminho.write_text(bruto, encoding="utf-8")
    recarregado = Script.model_validate(json.loads(caminho.read_text(encoding="utf-8")))
    assert recarregado.word_count == roteiro().word_count
