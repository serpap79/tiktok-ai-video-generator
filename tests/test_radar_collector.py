"""Testes do coletor: isolamento de falha e velocidade por diferenca."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent.memory.store import SignalStore
from agent.models import Signal
from agent.ports.radar import SourceUnavailable
from agent.radar.collector import Radar

AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


class FonteFalsa:
    def __init__(self, name: str, sinais: list[Signal] | None = None,
                 erro: Exception | None = None):
        self.name = name
        self._sinais = sinais or []
        self._erro = erro
        self.chamadas = 0

    def collect(self) -> list[Signal]:
        self.chamadas += 1
        if self._erro:
            raise self._erro
        return self._sinais


def sinal(term="tema", source="fonte", volume=100.0, velocity=None, seen_at=AGORA) -> Signal:
    return Signal(term=term, source=source, volume=volume, unit="pageviews",
                  velocity=velocity, seen_at=seen_at)


@pytest.fixture
def store(tmp_path) -> SignalStore:
    return SignalStore(tmp_path / "agent.db")


class TestIsolamentoDeFalha:
    def test_fonte_fora_do_ar_nao_derruba_a_coleta(self, store):
        """A janela de um trend e de horas: nao ha tempo de esperar um servico
        se recuperar, e o GDELT cai com frequencia."""
        radar = Radar([
            FonteFalsa("gdelt", erro=SourceUnavailable("429")),
            FonteFalsa("hacker_news", [sinal(source="hacker_news", velocity=12.0)]),
        ], store)

        report = radar.collect()

        assert report.failures == {"gdelt": "429"}
        assert report.sources_ok == ["hacker_news"]
        assert len(report.signals) == 1

    def test_bug_no_parser_tambem_e_isolado_mas_nomeado(self, store):
        """Erro inesperado nao pode virar falha silenciosa: o tipo vai no relatorio."""
        radar = Radar([
            FonteFalsa("quebrada", erro=KeyError("title")),
            FonteFalsa("boa", [sinal(source="boa", velocity=1.0)]),
        ], store)

        report = radar.collect()

        assert "KeyError" in report.failures["quebrada"]
        assert len(report.signals) == 1

    def test_todas_fora_devolve_relatorio_vazio_sem_estourar(self, store):
        report = Radar([FonteFalsa("a", erro=SourceUnavailable("x"))], store).collect()
        assert report.signals == []
        assert report.failures == {"a": "x"}


class TestVelocidadePorDiferenca:
    def test_calcula_a_partir_da_coleta_anterior(self, store):
        antes = AGORA - timedelta(hours=4)
        store.record([sinal(term="Bonsai", volume=1000.0, seen_at=antes)])

        radar = Radar([FonteFalsa("fonte", [sinal(term="Bonsai", volume=1400.0)])], store)
        (s,) = radar.collect().signals

        assert s.velocity == 100.0  # (1400 - 1000) / 4h

    def test_sem_coleta_anterior_a_velocidade_fica_desconhecida(self, store):
        """None significa "nao sei", e o curador precisa distinguir isso de zero:
        zero seria "medi e nao se moveu", que e uma afirmacao bem diferente."""
        (s,) = Radar([FonteFalsa("fonte", [sinal()])], store).collect().signals
        assert s.velocity is None

    def test_queda_produz_velocidade_negativa(self, store):
        """Assunto em queda e informacao util: evita entrar num tema que ja passou."""
        store.record([sinal(term="Antigo", volume=5000.0, seen_at=AGORA - timedelta(hours=5))])
        radar = Radar([FonteFalsa("fonte", [sinal(term="Antigo", volume=2500.0)])], store)
        (s,) = radar.collect().signals
        assert s.velocity == -500.0

    def test_intervalo_curto_demais_nao_vira_velocidade(self, store):
        """Duas coletas em poucos minutos dividem ruido por um intervalo minusculo
        e produzem velocidade absurda."""
        store.record([sinal(volume=1000.0, seen_at=AGORA - timedelta(minutes=5))])
        (s,) = Radar([FonteFalsa("fonte", [sinal(volume=1001.0)])], store).collect().signals
        assert s.velocity is None

    def test_velocidade_nativa_da_fonte_e_preservada(self, store):
        """O HN ja traz pontos/hora; o coletor nao pode sobrescrever com delta."""
        store.record([sinal(term="HN", source="hacker_news", volume=100.0,
                            seen_at=AGORA - timedelta(hours=2))])
        radar = Radar([FonteFalsa("hacker_news", [
            sinal(term="HN", source="hacker_news", volume=300.0, velocity=42.0)
        ])], store)
        (s,) = radar.collect().signals
        assert s.velocity == 42.0

    def test_a_serie_e_gravada_para_a_proxima_coleta(self, store):
        Radar([FonteFalsa("fonte", [sinal(), sinal(term="outro")])], store).collect()
        assert store.count() == 2


class TestStore:
    def test_previous_devolve_a_observacao_mais_recente_antes_do_corte(self, store):
        for h, v in ((10, 100.0), (5, 200.0), (1, 300.0)):
            store.record([sinal(volume=v, seen_at=AGORA - timedelta(hours=h))])

        volume, visto = store.previous("fonte:tema", before=AGORA - timedelta(hours=3))

        assert volume == 200.0
        assert visto == AGORA - timedelta(hours=5)

    def test_previous_sem_historico_devolve_none(self, store):
        assert store.previous("fonte:inexistente", before=AGORA) is None

    def test_chave_separa_o_mesmo_termo_em_fontes_diferentes(self, store):
        """"IA" no Trends e "IA" na Wikipedia sao series distintas, em unidades
        distintas: misturar produziria velocidade sem sentido."""
        store.record([
            sinal(term="IA", source="google_trends", volume=2000.0,
                  seen_at=AGORA - timedelta(hours=3)),
            sinal(term="IA", source="wikipedia", volume=50.0,
                  seen_at=AGORA - timedelta(hours=3)),
        ])
        assert store.previous("google_trends:ia", before=AGORA)[0] == 2000.0
        assert store.previous("wikipedia:ia", before=AGORA)[0] == 50.0
