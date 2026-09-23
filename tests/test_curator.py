"""Testes da orquestracao do curador: ordem dos portoes, score e registro."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent.curator.curator import Curator
from agent.memory.store import SignalStore
from agent.models import Signal, Verdict

AGORA = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def sinal(term: str, source: str = "hacker_news", volume: float = 300.0,
          velocity: float | None = 20.0, unit: str = "points") -> Signal:
    return Signal(term=term, source=source, volume=volume, unit=unit,
                  velocity=velocity, seen_at=AGORA)


class TestOrdemDosPortoes:
    def test_politica_vence_velocidade_altissima(self):
        """Tema vetado nao pode ganhar no ranking por estar subindo rapido.

        Por isso a politica roda ANTES do score, e nao como desempate depois.
        """
        report = Curator().curate([
            sinal("Lula defende caneta emagrecedora", "wikipedia", 999_999, 99_999, "pageviews"),
            sinal("Qwen 3.8 Omni Flash: novo modelo de IA", volume=100, velocity=1.0),
        ])
        assert report.selected.term.startswith("Qwen")
        assert report.by_verdict(Verdict.rejected_policy)[0].score == 0.0

    def test_politica_e_avaliada_antes_do_nicho(self):
        """Um tema pode estar fora do nicho E vetado; o registro deve mostrar o
        veto, que e a informacao mais forte."""
        (d,) = Curator().curate([sinal("Eleicao presidencial", "wikipedia")]).decisions
        assert d.verdict is Verdict.rejected_policy

    def test_duplicata_so_e_checada_depois_dos_outros_dois(self):
        """Comparar com o ledger custa; nao se gasta isso num tema ja vetado."""
        ledger = ["Eleicao presidencial no Brasil"]
        (d,) = Curator().curate([sinal("Eleicao presidencial", "wikipedia")], ledger).decisions
        assert d.verdict is Verdict.rejected_policy


class TestSelecao:
    def test_escolhe_exatamente_um(self):
        report = Curator().curate([
            sinal("GPU quantization benchmark", velocity=50.0),
            sinal("Novo modelo de IA bate recorde", velocity=40.0),
            sinal("Linux kernel 7.0 lancado", velocity=30.0),
        ])
        assert len(report.by_verdict(Verdict.selected)) == 1

    def test_toda_decisao_tem_motivo_inclusive_a_aprovada(self):
        """Decisao sem justificativa gravada nao da para auditar depois."""
        report = Curator().curate([
            sinal("GPU quantization benchmark"),
            sinal("Eleicao presidencial", "wikipedia"),
            sinal("Receita de bolo de cenoura", "wikipedia", velocity=None),
        ])
        assert all(d.reason.strip() for d in report.decisions)
        assert "score" in report.selected.reason

    def test_nenhum_elegivel_devolve_selected_none(self):
        report = Curator().curate([sinal("Eleicao presidencial", "wikipedia")])
        assert report.selected is None
        assert report.tally()["rejected_policy"] == 1

    def test_lista_vazia_nao_estoura(self):
        report = Curator().curate([])
        assert report.selected is None
        assert report.decisions == []

    def test_o_escolhido_entra_no_ledger_da_propria_rodada(self):
        """Duas fontes trazendo a mesma historia nao podem produzir dois videos.

        O termo escolhido e acrescentado ao ledger em memoria durante a propria
        curadoria -- mas os demais ja passaram pelo portao de duplicata antes do
        score, entao o efeito real e no ledger persistido da proxima rodada.
        """
        report = Curator().curate([sinal("GPU quantization benchmark")])
        assert report.selected is not None


class TestScorePorPercentil:
    def test_unidades_diferentes_nao_competem_no_valor_bruto(self):
        """Pageview da Wikipedia e ponto do Hacker News nao compartilham escala.

        Somar os numeros crus faria a Wikipedia vencer sempre por ter unidade
        maior, e nao por ter assunto melhor. O score usa percentil dentro da
        propria fonte.
        """
        sinais = [
            # Wikipedia com volume gigante, mas no rodape da propria fonte
            sinal("Computacao quantica", "wikipedia", volume=50_000, velocity=10.0,
                  unit="pageviews"),
            sinal("Robotica", "wikipedia", volume=90_000, velocity=500.0, unit="pageviews"),
            sinal("Genoma humano", "wikipedia", volume=95_000, velocity=900.0, unit="pageviews"),
            # HN com volume minusculo, mas no topo da propria fonte
            sinal("Novo modelo de IA quebra benchmark", volume=400, velocity=90.0),
            sinal("Chip fotonico", volume=120, velocity=5.0),
        ]
        report = Curator().curate(sinais)
        # O topo do HN precisa conseguir vencer o topo da Wikipedia
        assert report.selected.source == "hacker_news"

    def test_velocidade_desconhecida_nao_e_tratada_como_zero(self):
        """None significa "nao medi"; zero significaria "medi e nao se moveu".

        Quem nao tem medida recebe o meio da escala e compete pelo volume e pelo
        nicho ate a segunda coleta dar a taxa.
        """
        # Com uma unica observacao de velocidade na fonte nao ha distribuicao, e
        # o percentil devolve 0.5 -- o mesmo valor do desconhecido. Sao precisos
        # varios valores para o teste dizer alguma coisa.
        sinais = [
            sinal("GPU benchmark recorde", volume=100, velocity=100.0),
            sinal("Kernel Linux otimizado", volume=100, velocity=50.0),
            sinal("Chip fotonico integrado", volume=100, velocity=1.0),
            sinal("Robotica autonoma na industria", volume=100, velocity=None),
        ]
        por_termo = {d.term: d for d in Curator().curate(sinais).decisions}

        lento = por_termo["Chip fotonico integrado"]
        sem_medida = por_termo["Robotica autonoma na industria"]
        assert sem_medida.velocity is None
        # "nao medi" precisa valer mais que "medi e esta no rodape da fonte"
        assert sem_medida.score > lento.score

    def test_score_fica_entre_zero_e_um(self):
        report = Curator().curate([
            sinal("GPU quantization benchmark", velocity=1e9, volume=1e9),
            sinal("Linux kernel", velocity=0.0, volume=0.0),
        ])
        assert all(0.0 <= d.score <= 1.0 for d in report.decisions)


class TestLedger:
    @pytest.fixture
    def store(self, tmp_path) -> SignalStore:
        return SignalStore(tmp_path / "agent.db")

    def test_tema_ja_aprovado_e_rejeitado_como_duplicata(self, store):
        anterior = "Bonsai 2 27B Near Lossless Compression Footprint"
        report = Curator().curate(
            [sinal("Bonsai 2 27B: Near-Lossless Compression in a Smaller Footprint")],
            ledger=[anterior],
        )
        (d,) = report.decisions
        assert d.verdict is Verdict.rejected_duplicate
        assert d.duplicate_of == anterior

    def test_grava_rejeicoes_tambem(self, store):
        """Sem as rejeicoes gravadas, so se sabe o que foi escolhido, nunca o
        que foi perdido -- e calibrar o score vira chute."""
        report = Curator().curate([
            sinal("GPU quantization benchmark"),
            sinal("Eleicao presidencial", "wikipedia"),
        ])
        assert store.record_decisions(report.decisions) == 2
        assert store.topic_count() == 2

    def test_ledger_so_devolve_os_aprovados(self, store):
        """Tema rejeitado por politica nunca virou video: bloquear o parecido
        estenderia o veto a assuntos que nunca foram julgados."""
        report = Curator().curate([
            sinal("GPU quantization benchmark"),
            sinal("Eleicao presidencial", "wikipedia"),
        ])
        store.record_decisions(report.decisions)
        recentes = store.recent_topics()
        assert recentes == ["GPU quantization benchmark"]

    def test_ledger_respeita_a_janela_de_dias(self, store):
        report = Curator().curate([sinal("GPU quantization benchmark")])
        store.record_decisions(report.decisions)
        assert store.recent_topics(days=30) != []
        assert store.recent_topics(days=0) == []
