"""Metricas do M5, sem rede: serie temporal por publish_id.

A coleta e manual (lida no app) porque a Content Posting API, no escopo
video.upload da inbox, nao expoe endpoint de metricas. O que estes testes
travam: a curva (varias coletas por post, em ordem), o vinculo opcional com o
roteiro, e a rejeicao cedo de numero impossivel.
"""

from __future__ import annotations

import pytest

from agent.memory.store import SignalStore

PID = "v_inbox_file~v2.999"


class TestSerie:
    def test_coletas_formam_curva_em_ordem(self, tmp_path):
        store = SignalStore(tmp_path / "agent.db")
        store.record_metric(PID, 120)
        store.record_metric(PID, 340, avg_watch_s=41.2, completion_rate=0.55)

        serie = store.metrics_for(PID)
        assert [m["views"] for m in serie] == [120, 340]
        ultima = store.latest_metric(PID)
        assert ultima is not None and ultima["views"] == 340
        assert ultima["completion_rate"] == 0.55
        assert store.metric_count() == 2

    def test_sem_coleta_e_nada(self, tmp_path):
        assert SignalStore(tmp_path / "agent.db").latest_metric(PID) is None

    def test_script_id_fecha_o_loop(self, tmp_path):
        store = SignalStore(tmp_path / "agent.db")
        store.record_metric(PID, 10, script_id=2)
        assert store.latest_metric(PID)["script_id"] == 2


class TestValidacao:
    def test_numero_impossivel_falha_cedo(self, tmp_path):
        store = SignalStore(tmp_path / "agent.db")
        with pytest.raises(ValueError):
            store.record_metric("", 10)
        with pytest.raises(ValueError):
            store.record_metric(PID, -1)
        with pytest.raises(ValueError):
            store.record_metric(PID, 10, completion_rate=1.5)
        with pytest.raises(ValueError):
            store.record_metric(PID, 10, avg_watch_s=-3.0)
        assert store.metric_count() == 0
