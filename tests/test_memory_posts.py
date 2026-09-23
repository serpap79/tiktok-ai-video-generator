"""Testes da tabela posts: subidas gravadas, com motivo, inclusive as falhas."""

from __future__ import annotations

from agent.memory.store import SignalStore


def _banco(tmp_path) -> SignalStore:
    return SignalStore(tmp_path / "agent.db")


class TestPosts:
    def test_upload_gravado_e_recuperavel(self, tmp_path):
        store = _banco(tmp_path)
        store.record_post("v_inbox_file~v2.1", "output/corte.mp4", status="uploaded")
        ultimo = store.latest_post()
        assert ultimo is not None
        assert ultimo["publish_id"] == "v_inbox_file~v2.1"
        assert ultimo["status"] == "uploaded"
        assert store.post_count() == 1

    def test_falha_antes_do_init_tambem_fica_gravada(self, tmp_path):
        """Sem isso, tentativa que nao gerou publish_id some do historico."""
        store = _banco(tmp_path)
        store.record_post("", "output/sumiu.mp4", status="failed",
                          error="arquivo nao encontrado")
        assert store.post_count() == 1
        assert store.latest_post()["error"] == "arquivo nao encontrado"

    def test_status_atualiza_sem_apagar_o_envio(self, tmp_path):
        store = _banco(tmp_path)
        store.record_post("v_inbox_file~v2.2", "output/corte.mp4", status="uploaded")
        store.update_post_status("v_inbox_file~v2.2", status="PROCESSING")
        assert store.latest_post()["status"] == "PROCESSING"
        assert store.post_count() == 1

    def test_ultimo_por_video(self, tmp_path):
        store = _banco(tmp_path)
        store.record_post("pid-1", "output/a.mp4", status="uploaded")
        store.record_post("pid-2", "output/b.mp4", status="uploaded")
        assert store.latest_post("output/a.mp4")["publish_id"] == "pid-1"
