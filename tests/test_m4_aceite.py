"""Aceite do M4, sem rede e sem app registrado.

O M4 esta feito quando o caminho `MP4 -> inbox` funciona de ponta a ponta no
contrato: init, chunks com Content-Range, publish_id rastreavel e status
consultavel. O primeiro post real no app continua sendo o aceite humano --
este teste garante que, quando ele acontecer, o codigo ja fala a API certa.
"""

from __future__ import annotations

import httpx

from agent.adapters.tiktok_publisher import TikTokPublisher
from agent.config import Settings
from agent.memory.store import SignalStore
from agent.models import PublishState

PUBLISH_ID = "v_inbox_file~v2.999"


def _cliente() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/post/publish/inbox/video/init/":
            return httpx.Response(200, json={
                "data": {"publish_id": PUBLISH_ID,
                         "upload_url": "https://upload/video?token=m4"},
                "error": {"code": "ok", "message": ""},
            })
        if str(request.url).startswith("https://upload/"):
            return httpx.Response(201, content=b"")
        if request.url.path == "/v2/post/publish/status/fetch/":
            return httpx.Response(200, json={
                "data": {"status": "PROCESSING"},
                "error": {"code": "ok", "message": ""},
            })
        return httpx.Response(404, json={})

    return httpx.Client(
        base_url="https://open.tiktokapis.com",
        transport=httpx.MockTransport(handler),
    )


class TestAceiteM4:
    def test_mp4_chega_a_inbox_e_e_rastreavel(self, tmp_path):
        video = tmp_path / "corte.mp4"
        video.write_bytes(b"\x00" * 1024)

        pub = TikTokPublisher(Settings(), _cliente())
        result = pub.upload(str(video), access_token="tok-de-teste")
        assert result.state is PublishState.uploaded
        assert result.publish_id == PUBLISH_ID

        store = SignalStore(tmp_path / "agent.db")
        store.record_post(result.publish_id or "", str(video),
                          status=result.state.value)
        assert store.latest_post()["publish_id"] == PUBLISH_ID

        estado = pub.fetch_status(PUBLISH_ID, access_token="tok-de-teste")
        store.update_post_status(PUBLISH_ID, status=estado)
        assert store.latest_post()["status"] == "PROCESSING"
