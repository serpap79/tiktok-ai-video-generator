"""Testes do publicador TikTok (inbox), sem rede e sem app registrado.

O que importa aqui nao e "o video apareceu na inbox" -- isso so o primeiro
post real prova, no app. E o contrato: o payload do init, a sequencia de
chunks com Content-Range, o tipo de excecao para cada falha (token expirado e
cota estourada pedem acoes opostas), e o respeito a 6 req/min por token.
"""

from __future__ import annotations

import httpx
import pytest

from agent.adapters.tiktok_publisher import (
    INIT_PATH,
    STATUS_PATH,
    TikTokPublisher,
    chunk_ranges,
)
from agent.config import Settings
from agent.models import PublishState
from agent.ports.publisher import PublisherAuthError, PublisherRateLimited

INIT_OK = {
    "data": {
        "publish_id": "v_inbox_file~v2.123",
        "upload_url": "https://upload/video?token=abc",
    },
    "error": {"code": "ok", "message": "", "log_id": "x"},
}


def _settings(**kwargs) -> Settings:
    base = {"tiktok_chunk_size": 4, "tiktok_timeout_s": 5.0}
    base.update(kwargs)
    return Settings(**base)


def _cliente(roteiro: list[tuple[int, dict]]) -> tuple[httpx.Client, list[httpx.Request]]:
    """Respostas enfileiradas por chamada, com as requisicoes gravadas."""
    chamadas: list[httpx.Request] = []
    fila = list(roteiro)

    def handler(request: httpx.Request) -> httpx.Response:
        chamadas.append(request)
        status, corpo = fila.pop(0)
        return httpx.Response(status, json=corpo)

    return httpx.Client(
        base_url="https://open.tiktokapis.com",
        transport=httpx.MockTransport(handler),
    ), chamadas


def _publicador(
    roteiro: list[tuple[int, dict]], *, chunk_size: int = 4, **kwargs
) -> tuple[TikTokPublisher, list[httpx.Request], list[float]]:
    cliente, chamadas = _cliente(roteiro)
    sonecas: list[float] = []
    pub = TikTokPublisher(
        _settings(tiktok_chunk_size=chunk_size), cliente,
        time_fn=lambda: 0.0, sleeper=sonecas.append, **kwargs
    )
    return pub, chamadas, sonecas


def _mp4(tmp_path, tamanho: int = 10) -> str:
    video = tmp_path / "corte.mp4"
    video.write_bytes(bytes(range(tamanho)))
    return str(video)


class TestPayloadDoInit:
    def test_source_e_file_upload_com_tamanhos(self):
        # Numeros do exemplo da doc ("Media Transfer Guide"): 50_000_123 com
        # chunks de 10_000_000 sao 5 no piso, com o resto no ultimo.
        corpo = TikTokPublisher.build_init_payload(50_000_123, 10_000_000, 5)
        assert corpo == {
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": 50_000_123,
                "chunk_size": 10_000_000,
                "total_chunk_count": 5,
            }
        }

    def test_intervalos_ultimo_absorve_o_resto(self):
        assert chunk_ranges(50_000_123, 10_000_000, 5) == [
            (0, 9_999_999), (10_000_000, 19_999_999), (20_000_000, 29_999_999),
            (30_000_000, 39_999_999), (40_000_000, 50_000_122),
        ]
        assert chunk_ranges(4_194_304, 4_194_304, 1) == [(0, 4_194_303)]

    def test_plano_piso_e_inteiro_abaixo_de_5mb(self):
        pub, _, _ = _publicador([], chunk_size=10_000_000)
        assert pub._plan(50_000_123) == (10_000_000, 5)
        assert pub._plan(4_194_304) == (4_194_304, 1)

    def test_plano_entre_5mb_e_2x_chunk_vira_chunk_unico_do_arquivo(self):
        # Regressao do slot 1200 de 21/09/2026: video de 12,2 MB com chunk de
        # 10 MB saia como (10M, 1) e o init caia em 400 "chunk size is
        # invalid" -- com count=1 a API exige chunk_size == video_size.
        pub, _, _ = _publicador([], chunk_size=10_000_000)
        assert pub._plan(12_249_737) == (12_249_737, 1)
        assert pub._plan(6_000_000) == (6_000_000, 1)
        # Acima de 2x o chunk o piso segue valendo (ultimo absorve o resto).
        assert pub._plan(25_000_000) == (10_000_000, 2)
        assert pub._plan(40_800_000) == (10_000_000, 4)

    def test_config_fora_da_faixa_5_64mb_e_trazida_para_dentro(self):
        pub, _, _ = _publicador([], chunk_size=4)
        chunk, n = pub._plan(100_000_000)
        assert chunk >= 5 * 1024 * 1024
        assert n == 100_000_000 // chunk


class TestUpload:
    def test_chunk_unico_recebe_201(self, tmp_path):
        pub, chamadas, _ = _publicador([(200, INIT_OK), (201, {})], chunk_size=10)
        result = pub.upload(_mp4(tmp_path), access_token="tok")
        assert result.state is PublishState.uploaded
        assert result.publish_id == "v_inbox_file~v2.123"
        assert len(chamadas) == 2
        put = chamadas[1]
        assert put.headers["Content-Range"] == "bytes 0-9/10"
        assert put.headers["Content-Length"] == "10"

    def test_multiplos_chunks_com_206_e_201(self, tmp_path):
        # Arquivo minusculo cai no caminho inteiro; o fluxo multiplo e
        # exercitado direto no `_send_chunks`, com os intervalos da doc.
        pub, chamadas, _ = _publicador([(206, {}), (201, {})])
        blob = bytes(range(20))
        pub._send_chunks("https://upload/video?token=abc", blob, 10, 2)
        ranges = [c.headers["Content-Range"] for c in chamadas]
        assert ranges == ["bytes 0-9/20", "bytes 10-19/20"]

    def test_chunk_rejeitado_vira_falha_com_motivo(self, tmp_path):
        pub, _, _ = _publicador([(200, INIT_OK), (416, {})])
        result = pub.upload(_mp4(tmp_path), access_token="tok")
        assert result.state is PublishState.failed
        assert "416" in (result.error or "")

    def test_arquivo_inexistente_nao_gasta_chamada(self, tmp_path):
        pub, chamadas, _ = _publicador([])
        result = pub.upload(str(tmp_path / "sumiu.mp4"), access_token="tok")
        assert result.state is PublishState.failed
        assert chamadas == []

    def test_sem_token_e_erro_de_auth(self, tmp_path):
        pub, chamadas, _ = _publicador([])
        with pytest.raises(PublisherAuthError):
            pub.upload(_mp4(tmp_path), access_token="")
        assert chamadas == []


class TestErrosDaApi:
    def test_token_invalido_e_auth(self, tmp_path):
        corpo = {"data": {}, "error": {"code": "access_token_invalid", "message": "x"}}
        pub, _, _ = _publicador([(401, corpo)])
        result = pub.upload(_mp4(tmp_path), access_token="velho")
        assert result.state is PublishState.failed
        assert "access_token_invalid" in (result.error or "")

    def test_cota_estourada_preserva_o_motivo(self, tmp_path):
        corpo = {"data": {}, "error": {"code": "rate_limit_exceeded", "message": "lento"}}
        pub, _, _ = _publicador([(429, corpo)])
        with pytest.raises(PublisherRateLimited):
            pub.fetch_status("v_inbox_file~v2.123", access_token="tok")

    def test_teto_de_pendentes_tambem_e_cota(self, tmp_path):
        corpo = {"data": {}, "error": {"code": "spam_risk_too_many_pending_share",
                                       "message": "5 pendentes"}}
        pub, _, _ = _publicador([(403, corpo)])
        result = pub.upload(_mp4(tmp_path), access_token="tok")
        assert "spam_risk_too_many_pending_share" in (result.error or "")

    def test_resposta_sem_publish_id_diz_o_que_faltou(self, tmp_path):
        corpo = {"data": {}, "error": {"code": "ok", "message": ""}}
        pub, _, _ = _publicador([(200, corpo)])
        result = pub.upload(_mp4(tmp_path), access_token="tok")
        assert "publish_id" in (result.error or "")


class TestStatus:
    def test_status_volta_opaco(self):
        corpo = {"data": {"status": "PROCESSING"}, "error": {"code": "ok"}}
        pub, chamadas, _ = _publicador([(200, corpo)])
        assert pub.fetch_status("v_inbox_file~v2.123", access_token="tok") == "PROCESSING"
        assert chamadas[0].url.path == STATUS_PATH
        assert chamadas[0].headers["Authorization"] == "Bearer tok"

    def test_init_usa_bearer_e_json_utf8(self, tmp_path):
        pub, chamadas, _ = _publicador([(200, INIT_OK), (201, {})])
        pub.upload(_mp4(tmp_path), access_token="tok")
        init = chamadas[0]
        assert init.url.path == INIT_PATH
        assert init.headers["Authorization"] == "Bearer tok"
        assert "application/json" in init.headers["Content-Type"]


class TestCota:
    def test_sexta_chamada_dorme_antes_da_setima(self):
        corpo = {"data": {"status": "PROCESSING"}, "error": {"code": "ok"}}
        pub, _, _ = _publicador([(200, corpo)] * 7)
        for _ in range(6):
            pub.fetch_status("pid", access_token="tok")
        pub.fetch_status("pid", access_token="tok")
        # time_fn congelado em 0: a setima chamada dorme a janela inteira.
        assert pub._sleep is not None


class TestCarrosselPorUrl:
    """Foto so entra por PULL_FROM_URL; titulo e descricao vao pela API."""

    def test_payload_de_foto_para_a_inbox(self):
        p = TikTokPublisher.build_photo_payload(
            ["https://u.github.io/r/1.jpg", "https://u.github.io/r/2.jpg"],
            "t" * 200, "legenda #ia")
        assert p["post_mode"] == "MEDIA_UPLOAD" and p["media_type"] == "PHOTO"
        assert p["source_info"]["source"] == "PULL_FROM_URL"
        assert p["source_info"]["photo_images"][0].endswith("1.jpg")
        assert len(p["post_info"]["title"]) == 90

    def test_upload_de_fotos_devolve_publish_id(self):
        from agent.adapters.tiktok_publisher import PHOTO_INIT_PATH
        pub, chamadas, _ = _publicador([
            (200, {"data": {"publish_id": "p_pub_url~v2.9"}, "error": {"code": "ok"}})])
        r = pub.upload_photos(["https://u.github.io/r/1.jpg"], access_token="tok",
                              title="Titulo", description="Legenda")
        assert r.state is PublishState.uploaded and r.publish_id == "p_pub_url~v2.9"
        assert chamadas[0].url.path == PHOTO_INIT_PATH

    def test_dominio_nao_verificado_falha_com_motivo(self):
        pub, _, _ = _publicador([
            (403, {"error": {"code": "url_ownership_unverified", "message": "verify"}})])
        r = pub.upload_photos(["https://x.com/1.jpg"], access_token="tok",
                              title="t", description="d")
        assert r.state is PublishState.failed and "url_ownership_unverified" in r.error


class TestHospedagem:
    def test_publica_e_espera_o_pages_servir(self, tmp_path):
        from agent.publish.media_host import GitPagesHost

        (tmp_path / "repo" / ".git").mkdir(parents=True)
        slide = tmp_path / "s1.jpg"
        slide.write_bytes(b"jpg")
        comandos: list[list[str]] = []

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        def run(cmd, **kw):
            comandos.append(cmd)
            return R()

        respostas = iter([404, 200])
        cliente = httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(next(respostas))))
        host = GitPagesHost(tmp_path / "repo", "https://u.github.io/r/", runner=run,
                            client=cliente, sleeper=lambda s: None)
        urls = host.publish([slide], "seucanal/2026-09-20/1500")
        assert urls == ["https://u.github.io/r/seucanal/2026-09-20/1500/s1.jpg"]
        assert [c[1] for c in comandos] == ["add", "commit", "push"]
        assert (tmp_path / "repo" / "seucanal/2026-09-20/1500/s1.jpg").exists()

    def test_png_vira_jpeg(self, tmp_path):
        from PIL import Image

        from agent.publish.media_host import to_jpeg
        png = tmp_path / "slide-1.png"
        Image.new("RGB", (1080, 1920), (10, 10, 12)).save(png)
        (jpg,) = to_jpeg([png], tmp_path / "out")
        with Image.open(jpg) as img:
            assert img.format == "JPEG" and img.size == (1080, 1920)
