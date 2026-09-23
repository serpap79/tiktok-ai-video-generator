"""Testes do OAuth TikTok, sem rede e sem app registrado.

Contrato: a URL de autorizacao carrega client_key, escopo e redirect; a troca
e o refresh mandam form-encoded e devolvem os tokens; sem client_key/secret o
erro diz onde registrar o app, em vez de mandar POST anonimo.
"""

from __future__ import annotations

import httpx
import pytest

from agent.adapters.tiktok_oauth import TikTokOAuth, authorize_url
from agent.config import Settings
from agent.ports.publisher import PublisherAuthError


def _oauth(body: dict, status: int = 200, **kwargs) -> TikTokOAuth:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    client = httpx.Client(
        base_url="https://open.tiktokapis.com", transport=httpx.MockTransport(handler)
    )
    settings = Settings(
        tiktok_client_key="chave-teste",
        tiktok_client_secret="segredo-teste",
        tiktok_redirect_uri="https://exemplo/callback",
        **kwargs,
    )
    return TikTokOAuth(settings, client)


class TestAuthorizeUrl:
    def test_carrega_escopo_e_redirect(self):
        url = authorize_url("minha-key", "https://exemplo/callback")
        assert url.startswith("https://www.tiktok.com/v2/auth/authorize/")
        assert "client_key=minha-key" in url
        assert "scope=video.upload" in url
        assert "response_type=code" in url
        assert "redirect_uri=" in url

    def test_state_e_opcional(self):
        assert "state=" not in authorize_url("k", "https://exemplo/cb")
        assert "state=xyz" in authorize_url("k", "https://exemplo/cb", state="xyz")


class TestExchange:
    def test_troca_devolve_tokens(self):
        oauth = _oauth({"access_token": "act", "refresh_token": "rfr",
                        "open_id": "oid"})
        tokens = oauth.exchange_code("codigo-do-callback")
        assert tokens == {"access_token": "act", "refresh_token": "rfr",
                          "open_id": "oid"}

    def test_refresh_usa_o_grant_certo(self):
        vistos: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            vistos.append(request)
            return httpx.Response(200, json={"access_token": "novo"})

        client = httpx.Client(
            base_url="https://open.tiktokapis.com",
            transport=httpx.MockTransport(handler),
        )
        oauth = TikTokOAuth(Settings(tiktok_client_key="k", tiktok_client_secret="s"),
                            client)
        assert oauth.refresh("rfr-antigo")["access_token"] == "novo"
        assert "grant_type=refresh_token" in vistos[0].content.decode()

    def test_sem_credencial_nao_chama_a_rede(self):
        chamadas: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            chamadas.append(request)
            return httpx.Response(200, json={})

        client = httpx.Client(
            base_url="https://open.tiktokapis.com",
            transport=httpx.MockTransport(handler),
        )
        oauth = TikTokOAuth(
            Settings(tiktok_client_key="", tiktok_client_secret=""), client)
        with pytest.raises(PublisherAuthError, match="developers.tiktok.com"):
            oauth.exchange_code("codigo")
        assert chamadas == []

    def test_recusa_diz_o_http(self):
        oauth = _oauth({"error": {"message": "code expirado"}}, status=400)
        with pytest.raises(PublisherAuthError, match="400"):
            oauth.exchange_code("codigo-velho")


class TestRenovacaoAutomatica:
    """O token vale 24h: o piloto renova antes de cada post e grava no .env."""

    def test_renova_e_grava_os_dois_tokens_no_env(self, tmp_path):
        import httpx

        from agent.adapters.tiktok_oauth import refresh_and_store
        from agent.config import Settings

        env = tmp_path / ".env"
        env.write_text("# comentario\nAGENT_TIKTOK_ACCESS_TOKEN=velho\n"
                       "AGENT_TIKTOK_REFRESH_TOKEN=rt-velho\nOUTRA=1\n", encoding="utf-8")
        cfg = Settings(_env_file=None, tiktok_client_key="ck", tiktok_client_secret="cs",
                       tiktok_refresh_token="rt-velho", tiktok_access_token="velho")

        def handler(request: httpx.Request) -> httpx.Response:
            assert b"grant_type=refresh_token" in request.content
            assert b"refresh_token=rt-velho" in request.content
            return httpx.Response(200, json={
                "access_token": "act-novo", "refresh_token": "rt-novo",
                "open_id": "u", "expires_in": 86400})

        cliente = httpx.Client(base_url="https://open.tiktokapis.com",
                               transport=httpx.MockTransport(handler))
        assert refresh_and_store(cfg, env, client=cliente) == "act-novo"
        texto = env.read_text(encoding="utf-8")
        assert "AGENT_TIKTOK_ACCESS_TOKEN=act-novo" in texto
        assert "AGENT_TIKTOK_REFRESH_TOKEN=rt-novo" in texto
        assert "# comentario" in texto and "OUTRA=1" in texto
        assert cfg.tiktok_access_token == "act-novo"
        assert (env.stat().st_mode & 0o777) == 0o600

    def test_erro_em_string_vira_motivo_legivel(self):
        import httpx
        import pytest

        from agent.adapters.tiktok_oauth import TikTokOAuth
        from agent.config import Settings
        from agent.ports.publisher import PublisherAuthError

        cfg = Settings(_env_file=None, tiktok_client_key="ck", tiktok_client_secret="cs")
        cliente = httpx.Client(base_url="https://open.tiktokapis.com",
                               transport=httpx.MockTransport(lambda r: httpx.Response(
                                   400, json={"error": "invalid_grant",
                                              "error_description": "refresh expirado"})))
        with pytest.raises(PublisherAuthError, match="invalid_grant refresh expirado"):
            TikTokOAuth(cfg, client=cliente).refresh("rt")
