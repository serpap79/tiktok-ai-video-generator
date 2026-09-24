"""OAuth do TikTok (Login Kit) para o escopo `video.upload`.

Cubre solo lo que necesita M4: construir la URL de autorización, canjear el `code` por
tokens e renovar com o `refresh_token`. Os tokens nunca vao para o banco --
moram so no `.env` (`AGENT_TIKTOK_ACCESS_TOKEN`, `AGENT_TIKTOK_REFRESH_TOKEN`),
como todo secreto del proyecto.

O formato exato da troca (form-encoded em `/v2/oauth/token/`) segue o Login
Kit documentado, pero aún no se ha probado con el servicio real: no hay
una app registrada en esta máquina. Cuando exista, `agent publish` será la prueba
real, y cualquier divergencia aparecerá allí, no en una prueba simulada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from agent.config import Settings
from agent.config import settings as default_settings
from agent.ports.publisher import PublisherAuthError, PublisherError

AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_PATH = "/v2/oauth/token/"
BASE_URL = "https://open.tiktokapis.com"


def authorize_url(
    client_key: str,
    redirect_uri: str,
    *,
    scopes: tuple[str, ...] = ("video.upload",),
    state: str = "",
) -> str:
    """URL para a pessoa criadora autorizar o app no navegador."""
    params = {
        "client_key": client_key,
        "scope": ",".join(scopes),
        "response_type": "code",
        "redirect_uri": redirect_uri,
    }
    if state:
        params["state"] = state
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


class TikTokOAuth:
    """Troca e renova tokens. Cliente HTTP injetavel para teste sem rede."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ):
        self.settings = settings or default_settings
        self._client = client or httpx.Client(
            base_url=BASE_URL,
            timeout=httpx.Timeout(self.settings.tiktok_timeout_s),
        )

    def exchange_code(self, code: str, redirect_uri: str = "") -> dict[str, str]:
        """Troca o `code` da autorizacao por access + refresh token."""
        return self._token(
            {
                "client_key": self.settings.tiktok_client_key,
                "client_secret": self.settings.tiktok_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri or self.settings.tiktok_redirect_uri,
            }
        )

    def refresh(self, refresh_token: str = "") -> dict[str, str]:
        """Renova o access token sem passar pelo navegador de novo."""
        return self._token(
            {
                "client_key": self.settings.tiktok_client_key,
                "client_secret": self.settings.tiktok_client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token or self.settings.tiktok_refresh_token,
            }
        )

    def _token(self, form: dict[str, str]) -> dict[str, str]:
        if not form["client_key"] or not form["client_secret"]:
            raise PublisherAuthError(
                "sem client_key/client_secret; registre o app em developers.tiktok.com"
            )
        try:
            r = self._client.post(
                TOKEN_PATH,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise PublisherError(f"falha de rede no oauth: {exc}") from exc
        corpo: Any = r.json() if r.headers.get("content-type", "").startswith(
            "application/json"
        ) else {}
        if r.status_code != 200 or not isinstance(corpo, dict) or corpo.get("error"):
            erro = corpo.get("error", "") if isinstance(corpo, dict) else ""
            # O endpoint v2 devolve `error` como string + `error_description`,
            # como objeto; tratar ambos evita un AttributeError en lugar
            # do motivo real.
            detalhe = (erro.get("message", "") if isinstance(erro, dict)
                       else f"{erro} {corpo.get('error_description', '')}".strip())
            raise PublisherAuthError(f"oauth recusado (HTTP {r.status_code}): {detalhe}")
        try:
            return {
                "access_token": str(corpo["access_token"]),
                "refresh_token": str(corpo.get("refresh_token", "")),
                "open_id": str(corpo.get("open_id", "")),
            }
        except KeyError as exc:
            raise PublisherError(f"resposta oauth sem tokens: {corpo}") from exc


def refresh_and_store(settings: Settings | None = None, env_path: Path | None = None,
                      client: httpx.Client | None = None) -> str:
    """Renova o access token e grava o par novo no `.env`. Devolve o access token.

    O access token do TikTok vale 24h: o do primeiro post (19/09, ~11h) ja
    teria vencido no slot das 9h do dia seguinte. O piloto automatico chama
    isto antes de cada publicacao. O refresh token tambem pode girar -- por
    isso os DOIS sao gravados, e na mesma escrita atomica.
    """
    from agent.config import PROJECT_ROOT
    from agent.envfile import update_env

    cfg = settings or default_settings
    tokens = TikTokOAuth(cfg, client=client).refresh()
    novo_refresh = tokens["refresh_token"] or cfg.tiktok_refresh_token
    update_env(env_path or PROJECT_ROOT / ".env", {
        "AGENT_TIKTOK_ACCESS_TOKEN": tokens["access_token"],
        "AGENT_TIKTOK_REFRESH_TOKEN": novo_refresh,
    })
    cfg.tiktok_access_token = tokens["access_token"]
    cfg.tiktok_refresh_token = novo_refresh
    return tokens["access_token"]
