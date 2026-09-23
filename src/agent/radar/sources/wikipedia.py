"""Wikipedia pageviews em pt. Sem chave, sem conta.

Serve de **confirmacao**, nao de descoberta: a API so publica o dia fechado, com
atraso de ate ~48h. Um assunto que ja esta no topo aqui provavelmente ja passou
do pico de novidade. O valor e outro -- distinguir termo que o Trends mostra em
alta momentanea de assunto com interesse real e sustentado em portugues.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from agent.models import Signal
from agent.ports.radar import SourceUnavailable

ENDPOINT = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
HEADERS = {"User-Agent": "tiktok-viral-generator/0.1 (github.com/guilhermehrsilva)"}

# Paginas de servico do proprio projeto: sempre no topo, nunca sao assunto.
_PREFIXOS_IGNORADOS = (
    "Especial:", "Wikipédia:", "Wikipedia:", "Ajuda:", "Portal:",
    "Categoria:", "Ficheiro:", "Predefinição:", "Anexo:",
)
_TITULOS_IGNORADOS = {"Página_principal", "Main_Page"}

# A Wikipedia tem artigos de um caractere ("Q", "A"). Sao titulos legitimos la e
# inuteis como tema de video -- e estouram o min_length do contrato Signal. Um
# unico artigo assim derrubava a fonte inteira por ValidationError.
_TAMANHO_MINIMO_TERMO = 2


class WikipediaPageviews:
    name = "wikipedia"

    def __init__(self, client: httpx.Client | None = None, project: str = "pt.wikipedia",
                 top: int = 40):
        self._client = client or httpx.Client(timeout=httpx.Timeout(20.0), headers=HEADERS)
        self._project = project
        self._top = top

    def collect(self) -> list[Signal]:
        # A API so publica dia fechado; ontem costuma ainda nao existir.
        dia = datetime.now(UTC) - timedelta(days=2)
        url = f"{ENDPOINT}/{self._project}/all-access/{dia:%Y/%m/%d}"
        try:
            r = self._client.get(url, headers=HEADERS)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"wikipedia inacessivel: {exc}") from exc
        if r.status_code != 200:
            raise SourceUnavailable(f"wikipedia devolveu {r.status_code} para {dia:%Y-%m-%d}")
        try:
            payload = r.json()
        except ValueError as exc:
            raise SourceUnavailable("wikipedia devolveu resposta nao-JSON") from exc
        return self.parse(payload, now=datetime.now(UTC), top=self._top)

    @staticmethod
    def parse(payload: dict, now: datetime, top: int = 40) -> list[Signal]:
        itens = payload.get("items") or []
        if not itens:
            raise SourceUnavailable("wikipedia devolveu payload sem 'items'")

        sinais: list[Signal] = []
        for artigo in itens[0].get("articles") or []:
            titulo = (artigo.get("article") or "").strip()
            if not titulo or titulo in _TITULOS_IGNORADOS:
                continue
            if titulo.startswith(_PREFIXOS_IGNORADOS):
                continue
            termo = titulo.replace("_", " ").strip()
            if len(termo) < _TAMANHO_MINIMO_TERMO:
                continue
            views = artigo.get("views")
            if views is None:
                continue

            sinais.append(
                Signal(
                    term=termo,
                    source=WikipediaPageviews.name,
                    volume=float(views),
                    unit="pageviews",
                    # Nivel diario, nao taxa: a velocidade vem da coleta anterior.
                    velocity=None,
                    seen_at=now,
                    url=f"https://pt.wikipedia.org/wiki/{titulo}",
                )
            )
            if len(sinais) >= top:
                break
        return sinais
