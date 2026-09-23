"""Google Trends via feed RSS de buscas em alta. Sem chave, sem conta.

Cobre o Brasil, que o Hacker News nao cobre, e entrega **as materias ja
associadas a cada tema** -- titulo, veiculo e URL. Isso adianta parte do
trabalho do pesquisador (M3) sem custar uma requisicao a mais.

Nao e API oficial e pode mudar sem aviso. A oficial seguia em alpha por
inscricao em ago/2026 e o `pytrends` foi arquivado em abr/2025, entao este feed
e o que existe de gratuito e estavel na pratica.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import httpx

from agent.models import NewsItem, Signal
from agent.ports.radar import SourceUnavailable

ENDPOINT = "https://trends.google.com/trending/rss"

# O feed nao e servido para clientes sem user agent de navegador.
HEADERS = {"User-Agent": "Mozilla/5.0"}

# "200+", "2 mil+", "1 mi+" -> o numero e uma ordem de grandeza, nao uma medida.
_TRAFEGO = re.compile(r"([\d.,]+)\s*(mil|mi|k|m)?", re.IGNORECASE)
_MULTIPLICADOR = {"mil": 1_000, "k": 1_000, "mi": 1_000_000, "m": 1_000_000}


class GoogleTrends:
    name = "google_trends"

    def __init__(self, client: httpx.Client | None = None, geo: str = "BR"):
        self._client = client or httpx.Client(timeout=httpx.Timeout(20.0), headers=HEADERS)
        self._geo = geo

    def collect(self) -> list[Signal]:
        try:
            r = self._client.get(ENDPOINT, params={"geo": self._geo}, headers=HEADERS)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"google trends inacessivel: {exc}") from exc
        if r.status_code != 200:
            raise SourceUnavailable(f"google trends devolveu {r.status_code}")
        return self.parse(r.content, now=datetime.now(UTC))

    @staticmethod
    def parse(xml_bytes: bytes, now: datetime) -> list[Signal]:
        try:
            raiz = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise SourceUnavailable(f"google trends devolveu XML invalido: {exc}") from exc

        canal = raiz.find("channel")
        if canal is None:
            raise SourceUnavailable("google trends devolveu RSS sem <channel>")

        sinais: list[Signal] = []
        for item in canal.findall("item"):
            termo = (item.findtext("title") or "").strip()
            # O feed tambem tem tema de uma letra ("p", visto em 19/09/2026):
            # estoura o min_length do Signal e derrubava a fonte inteira -- o
            # mesmo defeito que a Wikipedia teve com o artigo "Q".
            if len(termo) < 2:
                continue

            materias = []
            for ni in item.findall("{*}news_item"):
                titulo = (ni.findtext("{*}news_item_title") or "").strip()
                url = (ni.findtext("{*}news_item_url") or "").strip()
                if titulo and url:
                    materias.append(
                        NewsItem(
                            title=titulo,
                            url=url,
                            source_name=(ni.findtext("{*}news_item_source") or "").strip(),
                        )
                    )

            sinais.append(
                Signal(
                    term=termo,
                    source=GoogleTrends.name,
                    volume=_parse_trafego(item.findtext("{*}approx_traffic")),
                    unit="searches",
                    # O feed da o nivel aproximado, nunca a taxa. A velocidade sai
                    # da comparacao com a coleta anterior, feita pelo coletor.
                    velocity=None,
                    seen_at=now,
                    news_items=materias,
                )
            )
        return sinais


def _parse_trafego(bruto: str | None) -> float:
    """Converte "2 mil+" em 2000.0. Devolve 0.0 quando o feed nao informa."""
    if not bruto:
        return 0.0
    m = _TRAFEGO.search(bruto.strip())
    if not m:
        return 0.0
    numero = m.group(1).replace(".", "").replace(",", ".")
    try:
        valor = float(numero)
    except ValueError:
        return 0.0
    sufixo = (m.group(2) or "").lower()
    return valor * _MULTIPLICADOR.get(sufixo, 1)
