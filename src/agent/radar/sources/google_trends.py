"""Google Trends via feed RSS de busquedas en tendencia. Sin clave, sin cuenta.

Cubre Espana, que Hacker News no cubre, y entrega **las noticias ya asociadas
a cada tema** -- titulo, medio y URL. Eso adelanta parte del trabajo del
investigador (M3) sin costar una peticion mas.

No es API oficial y puede cambiar sin aviso. La oficial seguia en alpha por
inscripcion en ago/2026 y `pytrends` fue archivado en abr/2025, asi que este
feed es lo que existe de gratuito y estable en la practica.
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

# "200+", "2 mil+", "1 M+" -> el numero es un orden de magnitud, no una medida.
# El feed ES escribe "2 mil+" y "1 M+"; el PT escribía "mi". Aceptamos ambos.
_TRAFEGO = re.compile(r"([\d.,]+)\s*(mil|mi|m|k)?", re.IGNORECASE)
_MULTIPLICADOR = {"mil": 1_000, "k": 1_000, "mi": 1_000_000, "m": 1_000_000}


class GoogleTrends:
    name = "google_trends"

    def __init__(self, client: httpx.Client | None = None, geo: str = "ES"):
        self._client = client or httpx.Client(timeout=httpx.Timeout(20.0), headers=HEADERS)
        self._geo = geo

    def collect(self) -> list[Signal]:
        try:
            r = self._client.get(ENDPOINT, params={"geo": self._geo}, headers=HEADERS)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"google trends inaccesible: {exc}") from exc
        if r.status_code != 200:
            raise SourceUnavailable(f"google trends devolvio {r.status_code}")
        return self.parse(r.content, now=datetime.now(UTC))

    @staticmethod
    def parse(xml_bytes: bytes, now: datetime) -> list[Signal]:
        try:
            raiz = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise SourceUnavailable(f"google trends devolvio XML invalido: {exc}") from exc

        canal = raiz.find("channel")
        if canal is None:
            raise SourceUnavailable("google trends devolvio RSS sin <channel>")

        sinais: list[Signal] = []
        for item in canal.findall("item"):
            termo = (item.findtext("title") or "").strip()
            # El feed tambien tiene tema de una letra ("p", visto en 19/09/2026):
            # rebasa el min_length del Signal y tiraba abajo la fuente entera --
            # el mismo defecto que tuvo Wikipedia con el articulo "Q".
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
                    # El feed da el nivel aproximado, nunca la tasa. La velocidad
                    # sale de la comparacion con la recolecta anterior, hecha
                    # por el colector.
                    velocity=None,
                    seen_at=now,
                    news_items=materias,
                )
            )
        return sinais


def _parse_trafego(bruto: str | None) -> float:
    """Convierte "2 mil+" en 2000.0. Devuelve 0.0 cuando el feed no informa."""
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
