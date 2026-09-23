"""Hacker News via API do Algolia. Sem chave, sem conta.

Fonte primaria do nicho: e a unica gratuita que entrega **velocidade nativa**.
`points` e `created_at_i` juntos dao pontos por hora sem precisar de coleta
anterior, entao a primeira execucao ja produz sinal util -- as outras fontes
precisam de pelo menos duas coletas para dizer qualquer coisa sobre movimento.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from agent.models import Signal
from agent.ports.radar import SourceUnavailable

ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"

# Abaixo disso e ruido: o HN tem muita submissao que nunca sai do /new.
MIN_POINTS = 100

# Uma historia recem-postada com poucos pontos produz pontos/hora altissimo por
# artefato de divisao. Meia hora de piso mata esse falso positivo.
MIN_AGE_HOURS = 0.5


class HackerNews:
    name = "hacker_news"

    def __init__(self, client: httpx.Client | None = None, hits: int = 50):
        self._client = client or httpx.Client(timeout=httpx.Timeout(15.0))
        self._hits = hits

    def collect(self) -> list[Signal]:
        params = {
            "tags": "story",
            "numericFilters": f"points>{MIN_POINTS}",
            "hitsPerPage": str(self._hits),
        }
        try:
            r = self._client.get(ENDPOINT, params=params)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"hacker news inacessivel: {exc}") from exc
        if r.status_code != 200:
            raise SourceUnavailable(f"hacker news devolveu {r.status_code}")
        try:
            payload = r.json()
        except ValueError as exc:
            raise SourceUnavailable("hacker news devolveu resposta nao-JSON") from exc

        return self.parse(payload, now=datetime.now(UTC))

    @staticmethod
    def parse(payload: dict, now: datetime) -> list[Signal]:
        sinais: list[Signal] = []
        for hit in payload.get("hits") or []:
            titulo = (hit.get("title") or "").strip()
            pontos = hit.get("points")
            criado = hit.get("created_at_i")
            if not titulo or pontos is None or not criado:
                continue

            idade_h = max((now.timestamp() - float(criado)) / 3600.0, MIN_AGE_HOURS)
            story_id = hit.get("objectID")
            sinais.append(
                Signal(
                    term=titulo,
                    source=HackerNews.name,
                    volume=float(pontos),
                    unit="points",
                    velocity=round(float(pontos) / idade_h, 2),
                    seen_at=now,
                    url=f"https://news.ycombinator.com/item?id={story_id}" if story_id else None,
                )
            )
        return sinais
