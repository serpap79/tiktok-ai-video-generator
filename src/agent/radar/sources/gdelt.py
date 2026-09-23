"""GDELT DOC 2.0: volume de cobertura jornalistica global. Sem chave.

Entra com disjuntor porque **e instavel na pratica**: devolveu 429 em 2 de 3
tentativas nos testes de viabilidade, sem chave e sem cota publicada. Depois de
algumas falhas seguidas a fonte e desligada pelo resto da execucao, para nao
gastar o orcamento de tempo da coleta batendo numa porta fechada.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from agent.models import Signal
from agent.ports.radar import SourceUnavailable

ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# Termos do nicho. O GDELT nao tem "o que esta em alta": ele responde consultas,
# entao o recorte tematico precisa vir de nos.
CONSULTAS_PADRAO = (
    "artificial intelligence",
    "quantum computing",
    "space launch",
    "semiconductor",
)


class Gdelt:
    name = "gdelt"

    def __init__(self, client: httpx.Client | None = None,
                 queries: tuple[str, ...] = CONSULTAS_PADRAO,
                 max_falhas: int = 2):
        self._client = client or httpx.Client(timeout=httpx.Timeout(25.0))
        self._queries = queries
        self._max_falhas = max_falhas

    def collect(self) -> list[Signal]:
        agora = datetime.now(UTC)
        sinais: list[Signal] = []
        falhas = 0

        for consulta in self._queries:
            if falhas >= self._max_falhas:
                # Disjuntor aberto: para de tentar, mas devolve o que ja coletou.
                break
            try:
                r = self._client.get(ENDPOINT, params={
                    "query": consulta, "mode": "artlist", "maxrecords": "20",
                    "format": "json", "timespan": "1d",
                })
            except httpx.HTTPError:
                falhas += 1
                continue

            if r.status_code != 200:
                falhas += 1
                continue
            try:
                artigos = (r.json() or {}).get("articles") or []
            except ValueError:
                # 200 com corpo vazio e comum aqui quando a consulta nao casa.
                falhas += 1
                continue

            if artigos:
                sinais.append(Signal(
                    term=consulta,
                    source=self.name,
                    volume=float(len(artigos)),
                    unit="articles",
                    velocity=None,
                    seen_at=agora,
                ))

        if not sinais and falhas >= self._max_falhas:
            raise SourceUnavailable(
                f"gdelt indisponivel apos {falhas} falhas (429 e comum sem chave)"
            )
        return sinais
