"""GDELT DOC 2.0: volumen de cobertura periodística mundial. Sin clave.

Entra con disyuntor porque **es inestable en la práctica**: devolvió 429 en 2 de 3
intentos en las pruebas de viabilidad, sin clave y sin cuota publicada. Tras varios
fallos consecutivos, la fuente se desactiva durante el resto de la ejecución para no
gastar el presupuesto de tiempo de la recolección golpeando una puerta cerrada.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from agent.models import Signal
from agent.ports.radar import SourceUnavailable

ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# Términos del nicho. GDELT no ofrece «qué está en tendencia»: responde consultas,
# por lo que el recorte temático debe proceder de nosotros.
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
                # Disyuntor abierto: deja de intentarlo, pero devuelve lo ya recolectado.
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
                # Una respuesta 200 con cuerpo vacío es habitual cuando la consulta no coincide.
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
                f"gdelt no disponible tras {falhas} fallos (429 es habitual sin clave)"
            )
        return sinais
