"""Pageviews de Wikipedia en espanol. Sin clave, sin cuenta.

Sirve de **confirmacion**, no de descubrimiento: la API solo publica el dia
cerrado, con retraso de hasta ~48h. Un asunto que ya esta arriba aqui
probablemente ya paso el pico de novedad. El valor es otro -- distinguir termino
que Trends muestra en alta momentanea de asunto con interes real y sostenido
en castellano.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from agent.models import Signal
from agent.ports.radar import SourceUnavailable

ENDPOINT = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
HEADERS = {"User-Agent": "tiktok-viral-generator/0.1"}

# Paginas de servicio de la propia enciclopedia: siempre arriba, nunca son
# asunto. En la Wikipedia en espanol el namespace de proyectos es "Wikipedia:"
# y la ayuda es "Ayuda:"; "Ficheiro" y "Predefinição" son de la edicion en
# portugues y se conservan por si el proyecto cambia.
_PREFIXOS_IGNORADOS = (
    "Especial:", "Wikipedia:", "Wikipédia:", "Ayuda:", "Portal:",
    "Categoría:", "Categoria:", "Archivo:", "Plantilla:", "Ficheiro:",
    "Predefinição:", "Anexo:",
)
_TITULOS_IGNORADOS = {"Portada", "Página_principal", "Main_Page"}

# La Wikipedia tiene articulos de un caracter ("Q", "A"). Son titulos legitimos
# alli e inutiles como tema de video -- y rebasan el min_length del contrato
# Signal. Un solo articulo asi tiraba abajo la fuente entera por ValidationError.
_TAMANHO_MINIMO_TERMO = 2


class WikipediaPageviews:
    name = "wikipedia"

    def __init__(self, client: httpx.Client | None = None, project: str = "es.wikipedia",
                 top: int = 40):
        self._client = client or httpx.Client(timeout=httpx.Timeout(20.0), headers=HEADERS)
        self._project = project
        self._top = top

    def collect(self) -> list[Signal]:
        # La API solo publica el dia cerrado; ayer suele todavia no existir.
        dia = datetime.now(UTC) - timedelta(days=2)
        url = f"{ENDPOINT}/{self._project}/all-access/{dia:%Y/%m/%d}"
        try:
            r = self._client.get(url, headers=HEADERS)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"wikipedia inaccesible: {exc}") from exc
        if r.status_code != 200:
            raise SourceUnavailable(f"wikipedia devolvio {r.status_code} para {dia:%Y-%m-%d}")
        try:
            payload = r.json()
        except ValueError as exc:
            raise SourceUnavailable("wikipedia devolvio respuesta no-JSON") from exc
        return self.parse(payload, now=datetime.now(UTC), top=self._top)

    @staticmethod
    def parse(payload: dict, now: datetime, top: int = 40) -> list[Signal]:
        itens = payload.get("items") or []
        if not itens:
            raise SourceUnavailable("wikipedia devolvio payload sin 'items'")

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
                    # Nivel diario, no tasa: la velocidad viene de la recolecta anterior.
                    velocity=None,
                    seen_at=now,
                    url=f"https://es.wikipedia.org/wiki/{titulo}",
                )
            )
            if len(sinais) >= top:
                break
        return sinais
