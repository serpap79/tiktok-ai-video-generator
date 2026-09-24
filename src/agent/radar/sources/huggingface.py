"""Modelos en tendencia en Hugging Face. Sin clave.

La fuente más directa de «qué está descargando ahora la comunidad de IA»: el
Hub publica `trendingScore` por modelo, que es una tasa (me gusta recientes), no
nivel -- velocidade nativa, como a do HN. Em 19/09/2026 o primeiro da lista
era o Ternary Bonsai 2 27B, o mesmo tema que o radar tinha achado pelo HN
dois dias antes.

Filtro que importa: variantes quantizadas (GGUF, AWQ, MLX...) dominam a
lista y son el mismo modelo. El nombre se normaliza sin el sufijo y las
variantes del mismo modelo se convierten en una sola señal.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import httpx

from agent.models import Signal
from agent.ports.radar import SourceUnavailable

ENDPOINT = "https://huggingface.co/api/models"
HEADERS = {"User-Agent": "tiktok-viral-generator/0.1 (radar)"}

_QUANT = re.compile(
    r"[-_.](gguf|awq|gptq|exl2|mlx|bnb|4bit|8bit|fp8|int4|int8|q4|q8|onnx)\b.*$",
    re.IGNORECASE)
# Menos que isso e experimento pessoal que entrou na lista por pouco.
MIN_LIKES = 50
# Un modelo de hace seis meses en tendencia es reposición, no noticia.
MAX_IDADE_DIAS = 60


def display_name(model_id: str) -> str:
    """'prism-ml/Ternary-Bonsai-2-27B-gguf' -> 'Ternary Bonsai 2 27B'."""
    nome = model_id.split("/", 1)[-1]
    nome = _QUANT.sub("", nome)
    return " ".join(nome.replace("_", " ").replace("-", " ").split())


class HuggingFaceTrending:
    name = "huggingface"

    def __init__(self, client: httpx.Client | None = None, limit: int = 30):
        self._client = client or httpx.Client(timeout=httpx.Timeout(15.0), headers=HEADERS)
        self._limit = limit

    def collect(self) -> list[Signal]:
        try:
            r = self._client.get(ENDPOINT, params={"sort": "trendingScore",
                                                    "limit": str(self._limit)})
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"hugging face inacessivel: {exc}") from exc
        if r.status_code != 200:
            raise SourceUnavailable(f"hugging face devolveu {r.status_code}")
        try:
            modelos = r.json()
        except ValueError as exc:
            raise SourceUnavailable("hugging face devolvió una respuesta que no es JSON") from exc
        return self.parse(modelos, now=datetime.now(UTC))

    @staticmethod
    def parse(modelos: list[dict], now: datetime) -> list[Signal]:
        vistos: set[str] = set()
        sinais: list[Signal] = []
        for m in modelos or []:
            mid = str(m.get("id") or "")
            likes = m.get("likes") or 0
            score = m.get("trendingScore")
            criado = m.get("createdAt") or ""
            if not mid or score is None or likes < MIN_LIKES:
                continue
            try:
                quando = datetime.fromisoformat(criado.replace("Z", "+00:00"))
            except ValueError:
                continue
            if now - quando > timedelta(days=MAX_IDADE_DIAS):
                continue
            nome = display_name(mid)
            chave = nome.casefold()
            if len(nome) < 3 or chave in vistos:
                continue
            vistos.add(chave)
            autor = mid.split("/", 1)[0] if "/" in mid else ""
            sinais.append(Signal(
                term=f"{nome} ({autor})" if autor else nome,
                source=HuggingFaceTrending.name,
                volume=float(likes), unit="likes",
                velocity=float(score),
                seen_at=now,
                url=f"https://huggingface.co/{mid}",
            ))
        return sinais


__all__ = ["HuggingFaceTrending", "display_name"]
