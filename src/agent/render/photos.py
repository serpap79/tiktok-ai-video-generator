"""Fotos ilustrativas para os slides: Pexels pela tag do pilar ou pelo assunto.

Cada slide ja carrega um `visual` (tag verbatim do vocabulario do canal), e o
carrossel pode trazer `broll` (o objeto concreto do assunto, em ingles). A
capa e o miolo usam o assunto; os outros slides, a identidade.

Escolha medida, nao "a primeira foto": o Pexels devolve `alt` (descricao da
foto), e a nota soma relevancia (palavras do termo no `alt`) com escuridao
(`avg_color` que a propria API informa -- sem baixar miniatura). O carrossel
de 19/09 tinha microscopio e cockpit atras de "passkeys": era a primeira
foto de uma tag generica.

Sem rede ou sem chave, o slide sai so com layout -- foto e enriquecimento,
nunca requisito que trave o render. Cache em disco por termo+foto.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

PEXELS_SEARCH = "https://api.pexels.com/v1/search"
_PALAVRA = re.compile(r"[a-z0-9]+")


def _chave() -> str:
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        from agent.config import settings
        key = settings.pexels_api_key
    if not key:
        raise ValueError("sem chave Pexels (PEXELS_API_KEY ou AGENT_PEXELS_API_KEY); "
                         "slides saem sem foto")
    return key


def search(query: str, *, per_page: int = 3) -> list[dict]:
    """Fotos portrait para a tag. Devolve dicts crus do Pexels."""
    params = urllib.parse.urlencode({
        "query": query, "orientation": "portrait", "per_page": per_page})
    req = urllib.request.Request(
        f"{PEXELS_SEARCH}?{params}",
        headers={"Authorization": _chave(),
                 # A API barra o UA padrao do urllib (403): sem isso, toda
                 # busca falha em silencio e o slide sai sem foto.
                 "User-Agent": "tiktok-viral-generator/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        corpo = json.load(r)
    return list(corpo.get("photos", []))


def portrait_url(photo: dict, w: int = 1080, h: int = 1920) -> str:
    """URL ja no corte 9:16 (a API do Pexels redimensiona por parametro)."""
    base = photo.get("src", {}).get("portrait", "")
    if not base:
        raise ValueError("foto sem src.portrait")
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}auto=compress&w={w}&h={h}&fit=crop"


def darkness(photo: dict) -> float:
    """1 - luminancia do `avg_color` que a API ja devolve (0 claro, 1 escuro)."""
    cor = str(photo.get("avg_color") or "").lstrip("#")
    if len(cor) != 6:
        return 0.5
    r, g, b = (int(cor[i:i + 2], 16) for i in (0, 2, 4))
    return round(1 - (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255, 3)


def relevance(query: str, photo: dict) -> float:
    alvo = {p for p in _PALAVRA.findall(query.lower()) if len(p) > 2}
    if not alvo:
        return 0.0
    texto = f"{photo.get('alt') or ''} {photo.get('url') or ''}".lower()
    presentes = set(_PALAVRA.findall(texto))
    return round(len(alvo & presentes) / len(alvo), 3)


def rank(query: str, photos: list[dict], *, subject: bool,
         exclude: set[int] | None = None) -> list[dict]:
    """Fotos ordenadas pela nota; assunto pesa relevancia, pilar pesa escuridao."""
    fora = exclude or set()

    def nota(p: dict) -> float:
        rel, esc = relevance(query, p), darkness(p)
        return 0.7 * rel + 0.3 * esc if subject else 0.35 * rel + 0.65 * esc

    return sorted((p for p in photos if int(p.get("id") or 0) not in fora),
                  key=lambda p: -nota(p))


def fetch(query: str, dest_dir: str | Path, *, subject: bool = False,
          exclude: set[int] | None = None, used: set[int] | None = None) -> Path | None:
    """Descarga la mejor foto del termino a la cache. None si falla.

    `used` recibe el id elegido: quien llama pasa el mismo conjunto para los
    cinco slides, y la misma foto no aparece dos veces en el carrusel.

    La cache se consulta ANTES de la red: un `{slug}.jpg` ya presente (sembrado
    a mano o de una corrida anterior) evita la llamada al API -- sin clave o con
    el API caido, el slide sale con su foto en vez de con layout desnudo.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in query.lower()).strip("-")
    sembrada = Path(dest_dir) / f"{slug}.jpg"
    if sembrada.exists():
        return sembrada
    try:
        fotos = rank(query, search(query, per_page=12), subject=subject,
                     exclude=(exclude or set()) | (used or set()))
        if not fotos:
            return None
        foto = fotos[0]
        pid = int(foto.get("id") or 0)
        destino = Path(dest_dir) / f"{slug}-{pid}.jpg"
        if used is not None:
            used.add(pid)
        if destino.exists():
            return destino
        req = urllib.request.Request(
            portrait_url(foto),
            headers={"User-Agent": "tiktok-viral-generator/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(destino, "wb") as f:
            f.write(r.read())
        return destino
    except Exception:
        return None


__all__ = ["darkness", "fetch", "portrait_url", "rank", "relevance", "search"]
