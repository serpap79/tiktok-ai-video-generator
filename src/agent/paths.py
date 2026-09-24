"""Donde vive cada artefacto: dia/hora/tema/formato, siempre.

`output/2026-09-19/1430-bonsai-2-27b-largo/roteiro.json` se encuentra sin grep, y
el tema repetido aparece en el listado antes de convertirse en video repetido.
El formato cierra el trio de la rutina (long/short/carousel) en la propia ruta.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

_SUCIO = re.compile(r"[^a-z0-9]+")


def slugify(topic: str, maximo: int = 40) -> str:
    base = _SUCIO.sub("-", topic.lower()).strip("-")
    return base[:maximo].rstrip("-") or "tema"


def run_dir(formato: str, topic: str, ahora: datetime | None = None,
            base: Path | str = "output") -> Path:
    ahora = ahora or datetime.now(UTC)
    dia = ahora.strftime("%Y-%m-%d")
    hora = ahora.strftime("%H%M")
    return Path(base) / dia / f"{hora}-{slugify(topic)}-{formato}"


__all__ = ["run_dir", "slugify"]
