"""Onde cada artefato mora: dia/hora/tema/formato, sempre.

`output/2026-09-19/1430-bonsai-2-27b-long/roteiro.json` se acha sem grep, e
o tema repetido aparece na listagem antes de virar video repetido. O formato
fecha o trio da rotina (long/short/carousel) no proprio caminho.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

_SUJO = re.compile(r"[^a-z0-9]+")


def slugify(topic: str, máximo: int = 40) -> str:
    base = _SUJO.sub("-", topic.lower()).strip("-")
    return base[:máximo].rstrip("-") or "tema"


def run_dir(formato: str, topic: str, agora: datetime | None = None,
            base: Path | str = "output") -> Path:
    agora = agora or datetime.now(UTC)
    dia = agora.strftime("%Y-%m-%d")
    hora = agora.strftime("%H%M")
    return Path(base) / dia / f"{hora}-{slugify(topic)}-{formato}"


__all__ = ["run_dir", "slugify"]
