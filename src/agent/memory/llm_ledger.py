"""Livro de cota dos LLMs: cada chamada, e quem esta esgotado ate quando.

Existe porque la cuota del nivel gratuito es el recurso escaso del proyecto, no
dinheiro. Medido em 19/09/2026: o gemini-2.5-flash tem 20 pedidos por dia no
el nivel gratuito. Un solo vídeo (investigación + guion + juez) se acerca a ese límite.
Com tres posts por dia, "qual modelo ainda tem cota agora?" precisa de
una respuesta grabada y no ensayo y error en cada etapa.

Duas tabelas no mesmo banco do resto da memoria:

- `llm_calls`: toda llamada, incluso la denegada, con etapa y coste. Permite
  saber si el prompt del investigador es caro o si el juez consume demasiada cuota
  demais -- e a fonte do relatorio diario.
- `llm_quota`: modelo esgotado e ate quando. Cada slot roda num processo
  proprio (timer do systemd), e sem isto o slot das 20h redescobriria, pagando
  un 429 por modelo, información que la ejecución de las 9:00 ya conocía.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY,
    stage         TEXT    NOT NULL,
    route         TEXT    NOT NULL,
    ok            INTEGER NOT NULL,
    error         TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_s     REAL    NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_route ON llm_calls(route, created_at DESC);

CREATE TABLE IF NOT EXISTS llm_quota (
    route           TEXT PRIMARY KEY,
    exhausted_until TEXT NOT NULL,
    reason          TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""


class LLMLedger:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------ chamadas

    def record_call(self, stage: str, route: str, *, ok: bool, error: str = "",
                    input_tokens: int = 0, output_tokens: int = 0,
                    latency_s: float = 0.0, at: datetime | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO llm_calls (stage, route, ok, error, input_tokens,"
                " output_tokens, latency_s, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (stage, route, int(ok), error[:500] or None, input_tokens,
                 output_tokens, latency_s, (at or datetime.now(UTC)).isoformat()),
            )

    def calls_since(self, since: datetime) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT stage, route, ok, error, input_tokens, output_tokens, latency_s,"
                " created_at FROM llm_calls WHERE created_at >= ? ORDER BY id",
                (since.isoformat(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def usage_by_route(self, since: datetime) -> dict[str, dict[str, int]]:
        """Pedidos, falhas e tokens por modelo desde `since` (relatorio diario)."""
        saida: dict[str, dict[str, int]] = {}
        for c in self.calls_since(since):
            linha = saida.setdefault(c["route"], {"calls": 0, "failed": 0, "tokens": 0})
            linha["calls"] += 1
            linha["failed"] += 0 if c["ok"] else 1
            linha["tokens"] += int(c["input_tokens"]) + int(c["output_tokens"])
        return saida

    # ------------------------------------------------------------ cota

    def mark_exhausted(self, route: str, until: datetime, reason: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO llm_quota (route, exhausted_until, reason, updated_at)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(route) DO UPDATE SET"
                " exhausted_until = excluded.exhausted_until,"
                " reason = excluded.reason, updated_at = excluded.updated_at",
                (route, until.isoformat(), reason[:300], datetime.now(UTC).isoformat()),
            )

    def exhausted_until(self, route: str, now: datetime | None = None) -> datetime | None:
        """Ate quando o modelo esta esgotado, ou None se tem cota (ou nao sabemos)."""
        agora = now or datetime.now(UTC)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT exhausted_until FROM llm_quota WHERE route = ?", (route,)
            ).fetchone()
        if row is None:
            return None
        ate = _parse(row["exhausted_until"])
        return ate if ate > agora else None

    def quota_status(self, now: datetime | None = None) -> list[dict]:
        agora = now or datetime.now(UTC)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT route, exhausted_until, reason FROM llm_quota ORDER BY route"
            ).fetchall()
        return [dict(r) for r in rows if _parse(r["exhausted_until"]) > agora]


def next_pacific_midnight(now: datetime | None = None) -> datetime:
    """Reset diario del free tier de Gemini: medianoche en el horario del Pacífico.

    Documentado en ai.google.dev/gemini-api/docs/rate-limits ("RPD quotas
    reset at midnight Pacific time"). En hora de España eso cae a las 8h u 9h --
    antes del slot de las 10h, así que el día empieza siempre con cuota llena.
    """
    from zoneinfo import ZoneInfo

    pacifico = ZoneInfo("America/Los_Angeles")
    agora = (now or datetime.now(UTC)).astimezone(pacifico)
    amanha = (agora + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return amanha.astimezone(UTC)


def _parse(valor: str) -> datetime:
    dt = datetime.fromisoformat(valor)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


__all__ = ["LLMLedger", "next_pacific_midnight"]
