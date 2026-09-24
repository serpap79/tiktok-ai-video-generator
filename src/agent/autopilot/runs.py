"""Libro de los slots: qué produjo o publicó cada horario y por qué falló.

Una fila única por (día, slot). Esto hace idempotente el piloto: el temporizador
de systemd puede dispararse de nuevo (máquina encendida, `Persistent=true`) y un
slot ya publicado no puede convertirse en dos posts. También es lo que lee el planificador
para variar o dia -- formato, tipo e tema ja usados hoje.

Estados:
- `started`: en producción (o se interrumpió: se puede retomar);
- `produced`: paquete listo, pendiente de la hora de publicación;
- `published`: en la bandeja de entrada de TikTok (vídeo);
- `ready_manual`: paquete listo para publicar manualmente (carrusel sin alojamiento
  verificado: la API de fotos solo acepta URL de un dominio verificado);
- `failed`: no se publicó, con el motivo;
- `skipped`: ya estaba hecho.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS slot_runs (
    id           INTEGER PRIMARY KEY,
    day          TEXT NOT NULL,
    slot         TEXT NOT NULL,
    state        TEXT NOT NULL,
    topic        TEXT,
    source       TEXT,
    pillar       TEXT,
    format       TEXT,
    plan_json    TEXT,
    package_dir  TEXT,
    publish_id   TEXT,
    error        TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    UNIQUE(day, slot)
);
"""

FINAIS = ("published", "ready_manual")


class SlotRuns:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
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

    def get(self, day: str, slot: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM slot_runs WHERE day = ? AND slot = ?",
                               (day, slot)).fetchone()
        return dict(row) if row else None

    def begin(self, day: str, slot: str, *, force: bool = False) -> bool:
        """Abre (o reabre) el slot. False si ya terminó y no debe rehacerse."""
        atual = self.get(day, slot)
        if atual and atual["state"] in FINAIS and not force:
            return False
        agora = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO slot_runs (day, slot, state, started_at) VALUES (?, ?, 'started', ?)"
                " ON CONFLICT(day, slot) DO UPDATE SET state = 'started', error = NULL,"
                " started_at = excluded.started_at, finished_at = NULL",
                (day, slot, agora))
        return True

    def update(self, day: str, slot: str, **campos) -> None:
        if not campos:
            return
        if "plan" in campos:
            campos["plan_json"] = json.dumps(campos.pop("plan"), ensure_ascii=False, default=str)
        if campos.get("state") in (*FINAIS, "failed", "skipped"):
            campos.setdefault("finished_at", datetime.now(UTC).isoformat())
        colunas = ", ".join(f"{k} = ?" for k in campos)
        with self._conn() as conn:
            conn.execute(f"UPDATE slot_runs SET {colunas} WHERE day = ? AND slot = ?",
                         (*campos.values(), day, slot))

    def day(self, day: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM slot_runs WHERE day = ? ORDER BY slot",
                                (day,)).fetchall()
        return [dict(r) for r in rows]

    def used_today(self, day: str, except_slot: str = "") -> dict[str, list[str]]:
        """Formatos, tipos y temas que el día ya ha producido (para variar)."""
        feitos = [r for r in self.day(day)
                  if r["slot"] != except_slot and r["state"] in (*FINAIS, "produced")]
        return {
            "formats": [r["format"] for r in feitos if r["format"]],
            "pillars": [r["pillar"] for r in feitos if r["pillar"]],
            "topics": [r["topic"] for r in feitos if r["topic"]],
        }


__all__ = ["FINAIS", "SlotRuns"]
