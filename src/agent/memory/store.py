"""Memoria del agente en SQLite.

Cosas distintas, y no un almacén genérico: la **serie de señales** (del radar), el
**libro de temas** (del curador), los **dossiers** (del investigador) y los
**guiones** (del guionista). Cada una responde a una pregunta distinta --
"¿está subiendo?", "¿ya hablamos de esto?", "¿qué sabemos y de dónde?", "¿qué se
escribió y a qué coste?" -- y mezclarlas todas en una tabla de documentos haría
imposible responder cualquiera de ellas por SQL.

La serie de señales es lo que permite calcular velocidad
para fuentes que reportan nivel y no tasa (Wikipedia, Google Trends). Sin
historial, "500 mil pageviews" es un número sin significado: no se puede saber si
el asunto está subiendo o ya pasó.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.models import Carousel, CarouselReview, Decision, Dossier, Review, Script, Signal

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY,
    key         TEXT    NOT NULL,
    term        TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    volume      REAL    NOT NULL,
    unit        TEXT    NOT NULL,
    velocity    REAL,
    url         TEXT,
    seen_at     TEXT    NOT NULL
);
-- A consulta quente e "ultima observacao desta chave antes de agora".
CREATE INDEX IF NOT EXISTS idx_signals_key_seen ON signals(key, seen_at DESC);

CREATE TABLE IF NOT EXISTS topics (
    id          INTEGER PRIMARY KEY,
    term        TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    verdict     TEXT    NOT NULL,
    reason      TEXT    NOT NULL,
    score       REAL    NOT NULL,
    niche_fit   REAL    NOT NULL,
    url         TEXT,
    duplicate_of TEXT,
    decided_at  TEXT    NOT NULL
);
-- O ledger e sempre lido por "temas aprovados recentemente", nunca inteiro.
CREATE INDEX IF NOT EXISTS idx_topics_verdict_decided
    ON topics(verdict, decided_at DESC);

CREATE TABLE IF NOT EXISTS dossiers (
    id            INTEGER PRIMARY KEY,
    topic         TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    fact_count    INTEGER NOT NULL,
    source_count  INTEGER NOT NULL,
    -- Coste medido durante la investigación. Reconstruirlo después desde el log no sirve:
    -- el proveedor no devuelve consumo retroactivo y la evaluación M5 compara costes.
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_s     REAL    NOT NULL,
    -- JSON do Dossier inteiro. Guardar o contrato serializado, em vez de uma
    -- tabla de hechos normalizada, mantiene el dossier reproducible palabra por
    -- palabras, que es lo que el juez releerá en M5 para juzgar el mismo material.
    dossier_json  TEXT    NOT NULL,
    -- Hechos que las puertas derribaron, con motivo. Misma regla del libro de
    -- temas: sin lo descartado, solo se sabe lo que entró, nunca lo que se perdió.
    discarded_json TEXT   NOT NULL,
    failures_json TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dossiers_topic_created
    ON dossiers(topic, created_at DESC);

CREATE TABLE IF NOT EXISTS scripts (
    id            INTEGER PRIMARY KEY,
    topic         TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    dossier_id    INTEGER,
    word_count    INTEGER NOT NULL,
    -- Quantas tentativas o roteirista precisou para passar nos portoes
    -- mecánicos. Si cada ejecución gasta dos, el defecto está en el prompt, no en
    -- modelo -- e isso so aparece se o numero for gravado tambem no sucesso.
    attempts      INTEGER NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_s     REAL    NOT NULL,
    script_json   TEXT    NOT NULL,
    -- Violacoes de cada tentativa, inclusive as que foram corrigidas.
    attempts_json TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(id)
);
CREATE INDEX IF NOT EXISTS idx_scripts_topic_created
    ON scripts(topic, created_at DESC);

-- Tabla propia, y no columnas en `scripts`, porque el mismo guion puede ser
-- juzgado más de una vez: el eval del M5 compara jueces de modelos distintos
-- sobre el MISMO texto, y eso es una relación de uno a muchos.
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY,
    script_id     INTEGER,
    topic         TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    total         INTEGER NOT NULL,
    approved      INTEGER NOT NULL,
    -- Informe completo: nota y motivo de cada uno de los siete criterios. Es lo que
    -- permite, depois, agregar por criterio em vez de so pela soma.
    review_json   TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_s     REAL    NOT NULL,
    created_at    TEXT    NOT NULL,
    FOREIGN KEY (script_id) REFERENCES scripts(id)
);
CREATE INDEX IF NOT EXISTS idx_reviews_script ON reviews(script_id, created_at DESC);

-- Publicacao e uma relacao de um para muitos com o video: o mesmo MP4 pode ser
-- reenviado (ex. upload interrompido gerou outro publish_id), e o motivo de
-- cada tentativa precisa ficar gravado para calibrar -- mesma regra do ledger.
CREATE TABLE IF NOT EXISTS posts (
    id            INTEGER PRIMARY KEY,
    publish_id    TEXT    NOT NULL,
    video_path    TEXT    NOT NULL,
    status        TEXT    NOT NULL,
    error         TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_publish ON posts(publish_id, created_at DESC);

-- Métricas de la publicación, serie temporal: el mismo publish_id puede tener
-- varias coletas (views sobem com o tempo), e a curva e o unico sinal real de
-- viralidade. Leitura manual do app por enquanto: a Content Posting API, no
-- el ámbito video.upload de la bandeja no expone un endpoint de métricas. Research es
-- API está restringida a investigación académica. `script_id` cierra el lazo con el guion
-- quien generó el vídeo; NULL cuando no se conoce el vínculo.
CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY,
    publish_id      TEXT    NOT NULL,
    script_id       INTEGER,
    views           INTEGER NOT NULL,
    avg_watch_s     REAL,
    completion_rate REAL,
    collected_at    TEXT    NOT NULL,
    FOREIGN KEY (script_id) REFERENCES scripts(id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_publish ON metrics(publish_id, collected_at DESC);

-- Carrusel: guion de 5 slides + parecer incrustado. Parecer propio (y no
-- linha em `reviews`) porque a rubrica do carrossel tem 4 critérios, e
-- `reviews.review_json` valida como Review de 7. `format` em scripts existe
-- pelo mesmo motivo: o eval agrupa por formato sem desserializar JSON.
CREATE TABLE IF NOT EXISTS carousels (
    id            INTEGER PRIMARY KEY,
    topic         TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    approved      INTEGER NOT NULL,
    carousel_json TEXT    NOT NULL,
    review_json   TEXT,
    attempts_json TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_s     REAL    NOT NULL,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_carousels_topic_created
    ON carousels(topic, created_at DESC);
"""


class SignalStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Colunas de fatias novas em bancos criados por fatias velhas.

        CREATE TABLE IF NOT EXISTS no añade columnas a una tabla que ya
        existe. Sin esto, la base de datos real de la máquina (creada en M3) rechaza
        INSERT com `format` e o teste hermetico nunca acusaria, porque tmp
        sempre nasce do SCHEMA novo.
        """
        def cols(tabela: str) -> set[str]:
            return {r[1] for r in conn.execute(f"PRAGMA table_info({tabela})")}
        if "format" not in cols("scripts"):
            conn.execute("ALTER TABLE scripts ADD COLUMN format TEXT NOT NULL DEFAULT 'long'")
        for col in ("saves", "comments", "shares"):
            if col not in cols("metrics"):
                conn.execute(f"ALTER TABLE metrics ADD COLUMN {col} INTEGER")
        # Piloto automático: el post sabe de qué guion/formato/slot vino, y
        # es eso lo que deja las métricas volver al planificador de formato.
        for col, tipo in (("format", "TEXT"), ("script_id", "INTEGER"),
                          ("carousel_id", "INTEGER"), ("slot", "TEXT")):
            if col not in cols("posts"):
                conn.execute(f"ALTER TABLE posts ADD COLUMN {col} {tipo}")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(self, signals: Iterable[Signal]) -> int:
        rows = [
            (s.key, s.term, s.source, s.volume, s.unit, s.velocity,
             str(s.url) if s.url else None, s.seen_at.isoformat())
            for s in signals
        ]
        if not rows:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO signals (key, term, source, volume, unit, velocity, url, seen_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def previous(self, key: str, before: datetime) -> tuple[float, datetime] | None:
        """Ultima observacao desta chave antes de `before`, se houver."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT volume, seen_at FROM signals"
                " WHERE key = ? AND seen_at < ? ORDER BY seen_at DESC LIMIT 1",
                (key, before.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return float(row["volume"]), _parse_iso(row["seen_at"])

    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM signals").fetchone()["n"])

    # ------------------------------------------------------------ ledger de temas

    def record_decisions(self, decisions: Iterable[Decision]) -> int:
        """Graba toda decisión, aprobada o no.

        El rechazo guardado permite calibrar después la puntuación en lugar de
        adivinar: sin ella solo se sabe qué se eligió, nunca qué se perdió.
        """
        rows = [
            (d.term, d.source, d.verdict.value, d.reason, d.score, d.niche_fit,
             str(d.url) if d.url else None, d.duplicate_of, d.decided_at.isoformat())
            for d in decisions
        ]
        if not rows:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO topics (term, source, verdict, reason, score, niche_fit,"
                " url, duplicate_of, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def recent_topics(self, days: int = 30, limit: int = 500) -> list[str]:
        """Temas ja aprovados, para o deduplicador comparar.

        Solo entran los aprobados: un tema rechazado por política o por nicho no
        "ja foi coberto" -- ele nunca virou video, e bloquear o parecido seria
        estender o veto a assuntos que nunca foram julgados.
        """
        corte = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT term FROM topics WHERE verdict = 'selected' AND decided_at >= ?"
                " ORDER BY decided_at DESC LIMIT ?",
                (corte, limit),
            ).fetchall()
            # Lo que se volvió guion o carrusel también fue cubierto, incluso sin
            # haber pasado por el curador (`research --topic` y el piloto
            # automático escriben directo). Visto el 19/09: el short del cerebro
            # se produjo por la mañana y el tema volvió al topl tarde.
            produzidos = conn.execute(
                "SELECT topic FROM scripts WHERE created_at >= ?"
                " UNION SELECT topic FROM carousels WHERE created_at >= ?",
                (corte, corte),
            ).fetchall()
        termos = [r["term"] for r in rows]
        termos.extend(r["topic"] for r in produzidos if r["topic"] not in termos)
        return termos[:limit]

    def topic_count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM topics").fetchone()["n"])

    # ------------------------------------------------------------------ dossies

    def record_dossier(
        self,
        dossier: Dossier,
        *,
        model: str,
        provider: str,
        usage: tuple[int, int],
        latency_s: float,
        source_count: int,
        discarded: list[dict] | None = None,
        failures: dict[str, str] | None = None,
    ) -> int:
        """Graba el dossier con el coste medido. Devuelve el id de la fila."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO dossiers (topic, model, provider, fact_count, source_count,"
                " input_tokens, output_tokens, latency_s, dossier_json, discarded_json,"
                " failures_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dossier.topic, model, provider, len(dossier.facts), source_count,
                    usage[0], usage[1], latency_s,
                    dossier.model_dump_json(),
                    json.dumps(discarded or [], ensure_ascii=False),
                    json.dumps(failures or {}, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return int(cur.lastrowid or 0)

    def latest_dossier(self, topic: str | None = None) -> Dossier | None:
        """El dossier más reciente, del tema pedido o de cualquier tema.

        Es lo que conecta al investigador con el guionista sin pasar archivos manualmente: el
        guion sale de lo que quedó grabado, no de un JSON suelto en disco.
        """
        sql = "SELECT dossier_json FROM dossiers"
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return Dossier.model_validate_json(row["dossier_json"]) if row else None

    def dossier_count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM dossiers").fetchone()["n"])

    def latest_dossier_id(self, topic: str | None = None) -> int | None:
        """Id del dossier más reciente, para que el guion apunte a su fuente."""
        sql = "SELECT id FROM dossiers"
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["id"]) if row else None

    # ------------------------------------------------------------------ guiones

    def record_script(
        self,
        script: Script,
        *,
        model: str,
        provider: str,
        usage: tuple[int, int],
        latency_s: float,
        attempts: list[dict],
        dossier_id: int | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO scripts (topic, model, provider, dossier_id, word_count,"
                " attempts, input_tokens, output_tokens, latency_s, script_json,"
                " attempts_json, created_at, format)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    script.topic, model, provider, dossier_id, script.word_count,
                    len(attempts), usage[0], usage[1], latency_s,
                    script.model_dump_json(),
                    json.dumps(attempts, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                    script.format,
                ),
            )
        return int(cur.lastrowid or 0)

    def latest_script(self, topic: str | None = None) -> Script | None:
        sql = "SELECT script_json FROM scripts"
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return Script.model_validate_json(row["script_json"]) if row else None

    def script_count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM scripts").fetchone()["n"])

    # ------------------------------------------------------------------ pareceres

    def record_review(
        self,
        review: Review,
        *,
        usage: tuple[int, int],
        latency_s: float,
        script_id: int | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO reviews (script_id, topic, model, provider, total, approved,"
                " review_json, input_tokens, output_tokens, latency_s, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    script_id, review.topic, review.model, review.provider,
                    review.total, int(review.approved), review.model_dump_json(),
                    usage[0], usage[1], latency_s, datetime.now(UTC).isoformat(),
                ),
            )
        return int(cur.lastrowid or 0)

    def latest_review(self, topic: str | None = None) -> Review | None:
        sql = "SELECT review_json FROM reviews"
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return Review.model_validate_json(row["review_json"]) if row else None

    def review_count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM reviews").fetchone()["n"])

    # ------------------------------------------------------------------ posts

    def record_post(
        self,
        publish_id: str,
        video_path: str,
        *,
        status: str,
        error: str | None = None,
        format: str | None = None,
        script_id: int | None = None,
        carousel_id: int | None = None,
        slot: str | None = None,
    ) -> int:
        """Graba una subida a la inbox, con o sin publish_id de la API.

        Un fallo antes de la inicialización (p. ej., un archivo inexistente) también se guarda, con
        publish_id vacío: sin esto, un intento que no generó nada desaparece del
        historial y no entra en la calibración.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO posts (publish_id, video_path, status, error, created_at,"
                " format, script_id, carousel_id, slot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    publish_id, video_path, status, error,
                    datetime.now(UTC).isoformat(), format, script_id, carousel_id, slot,
                ),
            )
        return int(cur.lastrowid or 0)

    def format_performance_rows(self) -> list[dict]:
        """Última métrica de cada publicación con su formato (para el planificador).

        El formato viene del propio post (piloto automático) o del guion ligado
        pela metrica (`metrics-record --script-id`).
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT m.publish_id, COALESCE(p.format, s.format) AS format,"
                " m.views, m.completion_rate, m.saves, m.collected_at"
                " FROM metrics m"
                " LEFT JOIN (SELECT publish_id, MAX(format) AS format FROM posts"
                "            GROUP BY publish_id) p ON p.publish_id = m.publish_id"
                " LEFT JOIN scripts s ON s.id = m.script_id"
                " ORDER BY m.collected_at DESC, m.id DESC"
            ).fetchall()
        vistos: set[str] = set()
        saida: list[dict] = []
        for r in rows:
            if r["publish_id"] in vistos:
                continue
            vistos.add(r["publish_id"])
            saida.append(dict(r))
        return saida

    def update_post_status(
        self, publish_id: str, *, status: str, error: str | None = None
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE posts SET status = ?, error = ? WHERE publish_id = ?",
                (status, error, publish_id),
            )

    def latest_post(self, video_path: str | None = None) -> dict | None:
        sql = "SELECT publish_id, video_path, status, error, created_at FROM posts"
        params: tuple = ()
        if video_path:
            sql += " WHERE video_path = ?"
            params = (video_path,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def post_count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM posts").fetchone()["n"])

    # ------------------------------------------------------------------ guiones/informes (eval)

    def list_scripts(self, topic: str | None = None) -> list[dict]:
        """Filas de guion para el eval, con coste. Sin parsing aquí."""
        sql = ("SELECT id, topic, model, provider, word_count, attempts,"
               " input_tokens, output_tokens, latency_s, format FROM scripts")
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at, id"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def list_reviews(self, topic: str | None = None) -> list[dict]:
        """Linhas de parecer para o eval, com o JSON para agregar por criterio."""
        sql = ("SELECT id, topic, script_id, model, provider, approved, review_json,"
               " input_tokens, output_tokens, latency_s FROM reviews")
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at, id"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------ metricas (M5)

    def record_metric(
        self,
        publish_id: str,
        views: int,
        *,
        script_id: int | None = None,
        avg_watch_s: float | None = None,
        completion_rate: float | None = None,
        saves: int | None = None,
        comments: int | None = None,
        shares: int | None = None,
    ) -> int:
        """Graba una recogida de métricas. Devuelve el id de la fila.

        Falha cedo em numero impossivel: views negativo, completion fora de
        0..1 ou watch negativo entram na serie e corrompem a curva sem aviso.
        saves/comments/shares sao o placar do carrossel (e o desempate do
        vídeo): la finalización por sí sola no indica si la publicación produjo acción.
        """
        if not publish_id:
            raise ValueError("metrica sem publish_id")
        if views < 0:
            raise ValueError("views negativo")
        if avg_watch_s is not None and avg_watch_s < 0:
            raise ValueError("tempo medio de exibicao negativo")
        if completion_rate is not None and not 0.0 <= completion_rate <= 1.0:
            raise ValueError("completion_rate fora de 0..1")
        for nome, valor in (("saves", saves), ("comments", comments),
                            ("shares", shares)):
            if valor is not None and valor < 0:
                raise ValueError(f"{nome} negativo")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO metrics (publish_id, script_id, views, avg_watch_s,"
                " completion_rate, saves, comments, shares, collected_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    publish_id, script_id, views, avg_watch_s, completion_rate,
                    saves, comments, shares,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return int(cur.lastrowid or 0)

    def latest_metric(self, publish_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT publish_id, script_id, views, avg_watch_s,"
                " completion_rate, saves, comments, shares, collected_at"
                " FROM metrics"
                " WHERE publish_id = ? ORDER BY collected_at DESC, id DESC LIMIT 1",
                (publish_id,),
            ).fetchone()
        return dict(row) if row else None

    def metrics_for(self, publish_id: str) -> list[dict]:
        """La serie completa de una publicación, en orden de recolección (la curva)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT publish_id, script_id, views, avg_watch_s,"
                " completion_rate, saves, comments, shares, collected_at"
                " FROM metrics"
                " WHERE publish_id = ? ORDER BY collected_at, id",
                (publish_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def metric_count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM metrics").fetchone()["n"])

    # ------------------------------------------------------------------ carrosseis

    def record_carousel(
        self,
        carousel: Carousel,
        *,
        model: str,
        provider: str,
        usage: tuple[int, int],
        latency_s: float,
        attempts: list[dict],
        review: CarouselReview | None = None,
    ) -> int:
        """Graba carrusel con parecer incrustado (rúbrica propia, 4 criterios)."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO carousels (topic, model, provider, approved,"
                " carousel_json, review_json, attempts_json, input_tokens,"
                " output_tokens, latency_s, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    carousel.topic, model, provider,
                    int(review.approved) if review is not None else 0,
                    carousel.model_dump_json(),
                    review.model_dump_json() if review is not None else None,
                    json.dumps(attempts, ensure_ascii=False),
                    usage[0], usage[1], latency_s,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return int(cur.lastrowid or 0)

    def list_carousels(self, topic: str | None = None) -> list[dict]:
        sql = ("SELECT id, topic, model, provider, approved, carousel_json,"
               " review_json, input_tokens, output_tokens, latency_s"
               " FROM carousels")
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at, id"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def latest_carousel_id(self, topic: str | None = None) -> int | None:
        sql = "SELECT id FROM carousels"
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["id"]) if row else None

    def update_carousel_review(self, carousel_id: int, review: CarouselReview) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE carousels SET review_json = ?, approved = ? WHERE id = ?",
                (review.model_dump_json(), int(review.approved), carousel_id),
            )

    def latest_carousel(self, topic: str | None = None) -> Carousel | None:
        sql = "SELECT carousel_json FROM carousels"
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return Carousel.model_validate_json(row["carousel_json"]) if row else None


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    # Datas gravadas antes de uma normalizacao, ou por outra ferramenta, podem
    # vir sem tzinfo. Comparar naive com aware levanta TypeError no meio da
    # recolección, se asume UTC, que es lo que el agente siempre graba.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

