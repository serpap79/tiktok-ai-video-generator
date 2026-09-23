"""Memoria do agente em SQLite.

Coisas distintas, e nao um armazem generico: a **serie de sinais** (do radar), o
**ledger de temas** (do curador), os **dossies** (do pesquisador) e os
**roteiros** (do roteirista). Cada uma responde a uma pergunta diferente --
"esta subindo?", "ja falamos disso?", "o que sabemos e de onde?", "o que foi
escrito e a que custo?" -- e misturar todas numa tabela de documentos tornaria
impossivel responder qualquer uma delas por SQL.

A serie de sinais e o que permite calcular velocidade
para fontes que reportam nivel e nao taxa (Wikipedia, Google Trends). Sem
historico, "500 mil pageviews" e um numero sem significado: nao da para saber se
o assunto esta subindo ou ja passou.
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
    -- Custo medido na hora da pesquisa. Reconstruir isso do log depois nao da:
    -- o provedor nao devolve consumo retroativo, e o eval do M5 compara custo.
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_s     REAL    NOT NULL,
    -- JSON do Dossier inteiro. Guardar o contrato serializado, em vez de uma
    -- tabela de fatos normalizada, mantem o dossie reproduzivel palavra por
    -- palavra -- que e o que o juiz vai reler no M5 para julgar o mesmo material.
    dossier_json  TEXT    NOT NULL,
    -- Fatos que os portoes derrubaram, com motivo. Mesma regra do ledger de
    -- temas: sem o descartado, so se sabe o que entrou, nunca o que foi perdido.
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
    -- mecanicos. Se toda execucao gasta duas, o defeito esta no prompt, nao no
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

-- Tabela propria, e nao colunas em `scripts`, porque o mesmo roteiro pode ser
-- julgado mais de uma vez: o eval do M5 compara juizes de modelos diferentes
-- sobre o MESMO texto, e isso e uma relacao de um para muitos.
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY,
    script_id     INTEGER,
    topic         TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    total         INTEGER NOT NULL,
    approved      INTEGER NOT NULL,
    -- Parecer completo: nota e motivo de cada um dos sete criterios. E o que
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

-- Metricas do post publicado, serie temporal: o mesmo publish_id pode ter
-- varias coletas (views sobem com o tempo), e a curva e o unico sinal real de
-- viralidade. Leitura manual do app por enquanto: a Content Posting API, no
-- escopo video.upload da inbox, nao expoe endpoint de metricas -- e a Research
-- API e restrita a pesquisa academica. `script_id` fecha o loop com o roteiro
-- que gerou o video; NULL quando o vinculo nao e conhecido.
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

-- Carrossel: roteiro de 5 slides + parecer embutido. Parecer proprio (e nao
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

        CREATE TABLE IF NOT EXISTS nao adiciona coluna em tabela que ja
        existe -- sem isso, o banco real da maquina (criado no M3) rejeita
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
        # Piloto automatico: o post sabe de qual roteiro/formato/slot veio, e
        # e isso que deixa as metricas voltarem ao planejador de formato.
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
        """Grava toda decisao, aprovada ou nao.

        Rejeicao gravada e o que permite calibrar o score depois em vez de
        chutar: sem ela, so se sabe o que foi escolhido, nunca o que foi perdido.
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

        So os aprovados entram: um tema rejeitado por politica ou por nicho nao
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
            # O que virou roteiro ou carrossel tambem foi coberto, mesmo sem
            # ter passado pelo curador (`research --topic` e o piloto
            # automatico escrevem direto). Visto em 19/09: o short do cerebro
            # foi produzido de manha e o tema voltou ao topo a tarde.
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
        """Grava o dossie com o custo medido. Devolve o id da linha."""
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
        """O dossie mais recente, do tema pedido ou de qualquer tema.

        E o que liga o pesquisador ao roteirista sem passar arquivo na mao: o
        roteiro sai do que ficou gravado, nao de um JSON solto no disco.
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
        """Id do dossie mais recente, para o roteiro apontar para a fonte dele."""
        sql = "SELECT id FROM dossiers"
        params: tuple = ()
        if topic:
            sql += " WHERE topic = ?"
            params = (topic,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["id"]) if row else None

    # ------------------------------------------------------------------ roteiros

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
        """Grava uma subida a inbox, com ou sem publish_id da API.

        Falha antes do init (ex. arquivo inexistente) tambem e gravada, com
        publish_id vazio: sem isso, tentativa que nao gerou nada some do
        historico e nao entra na calibracao.
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
        """Ultima metrica de cada post com o formato dele (para o planejador).

        Formato vem do proprio post (piloto automatico) ou do roteiro ligado
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

    # ------------------------------------------------------------------ roteiros/pareceres (eval)

    def list_scripts(self, topic: str | None = None) -> list[dict]:
        """Linhas de roteiro para o eval, com custo. Sem parsing aqui."""
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
        """Grava uma coleta de metricas. Devolve o id da linha.

        Falha cedo em numero impossivel: views negativo, completion fora de
        0..1 ou watch negativo entram na serie e corrompem a curva sem aviso.
        saves/comments/shares sao o placar do carrossel (e o desempate do
        video): completion sozinho nao diz se o post gerou acao.
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
        """A serie inteira de um post, em ordem de coleta (a curva)."""
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
        """Grava carrossel com parecer embutido (rubrica propria, 4 critérios)."""
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
    # coleta, entao assume-se UTC, que e o que o agente sempre grava.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

