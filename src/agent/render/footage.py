"""Material de video escolhido por nos: relevante ao assunto e escuro como a marca.

Ate 19/09 o MoneyPrinterTurbo buscava no Pexels com os nossos termos e
pegava os primeiros resultados. Dois defeitos medidos nos frames do que foi
publicado:

- **correlacao**: o short sobre o cerebro ser dois orgaos abriu com uma mao
  3D segurando celular e uma cadeira de simulador -- o termo era de
  identidade ("futuristic clean UI"), e nada na tela falava de cerebro;
- **identidade**: fundo bege e branco no meio de um canal "dark", porque o
  primeiro resultado nao sabe o que e a marca.

Aqui cada termo vira uma busca, e cada candidato ganha nota por duas medidas
baratas que o Pexels ja entrega: **relevancia** (palavras do termo no slug da
pagina do video, que descreve a cena: `/video/brain-scan-on-a-monitor-123/`)
e **escuridao** (luminancia media da miniatura). B-roll do assunto pesa mais
relevancia; tag de pilar pesa mais escuridao. Clipe usado nos ultimos dias
perde pontos: os mesmos termos de pilar trariam os mesmos clipes todo dia.

Os clipes escolhidos sobem ao renderizador como material local, NA ORDEM da
narracao -- o MPT so monta.
"""

from __future__ import annotations

import io
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from agent.models import Script

SEARCH_URL = "https://api.pexels.com/videos/search"
HEADERS = {"User-Agent": "tiktok-viral-generator/1.0"}

# Segundos de tela por clipe no MPT (`video_clip_duration`): corte a cada 5s
# segura a atencao sem virar videoclipe.
CLIP_S = 5
# Folga sobre a duracao estimada: o TTS real varia ~15% do ritmo assumido.
FOLGA = 1.25
# Teto de download por clipe: 1080x1920 de 20s fica em ~10-25 MB.
MAX_BYTES = 60 * 1024 * 1024
# Dias em que um clipe usado perde pontos.
JANELA_REUSO_DIAS = 14

_PALAVRA = re.compile(r"[a-z0-9]+")


@dataclass
class Clip:
    term: str
    video_id: int
    page_url: str
    file_url: str
    width: int
    height: int
    duration: float
    thumb_url: str
    relevance: float = 0.0
    darkness: float = 0.0
    score: float = 0.0
    path: Path | None = None


@dataclass
class FootagePlan:
    clips: list[Clip] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[Path]:
        return [c.path for c in self.clips if c.path is not None]


def needed_clips(script: Script) -> int:
    """Clipes para cobrir a narracao sem repetir, com folga."""
    return max(3, math.ceil(script.estimated_duration_s * FOLGA / CLIP_S) + 1)


def timeline_terms(script: Script) -> list[tuple[str, bool]]:
    """Termos na ordem da tela: (termo, e_broll).

    O primeiro b-roll abre o video -- e a imagem do assunto que segura os
    primeiros segundos; o segundo entra no meio. As tags do pilar preenchem o
    resto, na ordem cronologica em que o roteirista as pos.
    """
    tags = [(t, False) for t in script.search_terms]
    broll = [(t, True) for t in script.broll]
    if not broll:
        return tags
    linha = [broll[0], *tags]
    if len(broll) > 1:
        linha.insert(max(1, len(linha) // 2), broll[1])
    return linha


def relevance(term: str, page_url: str) -> float:
    """Fracao das palavras do termo que aparecem no slug da pagina do video."""
    alvo = {_raiz(p) for p in _PALAVRA.findall(term.lower()) if len(p) > 2}
    if not alvo:
        return 0.0
    slug = page_url.rstrip("/").rsplit("/", 1)[-1].lower()
    presentes = {_raiz(p) for p in _PALAVRA.findall(slug)}
    return round(len(alvo & presentes) / len(alvo), 3)


def _raiz(palavra: str) -> str:
    return palavra[:-1] if palavra.endswith("s") and len(palavra) > 3 else palavra


def darkness_of(jpeg: bytes) -> float:
    """1 - luminancia media (0 = branco, 1 = preto) de uma miniatura."""
    from PIL import Image, ImageStat

    with Image.open(io.BytesIO(jpeg)) as img:
        media = ImageStat.Stat(img.convert("L").resize((48, 85))).mean[0]
    return round(1 - media / 255, 3)


def score(clip: Clip, *, is_broll: bool, reused: bool) -> float:
    if is_broll:
        nota = 0.7 * clip.relevance + 0.3 * clip.darkness
    else:
        nota = 0.35 * clip.relevance + 0.65 * clip.darkness
    return round(nota - (0.4 if reused else 0.0), 3)


def best_file(video: dict) -> tuple[str, int, int] | None:
    """O arquivo retrato mais proximo de 1080x1920 (sem baixar 4K a toa)."""
    arquivos = [f for f in video.get("video_files") or []
                if (f.get("width") or 0) and (f.get("height") or 0)
                and f["height"] > f["width"] and f.get("link")]
    if not arquivos:
        return None
    ideais = [f for f in arquivos if f["width"] >= 1080]
    if ideais:
        f = min(ideais, key=lambda f: f["width"])
    else:
        f = max(arquivos, key=lambda f: f["width"])
        if f["width"] < 720:
            return None
    return f["link"], int(f["width"]), int(f["height"])


class FootageLedger:
    """Clipes usados, para o canal nao repetir as mesmas cenas todo dia."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS footage_used ("
                " video_id INTEGER NOT NULL, term TEXT NOT NULL, used_at TEXT NOT NULL)")

    def recent(self, days: int = JANELA_REUSO_DIAS) -> set[int]:
        corte = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            return {int(r[0]) for r in conn.execute(
                "SELECT video_id FROM footage_used WHERE used_at >= ?", (corte,))}

    def record(self, clips: list[Clip]) -> None:
        agora = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO footage_used (video_id, term, used_at) VALUES (?, ?, ?)",
                [(c.video_id, c.term, agora) for c in clips])


class PexelsFootage:
    def __init__(self, api_key: str, cache_dir: Path, *,
                 client: httpx.Client | None = None,
                 ledger: FootageLedger | None = None, per_page: int = 15):
        if not api_key:
            raise ValueError("sem chave do Pexels (AGENT_PEXELS_API_KEY)")
        self._key = api_key
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, read=120.0),
                                              headers=HEADERS, follow_redirects=True)
        self._ledger = ledger
        self._per_page = per_page

    # ---------------------------------------------------------------- plano

    def plan(self, script: Script) -> FootagePlan:
        """Clipes na ordem da narracao, ja baixados. Termo sem resultado vira `miss`."""
        plano = FootagePlan()
        linha = timeline_terms(script)
        alvo = needed_clips(script)
        usados = self._ledger.recent() if self._ledger else set()
        escolhidos: set[int] = set()
        candidatos: dict[str, list[Clip]] = {}
        for termo, is_broll in linha:
            try:
                candidatos[termo] = self._rank(termo, is_broll, usados)
            except httpx.HTTPError as exc:
                plano.misses.append(f"{termo}: {type(exc).__name__}")
                candidatos[termo] = []
            if not candidatos[termo]:
                plano.misses.append(f"{termo}: sem clipe retrato >= 720p")

        # Distribui o alvo pelos termos em rodadas: todos os termos entram uma
        # vez antes de qualquer um repetir, e a ordem da narracao se mantem
        # dentro de cada rodada.
        por_termo = {t: [c for c in cs] for t, cs in candidatos.items()}
        while len(plano.clips) < alvo and any(por_termo.values()):
            for termo, _ in linha:
                fila = por_termo.get(termo) or []
                while fila and fila[0].video_id in escolhidos:
                    fila.pop(0)
                if not fila or len(plano.clips) >= alvo:
                    continue
                clip = fila.pop(0)
                escolhidos.add(clip.video_id)
                plano.clips.append(clip)
        plano.clips = self._ordenar(plano.clips, linha)
        for clip in plano.clips:
            try:
                clip.path = self._baixar(clip)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                plano.misses.append(f"{clip.term} #{clip.video_id}: {exc}")
        plano.clips = [c for c in plano.clips if c.path is not None]
        if self._ledger is not None and plano.clips:
            self._ledger.record(plano.clips)
        return plano

    @staticmethod
    def _ordenar(clips: list[Clip], linha: list[tuple[str, bool]]) -> list[Clip]:
        """Agrupa por termo na ordem da narracao: a cena acompanha a fala."""
        ordem = {t: i for i, (t, _) in enumerate(linha)}
        return sorted(clips, key=lambda c: ordem.get(c.term, len(ordem)))

    # ---------------------------------------------------------------- busca

    def _rank(self, termo: str, is_broll: bool, usados: set[int]) -> list[Clip]:
        r = self._client.get(SEARCH_URL, params={
            "query": termo, "orientation": "portrait", "size": "medium",
            "per_page": str(self._per_page)}, headers={"Authorization": self._key})
        r.raise_for_status()
        clips: list[Clip] = []
        for v in r.json().get("videos") or []:
            arquivo = best_file(v)
            if arquivo is None or (v.get("duration") or 0) < CLIP_S:
                continue
            link, w, h = arquivo
            clips.append(Clip(
                term=termo, video_id=int(v["id"]), page_url=str(v.get("url") or ""),
                file_url=link, width=w, height=h, duration=float(v.get("duration") or 0),
                thumb_url=str(v.get("image") or ""),
                relevance=relevance(termo, str(v.get("url") or ""))))
        if is_broll:
            # B-roll sem nenhuma palavra do termo no slug e imagem aleatoria: no
            # video de Marte (19/09), "Thyles Rupes" trouxe flores, vespa e
            # lagarta. Sem clipe relevante, o termo sai e a identidade cobre.
            clips = [c for c in clips if c.relevance > 0]
        # Miniatura so dos mais relevantes: e a medida que custa requisicao.
        clips.sort(key=lambda c: -c.relevance)
        for c in clips[:8]:
            c.darkness = self._escuridao(c.thumb_url)
            c.score = score(c, is_broll=is_broll, reused=c.video_id in usados)
        for c in clips[8:]:
            c.score = score(c, is_broll=is_broll, reused=c.video_id in usados) - 0.2
        clips.sort(key=lambda c: -c.score)
        return clips

    def _escuridao(self, url: str) -> float:
        if not url:
            return 0.5
        pequena = re.sub(r"([?&])(w|h)=\d+", r"\1\2=0", url)
        pequena = pequena.replace("w=0", "w=90").replace("h=0", "h=160")
        try:
            r = self._client.get(pequena)
            r.raise_for_status()
            return darkness_of(r.content)
        except (httpx.HTTPError, OSError, ValueError):
            return 0.5

    def _baixar(self, clip: Clip) -> Path:
        destino = self._cache / f"pexels-{clip.video_id}-{clip.width}.mp4"
        if destino.exists() and destino.stat().st_size > 0:
            return destino
        parcial = destino.with_suffix(".part")
        total = 0
        with self._client.stream("GET", clip.file_url) as r:
            r.raise_for_status()
            with parcial.open("wb") as fh:
                for pedaco in r.iter_bytes(1 << 16):
                    total += len(pedaco)
                    if total > MAX_BYTES:
                        raise ValueError(f"clipe passou de {MAX_BYTES // 2**20} MB")
                    fh.write(pedaco)
        parcial.replace(destino)
        return destino


__all__ = ["CLIP_S", "Clip", "FootageLedger", "FootagePlan", "PexelsFootage",
           "best_file", "darkness_of", "needed_clips", "relevance", "score",
           "timeline_terms"]
