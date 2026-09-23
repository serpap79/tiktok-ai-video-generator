"""Feeds RSS de tecnologia e ciencia, agrupados por historia. Sem chave.

Por que RSS se ja ha Hacker News: o HN mede o que interessa a desenvolvedor
americano; o publico do canal e brasileiro. Tecnoblog, Olhar Digital,
Canaltech, TecMundo, TechTudo e Inovacao Tecnologica dizem o que a imprensa de
tech BRASILEIRA esta cobrindo, e os feeds globais (Verge, TechCrunch, Ars, MIT
TR, Wired, GitHub Blog, The New Stack, GameSpot, ScienceDaily, NASA) dizem o
que o mundo cobre. Todos responderam 200 em 20/09/2026, salvo VentureBeat
(429 com bot protection -- fora ate liberar; motivo registrado, nao falha
silenciosa).

Varios desses feeds espelham perfis que o autor acompanha no X
(@olhardigital, @Tec_Mundo, @TechTudo, @canaltech, @TechCrunch, @verge,
@WIRED, @VentureBeat, @GameSpot, @thenewstack, @GithubProjects). A API do X
e paga e quebra a restricao de $0/mes, entao o radar le o mesmo conteudo na
origem RSS em vez de ler no X.

RSS nao tem ponto nem pageview. O sinal de engajamento aqui e outro, e e
medido: **quantos veiculos diferentes publicaram a mesma historia nas ultimas
horas**. Uma materia isolada e pauta de um veiculo; a mesma historia em
quatro redacoes em seis horas e assunto do dia. Isso vira `volume`
(veiculos) e `velocity` (veiculos por hora desde a primeira publicacao) --
velocidade nativa, ja na primeira coleta, como a do HN.

De brinde, o grupo entrega as outras materias como `news_items`: o
pesquisador recebe varias fontes da mesma historia sem buscador pago.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import httpx

from agent.models import NewsItem, Signal
from agent.ports.radar import SourceUnavailable
from agent.text import content_tokens

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; tiktok-viral-generator/0.1; radar)"}
_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    # Rotulo de fonte do Signal: separa o percentil do curador por familia de
    # feed (redacao brasileira nao compete em escala com feed global).
    kind: str


FEEDS: tuple[Feed, ...] = (
    Feed("tecnoblog", "https://tecnoblog.net/feed/", "rss_tech_br"),
    Feed("olhardigital", "https://olhardigital.com.br/feed/", "rss_tech_br"),
    Feed("canaltech", "https://canaltech.com.br/rss/", "rss_tech_br"),
    Feed("tecmundo", "https://rss.tecmundo.com.br/feed", "rss_tech_br"),
    Feed("techtudo", "https://www.techtudo.com.br/rss/techtudo/", "rss_tech_br"),
    Feed("inovacaotecnologica", "https://www.inovacaotecnologica.com.br/boletim/rss.xml",
         "rss_tech_br"),
    Feed("theverge", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
         "rss_tech"),
    Feed("techcrunch", "https://techcrunch.com/category/artificial-intelligence/feed/",
         "rss_tech"),
    Feed("arstechnica", "https://feeds.arstechnica.com/arstechnica/technology-lab",
         "rss_tech"),
    Feed("technologyreview", "https://www.technologyreview.com/feed/", "rss_tech"),
    Feed("wired", "https://www.wired.com/feed/rss", "rss_tech"),
    Feed("githubblog", "https://github.blog/feed/", "rss_tech"),
    Feed("thenewstack", "https://thenewstack.io/feed/", "rss_tech"),
    Feed("gamespot", "https://www.gamespot.com/feeds/news/", "rss_tech"),
    Feed("sciencedaily", "https://www.sciencedaily.com/rss/top/technology.xml",
         "rss_ciencia"),
    Feed("nasa", "https://www.nasa.gov/news-release/feed/", "rss_ciencia"),
)

# Janela de "agora": materia de tres dias atras nao e tendencia, e arquivo.
JANELA_H = 36
# Similaridade de titulo para duas materias contarem como a mesma historia.
# Titulos de veiculos diferentes compartilham o nome proprio e pouco mais;
# 0,3 junta "OpenAI lanca GPT-6" com "GPT-6 chega com..." sem juntar tudo
# que fala de "IA".
LIMIAR_GRUPO = 0.3
# Piso de uma hora na idade: materia de 10 minutos em 2 veiculos daria
# "12 veiculos por hora" por artefato de divisao.
IDADE_MINIMA_H = 1.0


@dataclass
class Item:
    title: str
    url: str
    feed: Feed
    published: datetime
    tokens: frozenset[str]


class RssFeeds:
    name = "rss"

    def __init__(self, client: httpx.Client | None = None,
                 feeds: tuple[Feed, ...] = FEEDS, window_h: float = JANELA_H):
        self._client = client or httpx.Client(timeout=httpx.Timeout(15.0), headers=HEADERS,
                                              follow_redirects=True)
        self._feeds = feeds
        self._window_h = window_h

    def collect(self) -> list[Signal]:
        agora = datetime.now(UTC)
        itens: list[Item] = []
        falhas: list[str] = []
        for feed in self._feeds:
            try:
                r = self._client.get(feed.url, headers=HEADERS)
            except httpx.HTTPError as exc:
                falhas.append(f"{feed.name}: {type(exc).__name__}")
                continue
            if r.status_code != 200:
                falhas.append(f"{feed.name}: HTTP {r.status_code}")
                continue
            try:
                itens.extend(parse_feed(r.content, feed, now=agora, window_h=self._window_h))
            except SourceUnavailable as exc:
                falhas.append(f"{feed.name}: {exc}")
        if not itens and falhas:
            raise SourceUnavailable("nenhum feed respondeu: " + "; ".join(falhas[:4]))
        return cluster(itens, now=agora)


def parse_feed(xml_bytes: bytes, feed: Feed, *, now: datetime,
               window_h: float = JANELA_H) -> list[Item]:
    """Itens RSS 2.0 ou Atom dentro da janela. Item sem data fica de fora:
    sem data nao ha como dizer se e de hoje."""
    try:
        raiz = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise SourceUnavailable(f"XML invalido: {exc}") from exc
    corte = now - timedelta(hours=window_h)
    saida: list[Item] = []
    for no in list(raiz.iter("item")) + list(raiz.iter(f"{_ATOM}entry")):
        titulo = " ".join((no.findtext("title") or no.findtext(f"{_ATOM}title") or "").split())
        link = (no.findtext("link") or "").strip()
        if not link:
            atom_link = no.find(f"{_ATOM}link")
            link = (atom_link.get("href") if atom_link is not None else "") or ""
        data = _data(no.findtext("pubDate") or no.findtext(f"{_ATOM}published")
                     or no.findtext(f"{_ATOM}updated") or "")
        if not titulo or not link.startswith("http") or data is None:
            continue
        if data < corte or data > now + timedelta(hours=2):
            continue
        saida.append(Item(titulo, link, feed, data, frozenset(content_tokens(titulo))))
    return saida


def cluster(itens: list[Item], *, now: datetime) -> list[Signal]:
    """Agrupa a mesma historia entre veiculos e devolve um Signal por grupo.

    Guloso e deterministico: itens em ordem de publicacao, cada um entra no
    primeiro grupo com titulo parecido o bastante. O representante e o item
    mais antigo do grupo (quem deu primeiro), e o `volume` conta VEICULOS
    distintos -- o mesmo site republicando nao e corroboracao.
    """
    grupos: list[list[Item]] = []
    for item in sorted(itens, key=lambda i: i.published):
        for g in grupos:
            if any(_jaccard(item.tokens, outro.tokens) >= LIMIAR_GRUPO for outro in g):
                g.append(item)
                break
        else:
            grupos.append([item])

    sinais: list[Signal] = []
    for g in grupos:
        rep = g[0]
        veiculos = {i.feed.name for i in g}
        idade_h = max((now - rep.published).total_seconds() / 3600, IDADE_MINIMA_H)
        outros = [i for i in g[1:] if i.feed.name != rep.feed.name]
        vistos: set[str] = set()
        materias: list[NewsItem] = []
        for i in outros:
            if i.feed.name in vistos:
                continue
            vistos.add(i.feed.name)
            try:
                materias.append(NewsItem(title=i.title, url=i.url, source_name=i.feed.name))
            except ValueError:
                continue
        try:
            sinais.append(Signal(
                term=rep.title, source=rep.feed.kind, volume=float(len(veiculos)),
                unit="veiculos", velocity=round(len(veiculos) / idade_h, 3),
                seen_at=now, url=rep.url, news_items=materias,
            ))
        except ValueError:
            continue
    return sinais


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _data(texto: str) -> datetime | None:
    texto = texto.strip()
    if not texto:
        return None
    try:
        dt = parsedate_to_datetime(texto)
    except (TypeError, ValueError, IndexError):
        try:
            dt = datetime.fromisoformat(texto.replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


__all__ = ["FEEDS", "Feed", "Item", "RssFeeds", "cluster", "parse_feed"]
