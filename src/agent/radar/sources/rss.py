"""Feeds RSS de tecnologia y ciencia, agrupados por historia. Sin clave.

Por que RSS si ya esta Hacker News: el HN mide lo que interesa a un
desarrollador americano; el publico del canal es espanol. Xataka, El Androide
Libre, Genbeta, Hipertextual, Computer Hoy y MuyComputer dicen lo que la
prensa tech ESPANOLA esta cubriendo, y los feeds globales (Verge,
TechCrunch, Ars, MIT TR, Wired, GitHub Blog, The New Stack, GameSpot,
ScienceDaily, NASA) dicen lo que el mundo cubre. Todos respondieron 200 en
20/09/2026, salvo VentureBeat (429 con bot protection -- fuera hasta liberar;
motivo registrado, no fallo silencioso).

Varios de estos feeds reflejan perfiles que el autor sigue en X (@xataka,
@hipertextual, @El_Androide_Libre, @genbeta_es, @ComputerHoy, @verge,
@WIRED, @TechCrunch, @thenewstack, @GithubProjects). La API de X es de pago y
rompe la restriccion de $0/mes, asi que el radar lee el mismo contenido en el
origen RSS en lugar de leerlo en X.

RSS no tiene punto ni pageview. La senal de engagement aqui es otra, y se
mide: **cuantos medios distintos publicaron la misma historia en las ultimas
horas**. Una noticia aislada es pauta de un medio; la misma historia en cuatro
redacciones en seis horas es asunto del dia. Eso se convierte en `volume`
(medios) y `velocity` (medios por hora desde la primera publicacion) --
velocidad nativa, ya en la primera recolecta, como la del HN.

De regalo, el grupo entrega las otras noticias como `news_items`: el
investigador recibe varias fuentes de la misma historia sin buscador de pago.
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
    # Etiqueta de fuente del Signal: separa el percentil del curador por familia
    # de feed (redaccion espanola no compite en escala con feed global).
    kind: str


FEEDS: tuple[Feed, ...] = (
    Feed("xataka", "https://xataka.com/feednews", "rss_tech_es"),
    Feed("elpais_tec", "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/tecnologia/portada",
         "rss_tech_es"),
    Feed("elmundo_tec", "https://e00-elmundo.uecdn.es/rss/tecnologia.xml", "rss_tech_es"),
    Feed("genbeta", "https://www.genbeta.com/feednews", "rss_tech_es"),
    Feed("hipertextual", "https://hipertextual.com/feed", "rss_tech_es"),
    Feed("computerhoy", "https://www.computerhoy.com/rss.xml", "rss_tech_es"),
    Feed("muycomputer", "https://www.muycomputer.com/feed/", "rss_tech_es"),
    Feed("microsiervos", "https://www.microsiervos.com/xml/rss.xml", "rss_tech_es"),
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

# Ventana de "ahora": noticia de tres dias atras no es tendencia, es archivo.
JANELA_H = 36
# Similaridad de titulo para que dos noticias cuenten como la misma historia.
# Titulos de medios distintos comparten el nombre propio y poco mas; 0,3 junta
# "OpenAI lanza GPT-6" con "GPT-6 llega con..." sin juntar todo lo que habla
# de "IA".
UMBRAL_GRUPO = 0.3
# Suelo de una hora en la edad: noticia de 10 minutos en 2 medios daria
# "12 medios por hora" por artefacto de division.
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
        ahora = datetime.now(UTC)
        itens: list[Item] = []
        fallos: list[str] = []
        for feed in self._feeds:
            try:
                r = self._client.get(feed.url, headers=HEADERS)
            except httpx.HTTPError as exc:
                fallos.append(f"{feed.name}: {type(exc).__name__}")
                continue
            if r.status_code != 200:
                fallos.append(f"{feed.name}: HTTP {r.status_code}")
                continue
            try:
                itens.extend(parse_feed(r.content, feed, now=ahora, window_h=self._window_h))
            except SourceUnavailable as exc:
                fallos.append(f"{feed.name}: {exc}")
        if not itens and fallos:
            raise SourceUnavailable("ningun feed respondio: " + "; ".join(fallos[:4]))
        return cluster(itens, now=ahora)


def parse_feed(xml_bytes: bytes, feed: Feed, *, now: datetime,
               window_h: float = JANELA_H) -> list[Item]:
    """Items RSS 2.0 o Atom dentro de la ventana. Item sin fecha se queda
    fuera: sin fecha no hay forma de saber si es de hoy."""
    try:
        raiz = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise SourceUnavailable(f"XML invalido: {exc}") from exc
    corte = now - timedelta(hours=window_h)
    salida: list[Item] = []
    for nodo in list(raiz.iter("item")) + list(raiz.iter(f"{_ATOM}entry")):
        titulo = " ".join((nodo.findtext("title") or nodo.findtext(f"{_ATOM}title") or "").split())
        link = (nodo.findtext("link") or "").strip()
        if not link:
            atom_link = nodo.find(f"{_ATOM}link")
            link = (atom_link.get("href") if atom_link is not None else "") or ""
        data = _data(nodo.findtext("pubDate") or nodo.findtext(f"{_ATOM}published")
                     or nodo.findtext(f"{_ATOM}updated") or "")
        if not titulo or not link.startswith("http") or data is None:
            continue
        if data < corte or data > now + timedelta(hours=2):
            continue
        salida.append(Item(titulo, link, feed, data, frozenset(content_tokens(titulo))))
    return salida


def cluster(itens: list[Item], *, now: datetime) -> list[Signal]:
    """Agrupa la misma historia entre medios y devuelve un Signal por grupo.

    Goloso y determinista: items en orden de publicacion, cada uno entra en el
    primer grupo con titulo parecido lo bastante. El representante es el item
    mas antiguo del grupo (quien dio primero), y el `volume` cuenta MEDIOS
    distintos -- el mismo sitio republicando no es corroboracion.
    """
    grupos: list[list[Item]] = []
    for item in sorted(itens, key=lambda i: i.published):
        for g in grupos:
            if any(_jaccard(item.tokens, otro.tokens) >= UMBRAL_GRUPO for otro in g):
                g.append(item)
                break
        else:
            grupos.append([item])

    senales: list[Signal] = []
    for g in grupos:
        rep = g[0]
        medios = {i.feed.name for i in g}
        edad_h = max((now - rep.published).total_seconds() / 3600, IDADE_MINIMA_H)
        otros = [i for i in g[1:] if i.feed.name != rep.feed.name]
        vistos: set[str] = set()
        noticias: list[NewsItem] = []
        for i in otros:
            if i.feed.name in vistos:
                continue
            vistos.add(i.feed.name)
            try:
                noticias.append(NewsItem(title=i.title, url=i.url, source_name=i.feed.name))
            except ValueError:
                continue
        try:
            senales.append(Signal(
                term=rep.title, source=rep.feed.kind, volume=float(len(medios)),
                unit="medios", velocity=round(len(medios) / edad_h, 3),
                seen_at=now, url=rep.url, news_items=noticias,
            ))
        except ValueError:
            continue
    return senales


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
