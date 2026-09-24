"""De donde salen las 3-5 fuentes de un tema, sin clave y sin buscador de pago.

No existe API de busqueda web gratuita que sirva: Google y Bing cobran, y
raspar SERP se rompe en una semana. Lo que existe gratis, y ya esta en la
stack, es:

1. **las noticias que el RSS de Google Trends anexa** a cada tema (titulo,
   medio y URL). Vienen gratis en la propia recolecta del radar -- y es la
   razon de que `NewsItem` exista desde el M1.
2. **la URL del articulo detras del item de Hacker News**, via la API de
   Algolia. El `Signal` del HN guarda el link de la discusion, que es donde
   estan los comentarios; el articulo en si sale de `/items/{id}`, tambien sin
   clave.
3. **el GDELT DOC 2.0 en modo artlist**, que responde consulta por palabra
   clave sobre cobertura periodistica global. Ya entra en el radar como fuente
   de volumen; aqui responde "quien mas escribio sobre esto".

Ninguna por si sola cubre todo tema: item del HN sin noticia asociada, tema de
Trends no pasa por Algolia, y el GDELT devuelve 429 con frecuencia. Por eso las
tres corren y el informe dice cuales fallaron, en vez de que una de ellas sea
obligatoria.

El techo de fuentes por dominio existe porque cinco paginas del mismo sitio no
son cinco fuentes -- es una fuente contada cinco veces, y el juez no tiene como
saber la diferencia mirando solo el dossier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from agent.models import Decision
from agent.text import tokens

ALGOLIA_ITEM = "https://hn.algolia.com/api/v1/items"

# Fuentes cuya URL de senal no es texto sobre el tema: la entrada mas vista de
# la Wikipedia si es sobre el asunto, pero la senal de Trends no tiene URL, y
# el GDELT apunta a la consulta. Solo estas quedan fuera de la fuente primaria.
_SEM_PAGINA_PROPRIA = frozenset({"google_trends", "gdelt"})
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"

_ID_HN = re.compile(r"[?&]id=(\d+)")

# Tres palavras em AND ja recortam bem no GDELT; quatro costumam devolver zero.
MAX_TERMOS_CONSULTA = 3


@dataclass
class Candidate:
    """Uma fonte candidata, antes de ser lida."""

    url: str
    title: str = ""
    source_name: str = ""
    origin: str = ""


@dataclass
class DiscoveryReport:
    candidates: list[Candidate] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def domains(self) -> set[str]:
        return {_dominio(c.url) for c in self.candidates}


def discover(
    decision: Decision,
    *,
    client: httpx.Client | None = None,
    limit: int = 5,
    per_domain: int = 2,
) -> DiscoveryReport:
    """Junta las tres estrategias, en el orden en que la fuente es mas fiable.

    El orden importa: lo que viene primero se lee primero y, si el presupuesto
    de fuentes se acaba, es lo que queda en el dossier. Primero la fuente
    primaria del tema (el articulo que origino la discusion, o la noticia que
    la propia fuente asocio); el GDELT entra al final, como corroboracion.
    """
    report = DiscoveryReport()
    cliente = client or httpx.Client(timeout=httpx.Timeout(20.0))
    brutos: list[Candidate] = []

    if decision.source == "hacker_news" and decision.url:
        try:
            primaria = hn_story(str(decision.url), client=cliente)
        except SourceLookupFailed as exc:
            report.failures["hacker_news"] = str(exc)
        else:
            if primaria:
                brutos.append(primaria)
    elif decision.url and decision.source not in _SEM_PAGINA_PROPRIA:
        # RSS, Hugging Face, Wikipedia (un dia como hoy / archivo): la URL de
        # la senal YA es la noticia, el model card o la entrada -- la fuente
        # primaria.
        url = str(decision.url)
        brutos.append(Candidate(url=url, title=decision.term,
                                source_name=_dominio(url), origin=decision.source))

    brutos.extend(
        Candidate(
            url=str(item.url),
            title=item.title,
            source_name=item.source_name,
            origin="google_trends",
        )
        for item in decision.news_items
    )

    try:
        brutos.extend(gdelt_articles(decision.term, client=cliente))
    except SourceLookupFailed as exc:
        report.failures["gdelt"] = str(exc)

    report.candidates = _peneirar(brutos, limit=limit, per_domain=per_domain)
    return report


class SourceLookupFailed(RuntimeError):
    """La estrategia de descubrimiento no respondio. No tira abajo las otras."""


def hn_story(item_url: str, *, client: httpx.Client) -> Candidate | None:
    """Del link de la discusion al articulo que discute.

    Devuelve None para Ask HN y Show HN sin link: ahi la discusion **es** la
    fuente, y ya esta en el dossier por la propia URL de la senal.
    """
    m = _ID_HN.search(item_url)
    if not m:
        return None
    try:
        r = client.get(f"{ALGOLIA_ITEM}/{m.group(1)}")
    except httpx.HTTPError as exc:
        raise SourceLookupFailed(f"algolia inaccesible: {exc}") from exc
    if r.status_code != 200:
        raise SourceLookupFailed(f"algolia devolvio {r.status_code}")
    try:
        cuerpo = r.json() or {}
    except ValueError as exc:
        raise SourceLookupFailed("algolia devolvio respuesta no-JSON") from exc

    url = (cuerpo.get("url") or "").strip()
    if not url:
        return None
    return Candidate(
        url=url,
        title=(cuerpo.get("title") or "").strip(),
        source_name=_dominio(url),
        origin="hacker_news",
    )


def gdelt_articles(
    term: str, *, client: httpx.Client, max_records: int = 10
) -> list[Candidate]:
    """Quien mas escribio sobre el tema, segun el GDELT.

    La consulta se monta con los tokens mas distintivos del termino, y no con
    el titulo entero: titulo de noticia en AND no casa con nada. Si tres tokens
    devuelven vacio, prueba con dos -- una segunda llamada es mas barata que un
    dossier sin corroboracion.
    """
    consultas = _consultas(term)
    if not consultas:
        return []

    ultimo_error = ""
    for consulta in consultas:
        try:
            r = client.get(GDELT_DOC, params={
                "query": consulta, "mode": "artlist", "maxrecords": str(max_records),
                "format": "json", "timespan": "7d", "sort": "hybridrel",
            })
        except httpx.HTTPError as exc:
            ultimo_error = f"gdelt inaccesible: {exc}"
            continue
        if r.status_code != 200:
            ultimo_error = f"gdelt devolvio {r.status_code} (429 es comun sin clave)"
            continue
        try:
            articulos = (r.json() or {}).get("articles") or []
        except ValueError:
            # 200 con cuerpo vacio es la manera del GDELT de decir "consulta sin match".
            articulos = []

        hallados = [
            Candidate(
                url=(a.get("url") or "").strip(),
                title=(a.get("title") or "").strip(),
                source_name=(a.get("domain") or "").strip(),
                origin="gdelt",
            )
            for a in articulos
            if (a.get("url") or "").startswith("http")
        ]
        if hallados:
            return hallados

    if ultimo_error:
        raise SourceLookupFailed(ultimo_error)
    return []


def _consultas(term: str) -> list[str]:
    """Tokens distintivos primero, en dos anchuras: tres terminos y dos."""
    # "5,9 GB" se tokeniza en "5" y "9": digito suelto casa con cualquier
    # noticia y solo gasta una posicion de la consulta. Suelo de dos caracteres
    # para quien tiene digito, tres para el resto.
    candidatos = [
        t for t in tokens(term)
        if len(t) >= 3 or (len(t) >= 2 and any(c.isdigit() for c in t))
    ]
    # Token largo distingue mas que token corto, y digito dentro de token largo
    # suele ser nombre de producto o version ("27b", "5090") -- que es justo lo
    # que separa esta historia de otra sobre el mismo asunto.
    distintivos = sorted(
        dict.fromkeys(candidatos),
        key=lambda t: len(t) + (2 if any(c.isdigit() for c in t) else 0),
        reverse=True,
    )[:MAX_TERMOS_CONSULTA]
    if not distintivos:
        return []
    anchas = [" ".join(distintivos)]
    if len(distintivos) > 2:
        anchas.append(" ".join(distintivos[:2]))
    return anchas


def _peneirar(brutos: list[Candidate], *, limit: int, per_domain: int) -> list[Candidate]:
    vistos: set[str] = set()
    por_dominio: dict[str, int] = {}
    salida: list[Candidate] = []

    for c in brutos:
        if not c.url.startswith("http"):
            continue
        clave = _clave(c.url)
        if clave in vistos:
            continue
        dominio = _dominio(c.url)
        if por_dominio.get(dominio, 0) >= per_domain:
            continue
        vistos.add(clave)
        por_dominio[dominio] = por_dominio.get(dominio, 0) + 1
        salida.append(c)
        if len(salida) >= limit:
            break
    return salida


def _clave(url: str) -> str:
    """URL sin esquema, sin www, sin barra final y sin query de rastreo."""
    resto = url.split("://", 1)[-1].removeprefix("www.")
    return resto.split("?", 1)[0].rstrip("/").casefold()


def _dominio(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0].removeprefix("www.").casefold()
