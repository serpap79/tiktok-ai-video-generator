"""Investigador: de un tema elegido a un dossier con fuente en cada hecho.

La decision de proyecto que sostiene todo lo demas: **una llamada de modelo por
fuente, y la URL la estampamos nosotros, no se le pide a el.** El modelo recibe
el texto de una página y devuelve afirmaciones sobre esa página; de dónde proceden
el texto es informacion que ya tenemos. Pedir `source_url` al modelo invitaria
el error mas caro posible en este proyecto -- hecho real con fuente cambiada,
que parece anclado, pasa al juez y solo aparece cuando alguien pincha el link.

Con esto, "no existe Fact sin URL verificable" deja de depender de que el
modelo haya sido honesto y pasa a ser estructural. Lo que todavia depende de el
es la fidelidad de la afirmacion al texto, y es ahi donde entran las dos puertas
deterministas:

1. **el pasaje citado tiene que existir en la página.** El modelo devuelve,
   junto a cada afirmacion, el pasaje literal que la sostiene. Comprobar el
   pasaje es `in` sobre una string -- barato e imposible de enganar.
2. **todo numero de la afirmacion tiene que estar en la fuente**
   (`grounding.py`).

Las dos derriban el hecho con motivo grabado, nunca en silencio. Dossier corto
con motivo registrado es calibrable; dossier lleno de hechos flojos no lo es.

Hay una tercera regla, y vino de una ejecucion real: **un pasaje sostiene un
hecho solo**. Sin eso, el modelo divide una frase de changelog en cuatro
afirmaciones y entrega un dossier que parece lleno y no da asunto para 60
segundos de video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import ValidationError

from agent.models import Decision, Dossier, Fact
from agent.ports.llm import LLM, LLMError, Usage, parse_json_object
from agent.research import grounding, sources
from agent.research.fetch import Page, PageFetcher, PageUnavailable
from agent.research.sources import Candidate
from agent.text import tokens

SISTEMA = (
    "Eres investigador de un canal de tech, IA y ciencia. Tu unica funcion es "
    "extraer afirmaciones factuales del texto que recibes, sin anadir nada que "
    "el texto no diga. No opinas, no contextualizas con conocimiento propio y "
    "no rellenas huecos con lo que suele ser verdad. Si el texto no trata del "
    "tema pedido, devuelve una lista vacia."
)

# El schema es el mismo para los dos proveedores: Gemini lo recibe como
# responseSchema nativo y Groq como texto en el system. Pedir `quote` ANTES de
# `claim` es deliberado -- el orden de las claves es el orden en que el modelo
# escribe, y elegir el pasaje primero es lo que hace que la afirmacion salga de
# el, en vez de buscar el pasaje despues para justificar lo que ya habia
# escrito.
SCHEMA_HECHOS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["quote", "claim"],
            },
        }
    },
    "required": ["facts"],
}

_ESPACIOS = re.compile(r"\s+")

# Pasaje demasiado corto casa con cualquier cosa y no prueba nada ("5,9 GB"
# aparece también en el menú). Si es demasiado largo, se convierte en la página entera
# pegada.
MIN_PASAJE = 25
MAX_PASAJE = 400

# Caracteres de la pagina que van al modelo despues del foco en el tema (~1,2K
# tokens). La página completa sigue con `research_page_chars` para la comprobación.
FOCO_CHARS = 5000


@dataclass
class Discarded:
    """Hecho que el modelo produjo y una puerta derribo, con el motivo."""

    claim: str
    reason: str
    source_url: str


@dataclass
class ResearchReport:
    """Lo que la investigacion produjo, lo que descarto y cuanto costo.

    El coste entra en el informe porque el M3 es la primera etapa que gasta
    cuota, y el eval del M5 compara proveedores por calidad **y** por consumo.
    Numero medido en el momento es mas fiable que reconstruido despues del log.
    """

    topic: str
    dossier: Dossier | None = None
    pages: list[Page] = field(default_factory=list)
    discarded: list[Discarded] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    model: str = ""
    # `proveedor:modelo` que respondio por ultima vez cuando el LLM va por rutas.
    route: str = ""

    @property
    def facts(self) -> list[Fact]:
        return list(self.dossier.facts) if self.dossier else []

    @property
    def source_count(self) -> int:
        """Fuentes distintas que sostienen el dossier, por dominio."""
        return len({_dominio(str(f.source_url)) for f in self.facts})

    @property
    def ok(self) -> bool:
        return self.dossier is not None


class Researcher:
    def __init__(
        self,
        llm: LLM,
        fetcher: PageFetcher | None = None,
        client: httpx.Client | None = None,
        max_sources: int = 5,
        max_facts_per_source: int = 4,
    ):
        self._llm = llm
        self._fetcher = fetcher or PageFetcher()
        self._client = client
        self._max_sources = max_sources
        self._max_facts = max_facts_per_source

    def research(
        self, decision: Decision, candidates: list[Candidate] | None = None,
        target_facts: int | None = None,
    ) -> ResearchReport:
        """Lee las fuentes en orden y extrae hechos, una llamada por fuente.

        `target_facts`: deja de leer cuando ya hay hechos suficientes de al
        menos dos fuentes (o dos mas que el objetivo de una fuente sola). Cada
        fuente es una llamada al modelo; leer una quinta página cuando las tres primeras
        primeras ya dieron seis hechos es cuota gastada sin ganancia de guion.
        """
        report = ResearchReport(topic=decision.term, model=getattr(self._llm, "model", ""))

        if candidates is None:
            descubrimiento = sources.discover(
                decision, client=self._client, limit=self._max_sources
            )
            candidatos = descubrimiento.candidates
            report.failures.update(descubrimiento.failures)
        else:
            candidatos = candidates[: self._max_sources]

        if not candidatos:
            report.failures.setdefault(
                "descubrimiento", "ninguna fuente candidata para el tema"
            )
            return report

        hechos: list[Fact] = []
        for candidato in candidatos:
            try:
                página = self._fetcher.fetch(candidato.url, candidato.source_name)
            except PageUnavailable as exc:
                report.failures[candidato.url] = str(exc)
                continue

            if not página.usable:
                report.failures[candidato.url] = (
                    f"texto demasiado corto ({len(página.text)} caracteres) para extraer un hecho"
                )
                continue

            report.pages.append(página)
            try:
                nuevos, descartados, uso, latencia = self._extract(página, decision.term)
            except LLMError as exc:
                # Cuota rebasada o filtro de contenido del proveedor. Una fuente
                # perdida no invalida las otras, igual que en el radar.
                report.failures[candidato.url] = f"{type(exc).__name__}: {exc}"
                continue

            hechos.extend(nuevos)
            report.discarded.extend(descartados)
            report.usage = report.usage + uso
            report.latency_s = round(report.latency_s + latencia, 3)

            if target_facts and _suficiente(hechos, target_facts):
                break

        if hechos:
            report.dossier = Dossier(
                topic=decision.term, facts=hechos, collected_at=datetime.now(UTC)
            )
        return report

    # ------------------------------------------------------------------ extraccion

    def _extract(
        self, page: Page, topic: str
    ) -> tuple[list[Fact], list[Discarded], Usage, float]:
        respuesta = self._llm.complete(
            build_prompt(page, topic, self._max_facts),
            system=SISTEMA,
            schema=SCHEMA_HECHOS,
            temperature=0.1,
            max_output_tokens=1536,
        )
        cuerpo = parse_json_object(respuesta.text)
        crudos = cuerpo.get("facts")
        if not isinstance(crudos, list):
            raise LLMError("respuesta sin la lista 'facts'")

        # La página completa es el pajar de las comprobaciones: el título suele contener el
        # numero del titular, que el cuerpo repite en otra forma.
        pajar = _normalizar(f"{page.title}\n{page.text}")
        hechos: list[Fact] = []
        descartados: list[Discarded] = []
        vistos: set[str] = set()
        pasajes: set[str] = set()

        for crudo in crudos[: self._max_facts]:
            if not isinstance(crudo, dict):
                continue
            claim = _limpiar(crudo.get("claim"))
            quote = _limpiar(crudo.get("quote"))
            if not claim:
                continue

            clave = claim.casefold()
            if clave in vistos:
                continue
            vistos.add(clave)

            motivo = _reprobar(claim, quote, pajar)
            if motivo:
                descartados.append(Discarded(claim=claim, reason=motivo, source_url=page.url))
                continue

            clave_pasaje = _normalizar(quote)
            if clave_pasaje in pasajes:
                # Medido en la primera ejecucion real (18/09/2026): de una unica
                # frase de changelog salieron cuatro "hechos", tres apoyados en
                # el MISMO pasaje. Un dossier asi parece lleno y no sostiene 60
                # segundos de narracion -- el guionista dio tres veces con la
                # pared.
                descartados.append(Discarded(
                    claim=claim,
                    reason="el mismo pasaje ya sostiene otro hecho de esta fuente; "
                           "una frase no se convierte en varios hechos",
                    source_url=page.url,
                ))
                continue
            pasajes.add(clave_pasaje)

            try:
                hechos.append(Fact(
                    claim=claim,
                    source_url=page.url,
                    source_name=page.source_name or _dominio(page.url),
                    quote=quote[:MAX_PASAJE],
                ))
            except ValidationError as exc:
                # Afirmacion demasiado corta para el contrato, o URL que el
                # Pydantic rechaza. Es descarte de dominio, no bug: entra en el
                # informe.
                descartados.append(Discarded(
                    claim=claim, reason=f"el contrato Fact rechazo: {_primer_error(exc)}",
                    source_url=page.url,
                ))

        return hechos, descartados, respuesta.usage, respuesta.latency_s


def _suficiente(hechos: list[Fact], objetivo: int) -> bool:
    dominios = {_dominio(str(f.source_url)) for f in hechos}
    return (len(hechos) >= objetivo and len(dominios) >= 2) or len(hechos) >= objetivo + 2


def focus(texto: str, topic: str, limite: int) -> str:
    """Los parrafos que hablan del tema, en orden, hasta `limite` caracteres.

    La pagina ya llega cortada en `research_page_chars`, pero el corte era
    ciego: los primeros 8 mil caracteres de una noticia incluyen leyenda de
    foto, "lee tambien" y el parrafo sobre otro producto. Mantener la entrada y
    los parrafos con terminos del tema (y los con numero, que es lo que se
    convierte en hecho) corta token de entrada sin cortar el hecho. La puerta
    de pasaje sigue comprobando contra la pagina ENTERA.
    """
    if len(texto) <= limite:
        return texto
    terminos = {t for t in tokens(topic) if len(t) >= 3 or any(c.isdigit() for c in t)}
    parrafos = [p.strip() for p in re.split(r"\n{2,}|\n", texto) if p.strip()]
    if not parrafos:
        return texto[:limite]
    notas = []
    for i, par in enumerate(parrafos):
        normal = " " + " ".join(tokens(par, drop_stopwords=False)) + " "
        nota = sum(1 for t in terminos if f" {t} " in normal)
        nota += 0.5 if re.search(r"\d", par) else 0.0
        notas.append((i, nota))
    elegidos = {0}
    total = len(parrafos[0])
    for i, nota in sorted(notas, key=lambda x: (-x[1], x[0])):
        if nota <= 0 or i in elegidos:
            continue
        if total + len(parrafos[i]) > limite:
            continue
        elegidos.add(i)
        total += len(parrafos[i])
    # Poco texto caso (pagina que habla del tema con otras palabras): completa
    # en el orden de la página, mejor que enviar solo la entrada.
    for i in range(len(parrafos)):
        if total >= limite * 0.6:
            break
        if i not in elegidos and total + len(parrafos[i]) <= limite:
            elegidos.add(i)
            total += len(parrafos[i])
    return "\n\n".join(parrafos[i] for i in sorted(elegidos))


def _reprobar(claim: str, quote: str, pajar: str) -> str:
    """Motivo por el que el hecho no entra, o cadena vacia si pasa."""
    if len(quote) < MIN_PASAJE:
        return f"pasaje de apoyo ausente o demasiado corto ({len(quote)} caracteres)"

    if _normalizar(quote) not in pajar:
        # El modelo parafraseo donde debia copiar. No se puede saber si la
        # afirmacion es verdadera, y "no se puede saber" reprueba.
        return f"el pasaje citado no existe en la página: '{quote[:80]}'"

    ausentes = grounding.missing_numbers(claim, f"{quote}\n{pajar}")
    if ausentes:
        return f"numero sin respaldo en la fuente: {', '.join(ausentes[:4])}"
    return ""


def build_prompt(page: Page, topic: str, max_facts: int) -> str:
    """Compone el prompt de extraccion. Funcion libre para que el test inspeccione el texto."""
    return (
        f"TEMA EN INVESTIGACION: {topic}\n\n"
        f"FUENTE: {page.source_name} — {page.title or 'sin titulo'}\n"
        "TEXTO DE LA FUENTE (delimitado por <<< >>>):\n"
        f"<<<\n{focus(page.text, topic, FOCO_CHARS)}\n>>>\n\n"
        "TAREA\n"
        f"Extrae como maximo {max_facts} afirmaciones factuales de este texto sobre el tema.\n"
        "Para cada afirmacion, devuelve dos campos:\n"
        "- quote: el pasaje LITERAL del texto de arriba que sostiene la afirmacion, "
        "copiado caracter a caracter, en el idioma original, entre 25 y 400 "
        "caracteres. No reescribas, no traduzcas, no resumas.\n"
        "- claim: la afirmacion en castellano, completa y comprensible sola, "
        "preservando todo numero exactamente como aparece en el pasaje.\n\n"
        "REGLAS\n"
        "- Cada afirmacion tiene que venir de un pasaje DIFERENTE del texto. No "
        "dividas la misma frase en varias afirmaciones: si el texto solo sostiene "
        "una, devuelve una.\n"
        "- Prefiere afirmaciones con numero, fecha, medida o nombre propio.\n"
        "- Cuando el texto diga quien creo, lanzo o mantiene el asunto (empresa, "
        "proyecto, persona), incluye eso en la afirmacion: el guion necesita "
        "nombrar al sujeto, y solo se ancla lo que esta en el dossier.\n"
        "- No inventes numero, no conviertas unidad y no redondees.\n"
        "- No afirmes nada que el texto no diga, aunque sepas que es verdad.\n"
        "- Si el texto no trata del tema en investigacion, devuelve facts como lista vacia."
    )


def _limpiar(valor: object) -> str:
    return " ".join(str(valor).split()) if isinstance(valor, str) else ""


def _normalizar(texto: str) -> str:
    """Espacio colapsado y minusculas, para que el pasaje case pese a formato."""
    return _ESPACIOS.sub(" ", texto).casefold()


def _dominio(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")


def _primer_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    return f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}"
