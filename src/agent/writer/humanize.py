"""Humanizacion: quita el vicio de texto de IA de la narracion, sin inventar nada.

Integracion del `blader/humanizer` (MIT, 50k estrellas) como etapa del
guionista -- no como dependencia: alli es un skill de prompt para agente, aqui
se vuelve una pasada con frenos de grounding. La atribucion es idea de alli;
las reglas de seguridad son nuestras.

Dos mitades, como el resto del guionista:

1. **Scan determinista** (`scan`): los tells fuertes y de bajo falso-positivo,
   adaptados al castellano hablado. Sin hallazgo, sin llamada -- la cuota del
   free tier no se gasta con texto que ya suena humano.
2. **Reescritura en una llamada** (`humanize`): el modelo recibe los pasajes
   marcados y reescribe hook/body/closing bajo los mismos frenos del guion
   (hechos, numeros, franja de palabras). Si la reescritura rompe numero o
   franja, vale el original con motivo -- reescritura que inventa dato es peor
   que vicio de estilo.

Lo que NO pasa por aqui: `search_terms` (vocabulario cerrado, no prosa) y
slides de carrusel (lineas cortas de impacto, no parrafos).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.models import Dossier
from agent.ports.llm import LLM, Completion, LLMError, Usage, parse_json_object
from agent.research.grounding import missing_numbers
from agent.text import normalize

# Tells fuertes en castellano hablado. Lista corta a proposito: cada item aqui
# dispara UNA llamada de modelo, asi que el falso-positivo cuesta cuota. Solo
# entra lo que casi nunca aparece en habla natural de canal tech.
_TELLS: tuple[tuple[str, str], ...] = (
    ("no es solo", "contraste escenificado (no-es-X-es-Y)"),
    ("no se trata de", "contraste escenificado"),
    ("sumergir", "palabra de IA (sumergirse/delve)"),
    ("sumergete", "palabra de IA (sumergirse/delve)"),
    ("panorama", "palabra de IA (landscape figurado)"),
    ("constituye", "palabra de IA (testament)"),
    ("deslumbrante", "lenguaje de anuncio"),
    ("vibrante", "lenguaje de anuncio"),
    ("en la era", "dicho profundo (saying)"),
    ("en resumen", "cierre de cartilla"),
    ("para concluir", "cierre de cartilla"),
    ("concluyendo", "cierre de cartilla"),
    ("es importante senalar", "relleno (filler)"),
    ("cabe senalar", "relleno (filler)"),
    ("vale la pena destacar", "relleno (filler)"),
    ("en el mundo de hoy", "preambulo escenificado"),
    ("en la era digital", "preambulo escenificado"),
    ("sigue para mas", "CTA generico (la rubrica reprueba)"),
)

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]"
)


def scan(narration: str) -> list[str]:
    """Pasajes con vicio, en texto que va al modelo. Vacio = nada que hacer."""
    baja = normalize(narration)
    hallados = [f"{t!r} ({motivo})" for t, motivo in _TELLS if t in baja]
    if _EMOJI.search(narration):
        hallados.append("emoji en el texto (el TTS no habla emoji)")
    return hallados


HUMANIZE_SYSTEM = (
    "Revisas narracion de video corto de canal espanol de tech, IA y ciencia. "
    "Suena como una persona hablando, no como un chatbot escribiendo. Nunca "
    "inventas hecho, numero, nombre o fecha: todo lo que afirmas ya esta en el "
    "texto."
)


def build_prompt(hook: str, body: str, closing: str,
                 flags: list[str], min_words: int, max_words: int) -> str:
    objetivos = "\n".join(f"- {f}" for f in flags)
    return (
        "TAREA\nReescribe la narracion de abajo manteniendo TODOS los hechos, "
        "numeros y nombres con los mismos valores. Puedes cortar relleno, unir "
        "frase y cambiar palabra enlatada por habla natural corta.\n"
        f"Pasajes marcados:\n{objetivos}\n"
        "REGLAS\n"
        f"- Entre {min_words} y {max_words} palabras en total (hook + body + closing).\n"
        "- Castellano hablado, frase corta, voz activa. Sin emoji.\n"
        "- Sin contraste escenificado ('no es X, es Y'), sin preambulo "
        "('sumergete en', 'en el mundo de hoy'), sin cierre de cartilla "
        "('en resumen').\n"
        "- Termina en el ultimo hecho concreto, no en optimismo vago.\n"
        "- NO escribas indices como [0] en el texto.\n"
        "NARRACION\n"
        f"hook: {hook}\nbody: {body}\nclosing: {closing}\n"
        'Devuelve JSON {"hook": ..., "body": ..., "closing": ...} y nada mas.'
    )


@dataclass
class HumanizeReport:
    hook: str
    body: str
    closing: str
    changed: bool = False
    notes: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


def humanize(hook: str, body: str, closing: str, dossier: Dossier,
             llm: LLM, min_words: int, max_words: int) -> HumanizeReport:
    """Una pasada de humanizacion. Original intacto si la reescritura falla."""
    original = HumanizeReport(hook=hook, body=body, closing=closing)
    flags = scan("\n".join((hook, body, closing)))
    if not flags:
        return original
    try:
        respuesta: Completion = llm.complete(
            build_prompt(hook, body, closing, flags, min_words, max_words),
            system=HUMANIZE_SYSTEM,
            schema={"type": "object",
                    "properties": {"hook": {"type": "string"},
                                   "body": {"type": "string"},
                                   "closing": {"type": "string"}},
                    "required": ["hook", "body", "closing"]},
            temperature=0.7,
            max_output_tokens=2048,
        )
    except LLMError as exc:
        original.notes.append(f"humanizacion saltada (proveedor): {exc}")
        return original
    original.usage = respuesta.usage
    original.latency_s = respuesta.latency_s
    if respuesta.truncated:
        original.notes.append("reescritura cortada; vale el original")
        return original
    try:
        cuerpo = parse_json_object(respuesta.text)
    except LLMError as exc:
        original.notes.append(f"reescritura fuera de formato; vale el original: {exc}")
        return original
    nuevo = HumanizeReport(
        hook=" ".join(str(cuerpo.get("hook", "")).split()),
        body=" ".join(str(cuerpo.get("body", "")).split()),
        closing=" ".join(str(cuerpo.get("closing", "")).split()),
        usage=respuesta.usage, latency_s=respuesta.latency_s,
    )
    if not (nuevo.hook and nuevo.body and nuevo.closing):
        nuevo.notes.append("reescritura con campo vacio; vale el original")
        return original
    palabras = len(f"{nuevo.hook} {nuevo.body} {nuevo.closing}".split())
    if not (min_words <= palabras <= max_words):
        nuevo.notes.append(
            f"reescritura con {palabras} palabras (franja {min_words}-{max_words}); "
            "vale el original")
        nuevo.hook, nuevo.body, nuevo.closing = hook, body, closing
        return nuevo
    fuentes = "\n".join(f"{f.claim}\n{f.quote}" for f in dossier.facts)
    sueltos = missing_numbers(f"{nuevo.hook} {nuevo.body} {nuevo.closing}", fuentes)
    if sueltos:
        nuevo.notes.append(
            f"la reescritura cambio numero ({', '.join(sueltos)}); vale el original")
        nuevo.hook, nuevo.body, nuevo.closing = hook, body, closing
        return nuevo
    nuevo.changed = True
    nuevo.notes.append(f"tells corregidos: {len(flags)}")
    return nuevo


__all__ = ["HumanizeReport", "HUMANIZE_SYSTEM", "build_prompt", "humanize", "scan"]
