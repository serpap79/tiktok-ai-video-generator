"""Guionista: de un dossier grabado a un Script listo para el renderizador.

La etapa tiene dos mitades de naturaleza distinta, y mezclarlas es el error que
este archivo evita.

La primera es **juicio**: hook que abre hueco, punto de vista propio,
castellano hablado. Eso es trabajo del juez (porcion 3), con rubrica, y no se
puede decidir por regla.

La segunda es **mecanica**: contar palabra, comprobar si el termino de busqueda
esta en ASCII y salio del vocabulario visual del canal (un pilar solo),
comprobar si el indice de hecho existe en el dossier. Eso no necesita juez
ninguno, y gastar una ronda de revision del juez con error de conteo seria
desperdicio de cuota. Por eso el guionista tiene su propio lazo de correccion,
con el defecto medido devuelto al modelo en texto, y solo entrega al juez un
guion que ya pasa en lo que es verificable.

La franja de duracion es requisito de monetizacion, no gusto: video por debajo
de 60s no es elegible al Creator Rewards. Se estima aqui por el ritmo de habla
(WORDS_PER_SECOND) y se **mide de verdad** solo despues del TTS, por el
renderizador -- y es la medida la que manda.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent.brand.brand import pillar_brief, voice_brief
from agent.brand.checks import check_emoji_muletilla, check_hook, check_numbers
from agent.models import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    SHORT_MAX_DURATION_S,
    SHORT_MIN_DURATION_S,
    WORDS_PER_SECOND,
    Dossier,
    Fact,
    Script,
)
from agent.ports.llm import LLM, Completion, LLMError, Usage, parse_json_object
from agent.research.grounding import missing_numbers
from agent.research.subject import missing_subject
from agent.writer.humanize import humanize as humanize_narration
from agent.writer.visuals import compact_brief, suggest_pillar, validate_broll, validate_terms

# Franja de palabras que corresponde a la franja de duracion exigida, con 2s de
# margen en cada punta: la voz real varia unos por ciento del ritmo medido, y
# 59s no monetiza.
MARGEN_S = 2
MIN_PALABRAS = int((MIN_DURATION_S + MARGEN_S) * WORDS_PER_SECOND)
MAX_PALABRAS = int((MAX_DURATION_S - MARGEN_S) * WORDS_PER_SECOND)
# Alias historicos: los tests y los comandos importan estas constantes por su
# nombre original del modulo.
MIN_PALAVRAS = MIN_PALABRAS
MAX_PALAVRAS = MAX_PALABRAS
MIN_WORDS = MIN_PALABRAS
MAX_WORDS = MAX_PALABRAS

# short no monetiza (25s << 60s): su funcion es alcance, no ingreso.
# Subio de 30-50 a 58-70 palabras en 20/09/2026, cuando la parrilla paso a
# pedir corto de ~25s (era ~15s). A 2,57 palabras/s medidas, 58-70 palabras
# dan 23-27s. Final en bucle, para el replay automatico.
MIN_PALABRAS_CORTO = math.ceil(SHORT_MIN_DURATION_S * WORDS_PER_SECOND)
MAX_PALABRAS_CORTO = math.floor(SHORT_MAX_DURATION_S * WORDS_PER_SECOND)
BANDS: dict[str, tuple[int, int]] = {
    "long": (MIN_PALABRAS, MAX_PALABRAS),
    "short": (MIN_PALABRAS_CORTO, MAX_PALABRAS_CORTO),
}
# Un clip de b-roll cubre ~5s, asi que el corto de 25s pide mas terminos que el
# de 15s -- con 2 terminos el mismo clip volvia tres veces en el mismo video.
TERMS_PER_MODE: dict[str, tuple[int, int]] = {
    "long": (4, 8),
    "short": (3, 5),
}

# "[0]", "[1, 2]": el indice del hecho colado dentro del texto. Medido en la
# primera ejecucion real (18/09/2026): el modelo escribio "...en tu proyecto
# [0]." y "...en un proyecto [0, 3]." -- y el TTS lo diria "cero" y "uno" en
# voz alta.
_MARCADOR_DE_CITA = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")

# Por debajo de esto el dossier no sostiene 60 segundos de narracion. Medido:
# un dossier de 4 hechos sacados de UNA frase de changelog llevo al guionista a
# tres intentos, todos entre 104 y 157 palabras, sin nunca alcanzar las 150 --
# porque no habia asunto, y no porque la instruccion estuviera mal.
MIN_HECHOS_PARA_GUION = 3
# Alias historico.
MIN_FATOS_PARA_ROTEIRO = MIN_HECHOS_PARA_GUION

# Intentos totales, contando el primero. Dos correcciones bastan para defecto
# mecanico; si el modelo no acierta el conteo en tres intentos, el problema no
# es la instruccion, e insistir solo quema cuota del free tier.
MAX_INTENTOS = 3
# Alias historico.
MAX_TENTATIVAS = MAX_INTENTOS

SISTEMA = (
    "Eres guionista de un canal espanol de tech, IA y ciencia. Escribes para ser "
    "oido, no leido: frase corta, voz activa, cero jerga sin explicar. Solo "
    "afirmas lo que esta en el dossier que recibes. Numero que no esta en el "
    "dossier no entra en el guion, ni como aproximacion."
)

SCHEMA_GUION: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hook": {"type": "string"},
        "body": {"type": "string"},
        "closing": {"type": "string"},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "broll": {"type": "array", "items": {"type": "string"}},
        "caption": {"type": "string"},
        "used_facts": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["hook", "body", "closing", "search_terms", "broll", "caption",
                 "used_facts"],
}


@dataclass
class Attempt:
    """Un intento y lo que vio. Vacio significa que fue aceptado.

    `narration` guarda el texto reprobado. Sin el, entender POR QUE una puerta
    reprobo exige correr de nuevo y pagar la cuota otra vez -- fue lo que paso
    en la primera ejecucion real, con una puerta acusando "numero 0, 1, 2" sin
    que hubiera forma de ver de donde venian los numeros.
    """

    violations: list[str] = field(default_factory=list)
    word_count: int = 0
    narration: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


@dataclass
class WriteReport:
    """El guion, los intentos que hicieron falta, y el coste de todos.

    Los intentos quedan grabados porque dicen donde el prompt esta debil: si
    toda ejecucion gasta dos rondas para acertar el conteo de palabras, el
    defecto esta en la instruccion, no en el modelo -- y eso solo aparece si el
    intervalo se registra en vez de descartarse en el exito.
    """

    topic: str
    script: Script | None = None
    attempts: list[Attempt] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    # Pasada de humanizacion despues del aceite mecanico. No es intento: no
    # reprueba, solo mejora -- o mantiene el original con motivo.
    humanized: bool = False
    humanize_notes: list[str] = field(default_factory=list)
    humanize_usage: Usage = field(default_factory=Usage)
    humanize_latency_s: float = 0.0
    # Motivo de ni siquiera haber intentado. Distinto de intento reprobado: aqui
    # ninguna llamada se hizo, y el coste es cero.
    refusal: str = ""

    @property
    def ok(self) -> bool:
        return self.script is not None

    @property
    def usage(self) -> Usage:
        total = Usage()
        for a in self.attempts:
            total = total + a.usage
        return total + self.humanize_usage

    @property
    def latency_s(self) -> float:
        return round(
            sum(a.latency_s for a in self.attempts) + self.humanize_latency_s, 3)

    @property
    def violations(self) -> list[str]:
        """Violaciones del ultimo intento: el motivo de haber fallado."""
        return self.attempts[-1].violations if self.attempts else []


class Screenwriter:
    def __init__(self, llm: LLM, max_attempts: int = MAX_INTENTOS):
        self._llm = llm
        self._max_attempts = max_attempts

    def write(self, dossier: Dossier, notes: list[str] | None = None,
              mode: str = "long", polish: bool = True, pillar: str = "",
              previous: str = "") -> WriteReport:
        """Escribe el guion. `notes` son las notas de revision del juez.

        Entran por el mismo canal de las violaciones mecanicas -- el modelo
        recibe una lista de defectos que corregir y no necesita saber cual de
        ellos fue contado y cual juzgado.
        """
        if mode not in BANDS:
            raise ValueError(f"modo desconocido: {mode!r}; usa long o short")
        report = WriteReport(
            topic=dossier.topic,
            model=getattr(self._llm, "model", ""),
            provider=getattr(self._llm, "provider", ""),
        )

        report.refusal = thin_dossier_reason(dossier, mode)
        if report.refusal:
            # Medir antes de pagar, como hacen el curador y el juez: dossier que
            # no sostiene 60s de narracion no se convierte en guion por
            # insistencia, y probar tres veces solo gastaria cuota para llegar a
            # la misma pared.
            return report

        correccion: list[str] = list(notes or [])
        # Texto del intento anterior: sin el, "manten lo que estaba bien y
        # arregla lo apuntado" le pedia al modelo mantener algo que no veia --
        # cada correccion era una escrita desde cero, y el conteo de palabras
        # oscilaba en vez de converger.
        anterior = previous

        for _ in range(self._max_attempts):
            try:
                respuesta = self._llm.complete(
                    build_prompt(dossier, correccion, mode, pillar=pillar,
                                 previous=anterior if correccion else ""),
                    system=SISTEMA,
                    schema=SCHEMA_GUION,
                    temperature=0.6,
                    max_output_tokens=2048,
                )
            except LLMError:
                # Cuota rebasada o filtro de contenido: no hay nada que corregir
                # en el prompt, asi que sube a quien llamo. Defecto de forma en
                # la respuesta se trata en _evaluar, como violacion corregible.
                raise

            # Con el router, quien respondio solo se conoce despues de la llamada.
            report.model, report.provider = respuesta.model, respuesta.provider
            intento, script = self._evaluar(respuesta, dossier, mode, pillar)
            report.attempts.append(intento)
            if not intento.violations:
                report.script = script
                if polish and script is not None:
                    self._pulir(report, script, dossier, mode)
                return report
            correccion = intento.violations
            anterior = intento.narration or anterior

        return report

    def _pulir(self, report: WriteReport, script: Script,
               dossier: Dossier, mode: str) -> None:
        """Pasada de humanizacion. Original intacto si la reescritura falla."""
        minimo, maximo = BANDS[mode]
        rel = humanize_narration(
            script.hook, script.body, script.closing, dossier,
            self._llm, minimo, maximo)
        report.humanize_usage = rel.usage
        report.humanize_latency_s = rel.latency_s
        report.humanize_notes = rel.notes
        if rel.changed:
            report.script = script.model_copy(update={
                "hook": rel.hook, "body": rel.body, "closing": rel.closing})
            report.humanized = True

    # ------------------------------------------------------------------ evaluacion

    def _evaluar(self, respuesta: Completion, dossier: Dossier,
                 mode: str = "long", pillar: str = "") -> tuple[Attempt, Script | None]:
        intento = Attempt(usage=respuesta.usage, latency_s=respuesta.latency_s)
        if respuesta.truncated:
            intento.violations.append(
                "la respuesta fue cortada por limite de tokens; escribe mas corto"
            )
            return intento, None

        try:
            cuerpo = parse_json_object(respuesta.text)
        except LLMError as exc:
            # Defecto de forma, no de proveedor: el lazo lo arregla, y gastar una
            # ronda del juez con JSON roto seria desperdicio de cuota.
            intento.violations.append(f"la respuesta no vino como objeto JSON: {exc}")
            return intento, None

        usados, fuera = _resolver_hechos(cuerpo.get("used_facts"), dossier.facts)

        try:
            script = Script(
                topic=dossier.topic,
                hook=_texto(cuerpo.get("hook")),
                body=_texto(cuerpo.get("body")),
                closing=_texto(cuerpo.get("closing")),
                search_terms=_terminos(cuerpo.get("search_terms")),
                facts=usados,
                format=mode,
                broll=_terminos(cuerpo.get("broll"))[:3],
                pillar=pillar,
                caption=_leyenda(cuerpo.get("caption")),
            )
        except ValidationError as exc:
            intento.violations.extend(_violaciones_de_contrato(exc))
            return intento, None

        intento.word_count = script.word_count
        intento.narration = script.narration
        intento.violations.extend(_violaciones_mecanicas(script, dossier, fuera, mode))
        return intento, (script if not intento.violations else None)


def _violaciones_mecanicas(script: Script, dossier: Dossier, fuera: list[int],
                           mode: str = "long") -> list[str]:
    """Lo que se puede comprobar sin juicio. El texto vuelve al modelo."""
    problemas: list[str] = []
    minimo, maximo = BANDS[mode]
    tmin, tmax = TERMS_PER_MODE[mode]

    marcadores = _MARCADOR_DE_CITA.findall(script.narration)
    if marcadores:
        problemas.append(
            f"la narracion contiene marcador de cita ({', '.join(marcadores[:4])}). "
            "El texto lo habla un sintetizador: diria esos numeros en voz alta. "
            "El indice del hecho va SOLO en el campo used_facts."
        )

    if not (minimo <= script.word_count <= maximo):
        objetivo = (minimo + maximo) // 2
        problemas.append(
            f"la narracion tiene {script.word_count} palabras "
            f"(~{script.estimated_duration_s:.0f}s) y necesita tener entre {minimo} y "
            f"{maximo}. Reescribe con unas {objetivo} palabras."
        )

    if not (tmin <= len(script.search_terms) <= tmax):
        problemas.append(
            f"search_terms tiene {len(script.search_terms)} terminos y el modo {mode} "
            f"pide entre {tmin} y {tmax}, en orden cronologico."
        )

    if fuera:
        problemas.append(
            f"used_facts apunta a indice que no existe en el dossier: {fuera}. "
            f"Los indices validos van de 0 a {len(dossier.facts) - 1}."
        )
    if not script.facts:
        problemas.append(
            "used_facts esta vacio: todo guion necesita apoyarse en al menos "
            "un hecho del dossier, con fuente."
        )

    sueltos = _numeros_sin_dossier(script, dossier)

    if sueltos:
        problemas.append(
            f"la narracion cita numero que no esta en el dossier: {', '.join(sueltos)}. "
            "Usa solo los numeros de los hechos, sin convertir unidad y sin redondear."
        )

    problemas.extend(validate_terms(script.search_terms))
    problemas.extend(validate_broll(script.broll, mode))
    problemas.extend(caption_problems(script))

    sin_sujeto = missing_subject(script.narration, dossier.topic)
    if sin_sujeto:
        problemas.append(
            "el guion habla de 'un modelo' sin nombrarlo: cita "
            + ", ".join(f"{t!r}" for t in sin_sujeto) + " (nombre y creador, "
            "conforme el dossier -- nunca inventes). Sin nombre no hay busqueda ni "
            "credibilidad.")

    fallo_gancho = check_hook(script.hook)
    if fallo_gancho is not None:
        problemas.append(fallo_gancho)
    problemas.extend(check_numbers(script.narration))
    problemas.extend(check_emoji_muletilla(script.narration))

    return problemas


def _numeros_sin_dossier(script: Script, dossier: Dossier) -> list[str]:
    """Numero en digito en la narracion tiene que estar en algun hecho del dossier.

    Es la misma puerta que el investigador usa para comprobar hecho contra
    pagina, con el dossier en el lugar de la pagina -- la pregunta es identica
    ("este numero existe en la fuente?") y tener dos implementaciones de ella
    garantizaria dos respuestas.

    Limite conocido: coge solo lo que esta escrito en digito. La narracion buena
    escribe numero con letras para el TTS ("cinco coma nueve gigabytes"), y
    comprobar eso exigiria convertir numeral en castellano de vuelta a digito.
    Quien cubre ese caso es el criterio 2 de la rubrica del juez, con el dossier
    en mano -- esta puerta solo garantiza que lo barato de comprobar nunca pase
    mal.
    """
    fuentes = "\n".join(f"{f.claim}\n{f.quote}" for f in dossier.facts)
    # El marcador de cita sale antes de la cuenta: tiene violacion propia, y
    # dejarlo aqui haria la puerta acusar "numero 0, 1, 2 sin respaldo" -- que es
    # verdad y no ayuda a nadie a entender que hacer.
    narracion = _MARCADOR_DE_CITA.sub(" ", script.narration)
    return missing_numbers(narracion, fuentes)


# Hechos minimos por formato. El corto es "una sola idea, 1 o 2 hechos": exigir
# 3 de el rechazaba justo el dossier que solo sirve para corto (visto en el
# test del piloto en 20/09 -- el formato eligio corto y el guionista rechazo).
MIN_HECHOS_POR_MODO = {"long": MIN_HECHOS_PARA_GUION, "carousel": MIN_HECHOS_PARA_GUION,
                       "short": 1}


def thin_dossier_reason(dossier: Dossier, mode: str = "long") -> str:
    """Motivo para no intentar escribir, o cadena vacia si se puede intentar.

    La franja de 60-90s exige unas 150 palabras de contenido. Dossier con dos
    hechos sacados de la misma frase no tiene eso, y el guionista solo tiene
    dos salidas: llenar de relleno, o inventar. Las dos son peores que rechazar
    con motivo.

    El numero de FUENTES no entra: una fuente rica rinde guion (el guion de
    referencia del M0 tiene cinco hechos de un unico release). Lo que cuenta es
    cuantos hechos distintos existen.
    """
    minimo = MIN_HECHOS_POR_MODO.get(mode, MIN_HECHOS_PARA_GUION)
    if len(dossier.facts) < minimo:
        objetivo = (f"la franja de {MIN_DURATION_S}-{MAX_DURATION_S}s" if mode == "long"
                    else f"el formato {mode}")
        return (
            f"dossier fino: {len(dossier.facts)} hecho(s), y {objetivo} pide al menos "
            f"{minimo}. Investiga otras fuentes antes de escribir el guion."
        )
    return ""


def _resolver_hechos(indices: object, facts: list[Fact]) -> tuple[list[Fact], list[int]]:
    """Traduce los indices que el modelo devolvio a hechos del dossier.

    El modelo apunta, nunca copia: si pudiera reescribir el hecho, la
    afirmacion del guion dejaria de ser rastreable a lo que la fuente dice --
    que es todo el punto de que el dossier exista.
    """
    if not isinstance(indices, list):
        return [], []

    usados: list[Fact] = []
    fuera: list[int] = []
    for crudo in indices:
        if isinstance(crudo, bool) or not isinstance(crudo, int):
            continue
        if 0 <= crudo < len(facts):
            if facts[crudo] not in usados:
                usados.append(facts[crudo])
        else:
            fuera.append(crudo)
    return usados, fuera


def _violaciones_de_contrato(exc: ValidationError) -> list[str]:
    salida: list[str] = []
    for error in exc.errors():
        campo = ".".join(str(p) for p in error["loc"]) or "guion"
        salida.append(f"el campo {campo} no respeta el contrato: {error['msg']}")
    return salida


# Ejemplo de ~25s que cabe en la franja: el modelo imita el TAMANO, no solo el
# tono -- medido en 19/09/2026, sin ejemplo de extension escribia el corto en
# el tamano del largo. Tiene 65 palabras a proposito, el medio de la franja.
SHORT_EXAMPLE = (
    "hook: Un modelo gigante cabe en tu bolsillo?\n"
    "body: El Bonsai 2 tiene veintisiete mil millones de parametros en solo "
    "cinco coma nueve gigabytes. Es nueve veces mas pequeno que el original y "
    "mantiene casi todo el rendimiento. En la practica, funciona en un portatil "
    "comun, sin nube, sin cuota y sin mandar tus datos al servidor de nadie.\n"
    "closing: Gigante en el bolsillo: que mas se encogera?")

# Cuantos datos numericos aguanta el video. El largo del 19/09 apilo siete
# ("ciento cuarenta y tres tokens por segundo... cero punto setecientos
# catorce miliwatios-hora") y se convirtio en ficha tecnica leida en voz alta.
MAX_NUMEROS = {"long": 3, "short": 1}


def build_prompt(dossier: Dossier, correcciones: list[str] | None = None,
                 mode: str = "long", pillar: str = "", previous: str = "") -> str:
    """Compone el prompt del guion. Funcion libre para que el test inspeccione el texto.

    Orden: tema y dossier (el material), tarea y tipo de contenido (la forma),
    estetica y voz (la marca), reglas (lo que reprueba), correccion (solo a la
    vuelta).
    """
    hechos = "\n".join(
        f"[{i}] {f.claim}\n    fuente: {f.source_name}"
        + (f'\n    pasaje: "{f.quote}"' if f.quote else "")
        for i, f in enumerate(dossier.facts)
    )
    minimo, maximo = BANDS[mode]
    tmin, tmax = TERMS_PER_MODE[mode]
    objetivo = (minimo + maximo) // 2
    cierre = (
        "- closing: cierre con PUNTO DE VISTA PROPIO: apunta lo que las fuentes "
        "NO dicen, o la pregunta que dejan abierta, y se lo devuelve al "
        "espectador. Llamada concreta, nunca 'sigue para mas'.\n"
        if mode == "long" else
        "- closing: UNA frase que reconecta con la pregunta del hook Y apunta la "
        "implicacion que la fuente no desarrolla (ej.: 'Si son dos organos, cual "
        "de ellos decide por ti?'). Sin ella el video es resumen. Sin 'sigue para mas'.\n"
    )
    duracion_txt = (
        f"Equivale a {MIN_DURATION_S}-{MAX_DURATION_S}s hablados y es "
        "requisito de monetizacion, no preferencia.\n"
        if mode == "long" else
        "Video corto de alcance (~25s): una sola idea, 2 o 3 hechos como maximo. "
        "Cuenta las palabras antes de responder y corta frases enteras si te pasas "
        "del techo.\n"
    )
    partes = [
        f"TEMA: {dossier.topic}\n",
        f"DOSSIER (usa el indice para citar):\n{hechos}\n",
        "TAREA\n"
        "Guion de video vertical en castellano, para ser narrado. Devuelve:\n"
        "- hook: la PRIMERA FRASE abre un hueco de informacion y no lo responde. "
        "Hasta 12 palabras (regla de la marca). Nada de 'hoy os voy a hablar de'.\n"
        "- body: todo dato viene de un hecho del dossier. En la primera frase, "
        "contexto: nombra el asunto (nombre + quien lo construyo, SI el dossier lo "
        "dice) y por que importa para quien mira.\n"
        + cierre +
        f"- search_terms: de {tmin} a {tmax} terminos EN INGLES, en orden cronologico "
        "de la narracion, COPIADOS de la lista de ESTETICA.\n"
        "- broll: terminos EN INGLES del objeto concreto del asunto (regla abajo).\n"
        "- caption: leyenda del post en 2 lineas: un gancho ESCRITO distinto del habla "
        "y una frase de contexto (quien, que). Sin emoji, link ni hashtag.\n"
        "- used_facts: los indices de los hechos del dossier en los que el guion se apoya.\n",
        pillar_brief(pillar or "news", mode),
        compact_brief(suggest_pillar(dossier.topic), mode),
        *([] if mode != "short" else [
            "EJEMPLO DE TAMANO (25s: copia la extension, no el texto)\n" + SHORT_EXAMPLE]),
        voice_brief(),
        "REGLAS\n"
        f"- La narracion entera (hook + body + closing) tiene entre {minimo} y {maximo} "
        f"palabras, unas {objetivo}. " + duracion_txt +
        f"- Como maximo {MAX_NUMEROS[mode]} dato(s) numerico(s) en el video entero: elige "
        "el que el espectador recordaria y traducelo en comparacion. Ficha tecnica "
        "leida en voz alta pierde a la gente.\n"
        "- Numero con letras cuando suene mejor ('cinco coma nueve gigabytes'), "
        "sin cambiar el valor. No inventes numero, nombre, fecha ni cita.\n"
        "- Descripcion en ingles se convierte en castellano en el habla ('High-Resolution "
        "Stereo Camera' -> 'camara estereo de alta resolucion'); en ingles, solo nombre "
        "propio corto.\n"
        "- NO escribas indices en el texto ('[0]', '[1, 2]'): el sintetizador de voz "
        "los diria en voz alta. El indice va solo en el campo used_facts.\n"
        "- Sin emoji, sin hashtag, sin marcacion de escena. Solo lo que se dira.",
    ]
    if correcciones:
        bloque = "CORRIGE EL INTENTO ANTERIOR\n" + "\n".join(f"- {c}" for c in correcciones)
        if previous:
            bloque += f"\nTEXTO ANTERIOR (ajusta este, no empieces de cero):\n{previous}"
        partes.append(bloque + "\nManten lo que estaba bien y corrige solo lo apuntado.")
    return "\n".join(partes)


def _texto(valor: object) -> str:
    return " ".join(str(valor).split()) if isinstance(valor, str) else ""


def _leyenda(valor: object) -> str:
    """La leyenda preserva el salto de linea (son dos lineas por regla de marca)."""
    if not isinstance(valor, str):
        return ""
    lineas = [" ".join(linea.split()) for linea in valor.splitlines()]
    return "\n".join(linea for linea in lineas if linea)


_LINK = re.compile(r"https?://|www\.|\.com\b|\.es\b", re.IGNORECASE)


def caption_problems(script: Script) -> list[str]:
    """Reglas de la leyenda en la guia: 2 lineas, sin emoji/link/hashtag, sin repetir el audio."""
    leyenda = script.caption.strip()
    if not leyenda:
        return ["caption vacia: escribe 2 lineas (gancho escrito distinto del habla + "
                "una frase de contexto: quien, que)."]
    problemas: list[str] = []
    lineas = leyenda.splitlines()
    if len(lineas) > 2 or len(leyenda) > 220:
        problemas.append(f"caption con {len(lineas)} linea(s) y {len(leyenda)} caracteres: "
                         "como maximo 2 lineas cortas (regla de la marca).")
    if "#" in leyenda:
        problemas.append("caption con hashtag: las 5 hashtags de la marca entran solas.")
    if _LINK.search(leyenda):
        problemas.append("caption con link o dominio: la regla de la marca es sin link.")
    if check_emoji_muletilla(leyenda):
        problemas.append("caption con emoji o muletilla: la marca nunca los usa.")
    if script.hook and leyenda.splitlines()[0].strip().lower() == script.hook.strip().lower():
        problemas.append("caption repite el hook hablado: la primera linea es un gancho "
                         "ESCRITO distinto del audio (guia de la marca).")
    return problemas


def _terminos(valor: object) -> list[str]:
    if not isinstance(valor, list):
        return []
    return [" ".join(str(t).split()) for t in valor if isinstance(t, str) and t.strip()]
