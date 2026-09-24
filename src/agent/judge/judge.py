"""Juez: decide si el guion merece ser renderizado.

Dos criterios de la rubrica **no se le preguntan al modelo**, y eso es diseno y
no economia:

- **duracion** es conteo de palabra. Preguntarle a un LLM cuantos segundos
  tarda el texto en hablarse es cambiar una medida por un palo.
- **politica** ya tiene filtro escrito contra el radar real
  (`curator/policy.py`). Reutilizar el mismo filtro garantiza que el guion sea
  juzgado por la misma regla que bloqueo el tema, en vez de dos listas que
  divergen con el tiempo.

Un tercero, **fuente**, tiene una parte medida: si la narracion cita numero en
digito que no esta en ningun hecho del dossier, el criterio va a cero sin
consultar a nadie. El resto del criterio -- afirmacion que va mas alla del
dossier, numero escrito con letras -- sigue siendo lectura.

Y el orden entre las dos mitades es el mismo del curador: **medir es barato,
juzgar cuesta cuota**. Cuando la medida ya reprueba en criterio de requisito,
el modelo no se llama, y los criterios de lectura quedan marcados como no
evaluados. Pagar un informe para confirmar un reprobo ya decidido seria quemar
free tier -- y, de regalo, eso hace el reprobo demostrable sin ninguna clave de
API.

Quedan cuatro criterios que son juicio de verdad -- hook, punto de vista,
castellano hablado y cierre -- y para esos no existe alternativa a un lector.

El corte no es solo la suma. La suma sola permite compensacion equivocada: un
guion que es resumen de noticia (cero en punto de vista) llegaria a 12 de 14
con el resto perfecto, siendo exactamente el "AI slop" que el Creator Rewards
excluye. Por eso aprobar exige 11/14, **ningun criterio a cero** y ningun veto
violado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent.curator import policy
from agent.models import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    SHORT_MAX_DURATION_S,
    SHORT_MIN_DURATION_S,
    VETO_MINIMO,
    Criterion,
    CriterionScore,
    Dossier,
    Review,
    Script,
)
from agent.ports.llm import LLM, Completion, LLMError, Usage, parse_json_object
from agent.research.grounding import missing_numbers

# Dos oportunidades para el informe. Truncamiento y JSON malformado son
# intermitentes, y perder un guion ya escrito por eso costaria la ejecucion
# entera siguiente.
MAX_INTENTOS = 2

# Criterios que dependen de lectura. El orden es el de la rubrica original.
JULGADOS = (Criterion.hook, Criterion.fuente, Criterion.punto_de_vista,
            Criterion.es_es, Criterion.cta)

DESCRIPCIONES: dict[Criterion, str] = {
    Criterion.hook: (
        "La primera frase abre un hueco de informacion y no lo responde? "
        "2 = dan ganas de seguir escuchando; 1 = interesa pero regala el asunto "
        "de gratis; 0 = anuncia el tema ('hoy voy a hablar de'). "
        "Promesa que el video no paga (ej. anunciar '5 IAs' y mostrar una) "
        "puntua cero, incluso con frase buena."
    ),
    Criterion.fuente: (
        "Toda afirmacion factual de la narracion esta sostenida por un hecho del "
        "dossier? 2 = todas; 1 = alguna afirmacion va mas alla de lo que el "
        "dossier dice; 0 = hay afirmacion factual sin ningun apoyo en el dossier."
    ),
    Criterion.punto_de_vista: (
        "El guion tiene punto de vista propio, o es resumen de noticia? 2 = "
        "ofrece lectura, contraste o apunta lo que las fuentes NO dicen; 1 = "
        "casi solo recuento, con un comentario; 0 = resumen de la noticia."
    ),
    Criterion.es_es: (
        "Es castellano hablado? 2 = frase corta, voz activa, suena natural "
        "leido en voz alta; 1 = pasajes demasiado escritos, jerga sin explicar "
        "o numeros en secuencia (ficha tecnica leida en voz alta); 0 = trabado, "
        "traducido al pie de la letra."
    ),
    Criterion.cta: (
        "El cierre llama a algo concreto, que no sea 'sigue para mas'? 2 = "
        "pregunta o invitacion concreta ligada al tema; 1 = generica; "
        "0 = pide seguidor o no cierra."
    ),
}

SISTEMA = (
    "Eres editor de un canal espanol de tech, IA y ciencia, y evalua guiones "
    "antes de la produccion. Eres severo y especifico: nota alta es excepcion, y "
    "cada nota viene con el motivo escrito de forma que el guionista sepa que "
    "cambiar. No reescribes el guion, solo juzgas."
)

SCHEMA_INFORME: dict[str, Any] = {
    "type": "object",
    "properties": {
        criterio.value: {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["reason", "score"],
        }
        for criterio in JULGADOS
    },
    "required": [c.value for c in JULGADOS],
}


@dataclass
class ReviewReport:
    """El informe y el coste de producirlo."""

    review: Review | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.review is not None

    @property
    def approved(self) -> bool:
        return self.review is not None and self.review.approved


class Judge:
    def __init__(self, llm: LLM):
        self._llm = llm

    def review(self, script: Script, dossier: Dossier) -> ReviewReport:
        anticipado = review_measured_only(
            script, dossier,
            model=getattr(self._llm, "model", ""),
            provider=getattr(self._llm, "provider", ""),
        )
        if anticipado is not None:
            # Reprobado en la medida: ninguna llamada, coste cero.
            return ReviewReport(review=anticipado)

        uso = Usage()
        latencia = 0.0
        ultimo = ""

        for _ in range(MAX_INTENTOS):
            respuesta = self._llm.complete(
                build_prompt(script, dossier),
                system=SISTEMA,
                schema=SCHEMA_INFORME,
                temperature=0.0,  # el informe necesita ser reproducible para que el M5 compare
                max_output_tokens=1536,
            )
            uso = uso + respuesta.usage
            latencia = round(latencia + respuesta.latency_s, 3)
            try:
                juzgados = _notas_juzgadas(respuesta)
            except LLMError as exc:
                # Respuesta malformada y truncamiento son intermitentes: el guion
                # ya escrito no puede perderse por un JSON que abrio y no cerro.
                # Una segunda oportunidad cuesta menos que rehacer el guion
                # entero en la ejecucion siguiente.
                ultimo = str(exc)
                continue

            return ReviewReport(
                review=self._montar(script, _notas_medidas(script) + juzgados),
                usage=uso,
                latency_s=latencia,
            )

        raise LLMError(
            f"informe invalido en {MAX_INTENTOS} intentos ({ultimo}); "
            f"gastados {uso.total_tokens} tokens"
        )

    def _montar(self, script: Script, notas: list[CriterionScore]) -> Review:
        return Review(
            topic=script.topic,
            scores=notas,
            reviewed_at=datetime.now(UTC),
            model=getattr(self._llm, "model", ""),
            provider=getattr(self._llm, "provider", ""),
        )


def review_measured_only(
    script: Script, dossier: Dossier, *, model: str = "", provider: str = ""
) -> Review | None:
    """Informe completo sin consultar al modelo, cuando la medida ya reprueba.

    Devuelve None cuando el guion pasa los criterios de requisito -- ahi el
    informe depende de lectura y no hay forma de producirlo sin modelo.

    Es funcion libre, y no metodo, para que quien llama pueda descubrir que el
    guion ya reprobo **antes** de construir un adaptador y exigir clave de API.
    Es lo que permite reprobar la fixture adversarial del M3 sin ninguna cuenta
    en proveedor ninguno.
    """
    medidas = measured_scores(script, dossier)
    bloqueo = [s for s in medidas if s.score < VETO_MINIMO.get(s.criterion, 1)]
    if not bloqueo:
        return None
    return Review(
        topic=script.topic,
        scores=medidas + _no_evaluados(bloqueo),
        reviewed_at=datetime.now(UTC),
        model=model,
        provider=provider,
    )


def _notas_juzgadas(respuesta: Completion) -> list[CriterionScore]:
    cuerpo = parse_json_object(respuesta.text)
    notas: list[CriterionScore] = []
    for criterio in JULGADOS:
        crudo = cuerpo.get(criterio.value)
        if not isinstance(crudo, dict):
            raise LLMError(f"informe sin el criterio {criterio.value!r}")

        score = crudo.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 2:
            # Nota fuera de la escala no se redondea hacia dentro: la rubrica es
            # de 0 a 2, y aceptar 5 seria dejar al modelo redefinir el corte.
            raise LLMError(
                f"criterio {criterio.value!r} vino con nota invalida: {score!r}"
            )
        motivo = " ".join(str(crudo.get("reason") or "").split())
        # Motivo demasiado corto ("ok") rebasaba el min_length del CriterionScore
        # como ValidationError -- fuera del LLMError que el lazo trata -- y
        # tiraba abajo el slot entero. Hallado en el test del piloto (20/09).
        if len(motivo) < 3:
            motivo = (f"el modelo no justifico la nota ({motivo})" if motivo
                      else "el modelo no justifico la nota")
        notas.append(CriterionScore(criterion=criterio, score=score, reason=motivo))
    return notas


def measured_scores(script: Script, dossier: Dossier) -> list[CriterionScore]:
    """Los criterios de requisito, medidos sin consultar al modelo.

    Devuelve duracion y politica siempre, y fuente **solo cuando el anclaje
    numerico falla** -- porque solo en ese caso la medida decide el criterio
    sola. Cuando los numeros cuadran, fuente sigue siendo lectura y sale del
    informe del modelo.
    """
    notas = _notas_medidas(script)
    fuera = missing_numbers(script.narration, _texto_del_dossier(dossier))
    if fuera:
        notas.append(CriterionScore(
            criterion=Criterion.fuente,
            score=0,
            reason=f"numero citado sin respaldo en el dossier: {', '.join(fuera[:4])}",
            measured=True,
        ))
    return notas


def _no_evaluados(bloqueo: list[CriterionScore]) -> list[CriterionScore]:
    """Rellena la rubrica con los criterios que no fueron juzgados, y dice por que.

    La rubrica sigue completa -- informe con criterio faltante no valida -- pero
    cero aqui significa "no se", y `evaluated=False` lo marca para que la nota
    no vuelva al guionista como si fuera critica al texto.
    """
    motivo = (
        "no evaluado: el guion reprobo antes en "
        + ", ".join(sorted(s.criterion.value for s in bloqueo))
        + " (medido), y el informe del modelo no cambiaria el resultado"
    )
    ya_medidos = {s.criterion for s in bloqueo}
    return [
        CriterionScore(criterion=c, score=0, reason=motivo, evaluated=False)
        for c in JULGADOS
        if c not in ya_medidos
    ]


def _notas_medidas(script: Script) -> list[CriterionScore]:
    """Duracion y politica: medida, nunca lectura."""
    duracion = script.estimated_duration_s
    franja = ((SHORT_MIN_DURATION_S, SHORT_MAX_DURATION_S) if script.format == "short"
              else (MIN_DURATION_S, MAX_DURATION_S))
    en_franja = franja[0] <= duracion <= franja[1]
    # No existe medio punto para duracion: o el video esta en la franja del
    # formato, o no lo esta.
    nota_duracion = CriterionScore(
        criterion=Criterion.duracion,
        score=2 if en_franja else 0,
        reason=(
            f"{script.word_count} palabras, ~{duracion:.0f}s estimados "
            f"(franja exigida: {franja[0]}-{franja[1]}s)"
            + ("" if en_franja else "; fuera de la franja del formato")
        ),
        measured=True,
    )

    veredicto = policy.check(script.narration)
    nota_politica = CriterionScore(
        criterion=Criterion.politica,
        score=2 if veredicto.allowed else 0,
        reason=(
            "ningun termino de la lista de politica en la narracion"
            if veredicto.allowed
            else f"politica/{veredicto.rule}: '{veredicto.matched}' — {veredicto.reason}"
        ),
        measured=True,
    )
    return [nota_duracion, nota_politica]


def _texto_del_dossier(dossier: Dossier) -> str:
    return "\n".join(f"{f.claim}\n{f.quote}" for f in dossier.facts)


def build_prompt(script: Script, dossier: Dossier) -> str:
    """Compone el prompt del informe. Funcion libre para que el test inspeccione el texto."""
    hechos = "\n".join(
        f"[{i}] {f.claim} (fuente: {f.source_name})" for i, f in enumerate(dossier.facts)
    )
    rubrica = "\n".join(f"- {c.value}: {DESCRIPCIONES[c]}" for c in JULGADOS)
    lazo = (
        "\nCIERRE EN BUCLE: el video corto vive del replay automatico; el "
        "criterio cta vale 2 cuando el cierre reconecta con la pregunta del "
        "hook, y 0 cuando es CTA generico."
        if script.format == "short" else ""
    )
    tipo = _expectativa(script.pillar)
    return (
        f"TEMA: {script.topic}\n\n"
        f"DOSSIER DISPONIBLE PARA EL GUIONISTA:\n{hechos}\n\n"
        "GUION A EVALUAR\n"
        f"HOOK: {script.hook}\n\n"
        f"CUERPO: {script.body}\n\n"
        f"CIERRE: {script.closing}\n\n"
        "RUBRICA (nota de 0 a 2 en cada criterio)\n"
        f"{rubrica}{lazo}{tipo}\n\n"
        "Devuelve un objeto json con un campo por criterio, cada uno con 'reason' "
        "(una frase diciendo lo que hay que cambiar, en castellano) y 'score' (0, 1 o 2). "
        "Escribe la razon antes de la nota. No evalues duracion ni politica: "
        "esos dos se miden fuera de tu informe."
    )


def _expectativa(pillar: str) -> str:
    """La formula del tipo de contenido, para que el juez exija lo que el guionista recibio.

    Sin esto el juez juzgaria un tutorial con la regla de noticia: el punto de
    vista de un tutorial es "el error comun que lo anula todo", no una opinion.
    """
    if not pillar:
        return ""
    from agent.brand.brand import load

    p = load().pillars.get(pillar)
    if p is None:
        return ""
    return (f"\nTIPO {p.tag}: gancho esperado = {p.hook_formula}; batidas = "
            + " -> ".join(p.beats) + ". Usa esto para leer hook y punto_de_vista.")
