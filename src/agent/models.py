"""Modelos: contratos entre etapas del pipeline.

Cada etapa recibe y devuelve uno de estos modelos. Son la frontera que permite
probar una etapa sin levantar las otras, y son lo que queda grabado en memoria.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

# Ritmo de habla de la narracion es-ES. Usado solo para estimacion antes del
# TTS; la duracion real viene del renderizador. Referencia: las voces de
# edge-tts es-ES rondan 2,5-2,7 palabras/s; se recalibra con la primera medida
# real del renderizador propio.
WORDS_PER_SECOND = 2.57

# Franja exigida por el Creator Rewards: un video por debajo de 60s no es
# elegible a monetizacion, y por encima de ~90s la retencion cae sin ganancia
# de ingreso.
MIN_DURATION_S = 60
MAX_DURATION_S = 90

# Franja del corto. Vive aqui, junto con la del largo, porque estaba escrita en
# DOS sitios -- la franja de palabras en el `writer` y la de segundos en el
# `judge` (`(10, 20)`, clavada). Cuando la parrilla de 20/09/2026 pidio un
# corto de 25s, solo una de las dos se actualizo y el juez paso a reprobar todo
# corto que el guionista aprobaba, con el mensaje generico "ningun formato
# aprobado por el juez". Dos constantes para el mismo hecho siempre divergen;
# ahora es una.
SHORT_MIN_DURATION_S = 22
SHORT_MAX_DURATION_S = 28


class Fact(BaseModel):
    """Una afirmacion factual y la fuente que la sostiene.

    No existe Fact sin URL: es lo que separa contenido original de alucinacion,
    y es lo que el juez verifica en el criterio 2 de la rubrica.
    """

    claim: str = Field(min_length=10)
    source_url: HttpUrl
    source_name: str = Field(min_length=2)
    # Pasaje literal de la fuente que sostiene la afirmacion. Opcional porque el
    # guion de referencia del M0 se escribio a mano, sin el; el investigador
    # (M3) siempre lo rellena, y la puerta que comprueba si el pasaje existe de
    # verdad en la pagina es lo que hace la cita verificable sin nueva peticion.
    quote: str = ""


class Dossier(BaseModel):
    """Resultado de la etapa de investigacion: lo que sabemos y de donde."""

    topic: str
    facts: list[Fact]
    collected_at: datetime

    @field_validator("facts")
    @classmethod
    def _al_menos_un_hecho(cls, v: list[Fact]) -> list[Fact]:
        if not v:
            raise ValueError("dossier sin hecho no autoriza guion")
        return v

    @property
    def source_urls(self) -> set[str]:
        return {str(f.source_url) for f in self.facts}


class Script(BaseModel):
    """Guion listo para produccion.

    `search_terms` va directo al parametro `query` de Pexels, sin pasar por
    traduccion: tiene que estar **en ingles** y en **orden cronologico**
    casando con la narracion, porque el material del primer termino abre el
    video.
    """

    topic: str = Field(min_length=3)
    hook: str = Field(min_length=10, description="primeros ~1,5s; abre hueco de informacion")
    body: str = Field(min_length=50)
    closing: str = Field(min_length=10)
    search_terms: list[str] = Field(min_length=2, max_length=12)
    facts: list[Fact] = Field(default_factory=list)
    # long = 60-90s (monetiza), short = ~15s (alcance, no monetiza).
    format: str = "long"
    # B-roll del ASUNTO (objeto/lugar filmable, en ingles): lo que correlaciona
    # la imagen con el habla. `search_terms` es la firma del canal; `broll` es
    # la cosa de la que el video habla. Vacio en guion antiguo -- sigue valido.
    broll: list[str] = Field(default_factory=list, max_length=3)
    # Tipo de contenido (pilar de la marca) con el que se escribio: news, dato,
    # analisis, tutorial, futuro, vs, historia. Vacio en guion antiguo.
    pillar: str = ""
    # Leyenda del post (guia de marca: gancho escrito SIN repetir el audio +
    # una frase de contexto). Vacio en guion antiguo: la leyenda cae en el hook.
    caption: str = ""

    @field_validator("broll")
    @classmethod
    def _broll_en_ascii(cls, v: list[str]) -> list[str]:
        limpios = [" ".join(t.split()) for t in v if t and t.strip()]
        for termino in limpios:
            if not termino.isascii():
                raise ValueError(f"broll {termino!r} no es ASCII; Pexels espera ingles")
        return limpios

    @field_validator("search_terms")
    @classmethod
    def _terminos_en_ascii(cls, v: list[str]) -> list[str]:
        # Heuristica deliberadamente simple: un acento en un termino de busqueda
        # casi siempre significa que el modelo respondio en castellano, y Pexels
        # devuelve resultado malo o vacio. Fallar aqui es mas barato que
        # renderizar un video con material equivocado.
        for termino in v:
            if not termino.isascii():
                raise ValueError(
                    f"termino de busqueda {termino!r} no es ASCII; Pexels espera ingles"
                )
            if not termino.strip():
                raise ValueError("termino de busqueda vacio")
        return v

    @property
    def narration(self) -> str:
        """Texto que el TTS va a hablar, en el orden en que se hablara."""
        return "\n\n".join(p.strip() for p in (self.hook, self.body, self.closing))

    @property
    def word_count(self) -> int:
        return len(self.narration.split())

    @property
    def estimated_duration_s(self) -> float:
        """Estimacion pre-TTS. La duracion que vale es la del MP4 renderizado."""
        return self.word_count / WORDS_PER_SECOND

    @property
    def unsourced(self) -> list[str]:
        """No implementado aqui a proposito.

        Casar afirmacion con fuente exige juicio semantico, no string match:
        es trabajo del juez (M3), con el dossier en mano. Este modelo solo
        lleva los hechos para que el juez pueda hacerlo.
        """
        raise NotImplementedError("verificar fuente es responsabilidad del juez (M3)")


class Criterion(StrEnum):
    """Los criterios de las rubricas del juez.

    Son StrEnum y no cadena libre porque la rubrica es un contrato: el eval del
    M5 compara proveedores criterio a criterio, y una nota grabada con el nombre
    del criterio escrito de dos formas no se agrega.

    `flujo` es solo del carrusel (hilo narrativo entre slides); el video usa
    los otros siete.
    """

    hook = "hook"
    fuente = "fuente"
    duracion = "duracion"
    punto_de_vista = "punto_de_vista"
    politica = "politica"
    idioma = "idioma"
    cta = "cta"
    flujo = "flujo"


# Corte de la rubrica del video: 7 criterios, 0 a 2 cada uno. Literal, no
# derivado del tamano del enum: `flujo` es criterio de carrusel y no entra aqui.
RUBRIC_CUTOFF = 11
RUBRIC_MAX = 14

# Criterios que reprueban por exigencia, y no por calidad -- nota alta en los
# otros no compra aprobacion aqui. Fuente y duracion son requisito del Creator
# Rewards; politica es riesgo para el canal entero.
VETO_MINIMO: dict[Criterion, int] = {
    Criterion.fuente: 2,
    Criterion.duracion: 2,
    Criterion.politica: 2,
}


class CriterionScore(BaseModel):
    """La nota de un criterio y el motivo de ella.

    `reason` es obligatorio incluso en el 2. Nota sin justificacion no se puede
    auditar ni devolver al guionista como correccion, y la revision se vuelve
    "intentalo de nuevo".
    """

    criterion: Criterion
    score: int = Field(ge=0, le=2)
    reason: str = Field(min_length=3)
    # True cuando la nota salio de medicion nuestra, no del juicio del modelo.
    measured: bool = False
    # False cuando el informe se interrumpio antes de este criterio: el guion ya
    # habia reprobado en un criterio medido, y pagar el informe del modelo seria
    # cuota gastada para confirmar un reprobo ya decidido. Cero aqui significa
    # "no se", no "malo" -- y la distincion importa: nota no evaluada no vuelve
    # al guionista como correccion.
    evaluated: bool = True


class Review(BaseModel):
    """El informe del juez sobre un guion."""

    topic: str
    scores: list[CriterionScore]
    reviewed_at: datetime
    model: str = ""
    provider: str = ""

    @model_validator(mode="after")
    def _rubrica_completa(self) -> Review:
        vistos = [s.criterion for s in self.scores]
        if len(vistos) != len(set(vistos)):
            raise ValueError("rubrica con criterio repetido")
        # La rubrica del video tiene 7 criterios; `flujo` es solo del carrusel
        # (no importar JULGADOS del juez aqui: models no depende de judge).
        faltando = set(Criterion) - {Criterion.flujo} - set(vistos)
        if faltando:
            raise ValueError(
                "informe incompleto, falta: "
                + ", ".join(sorted(c.value for c in faltando))
            )
        return self

    @property
    def total(self) -> int:
        return sum(s.score for s in self.scores)

    @property
    def by_criterion(self) -> dict[Criterion, CriterionScore]:
        return {s.criterion: s for s in self.scores}

    @property
    def vetoed(self) -> list[CriterionScore]:
        """Criterios de exigencia que quedaron por debajo del minimo.

        Solo los que estan en VETO_MINIMO. Criterio de calidad a cero tambien
        reprueba, pero por otra regla (`zeroed`) -- llamar a los dos veto haria
        que el mensaje dijera que hook es requisito de monetizacion, lo cual es
        falso.
        """
        return [
            s for s in self.scores
            if s.criterion in VETO_MINIMO and s.score < VETO_MINIMO[s.criterion]
        ]

    @property
    def zeroed(self) -> list[CriterionScore]:
        return [s for s in self.scores if s.score == 0 and s.evaluated]

    @property
    def short_circuited(self) -> bool:
        """True cuando la medicion reprobo antes de consultar al modelo.

        `total` de un informe interrumpido no es comparable con el de un informe
        completo -- el eval del M5 necesita filtrar por esto antes de agregar
        nota.
        """
        return any(not s.evaluated for s in self.scores)

    @property
    def approved(self) -> bool:
        """Corte en 11/14, ningun criterio a cero y ningun veto violado.

        Las tres condiciones existen porque la suma sola permite compensacion
        equivocada: un guion que es resumen de noticia (0 en punto de vista)
        llegaria a 12 de 14 con el resto perfecto y pasaria -- siendo
        exactamente el "AI slop" que desmonetiza el canal.
        """
        return (
            self.total >= RUBRIC_CUTOFF
            and not self.zeroed
            and not self.vetoed
        )

    @property
    def revision_notes(self) -> list[str]:
        """Lo que se devuelve al guionista, en el orden en que cuesta mas caro.

        Veto primero: no sirve de nada mejorar el hook de un guion que cita
        numero sin fuente.
        """
        veto = {s.criterion for s in self.vetoed}
        ordenados = sorted(
            (s for s in self.scores if s.score < 2 and s.evaluated),
            key=lambda s: (s.criterion not in veto, s.score),
        )
        return [f"[{s.criterion.value} {s.score}/2] {s.reason}" for s in ordenados]


class RenderState(StrEnum):
    processing = "processing"
    complete = "complete"
    failed = "failed"


CAROUSEL_SLIDES = 5
# Techo de la marca ("frase corta"): 12 palabras en pantalla. La investigacion
# tolera 15, pero la identidad manda -- el prompt apunta a 10 para caber con
# margen.
CAROUSEL_MAX_WORDS_PER_SLIDE = 12


class Slide(BaseModel):
    """Un slide del carrusel: promesa corta + visual del pilar."""

    n: int = Field(ge=1, le=CAROUSEL_SLIDES)
    headline: str = Field(min_length=3)
    text: str = Field(min_length=3)
    # Tag verbatim del vocabulario visual (writer/visuals.py), un pilar solo.
    visual: str = Field(min_length=3)

    @property
    def word_count(self) -> int:
        return len(f"{self.headline} {self.text}".split())


class Carousel(BaseModel):
    """Guion de carrusel: 5 slides 1080x1920 + leyenda que tira de comentario."""

    topic: str = Field(min_length=3)
    slides: list[Slide] = Field(min_length=CAROUSEL_SLIDES,
                                max_length=CAROUSEL_SLIDES)
    caption: str = Field(min_length=10)
    facts: list[Fact] = Field(default_factory=list)
    format: str = "carousel"
    # Foto del asunto para la portada (slide 1) y el centro (slide 3), en ingles.
    broll: list[str] = Field(default_factory=list, max_length=3)
    pillar: str = ""

    @model_validator(mode="after")
    def _orden(self) -> Carousel:
        if [s.n for s in self.slides] != [1, 2, 3, 4, 5]:
            raise ValueError("slides fuera de orden 1-5")
        return self


CAROUSEL_CUTOFF = 8
CAROUSEL_MAX = 10


class CarouselReview(BaseModel):
    """Informe del carrusel: politica medido, 4 criterios leidos.

    Corte en 8/10 (margen de 2 puntos, como en el 6/8 anterior), ningun
    criterio a cero, politica sin veto. El resto del mecanico (5 slides, 15
    palabras, save en el 5, numero en el 1) ya paso en el guionista -- mandar
    eso al juez gastaria cuota para comprobar un `len()`.
    """

    topic: str
    scores: list[CriterionScore]
    reviewed_at: datetime
    model: str = ""
    provider: str = ""

    @property
    def total(self) -> int:
        return sum(s.score for s in self.scores)

    @property
    def by_criterion(self) -> dict[Criterion, CriterionScore]:
        return {s.criterion: s for s in self.scores}

    @property
    def zeroed(self) -> list[CriterionScore]:
        return [s for s in self.scores if s.score == 0 and s.evaluated]

    @property
    def approved(self) -> bool:
        pol = self.by_criterion.get(Criterion.politica)
        return (
            self.total >= CAROUSEL_CUTOFF
            and not self.zeroed
            and pol is not None and pol.score == 2
        )

    @property
    def short_circuited(self) -> bool:
        return any(not s.evaluated for s in self.scores)

    @property
    def revision_notes(self) -> list[str]:
        """Lo que se devuelve al guionista de carrusel, mas barato primero."""
        return [f"[{s.criterion.value} {s.score}/2] {s.reason}"
                for s in sorted(self.scores, key=lambda s: s.score)
                if s.score < 2 and s.evaluated]


class RenderResult(BaseModel):
    """Lo que devuelve el renderizador. `duration_s` es medida, no estimada."""

    state: RenderState
    video_path: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    # Medido con ffprobe. Un MP4 sin pista de narracion pasa cualquier
    # comprobacion de dimension y duracion, y no sirve para nada.
    has_audio: bool = False
    task_id: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _coherencia(self) -> RenderResult:
        if self.state is RenderState.complete and not self.video_path:
            raise ValueError("render completo sin video_path")
        if self.state is RenderState.failed and not self.error:
            raise ValueError("render fallo sin mensaje de error")
        return self

    @property
    def is_portrait_1080x1920(self) -> bool:
        return (self.width, self.height) == (1080, 1920)

    @property
    def duration_in_monetizable_range(self) -> bool:
        if self.duration_s is None:
            return False
        return MIN_DURATION_S <= self.duration_s <= MAX_DURATION_S


class NewsItem(BaseModel):
    """Noticia ya asociada a un termino por la propia fuente.

    El RSS de Google Trends lo entrega gratis junto a cada tema, lo que adelanta
    parte del trabajo del investigador (M3) sin costar una peticion mas.
    """

    title: str = Field(min_length=3)
    url: HttpUrl
    source_name: str = ""


class Signal(BaseModel):
    """Un termino en tendencia, tal como lo reporta una fuente.

    `volume` esta siempre en la unidad nativa de la fuente -- puntos del HN,
    busquedas estimadas de Trends, pageviews de Wikipedia. No son comparables
    entre si y el radar no intenta normalizar: convertir escalas diferentes en
    una nota unica es juicio, y el juicio es trabajo del curador (M2). El radar
    solo recolecta y mide.

    `velocity` es la unica magnitud comparable en forma, porque es siempre la
    misma derivada: unidad por hora. Queda `None` cuando la fuente no permite
    calcularla -- y `None` significa "desconocido", nunca cero.
    """

    term: str = Field(min_length=2)
    source: str = Field(min_length=2)
    volume: float = Field(ge=0)
    unit: str = Field(min_length=1)
    velocity: float | None = None
    seen_at: datetime
    url: HttpUrl | None = None
    news_items: list[NewsItem] = Field(default_factory=list)

    @field_validator("term")
    @classmethod
    def _termino_normalizado(cls, v: str) -> str:
        return " ".join(v.split()).strip()

    @property
    def key(self) -> str:
        """Clave estable para casar la misma historia entre recolectas."""
        return f"{self.source}:{self.term.casefold()}"

    @property
    def has_velocity(self) -> bool:
        return self.velocity is not None


class Verdict(StrEnum):
    """Desenlace de un candidato en el curador.

    El rechazo tiene tipo, no solo un booleano: "reprobo en politica" y "perdio
    el ranking" piden acciones opuestas. El primero nunca debe volver; el
    segundo puede ser el elegido manana.
    """

    selected = "selected"
    rejected_policy = "rejected_policy"
    rejected_niche = "rejected_niche"
    rejected_duplicate = "rejected_duplicate"
    not_selected = "not_selected"


class Decision(BaseModel):
    """Que decidio el curador sobre una senal, y por que.

    `reason` es obligatorio incluso en la aprobacion. Decision sin justificacion
    grabada no se puede auditar despues, y la auditoria es lo que permite
    corregir el score en vez de dar palos de ciego.
    """

    term: str = Field(min_length=2)
    source: str
    verdict: Verdict
    reason: str = Field(min_length=3)
    score: float = Field(ge=0, le=1)
    niche_fit: float = Field(ge=0, le=1)
    velocity: float | None = None
    volume: float = 0.0
    url: HttpUrl | None = None
    duplicate_of: str | None = None
    decided_at: datetime
    news_items: list[NewsItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _duplicado_apunta_al_original(self) -> Decision:
        if self.verdict is Verdict.rejected_duplicate and not self.duplicate_of:
            raise ValueError("el rechazo por duplicado debe apuntar al tema original")
        return self

    @property
    def approved(self) -> bool:
        return self.verdict is Verdict.selected


class PublishState(StrEnum):
    """Desenlace de la subida a la inbox.

    `uploaded` no significa "publicado": significa que el video llego a la inbox
    y la conclusion (leyenda, etiqueta AIGC, publicar) ocurre en la app, por la
    persona creadora. Automatizar mas alla seria Direct Post, que exige
    auditoria.
    """

    uploaded = "uploaded"
    failed = "failed"


class PublishResult(BaseModel):
    """Lo que devuelve el publicador. `publish_id` rastrea el post en la API."""

    state: PublishState
    publish_id: str | None = None
    video_path: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _coherencia(self) -> PublishResult:
        if self.state is PublishState.uploaded and not self.publish_id:
            raise ValueError("subida a la inbox sin publish_id")
        if self.state is PublishState.failed and not self.error:
            raise ValueError("publicacion fallo sin mensaje de error")
        return self
