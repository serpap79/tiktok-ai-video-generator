"""Contratos entre estagios do pipeline.

Cada estagio recebe e devolve um destes modelos. Sao a fronteira que permite
testar um estagio sem levantar os outros, e sao o que fica gravado na memoria.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

# Ritmo de fala da narracao pt-BR. Usado apenas para estimativa antes do TTS;
# a duracao real vem do renderizador. Medido em 20/09/2026 nas vozes do
# renderizador proprio (Francisca -4%, Antonio +4%): 2,57 palavras/s nas duas.
WORDS_PER_SECOND = 2.57

# Faixa exigida pelo Creator Rewards: video abaixo de 60s nao e elegivel a
# monetizacao, e acima de ~90s a retencao cai sem ganho de receita.
MIN_DURATION_S = 60
MAX_DURATION_S = 90

# Faixa do curto. Mora aqui, junto com a do longo, porque estava escrita em
# DOIS lugares -- a faixa de palavras no `writer` e a de segundos no `judge`
# (`(10, 20)`, cravado). Quando a grade de 20/09/2026 pediu curto de 25s, so
# uma das duas foi atualizada e o juiz passou a reprovar todo curto que o
# roteirista aprovava, com a mensagem generica "nenhum formato aprovado pelo
# juiz". Duas constantes para o mesmo fato sempre divergem; agora e uma.
SHORT_MIN_DURATION_S = 22
SHORT_MAX_DURATION_S = 28


class Fact(BaseModel):
    """Uma afirmacao factual e a fonte que a sustenta.

    Nao existe Fact sem URL: e o que separa conteudo original de alucinacao, e
    e o que o juiz verifica no criterio 2 da rubrica.
    """

    claim: str = Field(min_length=10)
    source_url: HttpUrl
    source_name: str = Field(min_length=2)
    # Passagem literal da fonte que sustenta a afirmacao. Opcional porque o
    # roteiro de referencia do M0 foi escrito a mao, sem ela; o pesquisador (M3)
    # sempre preenche, e o portao que confere se a passagem existe de fato na
    # pagina e o que torna a citacao verificavel sem nova requisicao.
    quote: str = ""


class Dossier(BaseModel):
    """Resultado do estagio de pesquisa: o que sabemos e de onde."""

    topic: str
    facts: list[Fact]
    collected_at: datetime

    @field_validator("facts")
    @classmethod
    def _pelo_menos_um_fato(cls, v: list[Fact]) -> list[Fact]:
        if not v:
            raise ValueError("dossie sem fato nao autoriza roteiro")
        return v

    @property
    def source_urls(self) -> set[str]:
        return {str(f.source_url) for f in self.facts}


class Script(BaseModel):
    """Roteiro pronto para producao.

    `search_terms` vai direto para o parametro `query` do Pexels, sem passar por
    traducao: precisa estar **em ingles** e em **ordem cronologica** casando com
    a narracao, porque o material do primeiro termo abre o video.
    """

    topic: str = Field(min_length=3)
    hook: str = Field(min_length=10, description="primeiros ~1,5s; abre lacuna de informacao")
    body: str = Field(min_length=50)
    closing: str = Field(min_length=10)
    search_terms: list[str] = Field(min_length=2, max_length=12)
    facts: list[Fact] = Field(default_factory=list)
    # long = 60-90s (monetiza), short = ~15s (alcance, nao monetiza).
    format: str = "long"
    # B-roll do ASSUNTO (objeto/lugar filmavel, em ingles): o que correlaciona
    # a imagem com a fala. `search_terms` e a assinatura do canal; `broll` e
    # a coisa de que o video fala. Vazio em roteiro antigo -- continua valido.
    broll: list[str] = Field(default_factory=list, max_length=3)
    # Tipo de conteudo (pilar da marca) com que foi escrito: news, fato,
    # analise, tutorial, futuro, vs, historia. Vazio em roteiro antigo.
    pillar: str = ""
    # Legenda do post (guia da marca: gancho escrito SEM repetir o audio + uma
    # frase de contexto). Vazio em roteiro antigo: a legenda cai no hook.
    caption: str = ""

    @field_validator("broll")
    @classmethod
    def _broll_em_ascii(cls, v: list[str]) -> list[str]:
        limpos = [" ".join(t.split()) for t in v if t and t.strip()]
        for termo in limpos:
            if not termo.isascii():
                raise ValueError(f"broll {termo!r} nao e ASCII; o Pexels espera ingles")
        return limpos

    @field_validator("search_terms")
    @classmethod
    def _termos_em_ascii(cls, v: list[str]) -> list[str]:
        # Heuristica deliberadamente simples: acento em termo de busca quase
        # sempre significa que o modelo respondeu em pt-BR, e o Pexels devolve
        # resultado ruim ou vazio. Falhar aqui e mais barato que renderizar
        # um video com material errado.
        for termo in v:
            if not termo.isascii():
                raise ValueError(
                    f"termo de busca {termo!r} nao e ASCII; o Pexels espera ingles"
                )
            if not termo.strip():
                raise ValueError("termo de busca vazio")
        return v

    @property
    def narration(self) -> str:
        """Texto que o TTS vai falar, na ordem em que sera falado."""
        return "\n\n".join(p.strip() for p in (self.hook, self.body, self.closing))

    @property
    def word_count(self) -> int:
        return len(self.narration.split())

    @property
    def estimated_duration_s(self) -> float:
        """Estimativa pre-TTS. A duracao que vale e a do MP4 renderizado."""
        return self.word_count / WORDS_PER_SECOND

    @property
    def unsourced(self) -> list[str]:
        """Nao implementado aqui de proposito.

        Casar afirmacao com fonte exige julgamento semantico, nao string match:
        e trabalho do juiz (M3), com o dossie em maos. Este modelo so carrega
        os fatos para que o juiz possa fazer isso.
        """
        raise NotImplementedError("verificacao de fonte e responsabilidade do juiz (M3)")


class Criterion(StrEnum):
    """Os critérios das rubricas do juiz.

    Sao StrEnum e nao string livre porque a rubrica e um contrato: o eval do M5
    compara provedores criterio a criterio, e nota gravada com o nome do
    criterio escrito de duas formas nao se agrega.

    `fluxo` e so do carrossel (fio narrativo entre slides); o video usa os
    outros sete.
    """

    hook = "hook"
    fonte = "fonte"
    duracao = "duracao"
    ponto_de_vista = "ponto_de_vista"
    politica = "politica"
    pt_br = "pt_br"
    cta = "cta"
    fluxo = "fluxo"


# Corte da rubrica do video: 7 critérios, 0 a 2 cada. Literal, nao derivado
# do tamanho do enum: `fluxo` e criterio de carrossel e nao entra aqui.
RUBRIC_CUTOFF = 11
RUBRIC_MAX = 14

# Critérios que reprovam por exigência, e não por qualidade -- nota alta nos
# outros nao compra aprovacao aqui. Fonte e duracao sao requisito do Creator
# Rewards; politica e risco para o canal inteiro.
VETO_MINIMO: dict[Criterion, int] = {
    Criterion.fonte: 2,
    Criterion.duracao: 2,
    Criterion.politica: 2,
}


class CriterionScore(BaseModel):
    """A nota de um critério e a razão dela.

    `reason` e obrigatorio inclusive no 2. Nota sem justificativa nao da para
    auditar nem para devolver ao roteirista como correcao, e a revisao vira
    "tente de novo".
    """

    criterion: Criterion
    score: int = Field(ge=0, le=2)
    reason: str = Field(min_length=3)
    # True quando a nota saiu de medicao nossa, nao do julgamento do modelo.
    measured: bool = False
    # False quando o parecer foi interrompido antes deste criterio: o roteiro ja
    # havia reprovado num criterio medido, e pagar o parecer do modelo seria cota
    # gasta para confirmar uma reprovacao ja decidida. Zero aqui significa "nao
    # sei", nao "ruim" -- e a distincao importa: nota nao avaliada nao volta ao
    # roteirista como correcao.
    evaluated: bool = True


class Review(BaseModel):
    """O parecer do juiz sobre um roteiro."""

    topic: str
    scores: list[CriterionScore]
    reviewed_at: datetime
    model: str = ""
    provider: str = ""

    @model_validator(mode="after")
    def _rubrica_completa(self) -> Review:
        vistos = [s.criterion for s in self.scores]
        if len(vistos) != len(set(vistos)):
            raise ValueError("rubrica com criterio repetido")
        # A rubrica do video tem 7 criterios; `fluxo` e so do carrossel
        # (nao importar JULGADOS do juiz aqui: models nao depende de judge).
        faltando = set(Criterion) - {Criterion.fluxo} - set(vistos)
        if faltando:
            raise ValueError(
                "parecer incompleto, falta: "
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
        """Critérios de exigência que ficaram abaixo do mínimo.

        So os que estao em VETO_MINIMO. Criterio de qualidade zerado tambem
        reprova, mas por outra regra (`zeroed`) -- chamar os dois de veto faria a
        mensagem dizer que hook e requisito de monetizacao, o que e falso.
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
        """True quando a medida reprovou antes de o modelo ser consultado.

        `total` de um parecer interrompido nao e comparavel com o de um parecer
        completo -- o eval do M5 precisa filtrar por isto antes de agregar nota.
        """
        return any(not s.evaluated for s in self.scores)

    @property
    def approved(self) -> bool:
        """Corte em 11/14, nenhum critério zerado, e nenhum veto violado.

        As tres condicoes existem porque soma sozinha permite compensacao
        errada: um roteiro que e resumo de noticia (0 em ponto de vista) chegaria
        a 12 de 14 com o resto perfeito e passaria -- sendo exatamente o "AI
        slop" que desmonetiza o canal.
        """
        return (
            self.total >= RUBRIC_CUTOFF
            and not self.zeroed
            and not self.vetoed
        )

    @property
    def revision_notes(self) -> list[str]:
        """O que devolver ao roteirista, na ordem em que custa mais caro.

        Veto primeiro: nao adianta melhorar o hook de um roteiro que cita
        numero sem fonte.
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
# Teto da marca ("frase curta"): 12 palavras na tela. A pesquisa tolera 15,
# mas a identidade manda -- o prompt mira 10 para caber com folga.
CAROUSEL_MAX_WORDS_PER_SLIDE = 12


class Slide(BaseModel):
    """Um slide do carrossel: promessa curta + visual do pilar."""

    n: int = Field(ge=1, le=CAROUSEL_SLIDES)
    headline: str = Field(min_length=3)
    text: str = Field(min_length=3)
    # Tag verbatim do vocabulario visual (writer/visuals.py), um pilar so.
    visual: str = Field(min_length=3)

    @property
    def word_count(self) -> int:
        return len(f"{self.headline} {self.text}".split())


class Carousel(BaseModel):
    """Roteiro de carrossel: 5 slides 1080x1920 + legenda que puxa comentario."""

    topic: str = Field(min_length=3)
    slides: list[Slide] = Field(min_length=CAROUSEL_SLIDES,
                                max_length=CAROUSEL_SLIDES)
    caption: str = Field(min_length=10)
    facts: list[Fact] = Field(default_factory=list)
    format: str = "carousel"
    # Foto do assunto para a capa (slide 1) e o miolo (slide 3), em ingles.
    broll: list[str] = Field(default_factory=list, max_length=3)
    pillar: str = ""

    @model_validator(mode="after")
    def _ordem(self) -> Carousel:
        if [s.n for s in self.slides] != [1, 2, 3, 4, 5]:
            raise ValueError("slides fora de ordem 1-5")
        return self


CAROUSEL_CUTOFF = 8
CAROUSEL_MAX = 10


class CarouselReview(BaseModel):
    """Parecer do carrossel: politica medido, 4 critérios lidos.

    Corte em 8/10 (folga de 2 pontos, como no 6/8 anterior), nenhum critério
    zerado, politica sem veto. O resto do mecanico (5 slides, 15 palavras,
    save no 5, numero no 1) ja passou no roteirista -- mandar isso ao juiz
    gastaria cota para conferir `len().
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
        """O que devolver ao roteirista de carrossel, mais barato primeiro."""
        return [f"[{s.criterion.value} {s.score}/2] {s.reason}"
                for s in sorted(self.scores, key=lambda s: s.score)
                if s.score < 2 and s.evaluated]


class RenderResult(BaseModel):
    """O que o renderizador devolve. `duration_s` e medida, nao estimada."""

    state: RenderState
    video_path: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    # Medido com ffprobe. Um MP4 sem trilha de narracao passa em qualquer
    # checagem de dimensao e duracao, e nao serve para nada.
    has_audio: bool = False
    task_id: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _coerencia(self) -> RenderResult:
        if self.state is RenderState.complete and not self.video_path:
            raise ValueError("render completo sem video_path")
        if self.state is RenderState.failed and not self.error:
            raise ValueError("render falhou sem mensagem de erro")
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
    """Materia ja associada a um termo pela propria fonte.

    O Google Trends RSS entrega isso de graca junto de cada tema, o que adianta
    parte do trabalho do pesquisador (M3) sem custar uma requisicao a mais.
    """

    title: str = Field(min_length=3)
    url: HttpUrl
    source_name: str = ""


class Signal(BaseModel):
    """Um termo em alta, como uma fonte o reporta.

    `volume` esta sempre na unidade nativa da fonte -- pontos do HN, buscas
    estimadas do Trends, pageviews da Wikipedia. Nao sao comparaveis entre si e
    o radar nao tenta normalizar: converter escalas diferentes numa nota unica e
    julgamento, e julgamento e trabalho do curador (M2). O radar so coleta e
    mede.

    `velocity` e a unica grandeza comparavel em forma, porque e sempre a mesma
    derivada: unidade por hora. Fica `None` quando a fonte nao permite calcula-la
    -- e `None` significa "desconhecido", nunca zero.
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
    def _termo_normalizado(cls, v: str) -> str:
        return " ".join(v.split()).strip()

    @property
    def key(self) -> str:
        """Chave estavel para casar a mesma historia entre coletas."""
        return f"{self.source}:{self.term.casefold()}"

    @property
    def has_velocity(self) -> bool:
        return self.velocity is not None


class Verdict(StrEnum):
    """Desfecho de um candidato no curador.

    Rejeicao tem tipo, nao so um booleano: "reprovou na politica" e "perdeu o
    ranking" pedem acoes opostas. O primeiro nunca deve voltar; o segundo pode
    ser o escolhido amanha.
    """

    selected = "selected"
    rejected_policy = "rejected_policy"
    rejected_niche = "rejected_niche"
    rejected_duplicate = "rejected_duplicate"
    not_selected = "not_selected"


class Decision(BaseModel):
    """O que o curador decidiu sobre um sinal, e por que.

    `reason` e obrigatorio inclusive na aprovacao. Decisao sem justificativa
    gravada nao da para auditar depois, e a auditoria e o que permite corrigir o
    score em vez de chutar.
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
    def _duplicata_aponta_o_original(self) -> Decision:
        if self.verdict is Verdict.rejected_duplicate and not self.duplicate_of:
            raise ValueError("rejeicao por duplicata precisa apontar o tema original")
        return self

    @property
    def approved(self) -> bool:
        return self.verdict is Verdict.selected


class PublishState(StrEnum):
    """Desfecho da subida para a inbox.

    `uploaded` nao significa "postado": significa que o video chegou a inbox e
    a conclusao (legenda, rotulo AIGC, publicar) acontece no app, pela pessoa
    criadora. Automatizar alem disso seria Direct Post, que exige auditoria.
    """

    uploaded = "uploaded"
    failed = "failed"


class PublishResult(BaseModel):
    """O que o publicador devolve. `publish_id` rastreia o post na API."""

    state: PublishState
    publish_id: str | None = None
    video_path: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _coerencia(self) -> PublishResult:
        if self.state is PublishState.uploaded and not self.publish_id:
            raise ValueError("upload para a inbox sem publish_id")
        if self.state is PublishState.failed and not self.error:
            raise ValueError("publicacao falhou sem mensagem de erro")
        return self
