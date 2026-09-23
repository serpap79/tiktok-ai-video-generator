"""Roteirista: de um dossie gravado para um Script pronto para o renderizador.

O estagio tem duas metades de natureza diferente, e misturar as duas e o erro
que este arquivo evita.

A primeira e **julgamento**: hook que abre lacuna, ponto de vista proprio,
portugues falado. Isso e trabalho do juiz (fatia 3), com rubrica, e nao da para
decidir por regra.

A segunda e **mecanica**: contar palavra, conferir se o termo de busca esta
em ASCII e saiu do vocabulario visual do canal (um pilar so), conferir se o
indice de fato existe no dossie. Isso nao precisa de juiz
nenhum, e gastar uma rodada de revisao do juiz com erro de contagem seria
desperdicio de cota. Por isso o roteirista tem seu proprio laco de correcao, com
o defeito medido devolvido ao modelo em texto, e so entrega ao juiz um roteiro
que ja passa no que e verificavel.

A faixa de duracao e requisito de monetizacao, nao gosto: video abaixo de 60s
nao e elegivel ao Creator Rewards. Ela e estimada aqui pelo ritmo de fala
(WORDS_PER_SECOND) e **medida de verdade** so depois do TTS, pelo renderizador --
e e a medida que manda.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent.brand.brand import pillar_brief, voice_brief
from agent.brand.checks import check_emoji_bordao, check_hook, check_numbers
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

# Faixa de palavras que corresponde a faixa de duracao exigida, com 2s de
# margem em cada ponta: a voz real varia alguns por cento do ritmo medido, e
# 59s nao monetiza.
MARGEM_S = 2
MIN_PALAVRAS = int((MIN_DURATION_S + MARGEM_S) * WORDS_PER_SECOND)
MAX_PALAVRAS = int((MAX_DURATION_S - MARGEM_S) * WORDS_PER_SECOND)

# short nao monetiza (25s << 60s): a funcao dele e alcance, nao receita.
# Subiu de 30-50 para 58-70 palavras em 20/09/2026, quando a grade passou a
# pedir curto de ~25s (era ~15s). A 2,57 palavras/s medidas, 58-70 palavras
# dao 23-27s. Final em loop, para o replay automatico.
MIN_PALAVRAS_CURTO = math.ceil(SHORT_MIN_DURATION_S * WORDS_PER_SECOND)
MAX_PALAVRAS_CURTO = math.floor(SHORT_MAX_DURATION_S * WORDS_PER_SECOND)
BANDS: dict[str, tuple[int, int]] = {
    "long": (MIN_PALAVRAS, MAX_PALAVRAS),
    "short": (MIN_PALAVRAS_CURTO, MAX_PALAVRAS_CURTO),
}
# Um clipe de b-roll cobre ~5s, entao o curto de 25s pede mais termos que o de
# 15s -- com 2 termos o mesmo clipe voltava tres vezes no mesmo video.
TERMS_PER_MODE: dict[str, tuple[int, int]] = {
    "long": (4, 8),
    "short": (3, 5),
}

# "[0]", "[1, 2]": o indice do fato echoado dentro do texto. Medido na primeira
# execucao real (18/09/2026): o modelo escreveu "...no seu projeto [0]." e
# "...em um projeto [0, 3]." -- e o TTS leria "zero" e "um" em voz alta.
_MARCADOR_DE_CITACAO = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")

# Abaixo disto o dossie nao sustenta 60 segundos de narracao. Medido: um dossie
# de 4 fatos tirados de UMA frase de changelog levou o roteirista a tres
# tentativas, todas entre 104 e 157 palavras, sem nunca alcancar as 150 -- porque
# nao havia assunto, e nao porque a instrucao estava ruim.
MIN_FATOS_PARA_ROTEIRO = 3

# Tentativas totais, contando a primeira. Duas correcoes bastam para defeito
# mecanico; se o modelo nao acerta a contagem em tres tentativas, o problema nao
# e a instrucao, e insistir so queima cota do free tier.
MAX_TENTATIVAS = 3

SISTEMA = (
    "Voce e roteirista de um canal brasileiro de tech, IA e ciencia. Escreve para "
    "ser ouvido, nao lido: frase curta, voz ativa, zero jargao nao explicado. "
    "Voce so afirma o que esta no dossie que recebe. Numero que nao esta no "
    "dossie nao entra no roteiro, nem como aproximacao."
)

SCHEMA_ROTEIRO: dict[str, Any] = {
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
    """Uma tentativa e o que ela violou. Vazio significa que ela foi aceita.

    `narration` guarda o texto reprovado. Sem ele, entender POR QUE um portao
    reprovou exige rodar de novo e pagar a cota outra vez -- foi o que aconteceu
    na primeira execucao real, com um portao acusando "numero 0, 1, 2" sem que
    houvesse como ver de onde os numeros vinham.
    """

    violations: list[str] = field(default_factory=list)
    word_count: int = 0
    narration: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


@dataclass
class WriteReport:
    """O roteiro, as tentativas que precisaram acontecer, e o custo de todas.

    As tentativas ficam gravadas porque elas dizem onde o prompt esta fraco: se
    toda execucao gasta duas rodadas para acertar a contagem de palavras, o
    defeito esta na instrucao, nao no modelo -- e isso so aparece se o intervalo
    for registrado em vez de descartado no sucesso.
    """

    topic: str
    script: Script | None = None
    attempts: list[Attempt] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    # Passada de humanizacao apos o aceite mecanico. Nao e tentativa: nao
    # reprova, so melhora -- ou mantem o original com motivo.
    humanized: bool = False
    humanize_notes: list[str] = field(default_factory=list)
    humanize_usage: Usage = field(default_factory=Usage)
    humanize_latency_s: float = 0.0
    # Motivo de nem ter tentado. Diferente de tentativa reprovada: aqui nenhuma
    # chamada foi feita, e o custo e zero.
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
        """Violacoes da ultima tentativa: o motivo de ter falhado."""
        return self.attempts[-1].violations if self.attempts else []


class Screenwriter:
    def __init__(self, llm: LLM, max_attempts: int = MAX_TENTATIVAS):
        self._llm = llm
        self._max_attempts = max_attempts

    def write(self, dossier: Dossier, notes: list[str] | None = None,
              mode: str = "long", polish: bool = True, pillar: str = "",
              previous: str = "") -> WriteReport:
        """Escreve o roteiro. `notes` sao as notas de revisao do juiz.

        Elas entram no mesmo canal das violacoes mecanicas -- o modelo recebe uma
        lista de defeitos a corrigir e nao precisa saber qual deles foi contado e
        qual foi julgado.
        """
        if mode not in BANDS:
            raise ValueError(f"modo desconhecido: {mode!r}; use long ou short")
        report = WriteReport(
            topic=dossier.topic,
            model=getattr(self._llm, "model", ""),
            provider=getattr(self._llm, "provider", ""),
        )

        report.refusal = thin_dossier_reason(dossier, mode)
        if report.refusal:
            # Medir antes de pagar, como o curador e o juiz fazem: dossie que nao
            # sustenta 60s de narracao nao vira roteiro por insistencia, e tentar
            # tres vezes so gastaria cota para chegar na mesma parede.
            return report

        correcao: list[str] = list(notes or [])
        # Texto da tentativa anterior: sem ele, "mantenha o que estava bom e
        # conserte o apontado" pedia ao modelo para manter algo que ele nao via
        # -- cada correcao era uma escrita do zero, e a contagem de palavras
        # oscilava em vez de convergir.
        anterior = previous

        for _ in range(self._max_attempts):
            try:
                resposta = self._llm.complete(
                    build_prompt(dossier, correcao, mode, pillar=pillar,
                                 previous=anterior if correcao else ""),
                    system=SISTEMA,
                    schema=SCHEMA_ROTEIRO,
                    temperature=0.6,
                    max_output_tokens=2048,
                )
            except LLMError:
                # Cota estourada ou filtro de conteudo: nao ha o que corrigir no
                # prompt, entao sobe para quem chamou. Defeito de forma na
                # resposta e tratado em _avaliar, como violacao corrigivel.
                raise

            # Com o roteador, quem respondeu so e conhecido depois da chamada.
            report.model, report.provider = resposta.model, resposta.provider
            tentativa, script = self._avaliar(resposta, dossier, mode, pillar)
            report.attempts.append(tentativa)
            if not tentativa.violations:
                report.script = script
                if polish and script is not None:
                    self._polir(report, script, dossier, mode)
                return report
            correcao = tentativa.violations
            anterior = tentativa.narration or anterior

        return report

    def _polir(self, report: WriteReport, script: Script,
               dossier: Dossier, mode: str) -> None:
        """Passada de humanizacao. Original intacto se a reescrita falhar."""
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

    # ------------------------------------------------------------------ avaliacao

    def _avaliar(self, resposta: Completion, dossier: Dossier,
                 mode: str = "long", pillar: str = "") -> tuple[Attempt, Script | None]:
        tentativa = Attempt(usage=resposta.usage, latency_s=resposta.latency_s)
        if resposta.truncated:
            tentativa.violations.append(
                "a resposta foi cortada por limite de tokens; escreva mais curto"
            )
            return tentativa, None

        try:
            corpo = parse_json_object(resposta.text)
        except LLMError as exc:
            # Defeito de forma, nao de provedor: o laco conserta isso, e gastar
            # uma rodada do juiz com JSON quebrado seria desperdicio de cota.
            tentativa.violations.append(f"a resposta nao veio como objeto JSON: {exc}")
            return tentativa, None

        usados, fora = _resolver_fatos(corpo.get("used_facts"), dossier.facts)

        try:
            script = Script(
                topic=dossier.topic,
                hook=_texto(corpo.get("hook")),
                body=_texto(corpo.get("body")),
                closing=_texto(corpo.get("closing")),
                search_terms=_termos(corpo.get("search_terms")),
                facts=usados,
                format=mode,
                broll=_termos(corpo.get("broll"))[:3],
                pillar=pillar,
                caption=_legenda(corpo.get("caption")),
            )
        except ValidationError as exc:
            tentativa.violations.extend(_violacoes_de_contrato(exc))
            return tentativa, None

        tentativa.word_count = script.word_count
        tentativa.narration = script.narration
        tentativa.violations.extend(_violacoes_mecanicas(script, dossier, fora, mode))
        return tentativa, (script if not tentativa.violations else None)


def _violacoes_mecanicas(script: Script, dossier: Dossier, fora: list[int],
                         mode: str = "long") -> list[str]:
    """O que da para conferir sem julgamento. Texto vai de volta ao modelo."""
    problemas: list[str] = []
    minimo, maximo = BANDS[mode]
    tmin, tmax = TERMS_PER_MODE[mode]

    marcadores = _MARCADOR_DE_CITACAO.findall(script.narration)
    if marcadores:
        problemas.append(
            f"a narracao contem marcador de citacao ({', '.join(marcadores[:4])}). "
            "O texto e falado por um sintetizador: ele leria esses numeros em voz "
            "alta. O indice do fato vai APENAS no campo used_facts."
        )

    if not (minimo <= script.word_count <= maximo):
        alvo = (minimo + maximo) // 2
        problemas.append(
            f"a narracao tem {script.word_count} palavras "
            f"(~{script.estimated_duration_s:.0f}s) e precisa ter entre {minimo} e "
            f"{maximo}. Reescreva com cerca de {alvo} palavras."
        )

    if not (tmin <= len(script.search_terms) <= tmax):
        problemas.append(
            f"search_terms tem {len(script.search_terms)} termos e o modo {mode} "
            f"pede entre {tmin} e {tmax}, em ordem cronologica."
        )

    if fora:
        problemas.append(
            f"used_facts aponta indice que nao existe no dossie: {fora}. "
            f"Os indices validos vao de 0 a {len(dossier.facts) - 1}."
        )
    if not script.facts:
        problemas.append(
            "used_facts esta vazio: todo roteiro precisa apoiar-se em pelo menos "
            "um fato do dossie, com fonte."
        )

    soltos = _numeros_sem_dossie(script, dossier)

    if soltos:
        problemas.append(
            f"a narracao cita numero que nao esta no dossie: {', '.join(soltos)}. "
            "Use so os numeros dos fatos, sem converter unidade e sem arredondar."
        )

    problemas.extend(validate_terms(script.search_terms))
    problemas.extend(validate_broll(script.broll, mode))
    problemas.extend(caption_problems(script))

    sem_sujeito = missing_subject(script.narration, dossier.topic)
    if sem_sujeito:
        problemas.append(
            "o roteiro fala de 'um modelo' sem nomear: cite "
            + ", ".join(f"{t!r}" for t in sem_sujeito) + " (nome e criador, "
            "conforme o dossie -- nunca invente). Sem nome nao ha busca nem "
            "credibilidade.")

    falha_gancho = check_hook(script.hook)
    if falha_gancho is not None:
        problemas.append(falha_gancho)
    problemas.extend(check_numbers(script.narration))
    problemas.extend(check_emoji_bordao(script.narration))

    return problemas


def _numeros_sem_dossie(script: Script, dossier: Dossier) -> list[str]:
    """Numero em digito na narracao precisa estar em algum fato do dossie.

    E o mesmo portao que o pesquisador usa para conferir fato contra pagina, com
    o dossie no lugar da pagina -- a pergunta e identica ("este numero existe na
    fonte?") e ter duas implementacoes dela garantiria duas respostas.

    Limite conhecido: pega so o que esta escrito em digito. A narracao boa
    escreve numero por extenso para o TTS ("cinco virgula nove gigabytes"), e
    conferir isso exigiria converter numeral em portugues de volta para digito.
    Quem cobre esse caso e o criterio 2 da rubrica do juiz, com o dossie em maos
    -- este portao so garante que o barato de conferir nunca passe errado.
    """
    fontes = "\n".join(f"{f.claim}\n{f.quote}" for f in dossier.facts)
    # O marcador de citacao sai antes da conta: ele tem violacao propria, e
    # deixa-lo aqui faria o portao acusar "numero 0, 1, 2 sem respaldo" -- que e
    # verdade e nao ajuda ninguem a entender o que fazer.
    narracao = _MARCADOR_DE_CITACAO.sub(" ", script.narration)
    return missing_numbers(narracao, fontes)


# Fatos minimos por formato. O curto e "uma ideia so, 1 ou 2 fatos": exigir
# 3 dele recusava justamente o dossie que so serve para curto (visto no teste
# do piloto em 20/09 -- o formato escolheu curto e o roteirista recusou).
MIN_FATOS_POR_MODO = {"long": MIN_FATOS_PARA_ROTEIRO, "carousel": MIN_FATOS_PARA_ROTEIRO,
                      "short": 1}


def thin_dossier_reason(dossier: Dossier, mode: str = "long") -> str:
    """Motivo para nao tentar escrever, ou string vazia se da para tentar.

    A faixa de 60-90s exige umas 150 palavras de conteudo. Dossie com dois fatos
    tirados da mesma frase nao tem isso, e o roteirista so tem duas saidas:
    encher de enrolacao, ou inventar. As duas sao piores que recusar com motivo.

    O numero de FONTES nao entra: uma fonte rica rende roteiro (o roteiro de
    referencia do M0 tem cinco fatos de um unico release). O que conta e quantos
    fatos distintos existem.
    """
    minimo = MIN_FATOS_POR_MODO.get(mode, MIN_FATOS_PARA_ROTEIRO)
    if len(dossier.facts) < minimo:
        alvo = (f"a faixa de {MIN_DURATION_S}-{MAX_DURATION_S}s" if mode == "long"
                else f"o formato {mode}")
        return (
            f"dossie fino: {len(dossier.facts)} fato(s), e {alvo} pede pelo menos "
            f"{minimo}. Pesquise outras fontes antes de roteirizar."
        )
    return ""


def _resolver_fatos(indices: object, facts: list[Fact]) -> tuple[list[Fact], list[int]]:
    """Traduz os indices que o modelo devolveu em fatos do dossie.

    O modelo aponta, nunca copia: se ele pudesse reescrever o fato, a afirmacao
    do roteiro deixaria de ser rastreavel ao que a fonte diz -- que e todo o
    ponto de o dossie existir.
    """
    if not isinstance(indices, list):
        return [], []

    usados: list[Fact] = []
    fora: list[int] = []
    for bruto in indices:
        if isinstance(bruto, bool) or not isinstance(bruto, int):
            continue
        if 0 <= bruto < len(facts):
            if facts[bruto] not in usados:
                usados.append(facts[bruto])
        else:
            fora.append(bruto)
    return usados, fora


def _violacoes_de_contrato(exc: ValidationError) -> list[str]:
    saida: list[str] = []
    for erro in exc.errors():
        campo = ".".join(str(p) for p in erro["loc"]) or "roteiro"
        saida.append(f"o campo {campo} nao respeita o contrato: {erro['msg']}")
    return saida


# Exemplo de ~25s que cabe na faixa: o modelo imita o TAMANHO, nao so o tom --
# medido em 19/09/2026, sem exemplo de extensao ele escrevia o curto no
# tamanho do longo. Tem 65 palavras de proposito, o meio da faixa.
SHORT_EXAMPLE = (
    "hook: Um modelo gigante cabe no seu bolso?\n"
    "body: O Bonsai 2 tem vinte e sete bilhões de parâmetros em só cinco "
    "vírgula nove gigabytes. É nove vezes menor que o original e mantém quase "
    "todo o desempenho. Na prática, ele roda num notebook comum, sem nuvem, "
    "sem mensalidade e sem mandar seus dados para o servidor de ninguém.\n"
    "closing: Gigante no bolso: o que mais vai encolher?")

# Quantos dados numericos o video comporta. O longo de 19/09 empilhou sete
# ("cento e quarenta e tres tokens por segundo... zero ponto setecentos e
# quatorze miliwatts-hora") e virou ficha tecnica lida em voz alta.
MAX_NUMEROS = {"long": 3, "short": 1}


def build_prompt(dossier: Dossier, correcoes: list[str] | None = None,
                 mode: str = "long", pillar: str = "", previous: str = "") -> str:
    """Monta o prompt do roteiro. Funcao livre para o teste inspecionar o texto.

    Ordem: tema e dossie (o material), tarefa e tipo de conteudo (a forma),
    estetica e voz (a marca), regras (o que reprova), correcao (so na volta).
    """
    fatos = "\n".join(
        f"[{i}] {f.claim}\n    fonte: {f.source_name}"
        + (f'\n    trecho: "{f.quote}"' if f.quote else "")
        for i, f in enumerate(dossier.facts)
    )
    minimo, maximo = BANDS[mode]
    tmin, tmax = TERMS_PER_MODE[mode]
    alvo = (minimo + maximo) // 2
    fechamento = (
        "- closing: fechamento com PONTO DE VISTA PROPRIO: aponte o que as fontes "
        "NAO dizem, ou a pergunta que elas deixam aberta, e devolva ao "
        "espectador. Chamada concreta, nunca 'siga para mais'.\n"
        if mode == "long" else
        "- closing: UMA frase que reconecta com a pergunta do hook E aponta a "
        "implicacao que a fonte nao desenvolve (ex.: 'Se sao dois orgaos, qual "
        "deles decide por voce?'). Sem ela o video e resumo. Sem 'siga para mais'.\n"
    )
    duracao_txt = (
        f"Isso equivale a {MIN_DURATION_S}-{MAX_DURATION_S}s falados e e "
        "requisito de monetizacao, nao preferencia.\n"
        if mode == "long" else
        "Video curto de alcance (~25s): uma ideia so, 2 ou 3 fatos no maximo. "
        "Conte as palavras antes de responder e corte frases inteiras se passar "
        "do teto.\n"
    )
    partes = [
        f"TEMA: {dossier.topic}\n",
        f"DOSSIE (use o indice para citar):\n{fatos}\n",
        "TAREFA\n"
        "Roteiro de video vertical em portugues do Brasil, para ser narrado. Devolva:\n"
        "- hook: a PRIMEIRA FRASE abre uma lacuna de informacao e nao a responde. "
        "Ate 12 palavras (regra da marca). Nada de 'hoje eu vou falar sobre'.\n"
        "- body: todo dado vem de um fato do dossie. Na primeira frase, contexto: "
        "nomeie o assunto (nome + quem construiu, SE o dossie disser) e por que "
        "importa para quem assiste.\n"
        + fechamento +
        f"- search_terms: de {tmin} a {tmax} termos EM INGLES, na ordem cronologica "
        "da narracao, COPIADOS da lista de ESTETICA.\n"
        "- broll: termos EM INGLES do objeto concreto do assunto (regra abaixo).\n"
        "- caption: legenda do post em 2 linhas: um gancho ESCRITO diferente da fala "
        "e uma frase de contexto (quem, o que). Sem emoji, link ou hashtag.\n"
        "- used_facts: os indices dos fatos do dossie em que o roteiro se apoia.\n",
        pillar_brief(pillar or "news", mode),
        compact_brief(suggest_pillar(dossier.topic), mode),
        *([] if mode != "short" else [
            "EXEMPLO DE TAMANHO (25s: copie a extensao, nao o texto)\n" + SHORT_EXAMPLE]),
        voice_brief(),
        "REGRAS\n"
        f"- A narracao inteira (hook + body + closing) tem entre {minimo} e {maximo} "
        f"palavras, cerca de {alvo}. " + duracao_txt +
        f"- No maximo {MAX_NUMEROS[mode]} dado(s) numerico(s) no video inteiro: escolha "
        "o que o espectador lembraria e traduza em comparacao. Ficha tecnica lida "
        "em voz alta perde a pessoa.\n"
        "- Numero por extenso quando soar melhor ('cinco virgula nove gigabytes'), "
        "sem mudar o valor. Nao invente numero, nome, data nem citacao.\n"
        "- Descricao em ingles vira portugues na fala ('High-Resolution Stereo Camera' -> "
        "'camera estereo de alta resolucao'); em ingles, so nome proprio curto.\n"
        "- NAO escreva indices no texto ('[0]', '[1, 2]'): o sintetizador de voz "
        "leria em voz alta. O indice vai apenas no campo used_facts.\n"
        "- Sem emoji, sem hashtag, sem marcacao de cena. So o que sera falado.",
    ]
    if correcoes:
        bloco = "CORRIJA A TENTATIVA ANTERIOR\n" + "\n".join(f"- {c}" for c in correcoes)
        if previous:
            bloco += f"\nTEXTO ANTERIOR (ajuste este, nao comece do zero):\n{previous}"
        partes.append(bloco + "\nMantenha o que estava bom e conserte apenas o apontado.")
    return "\n".join(partes)


def _texto(valor: object) -> str:
    return " ".join(str(valor).split()) if isinstance(valor, str) else ""


def _legenda(valor: object) -> str:
    """Legenda preserva a quebra de linha (sao duas linhas por regra da marca)."""
    if not isinstance(valor, str):
        return ""
    linhas = [" ".join(linha.split()) for linha in valor.splitlines()]
    return "\n".join(linha for linha in linhas if linha)


_LINK = re.compile(r"https?://|www\.|\.com\b|\.br\b", re.IGNORECASE)


def caption_problems(script: Script) -> list[str]:
    """Regras da legenda no guia: 2 linhas, sem emoji/link/hashtag, sem repetir o audio."""
    legenda = script.caption.strip()
    if not legenda:
        return ["caption vazia: escreva 2 linhas (gancho escrito diferente da fala + "
                "uma frase de contexto: quem, o que)."]
    problemas: list[str] = []
    linhas = legenda.splitlines()
    if len(linhas) > 2 or len(legenda) > 220:
        problemas.append(f"caption com {len(linhas)} linha(s) e {len(legenda)} caracteres: "
                         "no maximo 2 linhas curtas (regra da marca).")
    if "#" in legenda:
        problemas.append("caption com hashtag: as 5 hashtags da marca entram sozinhas.")
    if _LINK.search(legenda):
        problemas.append("caption com link ou dominio: a regra da marca e sem link.")
    if check_emoji_bordao(legenda):
        problemas.append("caption com emoji ou bordao: a marca nunca usa.")
    if script.hook and legenda.splitlines()[0].strip().lower() == script.hook.strip().lower():
        problemas.append("caption repete o hook falado: a primeira linha e um gancho ESCRITO "
                         "diferente do audio (guia da marca).")
    return problemas


def _termos(valor: object) -> list[str]:
    if not isinstance(valor, list):
        return []
    return [" ".join(str(t).split()) for t in valor if isinstance(t, str) and t.strip()]
