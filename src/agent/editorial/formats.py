"""Qual formato serve a este tema: video longo, curto ou carrossel.

A pergunta do autor foi: "o agente deve saber a prioridade de formato de
acordo com a informacao que recebe". Aqui a resposta e uma nota por formato
com tres partes, cada uma gravada no motivo:

1. **o que a informacao aguenta** (`fit`, medido no dossie): fato unico com
   numero forte cabe em 15s; cinco fatos com data e fonte sustentam 60-90s;
   quatro itens paralelos (numeros, passos) viram cinco slides. Dossie com
   menos de 3 fatos NAO vira longo nem carrossel -- e regra, nao nota.
2. **o que o tipo de conteudo pede** (`affinity`): tutorial e comparacao sao
   salvaveis (carrossel); historia e analise pedem arco (longo); curiosidade
   e noticia quente cabem no curto.
3. **o que o horario favorece** (`slot`): hipotese declarada em
   `editorial/slots.py`.

Por cima, duas correcoes: formato ja usado hoje perde pontos (variedade para
o publico E amostra para o eval) e o desempenho MEDIDO do canal entra quando
existir.

Desde 20/09/2026 a grade dos quatro slots **forca** o formato (curto de manha
e na tarde, longo no almoco e a noite), entao a nota abaixo so desempata
dentro do que o horario permite -- e a penalidade de repeticao nao muda
resultado nenhum quando ha um formato so permitido. Ela continua valendo para
`slot-extra` e para o dia em que a grade voltar a ser livre.

Nada disto diz o que viraliza no TikTok: nao ha API publica para isso. Os
pesos de 2 e 3 sao priors declarados; o de desempenho e o unico medido, e e
o que deve crescer com o tempo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.models import Dossier

FORMATS = ("long", "short", "carousel")

# Ordem forcada por horario (decisao do autor em 20/09/2026, segunda
# versao): quatro posts por dia, todos VIDEO, alternando curto e longo. O
# carrossel saiu da grade -- segue implementado e alcancavel por
# `slot-extra --format carrossel`, mas nao ocupa mais horario.
#
# O segundo elemento e emergencia, nao alternativa: dossie com menos de 3
# fatos nao sustenta um longo (`MIN_FATOS`), e nesse caso o slot sai curto em
# vez de falhar -- e o motivo gravado diz que foi emergencia. Os slots curtos
# nao precisam de reserva: curto exige 1 fato, que e o minimo que o
# pesquisador entrega.
FORMATO_FORCADO: dict[str, tuple[str, ...]] = {
    "0900": ("short",),
    "1200": ("long", "short"),
    "1600": ("short",),
    "1900": ("long", "short"),
}

# Minimo de fatos para cada formato. Abaixo de 3 o roteirista ja recusa longo
# e carrossel (writer.MIN_FATOS_PARA_ROTEIRO); aqui a regra so fica explicita.
MIN_FATOS = {"long": 3, "short": 1, "carousel": 3}

# Prior por horario (ver slots.py). Soma 1 em cada linha.
SLOT_PRIOR: dict[str, dict[str, float]] = {
    "0900": {"short": 0.55, "long": 0.30, "carousel": 0.15},
    "1200": {"long": 0.60, "short": 0.25, "carousel": 0.15},
    "1600": {"short": 0.50, "long": 0.30, "carousel": 0.20},
    "1900": {"long": 0.60, "short": 0.25, "carousel": 0.15},
}
# Prior de quem nao esta na grade (`slot-extra`, ou um slot novo antes de
# ganhar linha propria). Era `SLOT_PRIOR["1500"]` escrito direto na funcao, e
# aquele id deixou de existir quando a grade virou 09/12/16/19 -- um
# `KeyError` esperando o primeiro `slot-extra`.
PRIOR_PADRAO: dict[str, float] = {"short": 0.40, "long": 0.35, "carousel": 0.25}

# Afinidade tipo de conteudo x formato. Soma 1 em cada linha.
AFFINITY: dict[str, dict[str, float]] = {
    "news": {"short": 0.50, "long": 0.35, "carousel": 0.15},
    "fato": {"short": 0.55, "carousel": 0.25, "long": 0.20},
    "analise": {"long": 0.60, "carousel": 0.25, "short": 0.15},
    "tutorial": {"carousel": 0.60, "long": 0.30, "short": 0.10},
    "futuro": {"long": 0.45, "short": 0.35, "carousel": 0.20},
    "vs": {"carousel": 0.50, "short": 0.30, "long": 0.20},
    "historia": {"long": 0.60, "carousel": 0.20, "short": 0.20},
}

PESO_SLOT = 0.30
PESO_AFINIDADE = 0.30
PESO_ENCAIXE = 0.40
PENALIDADE_REPETIDO = 0.25
# Desempenho medido so entra com amostra: menos que isso por formato e ruido.
AMOSTRA_MINIMA = 3
TETO_DESEMPENHO = 0.15

_ANO = re.compile(r"\b(1[6-9]\d\d|20\d\d)\b")
_DIGITO = re.compile(r"\d")
_PASSO = re.compile(
    r"\b(passo|etapa|step|primeiro|segundo|terceiro|instal|configur|execut|rode|use |"
    r"usar|clique|digite|comando|command|ative|habilit)", re.IGNORECASE)


@dataclass(frozen=True)
class DossierFeatures:
    facts: int
    with_numbers: int
    domains: int
    steps: int
    dated: int

    @property
    def listness(self) -> float:
        """0-1: quanto os fatos se comportam como itens de lista."""
        return min(1.0, (self.with_numbers + self.steps) / 4)


def features(dossier: Dossier) -> DossierFeatures:
    claims = [f"{f.claim} {f.quote}" for f in dossier.facts]
    dominios = {str(f.source_url).split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")
                for f in dossier.facts}
    return DossierFeatures(
        facts=len(dossier.facts),
        with_numbers=sum(1 for f in dossier.facts if _DIGITO.search(f.claim)),
        domains=len(dominios),
        steps=sum(1 for c in claims if _PASSO.search(c)),
        dated=sum(1 for f in dossier.facts if _ANO.search(f.claim)),
    )


def fit(feat: DossierFeatures, formato: str) -> float:
    """O quanto a informacao sustenta o formato, de 0 a 1."""
    if feat.facts < MIN_FATOS[formato]:
        return 0.0
    if formato == "long":
        return round(0.5 * min(1.0, feat.facts / 5)
                     + 0.25 * min(1.0, feat.domains / 2)
                     + 0.25 * (1.0 if feat.dated or feat.facts >= 5 else 0.4), 3)
    if formato == "carousel":
        return round(0.5 * min(1.0, feat.facts / 4) + 0.5 * feat.listness, 3)
    # short: um fato forte basta; numero e o que prende em 15s.
    return round(0.6 + (0.4 if feat.with_numbers else 0.12), 3)


@dataclass
class FormatDecision:
    format: str
    scores: dict[str, float]
    parts: dict[str, dict[str, float]] = field(default_factory=dict)
    reason: str = ""

    def ranked(self) -> list[str]:
        return sorted(self.scores, key=lambda f: -self.scores[f])


def choose_format(feat: DossierFeatures, pillar: str, slot_id: str,
                  used_today: list[str] | None = None,
                  performance: dict[str, float] | None = None,
                  allowed: tuple[str, ...] = FORMATS,
                  em_ordem: bool = False) -> FormatDecision:
    """Formato com a maior nota; o motivo diz de onde veio cada parte.

    Com `em_ordem=True` a lista `allowed` deixa de ser um conjunto de
    permitidos e passa a ser **preferencia**: vale o primeiro que o dossie
    sustenta, e os seguintes so entram como emergencia. A nota continua sendo
    calculada e gravada no motivo -- ela apenas nao decide.

    Por que existe: `FORMATO_FORCADO` sempre foi descrito como ordem ("o
    segundo elemento e emergencia, nao alternativa") e nunca foi uma. A funcao
    escolhia pela nota dentro do permitido, e isso passou despercebido
    enquanto o unico slot forcado era o carrossel das 20h, que pontuava alto
    sozinho. Quando a grade de 20/09/2026 forcou `("long", "short")` no
    almoco e na noite, o curto passou a ganhar do longo em pilar de noticia
    (afinidade 0,50 contra 0,35) e os dois slots longos do dia sairiam curtos
    -- exatamente o contrario do que a grade pede. Descoberto conferindo a
    grade depois de montada, nao por teste.
    """
    usados = list(used_today or [])
    desempenho = performance or {}
    prior_slot = SLOT_PRIOR.get(slot_id, PRIOR_PADRAO)
    afinidade = AFFINITY.get(pillar, AFFINITY["news"])

    notas: dict[str, float] = {}
    partes: dict[str, dict[str, float]] = {}
    for f in allowed:
        encaixe = fit(feat, f)
        if encaixe == 0.0:
            continue
        p = {
            "slot": round(PESO_SLOT * prior_slot.get(f, 0.0), 3),
            "tipo": round(PESO_AFINIDADE * afinidade.get(f, 0.0), 3),
            "encaixe": round(PESO_ENCAIXE * encaixe, 3),
            "repetido": round(-PENALIDADE_REPETIDO * usados.count(f), 3),
            "medido": round(max(-TETO_DESEMPENHO,
                                min(TETO_DESEMPENHO, desempenho.get(f, 0.0))), 3),
        }
        partes[f] = p
        notas[f] = round(sum(p.values()), 3)

    if not notas:
        # So chega aqui com dossie sem fato (o pesquisador nao entrega isso)
        # ou `allowed` vazio. Curto e o formato que menos exige.
        return FormatDecision("short", {"short": 0.0}, {},
                              "nenhum formato sustentado pelo dossie; curto por padrao")

    if em_ordem:
        escolhido = next((f for f in allowed if f in notas), None)
        if escolhido is None:
            escolhido = max(notas, key=lambda f: (notas[f], -FORMATS.index(f)))
    else:
        escolhido = max(notas, key=lambda f: (notas[f], -FORMATS.index(f)))
    p = partes[escolhido]
    outros = ", ".join(f"{f} {notas[f]:.2f}" for f in sorted(notas, key=lambda x: -notas[x])
                       if f != escolhido)
    medido = ("desempenho medido do canal entrou" if any(desempenho.values())
              else "sem metrica propria com amostra ainda: pesos sao prior declarado")
    reason = (
        f"{escolhido} ({notas[escolhido]:.2f}) = horario {p['slot']:+.2f}, "
        f"tipo {pillar} {p['tipo']:+.2f}, dossie {p['encaixe']:+.2f} "
        f"({feat.facts} fatos, {feat.with_numbers} com numero, {feat.domains} fonte(s), "
        f"{feat.steps} passo(s), {feat.dated} com data)"
        + (f", repetido hoje {p['repetido']:+.2f}" if p["repetido"] else "")
        + (f", medido {p['medido']:+.2f}" if p["medido"] else "")
        + (f"; outros: {outros}" if outros else "")
        + f"; {medido}"
    )
    return FormatDecision(escolhido, notas, partes, reason)


def performance_from_metrics(rows: list[dict]) -> dict[str, float]:
    """Bonus por formato a partir das metricas lidas no app, com amostra.

    `rows`: uma linha por post com `format`, `views` e `completion_rate`
    (a ultima coleta de cada publish_id). Cada formato recebe a diferenca
    relativa da sua mediana de conclusao para a mediana geral, limitada a
    +-TETO_DESEMPENHO. Formato com menos de AMOSTRA_MINIMA posts fica em 0:
    dois videos bons nao sao tendencia.
    """
    por_formato: dict[str, list[float]] = {}
    for r in rows:
        taxa = r.get("completion_rate")
        if taxa is None or r.get("format") not in FORMATS:
            continue
        por_formato.setdefault(r["format"], []).append(float(taxa))
    todas = [v for vs in por_formato.values() for v in vs]
    if not todas:
        return {}
    geral = _mediana(todas)
    saida: dict[str, float] = {}
    for f, valores in por_formato.items():
        if len(valores) < AMOSTRA_MINIMA or geral <= 0:
            continue
        mediana = _mediana(valores)
        saida[f] = round(max(-TETO_DESEMPENHO, min(TETO_DESEMPENHO,
                                                   (mediana - geral) / geral)), 3)
    return saida


def _mediana(valores: list[float]) -> float:
    ordenados = sorted(valores)
    meio = len(ordenados) // 2
    if len(ordenados) % 2:
        return ordenados[meio]
    return (ordenados[meio - 1] + ordenados[meio]) / 2


__all__ = ["AFFINITY", "FORMATS", "FORMATO_FORCADO", "MIN_FATOS", "PRIOR_PADRAO",
            "SLOT_PRIOR", "DossierFeatures",
            "FormatDecision", "choose_format", "features", "fit",
            "performance_from_metrics"]
