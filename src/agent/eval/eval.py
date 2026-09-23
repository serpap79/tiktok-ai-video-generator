"""Agregacao offline do eval: escritor x juiz com custo medido, sem gastar cota.

O eval nao chama modelo nenhum. Ele le os roteiros e pareceres que `write`,
`judge` e `produce` ja gravaram e tabula por (provedor, modelo). O juiz nunca
recebe quem escreveu o roteiro -- a cegueira e estrutural, nao promessa -- e a
matriz pareada so usa parecer com `script_id` conhecido; parecer orfao (do
`judge --script <arquivo>`, sem linha de roteiro) entra no agregado do juiz,
mas nao na matriz.

Comparacao deliberadamente free tier x free tier (gemini x groq). O braco pago
(Claude) esta fora de escopo: sem API paga nesta maquina, nao ha numero a
publicar -- e numero ausente nao e zero, e por isso nem aparece na tabela.

Nota interrompida (`short_circuited`) nao entra em media nem em taxa de
aprovacao: o total dela nao e comparavel ao de um parecer completo, e mistura
os dois faria um juiz que reprova na medida parecer mais rigoroso do que e.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.models import Criterion, Review


def chave(provider: str, model: str) -> str:
    """`gemini/gemini-2.5-flash`. Modelo vazio vira `?`, nunca string vazia."""
    return f"{provider or '?'}/{model or '?'}"


@dataclass(frozen=True)
class ScriptRow:
    id: int
    topic: str
    model: str
    provider: str
    word_count: int
    attempts: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    format: str = "long"


@dataclass(frozen=True)
class CarouselRow:
    id: int
    topic: str
    model: str
    provider: str
    approved: bool
    input_tokens: int
    output_tokens: int
    latency_s: float


@dataclass
class WriterSummary:
    key: str
    format: str = "long"
    n: int = 0
    avg_words: float = 0.0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class CarouselSummary:
    key: str
    n: int = 0
    approval_rate: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


@dataclass(frozen=True)
class ReviewRow:
    id: int
    topic: str
    script_id: int | None
    model: str
    provider: str
    review: Review
    input_tokens: int
    output_tokens: int
    latency_s: float


@dataclass
class JudgeSummary:
    key: str
    n_total: int = 0
    n_full: int = 0
    avg_total: float = 0.0
    approval_rate: float = 0.0
    avg_by_criterion: dict[str, float] = field(default_factory=dict)


@dataclass
class PairedCell:
    writer: str
    judge: str
    n: int = 0
    avg_total: float = 0.0


@dataclass
class EvalReport:
    topics: list[str] = field(default_factory=list)
    writers: list[WriterSummary] = field(default_factory=list)
    judges: list[JudgeSummary] = field(default_factory=list)
    paired: list[PairedCell] = field(default_factory=list)
    carousels: list[CarouselSummary] = field(default_factory=list)
    n_scripts: int = 0
    n_reviews: int = 0
    n_carousels: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


def summarize_writers(scripts: list[ScriptRow]) -> list[WriterSummary]:
    agreg: dict[tuple[str, str], WriterSummary] = {}
    for s in scripts:
        k = (s.format or "long", chave(s.provider, s.model))
        w = agreg.setdefault(k, WriterSummary(key=k[1], format=k[0]))
        w.n += 1
        w.avg_words += s.word_count
        w.attempts += s.attempts
        w.input_tokens += s.input_tokens
        w.output_tokens += s.output_tokens
        w.latency_s += s.latency_s
    saidas = sorted(agreg.values(), key=lambda w: (w.format, w.key))
    for w in saidas:
        w.avg_words = round(w.avg_words / w.n, 1) if w.n else 0.0
        w.latency_s = round(w.latency_s, 3)
    return saidas


def summarize_carousels(rows: list[CarouselRow]) -> list[CarouselSummary]:
    agreg: dict[str, CarouselSummary] = {}
    for r in rows:
        k = chave(r.provider, r.model)
        c = agreg.setdefault(k, CarouselSummary(key=k))
        c.n += 1
        c.approval_rate += int(r.approved)
        c.input_tokens += r.input_tokens
        c.output_tokens += r.output_tokens
        c.latency_s += r.latency_s
    saidas = sorted(agreg.values(), key=lambda c: c.key)
    for c in saidas:
        c.approval_rate = round(c.approval_rate / c.n, 3) if c.n else 0.0
        c.latency_s = round(c.latency_s, 3)
    return saidas


def summarize_judges(reviews: list[ReviewRow]) -> list[JudgeSummary]:
    agreg: dict[str, JudgeSummary] = {}
    somas: dict[str, dict[str, float]] = {}
    contagem: dict[str, dict[str, int]] = {}
    for r in reviews:
        k = chave(r.provider, r.model)
        j = agreg.setdefault(k, JudgeSummary(key=k))
        j.n_total += 1
        if r.review.short_circuited:
            continue
        j.n_full += 1
        j.avg_total += r.review.total
        j.approval_rate += int(r.review.approved)
        sc = somas.setdefault(k, {})
        cc = contagem.setdefault(k, {})
        for s in r.review.scores:
            if s.evaluated:
                sc[s.criterion.value] = sc.get(s.criterion.value, 0.0) + s.score
                cc[s.criterion.value] = cc.get(s.criterion.value, 0) + 1
    saidas = sorted(agreg.values(), key=lambda j: j.key)
    for j in saidas:
        if j.n_full:
            j.avg_total = round(j.avg_total / j.n_full, 2)
            j.approval_rate = round(j.approval_rate / j.n_full, 3)
            sc = somas.get(j.key, {})
            cc = contagem.get(j.key, {})
            j.avg_by_criterion = {
                c: round(sc[c] / cc[c], 2) for c in sorted(sc) if cc[c]
            }
    return saidas


def paired_matrix(
    scripts: list[ScriptRow], reviews: list[ReviewRow]
) -> list[PairedCell]:
    """Media por par (escritor, juiz). So parecer completo com roteiro conhecido."""
    escritor_de = {s.id: chave(s.provider, s.model) for s in scripts}
    agreg: dict[tuple[str, str], PairedCell] = {}
    for r in reviews:
        if r.script_id is None or r.script_id not in escritor_de:
            continue
        if r.review.short_circuited:
            continue
        par = (escritor_de[r.script_id], chave(r.provider, r.model))
        cell = agreg.setdefault(par, PairedCell(writer=par[0], judge=par[1]))
        cell.n += 1
        cell.avg_total += r.review.total
    saidas = sorted(agreg.values(), key=lambda c: (c.writer, c.judge))
    for c in saidas:
        c.avg_total = round(c.avg_total / c.n, 2) if c.n else 0.0
    return saidas


def build_report(
    scripts: list[ScriptRow], reviews: list[ReviewRow],
    carousels: list[CarouselRow] | None = None,
) -> EvalReport:
    carousels = carousels or []
    return EvalReport(
        topics=sorted({s.topic for s in scripts} | {r.topic for r in reviews}
                      | {c.topic for c in carousels}),
        writers=summarize_writers(scripts),
        judges=summarize_judges(reviews),
        paired=paired_matrix(scripts, reviews),
        carousels=summarize_carousels(carousels),
        n_scripts=len(scripts),
        n_reviews=len(reviews),
        n_carousels=len(carousels),
        input_tokens=sum(s.input_tokens for s in scripts)
        + sum(r.input_tokens for r in reviews)
        + sum(c.input_tokens for c in carousels),
        output_tokens=sum(s.output_tokens for s in scripts)
        + sum(r.output_tokens for r in reviews)
        + sum(c.output_tokens for c in carousels),
        latency_s=round(
            sum(s.latency_s for s in scripts)
            + sum(r.latency_s for r in reviews)
            + sum(c.latency_s for c in carousels),
            3,
        ),
    )


def format_text(report: EvalReport) -> str:
    """Tabela pronta para o terminal e para colar no README."""
    linhas = [
        f"temas    : {len(report.topics)} ({', '.join(report.topics) or 'nenhum'})",
        f"roteiros : {report.n_scripts} | pareceres: {report.n_reviews} "
        f"| carrosseis: {report.n_carousels}",
        "",
        "escritor [formato] (roteiros, palavras medias, tentativas, tokens in/out, latencia):",
    ]
    for w in report.writers:
        linhas.append(
            f"  [{w.format}] {w.key}: n={w.n} palavras~{w.avg_words} "
            f"tent={w.attempts} tok={w.input_tokens}/{w.output_tokens} {w.latency_s}s"
        )
    linhas.append("")
    linhas.append(
        "juiz (pareceres completos/total, media, aprovacao, media por criterio):"
    )
    for j in report.judges:
        crits = " ".join(f"{c}={v}" for c, v in j.avg_by_criterion.items())
        linhas.append(
            f"  {j.key}: {j.n_full}/{j.n_total} media={j.avg_total} "
            f"aprov={j.approval_rate:.0%} {crits}".rstrip()
        )
    linhas.append("")
    linhas.append("pareado (mesmo roteiro, juizes diferentes):")
    if report.paired:
        for c in report.paired:
            linhas.append(
                f"  {c.writer} x {c.judge}: n={c.n} media={c.avg_total}"
            )
    else:
        linhas.append("  (sem parecer ligado a roteiro conhecido)")
    linhas.append("")
    linhas.append("carrossel (aprovacao, tokens in/out, latencia):")
    if report.carousels:
        for c in report.carousels:
            linhas.append(
                f"  {c.key}: n={c.n} aprov={c.approval_rate:.0%} "
                f"tok={c.input_tokens}/{c.output_tokens} {c.latency_s}s"
            )
    else:
        linhas.append("  (nenhum carrossel gravado)")
    linhas.append("")
    linhas.append(
        f"custo total: {report.input_tokens} tokens de entrada, "
        f"{report.output_tokens} de saida, {report.latency_s}s de modelo"
    )
    return "\n".join(linhas) + "\n"


__all__ = [
    "CarouselRow",
    "CarouselSummary",
    "Criterion",
    "EvalReport",
    "JudgeSummary",
    "PairedCell",
    "ReviewRow",
    "ScriptRow",
    "WriterSummary",
    "build_report",
    "chave",
    "format_text",
    "paired_matrix",
    "summarize_carousels",
    "summarize_judges",
    "summarize_writers",
]
