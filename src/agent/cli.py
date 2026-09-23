"""CLI do agente."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from agent.adapters.mpt_renderer import MptRenderer
from agent.config import settings
from agent.models import MAX_DURATION_S, MIN_DURATION_S, Decision, RenderState, Script
from agent.ports.renderer import RendererError

app = typer.Typer(add_completion=False, help="Agente de video curto para tech/IA/ciencia")


def _load_script(path: Path) -> Script:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return Script.model_validate(raw)


@app.command()
def render(
    script_path: Path = typer.Option(..., "--script", "-s", exists=True, readable=True),
    out_dir: Path = typer.Option(None, "--out-dir",
                                 help="pasta do pacote; padrao organiza por dia/hora/tema"),
) -> None:
    """Renderiza um roteiro em MP4 vertical e confere o aceite do M0."""
    from agent.paths import run_dir
    script = _load_script(script_path)
    typer.echo(f"tema      : {script.topic}")
    typer.echo(f"narracao  : {script.word_count} palavras "
               f"(~{script.estimated_duration_s:.0f}s estimados)")
    typer.echo(f"termos    : {', '.join(script.search_terms)}")
    typer.echo(f"fatos     : {len(script.facts)} com fonte")

    renderer = MptRenderer()
    if not renderer.health():
        typer.secho(
            f"renderizador nao responde em {settings.renderer_url}\n"
            "suba com: ./scripts/setup_renderer.sh --serve",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    typer.echo("renderizando (minutos, em CPU)...")
    try:
        result = renderer.render(script)
    except RendererError as exc:
        typer.secho(f"falha: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if result.state is RenderState.failed:
        typer.secho(f"renderizacao falhou: {result.error}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo(f"arquivo   : {result.video_path}")
    typer.echo(f"dimensoes : {result.width}x{result.height}")
    typer.echo(f"duracao   : {result.duration_s}s")
    typer.echo(f"narracao  : {'presente' if result.has_audio else 'AUSENTE'}")

    if out_dir is None:
        out_dir = run_dir(script.format or "long", script.topic,
                          base=settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "video.mp4"
    if result.video_path and Path(result.video_path) != destino:
        Path(result.video_path).replace(destino)
    script_path_out = out_dir / "roteiro.json"
    if Path(script_path) != script_path_out:
        script_path_out.write_text(
            Path(script_path).read_text(encoding="utf-8"), encoding="utf-8")
    typer.echo(f"pacote    : {out_dir}")

    # Aceite medido, nao presumido. A faixa depende do formato: short nao
    # e elegivel ao Rewards, e cobrar 60-90s dele reprovaria todo curto.
    formato = script.format or "long"
    faixa = (MIN_DURATION_S, MAX_DURATION_S) if formato == "long" else (10, 30)
    na_faixa = (result.duration_s is not None
                and faixa[0] <= result.duration_s <= faixa[1])
    checks = [
        ("9:16 em 1080x1920", result.is_portrait_1080x1920),
        (f"duracao entre {faixa[0]}s e {faixa[1]}s ({formato})", na_faixa),
        ("trilha de audio presente", result.has_audio),
    ]
    typer.echo("")
    ok = True
    for label, passed in checks:
        mark = "OK  " if passed else "FALHA"
        color = typer.colors.GREEN if passed else typer.colors.RED
        typer.secho(f"[{mark}] {label}", fg=color)
        ok = ok and passed

    if not ok:
        raise typer.Exit(code=1)


@app.command()
def radar(
    limit: int = typer.Option(15, "--limit", "-n", help="quantos sinais listar"),
) -> None:
    """Coleta sinais de tendencia das fontes gratuitas e grava a serie."""
    from agent.memory.store import SignalStore
    from agent.radar.collector import Radar, default_sources

    settings.ensure_dirs()
    report = Radar(default_sources(), SignalStore(settings.db_path)).collect()

    for nome, erro in report.failures.items():
        typer.secho(f"[fonte fora] {nome}: {erro}", fg=typer.colors.YELLOW)

    if not report.signals:
        typer.secho("nenhum sinal coletado", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Ordena por velocidade; sem velocidade vai para o fim, porque "desconhecido"
    # nao pode competir de igual para igual com uma medida.
    ordenados = sorted(
        report.signals, key=lambda s: (s.has_velocity, s.velocity or 0), reverse=True
    )

    typer.echo("")
    typer.echo(f"{'velocidade':>12}  {'volume':>10}  {'fonte':<14}  termo")
    typer.echo("-" * 92)
    for s in ordenados[:limit]:
        vel = f"{s.velocity:,.1f}/h" if s.has_velocity else "-"
        typer.echo(f"{vel:>12}  {s.volume:>10,.0f}  {s.source:<14}  {s.term[:44]}")

    typer.echo("")
    typer.echo(f"{len(report.signals)} sinais de {len(report.sources_ok)} fontes "
               f"({len(report.with_velocity)} com velocidade) em {report.elapsed_s}s")
    if report.failures:
        typer.echo(f"{len(report.failures)} fonte(s) fora; a coleta seguiu sem elas")


@app.command()
def curate(
    show: int = typer.Option(8, "--show", help="quantos rejeitados detalhar"),
    dry_run: bool = typer.Option(False, "--dry-run", help="nao grava no ledger"),
    top: int = typer.Option(1, "--top", help="candidatos distintos p/ a rotina do dia"),
    cooldown_days: int = typer.Option(30, "--cooldown-days",
                                      help="janela anti-repeticao do ledger"),
) -> None:
    """Coleta sinais e escolhe tema(s), registrando o motivo de cada decisao.

    `--top 3` lista os 3 melhores assuntos DISTINTOS (deduplicados entre si)
    para a rotina long+short+carrossel do dia. Repetir assunto so e valido
    como atualizacao explicita (`research --topic`), nunca pelo curador.
    """
    from agent.curator.curator import Curator
    from agent.memory.store import SignalStore
    from agent.models import Verdict
    from agent.radar.collector import Radar, default_sources

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)

    coleta = Radar(default_sources(), store).collect()
    for nome, erro in coleta.failures.items():
        typer.secho(f"[fonte fora] {nome}: {erro}", fg=typer.colors.YELLOW)
    if not coleta.signals:
        typer.secho("nenhum sinal coletado", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    ledger = store.recent_topics(days=cooldown_days)
    report = Curator().curate(coleta.signals, ledger=ledger)

    contagem = report.tally()
    typer.echo("")
    typer.echo(f"{len(coleta.signals)} sinais -> "
               + "  ".join(f"{k}={v}" for k, v in contagem.items() if v))
    typer.echo(f"ledger: {len(ledger)} temas aprovados nos ultimos {cooldown_days} dias")

    for verdict, cor in ((Verdict.rejected_policy, typer.colors.RED),
                         (Verdict.rejected_duplicate, typer.colors.YELLOW)):
        for d in report.by_verdict(verdict)[:show]:
            typer.secho(f"  [{verdict.value}] {d.term[:52]}", fg=cor)
            typer.secho(f"      {d.reason}", fg=typer.colors.BRIGHT_BLACK)

    escolhido = report.selected
    if escolhido is None:
        typer.secho("\nnenhum tema elegivel hoje", fg=typer.colors.RED)
        if not dry_run:
            store.record_decisions(report.decisions)
        raise typer.Exit(code=1)

    typer.echo("")
    typer.secho(f"TEMA: {escolhido.term}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {escolhido.reason}")
    if escolhido.url:
        typer.echo(f"  {escolhido.url}")
    if escolhido.news_items:
        typer.echo(f"  {len(escolhido.news_items)} materias ja associadas pela fonte")

    candidatos = [d for d in report.top(top) if d.term != escolhido.term]
    if candidatos:
        typer.echo("\n  rotina do dia (distintos, sem repetir):")
        for i, d in enumerate(candidatos, start=2):
            typer.echo(f"    {i}. {d.term[:58]} (score {d.score:.3f})")

    vice = [d for d in report.by_verdict(Verdict.not_selected)][:4]
    if vice:
        typer.echo("\n  proximos colocados:")
        for d in vice:
            typer.echo(f"    {d.score:.3f}  {d.term[:58]}")

    if dry_run:
        typer.secho("\n--dry-run: nada gravado no ledger", fg=typer.colors.YELLOW)
    else:
        n = store.record_decisions(report.decisions)
        typer.echo(f"\n{n} decisoes gravadas no ledger")


@app.command()
def research(
    topic: str = typer.Option(
        "", "--topic", "-t",
        help="pesquisa este tema; sem isso, roda o curador. "
             "Tema explicito e atualizacao intencional.",
    ),
    url: list[str] = typer.Option(
        [], "--url", "-u", help="fonte explicita (repetivel); pula a descoberta"
    ),
    llm: str = typer.Option("", "--llm", help="gemini ou groq; padrao vem do .env"),
    max_sources: int = typer.Option(0, "--sources", help="teto de fontes a ler"),
    dry_run: bool = typer.Option(False, "--dry-run", help="nao grava o dossie"),
) -> None:
    """Monta o dossie de um tema: 3-5 fontes, cada fato com URL e trecho conferidos."""
    from datetime import UTC, datetime

    from agent.adapters.llm_factory import build_llm
    from agent.memory.store import SignalStore
    from agent.models import Verdict
    from agent.ports.llm import LLMError
    from agent.research.fetch import PageFetcher
    from agent.research.researcher import Researcher
    from agent.research.sources import Candidate

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)
    teto = max_sources or settings.research_max_sources

    if topic:
        decision = Decision(
            term=topic, source="manual", verdict=Verdict.selected,
            reason="tema informado na linha de comando, sem passar pelo curador",
            score=0.0, niche_fit=0.0, decided_at=datetime.now(UTC),
        )
    else:
        decision = _curate_one(store)

    typer.echo(f"tema      : {decision.term}")
    typer.echo(f"origem    : {decision.source}")

    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model}")

    candidatos = (
        [Candidate(url=u, origin="manual") for u in url] if url else None
    )
    pesquisador = Researcher(
        modelo,
        fetcher=PageFetcher(max_chars=settings.research_page_chars),
        max_sources=teto,
        max_facts_per_source=settings.research_max_facts_per_source,
    )

    typer.echo("lendo fontes (uma chamada de modelo por fonte)...")
    report = pesquisador.research(decision, candidates=candidatos)

    for onde, motivo in report.failures.items():
        typer.secho(f"[fonte fora] {onde[:70]}: {motivo}", fg=typer.colors.YELLOW)

    if report.pages:
        typer.echo("")
        typer.echo("fontes lidas:")
        for pagina in report.pages:
            typer.echo(f"  {pagina.source_name:<22} {len(pagina.text):>6} car  {pagina.url[:70]}")

    # Descartado vem antes do dossie de proposito: e a parte que se perde se
    # ninguem olhar, e e o que diz se o portao esta calibrado ou estrangulando.
    if report.discarded:
        typer.echo("")
        typer.secho(f"{len(report.discarded)} fato(s) descartado(s) pelos portoes:",
                    fg=typer.colors.YELLOW)
        for d in report.discarded:
            typer.secho(f"  {d.claim[:76]}", fg=typer.colors.YELLOW)
            typer.secho(f"      {d.reason}", fg=typer.colors.BRIGHT_BLACK)

    custo = report.usage
    typer.echo("")
    typer.echo(f"custo     : {custo.input_tokens} tokens de entrada, "
               f"{custo.output_tokens} de saida, {report.latency_s}s de modelo")

    if not report.ok:
        typer.secho("nenhum fato com fonte sobreviveu; sem dossie", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    dossier = report.dossier
    typer.echo("")
    typer.secho(f"DOSSIE: {len(dossier.facts)} fatos de {report.source_count} fonte(s)",
                fg=typer.colors.GREEN, bold=True)
    for f in dossier.facts:
        typer.echo("")
        typer.secho(f"  {f.claim}", bold=True)
        typer.echo(f"      fonte : {f.source_name} — {f.source_url}")
        if f.quote:
            typer.secho(f'      trecho: "{f.quote[:100]}"', fg=typer.colors.BRIGHT_BLACK)

    if dry_run:
        typer.secho("\n--dry-run: dossie nao gravado", fg=typer.colors.YELLOW)
        return

    linha = store.record_dossier(
        dossier,
        model=modelo.model,
        provider=modelo.provider,
        usage=(custo.input_tokens, custo.output_tokens),
        latency_s=report.latency_s,
        source_count=report.source_count,
        discarded=[vars(d) for d in report.discarded],
        failures=report.failures,
    )
    typer.echo(f"\ndossie #{linha} gravado ({store.dossier_count()} na memoria)")


def _curate_one(store: Any) -> Decision:
    """Roda radar + curador e devolve o tema do dia, ou sai com erro."""
    from agent.curator.curator import Curator
    from agent.radar.collector import Radar, default_sources

    coleta = Radar(default_sources(), store).collect()
    for nome, erro in coleta.failures.items():
        typer.secho(f"[fonte fora] {nome}: {erro}", fg=typer.colors.YELLOW)
    if not coleta.signals:
        typer.secho("nenhum sinal coletado", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    report = Curator().curate(coleta.signals, ledger=store.recent_topics())
    store.record_decisions(report.decisions)
    escolhido = report.selected
    if escolhido is None:
        typer.secho("nenhum tema elegivel hoje", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return escolhido


@app.command()
def write(
    topic: str = typer.Option("", "--topic", "-t", help="tema; sem isso, usa o ultimo dossie"),
    llm: str = typer.Option("", "--llm", help="gemini ou groq; padrao vem do .env"),
    out: Path = typer.Option(None, "--out", "-o", help="grava o roteiro em JSON para render"),
    dry_run: bool = typer.Option(False, "--dry-run", help="nao grava na memoria"),
    mode: str = typer.Option("long", "--mode",
                               help="long (60-90s), short (~15s) ou carousel"),
    polish: bool = typer.Option(True, "--polish/--no-polish",
                                help="humanizacao apos o aceite mecanico"),
) -> None:
    """Escreve o roteiro a partir de um dossie ja gravado."""
    from agent.adapters.llm_factory import build_llm
    from agent.memory.store import SignalStore
    from agent.ports.llm import LLMError
    from agent.writer.writer import Screenwriter

    if mode not in ("long", "short", "carousel"):
        typer.secho(f"modo {mode!r} desconhecido; use long, short ou carousel.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)

    dossier = store.latest_dossier(topic or None)
    if dossier is None:
        alvo = f" para o tema {topic!r}" if topic else ""
        typer.secho(
            f"nenhum dossie{alvo} na memoria. Rode `uv run agent research` primeiro.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    typer.echo(f"tema      : {dossier.topic}")
    typer.echo(f"dossie    : {len(dossier.facts)} fatos de "
               f"{len(dossier.source_urls)} fonte(s)")

    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model} (modo {mode})")

    if mode == "carousel":
        _write_carousel_cmd(store, dossier, modelo, out, dry_run)
        return

    typer.echo("escrevendo (corrige sozinho o que e mecanico)...")
    try:
        report = Screenwriter(modelo).write(dossier, mode=mode, polish=polish)
    except LLMError as exc:
        typer.secho(f"falha do provedor: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if report.refusal:
        typer.secho(f"\n{report.refusal}", fg=typer.colors.RED)
        typer.echo("nenhuma chamada de modelo foi feita; custo zero")
        raise typer.Exit(code=1)

    # As tentativas corrigidas aparecem mesmo no sucesso: se toda execucao gasta
    # duas rodadas no mesmo defeito, o prompt e que esta fraco.
    for i, tentativa in enumerate(report.attempts, start=1):
        if tentativa.violations:
            typer.secho(f"tentativa {i} reprovada ({tentativa.word_count} palavras):",
                        fg=typer.colors.YELLOW)
            for v in tentativa.violations:
                typer.secho(f"  - {v}", fg=typer.colors.BRIGHT_BLACK)

    if report.humanized:
        typer.secho("humanizacao aplicada "
                    f"({'; '.join(report.humanize_notes)})", fg=typer.colors.GREEN)
    elif report.humanize_notes:
        typer.secho(f"humanizacao mantida no original: "
                    f"{'; '.join(report.humanize_notes)}", fg=typer.colors.YELLOW)

    custo = report.usage
    typer.echo(f"custo     : {custo.input_tokens} tokens de entrada, "
                f"{custo.output_tokens} de saida, {report.latency_s}s de modelo, "
                f"{len(report.attempts)} tentativa(s)")

    if not report.ok:
        typer.secho("\nnenhum roteiro passou nos portoes mecanicos", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    script = report.script
    typer.echo("")
    typer.secho(f"ROTEIRO: {script.word_count} palavras "
                f"(~{script.estimated_duration_s:.0f}s estimados)",
                fg=typer.colors.GREEN, bold=True)
    typer.echo("")
    typer.secho("  HOOK", bold=True)
    typer.echo(f"  {script.hook}")
    typer.echo("")
    typer.secho("  CORPO", bold=True)
    for paragrafo in script.body.split("\n"):
        if paragrafo.strip():
            typer.echo(f"  {paragrafo.strip()}")
    typer.echo("")
    typer.secho("  FECHAMENTO", bold=True)
    typer.echo(f"  {script.closing}")
    typer.echo("")
    typer.echo(f"  termos: {', '.join(script.search_terms)}")
    typer.echo(f"  fatos : {len(script.facts)} do dossie")

    if out is not None:
        out.write_text(
            script.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )
        typer.echo(f"\ngravado em {out}")
        typer.echo(f"renderize com: uv run agent render --script {out}")

    if dry_run:
        typer.secho("\n--dry-run: roteiro nao gravado na memoria", fg=typer.colors.YELLOW)
        return

    linha = store.record_script(
        script,
        model=modelo.model,
        provider=modelo.provider,
        usage=(custo.input_tokens, custo.output_tokens),
        latency_s=report.latency_s,
        attempts=[
            {"violations": a.violations, "word_count": a.word_count} for a in report.attempts
        ],
        dossier_id=store.latest_dossier_id(dossier.topic),
    )
    typer.echo(f"\nroteiro #{linha} gravado ({store.script_count()} na memoria)")


def _write_carousel_cmd(store: Any, dossier: Any, modelo: Any,
                        out: Path | None, dry_run: bool) -> None:
    """Ramo carrossel do `write`: 5 slides + legenda, sem juiz aqui."""
    from agent.ports.llm import LLMError
    from agent.writer.carousel import write_carousel

    typer.echo("escrevendo carrossel (5 slides, portoes proprios)...")
    try:
        report = write_carousel(dossier, modelo)
    except LLMError as exc:
        typer.secho(f"falha do provedor: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if report.refusal:
        typer.secho(f"\n{report.refusal}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    for i, tentativa in enumerate(report.attempts, start=1):
        if tentativa.violations:
            typer.secho(f"tentativa {i} reprovada:", fg=typer.colors.YELLOW)
            for v in tentativa.violations:
                typer.secho(f"  - {v}", fg=typer.colors.BRIGHT_BLACK)

    custo = report.usage
    typer.echo(f"custo     : {custo.input_tokens} tokens de entrada, "
               f"{custo.output_tokens} de saida, {report.latency_s}s de modelo")

    if not report.ok or report.carousel is None:
        typer.secho("\nnenhum carrossel passou nos portoes mecanicos",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    for s in report.carousel.slides:
        typer.echo(f"\n  [{s.n}/5] {s.headline}\n  {s.text}\n  ~ {s.visual}")
    typer.echo(f"\n  legenda: {report.carousel.caption}")

    if out is not None:
        out.write_text(
            report.carousel.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )
        typer.echo(f"\ngravado em {out}")
        typer.echo(f"julque com: uv run agent judge --script {out}")
        typer.echo(f"slides com: uv run agent carousel-render --carousel {out}")

    if dry_run:
        typer.secho("\n--dry-run: carrossel nao gravado na memoria",
                    fg=typer.colors.YELLOW)
        return
    linha = store.record_carousel(
        report.carousel, model=modelo.model, provider=modelo.provider,
        usage=(custo.input_tokens, custo.output_tokens),
        latency_s=report.latency_s, attempts=[a.violations for a in report.attempts],
    )
    typer.echo(f"\ncarrossel #{linha} gravado")


def _is_carousel_file(path: Path) -> bool:
    try:
        return "slides" in json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False


def _judge_carousel_cmd(store: Any, path: Path, llm: str, dry_run: bool) -> None:
    """Parecer de carrossel (rubrica propria, corte 8/10)."""
    from datetime import UTC, datetime

    from agent.adapters.llm_factory import build_llm
    from agent.judge.carousel import judge_carousel
    from agent.models import Carousel, CarouselReview, Dossier
    from agent.ports.llm import LLMError

    try:
        carrossel = Carousel.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        typer.secho(f"carrossel invalido: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    dossier = Dossier(topic=carrossel.topic, facts=carrossel.facts,
                      collected_at=datetime.now(UTC))
    if not dossier.facts:
        typer.secho("carrossel sem nenhum fato: nao ha dossie contra o que julgar",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    typer.echo(f"tema      : {carrossel.topic}")
    typer.echo(f"carrossel : {len(carrossel.slides)} slides")
    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model}")
    try:
        report = judge_carousel(carrossel, dossier, modelo)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    review: CarouselReview | None = report.review
    assert review is not None
    for s in review.scores:
        cor = (typer.colors.GREEN if s.score == 2
               else typer.colors.YELLOW if s.score == 1 else typer.colors.RED)
        typer.secho(f"  [{s.score}/2] {s.criterion.value}: {s.reason}", fg=cor)
    estado = "APROVADO" if review.approved else "REPROVADO"
    typer.secho(f"\n{estado}: {review.total}/10 (corte 8, nenhum zerado)",
                fg=typer.colors.GREEN if review.approved else typer.colors.RED)
    typer.echo(f"custo     : {report.usage.total_tokens} tokens, {report.latency_s}s")

    if not dry_run:
        linha = store.record_carousel(
            carrossel, model=modelo.model, provider=modelo.provider,
            usage=(report.usage.input_tokens, report.usage.output_tokens),
            latency_s=report.latency_s, attempts=[],
            review=review,
        )
        typer.echo(f"parecer de carrossel gravado (carrossel #{linha})")
    raise typer.Exit(code=0 if review.approved else 1)


def _mostrar_parecer(review) -> None:
    """Imprime a rubrica inteira, critério a critério.

    Sempre inteira, inclusive os criterios que tiraram 2: parecer resumido em
    "reprovado (9/14)" nao diz o que mudar, e e o motivo por criterio que volta
    ao roteirista na revisao.
    """
    from agent.models import RUBRIC_CUTOFF, RUBRIC_MAX

    typer.echo("")
    for s in review.scores:
        cor = (typer.colors.GREEN if s.score == 2
               else typer.colors.YELLOW if s.score == 1 else typer.colors.RED)
        # "julgado" para nota que nao foi julgada seria mentira no proprio
        # relatorio que existe para dar para auditar.
        marca = "pulado " if not s.evaluated else "medido " if s.measured else "julgado"
        typer.secho(f"  [{s.score}/2] {marca}  {s.criterion.value}", fg=cor)
        typer.secho(f"          {s.reason}", fg=typer.colors.BRIGHT_BLACK)

    typer.echo("")
    if review.approved:
        typer.secho(f"APROVADO: {review.total}/{RUBRIC_MAX} "
                    f"(corte {RUBRIC_CUTOFF}, nenhum criterio zerado)",
                    fg=typer.colors.GREEN, bold=True)
        return

    typer.secho(f"REPROVADO: {review.total}/{RUBRIC_MAX} (corte {RUBRIC_CUTOFF})",
                fg=typer.colors.RED, bold=True)
    for s in review.vetoed:
        typer.secho(f"  veto em {s.criterion.value}: e requisito, nao qualidade — "
                    "nota nos outros criterios nao compensa", fg=typer.colors.RED)
    for s in review.zeroed:
        if s not in review.vetoed:
            typer.secho(f"  zerado em {s.criterion.value}: criterio zerado reprova "
                        "mesmo com a soma no corte", fg=typer.colors.RED)


@app.command()
def judge(
    topic: str = typer.Option("", "--topic", "-t", help="tema; sem isso, usa o ultimo roteiro"),
    script_path: Path = typer.Option(
        None, "--script", "-s", exists=True, readable=True,
        help="julga este arquivo em vez do ultimo roteiro da memoria",
    ),
    llm: str = typer.Option("", "--llm", help="gemini ou groq; padrao vem do .env"),
    dry_run: bool = typer.Option(False, "--dry-run", help="nao grava o parecer"),
) -> None:
    """Aplica a rubrica a um roteiro (video) ou carrossel (detecta pelo arquivo)."""
    from datetime import UTC, datetime

    from agent.adapters.llm_factory import build_llm
    from agent.judge.judge import Judge, review_measured_only
    from agent.memory.store import SignalStore
    from agent.models import Dossier
    from agent.ports.llm import LLMError

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)

    if script_path is not None and _is_carousel_file(script_path):
        _judge_carousel_cmd(store, script_path, llm, dry_run)
        return

    if script_path is not None:
        script = _load_script(script_path)
        # O Script carrega os fatos que o roteirista usou, então um arquivo se
        # autojulga: e o que permite rodar a fixture adversarial sem a memoria.
        dossier = Dossier(
            topic=script.topic, facts=script.facts, collected_at=datetime.now(UTC)
        )
    else:
        script = store.latest_script(topic or None)
        if script is None:
            typer.secho("nenhum roteiro na memoria. Rode `uv run agent write` primeiro.",
                        fg=typer.colors.RED)
            raise typer.Exit(code=2)
        dossier = store.latest_dossier(script.topic) or Dossier(
            topic=script.topic, facts=script.facts, collected_at=datetime.now(UTC)
        )

    if not dossier.facts:
        typer.secho("roteiro sem nenhum fato: nao ha dossie contra o que julgar",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    typer.echo(f"tema      : {script.topic}")
    typer.echo(f"roteiro   : {script.word_count} palavras, "
               f"{len(script.facts)} fatos, ~{script.estimated_duration_s:.0f}s")

    # Medida primeiro: se o roteiro ja reprova num criterio de requisito, nao ha
    # motivo para exigir chave de API para confirmar isso.
    antecipado = review_measured_only(script, dossier)
    if antecipado is not None:
        typer.echo("modelo    : nao consultado (reprovou na medida)")
        _mostrar_parecer(antecipado)
        typer.echo("\ncusto     : 0 tokens")
        if not dry_run:
            linha = store.record_review(antecipado, usage=(0, 0), latency_s=0.0)
            typer.echo(f"parecer #{linha} gravado")
        raise typer.Exit(code=1)

    try:
        modelo = build_llm(llm or None)
        typer.echo(f"modelo    : {modelo.provider}/{modelo.model}")
        report = Judge(modelo).review(script, dossier)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    _mostrar_parecer(report.review)
    typer.echo(f"\ncusto     : {report.usage.total_tokens} tokens, {report.latency_s}s")

    if not dry_run:
        linha = store.record_review(
            report.review,
            usage=(report.usage.input_tokens, report.usage.output_tokens),
            latency_s=report.latency_s,
        )
        typer.echo(f"parecer #{linha} gravado")

    raise typer.Exit(code=0 if report.approved else 1)


@app.command()
def produce(
    topic: str = typer.Option("", "--topic", "-t", help="tema; sem isso, usa o ultimo dossie"),
    out: Path = typer.Option(None, "--out", "-o", help="grava o roteiro aprovado em JSON"),
    revisions: int = typer.Option(2, "--revisions", help="teto de rodadas de revisao"),
    llm: str = typer.Option("", "--llm", help="gemini ou groq; padrao vem do .env"),
    dry_run: bool = typer.Option(False, "--dry-run", help="nao grava na memoria"),
    mode: str = typer.Option("long", "--mode", help="long, short ou carousel"),
) -> None:
    """Escreve, julga e revisa ate o roteiro passar na rubrica ou estourar as rodadas."""
    from agent.adapters.llm_factory import build_llm
    from agent.judge.judge import Judge
    from agent.memory.store import SignalStore
    from agent.pipeline import produce as rodar
    from agent.ports.llm import LLMError
    from agent.writer.writer import Screenwriter

    if mode not in ("long", "short", "carousel"):
        typer.secho(f"modo {mode!r} desconhecido.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)

    dossier = store.latest_dossier(topic or None)
    if dossier is None:
        typer.secho("nenhum dossie na memoria. Rode `uv run agent research` primeiro.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if mode == "carousel":
        _produce_carousel_cmd(store, dossier, llm, out, revisions, dry_run)
        return

    typer.echo(f"tema      : {dossier.topic}")
    typer.echo(f"dossie    : {len(dossier.facts)} fatos de "
               f"{len(dossier.source_urls)} fonte(s)")
    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model} (modo {mode})")
    typer.echo(f"ate {revisions + 1} rodada(s) de roteiro + parecer...")

    report = rodar(dossier, Screenwriter(modelo), Judge(modelo),
                   max_revisions=revisions, mode=mode)
    if report.failure:
        typer.secho(f"falha no meio do laco — {report.failure}", fg=typer.colors.RED)

    for i, rodada in enumerate(report.rounds, start=1):
        typer.echo("")
        typer.secho(f"--- rodada {i} ---", bold=True)
        if rodada.write is not None and rodada.write.refusal:
            typer.secho(f"roteirista recusou: {rodada.write.refusal}", fg=typer.colors.RED)
        elif rodada.write is not None:
            tentativas = len(rodada.write.attempts)
            typer.echo(f"roteirista: {tentativas} tentativa(s) mecanica(s)")
            for t in rodada.write.attempts:
                for v in t.violations:
                    typer.secho(f"  - {v}", fg=typer.colors.BRIGHT_BLACK)
        if rodada.review is not None and rodada.review.review is not None:
            _mostrar_parecer(rodada.review.review)

    custo = report.usage
    typer.echo("")
    typer.echo(f"custo total: {custo.input_tokens} tokens de entrada, "
               f"{custo.output_tokens} de saida, {report.latency_s}s de modelo")

    if not report.approved:
        typer.secho("nenhum roteiro aprovado nas rodadas disponiveis; "
                    "o motivo de cada reprovacao esta acima", fg=typer.colors.RED)
        if not dry_run and report.script is not None and report.review is not None:
            _gravar(store, report, dossier)
        raise typer.Exit(code=1)

    script = report.script
    typer.echo("")
    typer.secho(f"ROTEIRO APROVADO: {script.word_count} palavras "
                f"(~{script.estimated_duration_s:.0f}s)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {script.hook}")
    typer.echo(f"  termos: {', '.join(script.search_terms)}")

    if out is not None:
        out.write_text(
            script.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )
        typer.echo(f"\ngravado em {out}")
        typer.echo(f"renderize com: uv run agent render --script {out}")

    if dry_run:
        typer.secho("\n--dry-run: nada gravado na memoria", fg=typer.colors.YELLOW)
        return
    _gravar(store, report, dossier)


def _produce_carousel_cmd(store: Any, dossier: Any, llm: str, out: Path | None,
                          revisions: int, dry_run: bool) -> None:
    """Laco de carrossel: escreve, julga, revisa ate passar (corte 8/10)."""
    from agent.adapters.llm_factory import build_llm
    from agent.pipeline import produce_carousel as rodar
    from agent.ports.llm import LLMError

    typer.echo(f"tema      : {dossier.topic}")
    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model} (modo carousel)")

    report = rodar(dossier, modelo, max_revisions=revisions)
    if report.failure:
        typer.secho(f"falha no meio do laco — {report.failure}", fg=typer.colors.RED)
    for i, rodada in enumerate(report.rounds, start=1):
        typer.echo("")
        typer.secho(f"--- rodada {i} ---", bold=True)
        if rodada.write is not None:
            for t in rodada.write.attempts:
                for v in t.violations:
                    typer.secho(f"  - {v}", fg=typer.colors.BRIGHT_BLACK)
        if rodada.review is not None and rodada.review.review is not None:
            for s in rodada.review.review.scores:
                typer.echo(f"  [{s.score}/2] {s.criterion.value}: {s.reason}")

    custo = report.usage
    typer.echo(f"\ncusto total: {custo.input_tokens} in, {custo.output_tokens} out, "
               f"{report.latency_s}s")
    if not report.approved:
        typer.secho("nenhum carrossel aprovado nas rodadas.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    carrossel = report.carousel
    assert carrossel is not None
    typer.secho("CARROSSEL APROVADO", fg=typer.colors.GREEN, bold=True)
    if out is not None:
        out.write_text(
            carrossel.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8")
        typer.echo(f"gravado em {out}")
        typer.echo(f"slides com: uv run agent carousel-render --carousel {out}")
    if dry_run:
        typer.secho("\n--dry-run: nada gravado na memoria", fg=typer.colors.YELLOW)
        return
    linha = store.record_carousel(
        carrossel, model=modelo.model, provider=modelo.provider,
        usage=(custo.input_tokens, custo.output_tokens),
        latency_s=report.latency_s,
        attempts=[a.violations for r in report.rounds if r.write for a in r.write.attempts],
        review=report.review,
    )
    typer.echo(f"carrossel #{linha} gravado")


def _gravar(store: Any, report: Any, dossier: Any) -> None:
    """Grava roteiro e parecer ligados, inclusive quando reprovado.

    Reprovado tambem entra: e o registro de em que critério a rubrica bate com
    mais frequencia, e sem ele calibrar a rubrica seria chute.
    """
    ultima = report.rounds[-1]
    script_id = store.record_script(
        report.script,
        model=ultima.write.model,
        provider=ultima.write.provider,
        usage=(report.usage.input_tokens, report.usage.output_tokens),
        latency_s=report.latency_s,
        attempts=[
            {"violations": t.violations, "word_count": t.word_count}
            for rodada in report.rounds if rodada.write is not None
            for t in rodada.write.attempts
        ],
        dossier_id=store.latest_dossier_id(dossier.topic),
    )
    review_id = store.record_review(
        report.review,
        usage=(report.usage.input_tokens, report.usage.output_tokens),
        latency_s=report.latency_s,
        script_id=script_id,
    )
    typer.echo(f"\nroteiro #{script_id} e parecer #{review_id} gravados")


@app.command("llm-health")
def llm_health() -> None:
    """Confere se as chaves e os ids de modelo configurados ainda respondem.

    Id de modelo de free tier e descontinuado sem aviso, e o erro aparece no meio
    de uma pesquisa, depois de gastar tempo lendo paginas. Este comando gasta uma
    chamada minuscula e verifica o artefato: quem respondeu, em quanto tempo,
    cobrando quantos tokens.
    """
    from agent.adapters.llm_factory import build_llm, configured
    from agent.ports.llm import LLMError, parse_json_object

    provedores = configured()
    if not provedores:
        typer.secho(
            "nenhuma chave de LLM configurada. A principal e paga com credito:\n"
            "  AGENT_OPENROUTER_API_KEY -> openrouter.ai/keys\n"
            "As reservas sao gratuitas:\n"
            "  AGENT_GROQ_API_KEY      -> console.groq.com/keys\n"
            "  AGENT_GEMINI_API_KEY    -> aistudio.google.com/apikey\n"
            "Grave no .env (git-ignored).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    ok = True
    for nome in provedores:
        try:
            modelo = build_llm(nome)
            resposta = modelo.complete(
                'Responda {"ok": "sim"} e nada mais.',
                schema={"type": "object", "properties": {"ok": {"type": "string"}},
                        "required": ["ok"]},
                max_output_tokens=256,
            )
            parse_json_object(resposta.text)
        except LLMError as exc:
            typer.secho(f"[FALHA] {nome}: {exc}", fg=typer.colors.RED)
            ok = False
            continue
        typer.secho(
            f"[OK  ] {nome}: {resposta.model} respondeu em {resposta.latency_s}s "
            f"({resposta.usage.total_tokens} tokens)",
            fg=typer.colors.GREEN,
        )

    if not ok:
        raise typer.Exit(code=1)


@app.command()
def health() -> None:
    """Verifica se o renderizador configurado esta de pe (ffmpeg ou MPT)."""
    if settings.renderer == "ffmpeg":
        from agent.adapters.ffmpeg_renderer import FfmpegRenderer
        from agent.render.post import ffmpeg_bin

        alive = FfmpegRenderer().health()
        typer.secho(f"ffmpeg ({ffmpeg_bin()}): {'ok' if alive else 'sem subtitles/concat'}",
                    fg=typer.colors.GREEN if alive else typer.colors.RED)
        mpt = MptRenderer().health()
        typer.echo(f"reserva MPT em {settings.renderer_url}: {'ok' if mpt else 'fora do ar'}")
        sys.exit(0 if alive else 1)
    alive = MptRenderer().health()
    typer.secho(
        f"{settings.renderer_url}: {'ok' if alive else 'fora do ar'}",
        fg=typer.colors.GREEN if alive else typer.colors.RED,
    )
    sys.exit(0 if alive else 1)


@app.command()
def publish(
    video: Path = typer.Option(..., "--video", "-v", exists=True, readable=True),
) -> None:
    """Sobe um MP4 para a inbox do TikTok (escopo video.upload, sem auditoria).

    O endpoint inbox so recebe os bytes: titulo, descricao e rotulo AIGC sao
    aplicados por voce no app, ao concluir o post pela notificacao da inbox.
    Por isso este comando termina com o checklist manual -- e o rotulo AIGC
    nao tem flag para desligar porque nao e opcional.
    """
    from agent.adapters.tiktok_publisher import TikTokPublisher
    from agent.memory.store import SignalStore
    from agent.models import PublishState

    token = settings.tiktok_access_token
    if not token:
        typer.secho(
            "sem AGENT_TIKTOK_ACCESS_TOKEN no .env (git-ignored).\n"
            "Registre o app em developers.tiktok.com, autorize o escopo "
            "video.upload e grave o token.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    typer.echo(f"video     : {video} ({video.stat().st_size} bytes)")
    typer.echo("subindo para a inbox (init + chunks)...")
    result = TikTokPublisher().upload(str(video), access_token=token)

    store = SignalStore(settings.db_path)
    store.record_post(
        result.publish_id or "",
        str(video),
        status=result.state.value,
        error=result.error,
    )

    if result.state is not PublishState.uploaded:
        typer.secho(f"subida falhou: {result.error}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"\npublish_id: {result.publish_id}", fg=typer.colors.GREEN)
    typer.echo("gravado em posts. Agora, no app do TikTok:")
    typer.echo("  1. abra a notificacao da inbox e conclua a edicao;")
    typer.echo("  2. LIGUE o rotulo de conteudo gerado por IA (obrigatorio);")
    typer.echo("  3. confira que a legenda cita as fontes do roteiro.")


@app.command("publish-status")
def publish_status(
    publish_id: str = typer.Option(..., "--publish-id"),
) -> None:
    """Consulta o estado de um post na API e atualiza a tabela posts."""
    from agent.adapters.tiktok_publisher import TikTokPublisher
    from agent.memory.store import SignalStore

    token = settings.tiktok_access_token
    if not token:
        typer.secho("sem AGENT_TIKTOK_ACCESS_TOKEN no .env.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    try:
        estado = TikTokPublisher().fetch_status(publish_id, access_token=token)
    except Exception as exc:
        typer.secho(f"falha: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    SignalStore(settings.db_path).update_post_status(publish_id, status=estado)
    typer.echo(f"{publish_id}: {estado}")


@app.command("tiktok-auth-url")
def tiktok_auth_url(
    state: str = typer.Option("", "--state"),
) -> None:
    """Imprime a URL para autorizar o app no navegador (escopo video.upload)."""
    from agent.adapters.tiktok_oauth import authorize_url

    if not settings.tiktok_client_key or not settings.tiktok_redirect_uri:
        typer.secho(
            "configure AGENT_TIKTOK_CLIENT_KEY e AGENT_TIKTOK_REDIRECT_URI no .env.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    typer.echo(authorize_url(
        settings.tiktok_client_key, settings.tiktok_redirect_uri, state=state,
    ))


@app.command()
def eval(
    topic: str = typer.Option("", "--topic", "-t", help="tema; sem isso, agrega tudo"),
) -> None:
    """Tabula escritor x juiz com custo medido, sem chamar modelo nenhum.

    Le roteiros e pareceres ja gravados e imprime: agregado por escritor, por
    juiz (parecer interrompido fora da media), matriz pareada no mesmo roteiro,
    custo total e ultima metrica de cada post. E relatorio, nao portao: sai 0
    mesmo com a base vazia.
    """
    from agent.eval.eval import (
        CarouselRow,
        ReviewRow,
        ScriptRow,
        build_report,
        format_text,
    )
    from agent.memory.store import SignalStore
    from agent.models import Review

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)

    scripts = [ScriptRow(**r) for r in store.list_scripts(topic or None)]
    reviews = []
    for r in store.list_reviews(topic or None):
        try:
            review = Review.model_validate_json(r["review_json"])
        except ValueError as exc:
            typer.secho(f"parecer #{r['id']} com JSON invalido: {exc}",
                        fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        reviews.append(ReviewRow(
            id=r["id"], topic=r["topic"], script_id=r["script_id"],
            model=r["model"], provider=r["provider"], review=review,
            input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
            latency_s=r["latency_s"],
        ))

    if not scripts and not reviews and not store.list_carousels(topic or None):
        typer.echo("nada a agregar: sem roteiros nem pareceres na memoria.")
        return

    carousels = [CarouselRow(
        id=r["id"], topic=r["topic"], model=r["model"], provider=r["provider"],
        approved=bool(r["approved"]), input_tokens=r["input_tokens"],
        output_tokens=r["output_tokens"], latency_s=r["latency_s"],
    ) for r in store.list_carousels(topic or None)]
    typer.echo(format_text(build_report(scripts, reviews, carousels)), nl=False)

    vistos: set[str] = set()
    with store._conn() as conn:
        posts = [dict(r) for r in conn.execute(
            "SELECT publish_id, video_path, status FROM posts"
            " ORDER BY created_at DESC, id DESC").fetchall()]
    if posts:
        typer.echo("posts (ultima metrica lida no app; a API da inbox nao expoe):")
        for p in posts:
            if p["publish_id"] in vistos:
                continue
            vistos.add(p["publish_id"])
            m = store.latest_metric(p["publish_id"])
            if m is None:
                typer.echo(f"  {p['publish_id']}: {p['status']} (sem metrica)")
            else:
                acao = "".join(
                    f" {k}={m[k]}" for k in ("saves", "comments", "shares")
                    if m.get(k) is not None)
                typer.echo(
                    f"  {p['publish_id']}: {p['status']} "
                    f"views={m['views']} "
                    f"watch~{m['avg_watch_s']}s "
                    f"completion={m['completion_rate']}{acao}"
                )


@app.command("metrics-record")
def metrics_record(
    publish_id: str = typer.Option(..., "--publish-id"),
    views: int = typer.Option(..., "--views", min=0),
    avg_watch: float = typer.Option(None, "--avg-watch", min=0.0,
                                    help="tempo medio de exibicao em segundos"),
    completion: float = typer.Option(None, "--completion", min=0.0, max=1.0,
                                     help="fracao 0..1 que assistiu ate o fim"),
    script_id: int = typer.Option(None, "--script-id",
                                  help="roteiro que gerou o video (fecha o loop)"),
    saves: int = typer.Option(None, "--saves", min=0,
                              help="salvamentos (placar do carrossel)"),
    comments: int = typer.Option(None, "--comments", min=0),
    shares: int = typer.Option(None, "--shares", min=0),
) -> None:
    """Grava uma coleta de metricas lida no app (views, watch, completion).

    Manual de proposito: no escopo video.upload da inbox nao ha endpoint de
    metricas, e a Research API e restrita a pesquisa academica. Cada coleta
    entra na serie do publish_id -- a curva, nao o numero isolado, e o sinal.
    saves/comments/shares sao o placar do carrossel e o desempate do video.
    """
    from agent.memory.store import SignalStore

    settings.ensure_dirs()
    try:
        linha = SignalStore(settings.db_path).record_metric(
            publish_id, views, script_id=script_id,
            avg_watch_s=avg_watch, completion_rate=completion,
            saves=saves, comments=comments, shares=shares,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"metrica #{linha} gravada para {publish_id}")


@app.command("carousel-render")
def carousel_render(
    carousel: Path = typer.Option(..., "--carousel", exists=True, readable=True),
    out_dir: Path = typer.Option(None, "--out-dir",
                                 help="pasta dos slides; padrao: output/carrossel-<ts>"),
    pillar: str = typer.Option("", "--pillar",
                               help="news|fato|analise|tutorial|futuro|vs; vazio sugere pelo tema"),
    fotos: bool = typer.Option(False, "--fotos/--sem-fotos",
                               help="fundo com foto Pexels pela tag do slide"),
) -> None:
    """Renderiza os 5 slides 1080x1920 do carrossel + caption.txt, local."""
    from datetime import datetime

    from agent.brand.brand import suggest_content_pillar
    from agent.models import Carousel as CarouselModel
    from agent.render.carousel import render_carousel
    from agent.render.photos import fetch as fetch_photo

    try:
        modelo = CarouselModel.model_validate_json(carousel.read_text(encoding="utf-8"))
    except ValueError as exc:
        typer.secho(f"carrossel invalido: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    pilar = pillar or suggest_content_pillar(modelo.topic)
    destino = out_dir or settings.output_dir / (
        f"carrossel-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    typer.echo(f"pilar     : {pilar}")
    fotos_map: dict[int, Path | None] = {}
    if fotos:
        cache = settings.output_dir / "fotos-cache"
        for s in modelo.slides:
            caminho = fetch_photo(s.visual, cache)
            fotos_map[s.n] = caminho
            typer.echo(f"  slide {s.n}: {s.visual} -> "
                       + (caminho.name if caminho else "sem foto (layout puro)"))
    slides = render_carousel(modelo, destino, pilar, fotos_map or None)
    for s in slides:
        typer.echo(f"  {s} ({s.stat().st_size} bytes)")
    typer.echo(f"legenda em {destino / 'caption.txt'}")
    _checar_slides(slides)


def _checar_slides(slides: list[Path]) -> None:
    """Aceite do carrossel: 5 PNG 1080x1920 com titulo legivel medido."""
    from PIL import Image

    from agent.render.carousel import ink_height, legible

    ok = len(slides) == 5
    for s in slides:
        with Image.open(s) as img:
            ok = ok and img.size == (1080, 1920) and s.stat().st_size > 0
        if not legible(s):
            typer.secho(f"[FALHA] {s.name}: titulo com {ink_height(s)}px de tinta "
                        "(ilegivel)", fg=typer.colors.RED)
            ok = False
    (typer.secho("[OK  ] 5 slides 1080x1920, titulo legivel", fg=typer.colors.GREEN)
     if ok else typer.secho("[FALHA] slides fora do aceite", fg=typer.colors.RED))
    if not ok:
        raise typer.Exit(code=1)


@app.command("voice-list")
def voice_list() -> None:
    """Biblioteca de vozes e estilos (tudo local, $0)."""
    from agent.brand.brand import load as load_brand
    from agent.voice.library import REJEITADAS, STYLES, VOICES

    typer.secho("VOZES (locutor real, um por modelo):", bold=True)
    for v in VOICES.values():
        typer.echo(f"  {v.id}: {v.nome} | {v.genero}, {v.idade_aparente} | "
                   f"{v.timbre} | licenca: {v.licenca}")
    typer.secho("\nESTILOS (interpretacao sobre os timbres):", bold=True)
    for s in STYLES.values():
        typer.echo(f"  {s.id}: voz={s.voz} vel={s.velocidade} "
                   f"pausa={s.pausa_frase_s}s -- {s.descricao}")
    brand = load_brand()
    typer.secho("\nAPRESENTADORES (elenco, nao marca):", bold=True)
    for p in brand.presenters.values():
        voz = p.library_voice or "SEM VOZ ABERTA (gap: feminina pt-BR nao existe)"
        typer.echo(f"  {p.id} ({p.name}): {p.role} | seed={p.seed} | voz={voz}")
    typer.echo("  formatos com avatar: "
               + ", ".join(brand.presenter_formats)
               + " -- fora deles, sem avatar")
    typer.secho("\nFora, com motivo:", bold=True)
    for k, motivo in REJEITADAS.items():
        typer.echo(f"  {k}: {motivo}")


@app.command("voice-fetch")
def voice_fetch(
    voice: str = typer.Option("", "--voice", help="id; vazio baixa todas"),
) -> None:
    """Baixa modelos de voz para data/voices/ (git-ignored, ~63 MB cada)."""
    import urllib.request

    from agent.voice.library import VOICES

    settings.ensure_dirs()
    destino = settings.data_dir / "voices"
    destino.mkdir(parents=True, exist_ok=True)
    alvos = [VOICES[voice]] if voice else list(VOICES.values())
    if voice and voice not in VOICES:
        typer.secho(f"voz {voice!r} desconhecida; veja `voice-list`.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)
    for v in alvos:
        pares = [(v.modelo_url, v.arquivo),
                 (v.config_url or v.modelo_url + ".json", v.arquivo + ".json")]
        for url, nome in pares:
            caminho = destino / nome
            if caminho.exists():
                typer.echo(f"  {nome}: ja existe, pulando")
                continue
            typer.echo(f"  baixando {nome} (~63 MB)...")
            urllib.request.urlretrieve(url, caminho)
        typer.secho(f"[OK  ] {v.id}", fg=typer.colors.GREEN)


def _voice_engine(voice_id: str):
    """Adaptador Piper a partir de data/voices/. Falha explicando o fetch."""
    from agent.voice.engine import PiperTTS
    from agent.voice.library import VOICES

    if voice_id not in VOICES:
        raise ValueError(f"voz {voice_id!r} desconhecida; veja `voice-list`")
    modelo = settings.data_dir / "voices" / VOICES[voice_id].arquivo
    if not modelo.exists():
        raise ValueError(f"modelo {modelo} ausente; rode `voice-fetch --voice {voice_id}`")
    return PiperTTS(modelo)


@app.command("voice-say")
def voice_say(
    voice: str = typer.Option(..., "--voice"),
    text: str = typer.Option("", "--text"),
    text_file: Path = typer.Option(None, "--text-file"),
    out: Path = typer.Option(..., "--out", "-o"),
    speed: float = typer.Option(1.0, "--speed", min=0.5, max=2.0),
    noise: float = typer.Option(0.667, "--noise", min=0.0, max=1.5),
) -> None:
    """Um texto, uma voz, um wav. Amostra rapida antes de narrar roteiro."""
    import time

    from agent.voice.narrate import Narrator

    corpo = text or (text_file.read_text(encoding="utf-8") if text_file else "")
    if not corpo.strip():
        typer.secho("texto vazio; use --text ou --text-file.", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    try:
        motor = _voice_engine(voice)
    except (ValueError, Exception) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    inicio = time.monotonic()
    nar = Narrator(motor).narrate(corpo, speed=speed, noise=noise)
    out.parent.mkdir(parents=True, exist_ok=True)
    nar.write_wav(str(out))
    parede = time.monotonic() - inicio
    typer.echo(f"{out}: {nar.duration_s:.1f}s de audio em {parede:.1f}s "
               f"(RTF {parede / nar.duration_s:.2f}, {nar.utterances} falas)")


@app.command("voice-narrate")
def voice_narrate(
    script: Path = typer.Option(..., "--script", "-s", exists=True, readable=True),
    style: str = typer.Option("documental", "--style"),
    voice: str = typer.Option("", "--voice", help="troca o locutor do estilo"),
    out: Path = typer.Option(..., "--out", "-o"),
) -> None:
    """Roteiro (hook+body+closing) para wav com o estilo do canal."""
    import time

    from agent.models import Script
    from agent.voice.library import STYLES
    from agent.voice.narrate import Narrator

    if style not in STYLES:
        typer.secho(f"estilo {style!r} desconhecido; veja `voice-list`.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)
    perfil = STYLES[style]
    try:
        roteiro = Script.model_validate_json(script.read_text(encoding="utf-8"))
    except ValueError as exc:
        typer.secho(f"roteiro invalido: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    try:
        motor = _voice_engine(voice or perfil.voz)
    except (ValueError, Exception) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    inicio = time.monotonic()
    nar = Narrator(motor, pausa_frase_s=perfil.pausa_frase_s).narrate(
        roteiro.narration, speed=perfil.velocidade, noise=perfil.ruido)
    out.parent.mkdir(parents=True, exist_ok=True)
    nar.write_wav(str(out))
    parede = time.monotonic() - inicio
    typer.echo(f"{out}: {nar.duration_s:.1f}s em {parede:.1f}s "
               f"(RTF {parede / nar.duration_s:.2f}) estilo={style}")


@app.command("brand-avatar")
def brand_avatar(
    presenter: str = typer.Option(..., "--presenter", help="iris ou theo"),
    angulo: str = typer.Option("frontal", "--angulo"),
    expressao: str = typer.Option("neutra", "--expressao"),
    gesto: str = typer.Option("parada", "--gesto"),
) -> None:
    """Imprime o prompt travado de geracao do avatar + checklist de uso.

    A geracao e externa (sem image-gen local $0): este comando garante que o
    prompt sai inteiro e que as travas viajam junto -- seed, negativo,
    enquadramento e limites.
    """
    from agent.brand.brand import avatar_prompt
    from agent.brand.brand import load as load_brand

    brand = load_brand()
    if presenter not in brand.presenters:
        typer.secho("apresentador desconhecido; use iris ou theo.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)
    try:
        typer.echo(avatar_prompt(presenter, angulo=angulo, expressao=expressao,
                                 gesto=gesto))
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    p = brand.presenters[presenter]
    typer.echo("\n--- travas ---")
    typer.echo(f"seed: {p.seed} (fixo, sempre) | formatos: {', '.join(p.formats)}")
    typer.echo("enquadramento: 30-36% da altura, direita, peito p/ cima, "
               "fundo transparente, texto do lado oposto")
    typer.echo("encenacao: grande na chamada, canto no corpo, volta no fechamento; "
               "nunca em perfil/logo/capa; rotulo AIGC ligado")
    typer.echo("depois de gerar: salve em brand/assets/presenters/source/<id>.jpg e "
               "rode scripts/make_presenter_cutouts.py (recorte + pontos do rosto)")
    typer.echo("\n--- negativo ---")
    typer.echo(brand.negative_prompt)


@app.command("rerender")
def rerender(
    script_path: Path = typer.Option(..., "--script", "-s", exists=True, readable=True),
    out_dir: Path = typer.Option(Path("output/_rerender"), "--out-dir"),
    pillar: str = typer.Option("", "--pillar",
                               help="padrao: o pilar gravado no proprio roteiro"),
) -> None:
    """Refaz o VIDEO de um roteiro ja aprovado, sem gastar LLM nem pauta.

    Existe porque a coisa mais cara de testar uma mudanca de render e a parte
    que nao mudou: radar, pesquisa, roteirista e juiz gastam cota, disputam
    pauta com os slots do dia e podem simplesmente nao aprovar nada -- o dia
    20/09/2026 acabou com quatro temas tentados e nenhum aprovado, so porque
    as rodadas anteriores ja tinham consumido as boas pautas.

    Um roteiro que **ja passou pelo juiz** e material de producao legitimo.
    Daqui para a frente ele refaz exatamente o que mudou: clipes, narracao,
    apresentador, legenda, trilha e pos-producao, pelo mesmo caminho que o
    piloto usa (`SlotRunner.render_video`) -- nao e uma segunda implementacao
    que pode divergir da de producao.
    """
    from agent.autopilot.runner import SlotRunner

    script = _load_script(script_path)
    pilar = pillar or script.pillar or "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"tema     : {script.topic}")
    typer.echo(f"pilar    : {pilar} | formato: {script.format}")
    typer.echo(f"narracao : {script.word_count} palavras")

    runner = SlotRunner()
    try:
        video, medida = runner.render_video(script, out_dir, pilar)
    except (OSError, RuntimeError, ValueError) as exc:
        typer.secho(f"falha: {type(exc).__name__}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho(f"\nvideo: {video}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {medida['width']}x{medida['height']}, {medida['duration_s']}s, "
               f"audio={'sim' if medida['has_audio'] else 'NAO'}")


@app.command("presenter-preview")
def presenter_preview(
    script_path: Path = typer.Option(..., "--script", "-s", exists=True, readable=True),
    presenter: str = typer.Option("theo", "--presenter", help="iris ou theo"),
    out_dir: Path = typer.Option(Path("output/_apresentador"), "--out-dir"),
    seconds: float = typer.Option(0.0, "--seconds",
                                  help="so os N primeiros segundos (0 = tudo)"),
    still: bool = typer.Option(False, "--still",
                               help="forca o retrato parado, ignorando o clipe base"),
) -> None:
    """Sintetiza so a camada do apresentador e uma folha de contato para olhar.

    Serve para conferir o artefato sem gastar uma rodada de LLM nem um slot: a
    narracao sai do TTS (gratis), a camada sai em `apresentador.mp4` e um
    quadro de cada encenacao vira `folha.png`. Foi assim que a boca escancarada
    e a silhueta em retangulo apareceram -- olhando o pixel.

    Usa o clipe base (`<id>_base.mp4`) quando ele existe, que e o caminho de
    producao desde 20/09/2026; `--still` forca a reserva sintetizada.
    """
    import subprocess
    import time

    from PIL import Image

    from agent.brand.brand import load as load_brand
    from agent.config import PROJECT_ROOT
    from agent.render import presenter as pr
    from agent.render import presenter_video as pv
    from agent.render.post import ffmpeg_bin
    from agent.render.subtitles import align
    from agent.voice.edge import VOZ_FEMININA, VOZ_MASCULINA, synthesize
    from agent.voice.pronounce import respell

    pasta = PROJECT_ROOT / "brand" / "assets" / "presenters"
    clipe = pasta / f"{presenter}_base.json"
    png = pasta / f"{presenter}.png"
    usa_clipe = (not still and clipe.exists()
                 and clipe.with_suffix(".mp4").exists())
    if not usa_clipe and not (png.exists() and png.with_suffix(".json").exists()):
        typer.secho(f"falta {clipe.name} (rode scripts/make_presenter_video.py) "
                    f"ou {png.name} (rode scripts/make_presenter_cutouts.py)",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    script = _load_script(script_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    falado = respell(script.narration)
    audio = out_dir / "narracao.mp3"
    voz = VOZ_FEMININA if presenter == "iris" else VOZ_MASCULINA
    typer.echo(f"narrando ({voz})...")
    tempos = synthesize(falado.text, audio, voice=voz)
    palavras = align(falado, tempos)
    # O ffprobe do lado do ffmpeg estatico -- o do Fedora nem sempre esta la
    # (ver setup_renderer.sh); sem nenhum dos dois, cai na estimativa.
    sonda = Path(ffmpeg_bin()).with_name("ffprobe")
    dur = float(subprocess.run(
        [str(sonda) if sonda.exists() else "ffprobe", "-v", "error",
         "-show_entries", "format=duration", "-of", "csv=p=0", str(audio)],
        capture_output=True, text=True).stdout.strip() or 0)
    if not dur:
        dur = script.estimated_duration_s
    batidas = pr.batidas_por_tempo(script.hook, script.closing, palavras, dur)
    if seconds:
        dur = min(dur, seconds)

    marca = load_brand()
    acento = marca.accent_for(script.pillar or "news")
    nome = marca.presenters[presenter].name if presenter in marca.presenters else ""
    typer.echo(f"gancho ate {batidas.hook_end:.1f}s, fechamento em "
               f"{batidas.closing_start:.1f}s, {dur:.1f}s no total")
    typer.echo("material: " + ("clipe base (boca sintetizada)" if usa_clipe
                               else "retrato parado (tudo sintetizado)"))

    inicio = time.monotonic()
    if usa_clipe:
        base = pv.carregar_base(clipe)
        camada = pv.render_layer(base, audio, out_dir / "apresentador.mp4",
                                 duracao=dur, batidas=batidas, accent=acento,
                                 nome=nome, palavras_faladas=tempos,
                                 ffmpeg=ffmpeg_bin(), trabalho=out_dir)
    else:
        camada = pr.render_layer(pr.carregar(png), audio,
                                 out_dir / "apresentador.mp4", duracao=dur,
                                 batidas=batidas, accent=acento, nome=nome,
                                 ffmpeg=ffmpeg_bin(), semente=script.topic)
    parede = time.monotonic() - inicio
    typer.echo(f"camada: {camada.frames} quadros em {parede:.0f}s "
               f"({parede / max(camada.frames, 1) * 1000:.0f} ms/quadro), "
               f"{camada.path.stat().st_size / 1e6:.1f} MB")
    typer.echo(f"legenda: y={camada.subtitle_y} a partir de "
               f"{camada.subtitle_start:.1f}s (cartao sai junto)")

    enc = pr.plan(batidas)
    n = max(1, int(dur * pr.FPS))
    env = pr.envoltoria(audio, n, pr.FPS, ffmpeg=ffmpeg_bin())
    marcos = [("chamada", min(batidas.hook_end * 0.6, dur - 0.1)),
              ("travessia", min(enc.legenda_inicio - 0.3, dur - 0.1)),
              ("corpo", min((enc.legenda_inicio + batidas.closing_start) / 2, dur - 0.1)),
              ("fechamento", max(batidas.closing_start + 1.2, dur - 1.0))]
    if usa_clipe:
        abertura, largura = pv.trilha(tempos, n, pr.FPS)
        abertura = pv.modular(abertura, env)
        anim = pv.AnimadorVideo(base, pv.Quadros(base, out_dir, ffmpeg=ffmpeg_bin()),
                                acento, nome)

        def um(t: float):
            k = min(int(t * pr.FPS), n - 1)
            return anim.quadro(t, float(abertura[k]), float(largura[k]), enc.marcas)
    else:
        anim_p = pr.Animador(pr.carregar(png), acento, nome)
        janelas = pr.piscadas(dur, script.topic)

        def um(t: float):
            k = min(int(t * pr.FPS), len(env) - 1)
            return anim_p.quadro(t, float(env[k]), pr.fecho_em(janelas, t), enc.marcas)

    tiras = []
    for _, t in marcos:
        quadro = um(t)
        tela = Image.new("RGB", (pr.W, pr.H), (18, 20, 26))
        tela.paste(quadro, (camada.x, camada.y), quadro)
        tiras.append(tela.resize((pr.W // 4, pr.H // 4)))
    folha = Image.new("RGB", (tiras[0].width * len(tiras), tiras[0].height))
    for i, t in enumerate(tiras):
        folha.paste(t, (i * t.width, 0))
    folha.save(out_dir / "folha.png")
    typer.echo(f"folha de contato ({', '.join(m for m, _ in marcos)}): "
               f"{out_dir / 'folha.png'}")


@app.command()
def status() -> None:
    """Resumo da memoria: contexto e qualidade num relance."""
    from agent.memory.store import SignalStore

    store = SignalStore(settings.db_path)
    with store._conn() as conn:
        def n(tabela: str) -> int:
            return int(conn.execute(f"SELECT COUNT(*) FROM {tabela}").fetchone()[0])
        temas = [r["term"] for r in conn.execute(
            "SELECT term FROM topics WHERE verdict='selected'"
            " ORDER BY decided_at DESC LIMIT 5").fetchall()]
        pend = [dict(r) for r in conn.execute(
            "SELECT publish_id, status FROM posts ORDER BY created_at DESC LIMIT 3"
        ).fetchall()]
    toks = 0
    for tabela in ("dossiers", "scripts", "reviews"):
        with store._conn() as conn:
            toks += conn.execute(
                f"SELECT COALESCE(SUM(input_tokens+output_tokens),0) FROM {tabela}"
            ).fetchone()[0]
    typer.echo(f"sinais={n('signals')} temas={n('topics')} dossies={n('dossiers')} "
               f"roteiros={n('scripts')} pareceres={n('reviews')} "
               f"carrosseis={n('carousels')} posts={n('posts')} metricas={n('metrics')}")
    typer.echo(f"tokens totais medidos: {toks}")
    if temas:
        typer.echo("ultimos temas: " + " | ".join(t[:48] for t in temas))
    for p in pend:
        m = store.latest_metric(p["publish_id"])
        extra = f" views={m['views']}" if m else ""
        typer.echo(f"  {p['publish_id'][:30]}: {p['status']}{extra}")


@app.command()
def preflight(
    video: Path = typer.Option(None, "--video",
                               help="mp4 do pacote (video); omita no carrossel"),
    script: Path = typer.Option(None, "--script", "-s",
                                help="roteiro.json ou carrossel.json do pacote"),
    slides_dir: Path = typer.Option(None, "--slides",
                                    help="pasta dos slides (carrossel)"),
) -> None:
    """Camada final antes de producao: confere o pacote sem julgar de novo.

    Portoes: parecer aprovado ligado ao texto, MP4 1080x1920 com audio na
    faixa do formato (ou 5 slides + caption no carrossel), fatos com fonte.
    O checklist humano (AIGC, legenda) sai no fim, sempre.
    """
    from agent.memory.store import SignalStore
    from agent.models import Carousel
    from agent.publish.preflight import (
        CHECKLIST,
        preflight_carousel,
        preflight_video,
    )

    if script is None:
        typer.secho("informe --script (roteiro ou carrossel do pacote).",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)
    if not script.exists():
        typer.secho(f"arquivo ausente: {script}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    import json as _json
    raw = _json.loads(script.read_text(encoding="utf-8"))
    store = SignalStore(settings.db_path)

    if "slides" in raw:
        carrossel = Carousel.model_validate(raw)
        rows = store.list_carousels(carrossel.topic)
        aprovado = any(r["approved"] for r in rows)
        ok_slides = True
        if slides_dir is not None:
            from PIL import Image
            pngs = sorted(slides_dir.glob("slide-*.png"))
            ok_slides = len(pngs) == 5 and all(
                Image.open(p).size == (1080, 1920) for p in pngs)
            ok_slides = ok_slides and (slides_dir / "caption.txt").exists()
        report = preflight_carousel(raw, aprovado, ok_slides)
    else:
        probe = None
        if video is not None and video.exists():
            from agent.adapters.mpt_renderer import probe_video
            try:
                probe = probe_video(video)
            except Exception as exc:
                probe = {"error": str(exc)[:120]}
        elif video is not None:
            typer.secho(f"video ausente: {video}", fg=typer.colors.RED)
        report = preflight_video(
            raw, store.list_scripts(raw.get("topic")),
            store.list_reviews(raw.get("topic")), probe)

    ok = True
    for g in report.gates:
        mark = "OK  " if g.passed else "FALHA"
        typer.secho(f"[{mark}] {g.label}"
                    + (f" -- {g.detail}" if g.detail else ""),
                    fg=typer.colors.GREEN if g.passed else typer.colors.RED)
        ok = ok and g.passed

    typer.echo("\nchecklist humano (nao automatizavel):")
    for item in CHECKLIST:
        typer.echo(f"  [ ] {item}")

    raise typer.Exit(code=0 if ok else 1)


@app.command("slot")
def slot_cmd(
    slot: str = typer.Option(..., "--slot", help="0900, 1500 ou 2000 (horario de Brasilia)"),
    day: str = typer.Option("", "--day", help="AAAA-MM-DD; padrao: hoje em Brasilia"),
    publish: bool = typer.Option(True, "--publish/--no-publish",
                                 help="sobe o video para a inbox do TikTok"),
    wait: bool = typer.Option(True, "--wait/--no-wait",
                              help="espera o horario do slot para publicar"),
    force: bool = typer.Option(False, "--force", help="refaz slot ja concluido"),
) -> None:
    """Produz e publica UM slot do dia, sozinho (e o que o timer do systemd chama).

    Radar -> tema para o horario -> pesquisa -> formato pela informacao ->
    roteiro + juiz (modelos roteados por cota) -> render medido -> espera a hora
    -> inbox do TikTok (video) ou pacote para postar (carrossel) -> aviso.
    """
    from datetime import date as _date

    from agent.autopilot.runner import SlotRunner
    from agent.editorial.slots import parse_slot, today

    try:
        alvo = parse_slot(slot)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    dia = _date.fromisoformat(day) if day else today()
    resultado = SlotRunner().run(alvo, dia, publish=publish, wait=wait, force=force)
    cor = (typer.colors.GREEN if resultado.state in ("published", "ready_manual", "produced")
           else typer.colors.YELLOW if resultado.state == "skipped" else typer.colors.RED)
    typer.secho(f"\nslot {resultado.slot} {resultado.day}: {resultado.state.upper()}", fg=cor,
                bold=True)
    for rotulo, valor in (("tema", resultado.topic), ("tipo", resultado.pillar),
                          ("formato", resultado.format), ("pacote", resultado.package_dir),
                          ("publish_id", resultado.publish_id), ("erro", resultado.error)):
        if valor:
            typer.echo(f"  {rotulo:<10}: {valor}")
    raise typer.Exit(code=0 if resultado.state != "failed" else 1)


@app.command("slot-extra")
def slot_extra(
    force: bool = typer.Option(False, "--force", help="refaz a rodada extra de hoje"),
    publish: bool = typer.Option(True, "--publish/--no-publish",
                                 help="sobe o video para a inbox do TikTok; "
                                      "--no-publish so produz o pacote (conferir "
                                      "render sem gastar rascunho da inbox)"),
    formato: str = typer.Option("", "--format",
                                help="video (long+short), carrossel (carousel+short), "
                                     "short ou long (um so, sem reserva); "
                                     "vazio escolhe pelo dossie"),
) -> None:
    """Rodada extra fora da grade: da noticia nova a inbox, sem tocar nos slots.

    Le os temas/pilares/formatos que o dia ja usou (nao repete), registra em
    linha propria (`extra`) que nenhum timer le, e publica na hora (--no-wait).
    """
    from datetime import datetime as _datetime

    from agent.autopilot.runner import SlotRunner
    from agent.editorial.slots import TZ, Slot, today

    agora = _datetime.now(TZ)
    alvo = Slot("extra", agora.time(), "extra",
                "rodada extra manual: do radar a inbox, fora da grade")
    forca = None
    if formato:
        chave = formato.strip().lower()
        if chave not in SlotRunner.FORMATOS_EXTRA:
            typer.secho(f"--format {formato!r} desconhecido; use "
                        + ", ".join(SlotRunner.FORMATOS_EXTRA) + ".",
                        fg=typer.colors.RED)
            raise typer.Exit(code=2)
        forca = {"extra": SlotRunner.FORMATOS_EXTRA[chave]}
    resultado = SlotRunner(formatos=forca).run(
        alvo, today(), publish=publish, wait=False, force=force)
    cor = (typer.colors.GREEN if resultado.state in ("published", "ready_manual", "produced")
           else typer.colors.YELLOW if resultado.state == "skipped" else typer.colors.RED)
    typer.secho(f"\nslot {resultado.slot} {resultado.day}: {resultado.state.upper()}", fg=cor,
                bold=True)
    for rotulo, valor in (("tema", resultado.topic), ("tipo", resultado.pillar),
                          ("formato", resultado.format), ("pacote", resultado.package_dir),
                          ("publish_id", resultado.publish_id), ("erro", resultado.error)):
        if valor:
            typer.echo(f"  {rotulo:<10}: {valor}")
    raise typer.Exit(code=0 if resultado.state != "failed" else 1)


@app.command("autopilot-status")
def autopilot_status(
    day: str = typer.Option("", "--day", help="AAAA-MM-DD; padrao: hoje em Brasilia"),
    detail: bool = typer.Option(False, "--detail", help="motivo completo de cada decisao"),
) -> None:
    """O dia do piloto: cada slot, o motivo das escolhas, cota e custo por modelo."""
    import json as _json
    from datetime import UTC, datetime, timedelta

    from agent.autopilot.runs import SlotRuns
    from agent.editorial.slots import SLOTS, TZ, today
    from agent.memory.llm_ledger import LLMLedger

    settings.ensure_dirs()
    dia = day or today().isoformat()
    runs = {r["slot"]: r for r in SlotRuns(settings.db_path).day(dia)}
    typer.secho(f"piloto automatico -- {dia} (horario de Brasilia)", bold=True)
    for sid, slot in SLOTS.items():
        r = runs.get(sid)
        if r is None:
            typer.echo(f"  {slot.at:%H:%M}  ainda nao rodou  ({slot.intent})")
            continue
        cor = {"published": typer.colors.GREEN, "ready_manual": typer.colors.CYAN,
               "produced": typer.colors.BLUE, "failed": typer.colors.RED}.get(
                   r["state"], typer.colors.YELLOW)
        typer.secho(f"  {slot.at:%H:%M}  {r['state']:<12} {r['format'] or '-':<8} "
                    f"[{r['pillar'] or '-'}] {(r['topic'] or '')[:60]}", fg=cor)
        if r["error"]:
            typer.secho(f"         erro: {r['error'][:160]}", fg=typer.colors.RED)
        if r["package_dir"]:
            typer.echo(f"         pacote: {r['package_dir']}")
        if detail and r["plan_json"]:
            plano = _json.loads(r["plan_json"])
            for chave in ("topic_reason", "format_reason", "render_warning"):
                if plano.get(chave):
                    typer.echo(f"         {chave}: {plano[chave]}")
    livro = LLMLedger(settings.db_path)
    inicio = datetime.fromisoformat(dia).replace(tzinfo=TZ)
    fim = (inicio + timedelta(days=1)).astimezone(UTC).isoformat()
    uso: dict[str, dict[str, int]] = {}
    for c in livro.calls_since(inicio.astimezone(UTC)):
        if c["created_at"] >= fim:
            continue
        linha = uso.setdefault(c["route"], {"calls": 0, "failed": 0, "tokens": 0})
        linha["calls"] += 1
        linha["failed"] += 0 if c["ok"] else 1
        linha["tokens"] += int(c["input_tokens"]) + int(c["output_tokens"])
    if uso:
        typer.secho("\nLLM no dia (chamadas / falhas / tokens):", bold=True)
        for rota, u in sorted(uso.items()):
            typer.echo(f"  {rota:<42} {u['calls']:>3} / {u['failed']:>2} / {u['tokens']:>7}")
    esgotados = livro.quota_status()
    if esgotados:
        typer.secho("\nsem cota agora:", bold=True)
        for e in esgotados:
            typer.echo(f"  {e['route']:<42} ate {e['exhausted_until'][:16]} UTC  "
                       f"({e['reason'][:60]})")


if __name__ == "__main__":
    app()
