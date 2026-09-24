"""CLI del agente."""

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

app = typer.Typer(add_completion=False, help="Agente de vídeo corto para tecnología, IA y ciencia")


def _load_script(path: Path) -> Script:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return Script.model_validate(raw)


@app.command()
def render(
    script_path: Path = typer.Option(..., "--script", "-s", exists=True, readable=True),
    out_dir: Path = typer.Option(None, "--out-dir",
                                 help="carpeta del paquete; organiza por día/hora/tema"),
) -> None:
    """Renderiza un guion a MP4 vertical y comprueba la calidad del M0."""
    from agent.paths import run_dir
    script = _load_script(script_path)
    typer.echo(f"tema      : {script.topic}")
    typer.echo(f"narracion : {script.word_count} palabras "
               f"(~{script.estimated_duration_s:.0f}s estimados)")
    typer.echo(f"términos  : {', '.join(script.search_terms)}")
    typer.echo(f"hechos    : {len(script.facts)} con fuente")

    renderer = MptRenderer()
    if not renderer.health():
        typer.secho(
            f"renderizador no responde en {settings.renderer_url}\n"
            "arranca con: ./scripts/setup_renderer.sh --serve",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    typer.echo("renderizando (minutos, en CPU)...")
    try:
        result = renderer.render(script)
    except RendererError as exc:
        typer.secho(f"falha: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if result.state is RenderState.failed:
        typer.secho(f"renderizado falló: {result.error}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo(f"archivo   : {result.video_path}")
    typer.echo(f"dimensiones : {result.width}x{result.height}")
    typer.echo(f"duracion   : {result.duration_s}s")
    typer.echo(f"narracion : {'presente' if result.has_audio else 'AUSENTE'}")

    if out_dir is None:
        out_dir = run_dir(script.format or "long", script.topic,
                          base=settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "video.mp4"
    if result.video_path and Path(result.video_path) != destino:
        Path(result.video_path).replace(destino)
    script_path_out = out_dir / "guion.json"
    if Path(script_path) != script_path_out:
        script_path_out.write_text(
            Path(script_path).read_text(encoding="utf-8"), encoding="utf-8")
    typer.echo(f"paquete   : {out_dir}")

    # Aceptación medida, no presumida. La franja depende del formato: el short no
    # es elegible al Rewards, y cobrarle 60-90s reprobaría todo corto.
    formato = script.format or "long"
    franja = (MIN_DURATION_S, MAX_DURATION_S) if formato == "long" else (10, 30)
    en_franja = (result.duration_s is not None
                 and franja[0] <= result.duration_s <= franja[1])
    checks = [
        ("9:16 en 1080x1920", result.is_portrait_1080x1920),
        (f"duración entre {franja[0]}s y {franja[1]}s ({formato})", en_franja),
        ("pista de audio presente", result.has_audio),
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
    limit: int = typer.Option(15, "--limit", "-n", help="cuántas señales listar"),
) -> None:
    """Recoge señales de tendencia de las fuentes gratuitas y graba la serie."""
    from agent.memory.store import SignalStore
    from agent.radar.collector import Radar, default_sources

    settings.ensure_dirs()
    report = Radar(default_sources(), SignalStore(settings.db_path)).collect()

    for nombre, error in report.failures.items():
        typer.secho(f"[fuente caída] {nombre}: {error}", fg=typer.colors.YELLOW)

    if not report.signals:
        typer.secho("ninguna señal colectada", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Ordena por velocidad; sin velocidad va al final, porque "desconocido"
    # no puede competir de igual a igual con una medida.
    ordenados = sorted(
        report.signals, key=lambda s: (s.has_velocity, s.velocity or 0), reverse=True
    )

    typer.echo("")
    typer.echo(f"{'velocidad':>12}  {'volumen':>10}  {'fuente':<14}  término")
    typer.echo("-" * 92)
    for s in ordenados[:limit]:
        vel = f"{s.velocity:,.1f}/h" if s.has_velocity else "-"
        typer.echo(f"{vel:>12}  {s.volume:>10,.0f}  {s.source:<14}  {s.term[:44]}")

    typer.echo("")
    typer.echo(f"{len(report.signals)} señales de {len(report.sources_ok)} fuentes "
               f"({len(report.with_velocity)} con velocidad) en {report.elapsed_s}s")
    if report.failures:
        typer.echo(f"{len(report.failures)} fuente(s) caída(s); la recogida siguió sin ellas")


@app.command()
def curate(
    show: int = typer.Option(8, "--show", help="cuantos rechazados detallar"),
    dry_run: bool = typer.Option(False, "--dry-run", help="no graba en el libro"),
    top: int = typer.Option(1, "--top", help="candidatos distintos para la rutina del dia"),
    cooldown_days: int = typer.Option(30, "--cooldown-days",
                                      help="ventana anti repeticion del libro"),
) -> None:
    """Recoge señales y elige tema(s), registrando el motivo de cada decisión.

    `--top 3` lista los 3 mejores temas DISTINTOS (deduplicados entre sí)
    para la rutina long+short+carrusel del día. Repetir tema solo es válido
    como actualización explícita (`research --topic`), nunca por el curador.
    """
    from agent.curator.curator import Curator
    from agent.memory.store import SignalStore
    from agent.models import Verdict
    from agent.radar.collector import Radar, default_sources

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)

    recoge = Radar(default_sources(), store).collect()
    for nombre, error in recoge.failures.items():
        typer.secho(f"[fuente caída] {nombre}: {error}", fg=typer.colors.YELLOW)
    if not recoge.signals:
        typer.secho("ninguna señal colectada", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    ledger = store.recent_topics(days=cooldown_days)
    report = Curator().curate(recoge.signals, ledger=ledger)

    conteo = report.tally()
    typer.echo("")
    typer.echo(f"{len(recoge.signals)} señales -> "
               + "  ".join(f"{k}={v}" for k, v in conteo.items() if v))
    typer.echo(f"libro: {len(ledger)} temas aprobados en los últimos {cooldown_days} días")

    for verdict, cor in ((Verdict.rejected_policy, typer.colors.RED),
                         (Verdict.rejected_duplicate, typer.colors.YELLOW)):
        for d in report.by_verdict(verdict)[:show]:
            typer.secho(f"  [{verdict.value}] {d.term[:52]}", fg=cor)
            typer.secho(f"      {d.reason}", fg=typer.colors.BRIGHT_BLACK)

    elegido = report.selected
    if elegido is None:
        typer.secho("\nningún tema elegible hoy", fg=typer.colors.RED)
        if not dry_run:
            store.record_decisions(report.decisions)
        raise typer.Exit(code=1)

    typer.echo("")
    typer.secho(f"TEMA: {elegido.term}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {elegido.reason}")
    if elegido.url:
        typer.echo(f"  {elegido.url}")
    if elegido.news_items:
        typer.echo(f"  {len(elegido.news_items)} noticias ya asociadas por la fuente")

    candidatos = [d for d in report.top(top) if d.term != elegido.term]
    if candidatos:
        typer.echo("\n  rutina del día (distintos, sin repetir):")
        for i, d in enumerate(candidatos, start=2):
            typer.echo(f"    {i}. {d.term[:58]} (score {d.score:.3f})")

    vice = [d for d in report.by_verdict(Verdict.not_selected)][:4]
    if vice:
        typer.echo("\n  siguientes colocados:")
        for d in vice:
            typer.echo(f"    {d.score:.3f}  {d.term[:58]}")

    if dry_run:
        typer.secho("\n--dry-run: nada grabado en el libro", fg=typer.colors.YELLOW)
    else:
        n = store.record_decisions(report.decisions)
        typer.echo(f"\n{n} decisiones grabadas en el libro")


@app.command()
def research(
    topic: str = typer.Option(
        "", "--topic", "-t",
        help="investiga este tema; sin eso, ejecuta el curador. "
             "Tema explícito es actualización intencional.",
    ),
    url: list[str] = typer.Option(
        [], "--url", "-u", help="fuente explícita (repetible); salta el descubrimiento"
    ),
    llm: str = typer.Option("", "--llm", help="gemini o  groq; por defecto viene del .env"),
    max_sources: int = typer.Option(0, "--sources", help="techo de fuentes a leer"),
    dry_run: bool = typer.Option(False, "--dry-run", help="no graba el dossier"),
) -> None:
    """Monta el dossier de un tema: 3-5 fuentes, cada hecho con URL y pasaje comprobados."""
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
    techo = max_sources or settings.research_max_sources

    if topic:
        decision = Decision(
            term=topic, source="manual", verdict=Verdict.selected,
            reason="tema informado en la línea de comando, sin pasar por el curador",
            score=0.0, niche_fit=0.0, decided_at=datetime.now(UTC),
        )
    else:
        decision = _curate_one(store)

    typer.echo(f"tema      : {decision.term}")
    typer.echo(f"origen    : {decision.source}")

    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model}")

    candidatos = (
        [Candidate(url=u, origin="manual") for u in url] if url else None
    )
    investigador = Researcher(
        modelo,
        fetcher=PageFetcher(max_chars=settings.research_page_chars),
        max_sources=techo,
        max_facts_per_source=settings.research_max_facts_per_source,
    )

    typer.echo("leyendo fuentes (una llamada de modelo por fuente)...")
    report = investigador.research(decision, candidates=candidatos)

    for donde, motivo in report.failures.items():
        typer.secho(f"[fuente caída] {donde[:70]}: {motivo}", fg=typer.colors.YELLOW)

    if report.pages:
        typer.echo("")
        typer.echo("fuentes leídas:")
        for pagina in report.pages:
            typer.echo(f"  {pagina.source_name:<22} {len(pagina.text):>6} car  {pagina.url[:70]}")

    # Lo descartado va antes del dossier a propósito: es la parte que se pierde si
    # nadie lo mira, y dice si la puerta está calibrada o  estrangulando.
    if report.discarded:
        typer.echo("")
        typer.secho(f"{len(report.discarded)} hecho(s) descartado(s) por las puertas:",
                    fg=typer.colors.YELLOW)
        for d in report.discarded:
            typer.secho(f"  {d.claim[:76]}", fg=typer.colors.YELLOW)
            typer.secho(f"      {d.reason}", fg=typer.colors.BRIGHT_BLACK)

    coste = report.usage
    typer.echo("")
    typer.echo(f"coste     : {coste.input_tokens} tokens de entrada, "
               f"{coste.output_tokens} de salida, {report.latency_s}s de modelo")

    if not report.ok:
        typer.secho("ningún hecho con fuente sobrevivió; sin dossier", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    dossier = report.dossier
    typer.echo("")
    typer.secho(f"DOSSIER: {len(dossier.facts)} hechos de {report.source_count} fuente(s)",
                fg=typer.colors.GREEN, bold=True)
    for f in dossier.facts:
        typer.echo("")
        typer.secho(f"  {f.claim}", bold=True)
        typer.echo(f"      fuente : {f.source_name} — {f.source_url}")
        if f.quote:
            typer.secho(f'      pasaje: "{f.quote[:100]}"', fg=typer.colors.BRIGHT_BLACK)

    if dry_run:
        typer.secho("\n--dry-run: dossier no grabado", fg=typer.colors.YELLOW)
        return

    fila = store.record_dossier(
        dossier,
        model=modelo.model,
        provider=modelo.provider,
        usage=(coste.input_tokens, coste.output_tokens),
        latency_s=report.latency_s,
        source_count=report.source_count,
        discarded=[vars(d) for d in report.discarded],
        failures=report.failures,
    )
    typer.echo(f"\ndossier #{fila} grabado ({store.dossier_count()} en memoria)")


def _curate_one(store: Any) -> Decision:
    """Ejecuta radar + curador y devuelve el tema del dia, o sale con error."""
    from agent.curator.curator import Curator
    from agent.radar.collector import Radar, default_sources

    recoge = Radar(default_sources(), store).collect()
    for nombre, error in recoge.failures.items():
        typer.secho(f"[fuente caída] {nombre}: {error}", fg=typer.colors.YELLOW)
    if not recoge.signals:
        typer.secho("ninguna señal colectada", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    report = Curator().curate(recoge.signals, ledger=store.recent_topics())
    store.record_decisions(report.decisions)
    elegido = report.selected
    if elegido is None:
        typer.secho("ningún tema elegible hoy", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return elegido


@app.command()
def write(
    topic: str = typer.Option("", "--topic", "-t", help="tema; sin eso, usa el último dossier"),
    llm: str = typer.Option("", "--llm", help="gemini o  groq; por defecto viene del .env"),
    out: Path = typer.Option(None, "--out", "-o", help="graba el guion en JSON para render"),
    dry_run: bool = typer.Option(False, "--dry-run", help="no graba en memoria"),
    mode: str = typer.Option("long", "--mode",
                               help="long (60-90s), short (~15s) o  carrusel"),
    polish: bool = typer.Option(True, "--polish/--no-polish",
                                help="humanización tras el aceite mecánico"),
) -> None:
    """Escribe el guion a partir de un dossier ya grabado."""
    from agent.adapters.llm_factory import build_llm
    from agent.memory.store import SignalStore
    from agent.ports.llm import LLMError
    from agent.writer.writer import Screenwriter

    if mode not in ("long", "short", "carousel"):
        typer.secho(f"modo {mode!r} desconocido; use long, short o  carrusel.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)

    dossier = store.latest_dossier(topic or None)
    if dossier is None:
        objetivo = f" para el tema {topic!r}" if topic else ""
        typer.secho(
            f"ningún dossier{objetivo} en memoria. Ejecuta `uv run agent research` primero.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    typer.echo(f"tema      : {dossier.topic}")
    typer.echo(f"dossier    : {len(dossier.facts)} hechos de "
               f"{len(dossier.source_urls)} fuente(s)")

    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model} (modo {mode})")

    if mode == "carousel":
        _write_carousel_cmd(store, dossier, modelo, out, dry_run)
        return

    typer.echo("escribiendo (corrige solo lo que es mecánico)...")
    try:
        report = Screenwriter(modelo).write(dossier, mode=mode, polish=polish)
    except LLMError as exc:
        typer.secho(f"fallo del proveedor: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if report.refusal:
        typer.secho(f"\n{report.refusal}", fg=typer.colors.RED)
        typer.echo("ninguna llamada de modelo fue hecha; coste cero")
        raise typer.Exit(code=1)

    # Los intentos corregidos aparecen incluso en el éxito: si toda ejecución gasta
    # dos rondas en el mismo defecto, el prompt es que está flojo.
    for i, intento in enumerate(report.attempts, start=1):
        if intento.violations:
            typer.secho(f"intento {i} reprobado ({intento.word_count} palabras):",
                        fg=typer.colors.YELLOW)
            for v in intento.violations:
                typer.secho(f"  - {v}", fg=typer.colors.BRIGHT_BLACK)

    if report.humanized:
        typer.secho("humanización aplicada "
                    f"({'; '.join(report.humanize_notes)})", fg=typer.colors.GREEN)
    elif report.humanize_notes:
        typer.secho(f"humanización mantenida en el original: "
                    f"{'; '.join(report.humanize_notes)}", fg=typer.colors.YELLOW)

    coste = report.usage
    typer.echo(f"coste     : {coste.input_tokens} tokens de entrada, "
                f"{coste.output_tokens} de salida, {report.latency_s}s de modelo, "
                f"{len(report.attempts)} intento(s)")

    if not report.ok:
        typer.secho("\nningún guion pasó en las puertas mecánicas", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    script = report.script
    typer.echo("")
    typer.secho(f"GUION: {script.word_count} palabras "
                f"(~{script.estimated_duration_s:.0f}s estimados)",
                fg=typer.colors.GREEN, bold=True)
    typer.echo("")
    typer.secho("  HOOK", bold=True)
    typer.echo(f"  {script.hook}")
    typer.echo("")
    typer.secho("  CUERPO", bold=True)
    for parrafo in script.body.split("\n"):
        if parrafo.strip():
            typer.echo(f"  {parrafo.strip()}")
    typer.echo("")
    typer.secho("  CIERRE", bold=True)
    typer.echo(f"  {script.closing}")
    typer.echo("")
    typer.echo(f"  términos: {', '.join(script.search_terms)}")
    typer.echo(f"  hechos : {len(script.facts)} del dossier")

    if out is not None:
        out.write_text(
            script.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )
        typer.echo(f"\ngrabado en {out}")
        typer.echo(f"renderiza con: uv run agent render --script {out}")

    if dry_run:
        typer.secho("\n--dry-run: guion no grabado en memoria", fg=typer.colors.YELLOW)
        return

    fila = store.record_script(
        script,
        model=modelo.model,
        provider=modelo.provider,
        usage=(coste.input_tokens, coste.output_tokens),
        latency_s=report.latency_s,
        attempts=[
            {"violations": a.violations, "word_count": a.word_count} for a in report.attempts
        ],
        dossier_id=store.latest_dossier_id(dossier.topic),
    )
    typer.echo(f"\nguion #{fila} grabado ({store.script_count()} en memoria)")


def _write_carousel_cmd(store: Any, dossier: Any, modelo: Any,
                        out: Path | None, dry_run: bool) -> None:
    """Rama carrusel del `write`: 5 slides + leyenda, sin juez aquí."""
    from agent.ports.llm import LLMError
    from agent.writer.carousel import write_carousel

    typer.echo("escribiendo carrusel (5 slides, puertas propias)...")
    try:
        report = write_carousel(dossier, modelo)
    except LLMError as exc:
        typer.secho(f"fallo del proveedor: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if report.refusal:
        typer.secho(f"\n{report.refusal}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    for i, intento in enumerate(report.attempts, start=1):
        if intento.violations:
            typer.secho(f"intento {i} reprobado:", fg=typer.colors.YELLOW)
            for v in intento.violations:
                typer.secho(f"  - {v}", fg=typer.colors.BRIGHT_BLACK)

    coste = report.usage
    typer.echo(f"coste     : {coste.input_tokens} tokens de entrada, "
               f"{coste.output_tokens} de salida, {report.latency_s}s de modelo")

    if not report.ok or report.carousel is None:
        typer.secho("\nningún carrusel pasó en las puertas mecánicas",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    for s in report.carousel.slides:
        typer.echo(f"\n  [{s.n}/5] {s.headline}\n  {s.text}\n  ~ {s.visual}")
    typer.echo(f"\n  leyenda: {report.carousel.caption}")

    if out is not None:
        out.write_text(
            report.carousel.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )
        typer.echo(f"\ngrabado en {out}")
        typer.echo(f"júzga con: uv run agent judge --script {out}")
        typer.echo(f"slides con: uv run agent carousel-render --carousel {out}")

    if dry_run:
        typer.secho("\n--dry-run: carrusel no grabado en memoria",
                    fg=typer.colors.YELLOW)
        return
    fila = store.record_carousel(
        report.carousel, model=modelo.model, provider=modelo.provider,
        usage=(coste.input_tokens, coste.output_tokens),
        latency_s=report.latency_s, attempts=[a.violations for a in report.attempts],
    )
    typer.echo(f"\ncarrusel #{fila} grabado")


def _is_carousel_file(path: Path) -> bool:
    try:
        return "slides" in json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False


def _judge_carousel_cmd(store: Any, path: Path, llm: str, dry_run: bool) -> None:
    """Informe de carrusel (rubrica propia, corte 8/10)."""
    from datetime import UTC, datetime

    from agent.adapters.llm_factory import build_llm
    from agent.judge.carousel import judge_carousel
    from agent.models import Carousel, CarouselReview, Dossier
    from agent.ports.llm import LLMError

    try:
        carrusel = Carousel.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        typer.secho(f"carrusel inválido: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    dossier = Dossier(topic=carrusel.topic, facts=carrusel.facts,
                      collected_at=datetime.now(UTC))
    if not dossier.facts:
        typer.secho("carrusel sin ningún hecho: no hay dossier contra lo que juzgar",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    typer.echo(f"tema      : {carrusel.topic}")
    typer.echo(f"carrusel : {len(carrusel.slides)} slides")
    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model}")
    try:
        report = judge_carousel(carrusel, dossier, modelo)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    review: CarouselReview | None = report.review
    assert review is not None
    for s in review.scores:
        cor = (typer.colors.GREEN if s.score == 2
               else typer.colors.YELLOW if s.score == 1 else typer.colors.RED)
        typer.secho(f"  [{s.score}/2] {s.criterion.value}: {s.reason}", fg=cor)
    estado = "APROBADO" if review.approved else "REPROBADO"
    typer.secho(f"\n{estado}: {review.total}/10 (corte 8, ninguno zerado)",
                fg=typer.colors.GREEN if review.approved else typer.colors.RED)
    typer.echo(f"coste     : {report.usage.total_tokens} tokens, {report.latency_s}s")

    if not dry_run:
        fila = store.record_carousel(
            carrusel, model=modelo.model, provider=modelo.provider,
            usage=(report.usage.input_tokens, report.usage.output_tokens),
            latency_s=report.latency_s, attempts=[],
            review=review,
        )
        typer.echo(f"informe de carrusel grabado (carrusel #{fila})")
    raise typer.Exit(code=0 if review.approved else 1)


def _mostrar_parecer(review) -> None:
    """Imprime la rúbrica entera, criterio a criterio.

    Siempre entera, incluidos los criterios que sacaron 2: un informe resumido en
    "reprobado (9/14)" no dice lo que cambiar, y es el motivo por criterio el que
    vuelve al guionista en la revisión.
    """
    from agent.models import RUBRIC_CUTOFF, RUBRIC_MAX

    typer.echo("")
    for s in review.scores:
        color = (typer.colors.GREEN if s.score == 2
                 else typer.colors.YELLOW if s.score == 1 else typer.colors.RED)
        # "juzgado" para una nota que no  fue juzgada sería mentira en el propio
        # informe que existe para ser auditado.
        marca = "saltado " if not s.evaluated else "medido " if s.measured else "juzgado"
        typer.secho(f"  [{s.score}/2] {marca}  {s.criterion.value}", fg=color)
        typer.secho(f"          {s.reason}", fg=typer.colors.BRIGHT_BLACK)

    typer.echo("")
    if review.approved:
        typer.secho(f"APROBADO: {review.total}/{RUBRIC_MAX} "
                    f"(corte {RUBRIC_CUTOFF}, ningún criterio zerado)",
                    fg=typer.colors.GREEN, bold=True)
        return

    typer.secho(f"REPROBADO: {review.total}/{RUBRIC_MAX} (corte {RUBRIC_CUTOFF})",
                fg=typer.colors.RED, bold=True)
    for s in review.vetoed:
        typer.secho(f"  veto en {s.criterion.value}: es requisito, no  calidad — "
                    "nota en los otros criterios no  compensa", fg=typer.colors.RED)
    for s in review.zeroed:
        if s not in review.vetoed:
            typer.secho(f"  zerado en {s.criterion.value}: criterio zerado reprueba "
                        "aunque la suma pase el corte", fg=typer.colors.RED)


@app.command()
def judge(
    topic: str = typer.Option("", "--topic", "-t", help="tema; sin eso, usa el último guion"),
    script_path: Path = typer.Option(
        None, "--script", "-s", exists=True, readable=True,
        help="juzga este archivo en vez del último guion de la memoria",
    ),
    llm: str = typer.Option("", "--llm", help="gemini o  groq; por defecto viene del .env"),
    dry_run: bool = typer.Option(False, "--dry-run", help="no graba el informe"),
) -> None:
    """Aplica la rúbrica a un guion (vídeo) o carrusel (lo detecta por el archivo)."""
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
        # El Script lleva los hechos que usó el guionista, así que un archivo se
        # autojuzga: es lo que permite correr la fixture adversarial sin la memoria.
        dossier = Dossier(
            topic=script.topic, facts=script.facts, collected_at=datetime.now(UTC)
        )
    else:
        script = store.latest_script(topic or None)
        if script is None:
            typer.secho("ningún guion en memoria. Ejecuta `uv run agent write` primero.",
                        fg=typer.colors.RED)
            raise typer.Exit(code=2)
        dossier = store.latest_dossier(script.topic) or Dossier(
            topic=script.topic, facts=script.facts, collected_at=datetime.now(UTC)
        )

    if not dossier.facts:
        typer.secho("guion sin ningún hecho: no hay dossier contra lo que juzgar",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    typer.echo(f"tema      : {script.topic}")
    typer.echo(f"guion   : {script.word_count} palabras, "
               f"{len(script.facts)} hechos, ~{script.estimated_duration_s:.0f}s")

    # Medida primero: si el guion ya reprueba en un criterio de requisito, no  hay
    # motivo para exigir clave de API para confirmar eso.
    anticipado = review_measured_only(script, dossier)
    if anticipado is not None:
        typer.echo("modelo    : no  consultado (reprobó en la medida)")
        _mostrar_parecer(anticipado)
        typer.echo("\ncoste     : 0 tokens")
        if not dry_run:
            fila = store.record_review(anticipado, usage=(0, 0), latency_s=0.0)
            typer.echo(f"informe #{fila} grabado")
        raise typer.Exit(code=1)

    try:
        modelo = build_llm(llm or None)
        typer.echo(f"modelo    : {modelo.provider}/{modelo.model}")
        report = Judge(modelo).review(script, dossier)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    _mostrar_parecer(report.review)
    typer.echo(f"\ncoste     : {report.usage.total_tokens} tokens, {report.latency_s}s")

    if not dry_run:
        fila = store.record_review(
            report.review,
            usage=(report.usage.input_tokens, report.usage.output_tokens),
            latency_s=report.latency_s,
        )
        typer.echo(f"informe #{fila} grabado")

    raise typer.Exit(code=0 if report.approved else 1)


@app.command()
def produce(
    topic: str = typer.Option("", "--topic", "-t", help="tema; sin eso, usa el último dossier"),
    out: Path = typer.Option(None, "--out", "-o", help="graba el guion aprobado en JSON"),
    revisions: int = typer.Option(2, "--revisions", help="techo de rondas de revisión"),
    llm: str = typer.Option("", "--llm", help="gemini o  groq; por defecto viene del .env"),
    dry_run: bool = typer.Option(False, "--dry-run", help="no graba en memoria"),
    mode: str = typer.Option("long", "--mode", help="long, short o  carrusel"),
) -> None:
    """Escribe, juzga y revisa hasta que el guion pase la rúbrica o se agoten las rondas."""
    from agent.adapters.llm_factory import build_llm
    from agent.judge.judge import Judge
    from agent.memory.store import SignalStore
    from agent.pipeline import produce as producir
    from agent.ports.llm import LLMError
    from agent.writer.writer import Screenwriter

    if mode not in ("long", "short", "carousel"):
        typer.secho(f"modo {mode!r} desconocido.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    settings.ensure_dirs()
    store = SignalStore(settings.db_path)

    dossier = store.latest_dossier(topic or None)
    if dossier is None:
        typer.secho("ningún dossier en memoria. Ejecuta `uv run agent research` primero.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if mode == "carousel":
        _produce_carousel_cmd(store, dossier, llm, out, revisions, dry_run)
        return

    typer.echo(f"tema      : {dossier.topic}")
    typer.echo(f"dossier    : {len(dossier.facts)} hechos de "
               f"{len(dossier.source_urls)} fuente(s)")
    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model} (modo {mode})")
    typer.echo(f"hasta {revisions + 1} ronda(s) de guion + informe...")

    report = producir(dossier, Screenwriter(modelo), Judge(modelo),
                   max_revisions=revisions, mode=mode)
    if report.failure:
        typer.secho(f"fallo a mitad del lazo — {report.failure}", fg=typer.colors.RED)

    for i, ronda in enumerate(report.rounds, start=1):
        typer.echo("")
        typer.secho(f"--- ronda {i} ---", bold=True)
        if ronda.write is not None and ronda.write.refusal:
            typer.secho(f"guionista rechazó: {ronda.write.refusal}", fg=typer.colors.RED)
        elif ronda.write is not None:
            intentos = len(ronda.write.attempts)
            typer.echo(f"guionista: {intentos} intento(s) mecánico(s)")
            for t in ronda.write.attempts:
                for v in t.violations:
                    typer.secho(f"  - {v}", fg=typer.colors.BRIGHT_BLACK)
        if ronda.review is not None and ronda.review.review is not None:
            _mostrar_parecer(ronda.review.review)

    coste = report.usage
    typer.echo("")
    typer.echo(f"coste total: {coste.input_tokens} tokens de entrada, "
               f"{coste.output_tokens} de salida, {report.latency_s}s de modelo")

    if not report.approved:
        typer.secho("ningún guion aprobado en las rondas disponibles; "
                    "el motivo de cada reprobación está arriba", fg=typer.colors.RED)
        if not dry_run and report.script is not None and report.review is not None:
            _grabar(store, report, dossier)
        raise typer.Exit(code=1)

    script = report.script
    typer.echo("")
    typer.secho(f"GUION APROBADO: {script.word_count} palabras "
                f"(~{script.estimated_duration_s:.0f}s)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {script.hook}")
    typer.echo(f"  términos: {', '.join(script.search_terms)}")

    if out is not None:
        out.write_text(
            script.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )
        typer.echo(f"\ngrabado en {out}")
        typer.echo(f"renderiza con: uv run agent render --script {out}")

    if dry_run:
        typer.secho("\n--dry-run: nada grabado en memoria", fg=typer.colors.YELLOW)
        return
    _grabar(store, report, dossier)


def _produce_carousel_cmd(store: Any, dossier: Any, llm: str, out: Path | None,
                          revisions: int, dry_run: bool) -> None:
    """Lazo de carrusel: escribe, juzga y revisa hasta pasar (corte 8/10)."""
    from agent.adapters.llm_factory import build_llm
    from agent.pipeline import produce_carousel as producir
    from agent.ports.llm import LLMError

    typer.echo(f"tema      : {dossier.topic}")
    try:
        modelo = build_llm(llm or None)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"modelo    : {modelo.provider}/{modelo.model} (modo carrusel)")

    report = producir(dossier, modelo, max_revisions=revisions)
    if report.failure:
        typer.secho(f"fallo a mitad del lazo — {report.failure}", fg=typer.colors.RED)
    for i, ronda in enumerate(report.rounds, start=1):
        typer.echo("")
        typer.secho(f"--- ronda {i} ---", bold=True)
        if ronda.write is not None:
            for t in ronda.write.attempts:
                for v in t.violations:
                    typer.secho(f"  - {v}", fg=typer.colors.BRIGHT_BLACK)
        if ronda.review is not None and ronda.review.review is not None:
            for s in ronda.review.review.scores:
                typer.echo(f"  [{s.score}/2] {s.criterion.value}: {s.reason}")

    coste = report.usage
    typer.echo(f"\ncoste total: {coste.input_tokens} in, {coste.output_tokens} out, "
               f"{report.latency_s}s")
    if not report.approved:
        typer.secho("ningún carrusel aprobado en las rondas.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    carrusel = report.carousel
    assert carrusel is not None
    typer.secho("CARRUSEL APROBADO", fg=typer.colors.GREEN, bold=True)
    if out is not None:
        out.write_text(
            carrusel.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8")
        typer.echo(f"grabado en {out}")
        typer.echo(f"slides con: uv run agent carousel-render --carousel {out}")
    if dry_run:
        typer.secho("\n--dry-run: nada grabado en memoria", fg=typer.colors.YELLOW)
        return
    fila = store.record_carousel(
        carrusel, model=modelo.model, provider=modelo.provider,
        usage=(coste.input_tokens, coste.output_tokens),
        latency_s=report.latency_s,
        attempts=[a.violations for r in report.rounds if r.write for a in r.write.attempts],
        review=report.review,
    )
    typer.echo(f"carrusel #{fila} grabado")


def _grabar(store: Any, report: Any, dossier: Any) -> None:
    """Graba guion e informe ligados, incluso cuando reprobado.

    Reprobado también entra: es el registro de en qué criterio la rúbrica falla con
    más frecuencia, y sin él calibrar la rúbrica sería un chute.
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
            for ronda in report.rounds if ronda.write is not None
            for t in ronda.write.attempts
        ],
        dossier_id=store.latest_dossier_id(dossier.topic),
    )
    review_id = store.record_review(
        report.review,
        usage=(report.usage.input_tokens, report.usage.output_tokens),
        latency_s=report.latency_s,
        script_id=script_id,
    )
    typer.echo(f"\nguion #{script_id} e informe #{review_id} grabados")


@app.command("llm-health")
def llm_health() -> None:
    """Comprueba si las claves y los identificadores de modelo configurados aún responden.

    El id de modelo de free tier se descontinua sin aviso, y el error aparece en
    medio de una investigacion, despues de gastar tiempo leyendo paginas. Este
    comando gasta una llamada minuscula y comprueba el artefacto: quien
    respondio, en cuanto tiempo, cobrando cuantos tokens.
    """
    from agent.adapters.llm_factory import build_llm, configured
    from agent.ports.llm import LLMError, parse_json_object

    proveedores = configured()
    if not proveedores:
        typer.secho(
            "ninguna clave de LLM configurada. La principal es pagada con crédito:\n"
            "  AGENT_OPENROUTER_API_KEY -> openrouter.ai/keys\n"
            "Las reservas son gratuitas:\n"
            "  AGENT_GROQ_API_KEY      -> console.groq.con/keys\n"
            "  AGENT_GEMINI_API_KEY    -> aistudio.google.con/apikey\n"
            "Grábala en .env (git-ignored).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    ok = True
    for nombre in proveedores:
        try:
            modelo = build_llm(nombre)
            respuesta = modelo.complete(
                'Responde {"ok": "sí"} y nada más.',
                schema={"type": "object", "properties": {"ok": {"type": "string"}},
                        "required": ["ok"]},
                max_output_tokens=256,
            )
            parse_json_object(respuesta.text)
        except LLMError as exc:
            typer.secho(f"[FALLO] {nombre}: {exc}", fg=typer.colors.RED)
            ok = False
            continue
        typer.secho(
            f"[OK  ] {nombre}: {respuesta.model} respondió en {respuesta.latency_s}s "
            f"({respuesta.usage.total_tokens} tokens)",
            fg=typer.colors.GREEN,
        )

    if not ok:
        raise typer.Exit(code=1)


@app.command()
def health() -> None:
    """Comprueba si el renderizador configurado está operativo (ffmpeg o MPT)."""
    if settings.renderer == "ffmpeg":
        from agent.adapters.ffmpeg_renderer import FfmpegRenderer
        from agent.render.post import ffmpeg_bin

        alive = FfmpegRenderer().health()
        typer.secho(f"ffmpeg ({ffmpeg_bin()}): {'ok' if alive else 'sin subtitles/concat'}",
                    fg=typer.colors.GREEN if alive else typer.colors.RED)
        mpt = MptRenderer().health()
        typer.echo(f"reserva MPT en {settings.renderer_url}: {'ok' if mpt else 'fuera del aire'}")
        sys.exit(0 if alive else 1)
    alive = MptRenderer().health()
    typer.secho(
        f"{settings.renderer_url}: {'ok' if alive else 'fuera del aire'}",
        fg=typer.colors.GREEN if alive else typer.colors.RED,
    )
    sys.exit(0 if alive else 1)


@app.command()
def publish(
    video: Path = typer.Option(..., "--video", "-v", exists=True, readable=True),
) -> None:
    """Sube un MP4 a la inbox de TikTok (scope video.upload, sin auditoría).

    El endpoint inbox solo recibe los bytes: título, descripción y rótulo AIGC son
    aplicados por ti en el app, al terminar el post por la notificación de la inbox.
    Por eso este comando termina con el checklist manual -- y el rótulo AIGC
    no  tiene flag para apagar porque no es opcional.
    """
    from agent.adapters.tiktok_publisher import TikTokPublisher
    from agent.memory.store import SignalStore
    from agent.models import PublishState

    token = settings.tiktok_access_token
    if not token:
        typer.secho(
            "sin AGENT_TIKTOK_ACCESS_TOKEN en .env (git-ignored).\n"
            "Registra el app en developers.tiktok.con, autoriza el scope "
            "video.upload y graba el token.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    typer.echo(f"video     : {video} ({video.stat().st_size} bytes)")
    typer.echo("subiendo a la inbox (init + chunks)...")
    result = TikTokPublisher().upload(str(video), access_token=token)

    store = SignalStore(settings.db_path)
    store.record_post(
        result.publish_id or "",
        str(video),
        status=result.state.value,
        error=result.error,
    )

    if result.state is not PublishState.uploaded:
        typer.secho(f"subida falló: {result.error}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"\npublish_id: {result.publish_id}", fg=typer.colors.GREEN)
    typer.echo("grabado en posts. Ahora, en el app del TikTok:")
    typer.echo("  1. abre la notificación de la inbox y concluye la edición;")
    typer.echo("  2. ACTIVA el rótulo de contenido generado por IA (obligatorio);")
    typer.echo("  3. confiere que la leyenda cita las fuentes del guion.")


@app.command("publish-status")
def publish_status(
    publish_id: str = typer.Option(..., "--publish-id"),
) -> None:
    """Consulta el estado de un post en la API y actualiza la tabla posts."""
    from agent.adapters.tiktok_publisher import TikTokPublisher
    from agent.memory.store import SignalStore

    token = settings.tiktok_access_token
    if not token:
        typer.secho("sin AGENT_TIKTOK_ACCESS_TOKEN en .env.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    try:
        estado = TikTokPublisher().fetch_status(publish_id, access_token=token)
    except Exception as exc:
        typer.secho(f"fallo: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    SignalStore(settings.db_path).update_post_status(publish_id, status=estado)
    typer.echo(f"{publish_id}: {estado}")


@app.command("tiktok-auth-url")
def tiktok_auth_url(
    state: str = typer.Option("", "--state"),
) -> None:
    """Imprime la URL para autorizar la app en el navegador (scope video.upload)."""
    from agent.adapters.tiktok_oauth import authorize_url

    if not settings.tiktok_client_key or not settings.tiktok_redirect_uri:
        typer.secho(
            "configura AGENT_TIKTOK_CLIENT_KEY y AGENT_TIKTOK_REDIRECT_URI en .env.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    typer.echo(authorize_url(
        settings.tiktok_client_key, settings.tiktok_redirect_uri, state=state,
    ))


@app.command()
def eval(
    topic: str = typer.Option("", "--topic", "-t", help="tema; sin eso, agrega todo"),
) -> None:
    """Tabula guionista x juez con coste medido, sin llamar a ningún modelo.

    Lee guiones e informes ya grabados e imprime: agregado por escritor, por
    juez (informe interrumpido fuera de la media), matriz pareada en el mismo guion,
    coste total y última métrica de cada post. Es informe, no  puerta: sale 0
    incluso con la base vacía.
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
            typer.secho(f"informe #{r['id']} con JSON inválido: {exc}",
                        fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        reviews.append(ReviewRow(
            id=r["id"], topic=r["topic"], script_id=r["script_id"],
            model=r["model"], provider=r["provider"], review=review,
            input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
            latency_s=r["latency_s"],
        ))

    if not scripts and not reviews and not store.list_carousels(topic or None):
        typer.echo("nada que agregar: sin guiones ni informes en memoria.")
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
        typer.echo("posts (última métrica leída en el app; la API de la inbox no la expone):")
        for p in posts:
            if p["publish_id"] in vistos:
                continue
            vistos.add(p["publish_id"])
            m = store.latest_metric(p["publish_id"])
            if m is None:
                typer.echo(f"  {p['publish_id']}: {p['status']} (sin métrica)")
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
                                    help="tiempo medio de reproducción en segundos"),
    completion: float = typer.Option(None, "--completion", min=0.0, max=1.0,
                                     help="fracción 0..1 que vio hasta el final"),
    script_id: int = typer.Option(None, "--script-id",
                                  help="guion que generó el video (cierra el lazo)"),
    saves: int = typer.Option(None, "--saves", min=0,
                              help="guardados (marcador del carrusel)"),
    comments: int = typer.Option(None, "--comments", min=0),
    shares: int = typer.Option(None, "--shares", min=0),
) -> None:
    """Graba una fila de métricas leída del app (views, watch, completion).

    Manual a propósito: en el scope video.upload de la inbox no  hay endpoint de
    métricas, y la Research API está restringida a investigación académica. Cada recogida
    entra en la serie del publish_id -- la curva, no  el número aislado, es la señal.
    saves/comments/shares son el marcador del carrusel y el desempate del video.
    """
    from agent.memory.store import SignalStore

    settings.ensure_dirs()
    try:
        fila = SignalStore(settings.db_path).record_metric(
            publish_id, views, script_id=script_id,
            avg_watch_s=avg_watch, completion_rate=completion,
            saves=saves, comments=comments, shares=shares,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(f"métrica #{fila} grabada para {publish_id}")


@app.command("carousel-render")
def carousel_render(
    carousel: Path = typer.Option(..., "--carousel", exists=True, readable=True),
    out_dir: Path = typer.Option(None, "--out-dir",
                                 help="carpeta de los slides; por defecto: output/carrusel-<ts>"),
    pillar: str = typer.Option("", "--pillar",
                               help="news|hechos|analisis|tutorial|futuro|vs; vacío = por tema"),
    fotos: bool = typer.Option(False, "--fotos/--sin-fotos",
                               help="fondo con foto Pexels por la etiqueta del slide"),
) -> None:
    """Renderiza las 5 diapositivas 1080x1920 del carrusel + caption.txt, en local."""
    from datetime import datetime

    from agent.brand.brand import suggest_content_pillar
    from agent.models import Carousel as CarouselModel
    from agent.render.carousel import render_carousel
    from agent.render.photos import fetch as fetch_photo

    try:
        modelo = CarouselModel.model_validate_json(carousel.read_text(encoding="utf-8"))
    except ValueError as exc:
        typer.secho(f"carrusel inválido: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    pilar = pillar or suggest_content_pillar(modelo.topic)
    destino = out_dir or settings.output_dir / (
        f"carrusel-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    typer.echo(f"pilar     : {pilar}")
    fotos_map: dict[int, Path | None] = {}
    if fotos:
        cache = settings.output_dir / "fotos-cache"
        for s in modelo.slides:
            camino = fetch_photo(s.visual, cache)
            fotos_map[s.n] = camino
            typer.echo(f"  slide {s.n}: {s.visual} -> "
                       + (camino.name if camino else "sin foto (layout puro)"))
    slides = render_carousel(modelo, destino, pilar, fotos_map or None)
    for s in slides:
        typer.echo(f"  {s} ({s.stat().st_size} bytes)")
    typer.echo(f"leyenda en {destino / 'caption.txt'}")
    _checar_slides(slides)


def _checar_slides(slides: list[Path]) -> None:
    """Aceite del carrusel: 5 PNG 1080x1920 con título legible medido."""
    from PIL import Image

    from agent.render.carousel import ink_height, legible

    ok = len(slides) == 5
    for s in slides:
        with Image.open(s) as img:
            ok = ok and img.size == (1080, 1920) and s.stat().st_size > 0
        if not legible(s):
            typer.secho(f"[FALLO] {s.name}: título con {ink_height(s)}px de tinta "
                        "(ilegible)", fg=typer.colors.RED)
            ok = False
    (typer.secho("[OK  ] 5 slides 1080x1920, título legible", fg=typer.colors.GREEN)
     if ok else typer.secho("[FALLO] slides fuera del aceite", fg=typer.colors.RED))
    if not ok:
        raise typer.Exit(code=1)


@app.command("voice-list")
def voice_list() -> None:
    """Biblioteca de voces y estilos (todo local, $0)."""
    from agent.brand.brand import load as load_brand
    from agent.voice.library import REJEITADAS, STYLES, VOICES

    typer.secho("VOZES (locutor real, un por modelo):", bold=True)
    for v in VOICES.values():
        typer.echo(f"  {v.id}: {v.nome} | {v.genero}, {v.idade_aparente} | "
                   f"{v.timbre} | licenca: {v.licenca}")
    typer.secho("\nESTILOS (interpretacao sobre os timbres):", bold=True)
    for s in STYLES.values():
        typer.echo(f"  {s.id}: voz={s.voz} vel={s.velocidade} "
                   f"pausa={s.pausa_frase_s}s -- {s.descricao}")
    brand = load_brand()
    typer.secho("\nAPRESENTADORES (elenco, no  marca):", bold=True)
    for p in brand.presenters.values():
        voz = p.library_voice or (
            "SIN VOZ ABIERTA (no hay una voz femenina de España en esta biblioteca)")
        typer.echo(f"  {p.id} ({p.name}): {p.role} | seed={p.seed} | voz={voz}")
    typer.echo("  formatos con avatar: "
               + ", ".join(brand.presenter_formats)
               + " -- fora deles, sem avatar")
    typer.secho("\nFora, con motivo:", bold=True)
    for k, motivo in REJEITADAS.items():
        typer.echo(f"  {k}: {motivo}")


@app.command("voice-fetch")
def voice_fetch(
    voice: str = typer.Option("", "--voice", help="id; vazio baixa todas"),
) -> None:
    """Descarga modelos de voz a data/voices/ (git-ignored, ~63 MB cada)."""
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
    """Adaptador Piper a partir de data/voices/. Falla explicando el descarga."""
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
    """Un texto, una voz, un wav. Muestra rapida antes de narrar el guion."""
    import time

    from agent.voice.narrate import Narrator

    corpo = text or (text_file.read_text(encoding="utf-8") if text_file else "")
    if not corpo.strip():
        typer.secho("texto vazio; use --text o  --text-file.", fg=typer.colors.RED)
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
    typer.echo(f"{out}: {nar.duration_s:.1f}s de audio en {parede:.1f}s "
               f"(RTF {parede / nar.duration_s:.2f}, {nar.utterances} falas)")


@app.command("voice-narrate")
def voice_narrate(
    script: Path = typer.Option(..., "--script", "-s", exists=True, readable=True),
    style: str = typer.Option("documental", "--style"),
    voice: str = typer.Option("", "--voice", help="cambia el locutor del estilo"),
    out: Path = typer.Option(..., "--out", "-o"),
) -> None:
    """Guion (hook+body+closing) a wav con el estilo del canal."""
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
        guion = Script.model_validate_json(script.read_text(encoding="utf-8"))
    except ValueError as exc:
        typer.secho(f"guion invalido: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    try:
        motor = _voice_engine(voice or perfil.voz)
    except (ValueError, Exception) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    inicio = time.monotonic()
    nar = Narrator(motor, pausa_frase_s=perfil.pausa_frase_s).narrate(
        guion.narration, speed=perfil.velocidad, noise=perfil.ruido)
    out.parent.mkdir(parents=True, exist_ok=True)
    nar.write_wav(str(out))
    parede = time.monotonic() - inicio
    typer.echo(f"{out}: {nar.duration_s:.1f}s en {parede:.1f}s "
               f"(RTF {parede / nar.duration_s:.2f}) estilo={style}")


@app.command("brand-avatar")
def brand_avatar(
    presenter: str = typer.Option(..., "--presenter", help="iris o  theo"),
    angulo: str = typer.Option("frontal", "--angulo"),
    expresion: str = typer.Option("neutra", "--expresion"),
    gesto: str = typer.Option("parada", "--gesto"),
) -> None:
    """Imprime el prompt fijo de generación del avatar + checklist de uso.

    La generación es externa (sin image-gen local $0): este comando garantiza que el
    prompt sale entero y que las trabas viajan junto -- seed, negativo,
    enquadramiento y límites.
    """
    from agent.brand.brand import avatar_prompt
    from agent.brand.brand import load as load_brand

    brand = load_brand()
    if presenter not in brand.presenters:
        typer.secho("presentador desconocido; usa iris o  theo.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)
    try:
        typer.echo(avatar_prompt(presenter, angulo=angulo, expresion=expresion,
                                 gesto=gesto))
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    p = brand.presenters[presenter]
    typer.echo("\n--- trabas ---")
    typer.echo(f"seed: {p.seed} (fijo, siempre) | formatos: {', '.join(p.formats)}")
    typer.echo("enquadramiento: 30-36% de la altura, derecha, pecho arriba, "
               "fondo transparente, texto del lado opuesto")
    typer.echo("escenificación: grande en la llamada, rincón en el cuerpo, vuelve en el cierre; "
               "nunca en perfil/logo/capa; rótulo AIGC activado")
    typer.echo("después de generar: guarda en brand/assets/presenters/source/<id>.jpg y "
               "corre scripts/make_presenter_cutouts.py (recorte + puntos del rosto)")
    typer.echo("\n--- negativo ---")
    typer.echo(brand.negative_prompt)


@app.command("rerender")
def rerender(
    script_path: Path = typer.Option(..., "--script", "-s", exists=True, readable=True),
    out_dir: Path = typer.Option(Path("output/_rerender"), "--out-dir"),
    pillar: str = typer.Option("", "--pillar",
                               help="por defecto: el pilar grabado en el propio guion"),
) -> None:
    """Rehace el VIDEO de un guion ya aprobado, sin gastar LLM ni parrilla.

    Existe porque lo más caro de probar un cambio de render es la parte
    que no  cambió: radar, investigación, guionista y juez gastan cuota, disputan
    pauta con los slots del día y pueden simplemente no  aprobar nada -- el día
    20/09/2026 terminó con cuatro temas intentados y ninguno aprobado, solo porque
    las rondas anteriores ya habían consumido las buenas pautas.

    Un guion que **ya pasó por el juez** es material de producción legítimo.
    De aquí en adelante rehace exactamente lo que cambió: clips, narración,
    presentador, leyenda, pista y posproducción, por el mismo camino que el
    piloto usa (`SlotRunner.render_video`) -- no es una segunda implementación
    que puede divergir de la de producción.
    """
    from agent.autopilot.runner import SlotRunner

    script = _load_script(script_path)
    pilar = pillar or script.pillar or "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"tema     : {script.topic}")
    typer.echo(f"pilar    : {pilar} | formato: {script.format}")
    typer.echo(f"narración : {script.word_count} palabras")

    runner = SlotRunner()
    try:
        video, medida = runner.render_video(script, out_dir, pilar)
    except (OSError, RuntimeError, ValueError) as exc:
        typer.secho(f"fallo: {type(exc).__name__}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho(f"\nvideo: {video}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {medida['width']}x{medida['height']}, {medida['duration_s']}s, "
               f"audio={'sí' if medida['has_audio'] else 'NO'}")


@app.command("presenter-preview")
def presenter_preview(
    script_path: Path = typer.Option(..., "--script", "-s", exists=True, readable=True),
    presenter: str = typer.Option("theo", "--presenter", help="iris o theo"),
    out_dir: Path = typer.Option(Path("output/_presentador"), "--out-dir"),
    seconds: float = typer.Option(0.0, "--seconds",
                                  help="solo los N primeros segundos (0 = todo)"),
    still: bool = typer.Option(False, "--still",
                               help="fuerza el retrato quieto, ignorando el clip base"),
) -> None:
    """Sintetiza solo la capa del presentador y una hoja de contactos para mirar.

    Sirve para comprobar el artefacto sin gastar una ronda de LLM ni un slot: la
    narracion sale del TTS (gratis), la capa sale en `presentador.mp4` y un
    cuadro de cada escenificacion se convierte en `hoja.png`. Fue asi como
    aparecieron la boca escancarada y la silueta en rectángulo -- mirando el
    pixel.

    Usa el clip base (`<id>_base.mp4`) cuando existe, que es el camino de
    producción desde el 20/09/2026; `--still` fuerza la reserva sintetizada.
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

    carpeta = PROJECT_ROOT / "brand" / "assets" / "presenters"
    clips = carpeta / f"{presenter}_base.json"
    png = carpeta / f"{presenter}.png"
    usa_clip = (not still and clips.exists()
                and clips.with_suffix(".mp4").exists())
    if not usa_clip and not (png.exists() and png.with_suffix(".json").exists()):
        typer.secho(f"falta {clips.name} (corre scripts/make_presenter_video.py) "
                    f"o {png.name} (corre scripts/make_presenter_cutouts.py)",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    script = _load_script(script_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    falado = respell(script.narration)
    audio = out_dir / "narracion.mp3"
    voz = VOZ_FEMININA if presenter == "iris" else VOZ_MASCULINA
    typer.echo(f"narrando ({voz})...")
    tiempos = synthesize(falado.text, audio, voice=voz)
    palabras = align(falado, tiempos)
    # El ffprobe del lado del ffmpeg estático -- el del Fedora no siempre está ahí
    # (ver setup_renderer.sh); sin ninguno de los dos, cae en la estimativa.
    sonda = Path(ffmpeg_bin()).with_name("ffprobe")
    dur = float(subprocess.run(
        [str(sonda) if sonda.exists() else "ffprobe", "-v", "error",
         "-show_entries", "format=duration", "-of", "csv=p=0", str(audio)],
        capture_output=True, text=True).stdout.strip() or 0)
    if not dur:
        dur = script.estimated_duration_s
    batidas = pr.tiempos_por_palabras(script.hook, script.closing, palabras, dur)
    if seconds:
        dur = min(dur, seconds)

    marca = load_brand()
    acento = marca.accent_for(script.pillar or "news")
    nombre = marca.presenters[presenter].name if presenter in marca.presenters else ""
    typer.echo(f"gancho hasta {batidas.hook_end:.1f}s, cierre en "
               f"{batidas.closing_start:.1f}s, {dur:.1f}s en total")
    typer.echo("material: " + ("clips base (boca sintetizada)" if usa_clip
                               else "retrato parado (todo sintetizado)"))

    inicio = time.monotonic()
    if usa_clip:
        base = pv.cargar_base(clips)
        capa = pv.render_layer(base, audio, out_dir / "presentador.mp4",
                               duracion=dur, batidas=batidas, accent=acento,
                               nombre=nombre, palabras_habladas=tiempos,
                               ffmpeg=ffmpeg_bin(), trabajo=out_dir)
    else:
        capa = pr.render_layer(pr.cargar(png), audio,
                               out_dir / "presentador.mp4", duracion=dur,
                               batidas=batidas, accent=acento, nombre=nombre,
                               ffmpeg=ffmpeg_bin(), semilla=script.topic)
    pared = time.monotonic() - inicio
    typer.echo(f"capa: {capa.frames} cuadros en {pared:.0f}s "
               f"({pared / max(capa.frames, 1) * 1000:.0f} ms/cuadro), "
               f"{capa.path.stat().st_size / 1e6:.1f} MB")
    typer.echo(f"leyenda: y={capa.subtitle_y} a partir de "
               f"{capa.subtitle_start:.1f}s (tarjeta sale junto)")

    enc = pr.plan(batidas)
    n = max(1, int(dur * pr.FPS))
    env = pr.envolvente(audio, n, pr.FPS, ffmpeg=ffmpeg_bin())
    hitos = [("llamada", min(batidas.hook_end * 0.6, dur - 0.1)),
             ("travesía", min(enc.leyenda_inicio - 0.3, dur - 0.1)),
             ("cuerpo", min((enc.leyenda_inicio + batidas.closing_start) / 2, dur - 0.1)),
             ("cierre", max(batidas.closing_start + 1.2, dur - 1.0))]
    if usa_clip:
        apertura, anchura = pv.pista(tiempos, n, pr.FPS)
        apertura = pv.modular(apertura, env)
        anim = pv.AnimadorVideo(base, pv.Cuadros(base, out_dir, ffmpeg=ffmpeg_bin()),
                                acento, nombre)

        def un(t: float):
            k = min(int(t * pr.FPS), n - 1)
            return anim.cuadro(t, float(apertura[k]), float(anchura[k]), enc.hitos)
    else:
        anim_p = pr.Animador(pr.cargar(png), acento, nombre)
        ventanas = pr.parpadeos(dur, script.topic)

        def un(t: float):
            k = min(int(t * pr.FPS), len(env) - 1)
            return anim_p.cuadro(t, float(env[k]), pr.cierre_en(ventanas, t), enc.hitos)

    tiras = []
    for _, t in hitos:
        cuadro = un(t)
        pantalla = Image.new("RGB", (pr.W, pr.H), (18, 20, 26))
        pantalla.paste(cuadro, (capa.x, capa.y), cuadro)
        tiras.append(pantalla.resize((pr.W // 4, pr.H // 4)))
    hoja = Image.new("RGB", (tiras[0].width * len(tiras), tiras[0].height))
    for i, t in enumerate(tiras):
        hoja.paste(t, (i * t.width, 0))
    hoja.save(out_dir / "hoja.png")
    typer.echo(f"hoja de contactos ({', '.join(m for m, _ in hitos)}): "
               f"{out_dir / 'hoja.png'}")


@app.command()
def status() -> None:
    """Resumen de la memoria: contexto y calidad de un vistazo."""
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
    typer.echo(f"señales={n('signals')} temas={n('topics')} dossiers={n('dossiers')} "
               f"guiones={n('scripts')} informes={n('reviews')} "
               f"carruseles={n('carousels')} posts={n('posts')} métricas={n('metrics')}")
    typer.echo(f"tokens totales medidos: {toks}")
    if temas:
        typer.echo("últimos temas: " + " | ".join(t[:48] for t in temas))
    for p in pend:
        m = store.latest_metric(p["publish_id"])
        extra = f" views={m['views']}" if m else ""
        typer.echo(f"  {p['publish_id'][:30]}: {p['status']}{extra}")


@app.command()
def preflight(
    video: Path = typer.Option(None, "--video",
                               help="mp4 del paquete (video); omítelo en carrusel"),
    script: Path = typer.Option(None, "--script", "-s",
                                help="guion.json o  carrusel.json del paquete"),
    slides_dir: Path = typer.Option(None, "--slides",
                                    help="carpeta de los slides (carrusel)"),
) -> None:
    """Última capa antes de producción: comprueba el paquete sin juzgar de nuevo.

    Puertas: informe aprobado ligado al texto, MP4 1080x1920 con audio en la
    franja del formato (o  5 slides + caption en carrusel), hechos con fuente.
    El checklist humano (AIGC, leyenda) sale al final, siempre.
    """
    from agent.memory.store import SignalStore
    from agent.models import Carousel
    from agent.publish.preflight import (
        CHECKLIST,
        preflight_carousel,
        preflight_video,
    )

    if script is None:
        typer.secho("pasa --script (guion o  carrusel del paquete).",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)
    if not script.exists():
        typer.secho(f"archivo ausente: {script}", fg=typer.colors.RED)
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
        mark = "OK  " if g.passed else "FALLO"
        typer.secho(f"[{mark}] {g.label}"
                    + (f" -- {g.detail}" if g.detail else ""),
                    fg=typer.colors.GREEN if g.passed else typer.colors.RED)
        ok = ok and g.passed

    typer.echo("\nchecklist humano (no  automatizable):")
    for item in CHECKLIST:
        typer.echo(f"  [ ] {item}")

    raise typer.Exit(code=0 if ok else 1)


@app.command("slots")
def slot_cmd(
    slots: str = typer.Option(..., "--slots", help="0900, 1500 o  2000 (hora de Madrid)"),
    day: str = typer.Option("", "--day", help="AAAA-MM-DD; por defecto: hoy en Madrid"),
    publish: bool = typer.Option(True, "--publish/--no-publish",
                                 help="sube el video a la inbox del TikTok"),
    wait: bool = typer.Option(True, "--wait/--no-wait",
                              help="espera la hora del slot para publicar"),
    force: bool = typer.Option(False, "--force", help="rehace slots ya concluido"),
) -> None:
    """Produce y publica UN slot del día, solo (es lo que llama el timer del systemd).

    Radar -> tema para el horario -> investigación -> formato por la información ->
    guion + juez (modelos enrutados por cuota) -> render medido -> espera la hora
    -> inbox del TikTok (video) o  paquete para postear (carrusel) -> aviso.
    """
    from datetime import date as _date

    from agent.autopilot.runner import SlotRunner
    from agent.editorial.slots import parse_slot, today

    try:
        destino = parse_slot(slots)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    dia = _date.fromisoformat(day) if day else today()
    resultado = SlotRunner().run(destino, dia, publish=publish, wait=wait, force=force)
    color = (typer.colors.GREEN if resultado.state in ("published", "ready_manual", "produced")
           else typer.colors.YELLOW if resultado.state == "skipped" else typer.colors.RED)
    typer.secho(f"\nslot {resultado.slots} {resultado.day}: {resultado.state.upper()}", fg=color,
                bold=True)
    for rotulo, valor in (("tema", resultado.topic), ("tipo", resultado.pillar),
                          ("formato", resultado.format), ("paquete", resultado.package_dir),
                          ("publish_id", resultado.publish_id), ("error", resultado.error)):
        if valor:
            typer.echo(f"  {rotulo:<10}: {valor}")
    raise typer.Exit(code=0 if resultado.state != "failed" else 1)


@app.command("slots-extra")
def slot_extra(
    force: bool = typer.Option(False, "--force", help="rehace la ronda extra de hoy"),
    publish: bool = typer.Option(True, "--publish/--no-publish",
                                 help="sube el video a la inbox del TikTok; "
                                      "--no-publish solo produce el paquete (conferir "
                                      "render sin gastar borrador de la inbox)"),
    formato: str = typer.Option("", "--format",
                                help="video (long+short), carrusel (carousel+short), "
                                     "short o  long (uno solo, sin reserva); "
                                     "vacío elige por el dossier"),
) -> None:
    """Ronda extra fuera de la parrilla: de la noticia nueva a la inbox, sin tocar los slots.

    Lee los temas/pilares/formatos que el día ya usó (no  repite), registra en
    fila propia (`extra`) que ningún timer lee, y publica en la hora (--no-wait).
    """
    from datetime import datetime as _datetime

    from agent.autopilot.runner import SlotRunner
    from agent.editorial.slots import TZ, Slot, today

    ahora = _datetime.now(TZ)
    destino = Slot("extra", ahora.time(), "extra",
                "ronda extra manual: del radar a inbox, fuera de la parrilla")
    fuerza = None
    if formato:
        clave = formato.strip().lower()
        if clave not in SlotRunner.FORMATOS_EXTRA:
            typer.secho(f"--format {formato!r} desconocido; usa "
                        + ", ".join(SlotRunner.FORMATOS_EXTRA) + ".",
                        fg=typer.colors.RED)
            raise typer.Exit(code=2)
        fuerza = {"extra": SlotRunner.FORMATOS_EXTRA[clave]}
    resultado = SlotRunner(formatos=fuerza).run(
        destino, today(), publish=publish, wait=False, force=force)
    color = (typer.colors.GREEN if resultado.state in ("published", "ready_manual", "produced")
           else typer.colors.YELLOW if resultado.state == "skipped" else typer.colors.RED)
    typer.secho(f"\nslot {resultado.slots} {resultado.day}: {resultado.state.upper()}", fg=color,
                bold=True)
    for rotulo, valor in (("tema", resultado.topic), ("tipo", resultado.pillar),
                          ("formato", resultado.format), ("paquete", resultado.package_dir),
                          ("publish_id", resultado.publish_id), ("error", resultado.error)):
        if valor:
            typer.echo(f"  {rotulo:<10}: {valor}")
    raise typer.Exit(code=0 if resultado.state != "failed" else 1)


@app.command("autopilot-status")
def autopilot_status(
    day: str = typer.Option("", "--day", help="AAAA-MM-DD; por defecto: hoy en Madrid"),
    detail: bool = typer.Option(False, "--detail", help="motivo completo de cada decisión"),
) -> None:
    """El día del piloto: cada slot, el motivo de las elecciones, cuota y coste por modelo."""
    import json as _json
    from datetime import UTC, datetime, timedelta

    from agent.autopilot.runs import SlotRuns
    from agent.editorial.slots import SLOTS, TZ, today
    from agent.memory.llm_ledger import LLMLedger

    settings.ensure_dirs()
    dia = day or today().isoformat()
    runs = {r["slots"]: r for r in SlotRuns(settings.db_path).day(dia)}
    typer.secho(f"piloto automático -- {dia} (hora de Madrid)", bold=True)
    for sid, slots in SLOTS.items():
        r = runs.get(sid)
        if r is None:
            typer.echo(f"  {slots.at:%H:%M}  aún no  rodó  ({slots.intent})")
            continue
        color = {"published": typer.colors.GREEN, "ready_manual": typer.colors.CYAN,
               "produced": typer.colors.BLUE, "failed": typer.colors.RED}.get(
                   r["state"], typer.colors.YELLOW)
        typer.secho(f"  {slots.at:%H:%M}  {r['state']:<12} {r['format'] or '-':<8} "
                    f"[{r['pillar'] or '-'}] {(r['topic'] or '')[:60]}", fg=color)
        if r["error"]:
            typer.secho(f"         error: {r['error'][:160]}", fg=typer.colors.RED)
        if r["package_dir"]:
            typer.echo(f"         paquete: {r['package_dir']}")
        if detail and r["plan_json"]:
            plano = _json.loads(r["plan_json"])
            for clave in ("topic_reason", "format_reason", "render_warning"):
                if plano.get(clave):
                    typer.echo(f"         {clave}: {plano[clave]}")
    libro = LLMLedger(settings.db_path)
    inicio = datetime.fromisoformat(dia).replace(tzinfo=TZ)
    fin = (inicio + timedelta(days=1)).astimezone(UTC).isoformat()
    uso: dict[str, dict[str, int]] = {}
    for c in libro.calls_since(inicio.astimezone(UTC)):
        if c["created_at"] >= fin:
            continue
        fila = uso.setdefault(c["route"], {"calls": 0, "failed": 0, "tokens": 0})
        fila["calls"] += 1
        fila["failed"] += 0 if c["ok"] else 1
        fila["tokens"] += int(c["input_tokens"]) + int(c["output_tokens"])
    if uso:
        typer.secho("\nLLM en el día (llamadas / fallos / tokens):", bold=True)
        for rota, u in sorted(uso.items()):
            typer.echo(f"  {rota:<42} {u['calls']:>3} / {u['failed']:>2} / {u['tokens']:>7}")
    agotados = libro.quota_status()
    if agotados:
        typer.secho("\nsin cuota ahora:", bold=True)
        for e in agotados:
            typer.echo(f"  {e['route']:<42} hasta {e['exhausted_until'][:16]} UTC  "
                       f"({e['reason'][:60]})")


if __name__ == "__main__":
    app()
