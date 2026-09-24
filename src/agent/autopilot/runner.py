"""Un slot del día, de punta a punta: pauta -> investigación -> formato ->
guion -> juez -> render -> paquete -> publicación -> aviso.

Cada etapa ya existía como comando manual; lo que este módulo añade es lo que
un humano hacía entre un comando y otro:

- **decidir** tema, tipo de contenido y formato para ESTA hora, sabiendo lo
  que el día ya ha publicado (`editorial/`), con el motivo grabado;
- **caer al siguiente** cuando algo no sostiene: tema sin dossier se convierte
  en el segundo de la lista; guion que no pasa en el formato preferido prueba
  el segundo formato; modelo sin cuota pasa al siguiente de la ruta
  (`router.py`);
- **medir el artefacto** antes de publicar (dimensión, audio, duración, título
  legible en la diapositiva) -- regla del proyecto desde el MP4 mudo;
- **esperar la hora** y publicar, renovando el token de TikTok antes.

Toda decisión va a `slot_runs.plan_json` y a `meta.json` en el paquete: el
post de las 20h tiene respuesta para "por qué este tema, por qué largo, qué
modelo escribió y qué costó".
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from agent.autopilot.notify import notify
from agent.autopilot.runs import SlotRuns
from agent.config import PROJECT_ROOT, Settings
from agent.config import settings as default_settings
from agent.editorial.formats import (
    FORMATO_FORZADO,
    FORMATS,
    choose_format,
    features,
    performance_from_metrics,
)
from agent.editorial.planner import TopicChoice, rank_topics
from agent.editorial.slots import TZ, Slot
from agent.memory.llm_ledger import LLMLedger
from agent.memory.store import SignalStore
from agent.models import CAROUSEL_MAX, Carousel, Decision, Dossier, PublishState, Script, Verdict
from agent.paths import slugify
from agent.ports.llm import LLM, Completion

# Cuántos temas intentar antes de rendir el slot, y cuántos formatos por tema.
MAX_TEMAS = 4
MAX_FORMATOS_POR_TEMA = 2
# Hechos que la investigación busca antes de dejar de leer fuentes.
OBJETIVO_HECHOS = 6
# Espera máxima por la hora del post: más que eso es reloj/timer erróneo, y
# publicar ya es mejor que dormir toda la tarde.
ESPERA_MAXIMA_S = 3 * 3600
# Retraso máximo para producir aún el slot. Con `Persistent=true`, encender la
# máquina a las 14h dispararía el slot de las 10h: dos posts en media hora
# compiten entre sí por la misma audiencia. Pasado eso, el slot se salta con
# motivo.
ATRASO_MAXIMO_S = 2 * 3600
# Franjas de aceptación del MP4 medido. El largo por debajo de 60s pierde la
# elegibilidad al Rewards, pero sigue siendo contenido: entre 55 y 60 publica
# con aviso.
RANGO_DURO = {"long": (55.0, 100.0), "short": (8.0, 35.0)}
RANGO_IDEAL = {"long": (60.0, 90.0), "short": (10.0, 30.0)}

# Tipos de contenido narrados por la voz masculina, en el reparto de la guía:
# Theo en tutorial y VS; el resto en la voz por defecto del canal (femenina,
# como Nova).
PILARES_THEO = frozenset({"tutorial", "vs"})
# Caché de clips y fotos: ~150 MB por vídeo largo. Tres slots al día llenan el
# disco en semanas; lo que no se usa en 10 días sale (el libro de reuso solo
# mira 14 días, así que el clip borrado tampoco volvería a ser elegido).
CACHE_DIAS = 10


@dataclass
class SlotResult:
    state: str
    slot: str
    day: str
    topic: str = ""
    pillar: str = ""
    format: str = ""
    package_dir: str = ""
    publish_id: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)


class SlotFailed(RuntimeError):
    """El slot no tiene qué publicar; el motivo va al libro y al aviso."""


class CrossFamilyJudge:
    """Puerto LLM del juez que prefiere la familia distinta a la del guionista.

    El proveedor del guionista solo se conoce después de que escribió (la ruta
    pudo haber caído en Groq), así que el orden del juez se decide en el
    momento de cada llamada.
    """

    def __init__(self, judge: LLM, writer: LLM):
        self._judge = judge
        self._writer = writer
        self._ultimo: LLM | None = None

    @property
    def provider(self) -> str:
        return getattr(self._ultimo or self._judge, "provider", "")

    @property
    def model(self) -> str:
        return getattr(self._ultimo or self._judge, "model", "")

    def complete(self, prompt: str, **kwargs) -> Completion:
        objetivo = self._judge
        preferir = getattr(self._judge, "prefer_other_than", None)
        escritor = getattr(self._writer, "provider", "")
        if callable(preferir) and escritor:
            objetivo = preferir(escritor)
        self._ultimo = objetivo
        return objetivo.complete(prompt, **kwargs)


class SlotRunner:
    # Orden de formatos fuera de la parrilla (ej. `slot-extra --format video`).
    # Misma semántica de FORMATO_FORZADO: prueba en orden, el último es
    # emergencia.
    FORMATOS_EXTRA: dict[str, tuple[str, ...]] = {
        "video": ("long", "short"),
        "carrusel": ("carousel", "short"),
        # Un formato solo, sin reserva: sirve para comprobar un cambio de
        # render en una pieza corta en vez de esperar un largo de 90s en cada
        # iteración.
        "short": ("short",),
        "long": ("long",),
    }

    def __init__(self, settings: Settings | None = None, *,
                 store: SignalStore | None = None, runs: SlotRuns | None = None,
                 ledger: LLMLedger | None = None,
                 now: Callable[[], datetime] | None = None,
                 sleep: Callable[[float], None] | None = None,
                 log: Callable[[str], None] | None = None,
                 formatos: dict[str, tuple[str, ...]] | None = None):
        self.cfg = settings or default_settings
        self.cfg.ensure_dirs()
        self.store = store or SignalStore(self.cfg.db_path)
        self.runs = runs or SlotRuns(self.cfg.db_path)
        self.ledger = ledger or LLMLedger(self.cfg.db_path)
        self._now = now or (lambda: datetime.now(TZ))
        self._sleep = sleep or time.sleep
        self._log_fn = log or (lambda m: print(m, flush=True))
        self.trail: list[str] = []
        self._formatos = formatos or {}

    def log(self, msg: str) -> None:
        linea = f"[{self._now().astimezone(TZ):%H:%M:%S}] {msg}"
        self.trail.append(linea)
        self._log_fn(linea)

    # ================================================================ flujo

    @staticmethod
    def _hora(slot: Slot) -> str:
        """Etiqueta horaria para avisos: '10h' en la parrilla, el id fuera de ella."""
        return f"{slot.id[:2]}h" if slot.id[:2].isdigit() else slot.id

    def run(self, slot: Slot, day: date, *, publish: bool = True, wait: bool = True,
            force: bool = False) -> SlotResult:
        dia = day.isoformat()
        retraso = (self._now() - slot.when(day)).total_seconds()
        if publish and wait and not force and retraso > ATRASO_MAXIMO_S:
            motivo = (f"slot {slot.id} disparó {retraso / 3600:.1f}h después de la hora "
                      "(¿máquina apagada?); saltado para no pegarse a otro post")
            actual = self.runs.get(dia, slot.id)
            if actual is None or actual["state"] not in ("published", "ready_manual"):
                self.runs.begin(dia, slot.id, force=True)
                self.runs.update(dia, slot.id, state="skipped", error=motivo)
            self.log(motivo)
            return SlotResult("skipped", slot.id, dia, error=motivo)
        if not self.runs.begin(dia, slot.id, force=force):
            previo = self.runs.get(dia, slot.id) or {}
            self.log(f"slot {slot.id} de {dia} ya concluido ({previo.get('state')}); "
                     "nada que hacer")
            return SlotResult("skipped", slot.id, dia, topic=previo.get("topic") or "")
        self.log(f"slot {slot.id} ({slot.label}) de {dia}: {slot.intent}")
        self._limpiar_cache()
        plan: dict = {"slot": slot.id, "day": dia}
        try:
            resultado = self._producir_y_publicar(slot, day, plan, publish=publish, wait=wait)
        except Exception as exc:  # noqa: BLE001 -- el slot nunca muere sin registro
            motivo = f"{type(exc).__name__}: {exc}"
            self.log(f"FALLO: {motivo}")
            plan["error_trace"] = traceback.format_exc()[-2000:]
            plan["log"] = self.trail
            self.runs.update(dia, slot.id, state="failed", error=motivo[:500], plan=plan)
            notify(f"Tu Canal: slot {self._hora(slot)} falló",
                   f"{motivo}\n\nNo se publicó nada en esta franja. Detalles: "
                   f"uv run agent autopilot-status",
                   ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
                   urgent=True)
            return SlotResult("failed", slot.id, dia, error=motivo)
        return resultado

    def _producir_y_publicar(self, slot: Slot, day: date, plan: dict, *,
                             publish: bool, wait: bool) -> SlotResult:
        dia = day.isoformat()
        usados = self.runs.used_today(dia, except_slot=slot.id)
        plan["used_today"] = usados

        candidatos = self.candidates(usados)
        elecciones = rank_topics(candidatos, slot.id, usados["pillars"], limit=10)
        elecciones, nota_interes = self.rerank(elecciones)
        self.log(f"pauta: {nota_interes}")
        plan["interest"] = nota_interes
        plan["shortlist"] = [
            {"term": c.decision.term, "source": c.decision.source, "pillar": c.pillar,
             "score": c.score, "reason": c.reason} for c in elecciones[:8]]
        if not elecciones:
            raise SlotFailed("ningún tema elegible en el radar (política, nicho y repetición)")

        paquete = None
        intentos: list[dict] = []
        for eleccion in elecciones[:MAX_TEMAS]:
            self.log(f"tema: {eleccion.decision.term[:80]} [{eleccion.pillar}] "
                     f"({eleccion.decision.source}, nota {eleccion.score:.2f})")
            try:
                paquete = self._intentar_tema(eleccion, slot, day, usados, plan, intentos)
            except SlotFailed as exc:
                intentos.append({"term": eleccion.decision.term, "fallo": str(exc)})
                self.log(f"  descartado: {exc}")
                continue
            if paquete is not None:
                break
        plan["attempts"] = intentos
        if paquete is None:
            fallos = [t["fallo"][:80] for t in intentos if t.get("fallo")]
            raise SlotFailed(f"{len(fallos)} tema(s) intentados, ninguno se volvió pieza "
                             "aprobada: " + "; ".join(fallos))

        # ------------------------------------------------ hora del post
        objetivo = slot.when(day)
        ahora = self._now()
        if wait and publish and ahora < objetivo:
            espera = (objetivo - ahora).total_seconds()
            if espera <= ESPERA_MAXIMA_S:
                self.log(f"paquete listo; esperando {espera / 60:.0f} min hasta "
                         f"{objetivo:%H:%M}")
                self.runs.update(dia, slot.id, state="produced")
                self._sleep(espera)
        return self._publicar(paquete, slot, day, plan, publish=publish)

    # ================================================================ etapas

    def candidates(self, usados: dict[str, list[str]]) -> list[Decision]:
        """Radar + curador, con lo que el día ya cubrió en el libro de repetición."""
        from agent.curator.curator import Curator
        from agent.radar.collector import Radar, default_sources

        colecta = Radar(default_sources(), self.store).collect()
        self._senales = colecta.signals
        for nombre, error in colecta.failures.items():
            self.log(f"  [fuente caída] {nombre}: {error[:100]}")
        if not colecta.signals:
            raise SlotFailed("radar sin ninguna señal (todas las fuentes fuera)")
        libro = self.store.recent_topics(days=30) + usados["topics"]
        report = Curator().curate(colecta.signals, ledger=libro)
        # El "selected" del curador todavía no es elección del slot: grabarlo así
        # sacaría el tema del libro de las próximas horas sin que se vuelva vídeo.
        self.store.record_decisions([
            d.model_copy(update={"verdict": Verdict.not_selected})
            if d.verdict is Verdict.selected else d for d in report.decisions])
        self.log(f"radar: {len(colecta.signals)} señales, {len(report.eligible)} elegibles "
                 f"({', '.join(colecta.sources_ok)})")
        return report.top(12)

    def rerank(self, elecciones: list[TopicChoice]) -> tuple[list[TopicChoice], str]:
        """Nota de interés de la audiencia (1 llamada) combinada con radar + hora."""
        from agent.editorial.interest import rerank

        return rerank(elecciones, self.llm("ranker"))

    def llm(self, stage: str) -> LLM:
        """El puerto LLM de la etapa: ruta con cambio de modelo por cuota."""
        from agent.adapters.llm_factory import build_routed

        return build_routed(stage, self.cfg, self.ledger)

    def _intentar_tema(self, eleccion: TopicChoice, slot: Slot, day: date,
                       usados: dict[str, list[str]], plan: dict,
                       intentos: list[dict]) -> dict | None:
        decision = eleccion.decision
        from agent.research.related import related_items

        extras = related_items(decision, getattr(self, "_senales", []))
        if extras:
            decision = decision.model_copy(update={"news_items": [*decision.news_items, *extras]})
            self.log(f"  +{len(extras)} fuente(s) de la misma historia en la colecta: "
                     + ", ".join(n.source_name for n in extras))
        investigacion = self.research(decision, self.llm("research"))
        if investigacion.dossier is None:
            raise SlotFailed(f"investigación sin dossier ({len(investigacion.pages)} páginas "
                             f"leídas, fallos: {list(investigacion.failures)[:3]})")
        dossier = investigacion.dossier
        ruta = getattr(investigacion, "route", "") or investigacion.model
        dossier_id = self.store.record_dossier(
            dossier, model=ruta, provider=ruta.split(":", 1)[0] if ":" in ruta else "",
            usage=(investigacion.usage.input_tokens, investigacion.usage.output_tokens),
            latency_s=investigacion.latency_s, source_count=investigacion.source_count,
            discarded=[vars(d) for d in investigacion.discarded],
            failures=investigacion.failures)
        self.log(f"  dossier #{dossier_id}: {len(dossier.facts)} hechos de "
                 f"{investigacion.source_count} fuente(s), "
                 f"{len(investigacion.discarded)} descartes")

        feat = features(dossier)
        rendimiento = performance_from_metrics(self.store.format_performance_rows())
        forzados = self._formatos.get(slot.id, FORMATO_FORZADO.get(slot.id))
        # `en_orden` solo cuando hay orden forzada: ahí la parrilla manda, y la
        # nota apenas explica. Sin parrilla forzada la nota decide, como siempre.
        formato = choose_format(feat, eleccion.pillar, slot.id, usados["formats"], rendimiento,
                                allowed=forzados or FORMATS, en_orden=bool(forzados))
        motivo_orden = (f"orden forzada {list(forzados)}" if forzados else "orden por nota")
        self.log(f"  formato: {formato.reason} [{motivo_orden}]")

        registro = {"term": decision.term, "pillar": eleccion.pillar,
                    "format_decision": formato.reason, "scores": formato.scores,
                    "dossier_id": dossier_id, "facts": len(dossier.facts)}
        orden = (list(forzados) if forzados
                 else [f for f in formato.ranked() if f in FORMATS])
        # Con orden forzada, intenta en el orden dado saltándose lo que el
        # dossier no sostiene (fuera del ranked), pero siempre prueba el último
        # (emergencia).
        candidatos = ([f for f in orden if f in formato.ranked() or f == orden[-1]]
                      if forzados else orden)
        for fmt in candidatos[:MAX_FORMATOS_POR_TEMA]:
            # El escritor depende del formato: corto abre en Groq (techo de
            # palabras), el resto en el premium de OpenRouter. El juez cruza la
            # familia con el escritor de ese formato, no con el anterior.
            etapa = "writer_short" if fmt == "short" else "writer"
            writer = self.llm(etapa)
            judge = CrossFamilyJudge(self.llm("judge"), writer)
            pieza = self.write(dossier, fmt, eleccion.pillar, writer, judge)
            registro.setdefault("writes", []).append(pieza["summary"])
            if pieza["approved"]:
                plan.update({
                    "topic": decision.term, "source": decision.source,
                    "topic_reason": eleccion.reason, "pillar": eleccion.pillar,
                    "pillar_guesses": [vars(g) for g in eleccion.guesses[:3]],
                    "format": fmt, "format_reason": formato.reason,
                    "format_scores": formato.scores, "dossier_id": dossier_id,
                    "research": {"sources": [p.url for p in investigacion.pages],
                                 "discarded": len(investigacion.discarded),
                                 "failures": investigacion.failures,
                                 "tokens": investigacion.usage.total_tokens},
                })
                intentos.append(registro)
                return self._montar_paquete(pieza, decision, eleccion.pillar, fmt,
                                            dossier, slot, day, plan)
            self.log(f"  {fmt} reprobado: {pieza['summary'].get('motivo', '')[:120]}")
        intentos.append(registro)
        raise SlotFailed("ningún formato aprobado por el juez para este tema")

    def research(self, decision: Decision, llm: LLM):
        from agent.research.fetch import PageFetcher
        from agent.research.researcher import Researcher

        investigador = Researcher(
            llm, fetcher=PageFetcher(max_chars=self.cfg.research_page_chars),
            max_sources=self.cfg.research_max_sources,
            max_facts_per_source=self.cfg.research_max_facts_per_source)
        informe = investigador.research(decision, target_facts=OBJETIVO_HECHOS)
        # Qué modelo de la ruta respondió el último (el informe nace antes).
        informe.route = getattr(llm, "last_route", "")
        return informe

    def write(self, dossier: Dossier, fmt: str, pillar: str, writer: LLM,
              judge: LLM) -> dict:
        """Escribe y juzga en el formato; graba guion/dictamen incluso reprobados."""
        from agent.judge.judge import Judge
        from agent.pipeline import produce, produce_carousel
        from agent.writer.writer import Screenwriter

        dossier_id = self.store.latest_dossier_id(dossier.topic)
        if fmt == "carousel":
            rep = produce_carousel(dossier, writer, judge_llm=judge, pillar=pillar)
            carrusel = rep.carousel
            carousel_id = None
            if carrusel is not None:
                carousel_id = self.store.record_carousel(
                    carrusel, model=getattr(writer, "model", ""),
                    provider=getattr(writer, "provider", ""),
                    usage=(rep.usage.input_tokens, rep.usage.output_tokens),
                    latency_s=rep.latency_s,
                    attempts=[a.violations for r in rep.rounds if r.write
                              for a in r.write.attempts],
                    review=rep.review)
            resumen = {"format": fmt, "approved": rep.approved, "rounds": len(rep.rounds),
                       "tokens": rep.usage.total_tokens, "failure": rep.failure,
                       "writer": (
                           f"{getattr(writer, 'provider', '')}/"
                           f"{getattr(writer, 'model', '')}"),
                       "judge": f"{judge.provider}/{judge.model}",
                       "motivo": _motivo_carrusel(rep),
                       # Todo intento reprobado queda grabado (regla del
                       # proyecto): carrusel que no pasó las puertas no se
                       # vuelve fila en `carousels`, así que las violaciones
                       # van al plan del slot.
                       "attempts": [a.violations for r in rep.rounds if r.write
                                    for a in r.write.attempts]}
            return {"approved": rep.approved and carrusel is not None, "carousel": carrusel,
                    "carousel_id": carousel_id, "summary": resumen}

        rep = produce(dossier, Screenwriter(writer), Judge(judge), mode=fmt, pillar=pillar)
        script_id = None
        if rep.script is not None:
            ultima = next((r for r in reversed(rep.rounds) if r.write is not None), None)
            script_id = self.store.record_script(
                rep.script,
                model=ultima.write.model if ultima else "",
                provider=ultima.write.provider if ultima else "",
                usage=(rep.usage.input_tokens, rep.usage.output_tokens),
                latency_s=rep.latency_s,
                attempts=[{"violations": t.violations, "word_count": t.word_count}
                          for r in rep.rounds if r.write for t in r.write.attempts],
                dossier_id=dossier_id)
            if rep.review is not None:
                self.store.record_review(
                    rep.review, usage=(rep.usage.input_tokens, rep.usage.output_tokens),
                    latency_s=rep.latency_s, script_id=script_id)
        resumen = {"format": fmt, "approved": rep.approved, "rounds": len(rep.rounds),
                   "tokens": rep.usage.total_tokens, "failure": rep.failure,
                   "writer": f"{getattr(writer, 'provider', '')}/{getattr(writer, 'model', '')}",
                   "judge": f"{judge.provider}/{judge.model}",
                   "total": rep.review.total if rep.review else None,
                   "motivo": _motivo_video(rep)}
        return {"approved": rep.approved and rep.script is not None, "script": rep.script,
                "script_id": script_id, "summary": resumen}

    # ================================================================ paquete

    def _montar_paquete(self, pieza: dict, decision: Decision, pillar: str, fmt: str,
                        dossier: Dossier, slot: Slot, day: date, plan: dict) -> dict:
        dia = day.isoformat()
        carpeta = (self.cfg.output_dir / dia
                   / f"{slot.id}-{slugify(decision.term)}-{fmt}")
        carpeta.mkdir(parents=True, exist_ok=True)
        self.runs.update(dia, slot.id, topic=decision.term, source=decision.source,
                         pillar=pillar, format=fmt, package_dir=str(carpeta))
        # Ahora sí el tema fue elegido: entra en el libro de repetición.
        self.store.record_decisions([decision.model_copy(update={
            "verdict": Verdict.selected,
            "reason": f"piloto {dia} {slot.id}: {plan.get('topic_reason', '')}"[:500],
            "decided_at": datetime.now(UTC)})])

        if fmt == "carousel":
            carrusel: Carousel = pieza["carousel"]
            (carpeta / "carrusel.json").write_text(
                carrusel.model_dump_json(indent=2) + "\n", encoding="utf-8")
            slides = self.render_carousel(carrusel, carpeta / "slides", pillar)
            self._aceptacion_slides(slides)
            caption = (carpeta / "slides" / "caption.txt").read_text(encoding="utf-8")
            (carpeta / "caption.txt").write_text(caption, encoding="utf-8")
            artefacto = {"kind": "carousel", "slides": [str(s) for s in slides],
                         "carousel_id": pieza.get("carousel_id")}
        else:
            script: Script = pieza["script"]
            (carpeta / "guion.json").write_text(
                script.model_dump_json(indent=2) + "\n", encoding="utf-8")
            video, medida = self.render_video(script, carpeta, pillar)
            aviso = self._aceptacion_video(medida, fmt)
            if aviso:
                plan["render_warning"] = aviso
                self.log(f"  aviso: {aviso}")
            caption = _caption_video(script)
            (carpeta / "caption.txt").write_text(caption, encoding="utf-8")
            artefacto = {"kind": "video", "video": str(video), "probe": medida,
                         "script_id": pieza.get("script_id")}
        plan["artifact"] = artefacto
        plan["writes"] = pieza["summary"]
        plan["llm_trail"] = self._traza_llm()
        plan["log"] = self.trail
        (carpeta / "meta.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")
        self.runs.update(dia, slot.id, state="produced", plan=plan)
        self.log(f"  paquete listo en {carpeta}")
        return {"dir": carpeta, "format": fmt, "pillar": pillar, "topic": decision.term,
                "caption": caption, **artefacto}

    def render_video(self, script: Script, carpeta: Path, pillar: str) -> tuple[Path, dict]:
        """Clips elegidos -> narración + leyenda (ffmpeg) -> pista + marca -> MP4 medido.

        El renderizador propio es el estándar; el MPT entra si falla o si la
        selección de clips no entrega material.
        """
        from agent.adapters.mpt_renderer import probe_video
        from agent.brand.brand import load as load_brand
        from agent.render.footage import FootageLedger, PexelsFootage
        from agent.render.music import render_bed
        from agent.render.post import PostSpec, postprocess
        from agent.render.presenter import conferir_presentador

        trabajo = carpeta / "_work"
        materiales: list[Path] = []
        try:
            footage = PexelsFootage(self.cfg.pexels_api_key,
                                    self.cfg.data_dir / "footage-cache",
                                    ledger=FootageLedger(self.cfg.db_path))
            plan = footage.plan(script)
            materiales = plan.paths
            self.log(f"  clips: {len(materiales)} elegidos "
                     f"({', '.join(sorted({c.term for c in plan.clips}))[:120]})"
                     + (f"; faltas: {plan.misses[:3]}" if plan.misses else ""))
        except (ValueError, OSError) as exc:
            self.log(f"  selección de clips no disponible ({exc})")

        marca = load_brand()
        # `pillar` sigue siendo el id (string). El objeto va en otro nombre a
        # propósito: `presenter_for` y `accent_for` reciben el id, y pasar el
        # objeto hacía que las dos cayeran en el defecto en silencio -- ningún
        # presentador aparecía y todo vídeo salía en verde, incluso en los
        # pilares de acento cian.
        pilar_obj = marca.pillars.get(pillar)
        pedido = self._pedido_presentador(marca, pillar)

        # La voz sigue a quien aparece. Con ATLAS presentando todos los pilares
        # (20/09/2026), rostro masculino con narración femenina sería una
        # incoherencia mayor que cualquier defecto de boca -- entonces el pilar
        # solo decide la voz cuando no hay presentador en cuadro.
        voz = (self.cfg.narrator_voice_male
               if (pedido is not None or pillar in PILARES_THEO)
               else self.cfg.narrator_voice)
        resultado, capa = (None, None)
        if self.cfg.renderer == "ffmpeg" and materiales:
            resultado, capa = self._render_ffmpeg(script, materiales, voz, trabajo,
                                                  pedido)
        if resultado is None:
            resultado = self._render_mpt(script, materiales)
        bruto = Path(resultado.video_path)
        duracion = float(resultado.duration_s or script.estimated_duration_s)
        pista = render_bed(trabajo / "pista.wav", duracion, pillar, seed=script.topic)
        spec = PostSpec(hook=script.hook, tag=pilar_obj.tag if pilar_obj else "",
                        accent=marca.accent_for(pillar), handle=marca.handle,
                        presenter=capa)
        final = postprocess(bruto, carpeta / "video.mp4", spec, duration_s=duracion,
                            music_wav=pista, work_dir=trabajo)
        medida = probe_video(final)
        if capa is not None:
            aviso_pres = conferir_presentador(final, capa, trabajo)
            self.log(f"  {aviso_pres}")
        bruto.unlink(missing_ok=True)
        shutil.rmtree(trabajo, ignore_errors=True)
        self.log(f"  vídeo: {medida['width']}x{medida['height']}, "
                 f"{medida['duration_s']}s, "
                 f"audio={'sí' if medida['has_audio'] else 'NO'}")
        return final, medida

    def _pedido_presentador(self, marca, pillar: str):
        """Quien presenta este pilar, y con qué material.

        Los dos materiales se generan fuera del bucle, porque los dos dependen
        de rembg y mediapipe (>400 MB que el agente nunca necesita cargar tres
        veces al día):

        - `<id>_base.mp4` + `.json` (`scripts/make_presenter_video.py`) -- el
          clip fotorrealista. Preferido: el parpadeo y el balanceo de cabeza
          son humanos, generados como vídeo, y solo la boca es nuestra.
        - `<id>.png` + `.json` (`scripts/make_presenter_cutouts.py`) -- el
          retrato parado, del que todo es sintetizado. Reserva.

        Sin ninguno de los dos el vídeo sale sin presentador y el motivo va al
        log: avatar es acabado, y un slot sin él sigue siendo slot publicable.
        """
        from agent.adapters.ffmpeg_renderer import PresenterRequest

        presentador = marca.presenter_for(pillar)
        if presentador is None:
            return None
        carpeta = PROJECT_ROOT / "brand" / "assets" / "presenters"
        clip = carpeta / f"{presentador.id}_base.json"
        png = carpeta / f"{presentador.id}.png"
        hay_clip = clip.exists() and clip.with_suffix(".mp4").exists()
        hay_png = png.exists() and png.with_suffix(".json").exists()
        if not hay_clip and not hay_png:
            self.log(f"  presentador: {presentador.name} sin material "
                     f"(ni {clip.name} ni {png.name} -- ejecuta "
                     f"scripts/make_presenter_video.py)")
            return None
        self.log(f"  presentador: {presentador.name} "
                 + ("(clip base, boca sintetizada)" if hay_clip
                    else "(retrato parado sintetizado -- sin clip base)"))
        return PresenterRequest(cutout=png if hay_png else None,
                                name=presentador.name,
                                base=clip if hay_clip else None)

    def _render_ffmpeg(self, script: Script, materiales: list[Path], voz: str,
                       trabajo: Path, pedido=None):
        from agent.adapters.ffmpeg_renderer import FfmpegRenderer
        from agent.models import RenderState
        from agent.ports.renderer import RendererError

        renderer = FfmpegRenderer(self.cfg)
        self.log(f"  renderizando (ffmpeg, voz {voz})...")
        try:
            resultado = renderer.render(script, materials=materiales, voice=voz,
                                        work_dir=trabajo, presenter=pedido)
        except RendererError as exc:
            self.log(f"  renderizador propio falló ({exc}); probando el MPT")
            return None, None
        if resultado.state is RenderState.failed or not resultado.video_path:
            self.log(f"  renderizador propio falló ({resultado.error}); probando el MPT")
            return None, None
        if renderer.presenter_error:
            self.log(f"  presentador no entró: {renderer.presenter_error}")
        elif renderer.last_presenter is not None:
            c = renderer.last_presenter
            self.log(f"  presentador: {c.frames} cuadros a {c.fps} fps, "
                     f"leyenda en y={c.subtitle_y} desde {c.subtitle_start:.1f}s")
        if renderer.last_changes:
            cambios = sorted({f"{a}->{b}" for a, b in renderer.last_changes})
            self.log(f"  pronunciación: {', '.join(cambios)[:160]}")
        revision = renderer.last_check
        if revision is not None:
            if revision.error:
                self.log(f"  revisión de pronunciación: {revision.error}")
            elif revision.suspicious:
                self.log("  AVISO pronunciación: el Whisper no reconoció "
                         f"{revision.suspicious} -- candidatos al léxico")
                self._sospechosos(revision.suspicious, script)
            else:
                self.log("  revisión de pronunciación: todos los nombres reconocidos")
        return resultado, renderer.last_presenter

    def _render_mpt(self, script: Script, materiales: list[Path]):
        from agent.adapters.mpt_renderer import MptRenderer
        from agent.models import RenderState

        renderer = MptRenderer(self.cfg)
        if not renderer.health():
            raise SlotFailed(f"renderizador propio falló y el MPT está caído en "
                             f"{self.cfg.renderer_url}")
        self.log(f"  renderizando en el MPT (voz {self.cfg.voice_name})...")
        resultado = renderer.render(script, materials=materiales or None)
        if resultado.state is RenderState.failed or not resultado.video_path:
            raise SlotFailed(f"render falló: {resultado.error}")
        return resultado

    def _sospechosos(self, nombres: list[str], script: Script) -> None:
        """Registra nombre mal pronunciado para convertirse en entrada del léxico."""
        archivo = self.cfg.data_dir / "pronuncia_sospechosa.txt"
        with archivo.open("a", encoding="utf-8") as fh:
            for n in nombres:
                fh.write(f"{datetime.now(UTC):%Y-%m-%d %H:%M}\t{n}\t{script.topic[:60]}\n")

    def render_carousel(self, carrusel: Carousel, carpeta: Path, pillar: str) -> list[Path]:
        from agent.render.carousel import render_carousel
        from agent.render.photos import fetch

        cache = self.cfg.data_dir / "photos-cache"
        usadas: set[int] = set()
        fotos: dict[int, Path | None] = {}
        broll = list(carrusel.broll)
        for s in carrusel.slides:
            if s.n == 1 and broll:
                fotos[s.n] = fetch(broll[0], cache, subject=True, used=usadas)
            elif s.n == 3 and len(broll) > 1:
                fotos[s.n] = fetch(broll[1], cache, subject=True, used=usadas)
            else:
                fotos[s.n] = fetch(s.visual, cache, used=usadas)
        self.log(f"  fotos: {sum(1 for f in fotos.values() if f)}/5 diapositivas con foto")
        return render_carousel(carrusel, carpeta, pillar, fotos)

    # ================================================================ aceptación

    @staticmethod
    def _aceptacion_video(medida: dict, fmt: str) -> str:
        if (medida.get("width"), medida.get("height")) != (1080, 1920):
            raise SlotFailed(f"vídeo fuera de 1080x1920: {medida}")
        if not medida.get("has_audio"):
            raise SlotFailed("vídeo sin pista de audio (el defecto del MP4 mudo)")
        dur = medida.get("duration_s") or 0.0
        duro = RANGO_DURO.get(fmt, RANGO_DURO["long"])
        if not duro[0] <= dur <= duro[1]:
            raise SlotFailed(f"duración {dur}s fuera de lo aceptable {duro} para {fmt}")
        ideal = RANGO_IDEAL.get(fmt, RANGO_IDEAL["long"])
        if not ideal[0] <= dur <= ideal[1]:
            return f"duración {dur}s fuera de la franja ideal {ideal} del formato {fmt}"
        return ""

    @staticmethod
    def _aceptacion_slides(slides: list[Path]) -> None:
        from PIL import Image

        from agent.render.carousel import ink_height, legible

        if len(slides) != 5:
            raise SlotFailed(f"carrusel con {len(slides)} diapositivas")
        for s in slides:
            with Image.open(s) as img:
                if img.size != (1080, 1920):
                    raise SlotFailed(f"{s.name} fuera de 1080x1920")
            if not legible(s):
                raise SlotFailed(f"{s.name} con título ilegible ({ink_height(s)}px)")

    # ================================================================ publicación

    def _publicar(self, paquete: dict, slot: Slot, day: date, plan: dict, *,
                  publish: bool) -> SlotResult:
        dia = day.isoformat()
        carpeta: Path = paquete["dir"]
        base = SlotResult("produced", slot.id, dia, topic=paquete["topic"],
                          pillar=paquete["pillar"], format=paquete["format"],
                          package_dir=str(carpeta))
        if not publish:
            self.log("publicación desactivada (--no-publish): el paquete queda listo")
            self.runs.update(dia, slot.id, state="produced", plan=plan)
            return base

        if paquete["kind"] == "carousel":
            # La API de foto de TikTok solo acepta PULL_FROM_URL de prefijo
            # verificado. Con el alojamiento configurado (GitHub Pages), el
            # carrusel va a la inbox como el vídeo; sin ella, paquete manual.
            if self.cfg.media_repo_dir and self.cfg.media_base_url:
                resultado = self.publish_carousel(paquete, slot, day)
                plan["publish"] = {"state": resultado.state.value,
                                   "publish_id": resultado.publish_id,
                                   "error": resultado.error}
                if resultado.state is PublishState.uploaded:
                    self.runs.update(dia, slot.id, state="published", plan=plan,
                                     publish_id=resultado.publish_id)
                    notify(f"Tu Canal {self._hora(slot)}: carrusel en la inbox de TikTok",
                           f"{paquete['topic'][:90]}\n\nEn la app: abre la notificación de "
                           "la inbox, revisa la caption y ACTIVA la etiqueta de contenido "
                           "generado por IA.",
                           ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
                           package_dir=carpeta)
                    base.state, base.publish_id = "published", resultado.publish_id or ""
                    return base
                self.log(f"carrusel por API falló ({resultado.error}); va como paquete")
            self.runs.update(dia, slot.id, state="ready_manual", plan=plan)
            self._avisar_manual(paquete, slot)
            base.state = "ready_manual"
            return base

        resultado = self.publish_video(Path(paquete["video"]), paquete, slot)
        plan["publish"] = {"state": resultado.state.value, "publish_id": resultado.publish_id,
                           "error": resultado.error}
        if resultado.state is not PublishState.uploaded:
            self.runs.update(dia, slot.id, state="failed", plan=plan,
                             error=f"publicación: {resultado.error}"[:500])
            notify(f"Tu Canal: vídeo {self._hora(slot)} NO subió",
                   f"{resultado.error}\n\nEl vídeo está listo en {carpeta}/video.mp4 -- "
                   "súbelo a mano por la app si quieres mantener la hora.",
                   ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
                   package_dir=carpeta, urgent=True)
            base.state, base.error = "failed", resultado.error or ""
            return base

        self.runs.update(dia, slot.id, state="published", publish_id=resultado.publish_id,
                         plan=plan)
        notify(f"Tu Canal {self._hora(slot)}: vídeo en la inbox de TikTok",
               f"{paquete['topic'][:90]}\nFormato: {paquete['format']} "
               f"({paquete['probe'].get('duration_s')}s)\n\n"
               "En la app: abre la notificación de la inbox, pega la caption de abajo y "
               "ACTIVA la etiqueta de contenido generado por IA.\n\n" + paquete["caption"],
               ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
               package_dir=carpeta)
        base.state, base.publish_id = "published", resultado.publish_id or ""
        self.log(f"publicado en la inbox: {resultado.publish_id}")
        return base

    def publish_video(self, video: Path, paquete: dict, slot: Slot):
        from agent.adapters.tiktok_oauth import refresh_and_store
        from agent.adapters.tiktok_publisher import TikTokPublisher
        from agent.models import PublishResult
        from agent.ports.publisher import PublisherError

        if self.cfg.tiktok_refresh_token and self.cfg.tiktok_client_key:
            try:
                refresh_and_store(self.cfg)
                self.log("  token de TikTok renovado")
            except PublisherError as exc:
                self.log(f"  renovación del token falló ({exc}); probando con el actual")
        if not self.cfg.tiktok_access_token:
            return PublishResult(state=PublishState.failed, video_path=str(video),
                                 error="sin AGENT_TIKTOK_ACCESS_TOKEN en el .env")
        publicador = TikTokPublisher(self.cfg)
        try:
            resultado = publicador.upload(str(video), access_token=self.cfg.tiktok_access_token)
        except PublisherError as exc:
            resultado = PublishResult(state=PublishState.failed, video_path=str(video),
                                      error=str(exc))
        self.store.record_post(
            resultado.publish_id or "", str(video), status=resultado.state.value,
            error=resultado.error, format=paquete["format"],
            script_id=paquete.get("script_id"), slot=f"{slot.id}")
        if resultado.state is PublishState.uploaded and resultado.publish_id:
            self._sleep(20)
            try:
                estado = publicador.fetch_status(resultado.publish_id,
                                                 access_token=self.cfg.tiktok_access_token)
                self.store.update_post_status(resultado.publish_id, status=estado)
                self.log(f"  estado en la API: {estado}")
            except PublisherError as exc:
                self.log(f"  estado no disponible ahora: {exc}")
        return resultado

    def publish_carousel(self, paquete: dict, slot: Slot, day: date):
        """Diapositivas en JPEG en GitHub Pages -> PULL_FROM_URL -> inbox de TikTok."""
        from agent.adapters.tiktok_oauth import refresh_and_store
        from agent.adapters.tiktok_publisher import TikTokPublisher
        from agent.models import PublishResult
        from agent.ports.publisher import PublisherError
        from agent.publish.media_host import GitPagesHost, MediaHostError, to_jpeg

        carpeta: Path = paquete["dir"]
        try:
            jpgs = to_jpeg([Path(p) for p in paquete["slides"]], carpeta / "_jpg")
            host = GitPagesHost(self.cfg.media_repo_dir, self.cfg.media_base_url)
            urls = host.publish(jpgs, f"tucanal/{day.isoformat()}/{slot.id}")
        except (MediaHostError, OSError) as exc:
            return PublishResult(state=PublishState.failed, error=f"alojamiento: {exc}")
        if self.cfg.tiktok_refresh_token and self.cfg.tiktok_client_key:
            try:
                refresh_and_store(self.cfg)
            except PublisherError as exc:
                self.log(f"  renovación del token falló ({exc}); probando con el actual")
        titulo = paquete["caption"].splitlines()[0] if paquete["caption"] else paquete["topic"]
        resultado = TikTokPublisher(self.cfg).upload_photos(
            urls, access_token=self.cfg.tiktok_access_token, title=titulo,
            description=paquete["caption"])
        self.store.record_post(
            resultado.publish_id or "", ",".join(urls), status=resultado.state.value,
            error=resultado.error, format="carousel", carousel_id=paquete.get("carousel_id"),
            slot=slot.id)
        return resultado

    def _avisar_manual(self, paquete: dict, slot: Slot) -> None:
        carpeta: Path = paquete["dir"]
        notify(f"Tu Canal {self._hora(slot)}: carrusel listo para publicar",
               f"{paquete['topic'][:90]}\n\nDiapositivas en {carpeta}/slides (slide-1 a "
               "slide-5, en ese orden). En la app: modo foto, las 5 diapositivas, pega la "
               "caption de abajo y ACTIVA la etiqueta de contenido generado por IA.\n\n"
               + paquete["caption"],
               ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
               package_dir=carpeta)
        self.log("carrusel listo para publicar a mano (aviso enviado)")

    def _limpiar_cache(self) -> None:
        limite = time.time() - CACHE_DIAS * 86400
        borrados = 0
        for carpeta in ("footage-cache", "photos-cache"):
            for archivo in (self.cfg.data_dir / carpeta).glob("*"):
                try:
                    if archivo.is_file() and archivo.stat().st_mtime < limite:
                        archivo.unlink()
                        borrados += 1
                except OSError:
                    continue
        if borrados:
            self.log(f"cache: {borrados} archivo(s) con más de {CACHE_DIAS} días borrados")

    def _traza_llm(self) -> list[dict]:
        inicio = self._now().astimezone(UTC).replace(hour=0, minute=0, second=0,
                                                     microsecond=0)
        return [{k: c[k] for k in ("stage", "route", "ok", "error", "input_tokens",
                                   "output_tokens")}
                for c in self.ledger.calls_since(inicio)][-60:]


# ==================================================================== apoyo

def _caption_video(script: Script) -> str:
    """Caption del post: gancho, fuentes (dominio, sin link) y los 5 hashtags."""
    from agent.brand.brand import load as load_brand

    marca = load_brand()
    dominios: list[str] = []
    for f in script.facts:
        d = str(f.source_url).split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")
        if d not in dominios:
            dominios.append(d)
    # Guía de la marca: gancho escrito sin repetir el audio + contexto. Guion
    # antiguo (sin caption) cae en el hook.
    partes = [script.caption.strip() or script.hook.strip()]
    if dominios:
        partes.append("Fuentes: " + ", ".join(dominios[:3]))
    partes.append(" ".join(marca.hashtags))
    return "\n\n".join(partes) + "\n"


def _motivo_video(rep) -> str:
    if rep.failure:
        return rep.failure
    if rep.review is not None and not rep.review.approved:
        notas = rep.review.revision_notes[:2]
        return f"{rep.review.total}/14: " + " | ".join(notas)
    for r in reversed(rep.rounds):
        if r.write is not None and r.write.refusal:
            return r.write.refusal
        if r.write is not None and r.write.violations:
            return "puertas mecánicas: " + "; ".join(r.write.violations[:2])
    return ""


def _motivo_carrusel(rep) -> str:
    if rep.failure:
        return rep.failure
    if rep.review is not None and not rep.review.approved:
        return (f"{rep.review.total}/{CAROUSEL_MAX}: "
                + " | ".join(rep.review.revision_notes[:2]))
    for r in reversed(rep.rounds):
        if r.write is not None and r.write.refusal:
            return r.write.refusal
        if r.write is not None and r.write.attempts and r.write.attempts[-1].violations:
            return "puertas mecánicas: " + "; ".join(r.write.attempts[-1].violations[:2])
    return ""


__all__ = ["CrossFamilyJudge", "SlotFailed", "SlotResult", "SlotRunner"]
