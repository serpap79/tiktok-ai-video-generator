"""Um slot do dia, de ponta a ponta: pauta -> pesquisa -> formato -> roteiro ->
juiz -> render -> pacote -> publicacao -> aviso.

Cada etapa ja existia como comando manual; o que este modulo acrescenta e o
que um humano fazia entre um comando e outro:

- **decidir** tema, tipo de conteudo e formato para ESTE horario, sabendo o
  que o dia ja publicou (`editorial/`), com o motivo gravado;
- **cair para o proximo** quando algo nao sustenta: tema sem dossie vira o
  segundo da lista; roteiro que nao passa no formato preferido tenta o
  segundo formato; modelo sem cota vira o proximo da rota (`router.py`);
- **medir o artefato** antes de publicar (dimensao, audio, duracao, titulo
  legivel no slide) -- a regra do projeto desde o MP4 mudo;
- **esperar a hora** e publicar, renovando o token do TikTok antes.

Toda decisao vai para `slot_runs.plan_json` e para `meta.json` no pacote:
o post das 20h tem resposta para "por que este tema, por que longo, qual
modelo escreveu e o que custou".
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
    FORMATO_FORCADO,
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

# Quantos temas tentar antes de desistir do slot, e quantos formatos por tema.
MAX_TEMAS = 4
MAX_FORMATOS_POR_TEMA = 2
# Fatos que a pesquisa busca antes de parar de ler fontes.
ALVO_FATOS = 6
# Espera maxima pela hora do post: mais que isso e relogio/timer errado, e
# publicar ja e melhor que dormir a tarde inteira.
ESPERA_MAXIMA_S = 3 * 3600
# Atraso maximo para ainda produzir o slot. Com `Persistent=true`, ligar a
# maquina as 14h dispararia o slot das 9h: dois posts em meia hora competem
# entre si pelo mesmo publico. Passou disso, o slot e pulado com motivo.
ATRASO_MAXIMO_S = 2 * 3600
# Faixas de aceite do MP4 medido. O longo abaixo de 60s perde a elegibilidade
# ao Rewards, mas ainda e conteudo: entre 55 e 60 publica com aviso.
FAIXA_DURO = {"long": (55.0, 100.0), "short": (8.0, 35.0)}
FAIXA_IDEAL = {"long": (60.0, 90.0), "short": (10.0, 30.0)}

# Tipos de conteudo narrados pela voz masculina, no elenco do guia: Theo em
# tutorial e VS; o resto na voz padrao do canal (feminina, como a Iris).
PILARES_THEO = frozenset({"tutorial", "vs"})
# Cache de clipes e fotos: ~150 MB por video longo. Tres slots por dia enchem
# o disco em semanas; o que nao e usado ha 10 dias sai (o livro de reuso so
# olha 14 dias, entao o clipe apagado tambem ja nao seria escolhido).
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
    """O slot nao tem o que publicar; o motivo vai para o livro e para o aviso."""


class CrossFamilyJudge:
    """Porta LLM do juiz que prefere a familia diferente da do roteirista.

    O provedor do roteirista so e conhecido depois que ele escreveu (a rota
    pode ter caido para o Groq), entao a ordem do juiz e decidida na hora de
    cada chamada.
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
        alvo = self._judge
        preferir = getattr(self._judge, "prefer_other_than", None)
        escritor = getattr(self._writer, "provider", "")
        if callable(preferir) and escritor:
            alvo = preferir(escritor)
        self._ultimo = alvo
        return alvo.complete(prompt, **kwargs)


class SlotRunner:
    # Ordem de formatos por fora da grade (ex. `slot-extra --format video`).
    # Mesma semantica de FORMATO_FORCADO: tenta na ordem, ultimo e emergencia.
    FORMATOS_EXTRA: dict[str, tuple[str, ...]] = {
        "video": ("long", "short"),
        "carrossel": ("carousel", "short"),
        # Um formato so, sem reserva: serve para conferir uma mudanca de render
        # numa peca curta em vez de esperar um longo de 90s a cada iteracao.
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
        linha = f"[{self._now().astimezone(TZ):%H:%M:%S}] {msg}"
        self.trail.append(linha)
        self._log_fn(linha)

    # ================================================================ fluxo

    @staticmethod
    def _hora(slot: Slot) -> str:
        """Rotulo de horario para avisos: '09h' no grade, o id fora dela."""
        return f"{slot.id[:2]}h" if slot.id[:2].isdigit() else slot.id

    def run(self, slot: Slot, day: date, *, publish: bool = True, wait: bool = True,
            force: bool = False) -> SlotResult:
        dia = day.isoformat()
        atraso = (self._now() - slot.when(day)).total_seconds()
        if publish and wait and not force and atraso > ATRASO_MAXIMO_S:
            motivo = (f"slot {slot.id} disparou {atraso / 3600:.1f}h depois do horario "
                      "(maquina desligada?); pulado para nao colar em outro post")
            atual = self.runs.get(dia, slot.id)
            if atual is None or atual["state"] not in ("published", "ready_manual"):
                self.runs.begin(dia, slot.id, force=True)
                self.runs.update(dia, slot.id, state="skipped", error=motivo)
            self.log(motivo)
            return SlotResult("skipped", slot.id, dia, error=motivo)
        if not self.runs.begin(dia, slot.id, force=force):
            feito = self.runs.get(dia, slot.id) or {}
            self.log(f"slot {slot.id} de {dia} ja concluido ({feito.get('state')}); nada a fazer")
            return SlotResult("skipped", slot.id, dia, topic=feito.get("topic") or "")
        self.log(f"slot {slot.id} ({slot.label}) de {dia}: {slot.intent}")
        self._limpar_cache()
        plano: dict = {"slot": slot.id, "day": dia}
        try:
            resultado = self._produzir_e_publicar(slot, day, plano, publish=publish, wait=wait)
        except Exception as exc:  # noqa: BLE001 -- o slot nunca morre sem registro
            motivo = f"{type(exc).__name__}: {exc}"
            self.log(f"FALHA: {motivo}")
            plano["error_trace"] = traceback.format_exc()[-2000:]
            plano["log"] = self.trail
            self.runs.update(dia, slot.id, state="failed", error=motivo[:500], plan=plano)
            notify(f"Seu Canal: slot {self._hora(slot)} falhou",
                   f"{motivo}\n\nNada foi publicado neste horario. Detalhes: "
                   f"uv run agent autopilot-status",
                   ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
                   urgent=True)
            return SlotResult("failed", slot.id, dia, error=motivo)
        return resultado

    def _produzir_e_publicar(self, slot: Slot, day: date, plano: dict, *,
                             publish: bool, wait: bool) -> SlotResult:
        dia = day.isoformat()
        usados = self.runs.used_today(dia, except_slot=slot.id)
        plano["used_today"] = usados

        candidatos = self.candidates(usados)
        escolhas = rank_topics(candidatos, slot.id, usados["pillars"], limit=10)
        escolhas, nota_interesse = self.rerank(escolhas)
        self.log(f"pauta: {nota_interesse}")
        plano["interest"] = nota_interesse
        plano["shortlist"] = [
            {"term": c.decision.term, "source": c.decision.source, "pillar": c.pillar,
             "score": c.score, "reason": c.reason} for c in escolhas[:8]]
        if not escolhas:
            raise SlotFailed("nenhum tema elegivel no radar (politica, nicho e repeticao)")

        pacote = None
        tentativas: list[dict] = []
        for escolha in escolhas[:MAX_TEMAS]:
            self.log(f"tema: {escolha.decision.term[:80]} [{escolha.pillar}] "
                     f"({escolha.decision.source}, nota {escolha.score:.2f})")
            try:
                pacote = self._tentar_tema(escolha, slot, day, usados, plano, tentativas)
            except SlotFailed as exc:
                tentativas.append({"term": escolha.decision.term, "falha": str(exc)})
                self.log(f"  descartado: {exc}")
                continue
            if pacote is not None:
                break
        plano["attempts"] = tentativas
        if pacote is None:
            falhas = [t["falha"][:80] for t in tentativas if t.get("falha")]
            raise SlotFailed(f"{len(falhas)} tema(s) tentados, nenhum virou peca "
                             "aprovada: " + "; ".join(falhas))

        # ------------------------------------------------ hora do post
        alvo = slot.when(day)
        agora = self._now()
        if wait and publish and agora < alvo:
            espera = (alvo - agora).total_seconds()
            if espera <= ESPERA_MAXIMA_S:
                self.log(f"pacote pronto; esperando {espera / 60:.0f} min ate {alvo:%H:%M}")
                self.runs.update(dia, slot.id, state="produced")
                self._sleep(espera)
        return self._publicar(pacote, slot, day, plano, publish=publish)

    # ================================================================ etapas

    def candidates(self, usados: dict[str, list[str]]) -> list[Decision]:
        """Radar + curador, com o que o dia ja cobriu no ledger de repeticao."""
        from agent.curator.curator import Curator
        from agent.radar.collector import Radar, default_sources

        coleta = Radar(default_sources(), self.store).collect()
        self._sinais = coleta.signals
        for nome, erro in coleta.failures.items():
            self.log(f"  [fonte fora] {nome}: {erro[:100]}")
        if not coleta.signals:
            raise SlotFailed("radar sem nenhum sinal (todas as fontes fora)")
        ledger = self.store.recent_topics(days=30) + usados["topics"]
        report = Curator().curate(coleta.signals, ledger=ledger)
        # O "selected" do curador ainda nao e escolha do slot: gravar assim
        # tiraria o tema do ledger das proximas horas sem ele virar video.
        self.store.record_decisions([
            d.model_copy(update={"verdict": Verdict.not_selected})
            if d.verdict is Verdict.selected else d for d in report.decisions])
        self.log(f"radar: {len(coleta.signals)} sinais, {len(report.eligible)} elegiveis "
                 f"({', '.join(coleta.sources_ok)})")
        return report.top(12)

    def rerank(self, escolhas: list[TopicChoice]) -> tuple[list[TopicChoice], str]:
        """Nota de interesse do publico (1 chamada) combinada com radar + horario."""
        from agent.editorial.interest import rerank

        return rerank(escolhas, self.llm("ranker"))

    def llm(self, stage: str) -> LLM:
        """A porta LLM do estagio: rota com troca de modelo por cota."""
        from agent.adapters.llm_factory import build_routed

        return build_routed(stage, self.cfg, self.ledger)

    def _tentar_tema(self, escolha: TopicChoice, slot: Slot, day: date,
                     usados: dict[str, list[str]], plano: dict,
                     tentativas: list[dict]) -> dict | None:
        decisao = escolha.decision
        from agent.research.related import related_items

        extras = related_items(decisao, getattr(self, "_sinais", []))
        if extras:
            decisao = decisao.model_copy(update={"news_items": [*decisao.news_items, *extras]})
            self.log(f"  +{len(extras)} fonte(s) da mesma historia na coleta: "
                     + ", ".join(n.source_name for n in extras))
        pesquisa = self.research(decisao, self.llm("research"))
        if pesquisa.dossier is None:
            raise SlotFailed(f"pesquisa sem dossie ({len(pesquisa.pages)} paginas lidas, "
                             f"falhas: {list(pesquisa.failures)[:3]})")
        dossier = pesquisa.dossier
        rota = getattr(pesquisa, "route", "") or pesquisa.model
        dossier_id = self.store.record_dossier(
            dossier, model=rota, provider=rota.split(":", 1)[0] if ":" in rota else "",
            usage=(pesquisa.usage.input_tokens, pesquisa.usage.output_tokens),
            latency_s=pesquisa.latency_s, source_count=pesquisa.source_count,
            discarded=[vars(d) for d in pesquisa.discarded], failures=pesquisa.failures)
        self.log(f"  dossie #{dossier_id}: {len(dossier.facts)} fatos de "
                 f"{pesquisa.source_count} fonte(s), {len(pesquisa.discarded)} descartes")

        feat = features(dossier)
        desempenho = performance_from_metrics(self.store.format_performance_rows())
        forcados = self._formatos.get(slot.id, FORMATO_FORCADO.get(slot.id))
        # `em_ordem` so quando ha ordem forcada: ali a grade manda, e a nota
        # apenas explica. Sem grade forcada a nota decide, como sempre.
        formato = choose_format(feat, escolha.pillar, slot.id, usados["formats"], desempenho,
                                allowed=forcados or FORMATS, em_ordem=bool(forcados))
        motivo_ordem = (f"ordem forcada {list(forcados)}" if forcados else "ordem por nota")
        self.log(f"  formato: {formato.reason} [{motivo_ordem}]")

        registro = {"term": decisao.term, "pillar": escolha.pillar,
                    "format_decision": formato.reason, "scores": formato.scores,
                    "dossier_id": dossier_id, "facts": len(dossier.facts)}
        ordem = (list(forcados) if forcados
                 else [f for f in formato.ranked() if f in FORMATS])
        # Com ordem forcada, tenta na ordem dada pulando o que o dossie nao
        # sustenta (fora do ranked), mas sempre tenta o ultimo (emergencia).
        candidatos = ([f for f in ordem if f in formato.ranked() or f == ordem[-1]]
                      if forcados else ordem)
        for fmt in candidatos[:MAX_FORMATOS_POR_TEMA]:
            # O escritor depende do formato: curto abre no Groq (teto de
            # palavras), o resto no premium do OpenRouter. O juiz cruza a
            # familia com o escritor daquele formato, nao com o anterior.
            etapa = "writer_short" if fmt == "short" else "writer"
            writer = self.llm(etapa)
            judge = CrossFamilyJudge(self.llm("judge"), writer)
            peca = self.write(dossier, fmt, escolha.pillar, writer, judge)
            registro.setdefault("writes", []).append(peca["summary"])
            if peca["approved"]:
                plano.update({
                    "topic": decisao.term, "source": decisao.source,
                    "topic_reason": escolha.reason, "pillar": escolha.pillar,
                    "pillar_guesses": [vars(g) for g in escolha.guesses[:3]],
                    "format": fmt, "format_reason": formato.reason,
                    "format_scores": formato.scores, "dossier_id": dossier_id,
                    "research": {"sources": [p.url for p in pesquisa.pages],
                                 "discarded": len(pesquisa.discarded),
                                 "failures": pesquisa.failures,
                                 "tokens": pesquisa.usage.total_tokens},
                })
                tentativas.append(registro)
                return self._montar_pacote(peca, decisao, escolha.pillar, fmt,
                                           dossier, slot, day, plano)
            self.log(f"  {fmt} reprovado: {peca['summary'].get('motivo', '')[:120]}")
        tentativas.append(registro)
        raise SlotFailed("nenhum formato aprovado pelo juiz para este tema")

    def research(self, decisao: Decision, llm: LLM):
        from agent.research.fetch import PageFetcher
        from agent.research.researcher import Researcher

        pesquisador = Researcher(
            llm, fetcher=PageFetcher(max_chars=self.cfg.research_page_chars),
            max_sources=self.cfg.research_max_sources,
            max_facts_per_source=self.cfg.research_max_facts_per_source)
        relatorio = pesquisador.research(decisao, target_facts=ALVO_FATOS)
        # Qual modelo da rota respondeu por ultimo (o relatorio nasce antes).
        relatorio.route = getattr(llm, "last_route", "")
        return relatorio

    def write(self, dossier: Dossier, fmt: str, pillar: str, writer: LLM,
              judge: LLM) -> dict:
        """Escreve e julga no formato; grava roteiro/parecer mesmo reprovados."""
        from agent.judge.judge import Judge
        from agent.pipeline import produce, produce_carousel
        from agent.writer.writer import Screenwriter

        dossier_id = self.store.latest_dossier_id(dossier.topic)
        if fmt == "carousel":
            rep = produce_carousel(dossier, writer, judge_llm=judge, pillar=pillar)
            carrossel = rep.carousel
            carousel_id = None
            if carrossel is not None:
                carousel_id = self.store.record_carousel(
                    carrossel, model=getattr(writer, "model", ""),
                    provider=getattr(writer, "provider", ""),
                    usage=(rep.usage.input_tokens, rep.usage.output_tokens),
                    latency_s=rep.latency_s,
                    attempts=[a.violations for r in rep.rounds if r.write
                              for a in r.write.attempts],
                    review=rep.review)
            resumo = {"format": fmt, "approved": rep.approved, "rounds": len(rep.rounds),
                      "tokens": rep.usage.total_tokens, "failure": rep.failure,
                      "writer": f"{getattr(writer, 'provider', '')}/{getattr(writer, 'model', '')}",
                      "judge": f"{judge.provider}/{judge.model}",
                      "motivo": _motivo_carrossel(rep),
                      # Toda tentativa reprovada fica gravada (regra do projeto):
                      # carrossel que nao passou nos portoes nao vira linha em
                      # `carousels`, entao as violacoes vao para o plano do slot.
                      "attempts": [a.violations for r in rep.rounds if r.write
                                   for a in r.write.attempts]}
            return {"approved": rep.approved and carrossel is not None, "carousel": carrossel,
                    "carousel_id": carousel_id, "summary": resumo}

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
        resumo = {"format": fmt, "approved": rep.approved, "rounds": len(rep.rounds),
                  "tokens": rep.usage.total_tokens, "failure": rep.failure,
                  "writer": f"{getattr(writer, 'provider', '')}/{getattr(writer, 'model', '')}",
                  "judge": f"{judge.provider}/{judge.model}",
                  "total": rep.review.total if rep.review else None,
                  "motivo": _motivo_video(rep)}
        return {"approved": rep.approved and rep.script is not None, "script": rep.script,
                "script_id": script_id, "summary": resumo}

    # ================================================================ pacote

    def _montar_pacote(self, peca: dict, decisao: Decision, pillar: str, fmt: str,
                       dossier: Dossier, slot: Slot, day: date, plano: dict) -> dict:
        dia = day.isoformat()
        pasta = (self.cfg.output_dir / dia
                 / f"{slot.id}-{slugify(decisao.term)}-{fmt}")
        pasta.mkdir(parents=True, exist_ok=True)
        self.runs.update(dia, slot.id, topic=decisao.term, source=decisao.source,
                         pillar=pillar, format=fmt, package_dir=str(pasta))
        # Agora sim o tema foi escolhido: entra no ledger de repeticao.
        self.store.record_decisions([decisao.model_copy(update={
            "verdict": Verdict.selected,
            "reason": f"piloto {dia} {slot.id}: {plano.get('topic_reason', '')}"[:500],
            "decided_at": datetime.now(UTC)})])

        if fmt == "carousel":
            carrossel: Carousel = peca["carousel"]
            (pasta / "carrossel.json").write_text(
                carrossel.model_dump_json(indent=2) + "\n", encoding="utf-8")
            slides = self.render_carousel(carrossel, pasta / "slides", pillar)
            self._aceite_slides(slides)
            legenda = (pasta / "slides" / "caption.txt").read_text(encoding="utf-8")
            (pasta / "caption.txt").write_text(legenda, encoding="utf-8")
            artefato = {"kind": "carousel", "slides": [str(s) for s in slides],
                        "carousel_id": peca.get("carousel_id")}
        else:
            script: Script = peca["script"]
            (pasta / "roteiro.json").write_text(
                script.model_dump_json(indent=2) + "\n", encoding="utf-8")
            video, medida = self.render_video(script, pasta, pillar)
            aviso = self._aceite_video(medida, fmt)
            if aviso:
                plano["render_warning"] = aviso
                self.log(f"  aviso: {aviso}")
            legenda = _legenda_video(script)
            (pasta / "caption.txt").write_text(legenda, encoding="utf-8")
            artefato = {"kind": "video", "video": str(video), "probe": medida,
                        "script_id": peca.get("script_id")}
        plano["artifact"] = artefato
        plano["writes"] = peca["summary"]
        plano["llm_trail"] = self._trilha_llm()
        plano["log"] = self.trail
        (pasta / "meta.json").write_text(
            json.dumps(plano, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")
        self.runs.update(dia, slot.id, state="produced", plan=plano)
        self.log(f"  pacote pronto em {pasta}")
        return {"dir": pasta, "format": fmt, "pillar": pillar, "topic": decisao.term,
                "caption": legenda, **artefato}

    def render_video(self, script: Script, pasta: Path, pillar: str) -> tuple[Path, dict]:
        """Clipes escolhidos -> narracao + legenda (ffmpeg) -> trilha + marca -> MP4 medido.

        O renderizador proprio e o padrao; o MPT entra se ele falhar ou se a
        selecao de clipes nao entregar material.
        """
        from agent.adapters.mpt_renderer import probe_video
        from agent.brand.brand import load as load_brand
        from agent.render.footage import FootageLedger, PexelsFootage
        from agent.render.music import render_bed
        from agent.render.post import PostSpec, postprocess
        from agent.render.presenter import conferir_apresentador

        trabalho = pasta / "_work"
        materiais: list[Path] = []
        try:
            footage = PexelsFootage(self.cfg.pexels_api_key,
                                    self.cfg.data_dir / "footage-cache",
                                    ledger=FootageLedger(self.cfg.db_path))
            plano = footage.plan(script)
            materiais = plano.paths
            self.log(f"  clipes: {len(materiais)} escolhidos "
                     f"({', '.join(sorted({c.term for c in plano.clips}))[:120]})"
                     + (f"; faltas: {plano.misses[:3]}" if plano.misses else ""))
        except (ValueError, OSError) as exc:
            self.log(f"  selecao de clipes indisponivel ({exc})")

        marca = load_brand()
        # `pillar` continua sendo o id (string). O objeto vai em outro nome de
        # proposito: `presenter_for` e `accent_for` recebem o id, e passar o
        # objeto fazia as duas caírem no padrao em silencio -- nenhum
        # apresentador aparecia e todo video saia no verde, mesmo nos pilares
        # de acento ciano.
        pilar_obj = marca.pillars.get(pillar)
        pedido = self._pedido_apresentador(marca, pillar)

        # A voz segue quem aparece. Com o THEO apresentando todos os pilares
        # (20/09/2026), rosto masculino com narracao feminina seria uma
        # incoerencia maior que qualquer defeito de boca -- entao o pilar so
        # decide a voz quando nao ha apresentador no quadro.
        voz = (self.cfg.narrator_voice_male
               if (pedido is not None or pillar in PILARES_THEO)
               else self.cfg.narrator_voice)
        resultado, camada = (None, None)
        if self.cfg.renderer == "ffmpeg" and materiais:
            resultado, camada = self._render_ffmpeg(script, materiais, voz, trabalho,
                                                    pedido)
        if resultado is None:
            resultado = self._render_mpt(script, materiais)
        bruto = Path(resultado.video_path)
        duracao = float(resultado.duration_s or script.estimated_duration_s)
        trilha = render_bed(trabalho / "trilha.wav", duracao, pillar, seed=script.topic)
        spec = PostSpec(hook=script.hook, tag=pilar_obj.tag if pilar_obj else "",
                        accent=marca.accent_for(pillar), handle=marca.handle,
                        presenter=camada)
        final = postprocess(bruto, pasta / "video.mp4", spec, duration_s=duracao,
                            music_wav=trilha, work_dir=trabalho)
        medida = probe_video(final)
        if camada is not None:
            aviso_apr = conferir_apresentador(final, camada, trabalho)
            self.log(f"  {aviso_apr}")
        bruto.unlink(missing_ok=True)
        shutil.rmtree(trabalho, ignore_errors=True)
        self.log(f"  video: {medida['width']}x{medida['height']}, {medida['duration_s']}s, "
                 f"audio={'sim' if medida['has_audio'] else 'NAO'}")
        return final, medida

    def _pedido_apresentador(self, marca, pillar: str):
        """Quem apresenta este pilar, e com que material.

        Os dois materiais sao assados fora do laco, porque os dois dependem de
        rembg e mediapipe (>400 MB que o agente nunca precisa carregar tres
        vezes por dia):

        - `<id>_base.mp4` + `.json` (`scripts/make_presenter_video.py`) -- o
          clipe fotorrealista. Preferido: a piscada e o balanco de cabeca sao
          humanos, gerados como video, e so a boca e nossa.
        - `<id>.png` + `.json` (`scripts/make_presenter_cutouts.py`) -- o
          retrato parado, de que tudo e sintetizado. Reserva.

        Sem nenhum dos dois o video sai sem apresentador e o motivo vai para o
        log: avatar e acabamento, e slot sem ele ainda e slot publicavel.
        """
        from agent.adapters.ffmpeg_renderer import PresenterRequest

        apresentador = marca.presenter_for(pillar)
        if apresentador is None:
            return None
        pasta = PROJECT_ROOT / "brand" / "assets" / "presenters"
        clipe = pasta / f"{apresentador.id}_base.json"
        png = pasta / f"{apresentador.id}.png"
        tem_clipe = clipe.exists() and clipe.with_suffix(".mp4").exists()
        tem_png = png.exists() and png.with_suffix(".json").exists()
        if not tem_clipe and not tem_png:
            self.log(f"  apresentador: {apresentador.name} sem material "
                     f"(nem {clipe.name} nem {png.name} -- rode "
                     f"scripts/make_presenter_video.py)")
            return None
        self.log(f"  apresentador: {apresentador.name} "
                 + ("(clipe base, boca sintetizada)" if tem_clipe
                    else "(retrato parado sintetizado -- sem clipe base)"))
        return PresenterRequest(cutout=png if tem_png else None,
                                name=apresentador.name,
                                base=clipe if tem_clipe else None)

    def _render_ffmpeg(self, script: Script, materiais: list[Path], voz: str,
                       trabalho: Path, pedido=None):
        from agent.adapters.ffmpeg_renderer import FfmpegRenderer
        from agent.models import RenderState
        from agent.ports.renderer import RendererError

        renderer = FfmpegRenderer(self.cfg)
        self.log(f"  renderizando (ffmpeg, voz {voz})...")
        try:
            resultado = renderer.render(script, materials=materiais, voice=voz,
                                        work_dir=trabalho, presenter=pedido)
        except RendererError as exc:
            self.log(f"  renderizador proprio falhou ({exc}); tentando o MPT")
            return None, None
        if resultado.state is RenderState.failed or not resultado.video_path:
            self.log(f"  renderizador proprio falhou ({resultado.error}); tentando o MPT")
            return None, None
        if renderer.presenter_error:
            self.log(f"  apresentador nao entrou: {renderer.presenter_error}")
        elif renderer.last_presenter is not None:
            c = renderer.last_presenter
            self.log(f"  apresentador: {c.frames} quadros a {c.fps} fps, "
                     f"legenda em y={c.subtitle_y} a partir de {c.subtitle_start:.1f}s")
        if renderer.last_changes:
            trocas = sorted({f"{a}->{b}" for a, b in renderer.last_changes})
            self.log(f"  pronuncia: {', '.join(trocas)[:160]}")
        conferencia = renderer.last_check
        if conferencia is not None:
            if conferencia.error:
                self.log(f"  conferencia de pronuncia: {conferencia.error}")
            elif conferencia.suspicious:
                self.log("  AVISO pronuncia: o Whisper nao reconheceu "
                         f"{conferencia.suspicious} -- candidatos ao lexico")
                self._suspeitos(conferencia.suspicious, script)
            else:
                self.log("  conferencia de pronuncia: todos os nomes reconhecidos")
        return resultado, renderer.last_presenter

    def _render_mpt(self, script: Script, materiais: list[Path]):
        from agent.adapters.mpt_renderer import MptRenderer
        from agent.models import RenderState

        renderer = MptRenderer(self.cfg)
        if not renderer.health():
            raise SlotFailed(f"renderizador proprio falhou e o MPT esta fora em "
                             f"{self.cfg.renderer_url}")
        self.log(f"  renderizando no MPT (voz {self.cfg.voice_name})...")
        resultado = renderer.render(script, materials=materiais or None)
        if resultado.state is RenderState.failed or not resultado.video_path:
            raise SlotFailed(f"render falhou: {resultado.error}")
        return resultado

    def _suspeitos(self, nomes: list[str], script: Script) -> None:
        """Registra nome mal pronunciado para virar entrada do lexico."""
        arquivo = self.cfg.data_dir / "pronuncia_suspeita.txt"
        with arquivo.open("a", encoding="utf-8") as fh:
            for n in nomes:
                fh.write(f"{datetime.now(UTC):%Y-%m-%d %H:%M}\t{n}\t{script.topic[:60]}\n")

    def render_carousel(self, carrossel: Carousel, pasta: Path, pillar: str) -> list[Path]:
        from agent.render.carousel import render_carousel
        from agent.render.photos import fetch

        cache = self.cfg.data_dir / "photos-cache"
        usadas: set[int] = set()
        fotos: dict[int, Path | None] = {}
        broll = list(carrossel.broll)
        for s in carrossel.slides:
            if s.n == 1 and broll:
                fotos[s.n] = fetch(broll[0], cache, subject=True, used=usadas)
            elif s.n == 3 and len(broll) > 1:
                fotos[s.n] = fetch(broll[1], cache, subject=True, used=usadas)
            else:
                fotos[s.n] = fetch(s.visual, cache, used=usadas)
        self.log(f"  fotos: {sum(1 for f in fotos.values() if f)}/5 slides com foto")
        return render_carousel(carrossel, pasta, pillar, fotos)

    # ================================================================ aceite

    @staticmethod
    def _aceite_video(medida: dict, fmt: str) -> str:
        if (medida.get("width"), medida.get("height")) != (1080, 1920):
            raise SlotFailed(f"video fora de 1080x1920: {medida}")
        if not medida.get("has_audio"):
            raise SlotFailed("video sem trilha de audio (o defeito do MP4 mudo)")
        dur = medida.get("duration_s") or 0.0
        duro = FAIXA_DURO.get(fmt, FAIXA_DURO["long"])
        if not duro[0] <= dur <= duro[1]:
            raise SlotFailed(f"duracao {dur}s fora do aceitavel {duro} para {fmt}")
        ideal = FAIXA_IDEAL.get(fmt, FAIXA_IDEAL["long"])
        if not ideal[0] <= dur <= ideal[1]:
            return f"duracao {dur}s fora da faixa ideal {ideal} do formato {fmt}"
        return ""

    @staticmethod
    def _aceite_slides(slides: list[Path]) -> None:
        from PIL import Image

        from agent.render.carousel import ink_height, legible

        if len(slides) != 5:
            raise SlotFailed(f"carrossel com {len(slides)} slides")
        for s in slides:
            with Image.open(s) as img:
                if img.size != (1080, 1920):
                    raise SlotFailed(f"{s.name} fora de 1080x1920")
            if not legible(s):
                raise SlotFailed(f"{s.name} com titulo ilegivel ({ink_height(s)}px)")

    # ================================================================ publicacao

    def _publicar(self, pacote: dict, slot: Slot, day: date, plano: dict, *,
                  publish: bool) -> SlotResult:
        dia = day.isoformat()
        pasta: Path = pacote["dir"]
        base = SlotResult("produced", slot.id, dia, topic=pacote["topic"],
                          pillar=pacote["pillar"], format=pacote["format"],
                          package_dir=str(pasta))
        if not publish:
            self.log("publicacao desligada (--no-publish): pacote fica pronto")
            self.runs.update(dia, slot.id, state="produced", plan=plano)
            return base

        if pacote["kind"] == "carousel":
            # A API de foto do TikTok so aceita PULL_FROM_URL de prefixo
            # verificado. Com a hospedagem configurada (GitHub Pages), o
            # carrossel vai para a inbox como o video; sem ela, pacote manual.
            if self.cfg.media_repo_dir and self.cfg.media_base_url:
                resultado = self.publish_carousel(pacote, slot, day)
                plano["publish"] = {"state": resultado.state.value,
                                    "publish_id": resultado.publish_id,
                                    "error": resultado.error}
                if resultado.state is PublishState.uploaded:
                    self.runs.update(dia, slot.id, state="published", plan=plano,
                                     publish_id=resultado.publish_id)
                    notify(f"Seu Canal {self._hora(slot)}: carrossel na inbox do TikTok",
                           f"{pacote['topic'][:90]}\n\nNo app: abra a notificacao da inbox, "
                           "confira a legenda e LIGUE o rotulo de conteudo gerado por IA.",
                           ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
                           package_dir=pasta)
                    base.state, base.publish_id = "published", resultado.publish_id or ""
                    return base
                self.log(f"carrossel por API falhou ({resultado.error}); vai como pacote")
            self.runs.update(dia, slot.id, state="ready_manual", plan=plano)
            self._avisar_manual(pacote, slot)
            base.state = "ready_manual"
            return base

        resultado = self.publish_video(Path(pacote["video"]), pacote, slot)
        plano["publish"] = {"state": resultado.state.value, "publish_id": resultado.publish_id,
                            "error": resultado.error}
        if resultado.state is not PublishState.uploaded:
            self.runs.update(dia, slot.id, state="failed", plan=plano,
                             error=f"publicacao: {resultado.error}"[:500])
            notify(f"Seu Canal: video {self._hora(slot)} NAO subiu",
                   f"{resultado.error}\n\nO video esta pronto em {pasta}/video.mp4 -- "
                   "suba a mao pelo app se quiser manter o horario.",
                   ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
                   package_dir=pasta, urgent=True)
            base.state, base.error = "failed", resultado.error or ""
            return base

        self.runs.update(dia, slot.id, state="published", publish_id=resultado.publish_id,
                         plan=plano)
        notify(f"Seu Canal {self._hora(slot)}: video na inbox do TikTok",
               f"{pacote['topic'][:90]}\nFormato: {pacote['format']} "
               f"({pacote['probe'].get('duration_s')}s)\n\n"
               "No app: abra a notificacao da inbox, cole a legenda abaixo e LIGUE o "
               "rotulo de conteudo gerado por IA.\n\n" + pacote["caption"],
               ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
               package_dir=pasta)
        base.state, base.publish_id = "published", resultado.publish_id or ""
        self.log(f"publicado na inbox: {resultado.publish_id}")
        return base

    def publish_video(self, video: Path, pacote: dict, slot: Slot):
        from agent.adapters.tiktok_oauth import refresh_and_store
        from agent.adapters.tiktok_publisher import TikTokPublisher
        from agent.models import PublishResult
        from agent.ports.publisher import PublisherError

        if self.cfg.tiktok_refresh_token and self.cfg.tiktok_client_key:
            try:
                refresh_and_store(self.cfg)
                self.log("  token do TikTok renovado")
            except PublisherError as exc:
                self.log(f"  renovacao do token falhou ({exc}); tentando com o atual")
        if not self.cfg.tiktok_access_token:
            return PublishResult(state=PublishState.failed, video_path=str(video),
                                 error="sem AGENT_TIKTOK_ACCESS_TOKEN no .env")
        publicador = TikTokPublisher(self.cfg)
        try:
            resultado = publicador.upload(str(video), access_token=self.cfg.tiktok_access_token)
        except PublisherError as exc:
            resultado = PublishResult(state=PublishState.failed, video_path=str(video),
                                      error=str(exc))
        self.store.record_post(
            resultado.publish_id or "", str(video), status=resultado.state.value,
            error=resultado.error, format=pacote["format"],
            script_id=pacote.get("script_id"), slot=f"{slot.id}")
        if resultado.state is PublishState.uploaded and resultado.publish_id:
            self._sleep(20)
            try:
                estado = publicador.fetch_status(resultado.publish_id,
                                                 access_token=self.cfg.tiktok_access_token)
                self.store.update_post_status(resultado.publish_id, status=estado)
                self.log(f"  status na API: {estado}")
            except PublisherError as exc:
                self.log(f"  status indisponivel agora: {exc}")
        return resultado

    def publish_carousel(self, pacote: dict, slot: Slot, day: date):
        """Slides em JPEG no GitHub Pages -> PULL_FROM_URL -> inbox do TikTok."""
        from agent.adapters.tiktok_oauth import refresh_and_store
        from agent.adapters.tiktok_publisher import TikTokPublisher
        from agent.models import PublishResult
        from agent.ports.publisher import PublisherError
        from agent.publish.media_host import GitPagesHost, MediaHostError, to_jpeg

        pasta: Path = pacote["dir"]
        try:
            jpgs = to_jpeg([Path(p) for p in pacote["slides"]], pasta / "_jpg")
            host = GitPagesHost(self.cfg.media_repo_dir, self.cfg.media_base_url)
            urls = host.publish(jpgs, f"seucanal/{day.isoformat()}/{slot.id}")
        except (MediaHostError, OSError) as exc:
            return PublishResult(state=PublishState.failed, error=f"hospedagem: {exc}")
        if self.cfg.tiktok_refresh_token and self.cfg.tiktok_client_key:
            try:
                refresh_and_store(self.cfg)
            except PublisherError as exc:
                self.log(f"  renovacao do token falhou ({exc}); tentando com o atual")
        titulo = pacote["caption"].splitlines()[0] if pacote["caption"] else pacote["topic"]
        resultado = TikTokPublisher(self.cfg).upload_photos(
            urls, access_token=self.cfg.tiktok_access_token, title=titulo,
            description=pacote["caption"])
        self.store.record_post(
            resultado.publish_id or "", ",".join(urls), status=resultado.state.value,
            error=resultado.error, format="carousel", carousel_id=pacote.get("carousel_id"),
            slot=slot.id)
        return resultado

    def _avisar_manual(self, pacote: dict, slot: Slot) -> None:
        pasta: Path = pacote["dir"]
        notify(f"Seu Canal {self._hora(slot)}: carrossel pronto para postar",
               f"{pacote['topic'][:90]}\n\nSlides em {pasta}/slides (slide-1 a slide-5, "
               "nessa ordem). No app: modo foto, os 5 slides, cole a legenda abaixo e "
               "LIGUE o rotulo de conteudo gerado por IA.\n\n" + pacote["caption"],
               ntfy_topic=self.cfg.ntfy_topic, ntfy_server=self.cfg.ntfy_server,
               package_dir=pasta)
        self.log("carrossel pronto para postar a mao (aviso enviado)")

    def _limpar_cache(self) -> None:
        limite = time.time() - CACHE_DIAS * 86400
        removidos = 0
        for pasta in ("footage-cache", "photos-cache"):
            for arquivo in (self.cfg.data_dir / pasta).glob("*"):
                try:
                    if arquivo.is_file() and arquivo.stat().st_mtime < limite:
                        arquivo.unlink()
                        removidos += 1
                except OSError:
                    continue
        if removidos:
            self.log(f"cache: {removidos} arquivo(s) com mais de {CACHE_DIAS} dias removidos")

    def _trilha_llm(self) -> list[dict]:
        inicio = self._now().astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return [{k: c[k] for k in ("stage", "route", "ok", "error", "input_tokens",
                                   "output_tokens")}
                for c in self.ledger.calls_since(inicio)][-60:]


# ==================================================================== apoio

def _legenda_video(script: Script) -> str:
    """Legenda do post: gancho, fontes (dominio, sem link) e as 5 hashtags."""
    from agent.brand.brand import load as load_brand

    marca = load_brand()
    dominios: list[str] = []
    for f in script.facts:
        d = str(f.source_url).split("://", 1)[-1].split("/", 1)[0].removeprefix("www.")
        if d not in dominios:
            dominios.append(d)
    # Guia da marca: gancho escrito sem repetir o audio + contexto. Roteiro
    # antigo (sem caption) cai no hook.
    partes = [script.caption.strip() or script.hook.strip()]
    if dominios:
        partes.append("Fontes: " + ", ".join(dominios[:3]))
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
            return "portoes mecanicos: " + "; ".join(r.write.violations[:2])
    return ""


def _motivo_carrossel(rep) -> str:
    if rep.failure:
        return rep.failure
    if rep.review is not None and not rep.review.approved:
        return f"{rep.review.total}/{CAROUSEL_MAX}: " + " | ".join(rep.review.revision_notes[:2])
    for r in reversed(rep.rounds):
        if r.write is not None and r.write.refusal:
            return r.write.refusal
        if r.write is not None and r.write.attempts and r.write.attempts[-1].violations:
            return "portoes mecanicos: " + "; ".join(r.write.attempts[-1].violations[:2])
    return ""


__all__ = ["CrossFamilyJudge", "SlotFailed", "SlotResult", "SlotRunner"]
