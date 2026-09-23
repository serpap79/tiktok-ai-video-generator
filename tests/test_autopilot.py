"""Piloto automatico: um slot inteiro com pecas falsas -- sem rede, sem render.

Trava o que importa para o dia rodar sozinho: idempotencia (slot publicado
nao vira dois posts), queda para o proximo tema/formato, espera pela hora,
registro do motivo, e o slot que falha sem derrubar o processo.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from agent.adapters.scripted_llm import ScriptedLLM
from agent.autopilot.runner import SlotRunner
from agent.autopilot.runs import SlotRuns
from agent.config import Settings
from agent.editorial.slots import SLOTS, TZ
from agent.models import Decision, Dossier, Fact, PublishResult, PublishState, Verdict
from agent.research.researcher import ResearchReport
from tests.test_writer import resposta

DIA = date(2026, 9, 20)


def fatos(n: int) -> list[Fact]:
    return [Fact(claim=f"O Bonsai 27B tem o dado numero {i} medido pela fonte",
                 source_url=f"https://fonte{i % 2}.com/a", source_name="fonte",
                 quote=f"the Bonsai 27B has the measured figure number {i}")
            for i in range(n)]


def parecer_ok() -> str:
    from agent.judge.judge import JULGADOS
    return json.dumps({c.value: {"reason": "ok", "score": 2} for c in JULGADOS})


class RunnerFalso(SlotRunner):
    """Radar, pesquisa, LLM, render e publicacao substituidos por fakes."""

    def __init__(self, tmp: Path, *, temas: list[str], n_fatos: int = 6,
                 roteiros: list[str] | None = None, agora: datetime | None = None,
                 publicacao: PublishResult | None = None,
                 formatos: dict[str, tuple[str, ...]] | None = None):
        cfg = Settings(_env_file=None, data_dir=tmp / "data", output_dir=tmp / "out",
                       groq_api_key="", gemini_api_key="")
        self.dormidas: list[float] = []
        super().__init__(cfg, now=lambda: agora or datetime(2026, 9, 20, 19, 30, tzinfo=TZ),
                         sleep=self.dormidas.append, log=lambda m: None,
                         formatos=formatos)
        self._temas = temas
        self._n_fatos = n_fatos
        self._roteiros = list(roteiros or [])
        self._publicacao = publicacao
        self.publicados: list[Path] = []

    def candidates(self, usados):
        return [Decision(term=t, source="hacker_news", verdict=Verdict.not_selected,
                         reason="teste", score=0.9 - i * 0.1, niche_fit=0.8,
                         decided_at=datetime.now(UTC)) for i, t in enumerate(self._temas)
                if t not in usados["topics"]]

    def research(self, decisao, llm):
        rep = ResearchReport(topic=decisao.term)
        if self._n_fatos:
            rep.dossier = Dossier(topic=decisao.term, facts=fatos(self._n_fatos),
                                  collected_at=datetime.now(UTC))
        return rep

    def llm(self, stage):
        if stage in ("writer", "writer_short"):
            return ScriptedLLM(responses=self._roteiros, provider="groq")
        if stage == "ranker":
            return ScriptedLLM(responder=lambda _p: '{"notas": []}', provider="groq")
        return ScriptedLLM(responder=lambda _p: parecer_ok(), provider="gemini")

    def render_video(self, script, pasta, pillar):
        video = pasta / "video.mp4"
        video.write_bytes(b"mp4")
        duracao = 70.0 if script.format == "long" else 18.0
        return video, {"width": 1080, "height": 1920, "duration_s": duracao, "has_audio": True}

    def render_carousel(self, carrossel, pasta, pillar):
        from PIL import Image, ImageDraw

        pasta.mkdir(parents=True, exist_ok=True)
        slides = []
        for i in range(1, 6):
            slide = pasta / f"slide-{i}.png"
            img = Image.new("RGB", (1080, 1920), (20, 20, 30))
            # Faixa clara na zona do titulo (y 960-1440, x 70-1010): o aceite
            # de slides mede tinta clara ali (`render/carousel.py:ink_height`).
            ImageDraw.Draw(img).rectangle([80, 1000, 1000, 1300], fill=(255, 255, 255))
            img.save(slide)
            slides.append(slide)
        (pasta / "caption.txt").write_text("legenda de teste #ia #tech\n", encoding="utf-8")
        return slides

    def publish_video(self, video, pacote, slot):
        self.publicados.append(video)
        return self._publicacao or PublishResult(state=PublishState.uploaded,
                                                 publish_id="v_inbox_teste", video_path=str(video))


@pytest.fixture(autouse=True)
def sem_aviso(monkeypatch):
    monkeypatch.setattr("agent.autopilot.runner.notify", lambda *a, **k: [])


def roteiro_longo() -> str:
    return resposta(total=190)


class TestSlotInteiro:
    def test_produz_espera_a_hora_e_publica(self, tmp_path):
        from tests.test_short import resposta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"],
                        roteiros=[resposta_short()] * 3,
                        agora=datetime(2026, 9, 20, 15, 30, tzinfo=TZ))
        res = r.run(SLOTS["1600"], DIA)
        assert res.state == "published" and res.publish_id == "v_inbox_teste"
        assert res.format == "short"
        # Produziu as 15:30 e dormiu ate as 16:00 para publicar.
        assert r.dormidas and r.dormidas[0] == pytest.approx(30 * 60)
        linha = SlotRuns(r.cfg.db_path).get(DIA.isoformat(), "1600")
        plano = json.loads(linha["plan_json"])
        assert linha["state"] == "published"
        assert "format_reason" in plano and "topic_reason" in plano
        assert Path(linha["package_dir"], "caption.txt").exists()

    def test_slot_publicado_nao_vira_dois_posts(self, tmp_path):
        from tests.test_short import resposta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"],
                        roteiros=[resposta_short()] * 6)
        r.run(SLOTS["1600"], DIA, wait=False)
        de_novo = r.run(SLOTS["1600"], DIA, wait=False)
        assert de_novo.state == "skipped" and len(r.publicados) == 1

    def test_tema_sem_dossie_cai_para_o_proximo(self, tmp_path):
        from tests.test_short import resposta_short

        class Seletivo(RunnerFalso):
            def research(self, decisao, llm):
                if decisao.term.startswith("Sem"):
                    return ResearchReport(topic=decisao.term)
                return super().research(decisao, llm)

        r = Seletivo(tmp_path, temas=["Sem fonte nenhuma", "Bonsai 27B roda no bolso"],
                     roteiros=[resposta_short()] * 3)
        res = r.run(SLOTS["1600"], DIA, wait=False)
        assert res.state == "published" and res.topic.startswith("Bonsai")

    def test_dossie_fino_vira_curto_e_nao_longo(self, tmp_path):
        from tests.test_short import resposta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"], n_fatos=2,
                        roteiros=[resposta_short()] * 3)
        res = r.run(SLOTS["1900"], DIA, wait=False)
        assert res.format == "short"

    def test_a_noite_sai_video_longo_e_nao_carrossel(self, tmp_path):
        """A grade de 20/09/2026 tirou o carrossel dos horarios fixos.

        Ate a tarde daquele dia as 20h eram carrossel (pacote manual). Com
        quatro posts por dia e todos video, a noite virou o longo -- que e o
        formato que monetiza. O carrossel continua implementado e sai por
        `slot-extra --format carrossel`.
        """
        from tests.test_writer import resposta as resposta_long

        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"], n_fatos=4,
                        roteiros=[resposta_long()] * 3)
        res = r.run(SLOTS["1900"], DIA, wait=False)
        assert res.format == "long"
        assert res.state != "ready_manual"

    def test_carrossel_continua_alcancavel_fora_da_grade(self, tmp_path):
        """Tirar da grade nao e remover: o formato tem de seguir chamavel."""
        from datetime import time as _time

        from agent.editorial.slots import Slot
        from tests.test_carousel import parecer as parecer_carrossel
        from tests.test_carousel import slides

        class Manual(RunnerFalso):
            def llm(self, stage):
                if stage == "judge":
                    return ScriptedLLM(responder=lambda _p: parecer_carrossel(),
                                       provider="groq")
                return super().llm(stage)

        r = Manual(tmp_path, temas=["Bonsai 27B roda no bolso"],
                   formatos={"extra": ("carousel",)},
                   roteiros=[slides(
                       headlines=["5 dados do Bonsai em disco", "Pesa pouco",
                                  "Rende muito", "Roda rapido", "Salve para depois"],
                       texts=["Arraste e veja", "Ocupa 2 partes no disco",
                              "Mantem 4 de 5 pontos", "Chega a 3 vezes mais",
                              "Salve este resumo para rever"])] * 3)
        extra = Slot("extra", _time(12, 0), "extra", "rodada manual")
        res = r.run(extra, DIA, wait=False)
        assert res.format == "carousel" and res.state == "ready_manual"

    def test_falha_de_publicacao_fica_registrada(self, tmp_path):
        from tests.test_short import resposta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"],
                        roteiros=[resposta_short()] * 3,
                        publicacao=PublishResult(state=PublishState.failed,
                                                 error="access_token_invalid"))
        res = r.run(SLOTS["1600"], DIA, wait=False)
        assert res.state == "failed" and "access_token_invalid" in res.error
        assert SlotRuns(r.cfg.db_path).get(DIA.isoformat(), "1600")["state"] == "failed"

    def test_sem_nenhum_tema_falha_com_motivo_e_sem_excecao(self, tmp_path):
        r = RunnerFalso(tmp_path, temas=[])
        res = r.run(SLOTS["0900"], DIA, wait=False)
        assert res.state == "failed" and "nenhum tema" in res.error

    def test_slot_muito_atrasado_e_pulado(self, tmp_path):
        """Maquina ligada as 14h: o slot das 9h nao sai colado no das 15h."""
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"],
                        agora=datetime(2026, 9, 20, 14, 0, tzinfo=TZ))
        res = r.run(SLOTS["0900"], DIA)
        assert res.state == "skipped" and "depois do horario" in res.error

    def test_sem_publicar_deixa_o_pacote_pronto(self, tmp_path):
        from tests.test_short import resposta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"],
                        roteiros=[resposta_short()] * 3)
        res = r.run(SLOTS["1600"], DIA, publish=False, wait=False)
        assert res.state == "produced" and not r.publicados

    def test_dia_varia_o_formato(self, tmp_path):
        """O segundo slot do dia ve o formato do primeiro e perde pontos nele."""
        runs = SlotRuns(tmp_path / "x.db")
        runs.begin("2026-09-20", "0900")
        runs.update("2026-09-20", "0900", state="published", format="short",
                    pillar="news", topic="Tema A")
        usados = runs.used_today("2026-09-20", except_slot="1600")
        assert usados == {"formats": ["short"], "pillars": ["news"], "topics": ["Tema A"]}

    def test_slot_extra_roda_fora_da_grade(self, tmp_path):
        """Rodada extra: linha propria no dia, sem tocar nos slots dos timers."""
        from datetime import time

        from agent.editorial.slots import Slot
        from tests.test_short import resposta_short
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"],
                        roteiros=[resposta_short()] * 3)
        res = r.run(Slot("extra", time(10, 45), "extra", "teste"), DIA, wait=False)
        assert res.state == "published" and res.slot == "extra"
        linha = SlotRuns(r.cfg.db_path).get(DIA.isoformat(), "extra")
        assert linha["state"] == "published"
        assert SlotRunner._hora(SLOTS["0900"]) == "09h"
        assert SlotRunner._hora(
            Slot("extra", time(10, 45), "extra", "teste")) == "extra"

    def test_formatos_por_fora_da_grade_obedecem_ordem(self, tmp_path):
        """`slot-extra --format video` tenta long antes do short."""
        from datetime import time

        from agent.editorial.slots import Slot
        r = RunnerFalso(tmp_path, temas=["Bonsai 27B roda no bolso"],
                        roteiros=[roteiro_longo()] * 3,
                        formatos={"extra": ("long", "short")})
        res = r.run(Slot("extra", time(10, 45), "extra", "teste"), DIA, wait=False)
        assert res.state == "published" and res.format == "long"


class TestApresentadorNoSlot:
    """O elenco e o acento saem do ID do pilar, nao do objeto dele.

    Bug pago em 20/09/2026: o runner reatribuia `pilar` para o objeto
    `ContentPillar` e depois chamava `presenter_for(pilar)` e
    `accent_for(pilar)`. As duas recebem um id (string), entao caiam no padrao
    em silencio -- nenhum apresentador entrava e todo video saia no verde,
    mesmo nos pilares de acento ciano. Nenhum teste pegava porque os dois
    caminhos degradam sem erro.
    """

    def test_o_pilar_certo_traz_o_apresentador_certo(self, tmp_path):
        from agent.brand.brand import load as load_brand

        runner = RunnerFalso(tmp_path, temas=["tema"])
        marca = load_brand()
        for pilar in ("analise", "tutorial", "futuro", "historia"):
            pedido = runner._pedido_apresentador(marca, pilar)
            assert pedido is not None and pedido.name == "Théo", pilar

    def test_pilar_com_clipe_base_pede_o_clipe_e_nao_o_retrato(self, tmp_path):
        """O caminho de producao e o clipe filmado; o retrato e reserva.

        Mede o PEDIDO, nao o codigo de saida: se `base` vier vazio o video sai
        com o apresentador sintetizado e ninguem levanta erro -- e a mesma
        forma de falhar em silencio do bug do id do pilar.
        """
        from agent.brand.brand import load as load_brand

        runner = RunnerFalso(tmp_path, temas=["tema"])
        pedido = runner._pedido_apresentador(load_brand(), "news")
        assert pedido is not None
        assert pedido.base is not None and pedido.base.name == "theo_base.json"
        assert pedido.base.with_suffix(".mp4").exists()

    def test_objeto_do_pilar_no_lugar_do_id_nao_resolve_nada(self):
        """O sintoma exato do bug, travado: objeto no lugar do id devolve None."""
        from agent.brand.brand import load as load_brand

        marca = load_brand()
        assert marca.presenter_for("analise") is not None
        assert marca.presenter_for(marca.pillars["analise"]) is None

    def test_acento_do_pilar_de_ruptura_nao_e_o_verde_padrao(self):
        from agent.brand.brand import load as load_brand

        marca = load_brand()
        assert marca.accent_for("fato") != marca.accent_for("news")
        assert marca.accent_for(marca.pillars["fato"]) == marca.accent_primary
