"""Editorial: tipo de conteudo, formato pela informacao, tema por horario."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from agent.editorial.content import best, classify
from agent.editorial.formats import (
    DossierFeatures,
    choose_format,
    features,
    fit,
    performance_from_metrics,
)
from agent.editorial.planner import choose_pillar, rank_topics
from agent.editorial.slots import SLOTS, TZ, next_slot, parse_slot
from agent.models import Decision, Dossier, Fact, Verdict


def fato(claim: str, url: str = "https://a.com/x") -> Fact:
    return Fact(claim=claim, source_url=url, source_name="fonte",
                quote="trecho literal da fonte que sustenta")


def dossie(*claims: str, urls: tuple[str, ...] = ()) -> Dossier:
    return Dossier(topic="Tema", collected_at=datetime(2026, 9, 20, tzinfo=UTC),
                   facts=[fato(c, urls[i] if i < len(urls) else "https://a.com/x")
                          for i, c in enumerate(claims)])


def decisao(term: str, source: str = "hacker_news", score: float = 0.8) -> Decision:
    return Decision(term=term, source=source, verdict=Verdict.not_selected,
                    reason="teste", score=score, niche_fit=0.8,
                    decided_at=datetime(2026, 9, 20, tzinfo=UTC))


class TestTipoDeConteudo:
    @pytest.mark.parametrize(("tema", "fonte", "esperado"), [
        ("How to use AGENTS.md with Claude Code", "hacker_news", "tutorial"),
        ("GPT-6 vs Gemini 4: which one codes better?", "hacker_news", "vs"),
        ("Ha 55 anos: lancamento do primeiro microprocessador", "wikipedia_onthisday",
         "historia"),
        ("Scientists discover brain is two separate organs", "rss_ciencia", "fato"),
        ("I don't like passkeys", "hacker_news", "analise"),
        ("OpenAI launches GPT-6", "rss_tech", "news"),
        ("What AI will look like in 2030", "hacker_news", "futuro"),
    ])
    def test_pistas_levam_ao_pilar(self, tema, fonte, esperado):
        assert best(tema, fonte).pillar == esperado

    def test_sem_pista_e_noticia_com_motivo(self):
        (g,) = classify("Zyxwv")
        assert g.pillar == "news" and "sem pista" in g.reason

    def test_arquivo_puxa_historia(self):
        assert best("ENIAC: o computador de 30 toneladas", "arquivo").pillar == "historia"


class TestFormato:
    def test_dossie_fino_so_permite_curto(self):
        feat = features(dossie("Um fato unico com 42 por cento"))
        assert fit(feat, "long") == 0 and fit(feat, "carousel") == 0
        assert choose_format(feat, "analise", "2000").format == "short"

    def test_noite_com_arco_vira_longo(self):
        feat = features(dossie(
            "Em 1947 o transistor foi inventado no Bell Labs",
            "O transistor substituiu a valvula nos computadores",
            "Hoje um chip tem bilhoes de transistores",
            "A industria de semicondutores fatura 600 bilhoes",
            "O primeiro radio transistorizado chegou em 1954",
            urls=("https://a.com/1", "https://b.com/2")))
        d = choose_format(feat, "historia", "2000")
        assert d.format == "long"
        assert "historia" in d.reason and "prior declarado" in d.reason

    def test_tutorial_a_tarde_vira_carrossel(self):
        feat = features(dossie(
            "Passo 1: instale o CLI com um comando",
            "Passo 2: configure o arquivo AGENTS.md na raiz",
            "Passo 3: execute o agente no terminal",
            "O erro comum e esquecer de habilitar a leitura"))
        assert choose_format(feat, "tutorial", "1500").format == "carousel"

    def test_formato_repetido_no_dia_perde_pontos(self):
        feat = DossierFeatures(facts=5, with_numbers=3, domains=2, steps=0, dated=1)
        sem = choose_format(feat, "news", "0900", [])
        com = choose_format(feat, "news", "0900", [sem.format])
        assert com.scores[sem.format] < sem.scores[sem.format]
        assert "repetido hoje" in com.reason or com.format != sem.format

    def test_desempenho_medido_entra_com_amostra(self):
        linhas = ([{"format": "short", "completion_rate": 0.8}] * 3
                  + [{"format": "long", "completion_rate": 0.3}] * 3)
        bonus = performance_from_metrics(linhas)
        assert bonus["short"] > 0 > bonus["long"]
        assert performance_from_metrics([{"format": "short", "completion_rate": 0.9}]) == {}


class TestPlanejador:
    def test_noite_prefere_analise_a_noticia_de_mesmo_score(self):
        escolhas = rank_topics([decisao("OpenAI launches GPT-6"),
                                decisao("The case against AI regulation")], "2000")
        assert escolhas[0].pillar == "analise"

    def test_manha_prefere_noticia(self):
        escolhas = rank_topics([decisao("OpenAI launches GPT-6"),
                                decisao("The case against AI regulation")], "0900")
        assert escolhas[0].pillar == "news"

    def test_tipo_ja_usado_hoje_cede_para_alternativa_forte(self):
        palpites = classify("GPT-6 vs Gemini 4 compared: how to choose")
        assert palpites[0].pillar == "vs"
        escolhido = choose_pillar(palpites, used_today=[palpites[0].pillar])
        assert escolhido.pillar != palpites[0].pillar


class TestSlots:
    def test_quatro_horarios_de_brasilia(self):
        """A grade subiu de 3 para 4 posts em 20/09/2026, todos video."""
        assert [s.at.hour for s in SLOTS.values()] == [9, 12, 16, 19]
        assert SLOTS["0900"].when(date(2026, 9, 20)).utcoffset().total_seconds() == -3 * 3600

    @pytest.mark.parametrize("texto", ["0900", "9", "09", "9h", "09:00"])
    def test_parse_aceita_formas_comuns(self, texto):
        assert parse_slot(texto).id == "0900"

    def test_proximo_slot_vira_o_dia(self):
        slot, dia = next_slot(datetime(2026, 9, 20, 21, 0, tzinfo=TZ))
        assert (slot.id, dia) == ("0900", date(2026, 9, 21))


class TestInteresse:
    def test_interesse_reordena_com_motivo(self):
        import json

        from agent.adapters.scripted_llm import ScriptedLLM
        from agent.editorial.interest import rerank

        escolhas = rank_topics([decisao("Bend: a language with formal proofs", score=0.9),
                                decisao("How to automate spreadsheets with AI agents",
                                        score=0.8)], "1500")
        nichado = next(i for i, c in enumerate(escolhas) if "Bend" in c.decision.term)
        amplo = 1 - nichado
        llm = ScriptedLLM(responses=[json.dumps({"notas": [
            {"i": nichado, "motivo": "so para programador", "interesse": 2},
            {"i": amplo, "motivo": "todo mundo usa planilha", "interesse": 9}]})])
        novas, nota = rerank(escolhas, llm)
        assert "spreadsheets" in novas[0].decision.term
        assert "interesse 9/10" in novas[0].reason and "aplicado" in nota

    def test_sem_modelo_vale_o_radar(self):
        from agent.adapters.scripted_llm import ScriptedLLM
        from agent.editorial.interest import rerank

        escolhas = rank_topics([decisao("OpenAI launches GPT-6")], "0900")
        novas, nota = rerank(escolhas, ScriptedLLM(responses=[]))
        assert novas == escolhas and "indisponivel" in nota
