"""Editorial: tipo de contenido, formato según la información, tema por hora."""

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


def hecho(claim: str, url: str = "https://a.com/x") -> Fact:
    return Fact(claim=claim, source_url=url, source_name="fuente",
                quote="fragmento literal de la fuente que lo sostiene")


def dossier(*claims: str, urls: tuple[str, ...] = ()) -> Dossier:
    return Dossier(topic="Tema", collected_at=datetime(2026, 9, 20, tzinfo=UTC),
                   facts=[hecho(c, urls[i] if i < len(urls) else "https://a.com/x")
                          for i, c in enumerate(claims)])


def decision(term: str, source: str = "hacker_news", score: float = 0.8) -> Decision:
    return Decision(term=term, source=source, verdict=Verdict.not_selected,
                    reason="prueba", score=score, niche_fit=0.8,
                    decided_at=datetime(2026, 9, 20, tzinfo=UTC))


class TestTipoDeContenido:
    @pytest.mark.parametrize(("tema", "fuente", "esperado"), [
        ("How to use AGENTS.md with Claude Code", "hacker_news", "tutorial"),
        ("GPT-6 vs Gemini 4: which one codes better?", "hacker_news", "vs"),
        ("Hace 55 años: lanzamiento del primer microprocesador", "wikipedia_onthisday",
         "historia"),
        ("Scientists discover brain is two separate organs", "rss_ciencia", "dato"),
        ("I don't like passkeys", "hacker_news", "analisis"),
        ("OpenAI launches GPT-6", "rss_tech", "news"),
        ("What AI will look like in 2030", "hacker_news", "futuro"),
    ])
    def test_pistas_llevan_al_pilar(self, tema, fuente, esperado):
        assert best(tema, fuente).pillar == esperado

    def test_sin_pista_es_noticia_con_motivo(self):
        (g,) = classify("Zyxwv")
        assert g.pillar == "news" and "sin pista" in g.reason

    def test_archivo_tira_a_historia(self):
        assert best("ENIAC: el ordenador de 30 toneladas", "arquivo").pillar == "historia"


class TestFormato:
    def test_dossier_fino_solo_permite_corto(self):
        feat = features(dossier("Un hecho único con 42 por ciento"))
        assert fit(feat, "long") == 0 and fit(feat, "carousel") == 0
        assert choose_format(feat, "analisis", "2000").format == "short"

    def test_noche_con_arco_se_vuelve_largo(self):
        feat = features(dossier(
            "En 1947 se inventó el transistor en Bell Labs",
            "El transistor sustituyó a la válvula en los ordenadores",
            "Hoy un chip lleva billones de transistores",
            "La industria de semiconductores factura 600 mil millones",
            "La primera radio transistorizada llegó en 1954",
            urls=("https://a.com/1", "https://b.com/2")))
        d = choose_format(feat, "historia", "2000")
        assert d.format == "long"
        assert "historia" in d.reason and "prior declarado" in d.reason

    def test_tutorial_a_tarde_se_vuelve_carrusel(self):
        feat = features(dossier(
            "Paso 1: instala el CLI con un comando",
            "Paso 2: configura el archivo AGENTS.md en la raíz",
            "Paso 3: ejecuta el agente en el terminal",
            "El error común es olvidar habilitar la lectura"))
        assert choose_format(feat, "tutorial", "1500").format == "carousel"

    def test_formato_repetido_en_el_dia_pierde_puntos(self):
        feat = DossierFeatures(facts=5, with_numbers=3, domains=2, steps=0, dated=1)
        sin = choose_format(feat, "news", "0900", [])
        con = choose_format(feat, "news", "0900", [sin.format])
        assert con.scores[sin.format] < sin.scores[sin.format]
        assert "repetido hoy" in con.reason or con.format != sin.format

    def test_rendimiento_medido_entra_con_muestra(self):
        lineas = ([{"format": "short", "completion_rate": 0.8}] * 3
                  + [{"format": "long", "completion_rate": 0.3}] * 3)
        bonus = performance_from_metrics(lineas)
        assert bonus["short"] > 0 > bonus["long"]
        assert performance_from_metrics([{"format": "short", "completion_rate": 0.9}]) == {}


class TestPlanificador:
    def test_noche_prefiere_analisis_a_noticia_del_mismo_score(self):
        elecciones = rank_topics([decision("OpenAI launches GPT-6"),
                                  decision("The case against AI regulation")], "2000")
        assert elecciones[0].pillar == "analisis"

    def test_manana_prefiere_noticia(self):
        elecciones = rank_topics([decision("OpenAI launches GPT-6"),
                                  decision("The case against AI regulation")], "0900")
        assert elecciones[0].pillar == "news"

    def test_tipo_ya_usado_hoy_cede_a_alternativa_fuerte(self):
        apuestas = classify("GPT-6 vs Gemini 4 compared: how to choose")
        assert apuestas[0].pillar == "vs"
        elegido = choose_pillar(apuestas, used_today=[apuestas[0].pillar])
        assert elegido.pillar != apuestas[0].pillar


class TestSlots:
    def test_cuatro_horarios_de_espana(self):
        """La parrilla quedó en 4 posts en 20/09/2026, todos vídeo."""
        assert [s.at.hour for s in SLOTS.values()] == [10, 13, 17, 20]
        assert SLOTS["1000"].when(date(2026, 9, 20)).utcoffset().total_seconds() == 2 * 3600

    @pytest.mark.parametrize("texto", ["1000", "10", "10h", "10:00"])
    def test_parse_acepta_formas_comunes(self, texto):
        assert parse_slot(texto).id == "1000"

    def test_proximo_slot_al_dia_siguiente(self):
        slot, dia = next_slot(datetime(2026, 9, 20, 21, 0, tzinfo=TZ))
        assert (slot.id, dia) == ("1000", date(2026, 9, 21))


class TestInteres:
    def test_interes_reordena_con_motivo(self):
        import json

        from agent.adapters.scripted_llm import ScriptedLLM
        from agent.editorial.interest import rerank

        elecciones = rank_topics([decision("Bend: a language with formal proofs", score=0.9),
                                  decision("How to automate spreadsheets with AI agents",
                                           score=0.8)], "1000")
        nichado = next(i for i, c in enumerate(elecciones) if "Bend" in c.decision.term)
        amplio = 1 - nichado
        llm = ScriptedLLM(responses=[json.dumps({"notas": [
            {"i": nichado, "motivo": "solo para programadores", "interes": 2},
            {"i": amplio, "motivo": "todo el mundo usa hojas de calculo", "interes": 9}]})])
        nuevas, nota = rerank(elecciones, llm)
        assert "spreadsheets" in nuevas[0].decision.term
        assert "interés 9/10" in nuevas[0].reason and "aplicado" in nota

    def test_sin_modelo_vale_el_radar(self):
        from agent.adapters.scripted_llm import ScriptedLLM
        from agent.editorial.interest import rerank

        elecciones = rank_topics([decision("OpenAI launches GPT-6")], "1000")
        nuevas, nota = rerank(elecciones, ScriptedLLM(responses=[]))
        assert nuevas == elecciones and "disponible" in nota
