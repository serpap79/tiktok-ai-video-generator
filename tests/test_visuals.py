"""Vocabulario visual del canal: 4 pilares, etiquetas fijas, sin término genérico.

Lo que este archivo traba: toda etiqueta es ASCII (va directa a Pexels), término
fuera del pool reprueba con la lista de lo que vale, pilares mezclados reprueban
(un guion, un pilar), y la sugerencia por el asunto es orientación -- quien
decide es el modelo, la puerta solo comprueba pertenencia.
"""

from __future__ import annotations

from agent.writer.visuals import (
    PILLARS,
    brief,
    pillar_of,
    suggest_pillar,
    validate_terms,
)
from agent.writer.writer import build_prompt
from tests.test_writer import dossier


class TestPool:
    def test_toda_etiqueta_es_ascii_y_escena_concreta(self):
        tags = [t for p in PILLARS.values() for t in p.tags]
        assert len(tags) == len({t.lower() for t in tags})  # sin duplicada
        assert all(t.isascii() for t in tags)
        assert all(len(t.split()) >= 2 for t in tags)  # concepto se vuelve escena

    def test_cinco_pilares(self):
        """Los 4 de tech + E (ciencia y espacio, 19/09/2026)."""
        assert sorted(PILLARS) == ["A", "B", "C", "D", "E"]


class TestValidacion:
    def test_etiqueta_del_pool_con_mayusculas_pasa(self):
        assert validate_terms(["Neural Network Nodes", "  ai deep learning loop "]) == []

    def test_parafrasis_reprueba_trayendo_lo_que_vale(self):
        (problema,) = validate_terms(["neural network nodes", "server rack blue lights"])
        assert "ESTETICA" in problema or "pilar" in problema
        assert "server rack blue lights" in problema

    def test_pilares_mezclados_reprueban(self):
        problemas = validate_terms(["sentient ai", "matrix code rain"])
        assert any("mezclan" in p and "A" in p and "B" in p for p in problemas)

    def test_un_pilar_solo_pasa(self):
        assert validate_terms(["sentient ai", "bionic eye neon"]) == []


class TestPilar:
    def test_la_mayoria_gana(self):
        assert pillar_of(["sentient ai", "matrix code rain", "bionic eye neon"]) == "A"

    def test_ningun_casamiento_es_none(self):
        assert pillar_of(["innovation", "future"]) is None

    def test_sugerencia_por_asunto(self):
        assert suggest_pillar("Robot humanoide gana conciencia") == "A"
        assert suggest_pillar("Filtracion expone fallo de seguridad en servidor") == "B"
        assert suggest_pillar("Bonsai 2 27B: modelo de 27B en 5,9 GB") == "C"
        assert suggest_pillar("El ayuntamiento lanza aplicacion de autobuses") == "D"
        assert suggest_pillar("El lado violeta de Marte: hielo, polvo y luz") == "E"
        assert suggest_pillar("Robot humanoide explora Marte") == "A"


class TestPrompt:
    def test_brief_va_al_prompt_con_sugerencia_marcada(self):
        texto = build_prompt(dossier(), None)
        assert "ESTETICA" in texto
        assert "neural network nodes" in texto
        assert brief("C").splitlines()[0].startswith("ESTETICA")
